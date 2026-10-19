//! A pinned Iceberg snapshot with lazily loaded manifests and file metadata.
//! Pivot evaluates metadata bounds before reading surviving files' footers.

use planner::expression::Expression;
use std::sync::Arc;

use arrow_schema::{Field, Schema};
use dispatch::DataFlowDispatcher;
use iceberg::spec::{ManifestFile, SchemaRef, TableMetadataRef, Transform};
use iceberg::table::Table;
use object_storage::ExternalStoreFactory;
use parquet_engine::{FileRowGroups, ParquetTable, TableColumns, load_file_row_groups};
use planner::catalog::{Column, SchemaQualifiedTableName, TableRevision};
use planner::types::physical_arrow_type;

use crate::columns::table_columns;
use crate::deletes::{count_deleted_rows, read_deletion_vectors};
use crate::manifests::{DataFiles, FileDescriptor, ManifestList, read_manifests};
use crate::store::{TableStore, read_vended_s3_credentials};
use crate::{Error, Result};

pub(crate) struct LoadedTable {
    pub name: SchemaQualifiedTableName,
    /// The table's identity across every metadata version.
    pub uuid: String,
    /// The metadata file this table was loaded from. Every commit writes a new
    /// one, so it is the table's version: a cached plan is reused only while
    /// the table is still at the location the plan was bound at.
    pub metadata_location: String,
    pub(super) metadata: TableMetadataRef,
    columns: TableColumns,
    pub(super) partition_by: Vec<String>,
    pub(super) sort_by: Vec<String>,
    manifest_list: ManifestList,
    store: TableStore,
    dispatcher: DataFlowDispatcher,
}

impl LoadedTable {
    /// Load the schema and manifest list, validating unsupported snapshot features
    /// before pruning can hide them. Manifests and data footers remain lazy.
    pub(crate) fn load(
        name: &SchemaQualifiedTableName,
        table: &Table,
        store: TableStore,
        manifest_list_size: Option<u64>,
        dispatcher: &DataFlowDispatcher,
    ) -> Result<Self> {
        let table_name = name.to_string();
        let metadata = table.metadata_ref();
        let metadata_location = table
            .metadata_location()
            .ok_or_else(|| Error::StagedTable {
                table: table_name.clone(),
            })?
            .to_string();
        if metadata.encryption_keys_iter().len() > 0 {
            return Err(Error::EncryptedTable { table: table_name });
        }
        let schema = metadata.current_schema();
        let columns = table_columns(&table_name, schema)?;
        let partition_by = identity_columns(
            schema,
            metadata
                .default_partition_spec()
                .fields()
                .iter()
                .map(|field| (field.source_id, &field.transform)),
        );
        let sort_by = identity_columns(
            schema,
            metadata
                .default_sort_order()
                .fields
                .iter()
                .map(|field| (field.source_id, &field.transform)),
        );

        let manifest_list = ManifestList::load(
            &table_name,
            &metadata,
            manifest_list_size,
            &store,
            dispatcher,
        )?;
        Ok(Self {
            name: name.clone(),
            uuid: metadata.uuid().to_string(),
            metadata_location,
            metadata,
            columns,
            partition_by,
            sort_by,
            manifest_list,
            store,
            dispatcher: dispatcher.clone(),
        })
    }

    /// Select files using metadata bounds, then fetch their row groups, with
    /// the rows their deletion vectors delete marked on them. Manifest order,
    /// then file-local row-group order, makes the same filters yield the same
    /// row-group positions for scanning and materialization.
    pub(crate) fn load_parquet(&self, filters: &[Expression]) -> Result<ParquetTable> {
        let table_name = self.name.to_string();
        let invalid_partition_metadata = |source| Error::InvalidPartitionMetadata {
            table: table_name.clone(),
            source: Box::new(source),
        };
        let selected = self
            .manifest_list
            .select(filters)
            .map_err(invalid_partition_metadata)?;
        let files = self
            .load_data_files(selected.data.iter().copied())?
            .select(filters)
            .map_err(invalid_partition_metadata)?;
        tracing::debug!(table = %self.name, manifests = selected.data.len(), files = files.len(), "selected Iceberg files");
        let delete_manifests = read_manifests(selected.deletes, &self.store, &self.dispatcher)?;
        let deletion_vectors =
            read_deletion_vectors(delete_manifests, &files, &self.store, &self.dispatcher)?;
        let mut footers = self.read_footers(&files)?;
        for (footer, deleted) in footers.iter_mut().zip(deletion_vectors) {
            if let Some(deleted) = deleted {
                footer.mark_deleted_rows(deleted);
            }
        }
        let row_groups: Vec<_> = footers
            .into_iter()
            .flat_map(|footer| footer.row_groups)
            .collect();
        Ok(if row_groups.is_empty() {
            ParquetTable::empty(self.declared_schema())
        } else {
            ParquetTable::new(row_groups)
        })
    }

    /// The data files that `manifests` list.
    fn load_data_files<'a>(
        &self,
        manifests: impl IntoIterator<Item = &'a ManifestFile>,
    ) -> Result<DataFiles> {
        DataFiles::load(
            &self.name.to_string(),
            &self.metadata,
            manifests,
            &self.store,
            &self.dispatcher,
        )
    }

    pub(super) fn load_all_data_files(&self) -> Result<DataFiles> {
        self.load_data_files(self.manifest_list.data_manifests())
    }

    /// Read file metadata in parallel and return it in request order.
    pub(super) fn read_footers(&self, files: &[FileDescriptor]) -> Result<Vec<FileRowGroups>> {
        let data_files = files
            .iter()
            .map(|file| self.store.data_file(&file.path, file.length))
            .collect::<Result<Vec<_>>>()?;
        Ok(load_file_row_groups(
            &self.dispatcher,
            &data_files,
            self.columns.clone(),
        )?)
    }

    pub(crate) fn columns(&self) -> &[Column] {
        self.columns.columns()
    }

    pub(crate) fn nullability(&self) -> Vec<bool> {
        self.metadata
            .current_schema()
            .as_struct()
            .fields()
            .iter()
            .map(|field| !field.required)
            .collect()
    }

    fn declared_schema(&self) -> arrow_schema::SchemaRef {
        Arc::new(Schema::new(
            self.columns()
                .iter()
                .zip(self.nullability())
                .map(|(column, nullable)| {
                    Field::new(
                        &column.name,
                        physical_arrow_type(&column.col_type),
                        nullable,
                    )
                })
                .collect::<Vec<_>>(),
        ))
    }

    pub(crate) fn revision(&self) -> TableRevision {
        TableRevision {
            identity: self.uuid.clone(),
            version: self.metadata_location.clone(),
        }
    }

    /// The rows of the data files less the rows their deletion vectors
    /// delete, read from the manifests alone. Older writers may omit counts,
    /// and a delete manifest may fail to read or hold a delete file that is
    /// not applied; planning then has no statistic and COUNT uses the
    /// ordinary scan, which reports what is wrong.
    pub(crate) fn row_count(&self) -> Option<i64> {
        let rows = self.manifest_list.row_count()?;
        let delete_manifests = read_manifests(
            self.manifest_list.delete_manifests(),
            &self.store,
            &self.dispatcher,
        )
        .ok()?;
        let deleted = count_deleted_rows(delete_manifests).ok()?;
        rows.checked_sub(i64::try_from(deleted).ok()?)
    }

    /// The row count as the manifest list alone has it, with nothing read:
    /// exact when every delete file is a deletion vector, an estimate
    /// otherwise.
    pub(crate) fn estimated_row_count(&self) -> Option<i64> {
        self.manifest_list.estimated_row_count()
    }
}

/// The names of the columns `fields` (a partition spec's or a sort order's,
/// as source field id and transform) take by identity, in field order. A
/// transformed field (a bucket, a day) is not a column value, so it is left out.
fn identity_columns<'a>(
    schema: &SchemaRef,
    fields: impl Iterator<Item = (i32, &'a Transform)>,
) -> Vec<String> {
    fields
        .filter(|(_, transform)| **transform == Transform::Identity)
        .filter_map(|(source_id, _)| schema.name_by_field_id(source_id))
        .map(str::to_string)
        .collect()
}

/// The store `table`'s files are read through: opened with the credentials
/// the catalog vended for the table, or through `store_factory` when it
/// vended none. Opened when the table is indexed rather than when a query loads
/// it, since opening a store may ask the bucket for its region; and opened
/// again at every refresh, so vended credentials never outlive their session.
pub(crate) fn open_table_store(
    name: &SchemaQualifiedTableName,
    table: &Table,
    store_factory: &dyn ExternalStoreFactory,
) -> Result<TableStore> {
    let table_name = name.to_string();
    let metadata_location = table
        .metadata_location()
        .ok_or_else(|| Error::StagedTable {
            table: table_name.clone(),
        })?;
    let vended = read_vended_s3_credentials(&table_name, table.file_io().config().props())?;
    TableStore::open(&table_name, metadata_location, store_factory, vended)
}

/// The size of `table`'s current manifest list, or `None` when the table has
/// no current snapshot. The snapshot names its manifest list without a
/// length, and a ring read needs one, so the store is asked once, when the
/// table is indexed, rather than on every query over it.
pub(crate) fn fetch_manifest_list_size(
    name: &SchemaQualifiedTableName,
    table: &Table,
    store: &TableStore,
) -> Result<Option<u64>> {
    let Some(snapshot) = table.metadata().current_snapshot() else {
        return Ok(None);
    };
    let path = snapshot.manifest_list();
    let size = store
        .object_size(path)?
        .ok_or_else(|| Error::MissingMetadataObject {
            table: name.to_string(),
            path: path.to_string(),
        })?;
    Ok(Some(size))
}

#[cfg(test)]
mod tests {
    use iceberg::spec::{NestedField, PrimitiveType, Schema, Type};

    use super::*;

    #[test]
    fn only_identity_fields_are_columns_in_field_order() {
        let schema: SchemaRef = Arc::new(
            Schema::builder()
                .with_fields(vec![
                    NestedField::required(1, "id", Type::Primitive(PrimitiveType::Long)).into(),
                    NestedField::required(2, "region", Type::Primitive(PrimitiveType::String))
                        .into(),
                    NestedField::required(3, "day", Type::Primitive(PrimitiveType::Date)).into(),
                ])
                .build()
                .unwrap(),
        );
        let fields = [
            (2, Transform::Identity),
            (1, Transform::Bucket(16)),
            (3, Transform::Day),
            (1, Transform::Identity),
        ];

        let columns = identity_columns(
            &schema,
            fields
                .iter()
                .map(|(source_id, transform)| (*source_id, transform)),
        );

        assert_eq!(columns, ["region", "id"]);
    }
}
