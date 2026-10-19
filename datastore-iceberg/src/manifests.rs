//! A snapshot's manifests and data files, and which of them a scan reads: a
//! manifest is pruned by the partition summaries of its manifest-list entry, a
//! data file by its partition values and column bounds. A delete manifest is
//! summarized by the partitions of the data files its deletes apply to, so
//! the same pruning keeps the deletes of the data files kept.

use crate::Error;
use crate::pruning::{self, FieldRange};
use crate::store::TableStore;
use dispatch::DataFlowDispatcher;
use iceberg::expr::BoundPredicate;
use iceberg::spec::{
    DataContentType, DataFile, DataFileFormat, Datum, Literal, Manifest, ManifestContentType,
    ManifestFile, ManifestList as IcebergManifestList, NestedField, PartitionSpec,
    PartitionSpecRef, SchemaRef, TableMetadata,
};
use iceberg::{ErrorKind, Result};
use object_storage::load_objects;
use planner::expression::Expression;
use std::collections::HashMap;

/// The snapshot's manifests, in manifest-list order: its data manifests, and
/// the delete manifests with a file still live.
pub(crate) struct ManifestList {
    data_manifests: Vec<ListedManifest>,
    delete_manifests: Vec<ListedManifest>,
    schema: SchemaRef,
    partition_specs: Vec<PartitionSpecRef>,
}

/// The manifests a scan over some filters reads: the data manifests that may
/// list a file holding a row satisfying every filter, and the delete
/// manifests whose deletes may apply to such a file.
pub(crate) struct SelectedManifests<'a> {
    pub data: Vec<&'a ManifestFile>,
    pub deletes: Vec<&'a ManifestFile>,
}

/// A manifest the list names, with what its entry says about the partition
/// values of the manifest's files.
struct ListedManifest {
    file: ManifestFile,
    partition_ranges: Vec<FieldRange>,
}

impl ManifestList {
    /// Read the current snapshot's manifest list, of `size` bytes, and keep
    /// its manifests in order. A table without a snapshot has none. A delete
    /// manifest whose files were all removed carries nothing to apply and is
    /// left out.
    pub fn load(
        table: &str,
        metadata: &TableMetadata,
        size: Option<u64>,
        store: &TableStore,
        dispatcher: &DataFlowDispatcher,
    ) -> crate::Result<Self> {
        let mut manifests = Vec::new();
        if let Some(snapshot) = metadata.current_snapshot() {
            let path = snapshot.manifest_list();
            let size = size
                .expect("a table with a current snapshot is indexed with its manifest list's size");
            let bytes = load_objects(dispatcher, &[store.data_file(path, size)?])?
                .pop()
                .expect("load_objects returns one object per file")
                .bytes;
            let list = IcebergManifestList::parse_with_version(&bytes, metadata.format_version())
                .map_err(|source| Error::MalformedMetadataObject {
                path: path.to_string(),
                source: Box::new(source),
            })?;
            for entry in list.consume_entries() {
                // A delete manifest whose delete files were all removed has
                // nothing left to apply; skipping it saves reading it.
                if entry.content == ManifestContentType::Deletes
                    && !entry.has_added_files()
                    && !entry.has_existing_files()
                {
                    continue;
                }
                manifests.push(entry);
            }
        }
        Self::new(metadata, manifests).map_err(|source| Error::InvalidPartitionMetadata {
            table: table.to_string(),
            source: Box::new(source),
        })
    }

    /// Keep `manifests`, data and delete ones apart, each with its partition
    /// summaries read once.
    fn new(metadata: &TableMetadata, manifests: Vec<ManifestFile>) -> Result<Self> {
        let mut listed = Vec::new();
        let mut listed_deletes = Vec::new();
        for file in manifests {
            let partition_ranges = read_partition_summaries(metadata, &file)?;
            let manifest = ListedManifest {
                file,
                partition_ranges,
            };
            match manifest.file.content {
                ManifestContentType::Data => listed.push(manifest),
                ManifestContentType::Deletes => listed_deletes.push(manifest),
            }
        }
        Ok(Self {
            data_manifests: listed,
            delete_manifests: listed_deletes,
            schema: metadata.current_schema().clone(),
            partition_specs: metadata.partition_specs_iter().cloned().collect(),
        })
    }

    /// The manifests that may list a file holding a row that satisfies every
    /// filter, and the delete manifests whose deletes may apply to one.
    pub fn select(&self, filters: &[Expression]) -> Result<SelectedManifests<'_>> {
        let row_filters = pruning::bind_filters(&self.schema, filters);
        let partition_filters =
            pruning::project_filters_per_spec(&self.schema, &self.partition_specs, &row_filters)?;
        Ok(SelectedManifests {
            data: select_manifests(&self.data_manifests, &partition_filters),
            deletes: select_manifests(&self.delete_manifests, &partition_filters),
        })
    }

    /// The data manifests.
    pub fn data_manifests(&self) -> impl Iterator<Item = &ManifestFile> {
        self.data_manifests.iter().map(|manifest| &manifest.file)
    }

    /// The delete manifests with a file still live.
    pub fn delete_manifests(&self) -> impl Iterator<Item = &ManifestFile> {
        self.delete_manifests.iter().map(|manifest| &manifest.file)
    }

    /// The rows of the data files, deleted ones included. Unknown or
    /// overflowing counts make planning fall back to an ordinary scan.
    pub fn row_count(&self) -> Option<i64> {
        self.data_manifests().try_fold(0i64, |total, manifest| {
            let rows = manifest
                .added_rows_count?
                .checked_add(manifest.existing_rows_count?)?;
            total.checked_add(i64::try_from(rows).ok()?)
        })
    }

    /// The rows of the data files less the rows the delete manifests' entries
    /// count, from the manifest list alone: exact when every delete file is a
    /// deletion vector, an estimate otherwise, never below zero.
    pub fn estimated_row_count(&self) -> Option<i64> {
        let deleted = self.delete_manifests().try_fold(0i64, |total, manifest| {
            let rows = manifest
                .added_rows_count?
                .checked_add(manifest.existing_rows_count?)?;
            total.checked_add(i64::try_from(rows).ok()?)
        })?;
        Some(self.row_count()?.saturating_sub(deleted).max(0))
    }
}

/// Read and parse `manifests`, in the order given.
pub(crate) fn read_manifests<'a>(
    manifests: impl IntoIterator<Item = &'a ManifestFile>,
    store: &TableStore,
    dispatcher: &DataFlowDispatcher,
) -> crate::Result<Vec<Manifest>> {
    let objects = manifests
        .into_iter()
        .map(|manifest| {
            let length = manifest.manifest_length.max(0) as u64;
            store.data_file(&manifest.manifest_path, length)
        })
        .collect::<crate::Result<Vec<_>>>()?;
    let mut parsed = Vec::with_capacity(objects.len());
    for object in load_objects(dispatcher, &objects)? {
        let manifest = Manifest::parse_avro(&object.bytes).map_err(|source| {
            Error::MalformedMetadataObject {
                path: object.file.path.as_str().to_string(),
                source: Box::new(source),
            }
        })?;
        parsed.push(manifest);
    }
    Ok(parsed)
}

/// The manifests among `manifests` whose partition summaries may match the
/// filter projected onto their spec, in `partition_filters`; a manifest of a
/// spec with no projected filter is kept.
fn select_manifests<'a>(
    manifests: &'a [ListedManifest],
    partition_filters: &HashMap<i32, BoundPredicate>,
) -> Vec<&'a ManifestFile> {
    manifests
        .iter()
        .filter(|manifest| {
            partition_filters
                .get(&manifest.file.partition_spec_id)
                .is_none_or(|partition_filter| {
                    pruning::may_match(partition_filter, &manifest.partition_ranges)
                })
        })
        .map(|manifest| &manifest.file)
        .collect()
}

/// What the manifest-list entry `manifest` says about the partition values of
/// the manifest's files: a range for each field of the spec it was written
/// under whose values can prune. Nothing when the entry has no summaries, or
/// the table does not list the spec.
fn read_partition_summaries(
    metadata: &TableMetadata,
    manifest: &ManifestFile,
) -> Result<Vec<FieldRange>> {
    let (Some(partition_spec), Some(summaries)) = (
        metadata.partition_spec_by_id(manifest.partition_spec_id),
        &manifest.partitions,
    ) else {
        return Ok(Vec::new());
    };
    let mut ranges = Vec::new();
    for (position, field) in partition_spec.fields().iter().enumerate() {
        let Some(value_type) = pruning::find_partition_value_type(metadata.current_schema(), field)
        else {
            continue;
        };
        let summary = summaries.get(position).ok_or_else(|| {
            iceberg::Error::new(
                ErrorKind::DataInvalid,
                "manifest summary does not match its partition spec",
            )
        })?;
        let decode = |bound: &Option<iceberg::spec::ByteBuf>| {
            bound
                .as_ref()
                .map(|bytes| Datum::try_from_bytes(bytes, value_type.clone()))
                .transpose()
        };
        ranges.push(FieldRange {
            field_id: field.field_id,
            lower: decode(&summary.lower_bound)?,
            upper: decode(&summary.upper_bound)?,
            // A summary without bounds has no non-null value to bound.
            all_null: summary.contains_null
                && summary.lower_bound.is_none()
                && summary.upper_bound.is_none(),
        });
    }
    Ok(ranges)
}

/// Where a data file is and how long, which is all a footer read needs.
pub(crate) struct FileDescriptor {
    pub path: String,
    pub length: u64,
}

impl FileDescriptor {
    pub fn of(file: &DataFile) -> Self {
        Self {
            path: file.file_path().to_string(),
            length: file.file_size_in_bytes(),
        }
    }
}

/// The data files that some of the snapshot's manifests list, in manifest
/// order.
pub(crate) struct DataFiles {
    manifests: Vec<Manifest>,
    schema: SchemaRef,
    partition_specs: Vec<PartitionSpecRef>,
}

impl DataFiles {
    /// Read and parse `manifests`, data manifests, in the order given so
    /// repeated preparations assign the same indexes. A live delete file
    /// among them, or a file that is not Parquet, is rejected.
    pub fn load<'a>(
        table: &str,
        metadata: &TableMetadata,
        manifests: impl IntoIterator<Item = &'a ManifestFile>,
        store: &TableStore,
        dispatcher: &DataFlowDispatcher,
    ) -> crate::Result<Self> {
        let parsed = read_manifests(manifests, store, dispatcher)?;
        let mut manifests = Vec::with_capacity(parsed.len());
        for manifest in parsed {
            for entry in manifest.entries().iter().filter(|entry| entry.is_alive()) {
                let file = entry.data_file();
                if file.content_type() != DataContentType::Data {
                    return Err(Error::DeleteFile {
                        file: file.file_path().to_string(),
                    });
                }
                if file.file_format() != DataFileFormat::Parquet {
                    return Err(Error::NonParquetFile {
                        table: table.to_string(),
                        file: file.file_path().to_string(),
                        format: file.file_format().to_string(),
                    });
                }
            }
            manifests.push(manifest);
        }
        Ok(Self {
            manifests,
            schema: metadata.current_schema().clone(),
            partition_specs: metadata.partition_specs_iter().cloned().collect(),
        })
    }

    /// The non-empty files that may hold a row satisfying every filter. A file
    /// is described by its column bounds, and by the partition values of the
    /// spec its manifest was written under when the table lists that spec.
    pub fn select(&self, filters: &[Expression]) -> Result<Vec<FileDescriptor>> {
        let row_filters = pruning::bind_filters(&self.schema, filters);
        let compared_columns = pruning::list_compared_fields(&row_filters);
        let partition_filters =
            pruning::project_filters_per_spec(&self.schema, &self.partition_specs, &row_filters)?;
        let mut selected = Vec::new();
        for manifest in &self.manifests {
            let partition_spec = &manifest.metadata().partition_spec;
            let partition_filter = partition_filters.get(&partition_spec.spec_id());
            for file in list_live_files(manifest) {
                if file.data.record_count() == 0 {
                    continue;
                }
                let partition_values = read_partition_values(&self.schema, partition_spec, &file)?;
                let column_bounds = read_column_bounds(&file, &compared_columns);
                if partition_filter.is_none_or(|partition_filter| {
                    pruning::may_match(partition_filter, &partition_values)
                }) && row_filters
                    .iter()
                    .all(|row_filter| pruning::may_match(row_filter, &column_bounds))
                {
                    selected.push(FileDescriptor::of(file.data));
                }
            }
        }
        Ok(selected)
    }

    /// Every file the manifests list as part of the snapshot.
    pub fn iter(&self) -> impl Iterator<Item = LiveFile<'_>> {
        self.manifests.iter().flat_map(list_live_files)
    }
}

/// A data file a manifest lists as part of the snapshot.
pub(crate) struct LiveFile<'a> {
    pub manifest: &'a Manifest,
    pub data: &'a DataFile,
}

fn list_live_files(manifest: &Manifest) -> impl Iterator<Item = LiveFile<'_>> {
    manifest
        .entries()
        .iter()
        .filter(|entry| entry.is_alive())
        .map(move |entry| LiveFile {
            manifest,
            data: entry.data_file(),
        })
}

/// The partition values of `file`: a range for each field of
/// `partition_spec`, the spec the file was written under, whose values can
/// prune. Every row of the file shares a value, so it is both bounds.
fn read_partition_values(
    schema: &SchemaRef,
    partition_spec: &PartitionSpec,
    file: &LiveFile,
) -> Result<Vec<FieldRange>> {
    let partition = file.data.partition().fields();
    let mut ranges = Vec::new();
    for (position, field) in partition_spec.fields().iter().enumerate() {
        let Some(value_type) = pruning::find_partition_value_type(schema, field) else {
            continue;
        };
        let value = partition.get(position).ok_or_else(|| {
            iceberg::Error::new(
                ErrorKind::DataInvalid,
                "file partition does not match its spec",
            )
        })?;
        match value {
            Some(Literal::Primitive(value)) => {
                let value = pruning::literal_to_datum(value, &value_type);
                ranges.push(FieldRange {
                    field_id: field.field_id,
                    lower: value.clone(),
                    upper: value,
                    all_null: false,
                });
            }
            Some(_) => {}
            // A NULL partition value means the column is NULL in every row.
            None => ranges.push(FieldRange {
                field_id: field.field_id,
                lower: None,
                upper: None,
                all_null: true,
            }),
        }
    }
    Ok(ranges)
}

/// The bounds `file`'s manifest records for the columns `fields` of the
/// current schema, read in each column's current type.
fn read_column_bounds(file: &LiveFile, fields: &[&NestedField]) -> Vec<FieldRange> {
    let stored_schema = &file.manifest.metadata().schema;
    fields
        .iter()
        .map(|field| {
            let column_type = field
                .field_type
                .as_primitive_type()
                .expect("a column compared with a constant is primitive");
            let read = |bounds: &HashMap<i32, Datum>| {
                bounds
                    .get(&field.id)
                    .and_then(|bound| pruning::literal_to_datum(bound.literal(), column_type))
            };
            FieldRange {
                field_id: field.id,
                lower: read(file.data.lower_bounds()),
                upper: read(file.data.upper_bounds()),
                // Initial defaults are rejected on load: a column the file's
                // schema lacks is NULL in every row.
                all_null: stored_schema.field_by_id(field.id).is_none()
                    || file.data.null_value_counts().get(&field.id)
                        == Some(&file.data.record_count()),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests;
