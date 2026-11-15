//! The global, read-only `system` datastore.
//!
//! Its content spans every configured datastore, so it belongs to none of them:
//! it is served as a datastore of its own rather than as a schema each datastore
//! appears to carry. A query reaches it the usual way, `system.tables`, since a
//! two-part name resolves its first part as a datastore before a schema.
//!
//! It has no storage, so it is not a [`crate::datastore::Datastore`] the catalog holds
//! a handle to. Only its transaction is real, assembled by the composite from
//! the sub-transactions the query already opened, and from there the catalog
//! routes to it exactly as it routes to a stored datastore.

use std::fmt::{Debug, Formatter};
use std::sync::Arc;

use crate::datastore::{DatastoreColumnMetadata, DatastoreTableMetadata, DatastoreTransaction};
use arrow_array::{ArrayRef, BooleanArray, Int64Array, RecordBatch, StringViewArray};
use arrow_schema::{Field, Schema, SchemaRef};
use async_trait::async_trait;
use dispatch::{
    DataFlowDispatcher, MemoryBlockState, MemoryBlockStatus, OneShotNullaryFactory, Projection,
    RecordBatchOperatorSpec, memory_ctx, values_input,
};
use planner::catalog::{
    BoundTable, Column, CreateSchemaRequest, CreateTableRequest, DropTableRequest,
    DynamicScanPredicate, Error as CatalogError, Result, SchemaCreation, SchemaQualifiedTableName,
    TableCreation, TableDrop, TableReference, TableRevision,
};
use planner::types::{Type, physical_arrow_type, sql_type_name};

pub const DATASTORE_NAME: &str = "system";
const DATASTORES_NAME: &str = "datastores";
const TABLES_NAME: &str = "tables";
const COLUMNS_NAME: &str = "columns";
const TABLE_FILES_NAME: &str = "table_files";
const MEMORY_BLOCKS_NAME: &str = "memory_blocks";

/// The type `system.datastores` reports this datastore itself as. It stores
/// nothing, so it is neither a pivotlake datastore nor any other stored format.
const SYSTEM_DATASTORE_KIND: &str = "system";

/// A statement this datastore cannot serve. Its relations are assembled per
/// query from the datastores it is composed with, so there is nothing for a
/// write to change; each variant names what the statement targeted.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("cannot create table `{}` in the read-only `{}` datastore", .0, DATASTORE_NAME)]
    CreateTable(SchemaQualifiedTableName),
    #[error("cannot drop table `{}` from the read-only `{}` datastore", .0, DATASTORE_NAME)]
    DropTable(SchemaQualifiedTableName),
    #[error("cannot create schema `{}` in the read-only `{}` datastore", .0, DATASTORE_NAME)]
    CreateSchema(String),
}

/// One entry in the list this transaction is built from: how the catalog serves
/// one datastore, and the sub-transaction the query has open against it. The
/// `system` datastore is never one of these -- it is the datastore these are
/// handed to, and it describes itself.
pub struct DatastoreEntry {
    pub name: String,
    pub kind: String,
    pub data_path: String,
    pub transaction: Arc<dyn DatastoreTransaction>,
}

/// One datastore as `system.datastores` describes it. The datastore serving the
/// relation describes itself this way too, and it has no transaction, so this
/// is what the inventory carries rather than the whole [`DatastoreEntry`].
struct DatastoreDescription {
    name: String,
    kind: String,
    data_path: String,
}

/// One table paired with the datastore that owns it. Datastores only know their
/// schema-qualified names; the inventory adds this final qualifier.
struct GlobalTableMetadata {
    datastore_name: String,
    table: DatastoreTableMetadata,
}

/// Everything the relations are built from: the datastores this transaction
/// spans, and every table each of them holds. Assembled once per binding, so
/// the relations of one query describe one view of the catalog.
struct Inventory {
    datastores: Vec<DatastoreDescription>,
    tables: Vec<GlobalTableMetadata>,
}

/// How one relation is built from the global inventory, over the columns it
/// declares. What it returns decides how the relation is served: rows the
/// inventory already holds, or a scan that reads them when the query runs.
type BuildRelation = fn(TableReference, Vec<Column>, &Inventory) -> Box<dyn BoundTable>;

/// One relation this datastore serves, as it is registered: what reaches it,
/// what identifies it, the columns it serves in order, and how its rows are
/// come by. The columns are written out rather than built, so a relation can be
/// described (in `system.columns`, like any other table) without being built.
/// Contrast [`Relation`], which is one bound instance of a registered relation.
struct SystemRelation {
    name: &'static str,
    /// The relation's durable identity, in the shape a stored table's is: a
    /// datastore's manifest mints that one, and these are minted here, once and
    /// for good. A query keys on `system.tables.id` without caring which
    /// datastore the row describes, so changing one of these breaks whatever
    /// recorded it, exactly as reissuing a stored table's would.
    ///
    /// They are hand-picked rather than random, and spell what they identify
    /// where the hex alphabet allows: reading a join's output tells you which
    /// relation a row belongs to without looking the id up.
    id: &'static str,
    columns: &'static [(&'static str, Type)],
    build: BuildRelation,
}

/// Every relation this datastore serves. The one place a system table is
/// registered: binding one, and reporting it in the inventory, both read this.
const RELATIONS: [SystemRelation; 5] = [
    SystemRelation {
        name: DATASTORES_NAME,
        id: "da7aba5e-5e75-4a11-ab1e-5e1ec7edda7a",
        columns: &[
            ("name", Type::Utf8),
            ("type", Type::Utf8),
            ("data_path", Type::Utf8),
        ],
        build: build_datastores,
    },
    SystemRelation {
        name: TABLES_NAME,
        id: "007ab1e5-1157-4c1d-8055-f1e1d50fda7a",
        columns: &[
            ("datastore", Type::Utf8),
            ("schema", Type::Utf8),
            ("name", Type::Utf8),
            ("id", Type::Utf8),
            ("sorting_keys", Type::Utf8),
            ("partition_key", Type::Utf8),
            ("total_rows", Type::Int64),
            ("bytes", Type::Int64),
            ("bytes_uncompressed", Type::Int64),
        ],
        build: build_tables,
    },
    SystemRelation {
        name: COLUMNS_NAME,
        id: "f1e1d500-da7a-4ce5-bead-e4c77ab1e50f",
        columns: &[
            ("datastore", Type::Utf8),
            ("table_id", Type::Utf8),
            ("name", Type::Utf8),
            ("type", Type::Utf8),
            ("position", Type::Int64),
            ("bytes", Type::Int64),
            ("bytes_uncompressed", Type::Int64),
            ("is_partition_key", Type::Boolean),
            ("is_sort_key", Type::Boolean),
        ],
        build: build_columns,
    },
    SystemRelation {
        name: TABLE_FILES_NAME,
        id: "7ab1ef11-e500-4ded-b10b-de1e7edf11e5",
        columns: &[
            ("table_id", Type::Utf8),
            ("path", Type::Utf8),
            ("partition", Type::Utf8),
            ("bytes", Type::Int64),
            ("bytes_uncompressed", Type::Int64),
            ("min_max_stats", Type::Variant),
        ],
        build: build_table_files,
    },
    SystemRelation {
        name: MEMORY_BLOCKS_NAME,
        id: "a110ca7e-b10c-4bed-ba5e-b10c54110ca7",
        columns: &[
            ("slot", Type::Int64),
            ("node", Type::Int64),
            ("state", Type::Utf8),
            ("readers", Type::Int64),
            ("bytes", Type::Int64),
        ],
        build: build_memory_blocks,
    },
];

/// The `system` datastore's transaction: the sub-transaction of every other
/// datastore, since each of its relations is assembled from them all. The
/// composite opens them before building this, so a query that reads a datastore
/// directly and through `system.tables` reads one snapshot of it either way.
pub struct SystemTransaction {
    datastores: Vec<DatastoreEntry>,
}

impl Debug for SystemTransaction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SystemTransaction")
            .field(
                "datastores",
                &self
                    .datastores
                    .iter()
                    .map(|datastore| datastore.name.as_str())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl SystemTransaction {
    pub fn new(datastores: Vec<DatastoreEntry>) -> Self {
        Self { datastores }
    }

    /// Every datastore and every table of every datastore, this one included:
    /// what all the relations are built from. Describing itself is what lets
    /// `system.tables` list the datastore serving it.
    fn inventory(&self) -> Result<Inventory> {
        let datastores = self
            .datastores
            .iter()
            .map(|datastore| DatastoreDescription {
                name: datastore.name.clone(),
                kind: datastore.kind.clone(),
                data_path: datastore.data_path.clone(),
            })
            .chain([DatastoreDescription {
                name: DATASTORE_NAME.to_string(),
                kind: SYSTEM_DATASTORE_KIND.to_string(),
                // It reads the other datastores' catalogs and the running
                // server, and stores nothing of its own anywhere.
                data_path: String::new(),
            }])
            .collect();

        let mut tables = Vec::new();
        for datastore in &self.datastores {
            tables.extend(datastore.transaction.tables()?.into_iter().map(|table| {
                GlobalTableMetadata {
                    datastore_name: datastore.name.clone(),
                    table,
                }
            }));
        }
        tables.extend(self.tables()?.into_iter().map(|table| GlobalTableMetadata {
            datastore_name: DATASTORE_NAME.to_string(),
            table,
        }));

        Ok(Inventory { datastores, tables })
    }
}

#[async_trait]
impl DatastoreTransaction for SystemTransaction {
    /// Every relation lives in the default schema, the one an unqualified
    /// `system.<table>` resolves to.
    fn does_schema_exist(&self, schema: &str) -> planner::catalog::Result<bool> {
        Ok(schema == planner::DEFAULT_SCHEMA_NAME)
    }

    fn bind_table(
        &self,
        datastore: &str,
        name: &SchemaQualifiedTableName,
    ) -> planner::catalog::Result<Option<Box<dyn BoundTable>>> {
        if name.schema != planner::DEFAULT_SCHEMA_NAME {
            return Ok(None);
        }
        let reference = TableReference {
            datastore: datastore.to_string(),
            schema: name.schema.clone(),
            table: name.table.clone(),
        };
        let Some(relation) = RELATIONS
            .iter()
            .find(|relation| relation.name == name.table)
        else {
            return Ok(None);
        };
        Ok(Some((relation.build)(
            reference,
            declare_columns(relation.columns),
            &self.inventory()?,
        )))
    }

    /// Asked of the binding rather than of a second list of names, so the
    /// contract that every bindable table has a revision holds by construction.
    fn table_revision(
        &self,
        name: &SchemaQualifiedTableName,
    ) -> planner::catalog::Result<Option<TableRevision>> {
        Ok(self
            .bind_table(DATASTORE_NAME, name)?
            .map(|table| table.table_revision()))
    }

    /// The relations this datastore serves. None of them is stored, so none has
    /// files, a durable identifier, or bytes to report; each is named by the
    /// qualified name that reaches it, which is unique and stable for as long as
    /// it is served. The columns are the very ones binding serves, so
    /// `system.columns` describes these relations exactly as it does a stored
    /// table's.
    fn tables(&self) -> Result<Vec<DatastoreTableMetadata>> {
        Ok(RELATIONS
            .iter()
            .map(|relation| DatastoreTableMetadata {
                name: SchemaQualifiedTableName::new(planner::DEFAULT_SCHEMA_NAME, relation.name),
                id: relation.id.to_string(),
                columns: relation
                    .columns
                    .iter()
                    .enumerate()
                    .map(|(position, (name, column_type))| DatastoreColumnMetadata {
                        name: name.to_string(),
                        column_type: column_type.clone(),
                        position,
                        bytes: 0,
                        bytes_uncompressed: 0,
                        is_partition_key: false,
                        is_sort_key: false,
                    })
                    .collect(),
                sort_by: Vec::new(),
                partition_by: Vec::new(),
                total_rows: 0,
                bytes: 0,
                bytes_uncompressed: 0,
                files: Vec::new(),
            })
            .collect())
    }

    fn bind_create_table(&self, request: CreateTableRequest) -> Result<Box<dyn TableCreation>> {
        Err(CatalogError::Other(Box::new(Error::CreateTable(
            request.schema_qualified_name(),
        ))))
    }

    fn bind_drop_table(&self, request: DropTableRequest) -> Result<Box<dyn TableDrop>> {
        Err(CatalogError::Other(Box::new(Error::DropTable(
            request.schema_qualified_name(),
        ))))
    }

    fn bind_create_schema(&self, request: CreateSchemaRequest) -> Result<Box<dyn SchemaCreation>> {
        Err(CatalogError::Other(Box::new(Error::CreateSchema(
            request.name,
        ))))
    }
}

/// Build the `system.datastores` relation. A datastore is named once per
/// server, so its name is its identity and what the other relations join on.
fn build_datastores(
    reference: TableReference,
    columns: Vec<Column>,
    inventory: &Inventory,
) -> Box<dyn BoundTable> {
    let datastores = &inventory.datastores;
    build_table(
        reference,
        columns,
        vec![
            string_array(datastores.iter().map(|entry| entry.name.clone())),
            string_array(datastores.iter().map(|entry| entry.kind.clone())),
            string_array(datastores.iter().map(|entry| entry.data_path.clone())),
        ],
    )
}

/// Build the catalog-wide `system.tables` relation.
fn build_tables(
    reference: TableReference,
    columns: Vec<Column>,
    inventory: &Inventory,
) -> Box<dyn BoundTable> {
    let tables = &inventory.tables;
    build_table(
        reference,
        columns,
        vec![
            string_array(tables.iter().map(|entry| entry.datastore_name.clone())),
            string_array(tables.iter().map(|entry| entry.table.name.schema.clone())),
            string_array(tables.iter().map(|entry| entry.table.name.table.clone())),
            string_array(tables.iter().map(|entry| entry.table.id.clone())),
            // A key list is reported as its column names in order, so an
            // unsorted or unpartitioned table reports an empty string.
            string_array(tables.iter().map(|entry| entry.table.sort_by.join(","))),
            string_array(
                tables
                    .iter()
                    .map(|entry| entry.table.partition_by.join(",")),
            ),
            int_array(tables.iter().map(|entry| entry.table.total_rows as i64)),
            int_array(tables.iter().map(|entry| entry.table.bytes as i64)),
            int_array(
                tables
                    .iter()
                    .map(|entry| entry.table.bytes_uncompressed as i64),
            ),
        ],
    )
}

/// Build the catalog-wide `system.columns` relation: one row per declared
/// column, named by the table that declares it rather than by that table's
/// name, since a table's id is what `system.tables` joins on.
fn build_columns(
    reference: TableReference,
    columns: Vec<Column>,
    inventory: &Inventory,
) -> Box<dyn BoundTable> {
    let declared: Vec<_> = inventory
        .tables
        .iter()
        .flat_map(|entry| {
            entry
                .table
                .columns
                .iter()
                .map(move |column| (entry, column))
        })
        .collect();

    build_table(
        reference,
        columns,
        vec![
            string_array(
                declared
                    .iter()
                    .map(|(entry, _)| entry.datastore_name.clone()),
            ),
            string_array(declared.iter().map(|(entry, _)| entry.table.id.clone())),
            string_array(declared.iter().map(|(_, column)| column.name.clone())),
            string_array(
                declared
                    .iter()
                    .map(|(_, column)| sql_type_name(&column.column_type)),
            ),
            int_array(declared.iter().map(|(_, column)| column.position as i64)),
            int_array(declared.iter().map(|(_, column)| column.bytes as i64)),
            int_array(
                declared
                    .iter()
                    .map(|(_, column)| column.bytes_uncompressed as i64),
            ),
            boolean_array(declared.iter().map(|(_, column)| column.is_partition_key)),
            boolean_array(declared.iter().map(|(_, column)| column.is_sort_key)),
        ],
    )
}

/// Build the catalog-wide `system.table_files` relation: one row per data file,
/// named by the table that holds it rather than by that table's datastore,
/// since a file's owning table is what `system.tables` joins on.
fn build_table_files(
    reference: TableReference,
    columns: Vec<Column>,
    inventory: &Inventory,
) -> Box<dyn BoundTable> {
    let files: Vec<_> = inventory
        .tables
        .iter()
        .flat_map(|entry| entry.table.files.iter().map(move |file| (entry, file)))
        .collect();

    build_table(
        reference,
        columns,
        vec![
            string_array(files.iter().map(|(entry, _)| entry.table.id.clone())),
            string_array(files.iter().map(|(_, file)| file.path.clone())),
            string_array(files.iter().map(|(_, file)| file.partition.clone())),
            // A file is far smaller than an i64 holds, so the cast keeps every
            // size and spares clients an unsigned type they mostly lack.
            int_array(files.iter().map(|(_, file)| file.bytes as i64)),
            int_array(files.iter().map(|(_, file)| file.bytes_uncompressed as i64)),
            variant_array(files.iter().map(|(_, file)| file.min_max_stats.clone())),
        ],
    )
}

/// Build the `system.memory_blocks` relation: one row per block of the memory
/// ring, the fixed-size unit every cache and every operator allocates in.
///
/// Reporting the blocks themselves rather than a summary of them is what makes
/// the summary a query: blocks are all one size, so the share of memory in any
/// state is the share of rows in it, and no aggregate has to be anticipated
/// here. Nothing in the catalog describes memory, so the inventory the other
/// relations are built from is not read.
fn build_memory_blocks(
    reference: TableReference,
    columns: Vec<Column>,
    _inventory: &Inventory,
) -> Box<dyn BoundTable> {
    build_scanned_table(reference, columns, compile_memory_blocks_scan)
}

/// Compile the scan of the memory ring.
///
/// A worker's handle to the memory it allocates from is thread-local, so the
/// read has to happen on a worker.
fn compile_memory_blocks_scan(
    dispatcher: &DataFlowDispatcher,
    schema: SchemaRef,
    projection: Projection,
) -> RecordBatchOperatorSpec {
    // The first worker to be built reads the ring; the rest emit nothing.
    let mut reader = Some((schema, projection));
    let factories: Vec<_> = (0..dispatcher.worker_count())
        .map(|_| {
            let staged = reader.take();
            OneShotNullaryFactory::new(move || {
                let (schema, projection) = staged?;
                let blocks = memory_ctx().read_all_blocks();
                let batch = RecordBatch::try_new(schema, memory_block_arrays(&blocks))
                    .expect("system table columns have equal lengths and declared physical types");
                Some(
                    batch
                        .project(&projection.column_indices)
                        .expect("the projection indexes this relation's own columns"),
                )
            })
        })
        .collect();
    RecordBatchOperatorSpec::from_nullary(dispatcher, factories)
}

/// One row per block of the ring.
fn memory_block_arrays(blocks: &[MemoryBlockStatus]) -> Vec<ArrayRef> {
    // Every block is the same size, so the column repeats one value; it is
    // carried anyway so that summing memory needs no knowledge of that size.
    let bytes = dispatch::block_size_bytes() as i64;

    vec![
        int_array(blocks.iter().map(|block| block.slot as i64)),
        int_array(blocks.iter().map(|block| block.node as i64)),
        string_array(
            blocks
                .iter()
                .map(|block| state_name(block.state).to_string()),
        ),
        int_array(blocks.iter().map(|block| block.readers as i64)),
        int_array(std::iter::repeat_n(bytes, blocks.len())),
    ]
}

/// The name a block's state answers to in the `state` column. A cached block
/// answers to its cache's registered name (`compressed_cache`,
/// `decompressed_cache`, ...).
fn state_name(state: MemoryBlockState) -> &'static str {
    match state {
        MemoryBlockState::Free => "free",
        MemoryBlockState::Pinned => "pinned",
        MemoryBlockState::Cached { cache } => cache,
    }
}

/// Compile a relation that reads itself when scanned, into the operator that
/// emits its projected rows.
type CompileScan = fn(&DataFlowDispatcher, SchemaRef, Projection) -> RecordBatchOperatorSpec;

/// What every relation here has, whichever way its rows are come by: the name
/// that reaches it and the columns it serves. Fixed when the relation is
/// registered, so binding always knows the shape.
#[derive(Debug, Clone)]
struct Relation {
    reference: TableReference,
    columns: Vec<Column>,
    schema: SchemaRef,
}

impl Relation {
    fn new(reference: TableReference, columns: Vec<Column>) -> Self {
        Self {
            reference,
            schema: build_schema(&columns),
            columns,
        }
    }

    /// Names a binding of this relation, but dates nothing: its rows are read
    /// afresh every time, so there is no version to advance. Only a cached plan
    /// reads a revision, and one of these is never cached.
    fn table_revision(&self) -> TableRevision {
        TableRevision {
            identity: format!("{DATASTORE_NAME}.{}", self.reference.table),
            version: "0".to_string(),
        }
    }
}

/// A relation whose rows the transaction already held when the table bound.
/// Scanning it is handing them over, and their count is known without doing so.
#[derive(Debug, Clone)]
struct BoundRelation {
    relation: Relation,
    arrays: Vec<ArrayRef>,
}

impl BoundRelation {
    /// Every relation declares at least one column, so the first array's length
    /// is the row count.
    fn held_row_count(&self) -> usize {
        self.arrays
            .first()
            .expect("a system relation declares at least one column")
            .len()
    }
}

impl BoundTable for BoundRelation {
    fn table_reference(&self) -> TableReference {
        self.relation.reference.clone()
    }

    fn table_revision(&self) -> TableRevision {
        self.relation.table_revision()
    }

    fn is_plan_cacheable(&self) -> bool {
        false
    }

    fn compile_scan(
        &self,
        dispatcher: &DataFlowDispatcher,
        projection: Projection,
        _dynamic_filters: Vec<DynamicScanPredicate>,
        _emit_row_group_metadata: bool,
    ) -> Result<RecordBatchOperatorSpec> {
        let batch = RecordBatch::try_new(self.relation.schema.clone(), self.arrays.clone())
            .expect("system table columns have equal lengths and declared physical types");
        let projected = batch
            .project(projection.indices())
            .map_err(|error| planner::catalog::Error::Other(Box::new(error)))?;
        Ok(values_input(dispatcher, [projected]).record_batches())
    }

    fn columns(&self) -> Vec<Column> {
        self.relation.columns.clone()
    }

    fn nullability(&self) -> Vec<bool> {
        vec![false; self.relation.columns.len()]
    }

    fn clone_box(&self) -> Box<dyn BoundTable> {
        Box::new(self.clone())
    }

    /// Exact, and answered without scanning: the rows are already here.
    fn row_count(&self) -> Option<i64> {
        i64::try_from(self.held_row_count()).ok()
    }

    fn estimate_row_count(&self) -> Option<u64> {
        Some(self.held_row_count() as u64)
    }
}

/// A relation that reads itself when scanned, because its rows describe the
/// running server rather than the catalog. Nothing has read them at bind time,
/// so it has no count to offer and is scanned to be counted.
#[derive(Clone)]
struct ScannedRelation {
    relation: Relation,
    compile_function: CompileScan,
}

/// Shows the relation, not a function pointer's address.
impl Debug for ScannedRelation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "ScannedRelation({:?})", self.relation)
    }
}

impl BoundTable for ScannedRelation {
    fn table_reference(&self) -> TableReference {
        self.relation.reference.clone()
    }

    fn table_revision(&self) -> TableRevision {
        self.relation.table_revision()
    }

    fn is_plan_cacheable(&self) -> bool {
        false
    }

    fn compile_scan(
        &self,
        dispatcher: &DataFlowDispatcher,
        projection: Projection,
        _dynamic_filters: Vec<DynamicScanPredicate>,
        _emit_row_group_metadata: bool,
    ) -> Result<RecordBatchOperatorSpec> {
        Ok((self.compile_function)(
            dispatcher,
            self.relation.schema.clone(),
            projection,
        ))
    }

    fn columns(&self) -> Vec<Column> {
        self.relation.columns.clone()
    }

    fn nullability(&self) -> Vec<bool> {
        vec![false; self.relation.columns.len()]
    }

    fn clone_box(&self) -> Box<dyn BoundTable> {
        Box::new(self.clone())
    }

    fn row_count(&self) -> Option<i64> {
        None
    }

    fn estimate_row_count(&self) -> Option<u64> {
        None
    }
}

/// Build a relation whose rows the transaction already holds, from its declared
/// columns and one array per column, in that same order.
fn build_table(
    reference: TableReference,
    columns: Vec<Column>,
    arrays: Vec<ArrayRef>,
) -> Box<dyn BoundTable> {
    Box::new(BoundRelation {
        relation: Relation::new(reference, columns),
        arrays,
    })
}

/// Build a relation that reads itself when it is scanned, declaring its columns
/// now and the scan that will produce them then. The two must agree: the scan
/// emits one array per column, of that column's physical type.
fn build_scanned_table(
    reference: TableReference,
    columns: Vec<Column>,
    compile_function: CompileScan,
) -> Box<dyn BoundTable> {
    Box::new(ScannedRelation {
        relation: Relation::new(reference, columns),
        compile_function,
    })
}

/// The Arrow schema a relation's arrays are assembled against, one field per
/// declared column. Nothing a system table serves is null, so no field is.
fn build_schema(columns: &[Column]) -> SchemaRef {
    Arc::new(Schema::new(
        columns
            .iter()
            .map(|column| Field::new(&column.name, physical_arrow_type(&column.col_type), false))
            .collect::<Vec<_>>(),
    ))
}

/// A relation's declared columns, as the planner's [`Column`]s: the pairs the
/// registry spells out, in the order it spells them.
fn declare_columns(columns: &[(&str, Type)]) -> Vec<Column> {
    columns
        .iter()
        .map(|(name, col_type)| Column {
            name: name.to_string(),
            col_type: col_type.clone(),
        })
        .collect()
}

fn string_array(values: impl IntoIterator<Item = String>) -> ArrayRef {
    Arc::new(StringViewArray::from_iter_values(values))
}

/// Parse JSON documents into the canonical physical layout backing `VARIANT`.
/// File bounds are assembled by their owning datastore and are valid JSON by
/// construction, so a failure here is an invariant violation rather than a
/// query error.
fn variant_array(values: impl IntoIterator<Item = String>) -> ArrayRef {
    let json = string_array(values);
    planner::expression::json_to_canonical_variant(&json)
        .expect("datastore file bounds are valid JSON objects")
}

fn int_array(values: impl IntoIterator<Item = i64>) -> ArrayRef {
    Arc::new(Int64Array::from_iter_values(values))
}

fn boolean_array(values: impl IntoIterator<Item = bool>) -> ArrayRef {
    Arc::new(BooleanArray::from_iter(values.into_iter().map(Some)))
}
