use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use arrow_array::{ArrayRef, Int32Array, RecordBatch, StringViewArray};
use arrow_json::ArrayWriter;
use arrow_schema::{Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rstest::fixture;
use serde_json::Value;
use tempfile::TempDir;

use dispatch::{DataFlowDispatcher, Dispatch, Projection, RecordBatchOperatorSpec};
use parquet_engine::{ParquetTable, row_group_filter_from, table_input_with_filter};
use planner::catalog::{
    BoundTable, CatalogTransaction, Column, DynamicScanPredicate, TableReference, TableRevision,
};
use planner::types::Type;
use planner::{DEFAULT_DATASTORE_NAME, DEFAULT_SCHEMA_NAME, Planner, ScalarValue};

#[derive(Clone, Debug)]
struct TestTable {
    reference: TableReference,
    _dir: Arc<TempDir>,
    parquet_table: Arc<ParquetTable>,
    columns: Vec<Column>,
    nullability: Vec<bool>,
}

impl TestTable {
    fn new(dispatch: &Dispatch, name: &str, columns: &[(&str, Type, ArrayRef)]) -> Self {
        let fields: Vec<Field> = columns
            .iter()
            .map(|(name, col_type, array)| {
                let field = Field::new(*name, array.data_type().clone(), array.null_count() > 0);
                // A variant column keeps its extension tag: the parquet writer
                // stamps the VARIANT logical type from it, which the reader in
                // turn needs to hand the binary leaves back as binary.
                match col_type {
                    Type::Variant => field.with_metadata(variant_extension_metadata()),
                    _ => field,
                }
            })
            .collect();
        let schema = Arc::new(Schema::new(fields));
        let arrays: Vec<ArrayRef> = columns.iter().map(|(_, _, array)| array.clone()).collect();
        let batch = RecordBatch::try_new(schema, arrays).unwrap();

        let cols = columns
            .iter()
            .map(|(name, col_type, _)| Column {
                name: name.to_string(),
                col_type: col_type.clone(),
            })
            .collect();
        Self::from_batches(dispatch, name, cols, &[batch])
    }

    /// A table whose data spans one parquet file per batch. The batches may
    /// differ physically (e.g. a variant column shredded differently, or not
    /// at all, per file), as real ingested files do; each batch's own schema
    /// is written as-is.
    fn from_batches(
        dispatch: &Dispatch,
        name: &str,
        columns: Vec<Column>,
        batches: &[RecordBatch],
    ) -> Self {
        // Report the data's actual nullability, as a real binding derives it
        // from footers, so NULL-free test data keeps the fast paths.
        let nullability: Vec<bool> = (0..columns.len())
            .map(|i| {
                batches
                    .iter()
                    .any(|batch| batch.num_columns() > i && batch.column(i).null_count() > 0)
            })
            .collect();
        let dir = TempDir::new().unwrap();
        let props = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .build();
        let mut file_paths = Vec::new();
        for (i, batch) in batches.iter().enumerate() {
            let file_path = dir.path().join(format!("part{i}.parquet"));
            let file = std::fs::File::create(&file_path).unwrap();
            let mut writer =
                ArrowWriter::try_new(file, batch.schema(), Some(props.clone())).unwrap();
            writer.write(batch).unwrap();
            writer.close().unwrap();
            file_paths.push(file_path);
        }

        // Read the footers once, over the dispatch worker pool. This drives a
        // dataflow, so it runs on the coordinator (here), not via run_on_worker.
        // The declared columns travel with the load, as a catalog table's do,
        // so files storing text as unannotated binary read as text.
        let parquet_table = Arc::new(
            ParquetTable::from_files(dispatch.dispatcher(), &file_paths, &columns)
                .expect("ParquetTable::from_files failed"),
        );

        TestTable {
            reference: TableReference {
                datastore: DEFAULT_DATASTORE_NAME.to_string(),
                schema: DEFAULT_SCHEMA_NAME.to_string(),
                table: name.to_string(),
            },
            _dir: Arc::new(dir),
            parquet_table,
            columns,
            nullability,
        }
    }
}

impl BoundTable for TestTable {
    fn table_reference(&self) -> TableReference {
        self.reference.clone()
    }

    fn table_revision(&self) -> TableRevision {
        TableRevision {
            identity: format!("{}:{}", self.reference.datastore, self.reference.table),
            version: "0".to_string(),
        }
    }

    fn compile_scan(
        &self,
        dispatcher: &DataFlowDispatcher,
        projection: Projection,
        dynamic_filters: Vec<DynamicScanPredicate>,
        emit_row_group_metadata: bool,
    ) -> planner::catalog::Result<RecordBatchOperatorSpec> {
        Ok(table_input_with_filter(
            dispatcher,
            &self.parquet_table,
            projection,
            emit_row_group_metadata,
            row_group_filter_from(dynamic_filters),
        ))
    }

    fn columns(&self) -> Vec<Column> {
        self.columns.clone()
    }

    fn nullability(&self) -> Vec<bool> {
        self.nullability.clone()
    }

    // Test tables scan real parquet files, so they resolve pushed paths
    // against each file's shredding exactly like a production table.
    fn applies_variant_extracts(&self) -> bool {
        true
    }

    fn clone_box(&self) -> Box<dyn BoundTable> {
        Box::new(self.clone())
    }

    fn materialize(
        &self,
        input: RecordBatchOperatorSpec,
        projection: Projection,
    ) -> planner::catalog::Result<RecordBatchOperatorSpec> {
        Ok(parquet_engine::materialize(
            input,
            self.parquet_table.clone(),
            projection,
        ))
    }

    fn estimate_row_count(&self) -> Option<u64> {
        Some(
            self.parquet_table
                .row_groups()
                .iter()
                .map(|rg| rg.num_rows as u64)
                .sum(),
        )
    }
}

/// In-memory catalog used by tests. Pre-populated with `example_table` (see
/// the [`catalog`] fixture) and exposes [`TestCatalog::add_table`] so tests
/// that need extra schemas can register them at the start of the test.
#[derive(Debug)]
pub struct TestCatalog {
    tables: Mutex<HashMap<String, TestTable>>,
}

impl TestCatalog {
    fn new() -> Self {
        Self {
            tables: Mutex::new(HashMap::new()),
        }
    }

    /// Register an ad-hoc table backed by the given column data. Mirrors the
    /// shape of the old `make_planner_with_table` helper, but adds the table
    /// to the shared catalog so the existing `testing_planner` can plan
    /// queries against it.
    pub fn add_table(&self, dispatch: &Dispatch, name: &str, columns: &[(&str, Type, ArrayRef)]) {
        self.tables
            .lock()
            .unwrap()
            .insert(name.to_string(), TestTable::new(dispatch, name, columns));
    }

    /// Register a table whose data spans one parquet file per batch (the files
    /// may differ physically, e.g. a variant column shredded per file).
    pub fn add_table_files(
        &self,
        dispatch: &Dispatch,
        name: &str,
        columns: Vec<Column>,
        batches: &[RecordBatch],
    ) {
        self.tables.lock().unwrap().insert(
            name.to_string(),
            TestTable::from_batches(dispatch, name, columns, batches),
        );
    }
}

/// One test transaction: a frozen clone of the catalog's table map, mirroring
/// the production shape (one snapshot per query).
#[derive(Debug)]
struct TestTransaction {
    tables: HashMap<String, TestTable>,
}

impl CatalogTransaction for TestTransaction {
    // Every test table lives in the default schema, so that is the only schema
    // this catalog defines.
    fn does_schema_exist(&self, _datastore: &str, schema: &str) -> planner::catalog::Result<bool> {
        Ok(schema == DEFAULT_SCHEMA_NAME)
    }

    fn bind_table(
        &self,
        reference: &TableReference,
    ) -> planner::catalog::Result<Option<Box<dyn BoundTable>>> {
        Ok(self
            .tables
            .get(&reference.table)
            .cloned()
            .map(|t| Box::new(t) as _))
    }

    fn table_revision(
        &self,
        reference: &TableReference,
    ) -> planner::catalog::Result<Option<TableRevision>> {
        Ok(self
            .tables
            .contains_key(&reference.table)
            .then(|| TableRevision {
                identity: format!("{}:{}", reference.datastore, reference.table),
                version: "0".to_string(),
            }))
    }
}

impl TestCatalog {
    fn begin_transaction(&self) -> Arc<dyn CatalogTransaction> {
        Arc::new(TestTransaction {
            tables: self.tables.lock().unwrap().clone(),
        })
    }
}

/// The Arrow field metadata that marks a struct column as a Parquet variant.
fn variant_extension_metadata() -> HashMap<String, String> {
    use arrow_schema::extension::{EXTENSION_TYPE_METADATA_KEY, EXTENSION_TYPE_NAME_KEY};
    [
        (
            EXTENSION_TYPE_NAME_KEY.to_owned(),
            "arrow.parquet.variant".to_owned(),
        ),
        (EXTENSION_TYPE_METADATA_KEY.to_owned(), String::new()),
    ]
    .into()
}

fn int_col(values: Vec<i32>) -> ArrayRef {
    Arc::new(Int32Array::from(values))
}

fn str_col(values: Vec<&'static str>) -> ArrayRef {
    Arc::new(StringViewArray::from(values))
}

/// A `Planner` paired with a handle to its backing catalog and a private
/// `Dispatch` for the workers running compiled plans.
///
/// The first two are bundled in one fixture (rather than two separate
/// fixtures) because rstest does not share fixture instances between a
/// sub-fixture dependency and a direct test parameter — they would resolve
/// to different `TestCatalog` instances and any tables registered through
/// the test's `catalog` would never reach the planner.
///
/// `dispatch` is owned by the fixture so each test gets its own worker pool
/// and matching `MemoryContext`, and tear-down is automatic when the fixture
/// drops at end of test.
pub struct TestingPlanner {
    pub planner: Planner,
    #[allow(dead_code)]
    pub catalog: Arc<TestCatalog>,
    // Held only for its `Drop` side-effect; access the dispatcher via
    // [`TestingPlanner::dispatcher`].
    #[allow(dead_code)]
    dispatch: Dispatch,
}

#[allow(dead_code)] // not all test binaries call every helper
impl TestingPlanner {
    /// Borrow the dispatcher to hand to `Plan::compile`.
    pub fn dispatcher(&self) -> &DataFlowDispatcher {
        self.dispatch.dispatcher()
    }

    /// Register an ad-hoc table in the catalog. Uses this fixture's own
    /// `Dispatch` so the parquet write/open happens on a worker.
    pub fn add_table(&self, name: &str, columns: &[(&str, Type, ArrayRef)]) {
        self.catalog.add_table(&self.dispatch, name, columns);
    }

    /// Register a table whose data spans one parquet file per batch.
    #[allow(dead_code)] // not all test binaries call every helper
    pub fn add_table_files(&self, name: &str, columns: Vec<Column>, batches: &[RecordBatch]) {
        self.catalog
            .add_table_files(&self.dispatch, name, columns, batches);
    }

    /// Plan `sql` inside a fresh transaction on this fixture's catalog, the
    /// one-liner tests use instead of wiring the transaction themselves.
    pub fn plan(&mut self, sql: &str) -> Result<planner::Plan, planner::Error> {
        let transaction = self.catalog.begin_transaction();
        self.planner.plan(sql, transaction)
    }

    /// A fresh transaction on this fixture's catalog, for compiling plans.
    pub fn transaction(&self) -> Arc<dyn CatalogTransaction> {
        self.catalog.begin_transaction()
    }
}

/// Shared planner backed by a catalog seeded with `example_table`:
/// `(a Int32, b Int32, c Int32, name Utf8)`, 5 rows:
///
/// ```text
/// a | b  | c   | name
/// --+----+-----+--------
/// 1 | 10 | 100 | alice
/// 2 | 20 | 200 | bob
/// 3 | 30 | 300 | charlie
/// 4 | 40 | 400 | dave
/// 5 | 50 | 500 | alice
/// ```
///
/// Tests that need a different schema reach into the returned `catalog` and
/// call [`TestCatalog::add_table`] before planning.
#[fixture]
pub fn testing_planner() -> TestingPlanner {
    testing_planner_with_ring_slots(32)
}

/// The fixture with a caller-sized ring, for the few tests whose plans hold
/// more live buffers than the default 32 slots (e.g. several nested CTE
/// tees over multi-thousand-row tables).
#[allow(dead_code)] // not all test binaries call every helper
pub fn testing_planner_with_ring_slots(ring_slots: usize) -> TestingPlanner {
    // `TestTable::new` calls `ParquetTable::from_files`, which touches
    // `memory_ctx()` (compressed cache) and so must run on a worker — see
    // `TestTable::new` for the `run_on_worker` hop.
    let dispatch = Dispatch::spin_up(1, ring_slots, None);
    let catalog = Arc::new(TestCatalog::new());
    catalog.add_table(
        &dispatch,
        "example_table",
        &[
            ("a", Type::Int32, int_col(vec![1, 2, 3, 4, 5])),
            ("b", Type::Int32, int_col(vec![10, 20, 30, 40, 50])),
            ("c", Type::Int32, int_col(vec![100, 200, 300, 400, 500])),
            (
                "name",
                Type::Utf8,
                str_col(vec!["alice", "bob", "charlie", "dave", "alice"]),
            ),
        ],
    );
    let planner = Planner::from_datastore_names(
        vec![DEFAULT_DATASTORE_NAME.to_string()],
        DEFAULT_DATASTORE_NAME.to_string(),
    )
    .expect("planner context");
    TestingPlanner {
        planner,
        catalog,
        dispatch,
    }
}

/// Plan, compile, and run `sql`, returning the result rows as JSON.
#[allow(dead_code)] // not all test binaries call every helper
pub fn run(planner: &mut TestingPlanner, sql: &str) -> Vec<Value> {
    batches_to_json(&run_batches(planner, sql))
}

/// Plan, compile, and run `sql`, returning the raw result batches. Lets a test
/// inspect the output arrow schema rather than only the JSON-rendered values.
#[allow(dead_code)]
pub fn run_batches(planner: &mut TestingPlanner, sql: &str) -> Vec<RecordBatch> {
    // One transaction spans plan and compile, as a statement's does.
    let transaction = planner.transaction();
    planner
        .planner
        .plan(sql, transaction.clone())
        .unwrap()
        .compile(planner.dispatcher(), transaction.as_ref())
        .unwrap()
        .collect()
        .unwrap()
}

/// Plan `sql` with `parameters` as its placeholder values, compile, and run
/// it, returning the raw result batches.
#[allow(dead_code)]
pub fn run_batches_with_parameters(
    planner: &mut TestingPlanner,
    sql: &str,
    parameters: Vec<ScalarValue>,
) -> Vec<RecordBatch> {
    let transaction = planner.transaction();
    planner
        .planner
        .plan_with_parameters(sql, parameters, transaction.clone())
        .unwrap()
        .compile(planner.dispatcher(), transaction.as_ref())
        .unwrap()
        .collect()
        .unwrap()
}

/// Plan, compile, and run `sql`, returning its execution error.
#[allow(dead_code)]
pub fn run_expecting_error(planner: &mut TestingPlanner, sql: &str) -> String {
    let transaction = planner.transaction();
    planner
        .planner
        .plan(sql, transaction.clone())
        .unwrap()
        .compile(planner.dispatcher(), transaction.as_ref())
        .unwrap()
        .collect()
        .expect_err("query should fail")
        .to_string()
}

/// The value of a row's single column, by position rather than name. For tests
/// that check a computed select item's value, not its (DuckDB-derived) name.
#[allow(dead_code)]
pub fn only_column(row: &Value) -> &Value {
    let cols = row.as_object().expect("row is a JSON object");
    assert_eq!(cols.len(), 1, "only_column expects exactly one column");
    cols.values().next().unwrap()
}

// Used only by compile tests, but this module is shared across test binaries.
#[allow(dead_code)]
pub fn batches_to_json(batches: &[RecordBatch]) -> Vec<Value> {
    let mut writer = ArrayWriter::new(Vec::new());
    writer
        .write_batches(&batches.iter().collect::<Vec<_>>())
        .unwrap();
    writer.finish().unwrap();
    serde_json::from_slice(&writer.into_inner()).unwrap()
}
