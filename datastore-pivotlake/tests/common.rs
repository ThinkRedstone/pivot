#![allow(dead_code)]

use arrow_array::{Array, Int64Array, RecordBatch, StringViewArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use std::ops::Deref;
use std::sync::{Arc, Once};
use tempfile::TempDir;

use catalog::datastore::{Datastore, DatastoreTransaction};
use catalog::metastore::{DEFAULT_USER_NAME, Metastore, UserAuth};
use dispatch::{DataFlowDispatcher, Dispatch};
use parquet_engine::ParquetTable;
use planner::catalog::{Result as CatalogResult, SchemaQualifiedTableName};
use std::collections::HashMap;

/// A metastore serving no datastores and only the built-in trusted user:
/// these tests wrap an already-open `PivotlakeDatastore` in a `PivotCatalog`, so
/// the catalog never asks the metastore for datastores.
pub fn trust_metastore() -> Arc<dyn Metastore> {
    #[derive(Debug)]
    struct TrustMetastore;

    impl Metastore for TrustMetastore {
        fn open_datastores(
            &self,
            _dispatcher: &DataFlowDispatcher,
        ) -> catalog::metastore::Result<HashMap<String, Arc<dyn Datastore>>> {
            Ok(HashMap::new())
        }

        fn default_datastore_name(&self) -> &str {
            planner::DEFAULT_DATASTORE_NAME
        }

        fn user_auth(&self, username: &str) -> Option<UserAuth> {
            (username == DEFAULT_USER_NAME).then_some(UserAuth::Trust)
        }
    }

    Arc::new(TrustMetastore)
}

// The process-wide dispatcher, created on the first `init*` call.
// static DISPATCHER: OnceLock<Mutex<DataFlowDispatcher>> = OnceLock::new();

// pub fn init() {
//     init_with_workers(core_affinity::get_core_ids().unwrap().len());
// }

// pub fn init_with_workers(num_workers: usize) {
//     // Each test thread also drops result `RecordBatch`es here, so it needs its
//     // own `MemoryContext`. Create a 1-worker context and install it.
//     let factory = MemoryContextFactory::create_many(1)
//         .into_iter()
//         .next()
//         .unwrap();
//     init_memory_context(factory.create_memory_ctx());
//
//     DISPATCHER.get_or_init(|| {
//         let (exit, handles, dispatcher) = Dispatch::new(num_workers).into_parts();
//         Mutex::new(dispatcher)
//     });
// }

static INIT: Once = Once::new();

fn init_tracing() {
    INIT.call_once(|| {
        let filter = tracing_subscriber::EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("debug"));
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_test_writer()
            .init()
    });
}

/// RAII wrapper around the owned `Dispatch`: derefs to its
/// `DataFlowDispatcher` for `table_input(&dispatch, …)` usage and calls
/// [`Dispatch::exit`] on drop to clean up worker threads.
pub struct DispatchGuard(Option<Dispatch>);

impl Deref for DispatchGuard {
    type Target = DataFlowDispatcher;

    fn deref(&self) -> &Self::Target {
        self.0.as_ref().unwrap().dispatcher()
    }
}

impl Drop for DispatchGuard {
    fn drop(&mut self) {
        self.0.take().unwrap().exit();
    }
}

/// Spin a `Dispatch` up on the calling thread and return a guard for it.
///
/// The buffer count is a process total that `spin_up` divides across NUMA node
/// regions, and workers only allocate from their own region. A multi-node
/// runner therefore halves what each worker can reach, and tests with a few
/// megabytes of pinned results (cached footers, zero-copy string views) run a
/// 10-slot ring dry. 32 leaves headroom either way.
pub fn dispatch(workers: usize) -> DispatchGuard {
    dispatch_with_buffers(workers, 32)
}

/// Like [`dispatch`], but with an explicit file-cache buffer count — needed when
/// a test keeps many row groups (or footers) in flight at once, since each pins
/// a cache slot.
pub fn dispatch_with_buffers(workers: usize, buffers: usize) -> DispatchGuard {
    init_tracing();
    DispatchGuard(Some(Dispatch::spin_up(workers, buffers, None)))
}

/// One dispatch pool shared by every test in a binary, spun up on the first
/// call and left running for the binary's life. This is what a test file wants
/// when several of its tests each drive dataflows: a [`DispatchGuard`] tears its
/// pool down when the test that built it ends, while the buffers here are
/// pre-faulted once and reused. The `workers`/`buffers` of the first caller win.
pub fn shared_dispatcher(workers: usize, buffers: usize) -> DataFlowDispatcher {
    static DISPATCH: std::sync::OnceLock<Dispatch> = std::sync::OnceLock::new();
    init_tracing();
    DISPATCH
        .get_or_init(|| Dispatch::spin_up(workers, buffers, None))
        .dispatcher()
        .clone()
}

/// Commit one datastore transaction to completion. Writing commits hop to
/// Tokio's blocking pool, so direct datastore tests use this small runtime just
/// as the server's async query handler does.
pub fn commit_datastore_transaction(
    transaction: Arc<dyn DatastoreTransaction>,
) -> CatalogResult<()> {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(transaction.commit())
}

pub fn parquet_table(
    dispatch: &DispatchGuard,
    batches: &[RecordBatch],
    dictionary: bool,
) -> (TempDir, Arc<ParquetTable>) {
    parquet_table_compressed(dispatch, batches, dictionary, Compression::SNAPPY)
}

pub fn parquet_table_compressed(
    dispatch: &DispatchGuard,
    batches: &[RecordBatch],
    dictionary: bool,
    compression: Compression,
) -> (TempDir, Arc<ParquetTable>) {
    // Write the parquet file on the test thread (`ArrowWriter` is plain
    // `std::fs` IO — no `MemoryContext` needed).
    let dir = TempDir::new().unwrap();
    let schema = batches[0].schema();
    let path = dir.path().join("data.parquet");
    let props = WriterProperties::builder()
        .set_compression(compression)
        .set_dictionary_enabled(dictionary)
        .build();
    let mut writer =
        ArrowWriter::try_new(std::fs::File::create(&path).unwrap(), schema, Some(props)).unwrap();
    for batch in batches {
        writer.write(batch).unwrap();
    }
    writer.close().unwrap();

    let table = parquet_table_from_dir(dispatch, dir.path());
    (dir, table)
}

/// Reload `name` to its latest committed manifest and return its current row
/// groups for inspection. The background refresh does the same sweep; here we
/// drive it on a cloned-out table handle.
pub fn current_parquet(
    datastore: &datastore_pivotlake::PivotlakeDatastore,
    name: &str,
) -> Arc<ParquetTable> {
    let mut table = datastore
        .table_handle(&SchemaQualifiedTableName::in_default_schema(name))
        .expect("table exists");
    table.refresh().expect("manifest reload");
    table.build_scan_view(&[], &[]).expect("build scan view")
}

/// Where `name` keeps its own storage under a database rooted at
/// `database_root`: its Delta log, and every file written into it. A table's
/// directory is named for its identity, so a test that wants to look at what the
/// engine wrote asks the table rather than guessing the path.
pub fn table_dir(
    database_root: &std::path::Path,
    datastore: &datastore_pivotlake::PivotlakeDatastore,
    name: &str,
) -> std::path::PathBuf {
    database_root.join(
        datastore
            .table_handle(&SchemaQualifiedTableName::in_default_schema(name))
            .expect("table exists")
            .location(),
    )
}

/// Data-file locations for remote files given as fetchable URLs paired with the
/// total size that locates each footer.
pub fn remote_files(files: &[(url::Url, u64)]) -> Vec<object_storage::DataFile> {
    files
        .iter()
        .map(|(url, size)| object_storage::DataFile::remote(url.clone(), *size))
        .collect()
}

/// Every `*.parquet` file directly under `dir`, in listing order.
pub fn parquet_files_in(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    std::fs::read_dir(dir)
        .expect("readable directory")
        .map(|entry| entry.expect("readable entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "parquet") && path.is_file())
        .collect()
}

/// Load a `ParquetTable` from an already-populated directory. Drives the
/// metadata-fetch dataflow, so it runs on the coordinator (the test thread), not
/// inside `run_on_worker`.
pub fn parquet_table_from_dir(
    dispatch: &DispatchGuard,
    dir: &std::path::Path,
) -> Arc<ParquetTable> {
    Arc::new(
        ParquetTable::from_files(dispatch, &parquet_files_in(dir), &[])
            .expect("ParquetTable::from_files failed"),
    )
}

pub fn strings_and_ints(names: &[&str], values: &[i64]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("name", DataType::Utf8View, false),
            Field::new("value", DataType::Int64, false),
        ])),
        vec![
            Arc::new(StringViewArray::from(names.to_vec())),
            Arc::new(Int64Array::from(values.to_vec())),
        ],
    )
    .unwrap()
}

pub fn extract_count(batches: &[RecordBatch]) -> i64 {
    batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0)
}

pub fn collect_strings(batches: &[RecordBatch], col: usize) -> Vec<String> {
    batches
        .iter()
        .flat_map(|b| {
            let a = b
                .column(col)
                .as_any()
                .downcast_ref::<StringViewArray>()
                .unwrap();
            (0..a.len()).map(move |i| a.value(i).to_string())
        })
        .collect()
}

pub fn collect_i64s(batches: &[RecordBatch], col: usize) -> Vec<i64> {
    batches
        .iter()
        .flat_map(|b| {
            let a = b.column(col).as_any().downcast_ref::<Int64Array>().unwrap();
            (0..a.len()).map(move |i| a.value(i))
        })
        .collect()
}

pub fn collect_u64s(batches: &[RecordBatch], col: usize) -> Vec<u64> {
    batches
        .iter()
        .flat_map(|b| {
            let a = b
                .column(col)
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap();
            (0..a.len()).map(move |i| a.value(i))
        })
        .collect()
}

/// Write `batches` as Parquet into a database rooted at `database_root`, through
/// the engine's own write path: a table, an INSERT of the rows, and a commit.
/// This is how a test gets files our writer produced, and it exercises what
/// production runs rather than a separate entry point kept alive for tests.
/// Returns the table's own directory, holding the written files and the Delta
/// log that records them.
///
/// The columns are taken from the batches' schema, so a caller only has to pass
/// the rows it wants written.
pub fn write_parquet_files(
    dispatch: &DispatchGuard,
    database_root: &std::path::Path,
    batches: Vec<RecordBatch>,
) -> std::path::PathBuf {
    use planner::catalog::{Column, CreateTableRequest};

    let schema = batches[0].schema();
    let columns: Vec<Column> = schema
        .fields()
        .iter()
        .map(|field| Column {
            name: field.name().clone(),
            col_type: planner::types::type_from_physical(field.data_type())
                .expect("a test writes a column type the engine has"),
        })
        .collect();

    let datastore =
        datastore_pivotlake::PivotlakeDatastore::open(&database_root.to_string_lossy(), dispatch)
            .unwrap();
    let creation = datastore.clone().begin_transaction();
    creation
        .bind_create_table(CreateTableRequest {
            datastore_name: None,
            schema_name: None,
            name: "written".to_string(),
            columns,
            options: std::collections::HashMap::new(),
            if_not_exists: false,
        })
        .unwrap()
        .compile(dispatch)
        .unwrap()
        .collect()
        .unwrap();
    commit_datastore_transaction(creation).unwrap();

    let name = SchemaQualifiedTableName::in_default_schema("written");
    insert_batches(
        dispatch,
        datastore.clone().begin_transaction(),
        &name.table,
        batches,
    );

    table_dir(database_root, &datastore, &name.table)
}

/// INSERT `batches` into the table `name` through `transaction` and commit it,
/// the way one client's statement runs. The transaction comes from the caller so
/// a test can insert through one it opened earlier (before another statement
/// committed) rather than a fresh one.
pub fn insert_batches(
    dispatch: &DataFlowDispatcher,
    transaction: Arc<dyn DatastoreTransaction>,
    name: &str,
    batches: Vec<RecordBatch>,
) {
    let table = transaction
        .bind_table(
            planner::DEFAULT_DATASTORE_NAME,
            &SchemaQualifiedTableName::in_default_schema(name),
        )
        .expect("table lookup succeeds")
        .expect("the table was created");
    let rows = dispatch::values_input(dispatch, batches).record_batches();
    table
        .compile_insert(rows, dispatch)
        .unwrap()
        .collect()
        .unwrap();
    commit_datastore_transaction(transaction).unwrap();
}
