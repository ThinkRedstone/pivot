//! End-to-end tests for a persisted database: a `PivotlakeDatastore` opened on a
//! directory keeps a durable table manifest, so `CREATE TABLE` survives a
//! restart (a fresh `open`). Uses a local directory — the same path an
//! object-store database would take, minus the network.

mod common;
use common::*;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use catalog::datastore::DatastoreTransaction;
use datastore_pivotlake::PivotlakeDatastore;
use dispatch::Projection;
use parquet_engine::table_input;
use planner::DEFAULT_DATASTORE_NAME;
use planner::catalog::{
    Column, CreateTableRequest, Result as CatalogResult, SchemaQualifiedTableName,
};
use planner::types::Type;

fn write_parquet(path: &Path, batch: &arrow_array::RecordBatch) {
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer = ArrowWriter::try_new(
        std::fs::File::create(path).unwrap(),
        batch.schema(),
        Some(props),
    )
    .unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
}

fn write_delta_commit(table: &Path, version: u64, actions: &[serde_json::Value]) {
    let log = table.join("_delta_log");
    std::fs::create_dir_all(&log).unwrap();
    let mut body = actions
        .iter()
        .map(|action| serde_json::to_string(action).unwrap())
        .collect::<Vec<_>>()
        .join("\n");
    body.push('\n');
    std::fs::write(log.join(format!("{version:020}.json")), body).unwrap();
}

fn columns() -> Vec<Column> {
    vec![
        Column {
            name: "name".to_string(),
            col_type: Type::Utf8,
        },
        Column {
            name: "value".to_string(),
            col_type: Type::Int64,
        },
    ]
}

/// A `CREATE TABLE … WITH (with_pre_existing_parquets = '<dir>')` request.
fn adopting_request(name: &str, path: &Path, columns: Vec<Column>) -> CreateTableRequest {
    let mut options = HashMap::new();
    options.insert(
        "with_pre_existing_parquets".to_string(),
        path.to_string_lossy().into_owned(),
    );
    CreateTableRequest {
        datastore_name: None,
        schema_name: None,
        name: name.to_string(),
        columns,
        options,
        if_not_exists: false,
    }
}

/// A `CREATE TABLE name (cols)` request with no explicit path (lives under the
/// database root).
fn rooted_request(name: &str, columns: Vec<Column>) -> CreateTableRequest {
    CreateTableRequest {
        datastore_name: None,
        schema_name: None,
        name: name.to_string(),
        columns,
        options: HashMap::new(),
        if_not_exists: false,
    }
}

/// Mirror the server's `CREATE TABLE`: execute the footer-fetch dataflow, then
/// commit the transaction that persists and publishes the staged table.
fn create(
    dispatch: &DispatchGuard,
    datastore: &Arc<PivotlakeDatastore>,
    request: CreateTableRequest,
) -> CatalogResult<()> {
    let transaction = datastore.clone().begin_transaction();
    transaction
        .bind_create_table(request)?
        .compile(dispatch)?
        .collect()
        .map(|_| ())
        .map_err(|e| planner::catalog::Error::Other(Box::new(e)))?;
    commit_datastore_transaction(transaction)
}

#[test]
fn create_table_over_adopted_parquet_scans_rows() {
    let dispatch = dispatch(1);
    let db = TempDir::new().unwrap();
    let data = TempDir::new().unwrap();
    write_parquet(
        &data.path().join("a.parquet"),
        &strings_and_ints(&["a", "b", "c"], &[1, 2, 3]),
    );

    let datastore = PivotlakeDatastore::open(db.path().to_str().unwrap(), &dispatch).unwrap();
    create(
        &dispatch,
        &datastore,
        adopting_request("events", data.path(), columns()),
    )
    .unwrap();

    let table = datastore
        .clone()
        .begin_transaction()
        .table(
            DEFAULT_DATASTORE_NAME,
            &SchemaQualifiedTableName::in_default_schema("events"),
        )
        .expect("table created");
    assert_eq!(table.columns.len(), 2);
    let parquet = current_parquet(&datastore, "events");
    let results = table_input(&dispatch, &parquet, Projection::all(2), false)
        .collect()
        .unwrap();
    assert_eq!(results.iter().map(|b| b.num_rows()).sum::<usize>(), 3);
    let mut vals = collect_i64s(&results, 1);
    vals.sort();
    assert_eq!(vals, vec![1, 2, 3]);
}

#[test]
fn tables_persist_across_reopen() {
    let dispatch = dispatch(1);
    let db = TempDir::new().unwrap();
    let data = TempDir::new().unwrap();
    write_parquet(
        &data.path().join("a.parquet"),
        &strings_and_ints(&["a", "b"], &[10, 20]),
    );
    let db_uri = db.path().to_str().unwrap();

    // Create the table, then drop the datastore, only the on-disk manifest remains.
    {
        let datastore = PivotlakeDatastore::open(db_uri, &dispatch).unwrap();
        create(
            &dispatch,
            &datastore,
            adopting_request("events", data.path(), columns()),
        )
        .unwrap();
    }

    // Reopening (as a restart would) reloads the table and its data.
    let reopened = PivotlakeDatastore::open(db_uri, &dispatch).unwrap();
    let table = reopened
        .clone()
        .begin_transaction()
        .table(
            DEFAULT_DATASTORE_NAME,
            &SchemaQualifiedTableName::in_default_schema("events"),
        )
        .expect("table reloaded from the manifest");
    assert_eq!(table.columns.len(), 2);
    let parquet = current_parquet(&reopened, "events");
    let results = table_input(&dispatch, &parquet, Projection::all(2), false)
        .collect()
        .unwrap();
    assert_eq!(results.iter().map(|b| b.num_rows()).sum::<usize>(), 2);
}

#[test]
fn background_refresh_advances_to_latest_delta_snapshot() {
    let dispatch = dispatch(1);
    let db = TempDir::new().unwrap();
    let data = TempDir::new().unwrap();
    let first = data.path().join("a.parquet");
    let second = data.path().join("b.parquet");
    write_parquet(&first, &strings_and_ints(&["a"], &[1]));

    let datastore = PivotlakeDatastore::open(db.path().to_str().unwrap(), &dispatch).unwrap();
    create(
        &dispatch,
        &datastore,
        adopting_request("events", data.path(), columns()),
    )
    .unwrap();
    assert_eq!(current_parquet(&datastore, "events").row_groups().len(), 1);

    // The table's log lives under the database root, and names the adopted files
    // by their absolute path, so a hand-written commit does the same.
    let log_dir = table_dir(db.path(), &datastore, "events");
    write_parquet(&second, &strings_and_ints(&["b"], &[2]));
    let second_size = std::fs::metadata(&second).unwrap().len();
    write_delta_commit(
        &log_dir,
        1,
        &[serde_json::json!({
            "add": {
                "path": second.to_str().unwrap(),
                "partitionValues": {},
                "size": second_size,
                "modificationTime": 0,
                "dataChange": true
            }
        })],
    );

    assert!(datastore.refresh_from_store().unwrap());
    assert_eq!(current_parquet(&datastore, "events").row_groups().len(), 2);

    write_delta_commit(
        &log_dir,
        2,
        &[serde_json::json!({
            "remove": {
                "path": first.to_str().unwrap(),
                "deletionTimestamp": 0,
                "dataChange": true
            }
        })],
    );
    assert!(datastore.refresh_from_store().unwrap());
    assert_eq!(current_parquet(&datastore, "events").row_groups().len(), 1);
}

#[test]
fn rooted_table_is_created_empty_under_the_db_root_and_persists() {
    let dispatch = dispatch(1);
    let db = TempDir::new().unwrap();
    let db_uri = db.path().to_str().unwrap();

    {
        let datastore = PivotlakeDatastore::open(db_uri, &dispatch).unwrap();
        create(&dispatch, &datastore, rooted_request("t", columns())).unwrap();

        // A data-less table: registered, with no row groups (its data lives at
        // the directory named for its identity, which fills in once a file is
        // registered there).
        let name = SchemaQualifiedTableName::in_default_schema("t");
        let id = datastore
            .clone()
            .begin_transaction()
            .table_revision(&name)
            .expect("table lookup succeeds")
            .expect("table created")
            .identity;
        assert!(current_parquet(&datastore, "t").row_groups().is_empty());

        // The storage is named for the table's identity. The Delta log's own
        // `metaData` id is Kernel's, minted independently: the catalog owns
        // identity in its manifest and does not read it back from the log.
        let commit = std::fs::read_to_string(
            db.path()
                .join(format!("{id}/_delta_log/00000000000000000000.json")),
        )
        .unwrap();
        let metadata = commit
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .find(|action| action.get("metaData").is_some())
            .unwrap();
        let schema: serde_json::Value =
            serde_json::from_str(metadata["metaData"]["schemaString"].as_str().unwrap()).unwrap();
        let fields = schema["fields"].as_array().unwrap();
        assert!(
            fields
                .iter()
                .all(|field| field["nullable"].as_bool() == Some(false))
        );
        assert_eq!(fields[0]["type"], "string");
        assert_eq!(fields[1]["type"], "long");
    }

    // And it survives a reopen.
    let reopened = PivotlakeDatastore::open(db_uri, &dispatch).unwrap();
    assert!(
        reopened
            .clone()
            .begin_transaction()
            .table(
                DEFAULT_DATASTORE_NAME,
                &SchemaQualifiedTableName::in_default_schema("t"),
            )
            .is_some()
    );
}

#[test]
fn create_fetches_footers_across_workers_before_commit() {
    // Production runs CREATE TABLE with workers = cores: the files are stolen
    // and their footers read by many workers, the row groups fan in to one
    // selected worker, and only that worker stages the completed creation. The
    // coordinator commits it after the dataflow finishes. Cover that shape
    // (the other tests are single-worker).
    //
    // The working set (8 files × footer + column chunks, plus transient
    // decompression buffers) lands around ~32 slots and varies with worker
    // timing. 64 leaves enough headroom to avoid eviction churn under CI
    // scheduling.
    let dispatch = dispatch_with_buffers(4, 64);
    let db = TempDir::new().unwrap();
    let data = TempDir::new().unwrap();
    for i in 0..8i64 {
        write_parquet(
            &data.path().join(format!("{i}.parquet")),
            &strings_and_ints(&["a", "b", "c"], &[i, i + 1, i + 2]),
        );
    }

    let datastore = PivotlakeDatastore::open(db.path().to_str().unwrap(), &dispatch).unwrap();
    create(
        &dispatch,
        &datastore,
        adopting_request("events", data.path(), columns()),
    )
    .unwrap();

    assert_eq!(current_parquet(&datastore, "events").row_groups().len(), 8);
    let parquet = current_parquet(&datastore, "events");
    let results = table_input(&dispatch, &parquet, Projection::all(2), false)
        .collect()
        .unwrap();
    assert_eq!(results.iter().map(|b| b.num_rows()).sum::<usize>(), 24);

    // The no-data case drives the same fan-in with nothing to fetch: every
    // worker's sink finishes empty and the selected worker still commits.
    create(&dispatch, &datastore, rooted_request("empty", columns())).unwrap();
    assert!(
        datastore
            .clone()
            .begin_transaction()
            .table(
                DEFAULT_DATASTORE_NAME,
                &SchemaQualifiedTableName::in_default_schema("empty"),
            )
            .is_some()
    );
    assert!(current_parquet(&datastore, "empty").row_groups().is_empty());
}

#[test]
fn rejects_an_object_store_scheme_in_the_adopt_path() {
    let dispatch = dispatch(1);
    let db = TempDir::new().unwrap();
    let datastore = PivotlakeDatastore::open(db.path().to_str().unwrap(), &dispatch).unwrap();

    let err = create(
        &dispatch,
        &datastore,
        adopting_request("t", Path::new("s3://bucket/data"), columns()),
    )
    .unwrap_err()
    .to_string();
    assert!(
        err.contains("not a URL"),
        "expected scheme rejection: {err}"
    );
}

/// The INSERT path records each written file's Parquet statistics in its `Add`
/// action's `stats`: the row count, and per-column min/max and null count, so a
/// reader can count, prune by range, and skip nulls straight from the log.
#[test]
fn insert_persists_stats_in_add_stats() {
    let dispatch = dispatch(2);
    let db = TempDir::new().unwrap();
    // Columns `name` (a,b,c) and `value` (1,2,3), no nulls.
    let table_dir = write_parquet_files(
        &dispatch,
        db.path(),
        vec![strings_and_ints(&["a", "b", "c"], &[1, 2, 3])],
    );

    // The insert commit (v1) follows the empty CREATE (v0). Aggregate the stats
    // across however many files the write produced.
    let commit =
        std::fs::read_to_string(table_dir.join("_delta_log/00000000000000000001.json")).unwrap();
    let stats: Vec<serde_json::Value> = commit
        .lines()
        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
        .filter(|action| action.get("add").is_some())
        .map(|action| serde_json::from_str(action["add"]["stats"].as_str().unwrap()).unwrap())
        .collect();

    let total_records: i64 = stats
        .iter()
        .map(|s| s["numRecords"].as_i64().unwrap())
        .sum();
    assert_eq!(total_records, 3, "the log records the inserted row count");

    // Per-file bounds vary if the write split across workers; the tightest bounds
    // over all files must span the data.
    let min_value = stats
        .iter()
        .filter_map(|s| s["minValues"]["value"].as_i64())
        .min()
        .expect("a file records a min for `value`");
    let max_value = stats
        .iter()
        .filter_map(|s| s["maxValues"]["value"].as_i64())
        .max()
        .expect("a file records a max for `value`");
    assert_eq!(
        (min_value, max_value),
        (1, 3),
        "value min/max span the data"
    );

    let min_name = stats
        .iter()
        .filter_map(|s| s["minValues"]["name"].as_str())
        .min()
        .expect("a file records a min for `name`");
    let max_name = stats
        .iter()
        .filter_map(|s| s["maxValues"]["name"].as_str())
        .max()
        .expect("a file records a max for `name`");
    assert_eq!(
        (min_name, max_name),
        ("a", "c"),
        "name min/max span the data"
    );

    let total_nulls: i64 = stats
        .iter()
        .map(|s| s["nullCount"]["value"].as_i64().unwrap())
        .sum();
    assert_eq!(total_nulls, 0, "no nulls were inserted");
}
