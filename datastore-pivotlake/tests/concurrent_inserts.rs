//! Several clients INSERTing into one table at once. Each statement runs in its
//! own transaction, and each transaction froze the table at whatever version it
//! began at; the commits still have to line up one log version after another.

mod common;

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use dispatch::DataFlowDispatcher;
use tempfile::TempDir;

use catalog::datastore::DatastoreTransaction;
use common::{commit_datastore_transaction, insert_batches, shared_dispatcher};
use datastore_pivotlake::PivotlakeDatastore;
use planner::catalog::{Column, CreateTableRequest, SchemaQualifiedTableName};
use planner::types::Type;

const TABLE: &str = "events";

/// Eight INSERT dataflows run at once here, each pinning ring slots for the file
/// it writes, so the buffer count is well above the suite's usual default; the
/// extra space avoids heavy eviction churn while all eight are active.
fn dispatcher() -> DataFlowDispatcher {
    shared_dispatcher(4, 128)
}

/// A datastore holding one empty `events (id Int64)` table, created the way the
/// server runs `CREATE TABLE`. The table sits at Delta version 0, so a later
/// version number counts the commits made since.
fn open_datastore_with_empty_table() -> (TempDir, Arc<PivotlakeDatastore>) {
    let database = TempDir::new().unwrap();
    let datastore =
        PivotlakeDatastore::open(&database.path().to_string_lossy(), &dispatcher()).unwrap();
    let transaction = datastore.clone().begin_transaction();
    transaction
        .bind_create_table(CreateTableRequest {
            datastore_name: None,
            schema_name: None,
            name: TABLE.to_string(),
            columns: vec![Column {
                name: "id".to_string(),
                col_type: Type::Int64,
            }],
            options: HashMap::new(),
            if_not_exists: false,
        })
        .unwrap()
        .compile(&dispatcher())
        .unwrap()
        .collect()
        .unwrap();
    commit_datastore_transaction(transaction).unwrap();
    (database, datastore)
}

/// INSERT one row through `transaction` and commit it.
fn insert_row_in(transaction: Arc<dyn DatastoreTransaction>, id: i64) {
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)])),
        vec![Arc::new(Int64Array::from(vec![id]))],
    )
    .unwrap();
    insert_batches(&dispatcher(), transaction, TABLE, vec![batch]);
}

/// INSERT one row in a transaction of its own, opened right now.
fn insert_row(datastore: &Arc<PivotlakeDatastore>, id: i64) {
    insert_row_in(datastore.clone().begin_transaction(), id);
}

/// The version the datastore's live copy of the table sits at, without
/// refreshing it: every commit publishes its result, so this is the version of
/// the last commit made.
fn published_version(datastore: &PivotlakeDatastore) -> u64 {
    datastore
        .table_handle(&SchemaQualifiedTableName::in_default_schema(TABLE))
        .expect("table exists")
        .version()
}

fn count_committed_rows(datastore: &PivotlakeDatastore) -> i64 {
    common::current_parquet(datastore, TABLE)
        .row_groups()
        .iter()
        .map(|group| group.num_rows)
        .sum()
}

/// Eight clients INSERT at once. Every commit appends one version on top of the
/// one the previous commit produced, so all eight land: eight versions above the
/// CREATE, with every row present.
#[test]
fn parallel_inserts_into_one_table_all_land() {
    let (_database, datastore) = open_datastore_with_empty_table();

    std::thread::scope(|scope| {
        for id in 0..8 {
            let datastore = datastore.clone();
            scope.spawn(move || insert_row(&datastore, id));
        }
    });

    assert_eq!(published_version(&datastore), 8, "one version per INSERT");
    assert_eq!(count_committed_rows(&datastore), 8);
}

/// A transaction froze the table before another statement committed into it.
/// Its own commit appends onto the version that statement produced rather than
/// competing for the version it froze.
#[test]
fn an_insert_appends_onto_a_commit_made_after_its_transaction_began() {
    let (_database, datastore) = open_datastore_with_empty_table();
    let began_first = datastore.clone().begin_transaction();

    insert_row(&datastore, 1);
    insert_row_in(began_first, 2);

    assert_eq!(published_version(&datastore), 2);
    assert_eq!(
        count_committed_rows(&datastore),
        2,
        "neither INSERT was lost"
    );
}
