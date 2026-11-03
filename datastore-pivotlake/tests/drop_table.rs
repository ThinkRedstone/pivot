//! `DROP TABLE` at the datastore level: the drop removes only the catalog
//! entries (durably, and only at commit), the table's storage survives for
//! vacuum to reclaim after the retention window, and a refresh removes a table
//! another process dropped from the shared manifest.

mod common;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use catalog::datastore::DatastoreTransaction as _;
use common::{DispatchGuard, commit_datastore_transaction, dispatch, table_dir};
use datastore_pivotlake::{DEFAULT_VACUUM_POLL, PivotlakeDatastore, Vacuumer};
use planner::catalog::{
    Column, CreateTableRequest, DropTableRequest, Result as CatalogResult, SchemaQualifiedTableName,
};

/// Past the default 4-hour `deletedFileRetentionDuration`.
const FIVE_HOURS_MS: u64 = 5 * 60 * 60 * 1000;

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

/// Write a small SNAPPY Parquet file (single Int64 column) at `path`.
fn write_parquet(path: &Path) {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int64Array::from(vec![1i64, 2, 3])) as _],
    )
    .unwrap();
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer =
        ArrowWriter::try_new(std::fs::File::create(path).unwrap(), schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// `CREATE TABLE <name> (id Int64)`, adopting the Parquet files already under
/// `adopted_dir` when given, and return the table's own directory.
fn create_table(
    datastore: &Arc<PivotlakeDatastore>,
    dispatch: &DispatchGuard,
    db: &Path,
    name: &str,
    adopted_dir: Option<&Path>,
) -> PathBuf {
    let options = adopted_dir.map_or_else(HashMap::new, |dir| {
        HashMap::from([(
            "with_pre_existing_parquets".to_string(),
            dir.to_string_lossy().into_owned(),
        )])
    });
    let request = CreateTableRequest {
        datastore_name: None,
        schema_name: None,
        name: name.to_string(),
        columns: vec![Column {
            name: "id".to_string(),
            col_type: planner::types::Type::Int64,
        }],
        options,
        if_not_exists: false,
    };
    let transaction = datastore.clone().begin_transaction();
    transaction
        .bind_create_table(request)
        .unwrap()
        .compile(dispatch)
        .unwrap()
        .collect()
        .unwrap();
    commit_datastore_transaction(transaction).unwrap();
    table_dir(db, datastore, name)
}

fn drop_request(name: &str, if_exists: bool) -> DropTableRequest {
    DropTableRequest {
        datastore_name: None,
        schema_name: None,
        name: name.to_string(),
        if_exists,
    }
}

/// Drop a table, mirroring how the server runs `DROP TABLE`: execute the
/// staging dataflow, then commit the transaction that unregisters the table.
fn drop_table(
    datastore: &Arc<PivotlakeDatastore>,
    dispatch: &DispatchGuard,
    request: DropTableRequest,
) -> CatalogResult<()> {
    let transaction = datastore.clone().begin_transaction();
    transaction
        .bind_drop_table(request)?
        .compile(dispatch)?
        .collect()
        .map(|_| ())
        .map_err(|e| planner::catalog::Error::Other(Box::new(e)))?;
    commit_datastore_transaction(transaction)
}

#[test]
fn dropped_table_is_gone_from_the_catalog_but_keeps_its_storage() {
    let dispatch = dispatch(2);
    let db = TempDir::new().unwrap();
    let adopted_dir = db.path().join("adopted");
    std::fs::create_dir_all(&adopted_dir).unwrap();
    write_parquet(&adopted_dir.join("live.parquet"));
    let datastore = PivotlakeDatastore::open(&db.path().to_string_lossy(), &dispatch).unwrap();
    let table_dir = create_table(&datastore, &dispatch, db.path(), "t", Some(&adopted_dir));
    let name = SchemaQualifiedTableName::in_default_schema("t");

    drop_table(&datastore, &dispatch, drop_request("t", false)).unwrap();

    assert!(!datastore.contains_table(&name));
    assert!(
        table_dir.join("_delta_log").exists(),
        "a query planned before the drop may still read the table, so its storage stays for vacuum"
    );
    assert!(adopted_dir.join("live.parquet").exists());
}

#[test]
fn drop_is_visible_only_after_commit_and_survives_a_reopen() {
    let dispatch = dispatch(2);
    let db = TempDir::new().unwrap();
    let datastore = PivotlakeDatastore::open(&db.path().to_string_lossy(), &dispatch).unwrap();
    create_table(&datastore, &dispatch, db.path(), "t", None);
    let name = SchemaQualifiedTableName::in_default_schema("t");
    let transaction = datastore.clone().begin_transaction();
    transaction
        .bind_drop_table(drop_request("t", false))
        .unwrap()
        .compile(&dispatch)
        .unwrap()
        .collect()
        .unwrap();

    assert!(datastore.contains_table(&name));

    commit_datastore_transaction(transaction).unwrap();

    assert!(!datastore.contains_table(&name));
    drop(datastore);
    let reopened = PivotlakeDatastore::open(&db.path().to_string_lossy(), &dispatch).unwrap();
    assert!(!reopened.contains_table(&name));
}

#[test]
fn rolling_back_a_drop_keeps_the_table() {
    let dispatch = dispatch(2);
    let db = TempDir::new().unwrap();
    let datastore = PivotlakeDatastore::open(&db.path().to_string_lossy(), &dispatch).unwrap();
    create_table(&datastore, &dispatch, db.path(), "t", None);
    let transaction = datastore.clone().begin_transaction();
    transaction
        .bind_drop_table(drop_request("t", false))
        .unwrap()
        .compile(&dispatch)
        .unwrap()
        .collect()
        .unwrap();

    transaction.rollback();

    assert!(datastore.contains_table(&SchemaQualifiedTableName::in_default_schema("t")));
}

#[test]
fn dropping_a_missing_table_fails() {
    let dispatch = dispatch(2);
    let db = TempDir::new().unwrap();
    let datastore = PivotlakeDatastore::open(&db.path().to_string_lossy(), &dispatch).unwrap();

    let err = drop_table(&datastore, &dispatch, drop_request("missing", false))
        .unwrap_err()
        .to_string();

    assert!(err.contains("does not exist"), "{err}");
}

#[test]
fn dropping_a_missing_table_with_if_exists_succeeds() {
    let dispatch = dispatch(2);
    let db = TempDir::new().unwrap();
    let datastore = PivotlakeDatastore::open(&db.path().to_string_lossy(), &dispatch).unwrap();

    drop_table(&datastore, &dispatch, drop_request("missing", true)).unwrap();
}

#[test]
fn recreating_a_dropped_name_mints_a_fresh_identity() {
    let dispatch = dispatch(2);
    let db = TempDir::new().unwrap();
    let datastore = PivotlakeDatastore::open(&db.path().to_string_lossy(), &dispatch).unwrap();
    create_table(&datastore, &dispatch, db.path(), "t", None);
    let name = SchemaQualifiedTableName::in_default_schema("t");
    let first_id = datastore.table_handle(&name).unwrap().id();

    drop_table(&datastore, &dispatch, drop_request("t", false)).unwrap();
    create_table(&datastore, &dispatch, db.path(), "t", None);

    // A fresh identity is what lets a cached plan against the first
    // incarnation be told apart from the recreated table.
    assert_ne!(datastore.table_handle(&name).unwrap().id(), first_id);
}

#[test]
fn vacuum_reclaims_a_dropped_tables_storage_only_after_retention() {
    let dispatch = dispatch(2);
    let db = TempDir::new().unwrap();
    let adopted_dir = db.path().join("adopted");
    std::fs::create_dir_all(&adopted_dir).unwrap();
    write_parquet(&adopted_dir.join("live.parquet"));
    let datastore = PivotlakeDatastore::open(&db.path().to_string_lossy(), &dispatch).unwrap();
    let table_dir = create_table(&datastore, &dispatch, db.path(), "t", Some(&adopted_dir));
    drop_table(&datastore, &dispatch, drop_request("t", false)).unwrap();
    let vacuumer = Arc::new(Vacuumer::new(DEFAULT_VACUUM_POLL, datastore.clone()));

    vacuumer.vacuum_all(now_ms()).unwrap();
    assert!(
        table_dir.exists(),
        "within the retention window the dropped table's storage is kept"
    );

    vacuumer.vacuum_all(now_ms() + FIVE_HOURS_MS).unwrap();
    assert!(
        !table_dir.exists(),
        "past the retention window the dropped table's storage is deleted"
    );
    assert!(
        adopted_dir.join("live.parquet").exists(),
        "an adopted file lives outside the table's storage and is never deleted"
    );
}

#[test]
fn refresh_removes_a_table_another_process_dropped() {
    let dispatch = dispatch(2);
    let db = TempDir::new().unwrap();
    let datastore = PivotlakeDatastore::open(&db.path().to_string_lossy(), &dispatch).unwrap();
    create_table(&datastore, &dispatch, db.path(), "t", None);
    let name = SchemaQualifiedTableName::in_default_schema("t");
    // Rewrite the manifest the way another process's drop would, leaving this
    // datastore's in-memory index holding the table.
    let manifest_path = db.path().join("_pivot_manifest.json");
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    let schema_entry = manifest["schemas"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|schema| schema["name"] == planner::DEFAULT_SCHEMA_NAME)
        .unwrap();
    let id = schema_entry["table_ids"]
        .as_object_mut()
        .unwrap()
        .remove("t")
        .unwrap();
    manifest["table_locations"]
        .as_object_mut()
        .unwrap()
        .remove(id.as_str().unwrap());
    manifest["version"] = (manifest["version"].as_u64().unwrap() + 1).into();
    std::fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();

    let changed = datastore.refresh_from_store().unwrap();

    assert!(changed);
    assert!(!datastore.contains_table(&name));
}
