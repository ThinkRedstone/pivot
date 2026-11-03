//! Vacuum deletes files the current table version no longer references once they
//! have been unreferenced for the retention window, and never touches a live
//! file. A retired file is dated by its tombstone, an orphan by its storage
//! mtime. Age is judged against `now_ms`, threaded in so the window is exercised
//! without waiting on the wall clock.

mod common;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use catalog::datastore::DatastoreTransaction as _;
use common::{DispatchGuard, commit_datastore_transaction, dispatch, table_dir};
use datastore_pivotlake::{DEFAULT_VACUUM_POLL, PivotlakeDatastore, Vacuumer};
use planner::catalog::{Column, CreateTableRequest, SchemaQualifiedTableName};

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
    let schema = Arc::new(Schema::new(vec![Field::new(
        "Timestamp",
        DataType::Int64,
        false,
    )]));
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

/// `CREATE TABLE events (Timestamp Int64)`, adopting the Parquet files already
/// under `dir`, and return the table's own directory: where its log lives, and
/// where files written into it land.
fn create_events_table(
    datastore: &Arc<PivotlakeDatastore>,
    dispatch: &DispatchGuard,
    db: &Path,
    dir: &Path,
) -> PathBuf {
    let request = CreateTableRequest {
        datastore_name: None,
        schema_name: None,
        name: "events".to_string(),
        columns: vec![Column {
            name: "Timestamp".to_string(),
            col_type: planner::types::Type::Int64,
        }],
        options: HashMap::from([(
            "with_pre_existing_parquets".to_string(),
            dir.to_string_lossy().into_owned(),
        )]),
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
    table_dir(db, datastore, "events")
}

#[test]
fn unreferenced_file_past_retention_is_deleted_and_the_live_file_is_kept() {
    let dispatch = dispatch(2);
    let db = TempDir::new().unwrap();
    let adopted_dir = db.path().join("events");
    std::fs::create_dir_all(&adopted_dir).unwrap();
    write_parquet(&adopted_dir.join("live.parquet"));
    let datastore = PivotlakeDatastore::open(&db.path().to_string_lossy(), &dispatch).unwrap();
    let table_dir = create_events_table(&datastore, &dispatch, db.path(), &adopted_dir);
    // Dropped into the table's own directory after CREATE, so no Add references
    // it: an unreferenced orphan.
    write_parquet(&table_dir.join("orphan.parquet"));

    let vacuumer = Arc::new(Vacuumer::new(DEFAULT_VACUUM_POLL, datastore.clone()));
    vacuumer.vacuum_all(now_ms() + FIVE_HOURS_MS).unwrap();

    assert!(
        !table_dir.join("orphan.parquet").exists(),
        "an unreferenced file past the retention window is deleted"
    );
    assert!(
        adopted_dir.join("live.parquet").exists(),
        "the current version's live file is kept, however old"
    );
}

#[test]
fn unreferenced_file_within_retention_is_kept() {
    let dispatch = dispatch(2);
    let db = TempDir::new().unwrap();
    let adopted_dir = db.path().join("events");
    std::fs::create_dir_all(&adopted_dir).unwrap();
    write_parquet(&adopted_dir.join("live.parquet"));
    let datastore = PivotlakeDatastore::open(&db.path().to_string_lossy(), &dispatch).unwrap();
    let table_dir = create_events_table(&datastore, &dispatch, db.path(), &adopted_dir);
    write_parquet(&table_dir.join("orphan.parquet"));

    let vacuumer = Arc::new(Vacuumer::new(DEFAULT_VACUUM_POLL, datastore.clone()));
    vacuumer.vacuum_all(now_ms()).unwrap();

    assert!(
        table_dir.join("orphan.parquet").exists(),
        "an unreferenced file still within the retention window is kept"
    );
}

/// The adopted directory is never swept: those files are the user's, so an
/// unreferenced one there survives however old it is.
#[test]
fn a_file_in_the_adopted_directory_is_never_deleted() {
    let dispatch = dispatch(2);
    let db = TempDir::new().unwrap();
    let adopted_dir = db.path().join("events");
    std::fs::create_dir_all(&adopted_dir).unwrap();
    write_parquet(&adopted_dir.join("live.parquet"));
    let datastore = PivotlakeDatastore::open(&db.path().to_string_lossy(), &dispatch).unwrap();
    create_events_table(&datastore, &dispatch, db.path(), &adopted_dir);
    write_parquet(&adopted_dir.join("theirs.parquet"));

    let vacuumer = Arc::new(Vacuumer::new(DEFAULT_VACUUM_POLL, datastore.clone()));
    vacuumer.vacuum_all(now_ms() + FIVE_HOURS_MS).unwrap();

    assert!(adopted_dir.join("theirs.parquet").exists());
}

/// Compaction merges adopted files into the table's own storage and drops them
/// from the log, and the vacuum sweep that follows still leaves the files
/// themselves alone, however far past the retention window: they are the
/// directory owner's, not the table's.
#[test]
fn compaction_merges_adopted_files_without_deleting_them() {
    let dispatch = dispatch(2);
    let db = TempDir::new().unwrap();
    let adopted_dir = db.path().join("events");
    std::fs::create_dir_all(&adopted_dir).unwrap();
    for name in ["a.parquet", "b.parquet"] {
        write_parquet(&adopted_dir.join(name));
    }
    let datastore = PivotlakeDatastore::open(&db.path().to_string_lossy(), &dispatch).unwrap();
    let table_dir = create_events_table(&datastore, &dispatch, db.path(), &adopted_dir);
    let name = SchemaQualifiedTableName::in_default_schema("events");
    let inputs = datastore.table_files(&name).unwrap();
    assert_eq!(inputs.len(), 2, "both adopted files are live");

    let id = datastore.table_handle(&name).unwrap().id();
    datastore_pivotlake::compact_table_files(&datastore, id, &inputs, 128 * 1024, 128 * 1024)
        .unwrap();
    Arc::new(Vacuumer::new(DEFAULT_VACUUM_POLL, datastore.clone()))
        .vacuum_all(now_ms() + FIVE_HOURS_MS)
        .unwrap();

    // One merged file in the table's own directory, holding every row.
    let merged = datastore.table_files(&name).unwrap();
    assert_eq!(merged.len(), 1);
    assert!(table_dir.join(merged[0].path.as_str()).exists());
    // And both originals still sit where their owner left them.
    for name in ["a.parquet", "b.parquet"] {
        assert!(
            adopted_dir.join(name).exists(),
            "compaction must not delete {name}"
        );
    }
}

/// Rewrite the table's current files into one new file in the table's own
/// directory, retiring the inputs, and return the new file's path.
fn rewrite_table_files(
    datastore: &Arc<PivotlakeDatastore>,
    name: &SchemaQualifiedTableName,
) -> PathBuf {
    let inputs = datastore.table_files(name).unwrap();
    let id = datastore.table_handle(name).unwrap().id();
    datastore_pivotlake::compact_table_files(datastore, id, &inputs, 128 * 1024, 128 * 1024)
        .unwrap();
    let outputs = datastore.table_files(name).unwrap();
    assert_eq!(outputs.len(), 1);
    PathBuf::from(outputs[0].path.as_str())
}

/// Backdate a file's storage mtime past the retention window, so only its
/// tombstone can still date it as recently retired.
fn backdate_past_retention(path: &Path) {
    std::fs::File::options()
        .write(true)
        .open(path)
        .unwrap()
        .set_modified(SystemTime::now() - Duration::from_millis(FIVE_HOURS_MS))
        .unwrap();
}

/// A file that was live for longer than the window and was then retired is
/// kept while its tombstone is inside the window, however old its bytes are.
#[test]
fn a_retired_file_is_kept_while_its_tombstone_is_inside_the_window() {
    let dispatch = dispatch(2);
    let db = TempDir::new().unwrap();
    let adopted_dir = db.path().join("events");
    std::fs::create_dir_all(&adopted_dir).unwrap();
    write_parquet(&adopted_dir.join("a.parquet"));
    let datastore = PivotlakeDatastore::open(&db.path().to_string_lossy(), &dispatch).unwrap();
    let table_dir = create_events_table(&datastore, &dispatch, db.path(), &adopted_dir);
    let name = SchemaQualifiedTableName::in_default_schema("events");
    let retired = table_dir.join(rewrite_table_files(&datastore, &name));
    rewrite_table_files(&datastore, &name);
    backdate_past_retention(&retired);

    Arc::new(Vacuumer::new(DEFAULT_VACUUM_POLL, datastore.clone()))
        .vacuum_all(now_ms())
        .unwrap();

    assert!(
        retired.exists(),
        "a file retired inside the window is kept whatever its storage mtime says"
    );
}

/// A datastore that did not make the retiring commit learns the tombstone from
/// the log when it loads the table, and keeps the file the same way.
#[test]
fn a_reopened_datastore_dates_a_retired_file_by_the_log_tombstone() {
    let dispatch = dispatch(2);
    let db = TempDir::new().unwrap();
    let adopted_dir = db.path().join("events");
    std::fs::create_dir_all(&adopted_dir).unwrap();
    write_parquet(&adopted_dir.join("a.parquet"));
    let name = SchemaQualifiedTableName::in_default_schema("events");
    let retired = {
        let datastore = PivotlakeDatastore::open(&db.path().to_string_lossy(), &dispatch).unwrap();
        let table_dir = create_events_table(&datastore, &dispatch, db.path(), &adopted_dir);
        let retired = table_dir.join(rewrite_table_files(&datastore, &name));
        rewrite_table_files(&datastore, &name);
        retired
    };
    backdate_past_retention(&retired);
    let reopened = PivotlakeDatastore::open(&db.path().to_string_lossy(), &dispatch).unwrap();

    Arc::new(Vacuumer::new(DEFAULT_VACUUM_POLL, reopened))
        .vacuum_all(now_ms())
        .unwrap();

    assert!(
        retired.exists(),
        "a file whose tombstone was read from the log is kept inside the window"
    );
}

/// Once the tombstone is past the window the retired file goes, and the file
/// that replaced it stays.
#[test]
fn a_retired_file_is_deleted_once_its_tombstone_is_past_the_window() {
    let dispatch = dispatch(2);
    let db = TempDir::new().unwrap();
    let adopted_dir = db.path().join("events");
    std::fs::create_dir_all(&adopted_dir).unwrap();
    write_parquet(&adopted_dir.join("a.parquet"));
    let datastore = PivotlakeDatastore::open(&db.path().to_string_lossy(), &dispatch).unwrap();
    let table_dir = create_events_table(&datastore, &dispatch, db.path(), &adopted_dir);
    let name = SchemaQualifiedTableName::in_default_schema("events");
    let retired = table_dir.join(rewrite_table_files(&datastore, &name));
    let live = table_dir.join(rewrite_table_files(&datastore, &name));

    Arc::new(Vacuumer::new(DEFAULT_VACUUM_POLL, datastore.clone()))
        .vacuum_all(now_ms() + FIVE_HOURS_MS)
        .unwrap();

    assert!(
        !retired.exists(),
        "a file retired past the window is deleted"
    );
    assert!(live.exists(), "the file that replaced it is live and kept");
}
