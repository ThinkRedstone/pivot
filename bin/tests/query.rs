//! End-to-end blackbox tests: spin up the real `Server` (with a real
//! `dispatch` worker pool and `PivotlakeDatastore`), connect with a real Postgres
//! client (`tokio-postgres`), and exercise the full
//! `CREATE TABLE` → `SELECT` → wire-encoding flow.
//!
//! The shared server, serial guard, and connected client are all wired up by
//! the [`common::conn`] rstest fixture — tests just take `conn: Conn` and go
//! straight to setup/execute/assert.

mod common;

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow_array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use common::{Conn, conn, connect_client, server_port, table_dir};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rstest::rstest;
use tempfile::TempDir;
use tokio_postgres::{Client, SimpleQueryMessage};

/// Run `sql` and decode every `DataRow` in the response into
/// `Vec<Option<String>>` (text format). Non-row messages (e.g. `RowDescription`,
/// `CommandComplete`) are dropped — tests assert on data only.
async fn select_rows(client: &Client, sql: &str) -> Vec<Vec<Option<String>>> {
    let msgs = client.simple_query(sql).await.unwrap();
    msgs.into_iter()
        .filter_map(|m| match m {
            SimpleQueryMessage::Row(r) => Some(
                (0..r.len())
                    .map(|i| r.get(i).map(|s| s.to_string()))
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        })
        .collect()
}

/// Run `sql` and return the single value of its single row, for the counts and
/// aggregates a test asserts one number from.
async fn single_value(client: &Client, sql: &str) -> String {
    let rows = select_rows(client, sql).await;
    let [row] = rows.as_slice() else {
        panic!("expected one row from `{sql}`, got {}", rows.len());
    };
    row[0].clone().expect("a single non-null value")
}

/// The server-side message of a failed query. `tokio_postgres::Error` renders
/// as a bare "db error", so assertions on what the server said have to read the
/// `DbError` it carries.
fn extract_db_error_message(error: &tokio_postgres::Error) -> String {
    error
        .as_db_error()
        .map_or_else(|| error.to_string(), |db| db.message().to_string())
}

/// Write `batch` as a single parquet file inside a fresh tempdir and return
/// the directory (kept alive by the caller — drop it to clean up).
fn write_parquet(batch: &RecordBatch) -> TempDir {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("data.parquet");
    // The dispatch reader assumes snappy-compressed pages — match that.
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut writer =
        ArrowWriter::try_new(File::create(&path).unwrap(), batch.schema(), Some(props)).unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
    dir
}

/// (id BIGINT, name VARCHAR) batch with three rows.
fn people_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let id: ArrayRef = Arc::new(Int64Array::from(vec![1i64, 2, 3]));
    let name: ArrayRef = Arc::new(StringArray::from(vec!["alice", "bob", "carol"]));
    RecordBatch::try_new(schema, vec![id, name]).unwrap()
}

async fn create_people_table(client: &Client, table: &str, dir: &Path) {
    let path = dir.to_str().unwrap();
    client
        .simple_query(&format!(
            "CREATE TABLE {table} (id BIGINT, name VARCHAR) WITH (with_pre_existing_parquets = '{path}')"
        ))
        .await
        .unwrap();
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn create_table_and_query(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "people_filter", dir.path()).await;

    let rows = select_rows(&conn, "SELECT id, name FROM people_filter WHERE id = 2").await;

    assert_eq!(rows, vec![vec![Some("2".into()), Some("bob".into())]]);
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn create_table_if_not_exists_is_a_noop_on_an_existing_table(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "people_if_not_exists", dir.path()).await;

    conn.simple_query("CREATE TABLE IF NOT EXISTS people_if_not_exists (id BIGINT, name VARCHAR)")
        .await
        .unwrap();

    let rows = select_rows(&conn, "SELECT COUNT(*) FROM people_if_not_exists").await;
    assert_eq!(rows, vec![vec![Some("3".into())]]);
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn drop_table_removes_the_table(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE drop_me (id BIGINT)")
        .await
        .unwrap();

    conn.simple_query("DROP TABLE drop_me").await.unwrap();

    let err = conn
        .simple_query("SELECT * FROM drop_me")
        .await
        .unwrap_err();
    let message = extract_db_error_message(&err);
    assert!(message.contains("drop_me"), "{message}");
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn drop_table_keeps_the_files_for_vacuum(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE drop_keeps_files (id BIGINT)")
        .await
        .unwrap();
    conn.simple_query("INSERT INTO drop_keeps_files VALUES (1)")
        .await
        .unwrap();
    let dir = table_dir("drop_keeps_files");
    let live = live_log_files(&dir);
    assert_eq!(live.len(), 1);

    conn.simple_query("DROP TABLE drop_keeps_files")
        .await
        .unwrap();

    // A query planned before the drop may still be reading the table, so the
    // drop leaves its storage in place; vacuum reclaims it after retention.
    for file in live {
        assert!(dir.join(file).exists());
    }
    assert!(dir.join("_delta_log").exists());
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn drop_then_recreate_serves_the_new_table(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE drop_phoenix (id BIGINT)")
        .await
        .unwrap();
    conn.simple_query("INSERT INTO drop_phoenix VALUES (1), (2)")
        .await
        .unwrap();
    // Prime the plan cache against the first incarnation.
    let rows = select_rows(&conn, "SELECT COUNT(*) FROM drop_phoenix").await;
    assert_eq!(rows, vec![vec![Some("2".into())]]);

    conn.simple_query("DROP TABLE drop_phoenix").await.unwrap();
    conn.simple_query("CREATE TABLE drop_phoenix (id BIGINT)")
        .await
        .unwrap();

    // The recreated table is empty; a cached plan bound to the dropped
    // incarnation must not answer for it.
    let rows = select_rows(&conn, "SELECT COUNT(*) FROM drop_phoenix").await;
    assert_eq!(rows, vec![vec![Some("0".into())]]);
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn drop_of_a_missing_table_errors(#[future] conn: Conn) {
    let err = conn
        .simple_query("DROP TABLE no_such_table_to_drop")
        .await
        .unwrap_err();

    let message = extract_db_error_message(&err);
    assert!(message.contains("no_such_table_to_drop"), "{message}");
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn drop_if_exists_of_a_missing_table_succeeds(#[future] conn: Conn) {
    conn.simple_query("DROP TABLE IF EXISTS no_such_table_to_drop")
        .await
        .unwrap();
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn drop_table_cascade_is_rejected(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE drop_cascade (id BIGINT)")
        .await
        .unwrap();

    let err = conn
        .simple_query("DROP TABLE drop_cascade CASCADE")
        .await
        .unwrap_err();

    let message = extract_db_error_message(&err);
    assert!(message.contains("CASCADE"), "{message}");
    let rows = select_rows(&conn, "SELECT COUNT(*) FROM drop_cascade").await;
    assert_eq!(rows, vec![vec![Some("0".into())]]);
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn system_tables_refreshes_after_catalog_changes(#[future] conn: Conn) {
    let sql = "SELECT datastore, schema, name \
               FROM system.tables \
               WHERE name = 'system_catalog_refresh'";

    assert!(select_rows(&conn, sql).await.is_empty());

    conn.simple_query("CREATE TABLE system_catalog_refresh (id BIGINT)")
        .await
        .unwrap();

    assert_eq!(
        select_rows(&conn, sql).await,
        vec![vec![
            Some("default".into()),
            Some("main".into()),
            Some("system_catalog_refresh".into()),
        ]],
    );
}

/// `system` is a datastore, not a schema every datastore carries: its tables
/// answer to their own qualified name and to no other datastore's.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn system_tables_belong_to_the_system_datastore(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE system_qualified (id BIGINT)")
        .await
        .unwrap();

    let rows = select_rows(
        &conn,
        "SELECT name FROM system.main.tables WHERE name = 'system_qualified'",
    )
    .await;
    assert_eq!(rows, vec![vec![Some("system_qualified".into())]]);

    // `default` is a keyword, so the datastore it names has to be quoted.
    let shadowed = conn
        .simple_query("SELECT * FROM \"default\".system.tables")
        .await
        .unwrap_err();
    let shadowed_message = extract_db_error_message(&shadowed).to_lowercase();
    assert!(
        shadowed_message.contains("system") && shadowed_message.contains("does not exist"),
        "a datastore must not carry a `system` schema of its own: {shadowed_message}",
    );
}

/// `system.table_files` reports the files a table holds, keyed by the same id
/// `system.tables` gives that table, so the two join.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn system_table_files_lists_the_files_of_a_table(#[future] conn: Conn) {
    let sql = "SELECT f.bytes \
               FROM system.table_files f JOIN system.tables t ON f.table_id = t.id \
               WHERE t.name = 'system_files'";
    conn.simple_query("CREATE TABLE system_files (id BIGINT)")
        .await
        .unwrap();
    assert!(select_rows(&conn, sql).await.is_empty());

    conn.simple_query("INSERT INTO system_files VALUES (1), (2)")
        .await
        .unwrap();

    let sizes: Vec<i64> = select_rows(&conn, sql)
        .await
        .into_iter()
        .map(|row| row[0].as_ref().unwrap().parse().unwrap())
        .collect();
    assert!(
        !sizes.is_empty() && sizes.iter().all(|&size| size > 0),
        "expected the inserted file to be listed with its size, got: {sizes:?}",
    );
}

/// Every listed file exposes its typed per-column lower and upper bounds in one
/// variant object. Files may be split across workers, so the union of their
/// ranges is asserted rather than assuming one particular file layout.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn system_table_files_reports_column_bounds(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE system_file_bounds (id BIGINT, name VARCHAR)")
        .await
        .unwrap();
    conn.simple_query("INSERT INTO system_file_bounds VALUES (3, 'c'), (1, 'a'), (2, 'b')")
        .await
        .unwrap();

    let rows = select_rows(
        &conn,
        "SELECT f.min_max_stats \
         FROM system.table_files f JOIN system.tables t ON f.table_id = t.id \
         WHERE t.name = 'system_file_bounds'",
    )
    .await;
    assert!(!rows.is_empty(), "the inserted files carry bounds");

    let mut min_ids = Vec::new();
    let mut max_ids = Vec::new();
    let mut min_names = Vec::new();
    let mut max_names = Vec::new();
    for row in rows {
        let stats: serde_json::Value = serde_json::from_str(row[0].as_deref().unwrap()).unwrap();
        min_ids.push(
            stats["id"]["min"]
                .as_i64()
                .expect("id minimum stays numeric"),
        );
        max_ids.push(
            stats["id"]["max"]
                .as_i64()
                .expect("id maximum stays numeric"),
        );
        min_names.push(
            stats["name"]["min"]
                .as_str()
                .expect("name minimum stays textual")
                .to_string(),
        );
        max_names.push(
            stats["name"]["max"]
                .as_str()
                .expect("name maximum stays textual")
                .to_string(),
        );
    }

    assert_eq!(min_ids.into_iter().min(), Some(1));
    assert_eq!(max_ids.into_iter().max(), Some(3));
    assert_eq!(min_names.into_iter().min().as_deref(), Some("a"));
    assert_eq!(max_names.into_iter().max().as_deref(), Some("c"));

    let extracted = select_rows(
        &conn,
        "SELECT CAST(f.min_max_stats.id.min AS BIGINT), \
                CAST(f.min_max_stats.id.max AS BIGINT) \
         FROM system.table_files f JOIN system.tables t ON f.table_id = t.id \
         WHERE t.name = 'system_file_bounds'",
    )
    .await;
    let extracted_min = extracted
        .iter()
        .map(|row| row[0].as_deref().unwrap().parse::<i64>().unwrap())
        .min();
    let extracted_max = extracted
        .iter()
        .map(|row| row[1].as_deref().unwrap().parse::<i64>().unwrap())
        .max();
    assert_eq!((extracted_min, extracted_max), (Some(1), Some(3)));
}

/// `system.datastores` names every datastore the server serves, the one serving
/// the relation included.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn system_datastores_lists_every_served_datastore(#[future] conn: Conn) {
    let rows = select_rows(
        &conn,
        "SELECT name, type, data_path <> '' FROM system.datastores ORDER BY name",
    )
    .await;

    assert_eq!(
        rows,
        vec![
            vec![
                Some("default".into()),
                Some("pivotlake".into()),
                Some("t".into()),
            ],
            vec![
                Some("system".into()),
                Some("system".into()),
                Some("f".into()),
            ],
        ],
    );
}

/// `system.tables` reports how a table is laid out and what it holds, and
/// `system.columns` describes each of its columns, keyed by the same id the
/// table answers to.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn system_tables_and_columns_describe_a_table(#[future] conn: Conn) {
    conn.simple_query(
        "CREATE TABLE system_described (id BIGINT, region VARCHAR) \
         WITH (partition_by = 'region', sort_by = 'id')",
    )
    .await
    .unwrap();
    conn.simple_query("INSERT INTO system_described VALUES (1, 'eu'), (2, 'us')")
        .await
        .unwrap();

    let table = select_rows(
        &conn,
        "SELECT datastore, schema, sorting_keys, partition_key, total_rows, \
                bytes > 0, bytes_uncompressed > 0 \
         FROM system.tables WHERE name = 'system_described'",
    )
    .await;
    let columns = select_rows(
        &conn,
        "SELECT c.name, c.type, c.position, c.is_partition_key, c.is_sort_key, c.bytes > 0 \
         FROM system.columns c JOIN system.tables t ON c.table_id = t.id \
         WHERE t.name = 'system_described' ORDER BY c.position",
    )
    .await;

    assert_eq!(
        table,
        vec![vec![
            Some("default".into()),
            Some("main".into()),
            Some("id".into()),
            Some("region".into()),
            Some("2".into()),
            Some("t".into()),
            Some("t".into()),
        ]],
    );
    assert_eq!(
        columns,
        vec![
            vec![
                Some("id".into()),
                Some("BIGINT".into()),
                Some("0".into()),
                Some("f".into()),
                Some("t".into()),
                Some("t".into()),
            ],
            vec![
                Some("region".into()),
                Some("VARCHAR".into()),
                Some("1".into()),
                Some("t".into()),
                Some("f".into()),
                Some("t".into()),
            ],
        ],
    );
}

/// The system relations are described by the same relations they serve, and
/// keyed the same way: their id is the constant one the datastore mints, so a
/// join from `system.columns` reaches them exactly as it reaches a stored
/// table's columns.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn system_columns_describes_the_system_relations_by_their_id(#[future] conn: Conn) {
    let rows = select_rows(
        &conn,
        "SELECT t.id, c.name, c.type \
         FROM system.columns c JOIN system.tables t ON c.table_id = t.id \
         WHERE t.datastore = 'system' AND t.name = 'datastores' ORDER BY c.position",
    )
    .await;

    let id = "da7aba5e-5e75-4a11-ab1e-5e1ec7edda7a";
    assert_eq!(
        rows,
        vec![
            vec![Some(id.into()), Some("name".into()), Some("VARCHAR".into())],
            vec![Some(id.into()), Some("type".into()), Some("VARCHAR".into())],
            vec![
                Some(id.into()),
                Some("data_path".into()),
                Some("VARCHAR".into()),
            ],
        ],
    );
}

/// A partitioned table's files report the partition they hold, so the files of
/// one partition are a query rather than a path convention.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn system_table_files_reports_a_file_partition(#[future] conn: Conn) {
    conn.simple_query(
        "CREATE TABLE system_partitioned (id BIGINT, region VARCHAR) \
         WITH (partition_by = 'region')",
    )
    .await
    .unwrap();

    conn.simple_query("INSERT INTO system_partitioned VALUES (1, 'eu'), (2, 'us')")
        .await
        .unwrap();

    let partitions = select_rows(
        &conn,
        "SELECT f.partition \
         FROM system.table_files f JOIN system.tables t ON f.table_id = t.id \
         WHERE t.name = 'system_partitioned' ORDER BY f.partition",
    )
    .await;
    assert_eq!(
        partitions,
        vec![
            vec![Some("region=eu".into())],
            vec![Some("region=us".into())],
        ],
    );
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn system_datastore_is_read_only(#[future] conn: Conn) {
    let create_table = conn
        .simple_query("CREATE TABLE system.not_allowed (id BIGINT)")
        .await
        .unwrap_err();
    let create_table_message = extract_db_error_message(&create_table).to_lowercase();
    assert!(
        create_table_message.contains("system") && create_table_message.contains("read-only"),
        "expected a read-only system datastore error, got: {create_table_message}",
    );

    let insert = conn
        .simple_query(
            "INSERT INTO system.datastores \
             VALUES ('not_allowed', 'delta', '/tmp')",
        )
        .await
        .unwrap_err();
    let insert_message = extract_db_error_message(&insert).to_lowercase();
    assert!(
        insert_message.contains("does not support insert"),
        "expected a read-only virtual table error, got: {insert_message}",
    );

    let bare_tables = conn.simple_query("SELECT * FROM tables").await.unwrap_err();
    let bare_tables_message = extract_db_error_message(&bare_tables).to_lowercase();
    assert!(
        bare_tables_message.contains("tables") && bare_tables_message.contains("not exist"),
        "bare `tables` must not fall back to `system.tables`: {bare_tables_message}",
    );

    // `system` names Pivot's own datastore, which serves one schema, so this
    // reaches neither a DuckDB-internal catalog nor a second Pivot schema.
    let duckdb_system = conn
        .simple_query("SELECT * FROM system.information_schema.tables")
        .await
        .unwrap_err();
    let duckdb_system_message = extract_db_error_message(&duckdb_system).to_lowercase();
    assert!(
        duckdb_system_message.contains("information_schema")
            && duckdb_system_message.contains("does not exist"),
        "the system datastore must serve no other schema, got: {duckdb_system_message}",
    );
}

/// The rows must be the whole ring: every block once, none twice, none missed,
/// however many workers the scan is compiled across. Contiguous slots from zero
/// with no duplicates is exactly that.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn system_memory_blocks_reports_the_whole_ring_exactly_once(#[future] conn: Conn) {
    let slots: Vec<i64> = select_rows(&conn, "SELECT slot FROM system.memory_blocks ORDER BY slot")
        .await
        .into_iter()
        .map(|row| row[0].as_ref().unwrap().parse().unwrap())
        .collect();

    assert!(!slots.is_empty(), "the ring must have blocks");
    assert_eq!(
        slots,
        (0..slots.len() as i64).collect::<Vec<_>>(),
        "the workers together must report every block of the ring exactly once",
    );
}

/// `system.memory_blocks` reports one row per block of the memory ring, so
/// counting rows measures memory: every block is the same size, and every one
/// of them is accounted for under exactly one state.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn system_memory_blocks_accounts_for_every_block(#[future] conn: Conn) {
    let total: i64 = single_value(&conn, "SELECT count(*) FROM system.memory_blocks")
        .await
        .parse()
        .unwrap();

    let by_state = select_rows(
        &conn,
        "SELECT state, count(*), sum(bytes) \
         FROM system.memory_blocks GROUP BY state ORDER BY state",
    )
    .await;

    assert!(total > 0, "the ring must have blocks");
    let counted: i64 = by_state
        .iter()
        .map(|row| row[1].as_ref().unwrap().parse::<i64>().unwrap())
        .sum();
    assert_eq!(counted, total, "every block must have exactly one state");
    for row in &by_state {
        let state = row[0].as_ref().unwrap();
        assert!(
            ["free", "pinned", "compressed_cache", "decompressed_cache"].contains(&state.as_str()),
            "unexpected block state: {state}",
        );
        let blocks: i64 = row[1].as_ref().unwrap().parse().unwrap();
        let bytes: i64 = row[2].as_ref().unwrap().parse().unwrap();
        assert_eq!(bytes % blocks, 0, "blocks must all be one size");
    }
}

/// Reading a table puts its bytes in the caches, which the ring reports as
/// blocks lent to a cache rather than free ones.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn system_memory_blocks_shows_a_scan_caching_bytes(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "memory_blocks_people", dir.path()).await;

    conn.simple_query("SELECT name FROM memory_blocks_people")
        .await
        .unwrap();

    let cached: i64 = single_value(
        &conn,
        "SELECT count(*) FROM system.memory_blocks \
         WHERE state IN ('compressed_cache', 'decompressed_cache')",
    )
    .await
    .parse()
    .unwrap();
    assert!(cached > 0, "reading a table must leave its bytes cached");
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn insert_returns_affected_row_count(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "people_insert", dir.path()).await;

    let messages = conn
        .simple_query("INSERT INTO people_insert VALUES (4, 'dave'), (5, 'eve')")
        .await
        .unwrap();

    assert_eq!(messages.len(), 1, "INSERT should not return a data row");
    match &messages[0] {
        SimpleQueryMessage::CommandComplete(rows) => assert_eq!(*rows, 2),
        other => panic!("expected INSERT command completion, got {other:?}"),
    }

    let rows = select_rows(&conn, "SELECT id FROM people_insert ORDER BY id").await;
    assert_eq!(rows.len(), 5);
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn insert_stores_a_null_constant_as_a_null(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "people_null_insert", dir.path()).await;

    conn.simple_query("INSERT INTO people_null_insert VALUES (4, NULL), (NULL, NULL)")
        .await
        .unwrap();

    let typed_null = select_rows(
        &conn,
        "SELECT id, name FROM people_null_insert WHERE id = 4",
    )
    .await;
    assert_eq!(typed_null, vec![vec![Some("4".into()), None]]);
    let all_null = select_rows(
        &conn,
        "SELECT id, name FROM people_null_insert WHERE id IS NULL",
    )
    .await;
    assert_eq!(all_null, vec![vec![None, None]]);
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn insert_with_a_column_list_fills_only_the_named_columns(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE listed_insert (id BIGINT, name VARCHAR, note VARCHAR)")
        .await
        .unwrap();

    conn.simple_query("INSERT INTO listed_insert (note, id) VALUES ('first', 1)")
        .await
        .unwrap();

    let rows = select_rows(&conn, "SELECT id, name, note FROM listed_insert").await;
    assert_eq!(
        rows,
        vec![vec![Some("1".into()), None, Some("first".into())]]
    );
}

/// (k BIGINT, id BIGINT) table whose `k` is 3, NULL, 1, 2 for ids 1..=4, so
/// the id order of a query sorted by `k` shows where the NULL key landed.
async fn create_null_key_table(client: &Client, table: &str) {
    client
        .simple_query(&format!("CREATE TABLE {table} (k BIGINT, id BIGINT)"))
        .await
        .unwrap();
    client
        .simple_query(&format!(
            "INSERT INTO {table} VALUES (3, 1), (NULL, 2), (1, 3), (2, 4)"
        ))
        .await
        .unwrap();
}

/// The single `id` column of `sql`'s rows, in result order.
async fn ids_in_order(client: &Client, sql: &str) -> Vec<String> {
    select_rows(client, sql)
        .await
        .into_iter()
        .map(|row| row[0].clone().expect("a non-null id"))
        .collect()
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn order_by_places_the_null_key_where_the_null_clause_says(#[future] conn: Conn) {
    create_null_key_table(&conn, "order_null_clause").await;

    let asc_first = ids_in_order(
        &conn,
        "SELECT id FROM order_null_clause ORDER BY k NULLS FIRST",
    )
    .await;
    let asc_last = ids_in_order(
        &conn,
        "SELECT id FROM order_null_clause ORDER BY k NULLS LAST",
    )
    .await;
    let desc_first = ids_in_order(
        &conn,
        "SELECT id FROM order_null_clause ORDER BY k DESC NULLS FIRST",
    )
    .await;
    let desc_last = ids_in_order(
        &conn,
        "SELECT id FROM order_null_clause ORDER BY k DESC NULLS LAST",
    )
    .await;

    assert_eq!(asc_first, ["2", "3", "4", "1"]);
    assert_eq!(asc_last, ["3", "4", "1", "2"]);
    assert_eq!(desc_first, ["2", "1", "4", "3"]);
    assert_eq!(desc_last, ["1", "4", "3", "2"]);
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn order_by_without_a_null_clause_puts_the_null_key_last(#[future] conn: Conn) {
    create_null_key_table(&conn, "order_null_default").await;

    let asc = ids_in_order(&conn, "SELECT id FROM order_null_default ORDER BY k").await;
    let desc = ids_in_order(&conn, "SELECT id FROM order_null_default ORDER BY k DESC").await;

    assert_eq!(asc, ["3", "4", "1", "2"]);
    assert_eq!(desc, ["1", "4", "3", "2"]);
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn order_by_limit_keeps_the_null_key_when_nulls_sort_first(#[future] conn: Conn) {
    create_null_key_table(&conn, "order_null_limit").await;

    let asc_first = ids_in_order(
        &conn,
        "SELECT id FROM order_null_limit ORDER BY k NULLS FIRST LIMIT 2",
    )
    .await;
    let desc_first = ids_in_order(
        &conn,
        "SELECT id FROM order_null_limit ORDER BY k DESC NULLS FIRST LIMIT 2",
    )
    .await;
    let asc_last = ids_in_order(
        &conn,
        "SELECT id FROM order_null_limit ORDER BY k NULLS LAST LIMIT 2",
    )
    .await;

    assert_eq!(asc_first, ["2", "3"]);
    assert_eq!(desc_first, ["2", "1"]);
    assert_eq!(asc_last, ["3", "4"]);
}

/// The key of row `i` in [`write_null_key_parquet`]: NULL on every
/// thousandth row, otherwise a spread of the ids that is far from file order.
fn null_key(i: i64) -> Option<i64> {
    (i % 1_000 != 0).then_some((i * 7919) % 1_000_003)
}

/// (k BIGINT, id BIGINT) with `rows` rows keyed by [`null_key`], written in
/// small row groups so a sort spans many batches and workers.
fn write_null_key_parquet(rows: i64) -> TempDir {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, true),
        Field::new("id", DataType::Int64, false),
    ]));
    let k: ArrayRef = Arc::new(Int64Array::from(
        (0..rows).map(null_key).collect::<Vec<_>>(),
    ));
    let id: ArrayRef = Arc::new(Int64Array::from((0..rows).collect::<Vec<_>>()));
    let batch = RecordBatch::try_new(schema, vec![k, id]).unwrap();
    let dir = TempDir::new().unwrap();
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_max_row_group_row_count(Some(4096))
        .build();
    let file = File::create(dir.path().join("data.parquet")).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    dir
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn order_by_limit_over_many_batches_honours_the_null_clause(#[future] conn: Conn) {
    let rows = 200_000;
    let dir = write_null_key_parquet(rows);
    let path = dir.path().to_str().unwrap();
    conn.simple_query(&format!(
        "CREATE TABLE many_null_keys (k BIGINT, id BIGINT) WITH (with_pre_existing_parquets = '{path}')"
    ))
    .await
    .unwrap();
    let mut non_null_keys: Vec<i64> = (0..rows).filter_map(null_key).collect();
    non_null_keys.sort_unstable();
    let null_rows = (rows - non_null_keys.len() as i64) as usize;

    let asc_first = ids_in_order(
        &conn,
        "SELECT id FROM many_null_keys ORDER BY k NULLS FIRST, id LIMIT 3",
    )
    .await;
    let desc_first = select_rows(
        &conn,
        &format!(
            "SELECT k FROM many_null_keys ORDER BY k DESC NULLS FIRST LIMIT {}",
            null_rows + 1
        ),
    )
    .await;
    let asc_last = select_rows(
        &conn,
        "SELECT k FROM many_null_keys ORDER BY k NULLS LAST LIMIT 2",
    )
    .await;
    let full = select_rows(&conn, "SELECT k FROM many_null_keys ORDER BY k NULLS FIRST").await;

    assert_eq!(asc_first, ["0", "1000", "2000"]);
    assert!(desc_first[..null_rows].iter().all(|row| row[0].is_none()));
    assert_eq!(
        desc_first[null_rows][0],
        Some(non_null_keys.last().unwrap().to_string())
    );
    assert_eq!(asc_last[0][0], Some(non_null_keys[0].to_string()));
    assert_eq!(asc_last[1][0], Some(non_null_keys[1].to_string()));
    assert!(full[..null_rows].iter().all(|row| row[0].is_none()));
    let sorted_keys: Vec<i64> = full[null_rows..]
        .iter()
        .map(|row| row[0].as_deref().unwrap().parse().unwrap())
        .collect();
    assert_eq!(sorted_keys, non_null_keys);
}

/// A variant column the statement leaves out has to travel the write path as a
/// variant all the same: it is a struct of two leaves that only the Arrow
/// extension tag tells apart from any other struct, and the shredding stage
/// reads that tag.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn insert_with_a_column_list_leaves_an_unnamed_variant_null(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE listed_variant (n BIGINT, j VARIANT)")
        .await
        .unwrap();

    conn.simple_query("INSERT INTO listed_variant (n) VALUES (1)")
        .await
        .unwrap();

    let rows = select_rows(&conn, "SELECT n, j FROM listed_variant").await;
    assert_eq!(rows, vec![vec![Some("1".into()), None]]);
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn second_connection_sees_table_created_by_first(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    let reader = connect_client(server_port()).await;
    create_people_table(&conn, "people_shared", dir.path()).await;

    let rows = select_rows(&reader, "SELECT id, name FROM people_shared").await;

    assert_eq!(
        rows,
        vec![
            vec![Some("1".into()), Some("alice".into())],
            vec![Some("2".into()), Some("bob".into())],
            vec![Some("3".into()), Some("carol".into())],
        ],
    );
}

/// Decode a one-row, one-column integer result (e.g. `SELECT drop_cache()`).
async fn select_one_i64(client: &Client, sql: &str) -> i64 {
    let rows = select_rows(client, sql).await;
    assert_eq!(rows.len(), 1, "expected exactly one row from `{sql}`");
    rows[0][0]
        .as_deref()
        .expect("non-null scalar")
        .parse()
        .expect("integer scalar")
}

/// `SELECT drop_cache()` evicts pivot's compressed cache: after a scan populates it,
/// the first drop reports ≥1 region freed and an immediate second drop reports
/// 0 (nothing left to evict). Serialised against the shared server by the
/// `conn` fixture, so no other query touches the cache in between.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn drop_cache_evicts_compressed_cache(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "people_drop_cache", dir.path()).await;

    // Scan the table so its parquet region lands in the compressed cache.
    let rows = select_rows(&conn, "SELECT id, name FROM people_drop_cache").await;
    assert_eq!(rows.len(), 3);

    // First drop frees the cached region(s); an immediate second drop finds none.
    let first = select_one_i64(&conn, "SELECT drop_cache()").await;
    assert!(first >= 1, "expected ≥1 region evicted, got {first}");
    let second = select_one_i64(&conn, "SELECT drop_cache()").await;
    assert_eq!(second, 0, "second drop should find an empty cache");
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn global_min_max_answers_from_metadata(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "people_minmax", dir.path()).await;

    let rows = select_rows(&conn, "SELECT MIN(id), MAX(id) FROM people_minmax").await;

    assert_eq!(rows, vec![vec![Some("1".into()), Some("3".into())]]);
}

/// The unfiltered global MIN/MAX short-circuits to parquet row-group stats and
/// never reads a data page, so it leaves the compressed cache empty. Proven by
/// clearing the cache (of any footer-load residue), running the aggregate, and
/// finding nothing to evict afterwards.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn global_min_max_does_not_scan(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "people_minmax_noscan", dir.path()).await;

    select_one_i64(&conn, "SELECT drop_cache()").await;
    select_rows(&conn, "SELECT MIN(id), MAX(id) FROM people_minmax_noscan").await;
    let evicted = select_one_i64(&conn, "SELECT drop_cache()").await;

    assert_eq!(evicted, 0, "metadata short-circuit must not scan any pages");
}

/// A WHERE clause excludes rows, so the metadata short-circuit is unsound and
/// must not fire: the aggregate returns the *filtered* extremes (not the whole
/// table's) and actually scans pages (the cache has something to evict after).
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn filtered_min_max_does_not_short_circuit(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "people_minmax_filtered", dir.path()).await;

    select_one_i64(&conn, "SELECT drop_cache()").await;
    let rows = select_rows(
        &conn,
        "SELECT MIN(id), MAX(id) FROM people_minmax_filtered WHERE id > 1",
    )
    .await;
    let evicted = select_one_i64(&conn, "SELECT drop_cache()").await;

    assert_eq!(rows, vec![vec![Some("2".into()), Some("3".into())]]);
    assert!(
        evicted >= 1,
        "a filtered aggregate must scan, not read stats"
    );
}

/// A global MIN/MAX over a DOUBLE column returns the exact float extremes. (This
/// also guards the old metadata short-circuit, which only fires for
/// integer/temporal columns: a float column declines it and computes from the
/// scan, so the bounds are never cast through Int64 and truncated to [3, 9].)
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn global_min_max_over_double_column(#[future] conn: Conn) {
    let schema = Arc::new(Schema::new(vec![Field::new("f", DataType::Float64, false)]));
    let f: ArrayRef = Arc::new(arrow_array::Float64Array::from(vec![3.7f64, 9.2, 5.0]));
    let dir = write_parquet(&RecordBatch::try_new(schema, vec![f]).unwrap());
    conn.simple_query(&format!(
        "CREATE TABLE doubles (f DOUBLE) WITH (with_pre_existing_parquets = '{}')",
        dir.path().to_str().unwrap()
    ))
    .await
    .unwrap();

    let rows = select_rows(&conn, "SELECT MIN(f), MAX(f) FROM doubles").await;

    let min: f64 = rows[0][0].as_deref().unwrap().parse().unwrap();
    let max: f64 = rows[0][1].as_deref().unwrap().parse().unwrap();
    assert_eq!(min, 3.7);
    assert_eq!(max, 9.2);
}

/// Grouped SUM/AVG over a DOUBLE value column, with a REAL column alongside, end
/// to end through the SQL server.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn grouped_float_aggregates(#[future] conn: Conn) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("g", DataType::Int64, false),
        Field::new("d", DataType::Float64, false),
        Field::new("r", DataType::Float32, false),
    ]));
    let g: ArrayRef = Arc::new(Int64Array::from(vec![1i64, 1, 2]));
    let d: ArrayRef = Arc::new(arrow_array::Float64Array::from(vec![1.5f64, 2.5, 4.0]));
    let r: ArrayRef = Arc::new(arrow_array::Float32Array::from(vec![1.0f32, 3.0, 5.0]));
    let dir = write_parquet(&RecordBatch::try_new(schema, vec![g, d, r]).unwrap());
    conn.simple_query(&format!(
        "CREATE TABLE fmetrics (g BIGINT, d DOUBLE, r REAL) WITH (with_pre_existing_parquets = '{}')",
        dir.path().to_str().unwrap()
    ))
    .await
    .unwrap();

    // Clear any cache residue from prior tests so the grouped aggregate has the
    // whole ring (the shared test server runs with a small buffer pool).
    select_one_i64(&conn, "SELECT drop_cache()").await;
    let rows = select_rows(
        &conn,
        "SELECT g, SUM(d), AVG(d), MIN(r) FROM fmetrics GROUP BY g ORDER BY g",
    )
    .await;

    let parse = |v: &Option<String>| -> f64 { v.as_deref().unwrap().parse().unwrap() };
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0][0].as_deref(), Some("1"));
    assert_eq!(parse(&rows[0][1]), 4.0); // SUM(d) group 1: 1.5 + 2.5
    assert_eq!(parse(&rows[0][2]), 2.0); // AVG(d) group 1
    assert_eq!(parse(&rows[0][3]), 1.0); // MIN(r) group 1
    assert_eq!(parse(&rows[1][1]), 4.0); // SUM(d) group 2
}

/// DuckDB plans an integer SUM as sum_no_overflow when statistics prove the
/// accumulation cannot overflow; a CASE over the constants 0 and 1 makes that
/// provable regardless of table contents, and the plan must still convert.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn integer_sum_planned_as_sum_no_overflow(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "sum_no_overflow_people", dir.path()).await;

    let rows = select_rows(
        &conn,
        "SELECT SUM(CASE WHEN name = 'alice' THEN 1 ELSE 0 END) FROM sum_no_overflow_people",
    )
    .await;

    assert_eq!(rows, vec![vec![Some("1".to_string())]]);
}

/// A DECIMAL column scans, filters against a decimal constant, and renders
/// scale-correct NUMERIC text on the wire. The file is written by arrow-rs,
/// which stores the decimal as FIXED_LEN_BYTE_ARRAY.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn decimal_scan_filter_and_wire_format(#[future] conn: Conn) {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "v",
        DataType::Decimal128(10, 2),
        false,
    )]));
    let v: ArrayRef = Arc::new(
        arrow_array::Decimal128Array::from(vec![1234i128, -500, 1050])
            .with_precision_and_scale(10, 2)
            .unwrap(),
    );
    let dir = write_parquet(&RecordBatch::try_new(schema, vec![v]).unwrap());
    conn.simple_query(&format!(
        "CREATE TABLE prices (v DECIMAL(10,2)) WITH (with_pre_existing_parquets = '{}')",
        dir.path().to_str().unwrap()
    ))
    .await
    .unwrap();

    let rows = select_rows(&conn, "SELECT v FROM prices WHERE v > 10.00 ORDER BY v").await;

    assert_eq!(
        rows,
        vec![vec![Some("10.50".into())], vec![Some("12.34".into())]]
    );
}

/// A TIMESTAMPTZ column scans from a file another executor wrote (arrow-rs
/// stamps `isAdjustedToUTC`), filters against a literal with an explicit
/// offset, and renders the UTC session's `+00` text on the wire.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn timestamptz_scan_filter_and_wire_format(#[future] conn: Conn) {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "occurred",
        DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some("UTC".into())),
        false,
    )]));
    let at: ArrayRef = Arc::new(
        arrow_array::TimestampMicrosecondArray::from(vec![
            0i64,
            1_577_836_800_000_000,
            1_577_836_800_500_000,
        ])
        .with_timezone("UTC"),
    );
    let dir = write_parquet(&RecordBatch::try_new(schema, vec![at]).unwrap());
    conn.simple_query(&format!(
        "CREATE TABLE tz_events (occurred TIMESTAMPTZ) WITH (with_pre_existing_parquets = '{}')",
        dir.path().to_str().unwrap()
    ))
    .await
    .unwrap();

    // 02:00:00+02 is 2020-01-01 00:00:00 UTC, so the epoch row drops.
    let rows = select_rows(
        &conn,
        "SELECT occurred FROM tz_events WHERE occurred >= TIMESTAMPTZ '2020-01-01 02:00:00+02' ORDER BY occurred",
    )
    .await;

    assert_eq!(
        rows,
        vec![
            vec![Some("2020-01-01 00:00:00+00".into())],
            vec![Some("2020-01-01 00:00:00.5+00".into())],
        ]
    );
}

/// A TIMESTAMPTZ column accepts offset-bearing INSERT values, stores their UTC
/// microseconds, and reports TIMESTAMPTZ on the PostgreSQL wire.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn timestamptz_columns_accept_offset_inserts(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE inserted_events (occurred TIMESTAMPTZ)")
        .await
        .unwrap();
    conn.simple_query(
        "INSERT INTO inserted_events VALUES \
         (TIMESTAMPTZ '2020-01-01 02:00:00.5+02'), \
         (TIMESTAMPTZ '1970-01-01 00:00:01+00')",
    )
    .await
    .unwrap();

    let rows = select_rows(
        &conn,
        "SELECT occurred FROM inserted_events ORDER BY occurred",
    )
    .await;

    assert_eq!(
        rows,
        vec![
            vec![Some("1970-01-01 00:00:01+00".into())],
            vec![Some("2020-01-01 00:00:00.5+00".into())],
        ]
    );
}

/// A table declared with a plain TIMESTAMP column reads a file whose leaf is
/// stamped UTC-adjusted: the declaration decides zone-ness, so the values
/// render without an offset suffix.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn declared_timestamp_overrides_a_files_utc_flag(#[future] conn: Conn) {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "occurred",
        DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some("UTC".into())),
        false,
    )]));
    let at: ArrayRef = Arc::new(
        arrow_array::TimestampMicrosecondArray::from(vec![1_577_836_800_000_000i64])
            .with_timezone("UTC"),
    );
    let dir = write_parquet(&RecordBatch::try_new(schema, vec![at]).unwrap());
    conn.simple_query(&format!(
        "CREATE TABLE naive_events (occurred TIMESTAMP) WITH (with_pre_existing_parquets = '{}')",
        dir.path().to_str().unwrap()
    ))
    .await
    .unwrap();

    let rows = select_rows(&conn, "SELECT occurred FROM naive_events").await;

    assert_eq!(rows, vec![vec![Some("2020-01-01 00:00:00".into())]]);
}

/// `now()` and `CURRENT_TIMESTAMP` are TIMESTAMP WITH TIME ZONE, so both
/// render in the UTC session with the offset suffix.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn now_and_current_timestamp_are_timestamptz(#[future] conn: Conn) {
    let rows = select_rows(&conn, "SELECT now(), CURRENT_TIMESTAMP").await;

    for value in &rows[0] {
        let rendered = value.as_deref().unwrap();
        assert!(
            rendered.ends_with("+00"),
            "expected a UTC offset suffix, got: {rendered}"
        );
    }
}

/// The session time zone is fixed to UTC: UTC spellings are accepted, any
/// other zone is refused rather than quietly served UTC answers.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn set_timezone_accepts_only_utc(#[future] conn: Conn) {
    conn.simple_query("SET TimeZone = 'UTC'").await.unwrap();
    conn.simple_query("SET timezone = 'GMT'").await.unwrap();
    conn.simple_query("RESET TimeZone").await.unwrap();

    let error = conn
        .simple_query("SET TimeZone = 'Europe/Amsterdam'")
        .await
        .unwrap_err();

    assert!(
        extract_db_error_message(&error).contains("UTC only"),
        "got: {error}"
    );
}

/// Global SUM/MIN/MAX over a DECIMAL column keep the exact fixed-point values
/// and the bound output scale; AVG comes back as a double.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn decimal_aggregates(#[future] conn: Conn) {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "v",
        DataType::Decimal128(10, 2),
        false,
    )]));
    let v: ArrayRef = Arc::new(
        arrow_array::Decimal128Array::from(vec![1234i128, -500, 1050])
            .with_precision_and_scale(10, 2)
            .unwrap(),
    );
    let dir = write_parquet(&RecordBatch::try_new(schema, vec![v]).unwrap());
    conn.simple_query(&format!(
        "CREATE TABLE decimal_metrics (v DECIMAL(10,2)) WITH (with_pre_existing_parquets = '{}')",
        dir.path().to_str().unwrap()
    ))
    .await
    .unwrap();

    let rows = select_rows(
        &conn,
        "SELECT SUM(v), MIN(v), MAX(v), AVG(v) FROM decimal_metrics",
    )
    .await;

    assert_eq!(rows[0][0].as_deref(), Some("17.84"));
    assert_eq!(rows[0][1].as_deref(), Some("-5.00"));
    assert_eq!(rows[0][2].as_deref(), Some("12.34"));
    let avg: f64 = rows[0][3].as_deref().unwrap().parse().unwrap();
    assert!((avg - 17.84 / 3.0).abs() < 1e-9);
}

/// GROUP BY a DECIMAL key: groups form on the exact fixed-point value and the
/// key column renders at its declared scale.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn group_by_decimal_key(#[future] conn: Conn) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("g", DataType::Decimal128(4, 1), false),
        Field::new("x", DataType::Int64, false),
    ]));
    let g: ArrayRef = Arc::new(
        arrow_array::Decimal128Array::from(vec![15i128, 15, 20])
            .with_precision_and_scale(4, 1)
            .unwrap(),
    );
    let x: ArrayRef = Arc::new(Int64Array::from(vec![1i64, 2, 3]));
    let dir = write_parquet(&RecordBatch::try_new(schema, vec![g, x]).unwrap());
    conn.simple_query(&format!(
        "CREATE TABLE decimal_groups (g DECIMAL(4,1), x BIGINT) WITH (with_pre_existing_parquets = '{}')",
        dir.path().to_str().unwrap()
    ))
    .await
    .unwrap();

    let rows = select_rows(
        &conn,
        "SELECT g, SUM(x) FROM decimal_groups GROUP BY g ORDER BY g",
    )
    .await;

    assert_eq!(
        rows,
        vec![
            vec![Some("1.5".into()), Some("3".into())],
            vec![Some("2.0".into()), Some("3".into())],
        ]
    );
}

/// An unfiltered global COUNT(*) is answered from the sum of parquet row-group
/// row counts, with no scan.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn global_count_star_answers_from_metadata(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "people_count", dir.path()).await;

    select_one_i64(&conn, "SELECT drop_cache()").await;
    let count = select_one_i64(&conn, "SELECT COUNT(*) FROM people_count").await;
    let evicted = select_one_i64(&conn, "SELECT drop_cache()").await;

    assert_eq!(count, 3);
    assert_eq!(evicted, 0, "count from metadata must not scan any pages");
}

/// A WHERE clause excludes rows, so the COUNT(*) short-circuit is unsound and
/// must not fire: the count reflects the filter, not the whole table.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn filtered_count_star_is_correct(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "people_count_filtered", dir.path()).await;

    let count = select_one_i64(
        &conn,
        "SELECT COUNT(*) FROM people_count_filtered WHERE id > 1",
    )
    .await;

    assert_eq!(count, 2);
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn query_against_unknown_table_errors(#[future] conn: Conn) {
    let err = conn
        .simple_query("SELECT id FROM does_not_exist")
        .await
        .unwrap_err();

    assert!(
        err.to_string().to_lowercase().contains("does_not_exist")
            || err.to_string().to_lowercase().contains("not found")
            || err.code().is_some(),
        "expected a structured pgwire error, got: {err}",
    );
}

/// Create `table` and fill it with `docs`, one document per row.
///
/// The rows go in through the server, so the files in the table's directory are
/// what our own write path produces: the column is annotated VARIANT in the
/// footer and its shredding is chosen from the documents, which is what a table
/// reading them back needs. A plain Arrow writer cannot stand in here.
async fn write_shredded_variant(conn: &Conn, table: &str, docs: &[String]) {
    conn.simple_query(&format!("CREATE TABLE {table} (j VARIANT)"))
        .await
        .unwrap();
    let values: Vec<String> = docs
        .iter()
        .map(|doc| format!("('{}')", doc.replace('\'', "''")))
        .collect();
    conn.simple_query(&format!("INSERT INTO {table} VALUES {}", values.join(",")))
        .await
        .unwrap();
}

/// The `typed_value` leaves of every Parquet file in `dir`: empty when nothing
/// was shredded, so a caller can tell an opaque document from a laid-out one.
fn shredded_leaves(dir: &Path) -> Vec<String> {
    let mut leaves = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "parquet") {
            continue;
        }
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&path).unwrap()).unwrap();
        for column in reader.parquet_schema().columns() {
            let name = column.path().string();
            if name.contains("typed_value") {
                leaves.push(name);
            }
        }
    }
    leaves
}

/// 80 documents carrying `a.x`, 20 carrying nothing but `kind`. The second kind
/// is the one that matters: a row with none of the shredded paths, which the
/// reader can only tell apart by the null mask it rebuilds over `typed_value`.
fn mixed_documents() -> Vec<String> {
    let mut docs: Vec<String> = (0..80)
        .map(|i| format!("{{\"kind\":\"commit\",\"a\":{{\"x\":{i}}}}}"))
        .collect();
    docs.extend((0..20).map(|_| "{\"kind\":\"identity\"}".to_string()));
    docs
}

/// Copying a variant from one table into another has to reassemble it: the
/// source hands over the shape *it* shredded into, while the destination shreds
/// for the rows it gets. Rows that have none of the source's shredded paths are
/// what makes this more than a copy, and they must survive it.
///
/// The copy goes through a select list that computes a column, so the variant
/// travels the path that rebuilds output fields rather than the one that passes
/// them through untouched.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn insert_from_a_shredded_variant_reassembles_and_reshreds(#[future] conn: Conn) {
    write_shredded_variant(&conn, "variant_source", &mixed_documents()).await;
    conn.simple_query("CREATE TABLE variant_copy (n BIGINT, j VARIANT)")
        .await
        .unwrap();

    // The computed `1` matters: a select list of nothing but column references
    // is copied through with its fields intact, while one that computes anything
    // rebuilds them, and that is where a variant column can lose the tag saying
    // it is one.
    conn.simple_query("INSERT INTO variant_copy SELECT 1, j FROM variant_source")
        .await
        .unwrap();

    let kinds = select_rows(
        &conn,
        "SELECT CAST(j->'kind' AS VARCHAR) AS kind, count(*) FROM variant_copy \
         GROUP BY kind ORDER BY kind",
    )
    .await;
    assert_eq!(
        kinds,
        vec![
            vec![Some("commit".into()), Some("80".into())],
            vec![Some("identity".into()), Some("20".into())],
        ],
        "every document survives the copy, including the ones with no shredded path"
    );

    // Restricted to the documents that have the path: an aggregate over the
    // rows that do not would be feeding it NULLs, which is a separate gap.
    let total = select_rows(
        &conn,
        "SELECT sum(CAST(j->'a'->'x' AS BIGINT)) FROM variant_copy \
         WHERE CAST(j->'kind' AS VARCHAR) = 'commit'",
    )
    .await;
    assert_eq!(total, vec![vec![Some("3160".into())]], "0..80 summed");

    assert!(
        !shredded_leaves(&table_dir("variant_copy")).is_empty(),
        "the copy shreds for itself; an unshredded write would answer the same \
         queries while storing every document opaque"
    );
}

/// Dot access is the syntax DuckDB folds into the scan's projection, so this
/// query reaches the reader as a pushed-down path rather than as an expression
/// above the scan. It has to answer exactly what the arrow syntax answers.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn dot_access_pushed_into_the_scan_reads_the_same_values(#[future] conn: Conn) {
    write_shredded_variant(&conn, "variant_dot", &mixed_documents()).await;

    let pushed = select_rows(
        &conn,
        "SELECT CAST(j.a.x AS BIGINT) AS x FROM variant_dot \
         WHERE CAST(j.kind AS VARCHAR) = 'commit' ORDER BY x LIMIT 3",
    )
    .await;
    let above_scan = select_rows(
        &conn,
        "SELECT CAST(j->'a'->'x' AS BIGINT) AS x FROM variant_dot \
         WHERE CAST(j->'kind' AS VARCHAR) = 'commit' ORDER BY x LIMIT 3",
    )
    .await;

    assert_eq!(
        pushed,
        vec![
            vec![Some("0".into())],
            vec![Some("1".into())],
            vec![Some("2".into())]
        ]
    );
    assert_eq!(
        pushed, above_scan,
        "the pushed path answers what the arrow syntax does"
    );
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn cte_read_twice_runs_once_and_feeds_both_readers(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "people_cte", dir.path()).await;

    // Grouping by a unique column gives one row per id, so the self-join below
    // pairs each row of the CTE with itself: a reader that saw only some of the
    // rows loses whole output rows rather than changing a value. DuckDB
    // materializes this CTE rather than inlining a copy per reference, because
    // it ends in an aggregate.
    let rows = select_rows(
        &conn,
        "WITH totals AS (SELECT id, count(*) AS n FROM people_cte GROUP BY id) \
         SELECT a.id, a.n, b.n FROM totals a, totals b WHERE a.id = b.id ORDER BY a.id",
    )
    .await;

    assert_eq!(
        rows,
        vec![
            vec![Some("1".into()), Some("1".into()), Some("1".into())],
            vec![Some("2".into()), Some("1".into()), Some("1".into())],
            vec![Some("3".into()), Some("1".into()), Some("1".into())],
        ]
    );
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn one_query_reads_two_ctes_including_one_built_from_the_other(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    create_people_table(&conn, "people_nested_cte", dir.path()).await;

    // `doubled` is built from `totals`, so its definition sits inside the other
    // CTE's body, and the query reads both. Each is read more than once and
    // ends in an aggregate, which is what makes DuckDB materialize the pair
    // instead of inlining a copy per reference.
    let rows = select_rows(
        &conn,
        "WITH totals AS (SELECT id, count(*) AS n FROM people_nested_cte GROUP BY id), \
              doubled AS (SELECT id, sum(n) * 2 AS n2 FROM totals GROUP BY id) \
         SELECT t1.id, t2.n, d1.n2, d2.n2 \
         FROM totals t1, totals t2, doubled d1, doubled d2 \
         WHERE t1.id = t2.id AND t1.id = d1.id AND t1.id = d2.id ORDER BY t1.id",
    )
    .await;

    assert_eq!(
        rows,
        vec![
            vec![
                Some("1".into()),
                Some("1".into()),
                Some("2".into()),
                Some("2".into())
            ],
            vec![
                Some("2".into()),
                Some("1".into()),
                Some("2".into()),
                Some("2".into())
            ],
            vec![
                Some("3".into()),
                Some("1".into()),
                Some("2".into()),
                Some("2".into())
            ],
        ]
    );
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn table_in_a_created_schema_is_queryable(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    let path = dir.path().to_str().unwrap();
    conn.simple_query("CREATE SCHEMA analytics").await.unwrap();

    conn.simple_query(&format!(
        "CREATE TABLE analytics.people (id BIGINT, name VARCHAR) WITH (with_pre_existing_parquets = '{path}')"
    ))
    .await
    .unwrap();

    let rows = select_rows(&conn, "SELECT name FROM analytics.people WHERE id = 2").await;
    assert_eq!(rows, vec![vec![Some("bob".into())]]);
}

/// A write names its table the same way a read does, so the rows an INSERT
/// produces have to land in the table the qualified name resolves to and come
/// back from it alongside the ones already there.
#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn insert_into_a_schema_qualified_table(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    let path = dir.path().to_str().unwrap();
    conn.simple_query("CREATE SCHEMA staffing").await.unwrap();
    conn.simple_query(&format!(
        "CREATE TABLE staffing.people (id BIGINT, name VARCHAR) WITH (with_pre_existing_parquets = '{path}')"
    ))
    .await
    .unwrap();

    conn.simple_query("INSERT INTO staffing.people VALUES (4, 'dave')")
        .await
        .unwrap();

    let rows = select_rows(&conn, "SELECT id, name FROM staffing.people ORDER BY id").await;
    assert_eq!(
        rows,
        vec![
            vec![Some("1".into()), Some("alice".into())],
            vec![Some("2".into()), Some("bob".into())],
            vec![Some("3".into()), Some("carol".into())],
            vec![Some("4".into()), Some("dave".into())],
        ],
    );
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn a_schema_qualifies_a_table_name(#[future] conn: Conn) {
    let dir = write_parquet(&people_batch());
    let path = dir.path().to_str().unwrap();
    conn.simple_query("CREATE SCHEMA reporting").await.unwrap();
    conn.simple_query(&format!(
        "CREATE TABLE reporting.staff (id BIGINT, name VARCHAR) WITH (with_pre_existing_parquets = '{path}')"
    ))
    .await
    .unwrap();

    // The same bare name in the default schema is a different table.
    let err = conn.simple_query("SELECT id FROM staff").await.unwrap_err();

    let message = extract_db_error_message(&err);
    assert!(
        message.to_lowercase().contains("staff"),
        "expected an unknown-table error naming `staff`, got: {message}",
    );
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn query_against_unknown_schema_errors(#[future] conn: Conn) {
    let err = conn
        .simple_query("SELECT id FROM no_such_schema.people")
        .await
        .unwrap_err();

    let message = extract_db_error_message(&err);
    assert!(
        message.contains("no_such_schema"),
        "expected the error to name the missing schema, got: {message}",
    );
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn create_table_in_an_unknown_schema_errors(#[future] conn: Conn) {
    let err = conn
        .simple_query("CREATE TABLE absent_schema.t (id BIGINT)")
        .await
        .unwrap_err();

    let message = extract_db_error_message(&err);
    assert!(
        message.contains("absent_schema"),
        "expected the error to name the missing schema, got: {message}",
    );
}

/// The set of data files the Delta log currently considers live: every `add`
/// path across the commits, minus every `remove` path.
fn live_log_files(dir: &std::path::Path) -> std::collections::HashSet<String> {
    let mut live = std::collections::HashSet::new();
    let mut commits: Vec<_> = std::fs::read_dir(dir.join("_delta_log"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|e| e == "json"))
        .collect();
    commits.sort();
    for commit in commits {
        for line in std::fs::read_to_string(&commit).unwrap().lines() {
            let action: serde_json::Value = serde_json::from_str(line).unwrap();
            if let Some(add) = action.get("add") {
                live.insert(add["path"].as_str().unwrap().to_string());
            }
            if let Some(remove) = action.get("remove") {
                live.remove(remove["path"].as_str().unwrap());
            }
        }
    }
    live
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn compact_final_merges_small_insert_files(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE compact_me (id BIGINT)")
        .await
        .unwrap();
    for i in 0..100 {
        conn.simple_query(&format!("INSERT INTO compact_me VALUES ({i})"))
            .await
            .unwrap();
    }
    let dir = table_dir("compact_me");

    conn.simple_query("COMPACT compact_me FINAL").await.unwrap();

    assert_eq!(live_log_files(&dir).len(), 1);
    let rows = select_rows(&conn, "SELECT COUNT(*), SUM(id) FROM compact_me").await;
    assert_eq!(rows, vec![vec![Some("100".into()), Some("4950".into())]]);
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn compact_of_a_missing_table_errors(#[future] conn: Conn) {
    let result = conn.simple_query("COMPACT no_such_table FINAL").await;

    let message = format!("{:?}", result.unwrap_err());
    assert!(message.contains("no table named"), "{message}");
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn vacuum_runs_without_background_maintenance(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE vacuum_me (id BIGINT)")
        .await
        .unwrap();
    let orphan = table_dir("vacuum_me").join("old.parquet");
    std::fs::write(&orphan, b"orphan").unwrap();
    File::open(&orphan)
        .unwrap()
        .set_times(
            std::fs::FileTimes::new().set_modified(
                std::time::SystemTime::now() - std::time::Duration::from_secs(5 * 3600),
            ),
        )
        .unwrap();
    let port = server_port();

    let messages = tokio::task::spawn_blocking(move || {
        let mut client = common::RawConn::connect(port);
        client.query("VACUUM;");
        client.read_until_ready()
    })
    .await
    .unwrap();

    assert_eq!(common::command_tag(&messages).as_deref(), Some("VACUUM"));
    assert!(!orphan.exists());
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn compact_accepts_a_fully_qualified_table_name(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE compact_qualified (id BIGINT)")
        .await
        .unwrap();
    for i in 0..100 {
        conn.simple_query(&format!("INSERT INTO compact_qualified VALUES ({i})"))
            .await
            .unwrap();
    }

    // The default datastore's name is also a reserved SQL keyword, so the
    // qualified form needs it quoted.
    conn.simple_query("COMPACT \"default\".main.compact_qualified FINAL")
        .await
        .unwrap();

    assert_eq!(live_log_files(&table_dir("compact_qualified")).len(), 1);
    let rows = select_rows(&conn, "SELECT COUNT(*) FROM compact_qualified").await;
    assert_eq!(rows, vec![vec![Some("100".into())]]);
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn transaction_statements_are_accepted(#[future] conn: Conn) {
    conn.simple_query("BEGIN").await.unwrap();
    conn.simple_query("COMMIT").await.unwrap();
    conn.simple_query("ROLLBACK").await.unwrap();
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn transaction_statements_work_over_the_extended_protocol(#[future] conn: Conn) {
    let affected = conn.execute("BEGIN", &[]).await.unwrap();

    assert_eq!(affected, 0);
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn statements_between_begin_and_rollback_stay_committed(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE transaction_noop (id BIGINT)")
        .await
        .unwrap();

    conn.simple_query("BEGIN").await.unwrap();
    conn.simple_query("INSERT INTO transaction_noop VALUES (1)")
        .await
        .unwrap();
    conn.simple_query("ROLLBACK").await.unwrap();

    let count = single_value(&conn, "SELECT COUNT(*) FROM transaction_noop").await;
    assert_eq!(count, "1");
}
