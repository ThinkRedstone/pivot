//! Wire-level `COPY <table> FROM STDIN WITH (FORMAT arrow)` tests: the
//! copy-in sub-protocol, schema conformance, and rollback on failure.

mod common;

use arrow_array::Array;
use common::{Conn, RawConn, command_tag, conn, select_rows, server_port};
use futures::SinkExt;
use rstest::rstest;
use std::sync::Arc;

/// Run `actions` against a fresh raw connection off the async runtime.
async fn with_raw_conn<T: Send + 'static>(
    actions: impl FnOnce(&mut RawConn) -> T + Send + 'static,
) -> T {
    let port = server_port();
    tokio::task::spawn_blocking(move || {
        let mut raw = RawConn::connect(port);
        actions(&mut raw)
    })
    .await
    .unwrap()
}

/// Serialize `batches` as one Arrow IPC stream, the bytes a client sends for
/// COPY ... WITH (FORMAT arrow).
fn arrow_stream_bytes(batches: &[arrow_array::RecordBatch]) -> Vec<u8> {
    let mut bytes = Vec::new();
    let mut writer =
        arrow::ipc::writer::StreamWriter::try_new(&mut bytes, &batches[0].schema()).unwrap();
    for batch in batches {
        writer.write(batch).unwrap();
    }
    writer.finish().unwrap();
    drop(writer);
    bytes
}

fn people_arrow_batch(ids: Vec<i64>, names: Vec<&str>) -> arrow_array::RecordBatch {
    let schema = Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("id", arrow_schema::DataType::Int64, true),
        arrow_schema::Field::new("name", arrow_schema::DataType::Utf8, true),
    ]));
    arrow_array::RecordBatch::try_new(
        schema,
        vec![
            Arc::new(arrow_array::Int64Array::from(ids)),
            Arc::new(arrow_array::StringArray::from(names)),
        ],
    )
    .unwrap()
}

fn single_column_batch(column: arrow_array::ArrayRef, name: &str) -> arrow_array::RecordBatch {
    let schema = Arc::new(arrow_schema::Schema::new(vec![arrow_schema::Field::new(
        name,
        column.data_type().clone(),
        true,
    )]));
    arrow_array::RecordBatch::try_new(schema, vec![column]).unwrap()
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn copy_arrow_loads_batches(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE copy_arrow (id BIGINT, name VARCHAR)")
        .await
        .unwrap();
    // Mixed string lengths on purpose: short values inline into string views,
    // while values past the inline limit reference the shared buffer, and
    // both paths must survive the cast to the table's physical type.
    let bytes = arrow_stream_bytes(&[
        people_arrow_batch(
            vec![1, 2],
            vec!["alice", "bob-with-a-name-well-past-inlining"],
        ),
        people_arrow_batch(vec![3], vec!["carol"]),
    ]);

    let messages = with_raw_conn(move |raw| {
        raw.query("COPY copy_arrow FROM STDIN WITH (FORMAT arrow)");
        let (kind, payload) = raw.read_message();
        assert_eq!(kind, b'G', "expected CopyInResponse");
        assert_eq!(payload[0], 1, "arrow copies advertise a binary payload");
        // Split mid-message so the decoder must carry state across frames.
        let split = bytes.len() / 2;
        raw.copy_data(&bytes[..split]);
        raw.copy_data(&bytes[split..]);
        raw.copy_done();
        raw.read_until_ready()
    })
    .await;

    assert_eq!(command_tag(&messages).as_deref(), Some("COPY 3"));
    let rows = select_rows(&conn, "SELECT id, name FROM copy_arrow ORDER BY id").await;
    assert_eq!(
        rows,
        vec![
            vec![Some("1".into()), Some("alice".into())],
            vec![
                Some("2".into()),
                Some("bob-with-a-name-well-past-inlining".into())
            ],
            vec![Some("3".into()), Some("carol".into())],
        ]
    );
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn copy_arrow_variant_column_loads_documents(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE copy_variant (n BIGINT, j VARIANT)")
        .await
        .unwrap();
    let documents: arrow_array::ArrayRef = Arc::new(arrow_array::StringArray::from(vec![
        Some(r#"{"kind":"commit","size":7}"#),
        Some(r#"{"kind":"identity"}"#),
        None,
    ]));
    let variants = parquet_variant_compute::json_to_variant(&documents)
        .unwrap()
        .into_inner();
    let schema = Arc::new(arrow_schema::Schema::new(vec![
        arrow_schema::Field::new("n", arrow_schema::DataType::Int64, true),
        arrow_schema::Field::new("j", variants.data_type().clone(), true),
    ]));
    let batch = arrow_array::RecordBatch::try_new(
        schema,
        vec![
            Arc::new(arrow_array::Int64Array::from(vec![1, 2, 3])),
            Arc::new(variants),
        ],
    )
    .unwrap();
    let bytes = arrow_stream_bytes(&[batch]);

    let messages = with_raw_conn(move |raw| {
        raw.query("COPY copy_variant FROM STDIN WITH (FORMAT arrow)");
        let (kind, _) = raw.read_message();
        assert_eq!(kind, b'G', "expected CopyInResponse");
        raw.copy_data(&bytes);
        raw.copy_done();
        raw.read_until_ready()
    })
    .await;

    assert_eq!(command_tag(&messages).as_deref(), Some("COPY 3"));
    let rows = select_rows(
        &conn,
        "SELECT n, CAST(j->'kind' AS VARCHAR) FROM copy_variant ORDER BY n",
    )
    .await;
    assert_eq!(
        rows,
        vec![
            vec![Some("1".into()), Some("commit".into())],
            vec![Some("2".into()), Some("identity".into())],
            vec![Some("3".into()), None],
        ]
    );
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn queries_run_on_the_same_connection_after_a_copy(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE copy_then_query (id BIGINT, name VARCHAR)")
        .await
        .unwrap();
    let bytes = arrow_stream_bytes(&[people_arrow_batch(vec![1, 2], vec!["alice", "bob"])]);

    let after = with_raw_conn(move |raw| {
        raw.query("COPY copy_then_query FROM STDIN WITH (FORMAT arrow)");
        let (kind, _) = raw.read_message();
        assert_eq!(kind, b'G', "expected CopyInResponse");
        raw.copy_data(&bytes);
        raw.copy_done();
        let copied = raw.read_until_ready();
        assert_eq!(command_tag(&copied).as_deref(), Some("COPY 2"));
        raw.query("SELECT COUNT(*) FROM copy_then_query");
        raw.read_until_ready()
    })
    .await;

    assert!(
        after.iter().any(|(kind, _)| *kind == b'T'),
        "the same connection answers queries after the copy"
    );
    let count = after
        .iter()
        .find(|(kind, _)| *kind == b'D')
        .expect("a data row");
    assert!(
        count.1.ends_with(b"2"),
        "the copied rows are visible to the follow-up query"
    );
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn empty_copy_commits_zero_rows(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE copy_empty (id BIGINT)")
        .await
        .unwrap();

    let messages = with_raw_conn(|raw| {
        raw.query("COPY copy_empty FROM STDIN WITH (FORMAT arrow)");
        let (kind, _) = raw.read_message();
        assert_eq!(kind, b'G', "expected CopyInResponse");
        raw.copy_done();
        raw.read_until_ready()
    })
    .await;

    assert_eq!(command_tag(&messages).as_deref(), Some("COPY 0"));
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn copy_arrow_column_list_null_fills_the_rest(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE copy_arrow_columns (id BIGINT, name VARCHAR, note VARCHAR)")
        .await
        .unwrap();
    let batch = single_column_batch(
        Arc::new(arrow_array::StringArray::from(vec!["dave"])),
        "name",
    );
    let bytes = arrow_stream_bytes(&[batch]);

    let messages = with_raw_conn(move |raw| {
        raw.query("COPY copy_arrow_columns (name) FROM STDIN WITH (FORMAT arrow)");
        let (kind, _) = raw.read_message();
        assert_eq!(kind, b'G', "expected CopyInResponse");
        raw.copy_data(&bytes);
        raw.copy_done();
        raw.read_until_ready()
    })
    .await;

    assert_eq!(command_tag(&messages).as_deref(), Some("COPY 1"));
    let rows = select_rows(&conn, "SELECT id, name, note FROM copy_arrow_columns").await;
    assert_eq!(rows, vec![vec![None, Some("dave".into()), None]]);
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn copy_arrow_width_mismatch_rolls_back(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE copy_arrow_bad (id BIGINT, name VARCHAR)")
        .await
        .unwrap();
    let batch = single_column_batch(Arc::new(arrow_array::Int64Array::from(vec![1])), "id");
    let bytes = arrow_stream_bytes(&[batch]);

    let messages = with_raw_conn(move |raw| {
        raw.query("COPY copy_arrow_bad FROM STDIN WITH (FORMAT arrow)");
        let (kind, _) = raw.read_message();
        assert_eq!(kind, b'G', "expected CopyInResponse");
        raw.copy_data(&bytes);
        raw.copy_done();
        raw.read_until_ready()
    })
    .await;

    assert!(messages.iter().any(|(kind, _)| *kind == b'E'));
    let rows = select_rows(&conn, "SELECT COUNT(*) FROM copy_arrow_bad").await;
    assert_eq!(rows, vec![vec![Some("0".into())]], "nothing was committed");
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn copy_arrow_bad_cast_reports_and_rolls_back(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE copy_arrow_cast (id BIGINT)")
        .await
        .unwrap();
    let batch = single_column_batch(
        Arc::new(arrow_array::StringArray::from(vec!["7", "seven"])),
        "id",
    );
    let bytes = arrow_stream_bytes(&[batch]);

    let messages = with_raw_conn(move |raw| {
        raw.query("COPY copy_arrow_cast FROM STDIN WITH (FORMAT arrow)");
        let (kind, _) = raw.read_message();
        assert_eq!(kind, b'G', "expected CopyInResponse");
        raw.copy_data(&bytes);
        raw.copy_done();
        let error_and_ready = raw.read_until_ready();
        raw.query("SELECT 1");
        (error_and_ready, raw.read_until_ready())
    })
    .await;

    let (error_and_ready, after) = messages;
    assert!(
        error_and_ready.iter().any(|(kind, _)| *kind == b'E'),
        "expected an ErrorResponse"
    );
    assert!(
        after.iter().any(|(kind, _)| *kind == b'T'),
        "the connection stays usable after the failed COPY"
    );
    let rows = select_rows(&conn, "SELECT COUNT(*) FROM copy_arrow_cast").await;
    assert_eq!(rows, vec![vec![Some("0".into())]], "nothing was committed");
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn copy_fail_discards_everything(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE copy_cancelled (id BIGINT, name VARCHAR)")
        .await
        .unwrap();
    let bytes = arrow_stream_bytes(&[people_arrow_batch(vec![1, 2], vec!["a", "b"])]);

    let messages = with_raw_conn(move |raw| {
        raw.query("COPY copy_cancelled FROM STDIN WITH (FORMAT arrow)");
        let (kind, _) = raw.read_message();
        assert_eq!(kind, b'G', "expected CopyInResponse");
        raw.copy_data(&bytes);
        raw.copy_fail("client changed its mind");
        raw.read_until_ready()
    })
    .await;

    assert!(messages.iter().any(|(kind, _)| *kind == b'E'));
    let rows = select_rows(&conn, "SELECT COUNT(*) FROM copy_cancelled").await;
    assert_eq!(rows, vec![vec![Some("0".into())]]);
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn copy_to_a_missing_table_fails_before_copy_mode(#[future] conn: Conn) {
    let messages = with_raw_conn(|raw| {
        raw.query("COPY copy_no_such_table FROM STDIN WITH (FORMAT arrow)");
        raw.read_until_ready()
    })
    .await;

    assert!(messages.iter().any(|(kind, _)| *kind == b'E'));
    assert!(
        !messages.iter().any(|(kind, _)| *kind == b'G'),
        "the connection must not enter copy mode"
    );
    drop(conn);
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn other_copy_forms_do_not_enter_copy_mode(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE copy_other_forms (id BIGINT)")
        .await
        .unwrap();

    for sql in [
        "COPY copy_other_forms TO STDOUT",
        "COPY copy_other_forms FROM 'data.csv'",
        "SELECT 1; COPY copy_other_forms FROM STDIN WITH (FORMAT arrow)",
        "COPY copy_other_forms (no_such_column) FROM STDIN WITH (FORMAT arrow)",
    ] {
        let messages = with_raw_conn(move |raw| {
            raw.query(sql);
            raw.read_until_ready()
        })
        .await;

        assert!(
            messages.iter().any(|(kind, _)| *kind == b'E'),
            "{sql}: expected an error"
        );
        assert!(
            !messages.iter().any(|(kind, _)| *kind == b'G'),
            "{sql}: must not enter copy mode"
        );
    }
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn copy_format_rejections_name_the_supported_format(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE copy_format_errors (id BIGINT)")
        .await
        .unwrap();

    for (sql, expected) in [
        (
            "COPY copy_format_errors FROM STDIN",
            "text format is not supported yet",
        ),
        (
            "COPY copy_format_errors FROM STDIN (FORMAT text)",
            "text format is not supported yet",
        ),
        (
            "COPY copy_format_errors FROM STDIN (FORMAT csv)",
            "only arrow is",
        ),
        (
            "COPY copy_format_errors FROM STDIN (FORMAT arrow, DELIMITER '|')",
            "not valid for FORMAT arrow",
        ),
    ] {
        let messages = with_raw_conn(move |raw| {
            raw.query(sql);
            raw.read_until_ready()
        })
        .await;

        let error = messages
            .iter()
            .find(|(kind, _)| *kind == b'E')
            .unwrap_or_else(|| panic!("{sql}: expected an error"));
        assert!(
            String::from_utf8_lossy(&error.1).contains(expected),
            "{sql}: error should mention \"{expected}\""
        );
        assert!(
            !messages.iter().any(|(kind, _)| *kind == b'G'),
            "{sql}: must not enter copy mode"
        );
    }
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn disconnect_mid_copy_commits_nothing(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE copy_dropped (id BIGINT, name VARCHAR)")
        .await
        .unwrap();
    let bytes = arrow_stream_bytes(&[people_arrow_batch(vec![1, 2], vec!["a", "b"])]);

    with_raw_conn(move |raw| {
        raw.query("COPY copy_dropped FROM STDIN WITH (FORMAT arrow)");
        let (kind, _) = raw.read_message();
        assert_eq!(kind, b'G', "expected CopyInResponse");
        raw.copy_data(&bytes);
        // The connection drops here without CopyDone.
    })
    .await;

    tokio::time::sleep(std::time::Duration::from_millis(300)).await;
    let rows = select_rows(&conn, "SELECT COUNT(*) FROM copy_dropped").await;
    assert_eq!(rows, vec![vec![Some("0".into())]]);
}

/// Stream one Arrow IPC payload through `tokio_postgres::copy_in` and return
/// the reported row count.
async fn copy_in_arrow(conn: &Conn, sql: &str, bytes: Vec<u8>) -> u64 {
    let sink = conn.copy_in(sql).await.unwrap();
    let mut sink = Box::pin(sink);
    sink.send(bytes::Bytes::from(bytes)).await.unwrap();
    sink.as_mut().finish().await.unwrap()
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn tokio_postgres_copy_in_loads_rows_over_the_extended_protocol(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE copy_extended (id BIGINT, name VARCHAR)")
        .await
        .unwrap();
    let bytes = arrow_stream_bytes(&[people_arrow_batch(vec![1, 2, 3], vec!["a", "b", "c"])]);

    let rows = copy_in_arrow(
        &conn,
        "COPY copy_extended FROM STDIN WITH (FORMAT arrow)",
        bytes,
    )
    .await;

    assert_eq!(rows, 3);
    let names = select_rows(&conn, "SELECT name FROM copy_extended ORDER BY id").await;
    assert_eq!(
        names,
        vec![
            vec![Some("a".into())],
            vec![Some("b".into())],
            vec![Some("c".into())]
        ]
    );
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn extended_protocol_runs_a_statement_and_repeated_copies(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE copy_repeated (id BIGINT, name VARCHAR)")
        .await
        .unwrap();
    let copy_sql = "COPY copy_repeated FROM STDIN WITH (FORMAT arrow)";

    let inserted = conn
        .execute("INSERT INTO copy_repeated VALUES (1, 'seed')", &[])
        .await
        .unwrap();
    let first_bytes = arrow_stream_bytes(&[people_arrow_batch(vec![2], vec!["first"])]);
    let first = copy_in_arrow(&conn, copy_sql, first_bytes).await;
    let second_bytes = arrow_stream_bytes(&[people_arrow_batch(vec![3], vec!["second"])]);
    let second = copy_in_arrow(&conn, copy_sql, second_bytes).await;

    assert_eq!((inserted, first, second), (1, 1, 1));
    let rows = select_rows(&conn, "SELECT name FROM copy_repeated ORDER BY id").await;
    assert_eq!(
        rows,
        vec![
            vec![Some("seed".into())],
            vec![Some("first".into())],
            vec![Some("second".into())]
        ]
    );
}

#[rstest]
#[awt]
#[tokio::test(flavor = "multi_thread")]
async fn prepare_describes_statement_shapes(#[future] conn: Conn) {
    conn.simple_query("CREATE TABLE copy_described (id BIGINT, name VARCHAR)")
        .await
        .unwrap();

    let select = conn
        .prepare("SELECT id, name FROM copy_described")
        .await
        .unwrap();
    let copy = conn
        .prepare("COPY copy_described FROM STDIN WITH (FORMAT arrow)")
        .await
        .unwrap();

    assert!(select.params().is_empty());
    let names: Vec<_> = select.columns().iter().map(|c| c.name()).collect();
    assert_eq!(names, ["id", "name"]);
    assert!(copy.params().is_empty());
    assert!(copy.columns().is_empty(), "COPY describes as NoData");
}
