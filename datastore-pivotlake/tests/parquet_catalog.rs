mod common;

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use arrow_array::cast::AsArray;
use arrow_array::types::Int32Type;
use arrow_array::{
    ArrayRef, Int8Array, Int16Array, Int32Array, Int64Array, RecordBatch, Scalar, StringArray,
    StringViewArray, TimestampMicrosecondArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use dispatch::{DataFlowDispatcher, Dispatch, Projection};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use tempfile::TempDir;

use catalog::datastore::{Datastore, DatastoreTransaction};
use catalog::{DEFAULT_DATASTORE_NAME, PivotCatalog};
use common::{commit_datastore_transaction, current_parquet};
use datastore_pivotlake::{ColumnStatFilter, PartitionEqFilter, PivotlakeDatastore, TableBinding};
use object_storage::ObjectPath;
use planner::PlanNode;
use planner::Planner;
use planner::catalog::{
    BoundTable, CatalogTransaction, Column, CreateTableRequest, Result as CatalogResult,
    SchemaQualifiedTableName,
};
use planner::expression::{Compare, CompareType, Expression, Function, Ref, VariantGet};
use planner::operator::{Input, Operator};
use planner::types::Type;

/// A shared single-worker dispatch pool for the whole test binary, handed to
/// each `PivotlakeDatastore` so `create_table` can read footers once (via the
/// metadata-fetch dataflow) when the table is defined.
fn dispatcher() -> DataFlowDispatcher {
    static DISPATCH: OnceLock<Dispatch> = OnceLock::new();
    // The compressed cache is process-global, so size it well above the suite's
    // distinct-file count to avoid cross-test eviction churn.
    DISPATCH
        .get_or_init(|| Dispatch::spin_up(1, 128, None))
        .dispatcher()
        .clone()
}

/// Open an empty datastore on a test-owned directory. Keeping the `TempDir`
/// guard next to the datastore makes its storage lifetime explicit.
fn empty_datastore() -> (TempDir, Arc<PivotlakeDatastore>) {
    let database = TempDir::new().unwrap();
    let datastore =
        PivotlakeDatastore::open(&database.path().to_string_lossy(), &dispatcher()).unwrap();
    (database, datastore)
}

/// Entries written to a datastore other than its process-lifetime lock.
fn durable_datastore_entries(database: &Path) -> Vec<std::ffi::OsString> {
    std::fs::read_dir(database)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .filter(|name| name != ".pivot.lock")
        .collect()
}

/// Create a table, mirroring how the server runs `CREATE TABLE`: execute the
/// footer-fetch dataflow, then commit the transaction that persists and
/// publishes the staged table. Returns the datastore's own result so error-path
/// tests can still assert on the `Err`.
fn create_table(
    datastore: &Arc<PivotlakeDatastore>,
    request: CreateTableRequest,
) -> CatalogResult<()> {
    let transaction = datastore.clone().begin_transaction();
    transaction
        .bind_create_table(request)?
        .compile(&dispatcher())?
        .collect()
        .map(|_| ())
        .map_err(|e| planner::catalog::Error::Other(Box::new(e)))?;
    commit_datastore_transaction(transaction)
}

/// Present a single `PivotlakeDatastore` to the planner as a one-datastore
/// [`PivotCatalog`], the way the server wraps its datastores.
fn single_catalog(datastore: &Arc<PivotlakeDatastore>) -> Arc<PivotCatalog> {
    Arc::new(
        PivotCatalog::new(
            HashMap::from([(
                DEFAULT_DATASTORE_NAME.to_string(),
                datastore.clone() as Arc<dyn Datastore>,
            )]),
            DEFAULT_DATASTORE_NAME.to_string(),
            common::trust_metastore(),
        )
        .unwrap(),
    )
}

/// Drive a transaction commit to completion. A commit that wrote files hops to
/// the blocking pool, so it needs a Tokio runtime to run under.
fn commit_transaction_blocking(transaction: Arc<dyn CatalogTransaction>) {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(transaction.commit())
        .unwrap();
}

/// Write a parquet file containing each batch as its own row group, so a
/// per-row-group min/max prune is observable.
fn write_one_row_per_group(batches: &[RecordBatch]) -> TempDir {
    let dir = TempDir::new().unwrap();
    let props = WriterProperties::builder()
        .set_statistics_enabled(EnabledStatistics::Chunk)
        .set_max_row_group_row_count(Some(1))
        .build();
    let path = dir.path().join("data.parquet");
    let file = File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, batches[0].schema(), Some(props)).unwrap();
    for batch in batches {
        writer.write(batch).unwrap();
    }
    writer.close().unwrap();
    dir
}

/// Three-row table with one (id, name) row per row group.
fn three_row_table() -> (TempDir, Vec<Column>) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8View, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![10, 20, 30])) as ArrayRef,
            Arc::new(StringViewArray::from(vec!["a", "b", "c"])) as ArrayRef,
        ],
    )
    .unwrap();
    let dir = write_one_row_per_group(&[batch]);
    let columns = vec![
        Column {
            name: "id".to_string(),
            col_type: Type::Int32,
        },
        Column {
            name: "name".to_string(),
            col_type: Type::Utf8,
        },
    ];
    (dir, columns)
}

fn create_request(name: &str, path: &Path, columns: Vec<Column>) -> CreateTableRequest {
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

/// A `CREATE TABLE name (cols)` request that adopts nothing: the table starts
/// empty and is filled through the engine's own write path.
fn empty_request(name: &str, columns: Vec<Column>) -> CreateTableRequest {
    CreateTableRequest {
        datastore_name: None,
        schema_name: None,
        name: name.to_string(),
        columns,
        options: HashMap::new(),
        if_not_exists: false,
    }
}

/// Where a created table keeps its own storage under the database root: its
/// Delta log and every file written into it.
fn table_dir(database: &TempDir, datastore: &PivotlakeDatastore, name: &str) -> std::path::PathBuf {
    common::table_dir(database.path(), datastore, name)
}

fn int_constant(v: i32) -> Scalar<ArrayRef> {
    Scalar::new(Arc::new(Int32Array::new_scalar(v).into_inner()) as ArrayRef)
}

fn col_neq_filter(column_idx: usize, constant: Scalar<ArrayRef>) -> Expression {
    constant_comparison(column_idx, CompareType::NotEqual, constant)
}

fn col_eq_filter(column_idx: usize, constant: Scalar<ArrayRef>) -> Expression {
    constant_comparison(column_idx, CompareType::Equal, constant)
}

fn constant_comparison(
    column_idx: usize,
    compare_type: CompareType,
    constant: Scalar<ArrayRef>,
) -> Expression {
    // Build the same shape DuckDB pushes through the C++ bridge:
    // `Compare { Ref, Constant }`.
    Expression::Compare(Compare {
        left: Box::new(Expression::Ref(Ref {
            column_idx,
            return_type: Type::Int32,
            name: None,
        })),
        right: Box::new(Expression::Constant(constant)),
        compare_type,
        return_type: Type::Boolean,
    })
}

// Row groups that survive `table`'s pushed-down predicates over the table's
// current files (what `compile` would scan). Pruning is a pure in-memory filter
// — no dispatcher needed.
fn row_group_count(datastore: &PivotlakeDatastore, name: &str, table: &TableBinding) -> usize {
    table
        .pruned_parquet(&current_parquet(datastore, name))
        .row_groups()
        .len()
}

fn first_input(node: &PlanNode) -> Option<&Input> {
    if let Operator::Input(input) = &node.operator {
        return Some(input);
    }
    node.inputs.iter().find_map(first_input)
}

#[test]
fn create_table_succeeds_with_valid_path() {
    let (dir, columns) = three_row_table();
    let (_database, datastore) = empty_datastore();
    create_table(&datastore, create_request("t", dir.path(), columns)).unwrap();
    assert!(
        datastore
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
fn create_table_is_durable_and_visible_only_after_commit() {
    let (dir, columns) = three_row_table();
    let (database, datastore) = empty_datastore();
    let transaction = datastore.clone().begin_transaction();
    transaction
        .bind_create_table(create_request("t", dir.path(), columns))
        .unwrap()
        .compile(&dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    assert!(
        datastore
            .clone()
            .begin_transaction()
            .table(
                DEFAULT_DATASTORE_NAME,
                &SchemaQualifiedTableName::in_default_schema("t"),
            )
            .is_none()
    );
    assert!(durable_datastore_entries(database.path()).is_empty());

    commit_datastore_transaction(transaction).unwrap();

    assert!(
        datastore
            .clone()
            .begin_transaction()
            .table(
                DEFAULT_DATASTORE_NAME,
                &SchemaQualifiedTableName::in_default_schema("t"),
            )
            .is_some()
    );
    assert!(
        table_dir(&database, &datastore, "t")
            .join("_delta_log/00000000000000000000.json")
            .exists()
    );
    // The adopted files' own directory holds nothing but them.
    assert!(!dir.path().join("_delta_log").exists());
}

#[test]
fn rolling_back_create_table_discards_the_staged_creation() {
    let (dir, columns) = three_row_table();
    let (database, datastore) = empty_datastore();
    let transaction = datastore.clone().begin_transaction();
    transaction
        .bind_create_table(create_request("t", dir.path(), columns))
        .unwrap()
        .compile(&dispatcher())
        .unwrap()
        .collect()
        .unwrap();

    transaction.rollback();

    assert!(
        datastore
            .clone()
            .begin_transaction()
            .table(
                DEFAULT_DATASTORE_NAME,
                &SchemaQualifiedTableName::in_default_schema("t"),
            )
            .is_none()
    );
    assert!(!dir.path().join("_delta_log").exists());
    assert!(durable_datastore_entries(database.path()).is_empty());
}

#[test]
fn create_duplicate_table_fails() {
    let (dir, columns) = three_row_table();
    let (_database, datastore) = empty_datastore();
    create_table(&datastore, create_request("t", dir.path(), columns.clone())).unwrap();

    let err = create_table(&datastore, create_request("t", dir.path(), columns))
        .unwrap_err()
        .to_string();

    assert!(
        err.contains("already exists"),
        "expected duplicate rejection: {err}"
    );
}

#[test]
fn create_table_if_not_exists_keeps_the_existing_table() {
    let (dir, columns) = three_row_table();
    let (_database, datastore) = empty_datastore();
    create_table(&datastore, create_request("t", dir.path(), columns.clone())).unwrap();

    // A path-less request would produce an empty table if it actually created
    // one, so surviving row groups prove the statement was a no-op.
    let request = CreateTableRequest {
        datastore_name: None,
        schema_name: None,
        name: "t".to_string(),
        columns,
        options: HashMap::new(),
        if_not_exists: true,
    };
    create_table(&datastore, request).unwrap();

    assert_eq!(current_parquet(&datastore, "t").row_groups().len(), 3);
}

#[test]
fn create_table_without_a_path_makes_an_empty_table() {
    let (_dir, columns) = three_row_table();
    let (_database, datastore) = empty_datastore();
    let req = CreateTableRequest {
        datastore_name: None,
        schema_name: None,
        name: "t".to_string(),
        columns,
        options: HashMap::new(),
        if_not_exists: false,
    };
    // No path: the table lives under the (in-memory) database root with no data.
    create_table(&datastore, req).unwrap();
    assert!(
        datastore
            .clone()
            .begin_transaction()
            .table(
                DEFAULT_DATASTORE_NAME,
                &SchemaQualifiedTableName::in_default_schema("t"),
            )
            .is_some()
    );
    assert!(current_parquet(&datastore, "t").row_groups().is_empty());
}

#[test]
fn opening_an_unwritable_database_fails() {
    // Opening takes and records the datastore lock before loading catalog state,
    // so an unwritable root is rejected immediately.
    let bogus = "/definitely/not/a/real/path/for/datastore/tests";
    let err = PivotlakeDatastore::open(bogus, &dispatcher())
        .unwrap_err()
        .to_string();

    let err_lower = err.to_lowercase();
    assert!(
        err_lower.contains("permission denied") || err_lower.contains("read-only file system"),
        "expected the lock-file write failure, got: {err}"
    );
}

#[test]
fn create_table_fails_when_the_adopt_path_is_a_file() {
    let (dir, columns) = three_row_table();
    let file_path = dir.path().join("data.parquet");
    let (_database, datastore) = empty_datastore();
    // Listing a file as a directory fails, so the create errors.
    let err = create_table(&datastore, create_request("t", &file_path, columns))
        .unwrap_err()
        .to_string();
    assert!(
        err.to_lowercase().contains("not a directory"),
        "expected not-a-directory error: {err}"
    );
}

#[test]
fn create_table_rejects_a_url_path() {
    let (_dir, columns) = three_row_table();
    let (_database, datastore) = empty_datastore();
    // The adopted files live in the database's own storage, not the path's, so a
    // scheme is rejected regardless of the database's storage class.
    let req = CreateTableRequest {
        datastore_name: None,
        schema_name: None,
        name: "t".to_string(),
        columns,
        options: HashMap::from([(
            "with_pre_existing_parquets".to_string(),
            "s3://bucket/data".to_string(),
        )]),
        if_not_exists: false,
    };
    let err = create_table(&datastore, req).unwrap_err().to_string();
    assert!(
        err.contains("not a URL"),
        "expected scheme rejection: {err}"
    );
}

/// An option this datastore does not implement is an error, not something to
/// drop: a statement carried out with one of its clauses ignored produces a
/// table nobody asked for. `path`, the option `with_pre_existing_parquets` replaced, is
/// exactly the case that matters.
#[test]
fn create_table_rejects_an_unknown_option() {
    let (dir, columns) = three_row_table();
    let (_database, datastore) = empty_datastore();
    let req = CreateTableRequest {
        datastore_name: None,
        schema_name: None,
        name: "t".to_string(),
        columns,
        options: HashMap::from([(
            "path".to_string(),
            dir.path().to_string_lossy().into_owned(),
        )]),
        if_not_exists: false,
    };

    let err = create_table(&datastore, req).unwrap_err().to_string();

    assert!(
        err.contains("`path` is not a table option"),
        "expected the unknown option to be named: {err}"
    );
    assert!(
        err.contains("with_pre_existing_parquets"),
        "expected the known options to be listed: {err}"
    );
}

/// An adopt directory holding no Parquet is not an error: the table is created
/// empty, exactly as one that names no directory at all, and fills up as it is
/// written to.
#[test]
fn create_table_over_an_empty_adopt_directory_makes_an_empty_table() {
    let empty = TempDir::new().unwrap();
    let (_dir, columns) = three_row_table();
    let (_database, datastore) = empty_datastore();

    create_table(&datastore, create_request("t", empty.path(), columns)).unwrap();

    assert!(current_parquet(&datastore, "t").row_groups().is_empty());
    run_sql(&datastore, "INSERT INTO t VALUES (1, 'one')");
    datastore.refresh_from_store().unwrap();
    assert_eq!(
        common::extract_count(&run_sql(&datastore, "SELECT COUNT(*) FROM t")),
        1
    );
}

#[test]
fn pushdown_filter_always_returns_false() {
    let (dir, columns) = three_row_table();
    let (_database, datastore) = empty_datastore();
    create_table(&datastore, create_request("t", dir.path(), columns)).unwrap();
    let mut table = datastore
        .clone()
        .begin_transaction()
        .table(
            DEFAULT_DATASTORE_NAME,
            &SchemaQualifiedTableName::in_default_schema("t"),
        )
        .unwrap();
    let pushed = table
        .pushdown_filter(col_neq_filter(0, int_constant(20)))
        .unwrap();
    assert!(!pushed);
}

#[test]
fn pushdown_filter_prunes_row_group_with_only_excluded_value() {
    let (dir, columns) = three_row_table();
    let (_database, datastore) = empty_datastore();
    create_table(&datastore, create_request("t", dir.path(), columns)).unwrap();

    let mut table = datastore
        .clone()
        .begin_transaction()
        .table(
            DEFAULT_DATASTORE_NAME,
            &SchemaQualifiedTableName::in_default_schema("t"),
        )
        .unwrap();
    assert_eq!(row_group_count(&datastore, "t", &table), 3);

    table
        .pushdown_filter(col_neq_filter(0, int_constant(20)))
        .unwrap();

    // Row group whose single value is 20 has min == max == 20 and is pruned.
    assert_eq!(row_group_count(&datastore, "t", &table), 2);
}

/// DuckDB binds a comparison between a zone-less TIMESTAMP column and a
/// TIMESTAMPTZ constant as `CAST(column AS TIMESTAMPTZ) > constant`. The
/// planner connection is UTC, so the binding can peel that identity cast and
/// prune against the column's native timestamp statistics.
#[test]
fn timestamp_cast_pushdown_prunes_native_timestamp_row_groups() {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "occurred",
        DataType::Timestamp(TimeUnit::Microsecond, None),
        false,
    )]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(TimestampMicrosecondArray::from(vec![
            0_i64,
            60_000_000,
            120_000_000,
        ])) as ArrayRef],
    )
    .unwrap();
    let dir = write_one_row_per_group(&[batch]);
    let (_database, datastore) = empty_datastore();
    create_table(
        &datastore,
        create_request(
            "events",
            dir.path(),
            vec![Column {
                name: "occurred".to_string(),
                col_type: Type::Timestamp,
            }],
        ),
    )
    .unwrap();

    let catalog = single_catalog(&datastore);
    let transaction = catalog.begin_transaction();
    let mut planner = Planner::from_datastore_names(
        vec![DEFAULT_DATASTORE_NAME.to_string()],
        DEFAULT_DATASTORE_NAME.to_string(),
    )
    .unwrap();
    let plan = planner
        .plan(
            "SELECT occurred FROM events \
             WHERE occurred > TIMESTAMPTZ '1970-01-01 00:01:30+00'",
            transaction,
        )
        .unwrap();
    let input = first_input(&plan.root).expect("query has a table scan");

    // Compile the scan by itself, before the plan's runtime Filter. Only the
    // 120-second row group should survive statistics pruning.
    let batches = input
        .table
        .compile_scan(&dispatcher(), Projection::all(0), Vec::new(), false)
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 1);
}

/// A table of one zone-less TIMESTAMP column with the values 0s, 60s and
/// 120s, each in its own row group, so a range prune is observable.
fn minute_events_table() -> (TempDir, Arc<PivotlakeDatastore>) {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "occurred",
        DataType::Timestamp(TimeUnit::Microsecond, None),
        false,
    )]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(TimestampMicrosecondArray::from(vec![
            0_i64,
            60_000_000,
            120_000_000,
        ])) as ArrayRef],
    )
    .unwrap();
    let dir = write_one_row_per_group(&[batch]);
    let (database, datastore) = empty_datastore();
    create_table(
        &datastore,
        create_request(
            "events",
            dir.path(),
            vec![Column {
                name: "occurred".to_string(),
                col_type: Type::Timestamp,
            }],
        ),
    )
    .unwrap();
    (database, datastore)
}

/// Rows the query's table scan produces on its own, before the plan's runtime
/// Filter: what statistics pruning left of the table.
fn scanned_row_count(datastore: &Arc<PivotlakeDatastore>, sql: &str) -> usize {
    let catalog = single_catalog(datastore);
    let mut planner = Planner::from_datastore_names(
        vec![DEFAULT_DATASTORE_NAME.to_string()],
        DEFAULT_DATASTORE_NAME.to_string(),
    )
    .unwrap();
    let plan = planner.plan(sql, catalog.begin_transaction()).unwrap();
    let input = first_input(&plan.root).expect("query has a table scan");
    input
        .table
        .compile_scan(&dispatcher(), Projection::all(0), Vec::new(), false)
        .unwrap()
        .collect()
        .unwrap()
        .iter()
        .map(RecordBatch::num_rows)
        .sum()
}

/// DuckDB offers a two-sided range on one column as a single BETWEEN, so both
/// bounds must prune, not just a lone comparison.
#[test]
fn between_pushdown_prunes_row_groups_outside_both_bounds() {
    let (_database, datastore) = minute_events_table();

    let scanned = scanned_row_count(
        &datastore,
        "SELECT occurred FROM events \
         WHERE occurred BETWEEN '1970-01-01 00:00:30' AND '1970-01-01 00:01:30'",
    );

    assert_eq!(scanned, 1);
}

/// A lower and an upper comparison on the same column are folded into a
/// BETWEEN before pushdown, so they prune exactly like the BETWEEN spelling.
#[test]
fn paired_comparisons_pushdown_prune_like_between() {
    let (_database, datastore) = minute_events_table();

    let scanned = scanned_row_count(
        &datastore,
        "SELECT occurred FROM events \
         WHERE occurred >= '1970-01-01 00:00:30' AND occurred <= '1970-01-01 00:01:30'",
    );

    assert_eq!(scanned, 1);
}

/// A BETWEEN against TIMESTAMPTZ bounds wraps the column in the UTC identity
/// cast; both bounds still prune the native timestamp statistics.
#[test]
fn timestamp_cast_between_pushdown_prunes_native_timestamp_row_groups() {
    let (_database, datastore) = minute_events_table();

    let scanned = scanned_row_count(
        &datastore,
        "SELECT occurred FROM events \
         WHERE occurred BETWEEN TIMESTAMPTZ '1970-01-01 00:00:30+00' \
         AND TIMESTAMPTZ '1970-01-01 00:01:30+00'",
    );

    assert_eq!(scanned, 1);
}

/// Each bind hands out a fresh clone, so pushdown applied to one binding
/// must not leak into a subsequent one.
#[test]
fn second_bind_is_independent_of_first_bind_pushdown() {
    let (dir, columns) = three_row_table();
    let (_database, datastore) = empty_datastore();
    create_table(&datastore, create_request("t", dir.path(), columns)).unwrap();

    let mut first = datastore
        .clone()
        .begin_transaction()
        .table(
            DEFAULT_DATASTORE_NAME,
            &SchemaQualifiedTableName::in_default_schema("t"),
        )
        .unwrap();
    first
        .pushdown_filter(col_neq_filter(0, int_constant(20)))
        .unwrap();
    assert_eq!(row_group_count(&datastore, "t", &first), 2);

    // A fresh bind starts from the master entry's full row group set.
    let second = datastore
        .clone()
        .begin_transaction()
        .table(
            DEFAULT_DATASTORE_NAME,
            &SchemaQualifiedTableName::in_default_schema("t"),
        )
        .unwrap();
    assert_eq!(row_group_count(&datastore, "t", &second), 3);
}

#[test]
fn pushdown_filter_eq_prunes_row_groups_when_constant_outside_range() {
    let (dir, columns) = three_row_table();
    let (_database, datastore) = empty_datastore();
    create_table(&datastore, create_request("t", dir.path(), columns)).unwrap();

    // `id = 999` lies outside every (min,max) → all three row groups drop.
    let mut table = datastore
        .clone()
        .begin_transaction()
        .table(
            DEFAULT_DATASTORE_NAME,
            &SchemaQualifiedTableName::in_default_schema("t"),
        )
        .unwrap();
    table
        .pushdown_filter(col_eq_filter(0, int_constant(999)))
        .unwrap();
    assert_eq!(row_group_count(&datastore, "t", &table), 0);
}

#[test]
fn pushdown_filter_eq_keeps_only_matching_row_group() {
    let (dir, columns) = three_row_table();
    let (_database, datastore) = empty_datastore();
    create_table(&datastore, create_request("t", dir.path(), columns)).unwrap();

    // `id = 20` matches only the row group whose single value is 20.
    let mut table = datastore
        .clone()
        .begin_transaction()
        .table(
            DEFAULT_DATASTORE_NAME,
            &SchemaQualifiedTableName::in_default_schema("t"),
        )
        .unwrap();
    table
        .pushdown_filter(col_eq_filter(0, int_constant(20)))
        .unwrap();
    assert_eq!(row_group_count(&datastore, "t", &table), 1);
}

#[test]
fn pushdown_filter_eq_returns_false() {
    let (dir, columns) = three_row_table();
    let (_database, datastore) = empty_datastore();
    create_table(&datastore, create_request("t", dir.path(), columns)).unwrap();
    let mut table = datastore
        .clone()
        .begin_transaction()
        .table(
            DEFAULT_DATASTORE_NAME,
            &SchemaQualifiedTableName::in_default_schema("t"),
        )
        .unwrap();
    let pushed = table
        .pushdown_filter(col_eq_filter(0, int_constant(20)))
        .unwrap();
    assert!(!pushed);
}

#[test]
fn pushdown_filter_keeps_row_groups_when_constant_outside_range() {
    let (dir, columns) = three_row_table();
    let (_database, datastore) = empty_datastore();
    create_table(&datastore, create_request("t", dir.path(), columns)).unwrap();

    let mut table = datastore
        .clone()
        .begin_transaction()
        .table(
            DEFAULT_DATASTORE_NAME,
            &SchemaQualifiedTableName::in_default_schema("t"),
        )
        .unwrap();
    table
        .pushdown_filter(col_neq_filter(0, int_constant(999)))
        .unwrap();

    assert_eq!(row_group_count(&datastore, "t", &table), 3);
}

/// Write one Parquet file of `ids` (single row group) into `dir`, mirroring the
/// `three_row_table` schema. Returns the file's path.
fn write_ids(dir: &Path, file_name: &str, ids: &[i32]) -> std::path::PathBuf {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8View, false),
    ]));
    let names: Vec<String> = ids.iter().map(|i| format!("n{i}")).collect();
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(ids.to_vec())) as ArrayRef,
            Arc::new(StringViewArray::from(
                names.iter().map(String::as_str).collect::<Vec<_>>(),
            )) as ArrayRef,
        ],
    )
    .unwrap();
    let path = dir.join(file_name);
    let props = WriterProperties::builder()
        .set_statistics_enabled(EnabledStatistics::Chunk)
        .build();
    let mut writer =
        ArrowWriter::try_new(File::create(&path).unwrap(), batch.schema(), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    path
}

/// Append the file at `path` to table `name` through the test-only seeding API:
/// it writes the bytes into the table's location, CAS-commits the file on a
/// cloned-out handle, and publishes the committed copy back so the next
/// transaction's snapshot sees it. Recorded by its location-relative name.
/// (Production writers instead go through `PivotlakeDatastore::commit_to_table`.)
fn append(datastore: &PivotlakeDatastore, name: &str, path: &Path) {
    let bytes = std::fs::read(path).unwrap();
    let relative = ObjectPath::new(path.file_name().unwrap().to_string_lossy());
    let mut handle = datastore
        .table_handle(&SchemaQualifiedTableName::in_default_schema(name))
        .expect("table exists");
    handle.append_data_file(relative, &bytes, None).unwrap();
    datastore.publish_table(handle);
}

/// Plan and compile `sql` over `datastore` inside a fresh transaction (like
/// the server does per query), leaving execution and commit to the caller.
fn compile_sql(
    datastore: &Arc<PivotlakeDatastore>,
    sql: &str,
) -> (
    Arc<dyn CatalogTransaction>,
    dispatch::RecordBatchOperatorSpec,
) {
    let catalog = single_catalog(datastore);
    let transaction = catalog.begin_transaction();
    let mut planner = Planner::from_datastore_names(
        vec![DEFAULT_DATASTORE_NAME.to_string()],
        DEFAULT_DATASTORE_NAME.to_string(),
    )
    .expect("planner context");
    let compiled = planner
        .plan(sql, transaction.clone())
        .unwrap()
        .compile(&dispatcher(), transaction.as_ref())
        .unwrap();
    (transaction, compiled)
}

/// Run `sql` through a planner over `datastore` (inside a fresh transaction,
/// like the server does per query) and return the result batches.
fn run_sql(datastore: &Arc<PivotlakeDatastore>, sql: &str) -> Vec<RecordBatch> {
    let (transaction, compiled) = compile_sql(datastore, sql);
    let batches = compiled.collect().unwrap();
    commit_transaction_blocking(transaction);
    batches
}

/// Run `sql` like [`run_sql`] but expect the dataflow to fail, returning the
/// reported error. The transaction is dropped rather than committed.
fn run_sql_err(datastore: &Arc<PivotlakeDatastore>, sql: &str) -> String {
    let (_transaction, compiled) = compile_sql(datastore, sql);
    compiled.collect().unwrap_err().to_string()
}

/// Run `sql` like [`run_sql`], retaining the dataflow's IO/CPU tally.
fn run_sql_with_stats(
    datastore: &Arc<PivotlakeDatastore>,
    sql: &str,
) -> (Vec<RecordBatch>, dispatch::DataFlowStats) {
    let (transaction, compiled) = compile_sql(datastore, sql);
    let result = compiled.collect_with_stats().unwrap();
    commit_transaction_blocking(transaction);
    result
}

#[test]
fn insert_rows_are_visible_after_a_refresh() {
    let columns = vec![
        Column {
            name: "id".to_string(),
            col_type: Type::Int32,
        },
        Column {
            name: "name".to_string(),
            col_type: Type::Utf8,
        },
    ];
    let (_database, datastore) = empty_datastore();
    create_table(&datastore, empty_request("inserted", columns)).unwrap();

    let (inserted, stats) = run_sql_with_stats(
        &datastore,
        "INSERT INTO inserted VALUES (2, 'two'), (1, 'one')",
    );
    assert_eq!(common::extract_count(&inserted), 2);
    assert!(stats.disk_requests > 0, "the local INSERT issued a write");
    assert!(stats.disk_bytes > 0, "the write transferred parquet bytes");
    assert!(
        stats.disk_write_time > Duration::ZERO,
        "the write recorded IO time"
    );
    assert_eq!(stats.disk_read_time, Duration::ZERO);

    let inserted = run_sql(&datastore, "INSERT INTO inserted VALUES (3, 'three')");
    assert_eq!(common::extract_count(&inserted), 1);

    // An INSERT commits to the Delta log only; the background refresh brings the
    // new rows into the live set that later transactions bind against.
    datastore.refresh_from_store().unwrap();

    let rows = run_sql(&datastore, "SELECT id, name FROM inserted ORDER BY id");
    let ids = rows
        .iter()
        .flat_map(|batch| {
            let values = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            (0..values.len()).map(move |row| values.value(row))
        })
        .collect::<Vec<_>>();
    assert_eq!(ids, vec![1, 2, 3]);
    assert_eq!(
        common::collect_strings(&rows, 1),
        vec!["one", "two", "three"]
    );
}

/// A three-column table for the INSERT column-list tests: values arrive out of
/// table order and never fill `note`.
fn column_list_table() -> Vec<Column> {
    vec![
        Column {
            name: "id".to_string(),
            col_type: Type::Int32,
        },
        Column {
            name: "name".to_string(),
            col_type: Type::Utf8,
        },
        Column {
            name: "note".to_string(),
            col_type: Type::Utf8,
        },
    ]
}

#[test]
fn insert_with_a_column_list_reorders_values_and_nulls_the_rest() {
    let (_database, datastore) = empty_datastore();
    create_table(&datastore, empty_request("listed", column_list_table())).unwrap();

    let inserted = run_sql(
        &datastore,
        "INSERT INTO listed (name, id) VALUES ('two', 2), ('one', 1)",
    );
    datastore.refresh_from_store().unwrap();

    assert_eq!(common::extract_count(&inserted), 2);
    let rows = run_sql(&datastore, "SELECT name FROM listed ORDER BY id");
    assert_eq!(common::collect_strings(&rows, 0), vec!["one", "two"]);
    let unfilled = run_sql(&datastore, "SELECT COUNT(*) FROM listed WHERE note IS NULL");
    assert_eq!(common::extract_count(&unfilled), 2);
}

#[test]
fn insert_select_with_a_column_list_reorders_the_selected_rows() {
    let (_database, datastore) = empty_datastore();
    create_table(&datastore, empty_request("listed", column_list_table())).unwrap();
    create_table(&datastore, empty_request("source", column_list_table())).unwrap();
    run_sql(&datastore, "INSERT INTO source VALUES (1, 'one', 'first')");
    datastore.refresh_from_store().unwrap();

    let inserted = run_sql(
        &datastore,
        "INSERT INTO listed (note, id) SELECT name, id FROM source",
    );
    datastore.refresh_from_store().unwrap();

    assert_eq!(common::extract_count(&inserted), 1);
    let rows = run_sql(&datastore, "SELECT note FROM listed");
    assert_eq!(common::collect_strings(&rows, 0), vec!["one"]);
    let unfilled = run_sql(&datastore, "SELECT COUNT(*) FROM listed WHERE name IS NULL");
    assert_eq!(common::extract_count(&unfilled), 1);
}

/// Every unsigned width, each inserted at a value past the signed maximum of
/// the type Parquet stores it in — the values that only survive the round trip
/// if the file records the column's true width and signedness.
#[test]
fn unsigned_columns_insert_and_read_back() {
    let columns = vec![
        Column {
            name: "a".to_string(),
            col_type: Type::UInt8,
        },
        Column {
            name: "b".to_string(),
            col_type: Type::UInt16,
        },
        Column {
            name: "c".to_string(),
            col_type: Type::UInt32,
        },
        Column {
            name: "d".to_string(),
            col_type: Type::UInt64,
        },
    ];
    let (_database, datastore) = empty_datastore();
    create_table(&datastore, empty_request("unsigned", columns)).unwrap();

    let inserted = run_sql(
        &datastore,
        "INSERT INTO unsigned VALUES (255, 65535, 4294967295, 18446744073709551615)",
    );
    assert_eq!(common::extract_count(&inserted), 1);
    datastore.refresh_from_store().unwrap();

    let rows = run_sql(&datastore, "SELECT a, b, c, d FROM unsigned");
    let batch = &rows[0];
    assert_eq!(batch.column(0).as_ref(), &UInt8Array::from(vec![255u8]));
    assert_eq!(batch.column(1).as_ref(), &UInt16Array::from(vec![65535u16]));
    assert_eq!(
        batch.column(2).as_ref(),
        &UInt32Array::from(vec![4_294_967_295u32])
    );
    assert_eq!(
        batch.column(3).as_ref(),
        &UInt64Array::from(vec![18_446_744_073_709_551_615u64])
    );
}

/// The two integer widths narrower than a four-byte one: each stores its bits
/// in the wider physical type Parquet has, so these read back as themselves
/// only if the file kept the column's declared width. Both ends of each range
/// pin the sign extension they are written with.
#[test]
fn narrow_integer_columns_insert_and_read_back() {
    let data = TempDir::new().unwrap();
    let columns = vec![
        Column {
            name: "tiny".to_string(),
            col_type: Type::Int8,
        },
        Column {
            name: "small".to_string(),
            col_type: Type::Int16,
        },
    ];
    let (_database, datastore) = empty_datastore();
    create_table(&datastore, create_request("narrow", data.path(), columns)).unwrap();

    let inserted = run_sql(
        &datastore,
        "INSERT INTO narrow VALUES (-128, -32768), (127, 32767)",
    );
    assert_eq!(common::extract_count(&inserted), 2);
    datastore.refresh_from_store().unwrap();

    let rows = run_sql(&datastore, "SELECT tiny, small FROM narrow ORDER BY tiny");
    let batch = &rows[0];
    assert_eq!(
        batch.column(0).as_ref(),
        &Int8Array::from(vec![-128i8, 127])
    );
    assert_eq!(
        batch.column(1).as_ref(),
        &Int16Array::from(vec![-32768i16, 32767])
    );
}

/// A timestamp inserted through SQL reads back at full resolution, on both
/// sides of the epoch. What the leaf is annotated as is the writer's business,
/// covered in `writing::type_tests`.
#[test]
fn timestamp_column_inserts_and_reads_back() {
    let columns = vec![Column {
        name: "ts".to_string(),
        col_type: Type::Timestamp,
    }];
    let (_database, datastore) = empty_datastore();
    create_table(&datastore, empty_request("events", columns)).unwrap();

    let inserted = run_sql(
        &datastore,
        "INSERT INTO events VALUES (TIMESTAMP '2023-11-14 22:13:20.123456'), \
         (TIMESTAMP '1969-12-31 23:59:58.5')",
    );
    assert_eq!(common::extract_count(&inserted), 2);
    datastore.refresh_from_store().unwrap();

    let rows = run_sql(&datastore, "SELECT ts FROM events ORDER BY ts");
    assert_eq!(
        rows[0].column(0).as_ref(),
        &TimestampMicrosecondArray::from(vec![-1_500_000i64, 1_700_000_000_123_456])
    );
}

/// A TIMESTAMPTZ column stores UTC instants: a literal with an explicit offset
/// lands converted to UTC, and the column reads back zone-marked.
#[test]
fn timestamptz_column_inserts_and_reads_back_in_utc() {
    let columns = vec![Column {
        name: "occurred".to_string(),
        col_type: Type::TimestampTz,
    }];
    let (_database, datastore) = empty_datastore();
    create_table(&datastore, empty_request("events", columns)).unwrap();

    let inserted = run_sql(
        &datastore,
        "INSERT INTO events VALUES (TIMESTAMPTZ '2020-01-01 02:00:00.5+02'), \
         (TIMESTAMPTZ '1970-01-01 00:00:01+00')",
    );
    assert_eq!(common::extract_count(&inserted), 2);
    datastore.refresh_from_store().unwrap();

    let rows = run_sql(&datastore, "SELECT occurred FROM events ORDER BY occurred");
    // 02:00:00.5+02 is 00:00:00.5 UTC on 2020-01-01.
    assert_eq!(
        rows[0].column(0).as_ref(),
        &TimestampMicrosecondArray::from(vec![1_000_000i64, 1_577_836_800_500_000])
            .with_timezone("UTC")
    );
}

/// A value a column's type cannot represent fails the INSERT with an error
/// naming it, instead of silently landing as NULL.
#[test]
fn insert_of_an_unparseable_timestamp_names_the_value() {
    let data = TempDir::new().unwrap();
    let columns = vec![Column {
        name: "ts".to_string(),
        col_type: Type::Timestamp,
    }];
    let (_database, datastore) = empty_datastore();
    create_table(&datastore, create_request("events", data.path(), columns)).unwrap();

    let err = run_sql_err(&datastore, "INSERT INTO events VALUES ('not-a-timestamp')");

    assert!(
        err.contains("not-a-timestamp"),
        "the error names the offending value; got: {err}"
    );
}

/// An unsigned column filters on its own ordering: the bounds a row group
/// records are what a scan prunes by, and comparing them as the signed type the
/// values are stored in would drop the rows above the signed maximum.
#[test]
fn unsigned_column_filters_on_unsigned_ordering() {
    let columns = vec![Column {
        name: "big".to_string(),
        col_type: Type::UInt64,
    }];
    let (_database, datastore) = empty_datastore();
    create_table(&datastore, empty_request("wide", columns)).unwrap();

    run_sql(
        &datastore,
        "INSERT INTO wide VALUES (1), (9223372036854775808), (18446744073709551615)",
    );
    datastore.refresh_from_store().unwrap();

    let rows = run_sql(
        &datastore,
        "SELECT big FROM wide WHERE big > 9223372036854775807 ORDER BY big",
    );
    assert_eq!(
        rows[0].column(0).as_ref(),
        &UInt64Array::from(vec![9_223_372_036_854_775_808u64, u64::MAX])
    );
}

/// Partitioning by an unsigned column: the value the log records for each file
/// has to come back as the type the column declares, since that is what a later
/// scan compares its partition filters against.
#[test]
fn unsigned_partition_column_round_trips() {
    let (_database, datastore) = empty_datastore();
    create_table(
        &datastore,
        CreateTableRequest {
            datastore_name: None,
            schema_name: None,
            name: "parts".to_string(),
            columns: vec![
                Column {
                    name: "bucket".to_string(),
                    col_type: Type::UInt8,
                },
                Column {
                    name: "id".to_string(),
                    col_type: Type::Int32,
                },
            ],
            options: HashMap::from([("partition_by".to_string(), "bucket".to_string())]),
            if_not_exists: false,
        },
    )
    .unwrap();

    run_sql(&datastore, "INSERT INTO parts VALUES (200, 1), (201, 2)");
    datastore.refresh_from_store().unwrap();

    let rows = run_sql(&datastore, "SELECT id FROM parts WHERE bucket = 200");
    assert_eq!(rows[0].column(0).as_ref(), &Int32Array::from(vec![1]));
}

#[test]
fn insert_files_reach_the_log_only_when_the_transaction_commits() {
    let (_database, datastore) = empty_datastore();
    create_table(
        &datastore,
        CreateTableRequest {
            datastore_name: None,
            schema_name: None,
            name: "pending_insert".to_string(),
            columns: vec![Column {
                name: "id".to_string(),
                col_type: Type::Int32,
            }],
            options: HashMap::new(),
            if_not_exists: false,
        },
    )
    .unwrap();

    let catalog = single_catalog(&datastore);
    let transaction = catalog.begin_transaction();
    let mut planner = Planner::from_datastore_names(
        vec![DEFAULT_DATASTORE_NAME.to_string()],
        DEFAULT_DATASTORE_NAME.to_string(),
    )
    .expect("planner context");
    let inserted = planner
        .plan(
            "INSERT INTO pending_insert VALUES (1), (2)",
            transaction.clone(),
        )
        .unwrap()
        .compile(&dispatcher(), transaction.as_ref())
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(common::extract_count(&inserted), 2);

    // Uncommitted: the files are uploaded but not in the Delta log, so a refresh
    // finds nothing and the rows stay invisible.
    datastore.refresh_from_store().unwrap();
    let before_commit = run_sql(&datastore, "SELECT COUNT(*) FROM pending_insert");
    assert_eq!(common::extract_count(&before_commit), 0);

    // Committing writes the files to the log; the refresh then surfaces them.
    commit_transaction_blocking(transaction);
    datastore.refresh_from_store().unwrap();
    let after_commit = run_sql(&datastore, "SELECT COUNT(*) FROM pending_insert");
    assert_eq!(common::extract_count(&after_commit), 2);

    let rolled_back = catalog.begin_transaction();
    let discarded = planner
        .plan("INSERT INTO pending_insert VALUES (3)", rolled_back.clone())
        .unwrap()
        .compile(&dispatcher(), rolled_back.as_ref())
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(common::extract_count(&discarded), 1);
    rolled_back.rollback();

    // The rolled-back INSERT never reached the log, so a refresh leaves the count
    // at the two committed rows.
    datastore.refresh_from_store().unwrap();
    let after_rollback = run_sql(&datastore, "SELECT COUNT(*) FROM pending_insert");
    assert_eq!(common::extract_count(&after_rollback), 2);
}

#[test]
fn a_rolled_back_insert_removes_the_files_it_uploaded() {
    let (database, datastore) = empty_datastore();
    create_table(
        &datastore,
        CreateTableRequest {
            datastore_name: None,
            schema_name: None,
            name: "discarded_insert".to_string(),
            columns: vec![Column {
                name: "id".to_string(),
                col_type: Type::Int32,
            }],
            options: HashMap::new(),
            if_not_exists: false,
        },
    )
    .unwrap();
    let table_dir = common::table_dir(database.path(), &datastore, "discarded_insert");
    let (transaction, compiled) =
        compile_sql(&datastore, "INSERT INTO discarded_insert VALUES (1)");
    compiled.collect().unwrap();
    assert_eq!(common::parquet_files_in(&table_dir).len(), 1);

    transaction.rollback();

    assert!(common::parquet_files_in(&table_dir).is_empty());
}

/// A file appended after `CREATE TABLE` becomes visible to new binds, with
/// global row-group indices kept sequential.
#[test]
fn append_data_file_makes_new_file_visible_to_new_binds() {
    let (dir, columns) = three_row_table();
    let (_database, datastore) = empty_datastore();
    create_table(&datastore, create_request("t", dir.path(), columns)).unwrap();
    assert_eq!(current_parquet(&datastore, "t").row_groups().len(), 3);

    let new_file = write_ids(dir.path(), "later.parquet", &[40, 50]);
    append(&datastore, "t", &new_file);

    let parquet = current_parquet(&datastore, "t");
    let groups = parquet.row_groups();
    assert_eq!(groups.len(), 4);
    assert_eq!(groups.iter().map(|rg| rg.num_rows).sum::<i64>(), 5);
}

#[test]
fn table_revision_is_frozen_with_the_transaction_snapshot() {
    let (dir, columns) = three_row_table();
    let (_database, datastore) = empty_datastore();
    create_table(&datastore, create_request("t", dir.path(), columns)).unwrap();
    let before_append = datastore.clone().begin_transaction();
    let before_revision = before_append
        .table_revision(&SchemaQualifiedTableName::in_default_schema("t"))
        .unwrap()
        .unwrap();

    let new_file = write_ids(dir.path(), "later.parquet", &[40]);
    append(&datastore, "t", &new_file);
    let after_revision = datastore
        .clone()
        .begin_transaction()
        .table_revision(&SchemaQualifiedTableName::in_default_schema("t"))
        .unwrap()
        .unwrap();

    assert_eq!(before_revision.version, "0");
    assert_eq!(after_revision.version, "1");
    assert_eq!(before_revision.identity, after_revision.identity);
    assert_eq!(
        before_append
            .table_revision(&SchemaQualifiedTableName::in_default_schema("t"))
            .unwrap()
            .unwrap(),
        before_revision
    );
}

/// Appending the same path twice (a replayed flush notification) must not
/// double-count its rows. The second handle starts a version behind, so it CAS-
/// conflicts, refreshes, and sees the file already present.
#[test]
fn append_data_file_is_idempotent_per_path() {
    let (dir, columns) = three_row_table();
    let (_database, datastore) = empty_datastore();
    create_table(&datastore, create_request("t", dir.path(), columns)).unwrap();

    let new_file = write_ids(dir.path(), "later.parquet", &[40]);
    append(&datastore, "t", &new_file);
    // Appending the same file again is a no-op (it must not double-count).
    append(&datastore, "t", &new_file);

    assert_eq!(current_parquet(&datastore, "t").row_groups().len(), 4);
}

/// No table yet (a write arrives before `CREATE TABLE`): there is no handle to
/// append against.
#[test]
fn table_handle_for_a_missing_table_is_none() {
    let (_database, datastore) = empty_datastore();
    assert!(
        datastore
            .table_handle(&SchemaQualifiedTableName::in_default_schema("missing"))
            .is_none()
    );
}

/// Compaction's commit: the small files' row groups vanish, the merged file's
/// appear, and indices are renumbered — one atomic version swap. And a *second*
/// compacter that picked the same inputs must fail with `CommitConflict`
/// rather than re-add its output on top, which would double-count the rows.
#[test]
fn replace_data_files_swaps_compacted_inputs_for_merged_output() {
    let (dir, columns) = three_row_table();
    let (_database, datastore) = empty_datastore();
    create_table(&datastore, create_request("t", dir.path(), columns)).unwrap();
    let extra = write_ids(dir.path(), "extra.parquet", &[40]);
    append(&datastore, "t", &extra);
    assert_eq!(current_parquet(&datastore, "t").row_groups().len(), 4);

    let merged = write_ids(dir.path(), "merged.parquet", &[10, 20, 30, 40]);
    let merged_size = std::fs::metadata(&merged).unwrap().len();
    let added = vec![datastore_pivotlake::DeltaFileEntry::new(
        datastore_pivotlake::FileRef {
            path: ObjectPath::new(merged.to_str().unwrap()),
            size: merged_size,
        },
    )];
    // A losing compacter clones the table out at the version where the inputs
    // are present, before the winning swap lands.
    let mut loser = datastore
        .table_handle(&SchemaQualifiedTableName::in_default_schema("t"))
        .unwrap();
    loser.refresh().unwrap();
    let mut winner = datastore
        .table_handle(&SchemaQualifiedTableName::in_default_schema("t"))
        .unwrap();
    winner.refresh().unwrap();
    // The two inputs under the paths the table records them by: the adopted file
    // absolutely, the appended one relative to the table's own location.
    let removed: Vec<ObjectPath> = winner.file_refs().into_iter().map(|f| f.path).collect();
    winner.replace_data_files(&removed, &added).unwrap();
    // The loser only discovers the inputs are gone after its CAS conflict +
    // refresh and returns a typed commit-conflict error.
    let loser_added = vec![datastore_pivotlake::DeltaFileEntry::new(
        datastore_pivotlake::FileRef {
            path: ObjectPath::new("merged-loser.parquet"),
            size: merged_size,
        },
    )];
    let loser_result = loser.replace_data_files(&removed, &loser_added);
    assert!(
        matches!(
            &loser_result,
            Err(datastore_pivotlake::Error::CommitConflict { .. })
        ),
        "unexpected losing compaction result: {loser_result:?}"
    );

    let parquet = current_parquet(&datastore, "t");
    let groups = parquet.row_groups();
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].num_rows, 4);
}

/// A writer's copy carries its own Delta snapshot forward across commits: each
/// commit rides the snapshot the previous one produced, so the copy commits
/// again and again with no reload in between and every commit lands exactly one
/// version on.
#[test]
fn one_copy_commits_repeatedly_without_reloading_between_commits() {
    let (dir, columns) = three_row_table();
    let (_database, datastore) = empty_datastore();
    create_table(&datastore, create_request("t", dir.path(), columns)).unwrap();
    let mut table = datastore
        .table_handle(&SchemaQualifiedTableName::in_default_schema("t"))
        .unwrap();
    let start = table.version();

    for id in 0..3 {
        let name = format!("extra{id}.parquet");
        let bytes = std::fs::read(write_ids(dir.path(), &name, &[id])).unwrap();
        table
            .append_data_file(ObjectPath::new(name), &bytes, None)
            .unwrap();
    }

    assert_eq!(table.version(), start + 3);
    assert_eq!(
        table.file_refs().len(),
        4,
        "the seed file plus three appends"
    );
}

/// A copy refreshes onto a version another copy committed, picking up its file;
/// refreshing an already-current copy reports no change.
#[test]
fn a_refresh_advances_a_copy_onto_another_copys_commit() {
    let (dir, columns) = three_row_table();
    let (_database, datastore) = empty_datastore();
    create_table(&datastore, create_request("t", dir.path(), columns)).unwrap();
    let mut reader = datastore
        .table_handle(&SchemaQualifiedTableName::in_default_schema("t"))
        .unwrap();
    let start = reader.version();

    append(
        &datastore,
        "t",
        &write_ids(dir.path(), "extra.parquet", &[40]),
    );

    assert_eq!(
        reader.version(),
        start,
        "the other copy's commit is not ours"
    );
    assert!(reader.refresh().unwrap(), "the refresh advances us onto it");
    assert_eq!(reader.version(), start + 1);
    assert_eq!(reader.file_refs().len(), 2);
    assert!(
        !reader.refresh().unwrap(),
        "a current copy does not advance"
    );
}

/// A refresh that cannot materialize the newer version leaves the copy exactly
/// where it was, files and version together. A copy moved onto a version whose
/// files it does not hold would never recover: its next refresh finds that
/// version already current and reconciles nothing, so it would serve, and
/// commit on top of, a file set missing the rows it claims.
#[test]
fn a_refresh_that_cannot_read_the_new_files_leaves_the_copy_untouched() {
    let (dir, columns) = three_row_table();
    let (_database, datastore) = empty_datastore();
    create_table(&datastore, create_request("t", dir.path(), columns)).unwrap();
    let mut reader = datastore
        .table_handle(&SchemaQualifiedTableName::in_default_schema("t"))
        .unwrap();
    let version = reader.version();

    // Another writer commits an `Add` for a file that is not in the store, so
    // any copy reloading onto that version fails to read its footer.
    let mut writer = datastore
        .table_handle(&SchemaQualifiedTableName::in_default_schema("t"))
        .unwrap();
    let absent = vec![datastore_pivotlake::DeltaFileEntry::new(
        datastore_pivotlake::FileRef {
            path: ObjectPath::new("absent.parquet"),
            size: 1,
        },
    )];
    assert!(writer.replace_data_files(&[], &absent).is_err());

    assert!(reader.refresh().is_err());
    assert_eq!(reader.version(), version, "the copy stays at its version");
    assert_eq!(
        reader.file_refs().len(),
        1,
        "with the files of that version"
    );
    assert_eq!(
        reader.build_scan_view(&[], &[]).unwrap().row_groups().len(),
        3,
        "and still scans them"
    );
}

/// Compaction merges a table's small files into one target-sized file over the
/// async upload path and swaps them in, preserving every row in one commit.
#[test]
fn compact_table_files_merges_small_files_into_one() {
    let columns = vec![Column {
        name: "id".to_string(),
        col_type: Type::Int32,
    }];
    let (_database, datastore) = empty_datastore();
    create_table(&datastore, empty_request("t", columns)).unwrap();
    run_sql(&datastore, "INSERT INTO t VALUES (10), (20), (30)");
    run_sql(&datastore, "INSERT INTO t VALUES (40), (50)");

    let mut table = datastore
        .table_handle(&SchemaQualifiedTableName::in_default_schema("t"))
        .unwrap();
    table.refresh().unwrap();
    let inputs = table.file_refs();
    assert_eq!(inputs.len(), 2, "two inserts wrote two files");
    let merged = datastore_pivotlake::compact_table_files(
        &datastore,
        table.id(),
        &inputs,
        128 * 1024,
        128 * 1024,
    )
    .unwrap();

    assert_eq!(merged.len(), 1, "the two inputs merge into one file");
    let parquet = current_parquet(&datastore, "t");
    assert_eq!(parquet.row_groups().len(), 1);
    assert_eq!(parquet.row_groups()[0].num_rows, 5);
}

/// An operational Delta commit error happens after compaction has uploaded its
/// output. The output is not referenced by any log version, so it must be
/// deleted before the error escapes.
#[test]
fn compact_table_files_deletes_uploaded_outputs_when_delta_commit_fails() {
    let (dir, columns) = three_row_table();
    let (database, datastore) = empty_datastore();
    create_table(&datastore, create_request("t", dir.path(), columns)).unwrap();
    let table_dir = table_dir(&database, &datastore, "t");

    let mut table = datastore
        .table_handle(&SchemaQualifiedTableName::in_default_schema("t"))
        .unwrap();
    table.refresh().unwrap();
    let inputs = table.file_refs();

    // Replace the Delta log directory with a file so the data upload succeeds
    // but creating the next commit version fails.
    let delta_log = table_dir.join("_delta_log");
    let saved_delta_log = table_dir.join("_delta_log.saved");
    std::fs::rename(&delta_log, &saved_delta_log).unwrap();
    File::create(&delta_log).unwrap();
    let result = datastore_pivotlake::compact_table_files(
        &datastore,
        table.id(),
        &inputs,
        128 * 1024,
        128 * 1024,
    );
    std::fs::remove_file(&delta_log).unwrap();
    std::fs::rename(&saved_delta_log, &delta_log).unwrap();

    assert!(result.is_err(), "the Delta commit must fail");
    assert!(
        std::fs::read_dir(&table_dir).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with("pivot-")
        }),
        "an uploaded output must not remain after any commit error"
    );
}

/// A binding's pushed-down predicates are a pure filter over whatever file set
/// they are applied to: the same binding prunes a wider, later file set just as
/// well, so predicate state and file state stay independent.
#[test]
fn pushed_predicate_prunes_latest_files_after_refresh() {
    let (dir, columns) = three_row_table();
    let (_database, datastore) = empty_datastore();
    create_table(&datastore, create_request("t", dir.path(), columns)).unwrap();
    let mut table = datastore
        .clone()
        .begin_transaction()
        .table(
            DEFAULT_DATASTORE_NAME,
            &SchemaQualifiedTableName::in_default_schema("t"),
        )
        .unwrap();
    table
        .pushdown_filter(col_eq_filter(0, int_constant(20)))
        .unwrap();
    assert_eq!(row_group_count(&datastore, "t", &table), 1);

    let new_file = write_ids(dir.path(), "later.parquet", &[40, 50]);
    append(&datastore, "t", &new_file);

    // `id = 20` still prunes to the single matching row group, now over 4 files.
    assert_eq!(row_group_count(&datastore, "t", &table), 1);
}

/// Restart reads the committed manifest, not the directory: files appended
/// after the `CREATE` survive a reopen.
#[test]
fn reopened_database_restores_appended_files_from_manifest() {
    let (data_dir, columns) = three_row_table();
    let db = TempDir::new().unwrap();
    {
        let datastore =
            PivotlakeDatastore::open(db.path().to_str().unwrap(), &dispatcher()).unwrap();
        create_table(&datastore, create_request("t", data_dir.path(), columns)).unwrap();
        let new_file = write_ids(data_dir.path(), "later.parquet", &[40]);
        append(&datastore, "t", &new_file);
    }
    let reopened = PivotlakeDatastore::open(db.path().to_str().unwrap(), &dispatcher()).unwrap();
    assert_eq!(current_parquet(&reopened, "t").row_groups().len(), 4);
}

/// Delta has no native sort spec, so `sort_by` rides in the table metadata's
/// configuration; a reopen must restore it or a writer silently stops sorting.
#[test]
fn reopened_database_restores_sort_by() {
    let (data_dir, columns) = three_row_table();
    let db = TempDir::new().unwrap();
    {
        let datastore =
            PivotlakeDatastore::open(db.path().to_str().unwrap(), &dispatcher()).unwrap();
        let mut request = create_request("t", data_dir.path(), columns);
        request
            .options
            .insert("sort_by".to_string(), "id".to_string());
        create_table(&datastore, request).unwrap();
    }

    let reopened = PivotlakeDatastore::open(db.path().to_str().unwrap(), &dispatcher()).unwrap();

    assert_eq!(
        reopened
            .table_handle(&SchemaQualifiedTableName::in_default_schema("t"))
            .unwrap()
            .sort_by(),
        ["id"]
    );
}

/// Only committed files exist: after a compaction swap, a leftover input
/// (e.g. a crash before the unlink) is invisible to a reopen — no double-read.
#[test]
fn unlogged_leftover_file_is_invisible_after_swap() {
    let (data_dir, columns) = three_row_table();
    let db = TempDir::new().unwrap();
    let datastore = PivotlakeDatastore::open(db.path().to_str().unwrap(), &dispatcher()).unwrap();
    create_table(&datastore, create_request("t", data_dir.path(), columns)).unwrap();

    // "Compact" the adopted file into merged.parquet but crash before deleting
    // the input: both files are on disk, only merged is in the manifest.
    let mut table = datastore
        .table_handle(&SchemaQualifiedTableName::in_default_schema("t"))
        .unwrap();
    let inputs: Vec<ObjectPath> = table.file_refs().into_iter().map(|f| f.path).collect();
    let merged = write_ids(data_dir.path(), "merged.parquet", &[10, 20, 30]);
    let added = vec![datastore_pivotlake::DeltaFileEntry::new(
        datastore_pivotlake::FileRef {
            path: ObjectPath::new(merged.to_str().unwrap()),
            size: std::fs::metadata(&merged).unwrap().len(),
        },
    )];
    table.replace_data_files(&inputs, &added).unwrap();

    drop(table);
    drop(datastore);
    let reopened = PivotlakeDatastore::open(db.path().to_str().unwrap(), &dispatcher()).unwrap();
    let parquet = current_parquet(&reopened, "t");
    let groups = parquet.row_groups();
    assert_eq!(groups.len(), 1, "only the committed merged file is read");
    assert_eq!(groups.iter().map(|rg| rg.num_rows).sum::<i64>(), 3);
}

/// Write `file_name` into `dir` with one row group per value in `ids`, so a file
/// can hold several row groups that all share one `id` (the shape the partition
/// writer emits: one file per partition value, many row groups inside it).
fn write_ids_one_group_each(dir: &Path, file_name: &str, ids: &[i32]) {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let batches: Vec<RecordBatch> = ids
        .iter()
        .map(|i| {
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![*i]))])
                .unwrap()
        })
        .collect();
    let props = WriterProperties::builder()
        .set_statistics_enabled(EnabledStatistics::Chunk)
        .set_max_row_group_row_count(Some(1))
        .build();
    let mut writer = ArrowWriter::try_new(
        File::create(dir.join(file_name)).unwrap(),
        schema,
        Some(props),
    )
    .unwrap();
    for batch in &batches {
        writer.write(batch).unwrap();
    }
    writer.close().unwrap();
}

/// A file whose every row group shares one partition value is fully pruned by a
/// filter on the partition column: each row group has min == max == that value,
/// so the existing min/max stats prune drops the whole irrelevant file without
/// any partition-specific read path. This is why partition_by needs no special
/// pruning today — one-partition-per-file makes it fall out of stats pruning.
#[test]
fn filter_on_partition_column_prunes_whole_single_partition_file() {
    let dir = TempDir::new().unwrap();
    write_ids_one_group_each(dir.path(), "part-1.parquet", &[1, 1, 1]);
    write_ids_one_group_each(dir.path(), "part-2.parquet", &[2, 2, 2]);
    let columns = vec![Column {
        name: "id".to_string(),
        col_type: Type::Int32,
    }];
    let (_database, datastore) = empty_datastore();
    create_table(&datastore, create_request("t", dir.path(), columns)).unwrap();

    let mut table = datastore
        .clone()
        .begin_transaction()
        .table(
            DEFAULT_DATASTORE_NAME,
            &SchemaQualifiedTableName::in_default_schema("t"),
        )
        .unwrap();
    assert_eq!(row_group_count(&datastore, "t", &table), 6);

    table
        .pushdown_filter(col_eq_filter(0, int_constant(1)))
        .unwrap();

    // `id = 1` excludes the part-2 file's three row groups entirely; only the
    // part-1 file's three survive.
    assert_eq!(row_group_count(&datastore, "t", &table), 3);
}

/// A table partitioned by `name` with one committed file per partition: a
/// one-row-group `keep` file and a three-row-group `drop` file, each tagged with
/// its partition tuple. The distinct group counts let a build's group count name
/// exactly which files it fetched. Files are written *after* `CREATE TABLE` (over
/// an empty dir) so they arrive through the partition-recording append, not as
/// untagged create-time discoveries.
fn table_partitioned_by_name() -> (TempDir, TempDir, Arc<PivotlakeDatastore>) {
    let dir = TempDir::new().unwrap();
    let request = CreateTableRequest {
        datastore_name: None,
        schema_name: None,
        name: "p".to_string(),
        columns: vec![
            Column {
                name: "id".to_string(),
                col_type: Type::Int32,
            },
            Column {
                name: "name".to_string(),
                col_type: Type::Utf8,
            },
        ],
        options: HashMap::from([("partition_by".to_string(), "name".to_string())]),
        if_not_exists: false,
    };
    let (database, datastore) = empty_datastore();
    create_table(&datastore, request).unwrap();

    write_ids_one_group_each(dir.path(), "keep.parquet", &[1]);
    write_ids_one_group_each(dir.path(), "drop.parquet", &[2, 2, 2]);
    let mut table = datastore
        .table_handle(&SchemaQualifiedTableName::in_default_schema("p"))
        .unwrap();
    table
        .append_data_file(
            ObjectPath::new("keep.parquet"),
            &std::fs::read(dir.path().join("keep.parquet")).unwrap(),
            Some(string_values("name", "keep")),
        )
        .unwrap();
    table
        .append_data_file(
            ObjectPath::new("drop.parquet"),
            &std::fs::read(dir.path().join("drop.parquet")).unwrap(),
            Some(string_values("name", "drop")),
        )
        .unwrap();
    (dir, database, datastore)
}

fn string_values(name: &str, value: &str) -> HashMap<String, Scalar<ArrayRef>> {
    HashMap::from([(
        name.to_string(),
        Scalar::new(Arc::new(StringViewArray::from(vec![value])) as ArrayRef),
    )])
}

fn name_eq(value: &str) -> PartitionEqFilter {
    PartitionEqFilter {
        column: "name".to_string(),
        value: Scalar::new(Arc::new(StringViewArray::from(vec![value])) as ArrayRef),
    }
}

#[test]
fn partition_filter_builds_only_the_matching_partitions_files() {
    let (_dir, _database, datastore) = table_partitioned_by_name();

    let mut table = datastore
        .table_handle(&SchemaQualifiedTableName::in_default_schema("p"))
        .unwrap();
    table.refresh().unwrap();
    let kept = table.build_scan_view(&[name_eq("keep")], &[]).unwrap();

    // Only the one-group `keep` file enters the scan view; the three-group
    // `drop` file's partition tuple can't match.
    assert_eq!(kept.row_groups().len(), 1);
}

#[test]
fn no_partition_filter_builds_every_partitions_files() {
    let (_dir, _database, datastore) = table_partitioned_by_name();

    let mut table = datastore
        .table_handle(&SchemaQualifiedTableName::in_default_schema("p"))
        .unwrap();
    table.refresh().unwrap();
    let all = table.build_scan_view(&[], &[]).unwrap();

    // Without a filter both files are in view: keep's 1 group + drop's 3.
    assert_eq!(all.row_groups().len(), 4);
}

/// Write `file_name` into `dir` with a constant `part` column and a varying `id`,
/// one row group per id, so a discovered file is single-partition on `part` with
/// observable per-group stats.
fn write_part_id_file(dir: &Path, file_name: &str, part: i32, ids: &[i32]) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("part", DataType::Int32, false),
        Field::new("id", DataType::Int32, false),
    ]));
    let props = WriterProperties::builder()
        .set_statistics_enabled(EnabledStatistics::Chunk)
        .set_max_row_group_row_count(Some(1))
        .build();
    let mut writer = ArrowWriter::try_new(
        File::create(dir.join(file_name)).unwrap(),
        schema.clone(),
        Some(props),
    )
    .unwrap();
    for id in ids {
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![part])),
                Arc::new(Int32Array::from(vec![*id])),
            ],
        )
        .unwrap();
        writer.write(&batch).unwrap();
    }
    writer.close().unwrap();
}

#[test]
fn create_over_partitioned_files_stamps_each_file_with_its_partition() {
    let dir = TempDir::new().unwrap();
    write_part_id_file(dir.path(), "part-1.parquet", 1, &[10, 11, 12]);
    write_part_id_file(dir.path(), "part-2.parquet", 2, &[20, 21, 22]);
    let columns = vec![
        Column {
            name: "part".to_string(),
            col_type: Type::Int32,
        },
        Column {
            name: "id".to_string(),
            col_type: Type::Int32,
        },
    ];
    let (_database, datastore) = empty_datastore();
    let mut request = create_request("t", dir.path(), columns);
    request
        .options
        .insert("partition_by".to_string(), "part".to_string());
    create_table(&datastore, request).unwrap();

    let mut table = datastore
        .table_handle(&SchemaQualifiedTableName::in_default_schema("t"))
        .unwrap();
    table.refresh().unwrap();
    let part_is_one = PartitionEqFilter {
        column: "part".to_string(),
        value: int_constant(1),
    };
    let kept = table.build_scan_view(&[part_is_one], &[]).unwrap();

    // Each discovered file's constant `part` was recorded as its partition, so the
    // part=1 filter drops the part=2 file's three row groups and keeps part=1's.
    assert_eq!(kept.row_groups().len(), 3);
}

#[test]
fn file_level_stats_prune_drops_a_whole_file_out_of_range() {
    let dir = TempDir::new().unwrap();
    write_ids_one_group_each(dir.path(), "low.parquet", &[1, 2, 3]);
    write_ids_one_group_each(dir.path(), "high.parquet", &[100, 200, 300]);
    let columns = vec![Column {
        name: "id".to_string(),
        col_type: Type::Int32,
    }];
    let (_database, datastore) = empty_datastore();
    // Created empty, then each file appended into the table's own location, so
    // the stats under test are the ones the append path records.
    create_table(
        &datastore,
        CreateTableRequest {
            datastore_name: None,
            schema_name: None,
            name: "t".to_string(),
            columns,
            options: HashMap::new(),
            if_not_exists: false,
        },
    )
    .unwrap();
    let mut table = datastore
        .table_handle(&SchemaQualifiedTableName::in_default_schema("t"))
        .unwrap();
    for name in ["low.parquet", "high.parquet"] {
        table
            .append_data_file(
                ObjectPath::new(name),
                &std::fs::read(dir.path().join(name)).unwrap(),
                None,
            )
            .unwrap();
    }
    table.refresh().unwrap();

    let id_below_fifty = ColumnStatFilter {
        column: "id".to_string(),
        compare_type: CompareType::Less,
        value: int_constant(50),
    };
    let kept = table.build_scan_view(&[], &[id_below_fifty]).unwrap();

    // The high file's aggregate min (100) proves no row is `< 50`, so all three
    // of its groups drop before row-group pruning; the low file's three survive.
    assert_eq!(kept.row_groups().len(), 3);
}

// -- Variant shredded-path pushdown --

/// Write `name` into `dir`: a single VARIANT `doc` column holding `docs` as
/// JSON, with `shred_path` shredded into a typed Int64 leaf when given, one
/// row group per document, so a per-group min/max prune is observable.
fn write_docs_file_into(dir: &Path, name: &str, docs: &[String], shred_path: Option<&str>) {
    use parquet_variant_compute::{ShreddedSchemaBuilder, json_to_variant, shred_variant};

    let json: ArrayRef = Arc::new(StringArray::from(
        docs.iter().map(String::as_str).collect::<Vec<_>>(),
    ));
    let variant = json_to_variant(&json).unwrap();
    let (field, array) = match shred_path {
        Some(path) => {
            let shred = ShreddedSchemaBuilder::new()
                .with_path(path, &DataType::Int64)
                .unwrap()
                .build();
            let shredded = shred_variant(&variant, &shred).unwrap();
            (shredded.field("doc"), shredded.into_inner())
        }
        None => (variant.field("doc"), variant.into_inner()),
    };
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![field])),
        vec![Arc::new(array) as ArrayRef],
    )
    .unwrap();

    let props = WriterProperties::builder()
        .set_statistics_enabled(EnabledStatistics::Chunk)
        .set_max_row_group_row_count(Some(1))
        .build();
    let file = File::create(dir.join(name)).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
}

/// One shredded-`age` file, one row group per age.
fn write_shredded_ages(ages: &[i64]) -> TempDir {
    let dir = TempDir::new().unwrap();
    let docs: Vec<String> = ages.iter().map(|a| format!(r#"{{"age":{a}}}"#)).collect();
    write_docs_file_into(dir.path(), "docs.parquet", &docs, Some("age"));
    dir
}

/// `CAST(doc-><path> AS BIGINT) <cmp> <value>`, the shape plan build pushes for
/// a typed variant comparison (the cast fused into a typed VariantGet).
fn variant_filter(path: &[&str], cmp: CompareType, value: i64) -> Expression {
    let constant = Scalar::new(Arc::new(Int64Array::new_scalar(value).into_inner()) as ArrayRef);
    Expression::Compare(Compare {
        left: Box::new(Expression::Function(Function::VariantGet(VariantGet {
            input: Box::new(Expression::Ref(Ref {
                column_idx: 0,
                return_type: Type::Variant,
                name: None,
            })),
            path: path.iter().map(|s| s.to_string()).collect(),
            as_type: Some(Type::Int64),
        }))),
        right: Box::new(Expression::Constant(constant)),
        compare_type: cmp,
        return_type: Type::Boolean,
    })
}

fn shredded_docs_datastore(dir: &Path) -> (TempDir, Arc<PivotlakeDatastore>, TableBinding) {
    let (database, datastore) = empty_datastore();
    let columns = vec![Column {
        name: "doc".to_string(),
        col_type: Type::Variant,
    }];
    create_table(&datastore, create_request("docs", dir, columns)).unwrap();
    let binding = datastore
        .clone()
        .begin_transaction()
        .table(
            DEFAULT_DATASTORE_NAME,
            &SchemaQualifiedTableName::in_default_schema("docs"),
        )
        .unwrap();
    (database, datastore, binding)
}

#[test]
fn variant_pushdown_equality_keeps_only_the_matching_row_group() {
    let dir = write_shredded_ages(&[10, 20, 30]);
    let (_database, datastore, mut table) = shredded_docs_datastore(dir.path());
    assert_eq!(row_group_count(&datastore, "docs", &table), 3);

    table
        .pushdown_filter(variant_filter(&["age"], CompareType::Equal, 20))
        .unwrap();

    // Only the row group whose shredded `age` leaf holds 20 survives.
    assert_eq!(row_group_count(&datastore, "docs", &table), 1);
}

#[test]
fn variant_pushdown_range_prunes_by_the_shredded_leaf() {
    let dir = write_shredded_ages(&[10, 20, 30]);
    let (_database, datastore, mut table) = shredded_docs_datastore(dir.path());

    table
        .pushdown_filter(variant_filter(&["age"], CompareType::Less, 20))
        .unwrap();

    // `age < 20` keeps only the group whose min is below 20 (the 10 group).
    assert_eq!(row_group_count(&datastore, "docs", &table), 1);
}

#[test]
fn variant_pushdown_prunes_json_null_and_missing_groups() {
    // Setup
    let dir = TempDir::new().unwrap();
    let docs = vec![
        r#"{"age":10}"#.to_string(),
        r#"{"age":null}"#.to_string(),
        r#"{}"#.to_string(),
        r#"{"age":30}"#.to_string(),
    ];
    write_docs_file_into(dir.path(), "docs.parquet", &docs, Some("age"));
    let (_database, datastore, mut table) = shredded_docs_datastore(dir.path());

    // Execute
    table
        .pushdown_filter(variant_filter(&["age"], CompareType::Equal, 30))
        .unwrap();

    // Assert
    assert_eq!(row_group_count(&datastore, "docs", &table), 1);
}

/// Soundness: a path this file doesn't shred has no typed leaf to read stats
/// from, so nothing is pruned and the upstream `Filter` still runs.
#[test]
fn variant_pushdown_keeps_all_groups_for_an_unshredded_path() {
    let dir = write_shredded_ages(&[10, 20, 30]);
    let (_database, datastore, mut table) = shredded_docs_datastore(dir.path());

    table
        .pushdown_filter(variant_filter(&["salary"], CompareType::Equal, 20))
        .unwrap();

    assert_eq!(row_group_count(&datastore, "docs", &table), 3);
}

/// Soundness: a row may store the path's value in the binary `value` fallback
/// instead of the typed leaf (the spec allows it for values of another
/// variant type), where the typed leaf's stats can't see it. A row group with
/// any non-null fallback must not be pruned.
#[test]
fn variant_pushdown_keeps_groups_whose_value_fallback_holds_data() {
    use arrow_array::{BinaryViewArray, StructArray};
    use arrow_schema::Fields;

    // Hand-built shredded shape, two rows per group: group 0 has one typed
    // age (10) and one row with `age` in the fallback `value`; group 1 is
    // fully shredded (20, 30) with an all-null fallback.
    let age_group = Fields::from(vec![
        Field::new("value", DataType::BinaryView, true),
        Field::new("typed_value", DataType::Int64, true),
    ]);
    let typed_group = Fields::from(vec![Field::new(
        "age",
        DataType::Struct(age_group.clone()),
        true,
    )]);
    let doc_fields = Fields::from(vec![
        Field::new("metadata", DataType::BinaryView, false),
        Field::new("value", DataType::BinaryView, true),
        Field::new("typed_value", DataType::Struct(typed_group.clone()), true),
    ]);
    let age = StructArray::new(
        age_group,
        vec![
            Arc::new(BinaryViewArray::from(vec![
                None,
                Some(b"unshredded".as_ref()),
                None,
                None,
            ])) as ArrayRef,
            Arc::new(Int64Array::from(vec![Some(10), None, Some(20), Some(30)])) as ArrayRef,
        ],
        None,
    );
    let typed_value = StructArray::new(typed_group, vec![Arc::new(age) as ArrayRef], None);
    let doc = StructArray::new(
        doc_fields.clone(),
        vec![
            Arc::new(BinaryViewArray::from(vec![b"m".as_ref(); 4])) as ArrayRef,
            Arc::new(BinaryViewArray::from(vec![None::<&[u8]>; 4])) as ArrayRef,
            Arc::new(typed_value) as ArrayRef,
        ],
        None,
    );
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "doc",
            DataType::Struct(doc_fields),
            true,
        )])),
        vec![Arc::new(doc) as ArrayRef],
    )
    .unwrap();
    let dir = TempDir::new().unwrap();
    let props = WriterProperties::builder()
        .set_statistics_enabled(EnabledStatistics::Chunk)
        .set_max_row_group_row_count(Some(2))
        .build();
    let file = File::create(dir.path().join("docs.parquet")).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    let (_database, datastore, mut table) = shredded_docs_datastore(dir.path());

    table
        .pushdown_filter(variant_filter(&["age"], CompareType::Equal, 20))
        .unwrap();

    // Group 0's typed stats ([10, 10]) exclude 20, but its fallback row could
    // still hold a matching age, so it survives alongside group 1.
    assert_eq!(row_group_count(&datastore, "docs", &table), 2);
}

/// Mixed files: the shredding file prunes by its typed leaf's stats while the
/// unshredded file (no typed leaf, no stats) keeps every group.
#[test]
fn variant_pushdown_prunes_only_the_file_that_shreds_the_path() {
    let dir = TempDir::new().unwrap();
    let shredded: Vec<String> = [10, 20, 30]
        .iter()
        .map(|a| format!(r#"{{"age":{a}}}"#))
        .collect();
    let unshredded = vec![r#"{"age":50}"#.to_string()];
    write_docs_file_into(dir.path(), "shredded.parquet", &shredded, Some("age"));
    write_docs_file_into(dir.path(), "plain.parquet", &unshredded, None);
    let (_database, datastore, mut table) = shredded_docs_datastore(dir.path());
    assert_eq!(row_group_count(&datastore, "docs", &table), 4);

    table
        .pushdown_filter(variant_filter(&["age"], CompareType::Equal, 20))
        .unwrap();

    // The shredded file keeps only its age=20 group; the unshredded file's
    // group has no typed leaf to prune by and must survive.
    assert_eq!(row_group_count(&datastore, "docs", &table), 2);
}

/// A nested path (`user.id`) prunes by the typed leaf two shredding levels
/// deep, resolved through each level's `typed_value` group.
#[test]
fn variant_pushdown_prunes_by_a_nested_shredded_path() {
    let dir = TempDir::new().unwrap();
    let docs: Vec<String> = [10, 20, 30]
        .iter()
        .map(|id| format!(r#"{{"user":{{"id":{id}}}}}"#))
        .collect();
    write_docs_file_into(dir.path(), "docs.parquet", &docs, Some("user.id"));
    let (_database, datastore, mut table) = shredded_docs_datastore(dir.path());
    assert_eq!(row_group_count(&datastore, "docs", &table), 3);

    table
        .pushdown_filter(variant_filter(&["user", "id"], CompareType::Equal, 20))
        .unwrap();

    assert_eq!(row_group_count(&datastore, "docs", &table), 1);
}

/// Row `i`'s payload in [`banded_table`], wide enough that a query selecting a
/// few rows out of it is one DuckDB late-materializes.
fn payload_of(row: usize) -> String {
    format!("payload-{row:08}-{}", "x".repeat(200))
}

/// A table cut into eight row groups whose `band` column is the row group's own
/// index, so a `band = k` predicate eliminates every other row group by its
/// statistics. Late materialization addresses rows by their row group's
/// position, which makes that pruning observable: a re-read that numbered row
/// groups differently from the scan returns another band's payloads.
fn banded_table() -> (TempDir, Vec<Column>) {
    let rows = 4096usize;
    let per_group = 512usize;
    let schema = Arc::new(Schema::new(vec![
        Field::new("band", DataType::Int32, false),
        Field::new("payload", DataType::Utf8View, false),
        Field::new("note", DataType::Utf8View, false),
    ]));
    let payloads: Vec<String> = (0..rows).map(payload_of).collect();
    let views: Vec<&str> = payloads.iter().map(String::as_str).collect();
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(
                (0..rows)
                    .map(|i| (i / per_group) as i32)
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringViewArray::from(views.clone())) as ArrayRef,
            Arc::new(StringViewArray::from(views)) as ArrayRef,
        ],
    )
    .unwrap();

    let dir = TempDir::new().unwrap();
    let props = WriterProperties::builder()
        .set_statistics_enabled(EnabledStatistics::Chunk)
        .set_compression(parquet::basic::Compression::SNAPPY)
        .set_max_row_group_row_count(Some(per_group))
        .build();
    let file = File::create(dir.path().join("data.parquet")).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let columns = vec![
        Column {
            name: "band".to_string(),
            col_type: Type::Int32,
        },
        Column {
            name: "payload".to_string(),
            col_type: Type::Utf8,
        },
        Column {
            name: "note".to_string(),
            col_type: Type::Utf8,
        },
    ];
    (dir, columns)
}

/// A late-materialized query whose predicate also prunes row groups returns the
/// rows it asked for.
///
/// The narrow scan hands the materializer a row group's position in the view it
/// scanned, which the pushed `band` predicate has pruned. Re-reading from the
/// unpruned view resolves those positions to different row groups, which shows
/// up here as another band's payloads rather than as an error.
#[test]
fn late_materialization_reads_the_scanned_row_groups() {
    let (dir, columns) = banded_table();
    let (_database, datastore) = empty_datastore();
    create_table(&datastore, create_request("t", dir.path(), columns)).unwrap();

    let batches = run_sql(
        &datastore,
        "SELECT payload, note FROM t WHERE band = 6 ORDER BY payload LIMIT 4",
    );

    let got: Vec<String> = batches
        .iter()
        .flat_map(|batch| {
            let column = batch.column(0).as_string_view();
            (0..batch.num_rows()).map(move |i| column.value(i).to_string())
        })
        .collect();
    let expected: Vec<String> = (6 * 512..6 * 512 + 4).map(payload_of).collect();
    assert_eq!(got, expected);
}

/// The defect stated without reference to any expected row: a `band = 6` query
/// must not return rows whose `band` is something else.
///
/// When the pushed `band` predicate prunes row groups, the late re-read
/// resolves the scan's row-group positions against an unpruned list and returns
/// a different row group's rows entirely, so the output contradicts the
/// predicate that produced it.
#[test]
fn late_materialization_rows_satisfy_their_own_predicate() {
    let (dir, columns) = banded_table();
    let (_database, datastore) = empty_datastore();
    create_table(&datastore, create_request("t", dir.path(), columns)).unwrap();

    let batches = run_sql(
        &datastore,
        "SELECT band, payload, note FROM t WHERE band = 6 ORDER BY payload LIMIT 4",
    );

    let bands: Vec<i32> = batches
        .iter()
        .flat_map(|batch| {
            let column = batch.column(0).as_primitive::<Int32Type>();
            (0..batch.num_rows()).map(move |i| column.value(i))
        })
        .collect();
    assert_eq!(bands, vec![6; 4]);
}
