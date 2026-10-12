use crate::common::*;
use dispatch::RowDelivery;
use insta::assert_snapshot;
use rstest::rstest;

// A user `IN`/`EXISTS` subquery lowers to a semi-join, which pivot runs as a
// probe-side semi join: the probe row comes out once, and no build column comes
// out at all.
#[rstest]
fn in_subquery_semijoin_is_a_join(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT a FROM example_table WHERE a IN (SELECT b FROM example_table WHERE b > 20)")
        .unwrap();

    assert_snapshot!(plan.to_string(), @"
    Projection(a:Int32)
      Join[probe semi](probe_keys: [0], build_keys: [0], probe_output: [0], build_output: [], publishes key bounds: [0])
        Input([a:Int32], dynamic: [DynamicFilter(#0 >= ?), DynamicFilter(#0 <= ?)], key bitsets: [#0 in build keys])
        Projection(b:Int32)
          Filter(b:Int32 > 20:Int32 -> Boolean)
            Input([b:Int32])
    ");
}

#[rstest]
fn simple_select(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT a, b FROM example_table")
        .unwrap();
    assert_snapshot!(plan.to_string(), @"
    Projection(a:Int32, b:Int32)
      Input([a:Int32, b:Int32])
    ");
}

#[rstest]
fn select_is_cacheable_when_its_table_revision_matches(mut testing_planner: TestingPlanner) {
    let plan = testing_planner.plan("SELECT a FROM example_table").unwrap();
    let transaction = testing_planner.transaction();

    assert!(plan.is_cacheable());
    assert!(
        plan.has_matching_table_revisions(transaction.as_ref())
            .unwrap()
    );
}

#[rstest]
fn table_free_query_is_cacheable(mut testing_planner: TestingPlanner) {
    let plan = testing_planner.plan("SELECT 1").unwrap();
    let transaction = testing_planner.transaction();

    assert!(plan.is_cacheable());
    assert!(
        plan.has_matching_table_revisions(transaction.as_ref())
            .unwrap()
    );
}

#[rstest]
fn query_with_now_is_not_cacheable(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT now() - interval '4 minutes'")
        .unwrap();

    assert!(!plan.is_cacheable());
}

#[rstest]
fn mutating_session_and_table_function_plans_are_not_cacheable(
    mut testing_planner: TestingPlanner,
) {
    let insert = testing_planner
        .plan("INSERT INTO example_table VALUES (6, 60, 600, 'eve')")
        .unwrap();
    let create = testing_planner
        .plan("CREATE TABLE cacheability_test (id INTEGER)")
        .unwrap();
    let set = testing_planner.plan("SET pivot_stats = true").unwrap();
    let create_user = testing_planner.plan("CREATE USER cache_test_user").unwrap();
    let table_function = testing_planner
        .plan("SELECT * FROM generate_series(1, 2)")
        .unwrap();

    assert!(!insert.is_cacheable());
    assert!(!create.is_cacheable());
    assert!(!set.is_cacheable());
    assert!(!create_user.is_cacheable());
    assert!(!table_function.is_cacheable());
}

#[rstest]
fn explain_wraps_the_explained_plan(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("EXPLAIN SELECT a, b FROM example_table")
        .unwrap();
    assert_snapshot!(plan.to_string(), @"
    Explain
      Projection(a:Int32, b:Int32)
        Input([a:Int32, b:Int32])
    ");
}

#[rstest]
fn explain_analyze_is_rejected(mut testing_planner: TestingPlanner) {
    let error = testing_planner
        .plan("EXPLAIN ANALYZE SELECT a, b FROM example_table")
        .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("EXPLAIN ANALYZE is not supported"),
        "{error}"
    );
}

#[rstest]
fn filter(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT a FROM example_table WHERE a <> b")
        .unwrap();
    assert_snapshot!(plan.to_string(), @"
    Projection(a:Int32)
      Projection(#0:Int32)
        Filter(a:Int32 <> b:Int32 -> Boolean)
          Input([a:Int32, b:Int32])
    ");
}

// DuckDB pushes the simple `a > 5` into the scan's table_filters while the
// column-to-column `a <> b` stays in the plan's Filter; the walk must fold
// both into ONE Filter (pushed condition first), never a stack of two - a
// stack materializes survivors once per level.
#[rstest]
fn pushed_and_residual_conditions_fold_into_one_filter(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT a FROM example_table WHERE a > 5 AND a <> b")
        .unwrap();
    assert_snapshot!(plan.to_string(), @"
    Projection(a:Int32)
      Projection(#0:Int32)
        Filter(a:Int32 > 5:Int32 -> Boolean AND a:Int32 <> b:Int32 -> Boolean)
          Input([a:Int32, b:Int32])
    ");
}

#[rstest]
fn order_by_limit_produces_top_n(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT a FROM example_table ORDER BY a DESC LIMIT 2")
        .unwrap();
    assert_snapshot!(plan.to_string(), @r#"
    TopN(limit: 2, offset: 0, order: "default".main.example_table.a:Int32 DESC NULLS LAST)
      Projection(a:Int32)
        Input([a:Int32], dynamic: [DynamicFilter(#0 > ?)])
    "#);
}

#[rstest]
fn order_by_without_limit_produces_order_by(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT a FROM example_table ORDER BY a DESC")
        .unwrap();
    assert_snapshot!(plan.to_string(), @r#"
    OrderBy("default".main.example_table.a:Int32 DESC NULLS LAST)
      Projection(a:Int32)
        Input([a:Int32])
    "#);
}

#[rstest]
fn simple_aggregate(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT COUNT(*) FROM example_table WHERE a <> 0")
        .unwrap();
    assert_snapshot!(plan.to_string(), @"
    Projection(count_star():Int64)
      Aggregate(groups: [], exprs: [count_star()])
        Filter(a:Int32 <> 0:Int32 -> Boolean)
          Input([a:Int32])
    ");
}

#[rstest]
fn aggregate_with_single_group(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT a, COUNT(*) FROM example_table GROUP BY a")
        .unwrap();
    assert_snapshot!(plan.to_string(), @"
    Projection(a:Int32, count_star():Int64)
      Aggregate(groups: [a:Int32], exprs: [count_star()])
        Input([a:Int32])
    ");
}

#[rstest]
fn aggregate_with_multiple_groups(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT a, b, COUNT(*) FROM example_table GROUP BY a, b")
        .unwrap();
    assert_snapshot!(plan.to_string(), @"
    Projection(a:Int32, b:Int32, count_star():Int64)
      Aggregate(groups: [a:Int32, b:Int32], exprs: [count_star()])
        Input([a:Int32, b:Int32])
    ");
}

#[rstest]
fn input_references_correct_columns(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT a, c FROM example_table")
        .unwrap();
    // `a` is column #0 and `c` is column #2 in the table; the Input scans
    // those two source columns, while the Projection references them by
    // their positions in the Input's output (#0 and #1).
    assert_snapshot!(plan.to_string(), @"
    Projection(a:Int32, c:Int32)
      Input([a:Int32, c:Int32])
    ");
}

#[rstest]
fn create_table_produces_create_table_operator(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("CREATE TABLE created_table (id INTEGER, name VARCHAR)")
        .unwrap();
    assert_snapshot!(
        plan.to_string(),
        @"CreateTable(created_table, [id:Int32, name:Utf8], options: {})
    "
    );
}

#[rstest]
fn create_table_preserves_timestamp_zone_semantics(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("CREATE TABLE events (local_time TIMESTAMP, occurred TIMESTAMPTZ)")
        .unwrap();
    assert_snapshot!(
        plan.to_string(),
        @"CreateTable(events, [local_time:Timestamp, occurred:TimestampTz], options: {})
    "
    );
}

#[rstest]
fn create_table_propagates_with_options(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("CREATE TABLE created_table (id INTEGER) WITH (path='/asdf', format='parquet')")
        .unwrap();
    assert_snapshot!(
        plan.to_string(),
        @r#"CreateTable(created_table, [id:Int32], options: {"format": "parquet", "path": "/asdf"})
    "#
    );
}

#[rstest]
fn drop_table_produces_drop_table_operator(mut testing_planner: TestingPlanner) {
    let plan = testing_planner.plan("DROP TABLE example_table").unwrap();

    assert_snapshot!(plan.to_string(), @"DropTable(example_table)
    ");
}

#[rstest]
fn drop_table_if_exists_of_a_missing_table_still_plans(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("DROP TABLE IF EXISTS no_such_table")
        .unwrap();

    assert_snapshot!(plan.to_string(), @"DropTable(no_such_table)
    ");
}

#[rstest]
fn drop_of_a_missing_table_without_if_exists_fails_at_binding(mut testing_planner: TestingPlanner) {
    let err = testing_planner
        .plan("DROP TABLE no_such_table")
        .unwrap_err()
        .to_string();

    assert!(err.contains("no_such_table"), "{err}");
}

#[rstest]
fn drop_of_a_non_table_entry_is_unsupported(mut testing_planner: TestingPlanner) {
    let err = testing_planner
        .plan("DROP SCHEMA some_schema")
        .unwrap_err()
        .to_string();

    assert!(err.contains("DROP Schema is not supported"), "{err}");
}

#[rstest]
fn create_user_produces_create_user_operator(mut testing_planner: TestingPlanner) {
    let plan = testing_planner.plan("CREATE USER alice").unwrap();

    assert_snapshot!(plan.to_string(), @"CreateUser(alice)
    ");
}

#[rstest]
fn create_user_with_password_redacts_it(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("CREATE USER alice PASSWORD 'secret'")
        .unwrap();

    let rendered = plan.to_string();

    assert!(!rendered.contains("secret"));
    assert_snapshot!(rendered, @"CreateUser(alice PASSWORD [redacted])
    ");
}

#[rstest]
fn create_user_with_invalid_utf8_password_fails_loudly(mut testing_planner: TestingPlanner) {
    let result = testing_planner.plan(r"CREATE USER alice PASSWORD E'\xff'");

    // Never silently alter an unrepresentable password: the scanner rejects
    // the bytes today, and the bridge refuses lossy conversion as well.
    result.expect_err("an unrepresentable password must not be silently altered");
}

#[rstest]
fn values_produces_values_operator(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("VALUES (1, 'one'), (2, 'two')")
        .unwrap();
    assert_snapshot!(plan.to_string(), @"
    Projection(col0:Int32, col1:Utf8)
      Values(rows: 2)
        DummyScan
    ");
}

#[rstest]
fn insert_values_produces_insert_and_values_operators(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("INSERT INTO example_table VALUES (6, 60, 600, 'eve'), (7, 70, 700, 'bob')")
        .unwrap();
    assert_snapshot!(plan.to_string(), @"
    Insert
      Projection(a:Int32, b:Int32, c:Int32, name:Utf8)
        Values(rows: 2)
          DummyScan
    ");
}

#[rstest]
fn insert_select_keeps_distinct_target_and_scan_bindings(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("INSERT INTO example_table SELECT a, b, c, name FROM example_table")
        .unwrap();
    assert_snapshot!(plan.to_string(), @"
    Insert
      Projection(a:Int32, b:Int32, c:Int32, name:Utf8)
        Input([a:Int32, b:Int32, c:Int32, name:Utf8])
    ");
}

#[rstest]
fn insert_with_a_column_list_widens_its_values_to_the_whole_table(
    mut testing_planner: TestingPlanner,
) {
    let plan = testing_planner
        .plan("INSERT INTO example_table (name, a) VALUES ('eve', 6)")
        .unwrap();
    assert_snapshot!(plan.to_string(), @"
    Insert
      Projection(a:Int32, NULL:Int32, NULL:Int32, name:Utf8)
        Projection(name:Utf8, a:Int32)
          Values(rows: 1)
            DummyScan
    ");
}

#[rstest]
fn insert_select_with_a_column_list_widens_the_select(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("INSERT INTO example_table (c, b) SELECT c, b FROM example_table")
        .unwrap();
    assert_snapshot!(plan.to_string(), @"
    Insert
      Projection(NULL:Int32, b:Int32, c:Int32, NULL:Utf8View)
        Projection(c:Int32, b:Int32)
          Input([c:Int32, b:Int32])
    ");
}

#[rstest]
fn insert_by_name_matches_the_select_output_names(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("INSERT INTO example_table BY NAME SELECT name, a FROM example_table")
        .unwrap();
    assert_snapshot!(plan.to_string(), @"
    Insert
      Projection(a:Int32, NULL:Int32, NULL:Int32, name:Utf8)
        Projection(name:Utf8, a:Int32)
          Input([name:Utf8, a:Int32])
    ");
}

/// A join Pivot cannot plan is reported by DuckDB's name for the join type,
/// not by its numeric type.
#[rstest]
fn an_unsupported_join_type_is_named_in_the_error(mut testing_planner: TestingPlanner) {
    let error = testing_planner
        .plan(
            "SELECT (SELECT max(b) FROM example_table inner_table \
             WHERE inner_table.a = example_table.a) FROM example_table",
        )
        .unwrap_err()
        .to_string();

    assert!(
        error.ends_with("Unsupported range join type: LEFT"),
        "{error}"
    );
}

#[rstest]
fn insert_default_values_is_rejected(mut testing_planner: TestingPlanner) {
    let error = testing_planner
        .plan("INSERT INTO example_table DEFAULT VALUES")
        .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("DEFAULT VALUES is not supported"),
        "{error}"
    );
}

#[rstest]
fn insert_naming_an_unknown_column_is_rejected(mut testing_planner: TestingPlanner) {
    let error = testing_planner
        .plan("INSERT INTO example_table (nope) VALUES (1)")
        .unwrap_err();

    assert!(error.to_string().contains("nope"), "{error}");
}

#[rstest]
fn combined_filter_order_limit(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT a FROM example_table WHERE a <> b ORDER BY a DESC LIMIT 2")
        .unwrap();
    assert_snapshot!(plan.to_string(), @r#"
    TopN(limit: 2, offset: 0, order: "default".main.example_table.a:Int32 DESC NULLS LAST)
      Projection(a:Int32)
        Projection(#0:Int32)
          Filter(a:Int32 <> b:Int32 -> Boolean)
            Input([a:Int32, b:Int32], dynamic: [DynamicFilter(#0 > ?)])
    "#);
}

#[rstest]
fn set_variable_is_parsed_by_duckdb(mut testing_planner: TestingPlanner) {
    let plan = testing_planner.plan("SET pivot_stats = true").unwrap();

    let set = plan
        .as_set_variable()
        .expect("SET should not compile to a query");

    assert_eq!(set.name, "pivot_stats");
    assert_eq!(set.value.as_deref(), Some("true"));
}

#[rstest]
fn reset_variable_carries_no_value(mut testing_planner: TestingPlanner) {
    let plan = testing_planner.plan("RESET pivot_stats").unwrap();

    let set = plan
        .as_set_variable()
        .expect("RESET should not compile to a query");

    assert_eq!(set.name, "pivot_stats");
    assert_eq!(set.value, None);
}

fn add_join_tables(planner: &TestingPlanner) {
    use arrow_array::{ArrayRef, Int64Array};
    use planner::types::Type;
    use std::sync::Arc;

    let int64_col = |values: Vec<i64>| -> ArrayRef { Arc::new(Int64Array::from(values)) };
    planner.add_table(
        "orders",
        &[
            ("o_key", Type::Int64, int64_col(vec![1, 2])),
            ("o_total", Type::Int64, int64_col(vec![10, 20])),
        ],
    );
    planner.add_table(
        "items",
        &[
            ("i_order", Type::Int64, int64_col(vec![1, 1, 2])),
            ("i_qty", Type::Int64, int64_col(vec![5, 6, 7])),
        ],
    );
}

/// A filtered correlated scalar subquery decorrelates into a LEFT delim join,
/// which the walk desugars into two synthetic CTEs: the outer side read by the
/// join's build and by a Distinct on the correlation column, whose output the
/// subquery side's delim scan reads. The projection above the join restores
/// DuckDB's LHS-then-RHS column order around the build-outer orientation.
#[rstest]
fn correlated_scalar_subquery_becomes_a_delim_join(mut testing_planner: TestingPlanner) {
    add_join_tables(&testing_planner);

    let plan = testing_planner
        .plan(
            "SELECT sum(i_qty) FROM items, orders WHERE o_key = i_order AND o_total = 10 \
             AND i_qty < (SELECT avg(i2.i_qty) FROM items i2 WHERE i2.i_order = o_key)",
        )
        .unwrap();

    assert_snapshot!(plan.to_string(), @"
    Projection(sum(i_qty):Int128)
      Aggregate(groups: [], exprs: [sum(i_qty:Int64)])
        Projection(#0:Int64)
          Filter(cast(i_qty:Int64 as Float64) < SUBQUERY:Float64 -> Boolean)
            Cte(#9223372036854775807, sites: 2)
              Join(probe_keys: [0], build_keys: [0], probe_output: [0, 1], build_output: [0, 1], publishes key bounds: [0])
                Input([i_order:Int64, i_qty:Int64], dynamic: [DynamicFilter(#0 >= ?), DynamicFilter(#0 <= ?)], key bitsets: [#0 in build keys])
                Filter(o_total:Int64 = 10:Int64 -> Boolean)
                  Input([o_key:Int64, o_total:Int64])
              Cte(#9223372036854775808, sites: 1)
                Distinct(keys: [0])
                  CteScan(#9223372036854775807)
                Join[probe outer](probe_keys: [0], build_keys: [1], probe_output: [1], build_output: [0])
                  CteScan(#9223372036854775807)
                  Projection(avg(i_qty):Float64, o_key:Int64)
                    Projection(#0:Int64, (cast(#1:Int128 as Float64) / cast(#2:Int64 as Float64)))
                      Aggregate(groups: [o_key:Int64], exprs: [sum(i_qty:Int64), count(i_qty:Int64)])
                        Join(probe_keys: [0], build_keys: [0], probe_output: [0, 1], build_output: [0], publishes key bounds: [0])
                          Input([i_order:Int64, i_qty:Int64], dynamic: [DynamicFilter(#0 >= ?), DynamicFilter(#0 <= ?)], key bitsets: [#0 in build keys])
                          CteScan(#9223372036854775808)
    ");
}

/// An IN list of five or more constants is rewritten by DuckDB into a mark
/// join against an in-memory chunk of them (the rewrite runs after the pass
/// that would have relaxed the marker into a semi join, so MARK is what
/// arrives). The chunk scans as an inline VALUES source on the build side
/// and the filter above reads the marker column.
#[rstest]
fn a_long_in_list_becomes_a_mark_join_against_inline_values(mut testing_planner: TestingPlanner) {
    use arrow_array::{ArrayRef, Int64Array};
    use planner::types::Type;
    use std::sync::Arc;
    let int64_col = |values: Vec<i64>| -> ArrayRef { Arc::new(Int64Array::from(values)) };
    testing_planner.add_table(
        "wide_items",
        &[("i_order", Type::Int64, int64_col(vec![7; 20]))],
    );

    let plan = testing_planner
        .plan("SELECT i_order FROM wide_items WHERE i_order IN (1, 2, 3, 4, 5, 7)")
        .unwrap();

    assert_snapshot!(plan.to_string(), @"
    Projection(i_order:Int64)
      Projection(#0:Int64)
        Filter(IN (...):Boolean)
          Join[probe mark](probe_keys: [0], build_keys: [0], probe_output: [0], build_output: [])
            Input([i_order:Int64])
            Values(rows: 6)
    ");
}

/// NOT EXISTS decorrelates into an ANTI delim join: the same synthetic-CTE
/// tee and Distinct as the other delim kinds, with the final join anti on the
/// probe (outer) side and no build columns.
#[rstest]
fn a_not_exists_subquery_becomes_an_anti_delim_join(mut testing_planner: TestingPlanner) {
    add_join_tables(&testing_planner);

    let plan = testing_planner
        .plan(
            "SELECT o_key FROM orders WHERE NOT EXISTS \
             (SELECT 1 FROM items WHERE i_order = o_key AND i_qty > 5)",
        )
        .unwrap();

    assert_snapshot!(plan.to_string(), @"
    Projection(o_key:Int64)
      Cte(#9223372036854775807, sites: 2)
        Input([o_key:Int64])
        Cte(#9223372036854775808, sites: 1)
          Distinct(keys: [0])
            CteScan(#9223372036854775807)
          Join[probe anti](probe_keys: [0], build_keys: [0], probe_output: [0], build_output: [])
            CteScan(#9223372036854775807)
            Projection(o_key:Int64)
              Join(probe_keys: [0], build_keys: [0], probe_output: [0], build_output: [0], publishes key bounds: [0])
                Filter(i_qty:Int64 > 5:Int64 -> Boolean)
                  Input([i_order:Int64, i_qty:Int64], dynamic: [DynamicFilter(#0 >= ?), DynamicFilter(#0 <= ?)], key bitsets: [#0 in build keys])
                CteScan(#9223372036854775808)
    ");
}

/// The same EXISTS with the outer side far smaller flips in DuckDB's plan:
/// the outer rows become the RHS and the dedup source, the join arrives as
/// RIGHT_SEMI, and the walk keeps its CTE tee on the outer side while the
/// subquery side probes, so the join lands build-side semi.
#[rstest]
fn a_small_outer_exists_becomes_a_flipped_delim_join(mut testing_planner: TestingPlanner) {
    use arrow_array::{ArrayRef, Int64Array};
    use planner::types::Type;
    use std::sync::Arc;
    let int64_col = |values: Vec<i64>| -> ArrayRef { Arc::new(Int64Array::from(values)) };
    testing_planner.add_table(
        "small_orders",
        &[("o_key", Type::Int64, int64_col(vec![1, 2]))],
    );
    testing_planner.add_table(
        "wide_items",
        &[
            ("i_order", Type::Int64, int64_col(vec![7; 1000])),
            ("i_qty", Type::Int64, int64_col((0..1000).collect())),
        ],
    );

    let plan = testing_planner
        .plan(
            "SELECT o_key FROM small_orders WHERE EXISTS \
             (SELECT 1 FROM wide_items WHERE i_order = o_key AND i_qty > 5)",
        )
        .unwrap();

    assert_snapshot!(plan.to_string(), @"
    Projection(o_key:Int64)
      Cte(#9223372036854775807, sites: 2)
        Input([o_key:Int64])
        Cte(#9223372036854775808, sites: 1)
          Distinct(keys: [0])
            CteScan(#9223372036854775807)
          Join[build semi](probe_keys: [0], build_keys: [0], probe_output: [], build_output: [0])
            Projection(o_key:Int64)
              Join(probe_keys: [0], build_keys: [0], probe_output: [0], build_output: [0], publishes key bounds: [0])
                Filter(i_qty:Int64 > 5:Int64 -> Boolean)
                  Input([i_order:Int64, i_qty:Int64], dynamic: [DynamicFilter(#0 >= ?), DynamicFilter(#0 <= ?)], key bitsets: [#0 in build keys])
                CteScan(#9223372036854775808)
            CteScan(#9223372036854775807)
    ");
}

/// A LEFT JOIN preserving the larger relation stays LEFT in DuckDB's plan and
/// lowers to the probe-side outer join: the preserved side streams as probe
/// and its unmatched rows pad with null build columns.
#[rstest]
fn a_left_join_preserving_the_larger_side_is_probe_outer(mut testing_planner: TestingPlanner) {
    use arrow_array::{ArrayRef, Int64Array};
    use planner::types::Type;
    use std::sync::Arc;
    let int64_col = |values: Vec<i64>| -> ArrayRef { Arc::new(Int64Array::from(values)) };
    testing_planner.add_table(
        "small_orders",
        &[("o_key", Type::Int64, int64_col(vec![1, 2]))],
    );
    testing_planner.add_table(
        "wide_items",
        &[("i_order", Type::Int64, int64_col(vec![7; 20]))],
    );

    let plan = testing_planner
        .plan("SELECT i_order, o_key FROM wide_items LEFT JOIN small_orders ON i_order = o_key")
        .unwrap();

    assert_snapshot!(plan.to_string(), @"
    Projection(i_order:Int64, o_key:Int64)
      Join[probe outer](probe_keys: [0], build_keys: [0], probe_output: [0], build_output: [0])
        Input([i_order:Int64])
        Input([o_key:Int64])
    ");
}

/// An ANTI JOIN preserving the larger relation keeps its orientation and
/// lowers to the probe-side anti join, which emits no build columns.
#[rstest]
fn an_anti_join_preserving_the_larger_side_is_probe_anti(mut testing_planner: TestingPlanner) {
    use arrow_array::{ArrayRef, Int64Array};
    use planner::types::Type;
    use std::sync::Arc;
    let int64_col = |values: Vec<i64>| -> ArrayRef { Arc::new(Int64Array::from(values)) };
    testing_planner.add_table(
        "small_orders",
        &[("o_key", Type::Int64, int64_col(vec![1, 2]))],
    );
    testing_planner.add_table(
        "wide_items",
        &[("i_order", Type::Int64, int64_col(vec![7; 20]))],
    );

    let plan = testing_planner
        .plan("SELECT i_order FROM wide_items ANTI JOIN small_orders ON i_order = o_key")
        .unwrap();

    assert_snapshot!(plan.to_string(), @"
    Projection(i_order:Int64)
      Join[probe anti](probe_keys: [0], build_keys: [0], probe_output: [0], build_output: [])
        Input([i_order:Int64])
        Input([o_key:Int64])
    ");
}

/// The same ANTI JOIN preserving the smaller relation flips in DuckDB's plan:
/// the preserved side becomes the build, the join arrives as RIGHT_ANTI, and
/// it lowers to the build-side anti join, which emits no probe columns.
#[rstest]
fn an_anti_join_preserving_the_smaller_side_is_build_anti(mut testing_planner: TestingPlanner) {
    use arrow_array::{ArrayRef, Int64Array};
    use planner::types::Type;
    use std::sync::Arc;
    let int64_col = |values: Vec<i64>| -> ArrayRef { Arc::new(Int64Array::from(values)) };
    testing_planner.add_table(
        "small_orders",
        &[("o_key", Type::Int64, int64_col(vec![1, 2]))],
    );
    testing_planner.add_table(
        "wide_items",
        &[("i_order", Type::Int64, int64_col(vec![7; 20]))],
    );

    let plan = testing_planner
        .plan("SELECT o_key FROM small_orders ANTI JOIN wide_items ON o_key = i_order")
        .unwrap();

    assert_snapshot!(plan.to_string(), @"
    Projection(o_key:Int64)
      Join[build anti](probe_keys: [0], build_keys: [0], probe_output: [], build_output: [0])
        Input([i_order:Int64])
        Input([o_key:Int64])
    ");
}

/// A SEMI JOIN preserving the smaller relation flips the same way: the
/// preserved side becomes the build, the join arrives as RIGHT_SEMI, and it
/// lowers to the build-side semi join, which emits no probe columns.
#[rstest]
fn a_semi_join_preserving_the_smaller_side_is_build_semi(mut testing_planner: TestingPlanner) {
    use arrow_array::{ArrayRef, Int64Array};
    use planner::types::Type;
    use std::sync::Arc;
    let int64_col = |values: Vec<i64>| -> ArrayRef { Arc::new(Int64Array::from(values)) };
    testing_planner.add_table(
        "small_orders",
        &[("o_key", Type::Int64, int64_col(vec![1, 2]))],
    );
    testing_planner.add_table(
        "wide_items",
        &[("i_order", Type::Int64, int64_col(vec![7; 20]))],
    );

    let plan = testing_planner
        .plan("SELECT o_key FROM small_orders SEMI JOIN wide_items ON o_key = i_order")
        .unwrap();

    assert_snapshot!(plan.to_string(), @"
    Projection(o_key:Int64)
      Join[build semi](probe_keys: [0], build_keys: [0], probe_output: [], build_output: [0], publishes key bounds: [0])
        Projection(#0:Int64)
          Input([i_order:Int64], dynamic: [DynamicFilter(#0 >= ?), DynamicFilter(#0 <= ?)], key bitsets: [#0 in build keys])
        Input([o_key:Int64])
    ");
}

#[rstest]
fn count_star_join_keeps_all_columns(mut testing_planner: TestingPlanner) {
    add_join_tables(&testing_planner);

    let plan = testing_planner
        .plan("SELECT count(*) FROM items JOIN orders ON i_order = o_key")
        .unwrap();

    assert_snapshot!(plan.to_string(), @"
    Projection(count_star():Int64)
      Aggregate(groups: [], exprs: [count_star()])
        Join(probe_keys: [0], build_keys: [0], probe_output: [0], build_output: [0], publishes key bounds: [0])
          Input([i_order:Int64], dynamic: [DynamicFilter(#0 >= ?), DynamicFilter(#0 <= ?)], key bitsets: [#0 in build keys])
          Input([o_key:Int64])
    ");
}

#[rstest]
fn join_output_folds_projection_maps(mut testing_planner: TestingPlanner) {
    add_join_tables(&testing_planner);

    let plan = testing_planner
        .plan("SELECT i_qty, o_total FROM items JOIN orders ON i_order = o_key")
        .unwrap();

    assert_snapshot!(plan.to_string(), @"
    Projection(i_qty:Int64, o_total:Int64)
      Join(probe_keys: [0], build_keys: [0], probe_output: [1], build_output: [1], publishes key bounds: [0])
        Input([i_order:Int64, i_qty:Int64], dynamic: [DynamicFilter(#0 >= ?), DynamicFilter(#0 <= ?)], key bitsets: [#0 in build keys])
        Input([o_key:Int64, o_total:Int64])
    ");
}

#[rstest]
fn join_with_unread_build_side_keeps_it_anyway(mut testing_planner: TestingPlanner) {
    add_join_tables(&testing_planner);

    let plan = testing_planner
        .plan("SELECT sum(i_order) FROM items JOIN orders ON i_order = o_key")
        .unwrap();

    assert_snapshot!(plan.to_string(), @"
    Projection(sum(i_order):Int128)
      Aggregate(groups: [], exprs: [sum(i_order:Int64)])
        Join(probe_keys: [0], build_keys: [0], probe_output: [0], build_output: [0], publishes key bounds: [0])
          Input([i_order:Int64], dynamic: [DynamicFilter(#0 >= ?), DynamicFilter(#0 <= ?)], key bitsets: [#0 in build keys])
          Input([o_key:Int64])
    ");
}

// A join predicate that references both sides but is not a bare comparison
// (an OR of per-side conjunctions) stays on the join as a single-expression
// condition and becomes the join's residual, evaluated on key-matched pairs.
// The per-side OR prefilters below the join are DuckDB's own derivation.
#[rstest]
fn join_with_expression_condition_becomes_the_joins_residual(mut testing_planner: TestingPlanner) {
    add_join_tables(&testing_planner);

    let plan = testing_planner
        .plan(
            "SELECT count(*) FROM items JOIN orders ON i_order = o_key \
             AND ((i_qty < 6 AND o_total > 5) OR (i_qty > 6 AND o_total < 25))",
        )
        .unwrap();

    assert_snapshot!(plan.to_string(), @"
    Projection(count_star():Int64)
      Aggregate(groups: [], exprs: [count_star()])
        Join(probe_keys: [0], build_keys: [0], probe_output: [0, 1], build_output: [0, 1], residual: ((i_qty:Int64 < 6:Int64 -> Boolean AND o_total:Int64 > 5:Int64 -> Boolean) OR (i_qty:Int64 > 6:Int64 -> Boolean AND o_total:Int64 < 25:Int64 -> Boolean)), publishes key bounds: [0])
          Filter((i_qty:Int64 < 6:Int64 -> Boolean OR i_qty:Int64 > 6:Int64 -> Boolean))
            Input([i_order:Int64, i_qty:Int64], dynamic: [DynamicFilter(#0 >= ?), DynamicFilter(#0 <= ?)], key bitsets: [#0 in build keys])
          Filter((o_total:Int64 > 5:Int64 -> Boolean OR o_total:Int64 < 25:Int64 -> Boolean))
            Input([o_key:Int64, o_total:Int64])
    ");
}

/// The delivery of every filter in `plan`, in tree order.
fn filter_deliveries(node: &planner::plan::PlanNode) -> Vec<RowDelivery> {
    let mut found = match &node.operator {
        planner::operator::Operator::Filter(f) => vec![f.delivery],
        _ => vec![],
    };
    for child in &node.inputs {
        found.extend(filter_deliveries(child));
    }
    found
}

// A filter feeding a LIMIT must hand rows over as it selects them: the LIMIT
// cancels its input once it has enough, and rows held back to fill a batch
// keep the scan reading. The projections in between are transparent, so the
// choice reaches the filter through them.
#[rstest]
fn filter_under_a_limit_delivers_immediately(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT a FROM example_table WHERE a <> b LIMIT 10")
        .unwrap();

    assert_eq!(filter_deliveries(&plan.root), vec![RowDelivery::Immediate]);
}

// An ORDER BY with a LIMIT is a Top-N, which publishes the boundary that
// prunes row groups from the scan, so it wants rows just as early.
#[rstest]
fn filter_under_a_top_n_delivers_immediately(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT a FROM example_table WHERE a <> b ORDER BY a LIMIT 10")
        .unwrap();

    assert_eq!(filter_deliveries(&plan.root), vec![RowDelivery::Immediate]);
}

// A group-by reads its whole input before emitting anything, so its filter
// coalesces into full batches instead.
#[rstest]
fn filter_under_a_group_by_coalesces(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT a, count(*) FROM example_table WHERE a <> b GROUP BY a")
        .unwrap();

    assert_eq!(filter_deliveries(&plan.root), vec![RowDelivery::Coalesced]);
}

// The nearest consumer decides: a LIMIT above a group-by changes nothing for
// the filter, which still feeds the group-by.
#[rstest]
fn group_by_under_a_limit_still_coalesces_its_filter(mut testing_planner: TestingPlanner) {
    let plan = testing_planner
        .plan("SELECT a, count(*) FROM example_table WHERE a <> b GROUP BY a LIMIT 10")
        .unwrap();

    assert_eq!(filter_deliveries(&plan.root), vec![RowDelivery::Coalesced]);
}
