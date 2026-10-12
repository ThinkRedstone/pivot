use std::sync::{Arc, Mutex};

use arrow_array::{
    ArrayRef, Date32Array, Float64Array, Int16Array, Int32Array, Int64Array, RecordBatch,
    StringViewArray,
};
use dispatch::Dispatch;

use crate::common::*;
use planner::Error as PlannerError;
use planner::catalog::{BoundTable, CreateTableRequest, CreateUserRequest};
use planner::types::Type;
use planner::{DEFAULT_DATASTORE_NAME, Planner};
use rstest::rstest;

fn int_col(values: Vec<i32>) -> ArrayRef {
    Arc::new(Int32Array::from(values))
}

fn str_col(values: Vec<&'static str>) -> ArrayRef {
    Arc::new(StringViewArray::from(values))
}

fn int16_col(values: Vec<i16>) -> ArrayRef {
    Arc::new(Int16Array::from(values))
}

#[rstest]
fn explain_emits_plan_text_without_running_the_query(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .plan("EXPLAIN SELECT a, b FROM example_table")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let lines: Vec<String> = batches_to_json(&results)
        .into_iter()
        .map(|row| row["QUERY PLAN"].as_str().unwrap().to_string())
        .collect();

    assert_eq!(
        lines,
        vec![
            "Projection(a:Int32, b:Int32)",
            "  Input([a:Int32, b:Int32])",
        ]
    );
}

#[rstest]
fn select_column_subset(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .plan("SELECT a, b FROM example_table")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["a"].as_i64().unwrap());

    assert_eq!(
        rows,
        serde_json::json!([
            {"a": 1, "b": 10},
            {"a": 2, "b": 20},
            {"a": 3, "b": 30},
            {"a": 4, "b": 40},
            {"a": 5, "b": 50},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn select_all_columns(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .plan("SELECT a, b, c FROM example_table")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["a"].as_i64().unwrap());

    assert_eq!(
        rows,
        serde_json::json!([
            {"a": 1, "b": 10, "c": 100},
            {"a": 2, "b": 20, "c": 200},
            {"a": 3, "b": 30, "c": 300},
            {"a": 4, "b": 40, "c": 400},
            {"a": 5, "b": 50, "c": 500},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn select_single_column(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .plan("SELECT c FROM example_table")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["c"].as_i64().unwrap());

    assert_eq!(
        rows,
        serde_json::json!([
            {"c": 100},
            {"c": 200},
            {"c": 300},
            {"c": 400},
            {"c": 500},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn filter_not_equal_columns(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "pairs_with_dup",
        &[
            ("a", Type::Int32, int_col(vec![1, 2, 3, 10])),
            ("b", Type::Int32, int_col(vec![10, 20, 30, 10])),
        ],
    );

    let results = testing_planner
        .plan("SELECT a, b FROM pairs_with_dup WHERE a <> b")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["a"].as_i64().unwrap());

    // (10, 10) is excluded.
    assert_eq!(
        rows,
        serde_json::json!([
            {"a": 1, "b": 10},
            {"a": 2, "b": 20},
            {"a": 3, "b": 30},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn filter_not_equal_no_matches(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "pairs_all_equal",
        &[
            ("a", Type::Int32, int_col(vec![1, 2, 3])),
            ("b", Type::Int32, int_col(vec![1, 2, 3])),
        ],
    );

    let results = testing_planner
        .plan("SELECT a FROM pairs_all_equal WHERE a <> b")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert!(rows.is_empty());
}

#[rstest]
fn filter_not_equal_all_pass(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .plan("SELECT a, b FROM example_table WHERE a <> b")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["a"].as_i64().unwrap());

    assert_eq!(
        rows,
        serde_json::json!([
            {"a": 1, "b": 10},
            {"a": 2, "b": 20},
            {"a": 3, "b": 30},
            {"a": 4, "b": 40},
            {"a": 5, "b": 50},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn filter_not_equal_constant(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .plan("SELECT a FROM example_table WHERE a <> 2")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["a"].as_i64().unwrap());

    assert_eq!(
        rows,
        serde_json::json!([
            {"a": 1},
            {"a": 3},
            {"a": 4},
            {"a": 5},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn filter_equal_constant(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .plan("SELECT a FROM example_table WHERE a = 3")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(
        rows,
        serde_json::json!([{"a": 3}]).as_array().unwrap().clone()
    );
}

#[rstest]
fn filter_equal_no_match(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .plan("SELECT a FROM example_table WHERE a = 999")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert!(rows.is_empty(), "expected no rows, got: {rows:?}");
}

// ---------------------------------------------------------------------------
// OrderBy tests
// ---------------------------------------------------------------------------

#[rstest]
fn order_by_ascending(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .plan("SELECT a, b FROM example_table ORDER BY a ASC")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(
        rows,
        serde_json::json!([
            {"a": 1, "b": 10},
            {"a": 2, "b": 20},
            {"a": 3, "b": 30},
            {"a": 4, "b": 40},
            {"a": 5, "b": 50},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn order_by_descending(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .plan("SELECT a, b FROM example_table ORDER BY a DESC")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(
        rows,
        serde_json::json!([
            {"a": 5, "b": 50},
            {"a": 4, "b": 40},
            {"a": 3, "b": 30},
            {"a": 2, "b": 20},
            {"a": 1, "b": 10},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn top_n_limit_1(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .plan("SELECT a FROM example_table ORDER BY a DESC LIMIT 1")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["a"], 5);
}

#[rstest]
fn top_n_limit_exceeds_row_count(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .plan("SELECT a FROM example_table ORDER BY a LIMIT 100")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 5);
}

#[rstest]
fn top_n_limit_2_ascending(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .plan("SELECT b FROM example_table ORDER BY a ASC LIMIT 2")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(
        rows,
        serde_json::json!([
            {"b": 10},
            {"b": 20},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

/// `k` has one NULL (at id 2) so the null's position is unambiguous.
fn add_nullable_keys(testing_planner: &mut TestingPlanner) {
    testing_planner.add_table(
        "nullable_keys",
        &[
            (
                "k",
                Type::Int32,
                Arc::new(Int32Array::from(vec![Some(3), None, Some(1), Some(2)])) as ArrayRef,
            ),
            ("id", Type::Int32, int_col(vec![1, 2, 3, 4])),
        ],
    );
}

fn ids(rows: &[serde_json::Value]) -> Vec<i64> {
    rows.iter().map(|r| r["id"].as_i64().unwrap()).collect()
}

#[rstest]
fn order_by_nulls_first_puts_the_null_key_before_every_value(mut testing_planner: TestingPlanner) {
    add_nullable_keys(&mut testing_planner);

    let rows = run(
        &mut testing_planner,
        "SELECT id FROM nullable_keys ORDER BY k NULLS FIRST",
    );

    assert_eq!(ids(&rows), vec![2, 3, 4, 1]);
}

#[rstest]
fn order_by_descending_nulls_first_puts_the_null_key_before_the_largest(
    mut testing_planner: TestingPlanner,
) {
    add_nullable_keys(&mut testing_planner);

    let rows = run(
        &mut testing_planner,
        "SELECT id FROM nullable_keys ORDER BY k DESC NULLS FIRST",
    );

    assert_eq!(ids(&rows), vec![2, 1, 4, 3]);
}

#[rstest]
fn order_by_without_a_null_clause_puts_the_null_key_last_in_both_directions(
    mut testing_planner: TestingPlanner,
) {
    add_nullable_keys(&mut testing_planner);

    let ascending = run(
        &mut testing_planner,
        "SELECT id FROM nullable_keys ORDER BY k",
    );
    let descending = run(
        &mut testing_planner,
        "SELECT id FROM nullable_keys ORDER BY k DESC",
    );

    assert_eq!(ids(&ascending), vec![3, 4, 1, 2]);
    assert_eq!(ids(&descending), vec![1, 4, 3, 2]);
}

#[rstest]
fn top_n_nulls_first_keeps_the_null_key(mut testing_planner: TestingPlanner) {
    add_nullable_keys(&mut testing_planner);

    let ascending = run(
        &mut testing_planner,
        "SELECT id FROM nullable_keys ORDER BY k NULLS FIRST LIMIT 2",
    );
    let descending = run(
        &mut testing_planner,
        "SELECT id FROM nullable_keys ORDER BY k DESC NULLS FIRST LIMIT 2",
    );

    assert_eq!(ids(&ascending), vec![2, 3]);
    assert_eq!(ids(&descending), vec![2, 1]);
}

// A bare `LIMIT` (no ORDER BY) keeps `limit` arbitrary rows.
#[rstest]
fn plain_limit_keeps_limit_rows(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .plan("SELECT a FROM example_table LIMIT 2")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 2);
}

// A `LIMIT` larger than the input returns every row.
#[rstest]
fn plain_limit_exceeds_row_count(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .plan("SELECT a FROM example_table LIMIT 100")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 5);
}

// `LIMIT … OFFSET …` with no ORDER BY: DuckDB late-materializes this (the narrow
// scan reads no columns), exercising the metadata-only scan + the row-id ORDER BY
// strip. example_table's `a` is 1..5 in row order; skipping 2 and taking 2 yields
// rows 3 and 4 (scan order, which a no-ORDER-BY query may return).
#[rstest]
fn plain_limit_with_offset(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .plan("SELECT a FROM example_table LIMIT 2 OFFSET 2")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let mut got: Vec<i64> = batches_to_json(&results)
        .iter()
        .map(|r| r["a"].as_i64().unwrap())
        .collect();
    got.sort();
    assert_eq!(got, vec![3, 4]);
}

// A grouped aggregate under a plain `LIMIT` (no ORDER BY): the limit pushdown
// caps each partition, but the surviving `Limit` still bounds the total. 4
// distinct groups, `LIMIT 2` -> exactly 2 rows.
#[rstest]
fn group_by_plain_limit(mut testing_planner: TestingPlanner) {
    add_events_table(&testing_planner);

    let results = testing_planner
        .plan("SELECT g, COUNT(*) FROM events GROUP BY g LIMIT 2")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 2);
}

// A two-key (int + string) GROUP BY ordered by `COUNT(*)` DESC with the group
// keys as tiebreakers, then LIMIT. The count-DESC primary key pushes a top-k
// into the grouped aggregate; the surviving TopN re-sorts under the full
// multi-key order.
#[rstest]
fn grouped_multikey_order_by_count_desc_limit(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "events",
        &[
            ("UserID", Type::Int32, int_col(vec![1, 1, 1, 2, 2, 3])),
            (
                "SearchPhrase",
                Type::Utf8,
                str_col(vec!["a", "a", "b", "a", "a", "c"]),
            ),
        ],
    );

    let results = testing_planner
        .plan(
            "SELECT UserID, SearchPhrase, COUNT(*) AS c FROM events \
             GROUP BY UserID, SearchPhrase ORDER BY c DESC, UserID, SearchPhrase LIMIT 10",
        )
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    let got: Vec<(i64, String, i64)> = rows
        .iter()
        .map(|r| {
            (
                r["UserID"].as_i64().unwrap(),
                r["SearchPhrase"].as_str().unwrap().to_string(),
                r["c"].as_i64().unwrap(),
            )
        })
        .collect();
    assert_eq!(
        got,
        vec![
            (1, "a".to_string(), 2),
            (2, "a".to_string(), 2),
            (1, "b".to_string(), 1),
            (3, "c".to_string(), 1),
        ]
    );
}

// Every group's `COUNT(*)` is identical here (50 keys, each appearing twice), so
// `ORDER BY c DESC, g LIMIT 3` is decided purely by the secondary key: the three
// smallest `g`. Pushing a per-group top-k by the count alone would keep arbitrary
// tied groups and drop the ones the secondary key selects, so a multi-key sort
// must not push the limit into the group-by. This guards that.
#[rstest]
fn grouped_multikey_order_by_breaks_count_ties_on_secondary_key(
    mut testing_planner: TestingPlanner,
) {
    let counts: Vec<i64> = (1..=50i64).flat_map(|g| [g, g]).collect();
    testing_planner.add_table(
        "g_ties",
        &[(
            "g",
            Type::Int64,
            Arc::new(Int64Array::from(counts)) as ArrayRef,
        )],
    );

    let results = testing_planner
        .plan("SELECT g, COUNT(*) AS c FROM g_ties GROUP BY g ORDER BY c DESC, g LIMIT 3")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    let got: Vec<(i64, i64)> = rows
        .iter()
        .map(|r| (r["g"].as_i64().unwrap(), r["c"].as_i64().unwrap()))
        .collect();
    assert_eq!(got, vec![(1, 2), (2, 2), (3, 2)]);
}

// `SELECT *` over a filtered Top-N is exactly the late-materialization shape:
// DuckDB scans only `a`/`name` for the predicate+sort, then materializes the
// full row for the survivors. Exercises multi-column materialize + reordering
// back to schema order, plus that metadata survives the narrow projection.
#[rstest]
fn select_star_filtered_top_n_late_materializes(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .plan("SELECT * FROM example_table WHERE name <> 'bob' ORDER BY a ASC LIMIT 2")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(
        rows,
        serde_json::json!([
            {"a": 1, "b": 10, "c": 100, "name": "alice"},
            {"a": 3, "b": 30, "c": 300, "name": "charlie"},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn grouped_avg_length_filtered_having_ordered(mut testing_planner: TestingPlanner) {
    // AVG(length(url)) per group, an empty-string WHERE filter, a HAVING on
    // COUNT(*), then ORDER BY the avg DESC. length() is byte count, so g=1
    // averages (5+2)/2 = 3.5 and g=2 averages (4+2)/2 = 3.0; the "" rows are
    // filtered before aggregation and g=3 (one row) is dropped by
    // HAVING COUNT(*) > 1.
    testing_planner.add_table(
        "pages",
        &[
            ("g", Type::Int32, int_col(vec![1, 1, 1, 2, 2, 3])),
            (
                "url",
                Type::Utf8,
                str_col(vec!["abcde", "fg", "", "wxyz", "uv", "zzzz"]),
            ),
        ],
    );

    let rows = batches_to_json(
        &testing_planner
            .plan(
                "SELECT g, AVG(length(url)) AS l, COUNT(*) AS c FROM pages \
                 WHERE url <> '' GROUP BY g HAVING COUNT(*) > 1 ORDER BY l DESC LIMIT 25",
            )
            .unwrap()
            .compile(
                testing_planner.dispatcher(),
                testing_planner.transaction().as_ref(),
            )
            .unwrap()
            .collect()
            .unwrap(),
    );

    assert_eq!(
        rows.len(),
        2,
        "g=3 should be dropped by HAVING; got {rows:?}"
    );
    let groups: Vec<i64> = rows.iter().map(|r| r["g"].as_i64().unwrap()).collect();
    let avgs: Vec<f64> = rows.iter().map(|r| r["l"].as_f64().unwrap()).collect();
    assert_eq!(groups, vec![1, 2]);
    assert_eq!(avgs, vec![3.5, 3.0]);
}

#[rstest]
fn group_by_int_column(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .plan("SELECT a, COUNT(*) FROM example_table GROUP BY a")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["a"].as_i64().unwrap());

    // Each value of a (1..=5) appears exactly once.
    assert_eq!(rows.len(), 5);
    for row in &rows {
        assert_eq!(row["count_star()"], 1);
    }
}

#[rstest]
fn group_by_string_column_with_duplicates(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .plan("SELECT name, COUNT(*) FROM example_table GROUP BY name")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["name"].as_str().unwrap().to_string());

    // alice appears twice, bob/charlie/dave each once.
    assert_eq!(rows.len(), 4);
    let alice = rows.iter().find(|r| r["name"] == "alice").unwrap();
    assert_eq!(alice["count_star()"], 2);
    let bob = rows.iter().find(|r| r["name"] == "bob").unwrap();
    assert_eq!(bob["count_star()"], 1);
}

// GROUP BY (Int64, Utf8) takes the dedicated int+string key extractor (the int
// beside the string's arena handle), not the row encoder. Plain LIMIT, no ORDER
// BY, so any 10 groups suffice.
#[rstest]
fn group_by_int64_string_key(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "hits",
        &[
            (
                "id",
                Type::Int64,
                Arc::new(Int64Array::from(vec![1i64, 1, 2, 2, 3])) as ArrayRef,
            ),
            ("name", Type::Utf8, str_col(vec!["a", "a", "b", "c", "c"])),
        ],
    );

    let mut rows = batches_to_json(
        &testing_planner
            .plan("SELECT id, name, COUNT(*) FROM hits GROUP BY id, name LIMIT 10")
            .unwrap()
            .compile(
                testing_planner.dispatcher(),
                testing_planner.transaction().as_ref(),
            )
            .unwrap()
            .collect()
            .unwrap(),
    );
    rows.sort_by_key(|r| {
        (
            r["id"].as_i64().unwrap(),
            r["name"].as_str().unwrap().to_string(),
        )
    });

    // Groups: (1,"a")=2, (2,"b")=1, (2,"c")=1, (3,"c")=1.
    assert_eq!(rows.len(), 4);
    assert_eq!(
        (
            rows[0]["id"].as_i64(),
            rows[0]["name"].as_str(),
            rows[0]["count_star()"].as_i64()
        ),
        (Some(1), Some("a"), Some(2))
    );
    assert_eq!(
        (
            rows[1]["id"].as_i64(),
            rows[1]["name"].as_str(),
            rows[1]["count_star()"].as_i64()
        ),
        (Some(2), Some("b"), Some(1))
    );
    assert_eq!(
        (
            rows[2]["id"].as_i64(),
            rows[2]["name"].as_str(),
            rows[2]["count_star()"].as_i64()
        ),
        (Some(2), Some("c"), Some(1))
    );
    assert_eq!(
        (
            rows[3]["id"].as_i64(),
            rows[3]["name"].as_str(),
            rows[3]["count_star()"].as_i64()
        ),
        (Some(3), Some("c"), Some(1))
    );
}

#[rstest]
fn group_by_int64_and_date_columns(mut testing_planner: TestingPlanner) {
    // A wide key tuple (Int64 id, Date d) routes through the row encoder, which
    // encodes the Date as its integer day count. (100,10) appears twice,
    // (100,20) once, (200,10) twice, so the date must be part of the key for
    // (100,10) and (100,20) to stay distinct.
    testing_planner.add_table(
        "idd",
        &[
            (
                "id",
                Type::Int64,
                Arc::new(Int64Array::from(vec![100i64, 100, 100, 200, 200])) as ArrayRef,
            ),
            (
                "d",
                Type::Date,
                Arc::new(Date32Array::from(vec![10i32, 10, 20, 10, 10])) as ArrayRef,
            ),
        ],
    );

    let results = testing_planner
        .plan("SELECT id, d, COUNT(*) FROM idd GROUP BY id, d")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    // The Date key surfaces as a real date (day 10 -> 1970-01-11, day 20 ->
    // 1970-01-21), even though it travels through the row encoder as an integer.
    rows.sort_by_key(|r| {
        (
            r["id"].as_i64().unwrap(),
            r["d"].as_str().unwrap().to_string(),
        )
    });

    assert_eq!(rows.len(), 3);
    assert_eq!(
        (
            rows[0]["id"].as_i64(),
            rows[0]["d"].as_str(),
            rows[0]["count_star()"].as_i64(),
        ),
        (Some(100), Some("1970-01-11"), Some(2))
    );
    assert_eq!(
        (
            rows[1]["id"].as_i64(),
            rows[1]["d"].as_str(),
            rows[1]["count_star()"].as_i64(),
        ),
        (Some(100), Some("1970-01-21"), Some(1))
    );
    assert_eq!(
        (
            rows[2]["id"].as_i64(),
            rows[2]["d"].as_str(),
            rows[2]["count_star()"].as_i64(),
        ),
        (Some(200), Some("1970-01-11"), Some(2))
    );
}

#[rstest]
fn group_by_count_distinct(mut testing_planner: TestingPlanner) {
    // g: 1,1,1,2,2,3   x: 10,10,20,30,30,40
    // distinct x per g: g=1 -> {10,20}=2, g=2 -> {30}=1, g=3 -> {40}=1
    testing_planner.add_table(
        "gx",
        &[
            ("g", Type::Int32, int_col(vec![1, 1, 1, 2, 2, 3])),
            ("x", Type::Int32, int_col(vec![10, 10, 20, 30, 30, 40])),
        ],
    );
    let results = testing_planner
        .plan("SELECT g, COUNT(DISTINCT x) FROM gx GROUP BY g")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["g"].as_i64().unwrap());
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0]["g"], 1);
    assert_eq!(rows[0]["count(DISTINCT x)"], 2);
    assert_eq!(rows[1]["g"], 2);
    assert_eq!(rows[1]["count(DISTINCT x)"], 1);
    assert_eq!(rows[2]["g"], 3);
    assert_eq!(rows[2]["count(DISTINCT x)"], 1);
}

#[rstest]
fn group_by_count_distinct_string_key(mut testing_planner: TestingPlanner) {
    // name: a,a,a,b,b   x: 1,1,2,5,5
    // distinct x per name: a -> {1,2}=2, b -> {5}=1
    testing_planner.add_table(
        "nx",
        &[
            ("name", Type::Utf8, str_col(vec!["a", "a", "a", "b", "b"])),
            ("x", Type::Int32, int_col(vec![1, 1, 2, 5, 5])),
        ],
    );

    let results = testing_planner
        .plan("SELECT name, COUNT(DISTINCT x) FROM nx GROUP BY name")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    // A single string group key counts through the dedicated `StringKeyExtractor`
    // (the same extractor a plain `GROUP BY name` uses), whose key column is
    // emitted as `name`.
    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["name"].as_str().unwrap().to_string());
    assert_eq!(rows.len(), 2);
    assert_eq!(
        (
            rows[0]["name"].as_str(),
            rows[0]["count(DISTINCT x)"].as_i64()
        ),
        (Some("a"), Some(2))
    );
    assert_eq!(
        (
            rows[1]["name"].as_str(),
            rows[1]["count(DISTINCT x)"].as_i64()
        ),
        (Some("b"), Some(1))
    );
}

#[rstest]
fn group_by_count_distinct_multi_column_key(mut testing_planner: TestingPlanner) {
    // (g, name): (1,x),(1,x),(2,y)   uid: 10,20,30
    // distinct uid per (g, name): (1,x) -> {10,20}=2, (2,y) -> {30}=1
    testing_planner.add_table(
        "gnx",
        &[
            ("g", Type::Int32, int_col(vec![1, 1, 2])),
            ("name", Type::Utf8, str_col(vec!["x", "x", "y"])),
            ("uid", Type::Int32, int_col(vec![10, 20, 30])),
        ],
    );

    let results = testing_planner
        .plan("SELECT g, name, COUNT(DISTINCT uid) FROM gnx GROUP BY g, name")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| {
        (
            r["g"].as_i64().unwrap(),
            r["name"].as_str().unwrap().to_string(),
        )
    });
    assert_eq!(rows.len(), 2);
    assert_eq!(
        (
            rows[0]["g"].as_i64(),
            rows[0]["name"].as_str(),
            rows[0]["count(DISTINCT uid)"].as_i64()
        ),
        (Some(1), Some("x"), Some(2))
    );
    assert_eq!(
        (
            rows[1]["g"].as_i64(),
            rows[1]["name"].as_str(),
            rows[1]["count(DISTINCT uid)"].as_i64()
        ),
        (Some(2), Some("y"), Some(1))
    );
}

#[rstest]
fn group_by_count_distinct_computed_key(mut testing_planner: TestingPlanner) {
    // GROUP BY a computed key (k * 2) forces the key to be materialised into a
    // leading column before the two-level dedup.
    // k: 1,1,2 -> k*2: 2,2,4 ; x: 10,20,30
    // distinct x per key: 2 -> {10,20}=2, 4 -> {30}=1
    testing_planner.add_table(
        "cg",
        &[
            ("k", Type::Int32, int_col(vec![1, 1, 2])),
            ("x", Type::Int32, int_col(vec![10, 20, 30])),
        ],
    );

    let results = testing_planner
        .plan("SELECT k * 2 AS p, COUNT(DISTINCT x) FROM cg GROUP BY k * 2")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["count(DISTINCT x)"].as_i64().unwrap());
    let counts: Vec<i64> = rows
        .iter()
        .map(|r| r["count(DISTINCT x)"].as_i64().unwrap())
        .collect();
    assert_eq!(rows.len(), 2);
    assert_eq!(counts, vec![1, 2]);
}

#[rstest]
fn global_count_distinct_string(mut testing_planner: TestingPlanner) {
    // example_table.name = alice, bob, charlie, dave, alice -> 4 distinct.
    let results = testing_planner
        .plan("SELECT COUNT(DISTINCT name) FROM example_table")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 1);
    let only_value = rows[0].as_object().unwrap().values().next().unwrap();
    assert_eq!(only_value, 4);
}

#[rstest]
fn global_count_distinct_int(mut testing_planner: TestingPlanner) {
    // distinct {7, 8, 9, 0} = 4 (includes 0, exercising the HashOnly extractor's
    // mix64(0)==0 / empty-sentinel edge through the keys-only count path).
    testing_planner.add_table(
        "ints",
        &[("v", Type::Int32, int_col(vec![7, 7, 7, 8, 9, 9, 0, 0]))],
    );
    let results = testing_planner
        .plan("SELECT COUNT(DISTINCT v) FROM ints")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 1);
    let only_value = rows[0].as_object().unwrap().values().next().unwrap();
    assert_eq!(only_value, 4);
}

#[rstest]
fn group_by_mixed_distinct(mut testing_planner: TestingPlanner) {
    // g: 1,1,2  x: 10,10,20  v: 5,5,7
    // per g: SUM(v), COUNT(*), COUNT(DISTINCT x)
    //   g=1 -> sum=10, count=2, distinct=1 ; g=2 -> sum=7, count=1, distinct=1
    testing_planner.add_table(
        "mixed",
        &[
            ("g", Type::Int32, int_col(vec![1, 1, 2])),
            ("x", Type::Int32, int_col(vec![10, 10, 20])),
            ("v", Type::Int32, int_col(vec![5, 5, 7])),
        ],
    );
    let results = testing_planner
        .plan("SELECT g, SUM(v), COUNT(*), COUNT(DISTINCT x) FROM mixed GROUP BY g")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["g"].as_i64().unwrap());
    assert_eq!(rows.len(), 2);
    // Assert the multiset of the three aggregate values so the test is robust to
    // their column order (all three values differ).
    let vals = |r: &serde_json::Value| -> Vec<i64> {
        ["sum(v)", "count_star()", "count(DISTINCT x)"]
            .iter()
            .map(|c| r[*c].as_i64().unwrap())
            .collect()
    };
    assert_eq!(rows[0]["g"], 1);
    let v1 = vals(&rows[0]);
    assert!(
        v1.contains(&10) && v1.contains(&2) && v1.contains(&1),
        "g=1 {v1:?}"
    );
    assert_eq!(rows[1]["g"], 2);
    let v2 = vals(&rows[1]);
    assert!(v2.contains(&7) && v2.contains(&1), "g=2 {v2:?}");
}

#[rstest]
fn group_by_two_counts_mixed_distinct(mut testing_planner: TestingPlanner) {
    // COUNT(*) and COUNT(w) compute the same per-row count, so they share one
    // inner partial and the outer re-folds each.
    // g: 1,1,2,2  x: 10,10,20,30  w (no nulls): 4,6,8,9
    //   g=1 -> count=2, count=2, distinct({10})=1
    //   g=2 -> count=2, count=2, distinct({20,30})=2
    testing_planner.add_table(
        "cc",
        &[
            ("g", Type::Int32, int_col(vec![1, 1, 2, 2])),
            ("x", Type::Int32, int_col(vec![10, 10, 20, 30])),
            ("w", Type::Int32, int_col(vec![4, 6, 8, 9])),
        ],
    );

    let results = testing_planner
        .plan("SELECT g, COUNT(*), COUNT(w), COUNT(DISTINCT x) FROM cc GROUP BY g")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["g"].as_i64().unwrap());
    assert_eq!(
        (
            rows[0]["count_star()"].as_i64(),
            rows[0]["count(w)"].as_i64(),
            rows[0]["count(DISTINCT x)"].as_i64()
        ),
        (Some(2), Some(2), Some(1))
    );
    assert_eq!(
        (
            rows[1]["count_star()"].as_i64(),
            rows[1]["count(w)"].as_i64(),
            rows[1]["count(DISTINCT x)"].as_i64()
        ),
        (Some(2), Some(2), Some(2))
    );
}

#[rstest]
fn group_by_count_two_int16_sums_coalesce(mut testing_planner: TestingPlanner) {
    // COUNT(*) and COUNT(c) fold to the same count, so the entry keeps three slots
    // [Count, SUM(i16), SUM(i16)] and the output re-expands the shared count into
    // both the v0 and v3 positions.
    // g: 1,1,2  a: 10,20,30  b: 1,2,3  c (no nulls): 7,8,9
    testing_planner.add_table(
        "cs",
        &[
            ("g", Type::Int32, int_col(vec![1, 1, 2])),
            ("a", Type::Int16, int16_col(vec![10, 20, 30])),
            ("b", Type::Int16, int16_col(vec![1, 2, 3])),
            ("c", Type::Int16, int16_col(vec![7, 8, 9])),
        ],
    );

    let results = testing_planner
        .plan("SELECT g, COUNT(*), SUM(a), SUM(b), COUNT(c) FROM cs GROUP BY g")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["g"].as_i64().unwrap());
    let row = |r: &serde_json::Value| {
        ["count_star()", "sum(a)", "sum(b)", "count(c)"]
            .iter()
            .map(|c| r[*c].as_i64().unwrap())
            .collect::<Vec<_>>()
    };
    assert_eq!(row(&rows[0]), vec![2, 30, 3, 2]);
    assert_eq!(row(&rows[1]), vec![1, 30, 3, 1]);
}

#[rstest]
fn group_order_by_sum_with_coalesced_count_limit(mut testing_planner: TestingPlanner) {
    // COUNT(c) coalesces into COUNT(*), narrowing the entry, while ORDER BY SUM(a)
    // LIMIT sorts/limits on a later aggregate: the coalesced output must still order
    // correctly (and any pushed-down Top-K slot remaps through the dedup).
    // g:1,1,2,3  a:10,20,5,7  -> sums g1=30, g3=7, g2=5 ; DESC LIMIT 2 keeps g1, g3
    testing_planner.add_table(
        "tk",
        &[
            ("g", Type::Int32, int_col(vec![1, 1, 2, 3])),
            ("a", Type::Int32, int_col(vec![10, 20, 5, 7])),
            ("c", Type::Int32, int_col(vec![1, 1, 1, 1])),
        ],
    );

    let results = testing_planner
        .plan(
            "SELECT g, COUNT(*), COUNT(c), SUM(a) AS s FROM tk GROUP BY g ORDER BY s DESC LIMIT 2",
        )
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 2);
    assert_eq!(
        (rows[0]["g"].as_i64(), rows[0]["s"].as_i64()),
        (Some(1), Some(30))
    );
    assert_eq!(
        (rows[1]["g"].as_i64(), rows[1]["s"].as_i64()),
        (Some(3), Some(7))
    );
}

// ---------------------------------------------------------------------------
// Combined operator tests
// ---------------------------------------------------------------------------

#[rstest]
fn filter_then_order_by(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .plan("SELECT a, b FROM example_table WHERE a <> b ORDER BY a DESC")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(
        rows,
        serde_json::json!([
            {"a": 5, "b": 50},
            {"a": 4, "b": 40},
            {"a": 3, "b": 30},
            {"a": 2, "b": 20},
            {"a": 1, "b": 10},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn filter_then_count(mut testing_planner: TestingPlanner) {
    // Every row has a == b, so WHERE a <> b yields 0 rows.
    testing_planner.add_table(
        "pairs_all_equal",
        &[
            ("a", Type::Int32, int_col(vec![1, 2, 3, 4, 5])),
            ("b", Type::Int32, int_col(vec![1, 2, 3, 4, 5])),
        ],
    );

    let results = testing_planner
        .plan("SELECT COUNT(*) FROM pairs_all_equal WHERE a <> b")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["count_star()"], 0);
}

#[rstest]
fn global_avg_is_lowered_to_sum_and_count(mut testing_planner: TestingPlanner) {
    // AVG never reaches the global aggregate operator as a dedicated kind:
    // DuckDB lowers it to sum/count plus a divide projection. After the
    // `AggKind::Avg` removal this must still compile (no
    // `UnsupportedAggregateExpression`) and produce the right average through
    // the sum + count slots. avg(a) over [1,2,3,4,5] = 3.0.
    let results = testing_planner
        .plan("SELECT AVG(a) FROM example_table")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 1);
    let avg = rows[0].as_object().unwrap().values().next().unwrap();
    assert_eq!(avg.as_f64().unwrap(), 3.0);
}

#[rstest]
fn global_sum_count_avg_together(mut testing_planner: TestingPlanner) {
    // The multi-aggregate global path with a SUM, a COUNT(*) and an AVG mixed.
    // Must compile and run end-to-end; count = 5 and avg(b) over
    // [10,20,30,40,50] = 30.0 (sum(a) is a Decimal128, skipped by the f64 scan).
    let results = testing_planner
        .plan("SELECT SUM(a), COUNT(*), AVG(b) FROM example_table")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 1);
    let vals: Vec<f64> = rows[0]
        .as_object()
        .unwrap()
        .values()
        .filter_map(|v| v.as_f64())
        .collect();
    assert!(vals.contains(&5.0), "expected count 5 in {vals:?}");
    assert!(vals.contains(&30.0), "expected avg(b) 30.0 in {vals:?}");
}

#[rstest]
fn global_min_max(mut testing_planner: TestingPlanner) {
    // MIN/MAX fold the extreme over the whole column (no GROUP BY): over
    // a=[1,2,3,4,5] and b=[10,20,30,40,50], min(a)=1 and max(b)=50.
    let results = testing_planner
        .plan("SELECT MIN(a), MAX(b) FROM example_table")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["min(a)"], 1);
    assert_eq!(rows[0]["max(b)"], 50);
}

#[rstest]
fn global_string_min_max(mut testing_planner: TestingPlanner) {
    // Global string MIN/MAX (no GROUP BY) fold the byte-lexicographic extreme over
    // the whole column, emitting one Utf8View row each.
    testing_planner.add_table(
        "gs_global",
        &[(
            "s",
            Type::Utf8,
            str_col(vec!["banana", "apple", "date", "cherry"]),
        )],
    );
    let results = testing_planner
        .plan("SELECT MIN(s), MAX(s) FROM gs_global")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["min(s)"], "apple");
    assert_eq!(rows[0]["max(s)"], "date");
}

#[rstest]
fn grouped_min_max(mut testing_planner: TestingPlanner) {
    // Per-group extremes through the dynamic value extractor's MIN/MAX slots.
    // g=1 -> v {10,40} (min 10, max 40); g=2 -> v {20,5} (min 5, max 20). The
    // consume fold and the partition merge are both kind-aware.
    testing_planner.add_table(
        "gv",
        &[
            ("g", Type::Int32, int_col(vec![1, 1, 2, 2])),
            ("v", Type::Int32, int_col(vec![10, 40, 20, 5])),
        ],
    );
    let results = testing_planner
        .plan("SELECT g, MIN(v), MAX(v) FROM gv GROUP BY g")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    // A single integer key uses the dedicated `IntKeyExtractor`, whose key column
    // carries the source column name `g` (multi-key groups use the row encoder's
    // `k0`, `k1`, …).
    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["g"].as_i64().unwrap());
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["g"], 1);
    assert_eq!(rows[0]["min(v)"], 10); // min
    assert_eq!(rows[0]["max(v)"], 40); // max
    assert_eq!(rows[1]["g"], 2);
    assert_eq!(rows[1]["min(v)"], 5); // min
    assert_eq!(rows[1]["max(v)"], 20); // max
}

#[rstest]
fn grouped_string_min_and_max(mut testing_planner: TestingPlanner) {
    // Per-group string extremes via StringExtreme (lazy arena persist + zero-copy
    // StringView output). Homogeneous direction only, so MIN and MAX are separate
    // queries. g=1 -> {"banana","apple"}; g=2 -> {"cherry","date"}.
    testing_planner.add_table(
        "gs",
        &[
            ("g", Type::Int32, int_col(vec![1, 1, 2, 2])),
            (
                "s",
                Type::Utf8,
                str_col(vec!["banana", "apple", "cherry", "date"]),
            ),
        ],
    );

    let run = |p: &mut TestingPlanner, sql: &str| {
        let results = p
            .plan(sql)
            .unwrap()
            .compile(p.dispatcher(), p.transaction().as_ref())
            .unwrap()
            .collect()
            .unwrap();
        let mut rows = batches_to_json(&results);
        rows.sort_by_key(|r| r["g"].as_i64().unwrap());
        rows
    };

    let mins = run(&mut testing_planner, "SELECT g, MIN(s) FROM gs GROUP BY g");
    assert_eq!(mins[0]["min(s)"], "apple");
    assert_eq!(mins[1]["min(s)"], "cherry");

    let maxes = run(&mut testing_planner, "SELECT g, MAX(s) FROM gs GROUP BY g");
    assert_eq!(maxes[0]["max(s)"], "banana");
    assert_eq!(maxes[1]["max(s)"], "date");
}

#[rstest]
fn grouped_string_max_order_by_limit(mut testing_planner: TestingPlanner) {
    // ORDER BY MAX(s) DESC LIMIT must rank by lexicographic order. A string
    // extreme's sort_key is its raw ArenaKey bits, so the group top-k pushdown
    // must NOT fire on it (plan.rs guards it); the full TopN sorts instead.
    // Per-group MAX: g1=avocado, g2=zebra, g3=melon -> DESC LIMIT 1 -> zebra.
    testing_planner.add_table(
        "gsl",
        &[
            ("g", Type::Int32, int_col(vec![1, 1, 2, 2, 3, 3])),
            (
                "s",
                Type::Utf8,
                str_col(vec!["apple", "avocado", "zebra", "yak", "mango", "melon"]),
            ),
        ],
    );

    let results = testing_planner
        .plan("SELECT g, MAX(s) FROM gsl GROUP BY g ORDER BY MAX(s) DESC LIMIT 1")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["max(s)"], "zebra");
}

/// A string extreme mixed with the opposite direction (or with integer
/// aggregates) routes to the runtime `Dynamic` value (widened to `i128` so the
/// `ArenaKey` cell fits) rather than the homogeneous `Compiled` tuple. Both the
/// string MIN+MAX mix and a string-extreme-beside-an-integer-extreme mix must
/// compile and return the right per-group values.
#[rstest]
fn mixed_string_extreme_via_dynamic(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "gs2",
        &[
            ("g", Type::Int32, int_col(vec![1, 1, 2, 2])),
            (
                "s",
                Type::Utf8,
                str_col(vec!["banana", "apple", "cherry", "date"]),
            ),
            ("v", Type::Int32, int_col(vec![10, 30, 5, 20])),
        ],
    );

    // MIN(s) + MAX(s): a string mix of opposite directions in one value.
    let results = testing_planner
        .plan("SELECT g, MIN(s), MAX(s) FROM gs2 GROUP BY g")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();
    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["g"].as_i64().unwrap());
    assert_eq!(rows[0]["min(s)"], "apple"); // g=1 min
    assert_eq!(rows[0]["max(s)"], "banana"); // g=1 max
    assert_eq!(rows[1]["min(s)"], "cherry"); // g=2 min
    assert_eq!(rows[1]["max(s)"], "date"); // g=2 max

    // MIN(s) (string) + MAX(v) (integer): a string extreme beside a numeric one.
    let mixed = testing_planner
        .plan("SELECT g, MIN(s), MAX(v) FROM gs2 GROUP BY g")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();
    let mut mixed_rows = batches_to_json(&mixed);
    mixed_rows.sort_by_key(|r| r["g"].as_i64().unwrap());
    assert_eq!(mixed_rows[0]["min(s)"], "apple"); // g=1 min(s)
    assert_eq!(mixed_rows[0]["max(v)"], 30); // g=1 max(v)
    assert_eq!(mixed_rows[1]["min(s)"], "cherry"); // g=2 min(s)
    assert_eq!(mixed_rows[1]["max(v)"], 20); // g=2 max(v)
}

#[rstest]
fn filter_then_top_n(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .plan("SELECT a FROM example_table WHERE a <> b ORDER BY a DESC LIMIT 2")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["a"], 5);
    assert_eq!(rows[1]["a"], 4);
}

/// Build a single-column `events` table whose group sizes are all distinct, so
/// `COUNT(*)` orderings are unambiguous: g=1 -> 4, g=2 -> 3, g=3 -> 2, g=4 -> 1.
fn add_events_table(testing_planner: &TestingPlanner) {
    testing_planner.add_table(
        "events",
        &[(
            "g",
            Type::Int32,
            int_col(vec![1, 1, 1, 1, 2, 2, 2, 3, 3, 4]),
        )],
    );
}

// `GROUP BY … ORDER BY COUNT(*) DESC LIMIT k OFFSET m`: the `OFFSET` window must
// survive the group → Top-N pipeline. DESC by count is g=1,2,3,4; skipping the
// top 2 and taking 2 leaves g=3 then g=4.
#[rstest]
fn group_order_by_count_desc_with_offset(mut testing_planner: TestingPlanner) {
    add_events_table(&testing_planner);
    let results = testing_planner
        .plan(
            "SELECT g, COUNT(*) FROM events \
             GROUP BY g ORDER BY COUNT(*) DESC LIMIT 2 OFFSET 2",
        )
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    let keys: Vec<i64> = rows.iter().map(|r| r["g"].as_i64().unwrap()).collect();
    assert_eq!(keys, vec![3, 4]);
}

// `GROUP BY … ORDER BY <group key> ASC LIMIT k OFFSET m`: ordering by the group
// key (not an aggregate) with an `OFFSET`. Keys ascending are 1,2,3,4; skipping
// 1 and taking 2 leaves g=2 then g=3.
#[rstest]
fn group_order_by_key_asc_with_offset(mut testing_planner: TestingPlanner) {
    add_events_table(&testing_planner);
    let results = testing_planner
        .plan(
            "SELECT g, COUNT(*) FROM events \
             GROUP BY g ORDER BY g ASC LIMIT 2 OFFSET 1",
        )
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    let keys: Vec<i64> = rows.iter().map(|r| r["g"].as_i64().unwrap()).collect();
    assert_eq!(keys, vec![2, 3]);
}

#[derive(Debug, Default)]
struct RecordingCatalog {
    created_tables: Arc<Mutex<Vec<CreateTableRequest>>>,
    created_users: Arc<Mutex<Vec<CreateUserRequest>>>,
}

/// The CREATE TABLE statements planned here reference no tables, so the
/// transaction resolves nothing; it only records the `create_table` it is asked
/// to compile (which now runs through the transaction, not the catalog).
#[derive(Debug)]
struct RecordingTransaction {
    created_tables: Arc<Mutex<Vec<CreateTableRequest>>>,
    created_users: Arc<Mutex<Vec<CreateUserRequest>>>,
}

impl planner::catalog::CatalogTransaction for RecordingTransaction {
    fn does_schema_exist(&self, _datastore: &str, schema: &str) -> planner::catalog::Result<bool> {
        Ok(schema == planner::DEFAULT_SCHEMA_NAME)
    }

    fn bind_table(
        &self,
        _reference: &planner::catalog::TableReference,
    ) -> planner::catalog::Result<Option<Box<dyn BoundTable>>> {
        Ok(None)
    }

    fn table_revision(
        &self,
        _reference: &planner::catalog::TableReference,
    ) -> planner::catalog::Result<Option<planner::catalog::TableRevision>> {
        Ok(None)
    }

    fn bind_create_table(
        &self,
        request: CreateTableRequest,
    ) -> planner::catalog::Result<Box<dyn planner::catalog::TableCreation>> {
        self.created_tables.lock().unwrap().push(request);
        Ok(Box::new(NoRowsCreation))
    }

    fn bind_create_user(
        &self,
        request: CreateUserRequest,
    ) -> planner::catalog::Result<Box<dyn planner::catalog::UserCreation>> {
        self.created_users.lock().unwrap().push(request);
        Ok(Box::new(NoRowsCreation))
    }
}

/// The resolved create a [`RecordingTransaction`] returns: it compiles to a
/// no-rows dataflow (CREATE TABLE yields no rows; a real datastore would commit
/// the table when this compiles).
struct NoRowsCreation;

impl planner::catalog::TableCreation for NoRowsCreation {
    fn compile(
        &self,
        dispatcher: &dispatch::DataFlowDispatcher,
    ) -> planner::catalog::Result<dispatch::RecordBatchOperatorSpec> {
        Ok(dispatch::RecordBatchOperatorSpec::from_nullary(
            dispatcher,
            (0..dispatcher.worker_count()).map(|_| NoRowsNullary::default()),
        ))
    }
}

/// A CREATE USER's dataflow is the same no-rows shape as a CREATE TABLE's.
impl planner::catalog::UserCreation for NoRowsCreation {
    fn compile(
        &self,
        dispatcher: &dispatch::DataFlowDispatcher,
    ) -> planner::catalog::Result<dispatch::RecordBatchOperatorSpec> {
        Ok(dispatch::RecordBatchOperatorSpec::from_nullary(
            dispatcher,
            (0..dispatcher.worker_count()).map(|_| NoRowsNullary::default()),
        ))
    }
}

impl RecordingCatalog {
    fn begin_transaction(&self) -> Arc<dyn planner::catalog::CatalogTransaction> {
        Arc::new(RecordingTransaction {
            created_tables: self.created_tables.clone(),
            created_users: self.created_users.clone(),
        })
    }
}

/// A nullary that emits nothing and finishes immediately — backs the empty
/// result of this test's `CREATE TABLE`.
#[derive(Default)]
struct NoRowsNullary {
    ran: bool,
}

impl dispatch::NullaryFactory<RecordBatch> for NoRowsNullary {
    type Nullary = NoRowsNullary;

    fn build_nullary(self) -> NoRowsNullary {
        self
    }
}

impl dispatch::Nullary<RecordBatch> for NoRowsNullary {
    fn run(
        &mut self,
        _sender: &mut dyn dispatch::Sender<RecordBatch>,
        _io: &mut dispatch::OperatorIO,
    ) -> dispatch::NullaryResult<dispatch::WorkStatus> {
        if self.ran {
            return Ok(dispatch::WorkStatus::Pending);
        }
        self.ran = true;
        Ok(dispatch::WorkStatus::Ran)
    }

    fn finish(
        &mut self,
        _sender: &mut dyn dispatch::Sender<RecordBatch>,
    ) -> dispatch::NullaryResult<bool> {
        Ok(self.ran)
    }
}

// CREATE TABLE tests use a custom recording catalog (which the shared
// `TestCatalog` can't impersonate without adding state we don't otherwise
// need), so they construct their own `Planner` rather than going through the
// `testing_planner` fixture.
#[test]
fn create_table_calls_catalog_once() {
    let dispatch = Dispatch::spin_up(1, 32, None);
    let catalog = Arc::new(RecordingCatalog::default());
    let mut planner = Planner::from_datastore_names(
        vec![DEFAULT_DATASTORE_NAME.to_string()],
        DEFAULT_DATASTORE_NAME.to_string(),
    )
    .expect("planner context");
    let transaction = catalog.begin_transaction();

    let results = planner
        .plan(
            "CREATE TABLE created_table (id INTEGER, name VARCHAR)",
            transaction.clone(),
        )
        .unwrap()
        .compile(dispatch.dispatcher(), transaction.as_ref())
        .unwrap()
        .collect()
        .unwrap();

    assert!(results.is_empty());

    let created = catalog.created_tables.lock().unwrap();
    assert_eq!(created.len(), 1);
    assert_eq!(created[0].name, "created_table");
    assert_eq!(created[0].columns.len(), 2);
    assert_eq!(created[0].columns[0].name, "id");
    assert_eq!(created[0].columns[0].col_type, Type::Int32);
    assert_eq!(created[0].columns[1].name, "name");
    assert_eq!(created[0].columns[1].col_type, Type::Utf8);
    assert!(created[0].options.is_empty());
    assert!(!created[0].if_not_exists);
}

#[test]
fn create_user_binds_through_the_transaction() {
    let dispatch = Dispatch::spin_up(1, 32, None);
    let catalog = Arc::new(RecordingCatalog::default());
    let mut planner = Planner::from_datastore_names(
        vec![DEFAULT_DATASTORE_NAME.to_string()],
        DEFAULT_DATASTORE_NAME.to_string(),
    )
    .expect("planner context");
    let transaction = catalog.begin_transaction();

    let results = planner
        .plan("CREATE USER walt PASSWORD 'blue-1'", transaction.clone())
        .unwrap()
        .compile(dispatch.dispatcher(), transaction.as_ref())
        .unwrap()
        .collect()
        .unwrap();

    assert!(results.is_empty());
    let created = catalog.created_users.lock().unwrap();
    assert_eq!(created.len(), 1);
    assert_eq!(created[0].name, "walt");
    assert_eq!(created[0].password.as_deref(), Some("blue-1"));
}

#[test]
fn create_table_passes_with_options_to_catalog() {
    let dispatch = Dispatch::spin_up(1, 32, None);
    let catalog = Arc::new(RecordingCatalog::default());
    let mut planner = Planner::from_datastore_names(
        vec![DEFAULT_DATASTORE_NAME.to_string()],
        DEFAULT_DATASTORE_NAME.to_string(),
    )
    .expect("planner context");
    let transaction = catalog.begin_transaction();

    let results = planner
        .plan(
            "CREATE TABLE created_table (id INTEGER) WITH (with_pre_existing_parquets='/asdf')",
            transaction.clone(),
        )
        .unwrap()
        .compile(dispatch.dispatcher(), transaction.as_ref())
        .unwrap()
        .collect()
        .unwrap();

    assert!(results.is_empty());

    let created = catalog.created_tables.lock().unwrap();
    assert_eq!(created.len(), 1);
    assert_eq!(created[0].name, "created_table");
    assert_eq!(created[0].columns.len(), 1);
    assert_eq!(created[0].columns[0].name, "id");
    assert_eq!(created[0].columns[0].col_type, Type::Int32);
    assert_eq!(
        created[0]
            .options
            .get("with_pre_existing_parquets")
            .map(String::as_str),
        Some("/asdf")
    );
}

// A grouped aggregate mixing string MIN(s), COUNT(*) and COUNT(DISTINCT) over a
// string group key, behind LIKE/<> filters, ordered by the count. Exercises the
// two-level COUNT(DISTINCT) lowering with a string group key (row-encoded inner
// key) and string-extreme partials.
#[rstest]
fn group_by_string_min_mixed_distinct_filtered_ordered(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "hits",
        &[
            (
                "phrase",
                Type::Utf8,
                str_col(vec!["x", "x", "y", "y", "", "z"]),
            ),
            (
                "url",
                Type::Utf8,
                str_col(vec!["a", "b", "c.google.d", "d", "e", "f"]),
            ),
            (
                "title",
                Type::Utf8,
                str_col(vec![
                    "Goog news",
                    "Goog maps",
                    "Goog x",
                    "miss",
                    "Goog z",
                    "Goog w",
                ]),
            ),
            (
                "user",
                Type::Int64,
                Arc::new(Int64Array::from(vec![1i64, 2, 3, 4, 5, 6])) as ArrayRef,
            ),
        ],
    );

    let rows = batches_to_json(
        &testing_planner
            .plan(
                "SELECT phrase, MIN(url), MIN(title), COUNT(*) AS c, COUNT(DISTINCT user) \
                 FROM hits WHERE title LIKE '%Goog%' AND url NOT LIKE '%.google.%' \
                 AND phrase <> '' GROUP BY phrase ORDER BY c DESC LIMIT 10",
            )
            .unwrap()
            .compile(
                testing_planner.dispatcher(),
                testing_planner.transaction().as_ref(),
            )
            .unwrap()
            .collect()
            .unwrap(),
    );

    // Surviving rows: (x,a,Goog news,u1), (x,b,Goog maps,u2), (z,f,Goog w,u6).
    // y rows drop (one URL has ".google.", one title misses "Goog"); "" drops.
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["phrase"], "x");
    assert_eq!(rows[0]["min(url)"], "a");
    assert_eq!(rows[0]["min(title)"], "Goog maps");
    assert_eq!(rows[0]["c"], 2);
    assert_eq!(rows[0][r#"count(DISTINCT "user")"#], 2);
    assert_eq!(rows[1]["phrase"], "z");
    assert_eq!(
        (
            rows[1]["min(url)"].as_str(),
            rows[1]["c"].as_i64(),
            rows[1][r#"count(DISTINCT "user")"#].as_i64()
        ),
        (Some("f"), Some(1), Some(1))
    );
}

// A numeric MIN/MAX alongside COUNT(DISTINCT): the two-level lowering re-folds the
// integer extreme of each subgroup, on the narrow (non-string) cell.
#[rstest]
fn group_by_numeric_min_max_mixed_distinct(mut testing_planner: TestingPlanner) {
    // g: 1,1,1,2,2   v: 30,10,20,7,5   user: 1,1,2,9,9
    // g=1 -> MIN 10, MAX 30, distinct users {1,2}=2; g=2 -> MIN 5, MAX 7, {9}=1.
    testing_planner.add_table(
        "gv",
        &[
            ("g", Type::Int32, int_col(vec![1, 1, 1, 2, 2])),
            ("v", Type::Int32, int_col(vec![30, 10, 20, 7, 5])),
            ("user", Type::Int32, int_col(vec![1, 1, 2, 9, 9])),
        ],
    );

    let mut rows = batches_to_json(
        &testing_planner
            .plan("SELECT g, MIN(v), MAX(v), COUNT(DISTINCT user) FROM gv GROUP BY g")
            .unwrap()
            .compile(
                testing_planner.dispatcher(),
                testing_planner.transaction().as_ref(),
            )
            .unwrap()
            .collect()
            .unwrap(),
    );
    rows.sort_by_key(|r| r["g"].as_i64().unwrap());

    assert_eq!(rows.len(), 2);
    assert_eq!(
        (
            rows[0]["min(v)"].as_i64(),
            rows[0]["max(v)"].as_i64(),
            rows[0][r#"count(DISTINCT "user")"#].as_i64()
        ),
        (Some(10), Some(30), Some(2))
    );
    assert_eq!(
        (
            rows[1]["min(v)"].as_i64(),
            rows[1]["max(v)"].as_i64(),
            rows[1][r#"count(DISTINCT "user")"#].as_i64()
        ),
        (Some(5), Some(7), Some(1))
    );
}

#[rstest]
fn generate_series_emits_inclusive_range(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .plan("SELECT * FROM generate_series(1, 5)")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["generate_series"].as_i64().unwrap());

    assert_eq!(
        rows,
        serde_json::json!([
            {"generate_series": 1},
            {"generate_series": 2},
            {"generate_series": 3},
            {"generate_series": 4},
            {"generate_series": 5},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn range_excludes_upper_bound_and_honors_step(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .plan("SELECT * FROM range(0, 10, 2)")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["range"].as_i64().unwrap());

    assert_eq!(
        rows,
        serde_json::json!([
            {"range": 0},
            {"range": 2},
            {"range": 4},
            {"range": 6},
            {"range": 8},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn generate_series_streams_across_chunk_boundaries(mut testing_planner: TestingPlanner) {
    // 20000 rows span several SERIES_CHUNK_ROWS (8192) batches, so this exercises
    // the multi-poll streaming path, not a single in-memory batch.
    let results = testing_planner

        .plan("SELECT count(*), min(generate_series), max(generate_series) FROM generate_series(1, 20000)")
        .unwrap()
        .compile(testing_planner.dispatcher(), testing_planner.transaction().as_ref())
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);

    assert_eq!(
        rows,
        vec![
            serde_json::json!({"count_star()": 20000, "min(generate_series)": 1, "max(generate_series)": 20000})
        ]
    );
}

#[rstest]
fn count_star_over_generate_series(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .plan("SELECT count(*) FROM generate_series(1, 5)")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);

    assert_eq!(rows, vec![serde_json::json!({"count_star()": 5})]);
}

#[rstest]
fn range_with_single_argument_starts_at_zero(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .plan("SELECT * FROM range(5)")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["range"].as_i64().unwrap());

    assert_eq!(
        rows,
        serde_json::json!([
            {"range": 0},
            {"range": 1},
            {"range": 2},
            {"range": 3},
            {"range": 4},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn timestamp_generate_series_is_not_registered(mut testing_planner: TestingPlanner) {
    let result = testing_planner.plan(
        "SELECT * FROM generate_series(\
            TIMESTAMP '2024-01-01', TIMESTAMP '2024-01-03', INTERVAL '1 day')",
    );

    let error = result.unwrap_err().to_string();

    assert!(error.contains("generate_series(BIGINT"), "{error}");
}

#[rstest]
fn zero_series_step_fails_during_binding(mut testing_planner: TestingPlanner) {
    let result = testing_planner.plan("SELECT * FROM generate_series(1, 5, 0)");

    let error = result.unwrap_err().to_string();

    assert!(error.contains("step must not be zero"), "{error}");
}

#[rstest]
fn count_star_over_empty_series(mut testing_planner: TestingPlanner) {
    // start > stop with a positive step yields no rows; count(*) must still be 0.
    let results = testing_planner
        .plan("SELECT count(*) FROM generate_series(5, 1)")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);

    assert_eq!(rows, vec![serde_json::json!({"count_star()": 0})]);
}

#[rstest]
fn generate_series_composes_with_aggregate(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .plan("SELECT count(*), sum(generate_series) FROM generate_series(1, 4)")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);

    assert_eq!(
        rows,
        vec![serde_json::json!({"count_star()": 4, "sum(generate_series)": 10})]
    );
}

#[rstest]
fn unsupported_aggregate_returns_error(mut testing_planner: TestingPlanner) {
    // `stddev` has no pivot lowering (unlike SUM/COUNT/MIN/MAX/AVG), so it must
    // surface as a plan-conversion error rather than silently mis-aggregating.
    let result = testing_planner.plan("SELECT STDDEV(b) FROM example_table");
    assert!(matches!(result, Err(PlannerError::PlanConversion(_))));
}

#[rstest]
fn projection_of_constant_broadcasts_to_every_row(mut testing_planner: TestingPlanner) {
    // Regression: `SELECT <constant>, <col>` takes the general projection path (the
    // literal is not a plain column ref). The constant evaluates to a single scalar
    // and must be broadcast to the batch's row count; before the fix it stayed
    // length-1, so assembling the output RecordBatch panicked with
    // "all columns in a record batch must have the same length". This is the path the
    // duckdb RemoveDerivedGroups optimizer exposes for `GROUP BY <const>` (e.g. q34).
    let results = testing_planner
        .plan("SELECT 1, name FROM example_table")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    // Every output column must be the full batch length (the invariant that failed).
    for batch in &results {
        for col in batch.columns() {
            assert_eq!(col.len(), batch.num_rows());
        }
    }

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 5, "all example_table rows returned");
    // The constant column is 1 for every row (broadcast, not a single 1).
    let keys: Vec<String> = rows[0].as_object().unwrap().keys().cloned().collect();
    assert!(
        keys.iter()
            .any(|k| rows.iter().all(|r| r[k].as_i64() == Some(1))),
        "a column should be the constant 1 broadcast to all rows"
    );
}

// A GROUP BY over an integer key and a float key. Grouping by a float column is
// unsupported (the row-key encoder only packs integers and strings), so this
// must error. The error should name the offending float key. Today it instead
// blames the first key (the perfectly valid Int64): grouped.rs builds the error
// as `DataTypeNotSupportedForGroupBy(keys[0].1)` regardless of which key
// actually failed to pack, so the message points at the wrong column.
#[rstest]
fn group_by_error_names_the_unsupported_key_not_the_first(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "mixed_keys",
        &[
            (
                "k_int",
                Type::Int64,
                Arc::new(Int64Array::from(vec![1_i64, 2, 3])) as ArrayRef,
            ),
            (
                "k_float",
                Type::Float64,
                Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])) as ArrayRef,
            ),
        ],
    );

    let compiled = testing_planner
        .plan("SELECT k_int, k_float, COUNT(*) FROM mixed_keys GROUP BY k_int, k_float")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        );
    let err = match compiled {
        Ok(_) => panic!("grouping by a float key should fail to compile"),
        Err(err) => err.to_string(),
    };

    assert!(
        err.contains("Float64"),
        "group-by error should name the unsupported float key (Float64), got: {err}"
    );
    assert!(
        err.contains("k_float"),
        "group-by error should name the offending column (k_float), got: {err}"
    );
}

/// A CTE the query reads twice: once row by row in the body, once reduced to
/// the scalar every row is compared against. DuckDB plans this as a
/// materialized CTE with two scans, and wraps the scalar subquery in its
/// single-row guard, so the whole shape runs through Cte/CteScan, the first
/// aggregate, and the lazy guard lowering together.
#[rstest]
fn cte_read_by_body_and_by_scalar_subquery(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "sales",
        &[
            ("seller", Type::Int32, int_col(vec![1, 2, 1])),
            (
                "amount",
                Type::Decimal {
                    precision: 15,
                    scale: 2,
                },
                Arc::new(
                    arrow_array::Decimal64Array::from(vec![1000i64, 2000, 3000])
                        .with_precision_and_scale(15, 2)
                        .unwrap(),
                ),
            ),
        ],
    );

    let rows = run(
        &mut testing_planner,
        "WITH totals(seller, total) AS (
             SELECT seller, sum(amount) FROM sales GROUP BY seller)
         SELECT seller, total FROM totals
         WHERE total = (SELECT max(total) FROM totals)",
    );

    assert_eq!(
        rows,
        serde_json::json!([{"seller": 1, "total": 40.0}])
            .as_array()
            .unwrap()
            .clone()
    );
}

#[rstest]
fn select_distinct_dedups_duplicate_values(mut testing_planner: TestingPlanner) {
    let rows = run(
        &mut testing_planner,
        "SELECT DISTINCT name FROM example_table",
    );

    let mut names: Vec<&str> = rows.iter().map(|r| r["name"].as_str().unwrap()).collect();
    names.sort_unstable();
    assert_eq!(names, vec!["alice", "bob", "charlie", "dave"]);
}

#[rstest]
fn select_distinct_over_multiple_columns(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "visits",
        &[
            (
                "id",
                Type::Int64,
                Arc::new(Int64Array::from(vec![1i64, 1, 2, 2, 2])) as ArrayRef,
            ),
            (
                "page",
                Type::Utf8,
                str_col(vec!["home", "home", "home", "cart", "cart"]),
            ),
        ],
    );

    let mut rows = run(&mut testing_planner, "SELECT DISTINCT id, page FROM visits");

    rows.sort_by_key(|r| {
        (
            r["id"].as_i64().unwrap(),
            r["page"].as_str().unwrap().to_string(),
        )
    });
    assert_eq!(
        rows,
        serde_json::json!([
            {"id": 1, "page": "home"},
            {"id": 2, "page": "cart"},
            {"id": 2, "page": "home"},
        ])
        .as_array()
        .unwrap()
        .clone()
    );
}

#[rstest]
fn group_by_without_aggregates_dedups_keys(mut testing_planner: TestingPlanner) {
    let rows = run(
        &mut testing_planner,
        "SELECT name FROM example_table GROUP BY name",
    );

    let mut names: Vec<&str> = rows.iter().map(|r| r["name"].as_str().unwrap()).collect();
    names.sort_unstable();
    assert_eq!(names, vec!["alice", "bob", "charlie", "dave"]);
}

// The projection reads nothing from the dedup, only its row count: one output
// row per group must still come through.
#[rstest]
fn constant_projection_over_group_by_without_aggregates(mut testing_planner: TestingPlanner) {
    let rows = run(
        &mut testing_planner,
        "SELECT 1 FROM example_table GROUP BY name",
    );

    assert_eq!(rows.len(), 4);
    assert!(rows.iter().all(|r| r["1"] == 1));
}

// The outer COUNT(*) reads none of the subquery's aggregate values, so the
// optimizer prunes them and leaves the inner GROUP BY with no aggregate
// expressions at all; counting its groups must still work.
#[rstest]
fn count_over_grouped_subquery_with_pruned_aggregates(mut testing_planner: TestingPlanner) {
    let rows = run(
        &mut testing_planner,
        "SELECT COUNT(*) FROM (SELECT name, max(a) FROM example_table GROUP BY name)",
    );

    assert_eq!(rows[0]["count_star()"], 4);
}
