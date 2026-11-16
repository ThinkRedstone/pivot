//! End-to-end tests of `unnest(...)` in a select list, over a list written out
//! in the query.

use crate::common::*;
use rstest::rstest;
use serde_json::{Value, json};

/// The rows as JSON text, sorted, so a test can compare without depending on
/// the order an unnest emits rows in.
fn sorted_rows(rows: Vec<Value>) -> Vec<String> {
    let mut rows: Vec<String> = rows.iter().map(Value::to_string).collect();
    rows.sort();
    rows
}

#[rstest]
fn unnest_repeats_each_row_once_per_element(mut testing_planner: TestingPlanner) {
    let rows = run(
        &mut testing_planner,
        "SELECT unnest(['x', 'y']) AS tag, a FROM example_table WHERE a <= 2",
    );

    assert_eq!(
        sorted_rows(rows),
        sorted_rows(vec![
            json!({"tag": "x", "a": 1}),
            json!({"tag": "y", "a": 1}),
            json!({"tag": "x", "a": 2}),
            json!({"tag": "y", "a": 2}),
        ])
    );
}

#[rstest]
fn group_by_the_unnested_element(mut testing_planner: TestingPlanner) {
    let rows = run(
        &mut testing_planner,
        "SELECT tag, sum(a) AS total \
         FROM (SELECT unnest(['x', 'y']) AS tag, a FROM example_table) \
         GROUP BY tag",
    );

    assert_eq!(
        sorted_rows(rows),
        sorted_rows(vec![
            json!({"tag": "x", "total": 15}),
            json!({"tag": "y", "total": 15}),
        ])
    );
}

#[rstest]
fn filter_on_the_unnested_element(mut testing_planner: TestingPlanner) {
    let rows = run(
        &mut testing_planner,
        "SELECT factor, a FROM (SELECT unnest([1, 2, 3]) AS factor, a FROM example_table) \
         WHERE factor = a",
    );

    assert_eq!(
        sorted_rows(rows),
        sorted_rows(vec![
            json!({"factor": 1, "a": 1}),
            json!({"factor": 2, "a": 2}),
            json!({"factor": 3, "a": 3}),
        ])
    );
}

#[rstest]
fn shorter_list_is_padded_with_nulls(mut testing_planner: TestingPlanner) {
    let rows = run(
        &mut testing_planner,
        "SELECT unnest([7]) AS short, unnest(['p', 'q']) AS long, a \
         FROM example_table WHERE a = 1",
    );

    assert_eq!(
        sorted_rows(rows),
        sorted_rows(vec![
            json!({"short": 7, "long": "p", "a": 1}),
            json!({"long": "q", "a": 1}),
        ])
    );
}

#[rstest]
fn null_element_becomes_a_null_row(mut testing_planner: TestingPlanner) {
    let rows = run(
        &mut testing_planner,
        "SELECT unnest([1, NULL]) AS element, a FROM example_table WHERE a = 1",
    );

    assert_eq!(
        sorted_rows(rows),
        sorted_rows(vec![json!({"element": 1, "a": 1}), json!({"a": 1})])
    );
}

#[rstest]
#[case::empty_list("CAST([] AS INTEGER[])")]
#[case::null_list("CAST(NULL AS INTEGER[])")]
fn list_without_elements_emits_no_rows(mut testing_planner: TestingPlanner, #[case] list: &str) {
    let rows = run(
        &mut testing_planner,
        &format!("SELECT unnest({list}) AS element, a FROM example_table"),
    );

    assert!(rows.is_empty());
}

#[rstest]
fn unnest_without_a_from_clause(mut testing_planner: TestingPlanner) {
    let rows = run(&mut testing_planner, "SELECT unnest([1, 2, 3]) AS element");

    assert_eq!(
        sorted_rows(rows),
        sorted_rows(vec![
            json!({"element": 1}),
            json!({"element": 2}),
            json!({"element": 3}),
        ])
    );
}

#[rstest]
fn unnest_of_a_list_built_from_columns(mut testing_planner: TestingPlanner) {
    let rows = run(
        &mut testing_planner,
        "SELECT unnest([a, b]) AS element, name FROM example_table WHERE a <= 2",
    );

    assert_eq!(
        sorted_rows(rows),
        sorted_rows(vec![
            json!({"element": 1, "name": "alice"}),
            json!({"element": 10, "name": "alice"}),
            json!({"element": 2, "name": "bob"}),
            json!({"element": 20, "name": "bob"}),
        ])
    );
}

#[rstest]
fn fields_of_unnested_named_tuples(mut testing_planner: TestingPlanner) {
    let rows = run(
        &mut testing_planner,
        "SELECT pair.label AS label, pair.amount AS amount \
         FROM (SELECT unnest([{'label': 'a', 'amount': a}, {'label': 'b', 'amount': b}]) AS pair \
               FROM example_table WHERE a = 1)",
    );

    assert_eq!(
        sorted_rows(rows),
        sorted_rows(vec![
            json!({"label": "a", "amount": 1}),
            json!({"label": "b", "amount": 10}),
        ])
    );
}

#[rstest]
fn tuples_of_constants_only(mut testing_planner: TestingPlanner) {
    let rows = run(
        &mut testing_planner,
        "SELECT pair.label AS label, pair.amount AS amount, a \
         FROM (SELECT unnest([{'label': 'x', 'amount': 7}, {'label': 'y', 'amount': 8}]) AS pair, a \
               FROM example_table WHERE a = 1)",
    );

    assert_eq!(
        sorted_rows(rows),
        sorted_rows(vec![
            json!({"label": "x", "amount": 7, "a": 1}),
            json!({"label": "y", "amount": 8, "a": 1}),
        ])
    );
}

// Each grouped row is turned into one (category, name) pair per grouping
// column, the pairs are filtered, and the counts are summed per pair.
#[rstest]
fn filter_and_regroup_by_the_fields_of_unnested_tuples(mut testing_planner: TestingPlanner) {
    let rows = run(
        &mut testing_planner,
        "SELECT pair.category AS category, pair.name AS name, sum(row_count) AS value \
         FROM ( \
             SELECT unnest([{'category': 'name', 'name': name}, \
                            {'category': 'big', 'name': CAST(a > 3 AS VARCHAR)}]) AS pair, \
                    row_count \
             FROM (SELECT name, a, count(*) AS row_count FROM example_table GROUP BY name, a) \
         ) \
         WHERE pair.name != 'bob' OR pair.category = 'big' \
         GROUP BY pair.category, pair.name",
    );

    assert_eq!(
        sorted_rows(rows),
        sorted_rows(vec![
            json!({"category": "name", "name": "alice", "value": 2}),
            json!({"category": "name", "name": "charlie", "value": 1}),
            json!({"category": "name", "name": "dave", "value": 1}),
            json!({"category": "big", "name": "false", "value": 3}),
            json!({"category": "big", "name": "true", "value": 2}),
        ])
    );
}

#[rstest]
fn unnest_of_a_list_not_written_out_in_the_query_is_rejected(mut testing_planner: TestingPlanner) {
    let error = testing_planner
        .plan("SELECT unnest(string_split(name, 'l')) AS part FROM example_table")
        .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("UNNEST is only supported over a list written out in the query"),
        "{error}"
    );
}

// The query reads no column of the table, so the scan below the unnest has
// nothing of its own to emit.
#[rstest]
#[case::unnest_alone("SELECT unnest([1, 2]) AS element FROM example_table", vec![1, 1, 1, 1, 1, 2, 2, 2, 2, 2])]
#[case::filtered_above(
    "SELECT element FROM (SELECT unnest([1, 2]) AS element FROM example_table) WHERE element = 2",
    vec![2, 2, 2, 2, 2]
)]
fn unnest_over_a_table_no_column_is_read_from(
    mut testing_planner: TestingPlanner,
    #[case] sql: &str,
    #[case] expected: Vec<i64>,
) {
    let rows = run(&mut testing_planner, sql);

    let mut elements: Vec<i64> = rows
        .iter()
        .map(|row| row["element"].as_i64().unwrap())
        .collect();
    elements.sort();
    assert_eq!(elements, expected);
}

// The tuples are never NULL but one of them has a NULL field, which has to
// group as NULL.
#[rstest]
fn group_by_a_null_field_of_unnested_tuples(mut testing_planner: TestingPlanner) {
    let rows = run(
        &mut testing_planner,
        "SELECT pair.amount AS amount, count(*) AS row_count \
         FROM (SELECT unnest([{'amount': NULL::INTEGER, 'label': name}, \
                              {'amount': 5, 'label': name}]) AS pair \
               FROM example_table) \
         GROUP BY 1",
    );

    let mut groups: Vec<(Option<i64>, i64)> = rows
        .iter()
        .map(|row| (row["amount"].as_i64(), row["row_count"].as_i64().unwrap()))
        .collect();
    groups.sort();
    assert_eq!(groups, vec![(None, 5), (Some(5), 5)]);
}
