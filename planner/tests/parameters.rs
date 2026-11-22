#[allow(dead_code)]
#[path = "common/mod.rs"]
mod common;

use common::*;
use planner::{BoundLogicalType, ExtraTypeInfo, LogicalTypeId, ScalarValue};
use rstest::rstest;
use serde_json::json;

#[rstest]
#[case(ScalarValue::Boolean(true), "true")]
#[case(ScalarValue::Int8(-7), "-7::TINYINT")]
#[case(ScalarValue::Int16(-300), "-300::SMALLINT")]
#[case(ScalarValue::Int32(70_000), "70000::INTEGER")]
#[case(ScalarValue::Int64(-5_000_000_000), "-5000000000::BIGINT")]
#[case(ScalarValue::UInt8(200), "200::UTINYINT")]
#[case(ScalarValue::UInt16(60_000), "60000::USMALLINT")]
#[case(ScalarValue::UInt32(4_000_000_000), "4000000000::UINTEGER")]
#[case(ScalarValue::UInt64(u64::MAX), "18446744073709551615::UBIGINT")]
#[case(
    ScalarValue::Int128(-170_141_183_460_469_231_731_687_303_715_884_105_727),
    "-170141183460469231731687303715884105727::HUGEINT"
)]
#[case(ScalarValue::Float32(1.5), "1.5::FLOAT")]
#[case(ScalarValue::Float64(-2.25), "-2.25::DOUBLE")]
#[case(ScalarValue::Decimal { value: -12_345, width: 10, scale: 2 }, "-123.45::DECIMAL(10,2)")]
#[case(
    ScalarValue::Decimal { value: 12_345_678_901_234_567_890_123, width: 30, scale: 3 },
    "12345678901234567890.123::DECIMAL(30,3)"
)]
#[case(ScalarValue::Utf8("hello".to_string()), "'hello'")]
#[case(ScalarValue::Date(19_723), "DATE '2024-01-01'")]
#[case(
    ScalarValue::Timestamp(1_704_067_200_000_001),
    "TIMESTAMP '2024-01-01 00:00:00.000001'"
)]
#[case(
    ScalarValue::TimestampTz(1_704_067_200_000_001),
    "TIMESTAMPTZ '2024-01-01 00:00:00.000001+00'"
)]
#[case(
    ScalarValue::Interval { months: 14, days: 3, micros: 4_000_000 },
    "INTERVAL '1 year 2 months 3 days 4 seconds'"
)]
#[case(ScalarValue::Variant(r#"{"a":1}"#.to_string()), r#"'{"a":1}'::VARIANT"#)]
fn select_returns_bound_parameter(
    mut testing_planner: TestingPlanner,
    #[case] parameter: ScalarValue,
    #[case] literal: &str,
) {
    let expected = run_batches(&mut testing_planner, &format!("SELECT {literal}"));

    let actual = run_batches_with_parameters(&mut testing_planner, "SELECT $1", vec![parameter]);

    assert_eq!(actual[0].columns(), expected[0].columns());
}

#[rstest]
#[case(LogicalTypeId::BOOLEAN, "BOOLEAN")]
#[case(LogicalTypeId::TINYINT, "TINYINT")]
#[case(LogicalTypeId::SMALLINT, "SMALLINT")]
#[case(LogicalTypeId::INTEGER, "INTEGER")]
#[case(LogicalTypeId::BIGINT, "BIGINT")]
#[case(LogicalTypeId::UTINYINT, "UTINYINT")]
#[case(LogicalTypeId::USMALLINT, "USMALLINT")]
#[case(LogicalTypeId::UINTEGER, "UINTEGER")]
#[case(LogicalTypeId::UBIGINT, "UBIGINT")]
#[case(LogicalTypeId::HUGEINT, "HUGEINT")]
#[case(LogicalTypeId::FLOAT, "FLOAT")]
#[case(LogicalTypeId::DOUBLE, "DOUBLE")]
#[case(LogicalTypeId::VARCHAR, "VARCHAR")]
#[case(LogicalTypeId::DATE, "DATE")]
#[case(LogicalTypeId::TIMESTAMP, "TIMESTAMP")]
#[case(LogicalTypeId::TIMESTAMP_TZ, "TIMESTAMPTZ")]
#[case(LogicalTypeId::INTERVAL, "INTERVAL")]
#[case(LogicalTypeId::VARIANT, "VARIANT")]
fn select_returns_null_parameter(
    mut testing_planner: TestingPlanner,
    #[case] type_id: LogicalTypeId,
    #[case] sql_type: &str,
) {
    let parameter = ScalarValue::Null(BoundLogicalType::plain(type_id));
    let expected = run_batches(&mut testing_planner, &format!("SELECT NULL::{sql_type}"));

    let actual = run_batches_with_parameters(&mut testing_planner, "SELECT $1", vec![parameter]);

    assert_eq!(actual[0].columns(), expected[0].columns());
}

#[rstest]
fn filter_compares_column_to_bound_parameters(mut testing_planner: TestingPlanner) {
    let parameters = vec![
        ScalarValue::Int32(3),
        ScalarValue::Utf8("alice".to_string()),
    ];

    let rows = batches_to_json(&run_batches_with_parameters(
        &mut testing_planner,
        "SELECT a FROM example_table WHERE a > $1 AND name = $2",
        parameters,
    ));

    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["a"], 5);
}

#[rstest]
fn select_reuses_parameter_among_several(mut testing_planner: TestingPlanner) {
    let parameters = vec![ScalarValue::Int32(3), ScalarValue::Int32(10)];

    let rows = batches_to_json(&run_batches_with_parameters(
        &mut testing_planner,
        "SELECT $1 + $2 + $1",
        parameters,
    ));

    assert_eq!(only_column(&rows[0]), 16);
}

fn build_integer_list(elements: Vec<ScalarValue>) -> ScalarValue {
    ScalarValue::List {
        element_type: BoundLogicalType::plain(LogicalTypeId::INTEGER),
        elements,
    }
}

#[rstest]
fn unnest_expands_list_parameter(mut testing_planner: TestingPlanner) {
    let list = build_integer_list(vec![
        ScalarValue::Int32(1),
        ScalarValue::Int32(2),
        ScalarValue::Int32(3),
    ]);

    let rows = batches_to_json(&run_batches_with_parameters(
        &mut testing_planner,
        "SELECT unnest($1) AS element ORDER BY element",
        vec![list],
    ));

    assert_eq!(
        rows,
        vec![
            json!({"element": 1}),
            json!({"element": 2}),
            json!({"element": 3})
        ]
    );
}

#[rstest]
fn unnest_keeps_null_element_of_list_parameter(mut testing_planner: TestingPlanner) {
    let list = build_integer_list(vec![
        ScalarValue::Int32(1),
        ScalarValue::Null(BoundLogicalType::plain(LogicalTypeId::INTEGER)),
    ]);

    let batches =
        run_batches_with_parameters(&mut testing_planner, "SELECT unnest($1)", vec![list]);

    let row_count: usize = batches.iter().map(|batch| batch.num_rows()).sum();
    let null_count: usize = batches
        .iter()
        .map(|batch| batch.column(0).null_count())
        .sum();
    assert_eq!((row_count, null_count), (2, 1));
}

#[rstest]
#[case::empty_list(build_integer_list(Vec::new()))]
#[case::null_list(ScalarValue::Null(BoundLogicalType {
    id: LogicalTypeId::LIST,
    extra: ExtraTypeInfo::List(Box::new(BoundLogicalType::plain(LogicalTypeId::INTEGER))),
}))]
fn unnest_of_list_parameter_without_elements_emits_no_rows(
    mut testing_planner: TestingPlanner,
    #[case] list: ScalarValue,
) {
    let batches =
        run_batches_with_parameters(&mut testing_planner, "SELECT unnest($1)", vec![list]);

    let row_count: usize = batches.iter().map(|batch| batch.num_rows()).sum();
    assert_eq!(row_count, 0);
}

#[rstest]
fn unnest_expands_scalar_parameters_in_written_out_list(mut testing_planner: TestingPlanner) {
    let parameters = vec![ScalarValue::Int32(7), ScalarValue::Int32(8)];

    let rows = batches_to_json(&run_batches_with_parameters(
        &mut testing_planner,
        "SELECT unnest([$1, $2]) AS element ORDER BY element",
        parameters,
    ));

    assert_eq!(rows, vec![json!({"element": 7}), json!({"element": 8})]);
}

#[rstest]
fn unnest_expands_list_parameter_per_table_row(mut testing_planner: TestingPlanner) {
    let list = build_integer_list(vec![ScalarValue::Int32(1), ScalarValue::Int32(2)]);

    let batches = run_batches_with_parameters(
        &mut testing_planner,
        "SELECT a, unnest($1) FROM example_table",
        vec![list],
    );

    let row_count: usize = batches.iter().map(|batch| batch.num_rows()).sum();
    assert_eq!(row_count, 10);
}

#[rstest]
fn select_of_list_parameter_is_rejected(mut testing_planner: TestingPlanner) {
    let list = build_integer_list(vec![ScalarValue::Int32(1)]);
    let transaction = testing_planner.transaction();

    let result = testing_planner
        .planner
        .plan_with_parameters("SELECT $1", vec![list], transaction);

    assert!(result.is_err());
}

#[rstest]
#[case(Vec::new())]
#[case(vec![ScalarValue::Int32(1), ScalarValue::Int32(2)])]
fn plan_rejects_parameter_count_mismatch(
    mut testing_planner: TestingPlanner,
    #[case] parameters: Vec<ScalarValue>,
) {
    let transaction = testing_planner.transaction();

    let result = testing_planner
        .planner
        .plan_with_parameters("SELECT $1", parameters, transaction);

    let message = result.err().unwrap().to_string();
    assert!(
        message.contains("statement takes 1 parameters"),
        "{message}"
    );
}
