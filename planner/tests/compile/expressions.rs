use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, Int32Array, StringViewArray};

use crate::common::*;
use planner::Error as PlannerError;
use planner::types::Type;
use rstest::rstest;

#[rstest]
fn filter_contains_substring(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .plan("SELECT name, b FROM example_table WHERE contains(name, 'ali')")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["b"].as_i64().unwrap());

    // "alice" appears at rows with b=10 and b=50.
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["name"], "alice");
    assert_eq!(rows[1]["name"], "alice");
}

#[rstest]
fn filter_contains_no_match(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .plan("SELECT name FROM example_table WHERE contains(name, 'zzz')")
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
fn filter_contains_matches_all(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "substrings",
        &[(
            "s",
            Type::Utf8,
            Arc::new(StringViewArray::from(vec!["aaa", "baab", "caaac"])) as ArrayRef,
        )],
    );

    let results = testing_planner
        .plan("SELECT s FROM substrings WHERE contains(s, 'aa')")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);
    assert_eq!(rows.len(), 3);
}

#[rstest]
fn contains_then_group_by(mut testing_planner: TestingPlanner) {
    // Filter to names containing "a" (alice, charlie, dave), then group by name.
    let results = testing_planner
        .plan("SELECT name, COUNT(*) FROM example_table WHERE contains(name, 'a') GROUP BY name")
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

    // alice (2 rows), charlie (1), dave (1) all contain "a"; bob does not.
    assert_eq!(rows.len(), 3);
    let alice = rows.iter().find(|r| r["name"] == "alice").unwrap();
    assert_eq!(alice["count_star()"], 2);
    assert!(rows.iter().all(|r| r["name"] != "bob"));
}

/// Epoch microseconds at 0s, 90s, 150s and 3690s: the minute-of-hour fixture
/// the computed-group-key tests share.
const SECOND_MARKS: [i64; 4] = [0, 90_000_000, 150_000_000, 3_690_000_000];

#[rstest]
fn group_by_minute_of_timestamp(mut testing_planner: TestingPlanner) {
    use arrow_array::Int64Array;
    // EventTime as epoch microseconds, at 0s, 90s, 150s and 3690s.
    // Minute-of-hour = (t mod 3600) / 60: 0 -> 0, 90 -> 1, 150 -> 2,
    // 3690 -> 1 (3690 mod 3600 = 90).
    testing_planner.add_table(
        "events",
        &[(
            "EventTime",
            Type::Timestamp,
            Arc::new(Int64Array::from(SECOND_MARKS.to_vec())) as ArrayRef,
        )],
    );

    let results = testing_planner
        .plan("SELECT extract(minute FROM EventTime) AS m, COUNT(*) FROM events GROUP BY m")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["m"].as_i64().unwrap());

    // Minutes 0 and 2 occur once; minute 1 occurs twice.
    assert_eq!(rows.len(), 3);
    assert_eq!(
        rows.iter()
            .map(|r| r["m"].as_i64().unwrap())
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    let minute_one = rows.iter().find(|r| r["m"] == 1).unwrap();
    assert_eq!(minute_one["count_star()"], 2);
}

#[rstest]
fn group_by_minute_of_timestamptz(mut testing_planner: TestingPlanner) {
    use arrow_array::Int64Array;
    // The same minute-of-hour fixture on a zone-carrying column: extraction
    // reads the UTC calendar, so the buckets match the zone-less case.
    testing_planner.add_table(
        "tz_events",
        &[(
            "EventTime",
            Type::TimestampTz,
            Arc::new(Int64Array::from(SECOND_MARKS.to_vec())) as ArrayRef,
        )],
    );

    let results = testing_planner
        .plan("SELECT extract(minute FROM EventTime) AS m, COUNT(*) FROM tz_events GROUP BY m")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["m"].as_i64().unwrap());

    assert_eq!(rows.len(), 3);
    assert_eq!(
        rows.iter()
            .map(|r| r["m"].as_i64().unwrap())
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    let minute_one = rows.iter().find(|r| r["m"] == 1).unwrap();
    assert_eq!(minute_one["count_star()"], 2);
}

/// A table with a minute-of-hour computed group key (`EventTime` 0,90,150,3690 →
/// minutes 0,1,2,1) and a value column. minute 1 holds two rows (v=20, v=40).
fn minute_grouped_table(testing_planner: &mut TestingPlanner, name: &str) {
    use arrow_array::Int64Array;
    testing_planner.add_table(
        name,
        &[
            (
                "EventTime",
                Type::Timestamp,
                Arc::new(Int64Array::from(SECOND_MARKS.to_vec())) as ArrayRef,
            ),
            (
                "v",
                Type::Int32,
                Arc::new(Int32Array::from(vec![10, 20, 30, 40])) as ArrayRef,
            ),
        ],
    );
}

#[rstest]
fn group_by_computed_key_sum(mut testing_planner: TestingPlanner) {
    minute_grouped_table(&mut testing_planner, "ev");

    let results = testing_planner
        .plan("SELECT extract(minute FROM EventTime) AS m, SUM(v) FROM ev GROUP BY m")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["m"].as_i64().unwrap());

    assert_eq!(
        rows.iter()
            .map(|r| r["m"].as_i64().unwrap())
            .collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    // minute 1 = 20 + 40 = 60.
    assert_eq!(
        rows.iter()
            .map(|r| r["sum(v)"].as_i64().unwrap())
            .collect::<Vec<_>>(),
        vec![10, 60, 30]
    );
}

#[rstest]
fn group_by_computed_key_min_max(mut testing_planner: TestingPlanner) {
    minute_grouped_table(&mut testing_planner, "ev");

    let results = testing_planner
        .plan("SELECT extract(minute FROM EventTime) AS m, MIN(v), MAX(v) FROM ev GROUP BY m")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["m"].as_i64().unwrap());

    // minute 1 = {20, 40}; the others are singletons.
    assert_eq!(
        rows.iter()
            .map(|r| r["min(v)"].as_i64().unwrap())
            .collect::<Vec<_>>(),
        vec![10, 20, 30]
    );
    assert_eq!(
        rows.iter()
            .map(|r| r["max(v)"].as_i64().unwrap())
            .collect::<Vec<_>>(),
        vec![10, 40, 30]
    );
}

#[rstest]
fn group_by_computed_key_multi_agg(mut testing_planner: TestingPlanner) {
    minute_grouped_table(&mut testing_planner, "ev");

    let results = testing_planner

        .plan("SELECT extract(minute FROM EventTime) AS m, SUM(v), COUNT(*), MAX(v) FROM ev GROUP BY m")
        .unwrap()
        .compile(testing_planner.dispatcher(), testing_planner.transaction().as_ref())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["m"].as_i64().unwrap());

    assert_eq!(
        rows.iter()
            .map(|r| r["sum(v)"].as_i64().unwrap())
            .collect::<Vec<_>>(),
        vec![10, 60, 30] // sum
    );
    assert_eq!(
        rows.iter()
            .map(|r| r["count_star()"].as_i64().unwrap())
            .collect::<Vec<_>>(),
        vec![1, 2, 1] // count(*)
    );
    assert_eq!(
        rows.iter()
            .map(|r| r["max(v)"].as_i64().unwrap())
            .collect::<Vec<_>>(),
        vec![10, 40, 30] // max
    );
}

#[rstest]
fn group_by_computed_key_avg(mut testing_planner: TestingPlanner) {
    minute_grouped_table(&mut testing_planner, "ev");

    // AVG over a computed group key, lowered by DuckDB to sum+count over the
    // key column.
    let results = testing_planner
        .plan(
            "SELECT extract(minute FROM EventTime) AS m, AVG(v) AS a FROM ev GROUP BY m ORDER BY m",
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
    // Ordered by minute; `a` is the AVG alias. minute 1 = avg(20, 40) = 30.
    let avgs: Vec<f64> = rows.iter().map(|r| r["a"].as_f64().unwrap()).collect();
    assert_eq!(avgs, vec![10.0, 30.0, 30.0]);
}

#[rstest]
fn group_by_plain_and_computed_key(mut testing_planner: TestingPlanner) {
    use arrow_array::Int64Array;
    // A plain column key mixed with a computed key. Groups
    // (uid, minute): (1,0)=v[10]; (1,1)=v[20,40]; (2,2)=v[30]. The multi-key row
    // encoder names the two key columns k0, k1.
    testing_planner.add_table(
        "t",
        &[
            (
                "uid",
                Type::Int32,
                Arc::new(Int32Array::from(vec![1, 1, 2, 1])) as ArrayRef,
            ),
            (
                "EventTime",
                Type::Timestamp,
                Arc::new(Int64Array::from(vec![
                    0i64,
                    90_000_000,
                    150_000_000,
                    90_000_000,
                ])) as ArrayRef,
            ),
            (
                "v",
                Type::Int32,
                Arc::new(Int32Array::from(vec![10, 20, 30, 40])) as ArrayRef,
            ),
        ],
    );

    let results = testing_planner

        .plan("SELECT uid, extract(minute FROM EventTime) AS m, SUM(v), COUNT(*) FROM t GROUP BY uid, m")
        .unwrap()
        .compile(testing_planner.dispatcher(), testing_planner.transaction().as_ref())
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| (r["uid"].as_i64().unwrap(), r["m"].as_i64().unwrap()));

    assert_eq!(
        rows.iter()
            .map(|r| (r["uid"].as_i64().unwrap(), r["m"].as_i64().unwrap()))
            .collect::<Vec<_>>(),
        vec![(1, 0), (1, 1), (2, 2)]
    );
    assert_eq!(
        rows.iter()
            .map(|r| r["sum(v)"].as_i64().unwrap())
            .collect::<Vec<_>>(),
        vec![10, 60, 30] // sum
    );
    assert_eq!(
        rows.iter()
            .map(|r| r["count_star()"].as_i64().unwrap())
            .collect::<Vec<_>>(),
        vec![1, 2, 1] // count(*)
    );
}

#[rstest]
fn group_by_column_and_arithmetic_key(mut testing_planner: TestingPlanner) {
    // A plain column grouped alongside an arithmetic expression of a DIFFERENT
    // column, so the computed key isn't functionally derived from a group key
    // (which DuckDB's RemoveDerivedGroups would prune). a=[1,1,2,1], b-1=[4,5,6,4]
    // → groups (a, b-1): (1,4)=2 rows, (1,5)=1, (2,6)=1.
    testing_planner.add_table(
        "t",
        &[
            (
                "a",
                Type::Int32,
                Arc::new(Int32Array::from(vec![1, 1, 2, 1])) as ArrayRef,
            ),
            (
                "b",
                Type::Int32,
                Arc::new(Int32Array::from(vec![5, 6, 7, 5])) as ArrayRef,
            ),
        ],
    );

    let results = testing_planner
        .plan("SELECT a, b - 1 AS d, COUNT(*) FROM t GROUP BY a, d")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| (r["a"].as_i64().unwrap(), r["d"].as_i64().unwrap()));

    assert_eq!(
        rows.iter()
            .map(|r| (r["a"].as_i64().unwrap(), r["d"].as_i64().unwrap()))
            .collect::<Vec<_>>(),
        vec![(1, 4), (1, 5), (2, 6)]
    );
    assert_eq!(
        rows.iter()
            .map(|r| r["count_star()"].as_i64().unwrap())
            .collect::<Vec<_>>(),
        vec![2, 1, 1]
    );
}

/// Every `extract(<part> FROM ts)` part, validated against values produced by
/// the DuckDB CLI at four timestamps (incl. a pre-epoch negative one and dates
/// that exercise the ISO-week year-boundary rollover):
///   t0 = 1704067200  2024-01-01 00:00:00 (Mon)
///   t1 = 1700000000  2023-11-14 22:13:20 (Tue)
///   t2 = 1262304000  2010-01-01 00:00:00 (Fri, ISO week 53 of 2009)
///   t3 =     -100000  1969-12-30 20:13:20 (Tue, ISO week 1 of 1970)
#[rstest]
fn extract_all_date_parts(mut testing_planner: TestingPlanner) {
    use arrow_array::Int64Array;
    let seconds = [1_704_067_200i64, 1_700_000_000, 1_262_304_000, -100_000];
    let timestamps: Vec<i64> = seconds.iter().map(|s| s * 1_000_000).collect();
    testing_planner.add_table(
        "ts",
        &[(
            "EventTime",
            Type::Timestamp,
            Arc::new(Int64Array::from(timestamps.clone())) as ArrayRef,
        )],
    );

    // (part, [expected for t0, t1, t2, t3]).
    let cases: &[(&str, [i64; 4])] = &[
        (
            "epoch",
            [1_704_067_200, 1_700_000_000, 1_262_304_000, -100_000],
        ),
        ("second", [0, 20, 0, 20]),
        ("millisecond", [0, 20_000, 0, 20_000]),
        ("microsecond", [0, 20_000_000, 0, 20_000_000]),
        ("minute", [0, 13, 0, 13]),
        ("hour", [0, 22, 0, 20]),
        ("day", [1, 14, 1, 30]),
        ("month", [1, 11, 1, 12]),
        ("quarter", [1, 4, 1, 4]),
        ("year", [2024, 2023, 2010, 1969]),
        ("decade", [202, 202, 201, 196]),
        ("century", [21, 21, 21, 20]),
        ("millennium", [3, 3, 3, 2]),
        ("dayofweek", [1, 2, 5, 2]),
        ("isodow", [1, 2, 5, 2]),
        ("dayofyear", [1, 318, 1, 364]),
        ("week", [1, 46, 53, 1]),
    ];

    for (part, expected) in cases {
        let results = testing_planner
            .plan(&format!(
                "SELECT extract({part} FROM EventTime) AS v, \
                 extract(epoch FROM EventTime) AS t FROM ts"
            ))
            .unwrap()
            .compile(
                testing_planner.dispatcher(),
                testing_planner.transaction().as_ref(),
            )
            .unwrap()
            .collect()
            .unwrap();

        // The select item aliases are `v` (the extracted value) and `t` (the
        // source instant in whole epoch seconds, which identifies the row as an
        // integer whatever resolution the column counts in). Map each row's
        // timestamp to its value, then compare against the expectation (row
        // order isn't fixed).
        let rows = batches_to_json(&results);
        for (i, &t) in seconds.iter().enumerate() {
            let row = rows
                .iter()
                .find(|r| r["t"].as_i64().unwrap() == t)
                .unwrap_or_else(|| panic!("{part}: no row for ts {t}"));
            assert_eq!(
                row["v"].as_i64().unwrap(),
                expected[i],
                "extract({part} FROM {t})"
            );
        }
    }
}

#[rstest]
fn filter_in_list_int(mut testing_planner: TestingPlanner) {
    // a IN (2, 4): rows with a=2 and a=4 survive.
    let results = testing_planner
        .plan("SELECT a FROM example_table WHERE a IN (2, 4)")
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
        rows.iter()
            .map(|r| r["a"].as_i64().unwrap())
            .collect::<Vec<_>>(),
        vec![2, 4]
    );
}

#[rstest]
fn filter_in_list_string(mut testing_planner: TestingPlanner) {
    // name IN ('alice', 'charlie'): alice (×2) and charlie (×1) survive.
    let results = testing_planner
        .plan("SELECT name FROM example_table WHERE name IN ('alice', 'charlie')")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let mut names = batches_to_json(&results)
        .iter()
        .map(|r| r["name"].as_str().unwrap().to_string())
        .collect::<Vec<_>>();
    names.sort();
    assert_eq!(names, vec!["alice", "alice", "charlie"]);
}

/// DuckDB's `InFilter::ToExpression` (the IN pushed into a scan) reconstructs
/// the membership test as a `COMPARE_IN` operator, which lowers to
/// `Expression::InList` rather than the `OR` conjunction the optimizer emits
/// for an un-pushed list. That path isn't reachable from the parquet-backed
/// test harness, so exercise `InList::compile` directly: `ts IN (-1, 6)` over
/// `[-1, 6, 3, 6]` should yield `[true, true, false, true]`.
#[test]
fn in_list_compare_in_compiles_to_membership_mask() {
    use arrow_array::{BooleanArray, Int16Array, RecordBatch, Scalar};
    use arrow_schema::{DataType, Field, Schema};
    use planner::expression::{Expression, InList, Ref};

    let constant =
        |v: i16| Expression::Constant(Scalar::new(Arc::new(Int16Array::from(vec![v])) as ArrayRef));
    let in_list = InList {
        input: Box::new(Expression::Ref(Ref {
            column_idx: 0,
            return_type: Type::Int16,
            name: None,
        })),
        values: vec![constant(-1), constant(6)],
    };

    let mut eval = in_list.compile().unwrap()();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("ts", DataType::Int16, false)])),
        vec![Arc::new(Int16Array::from(vec![-1i16, 6, 3, 6])) as ArrayRef],
    )
    .unwrap();

    let result = eval(&batch);
    let (arr, _) = result.as_datum().get();
    let mask = arr.as_any().downcast_ref::<BooleanArray>().unwrap();
    assert_eq!(
        (0..mask.len()).map(|i| mask.value(i)).collect::<Vec<_>>(),
        vec![true, true, false, true]
    );
}

#[rstest]
fn case_expression_in_projection(mut testing_planner: TestingPlanner) {
    // a = [1,2,3,4,5]; a < 3 -> 'low' (a=1,2), else 'high' (a=3,4,5).
    let results = testing_planner
        .plan("SELECT CASE WHEN a < 3 THEN 'low' ELSE 'high' END AS c FROM example_table")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let mut labels = batches_to_json(&results)
        .iter()
        .map(|r| only_column(r).as_str().unwrap().to_string())
        .collect::<Vec<_>>();
    labels.sort();
    assert_eq!(labels, vec!["high", "high", "high", "low", "low"]);
}

#[rstest]
fn case_expression_multi_arm_group_key(mut testing_planner: TestingPlanner) {
    // Three-arm CASE bucketing a into lo/mid/hi, grouped and counted:
    //   a=1     -> "lo"  (1 row)
    //   a=2,3   -> "mid" (2 rows)
    //   a=4,5   -> "hi"  (2 rows)
    let results = testing_planner
        .plan(
            "SELECT CASE WHEN a < 2 THEN 'lo' WHEN a < 4 THEN 'mid' ELSE 'hi' END AS c, COUNT(*) \
             FROM example_table GROUP BY 1",
        )
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["c"].as_str().unwrap().to_string());
    let counts: Vec<(String, i64)> = rows
        .iter()
        .map(|r| {
            (
                r["c"].as_str().unwrap().to_string(),
                r["count_star()"].as_i64().unwrap(),
            )
        })
        .collect();
    assert_eq!(
        counts,
        vec![
            ("hi".to_string(), 2),
            ("lo".to_string(), 1),
            ("mid".to_string(), 2),
        ]
    );
}

#[rstest]
fn arithmetic_in_projection(mut testing_planner: TestingPlanner) {
    // Covers all three operators, a nested expression, and a mixed-width
    // operand pair (Int32 column + BIGINT constant) that exercises the
    // Int64 coercion path.
    let results = testing_planner
        .plan("SELECT a + 1, b - 2, (a + b) * 2, a + 5000000000 FROM example_table")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["(a + 1)"].as_i64().unwrap());

    assert_eq!(rows.len(), 5);
    // First row: a=1, b=10.
    assert_eq!(rows[0]["(a + 1)"], 2); // 1 + 1
    assert_eq!(rows[0]["(b - 2)"], 8); // 10 - 2
    assert_eq!(rows[0]["((a + b) * 2)"], 22); // (1 + 10) * 2
    assert_eq!(rows[0]["(a + 5000000000)"], 5000000001i64); // 1 + 5000000000
    // Last row: a=5, b=50.
    assert_eq!(rows[4]["(a + 1)"], 6);
    assert_eq!(rows[4]["(b - 2)"], 48);
    assert_eq!(rows[4]["((a + b) * 2)"], 110);
    assert_eq!(rows[4]["(a + 5000000000)"], 5000000005i64);
}

#[rstest]
fn length_counts_bytes(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "strs",
        &[
            (
                "i",
                Type::Int32,
                Arc::new(Int32Array::from(vec![0, 1, 2, 3])) as ArrayRef,
            ),
            (
                "s",
                Type::Utf8,
                Arc::new(StringViewArray::from(vec![
                    "hello",
                    "héllo",
                    "",
                    "日本語abc",
                ])) as ArrayRef,
            ),
        ],
    );

    let results = testing_planner
        .plan("SELECT i, length(s) FROM strs")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["i"].as_i64().unwrap());

    // length() counts bytes, not characters: "héllo" is 6 bytes (é is 2),
    // "日本語abc" is 12 bytes (three 3-byte CJK chars + "abc").
    let lengths: Vec<i64> = rows
        .iter()
        .map(|r| r["length(s)"].as_i64().unwrap())
        .collect();
    assert_eq!(lengths, vec![5, 6, 0, 12]);
}

#[rstest]
fn arithmetic_does_not_truncate_floats(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "mixed",
        &[
            (
                "f",
                Type::Float64,
                Arc::new(Float64Array::from(vec![1.5, 2.5, 3.5])) as ArrayRef,
            ),
            (
                "n",
                Type::Int32,
                Arc::new(Int32Array::from(vec![10, 20, 30])) as ArrayRef,
            ),
        ],
    );

    // `f` is a DOUBLE column, `n` an INTEGER column (a float *constant* isn't
    // supported, so both operands must be columns). DuckDB casts `n` to DOUBLE,
    // but the bridge unwraps that column cast, so the operands reach the kernel
    // with mismatched types (Float64 array vs Int32 array) and hit the coercion
    // branch. Coercing both to Int64 there would truncate `f` (1.5 -> 1) before
    // adding; the result must keep the fractional input.
    let results = testing_planner
        .plan("SELECT f + n FROM mixed")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by(|x, y| {
        only_column(x)
            .as_f64()
            .unwrap()
            .partial_cmp(&only_column(y).as_f64().unwrap())
            .unwrap()
    });

    // f ∈ {1.5, 2.5, 3.5} + n ∈ {10, 20, 30} → {11.5, 22.5, 33.5}; an Int64
    // coercion would truncate to {11.0, 22.0, 33.0}.
    let sums: Vec<f64> = rows
        .iter()
        .map(|r| only_column(r).as_f64().unwrap())
        .collect();
    assert_eq!(sums, vec![11.5, 22.5, 33.5]);
}

#[rstest]
fn filter_not_contains(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .plan("SELECT name FROM example_table WHERE NOT contains(name, 'a')")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);

    // Only "bob" lacks an 'a'; alice (x2), charlie and dave are negated away.
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["name"], "bob");
}

#[rstest]
fn regexp_replace_extracts_group(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "urls",
        &[
            (
                "i",
                Type::Int32,
                Arc::new(Int32Array::from(vec![0, 1, 2])) as ArrayRef,
            ),
            (
                "url",
                Type::Utf8,
                Arc::new(StringViewArray::from(vec![
                    "https://www.example.com/path/x",
                    "http://foo.org/x",
                    "no-match-here",
                ])) as ArrayRef,
            ),
        ],
    );

    let results = testing_planner
        .plan(r"SELECT i, regexp_replace(url, '^https?://(?:www\.)?([^/]+)/.*$', '\1') FROM urls")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let mut rows = batches_to_json(&results);
    rows.sort_by_key(|r| r["i"].as_i64().unwrap());

    // `\1` substitutes the captured group; a row without a match passes
    // through unchanged.
    assert_eq!(
        rows[0][r#"regexp_replace(url, '^https?://(?:www\.)?([^/]+)/.*$', '\1')"#],
        "example.com"
    );
    assert_eq!(
        rows[1][r#"regexp_replace(url, '^https?://(?:www\.)?([^/]+)/.*$', '\1')"#],
        "foo.org"
    );
    assert_eq!(
        rows[2][r#"regexp_replace(url, '^https?://(?:www\.)?([^/]+)/.*$', '\1')"#],
        "no-match-here"
    );
}

#[rstest]
fn regexp_replace_first_match_only(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "rep",
        &[(
            "s",
            Type::Utf8,
            Arc::new(StringViewArray::from(vec!["aaa"])) as ArrayRef,
        )],
    );

    let results = testing_planner
        .plan("SELECT regexp_replace(s, 'a', 'b') FROM rep")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);

    // Without the 'g' option only the first occurrence is replaced.
    assert_eq!(rows.len(), 1);
    assert_eq!(*only_column(&rows[0]), "baa");
}

#[rstest]
fn regexp_replace_dollar_is_literal(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "rep",
        &[(
            "s",
            Type::Utf8,
            Arc::new(StringViewArray::from(vec!["x"])) as ArrayRef,
        )],
    );

    // `$` (even `$1`) is a literal in the SQL replacement, but the regex crate's
    // replacement dialect would read `$1` as a group reference, so it must be
    // escaped to `$$` — otherwise the `$1` would be consumed instead of kept.
    let results = testing_planner
        .plan(r"SELECT regexp_replace(s, 'x', 'a$1b') FROM rep")
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
    assert_eq!(*only_column(&rows[0]), "a$1b");
}

#[rstest]
fn regexp_jit_replace_extracts_group(mut testing_planner: TestingPlanner) {
    testing_planner.add_table(
        "urls",
        &[(
            "url",
            Type::Utf8,
            Arc::new(StringViewArray::from(vec![
                "https://www.example.com/path/x",
                "http://foo.org/x",
                "no-match-here",
            ])) as ArrayRef,
        )],
    );

    let results = testing_planner
        .plan(r"SELECT regexp_jit_replace(url, '^https?://(?:www\.)?([^/]+)/.*', '\1') FROM urls")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let mut got = batches_to_json(&results)
        .iter()
        .map(|r| only_column(r).as_str().unwrap().to_string())
        .collect::<Vec<_>>();
    got.sort();

    assert_eq!(got, vec!["example.com", "foo.org", "no-match-here"]);
}

#[rstest]
fn regexp_full_match_invalid_pattern_returns_error(mut testing_planner: TestingPlanner) {
    let result = testing_planner.plan("SELECT regexp_full_match(name, '(') FROM example_table");

    // DuckDB compiles a constant pattern while binding, so an unparsable one is
    // rejected during planning rather than reaching pivot's own compile.
    assert!(result.is_err(), "expected a planning error");
}

/// A non-constant pattern can't be compiled once per plan, so it is refused
/// rather than silently re-compiled per row.
#[rstest]
fn regexp_full_match_non_constant_pattern_returns_error(mut testing_planner: TestingPlanner) {
    let result = testing_planner.plan("SELECT regexp_full_match(name, name) FROM example_table");

    assert!(matches!(result, Err(PlannerError::PlanConversion(_))));
}

#[rstest]
fn unsupported_scalar_function_returns_error(mut testing_planner: TestingPlanner) {
    let result = testing_planner.plan("SELECT lower(name) FROM example_table");
    assert!(matches!(result, Err(PlannerError::PlanConversion(_))));
}

// Aggregates over expressions. example_table: a=[1..5], b=[10,20,30,40,50],
// c=[100..500], name. `a * b` per row: 10, 40, 90, 160, 250.

#[rstest]
fn global_sum_over_product(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .plan("SELECT SUM(a * b) AS s FROM example_table")
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
    assert_eq!(rows[0]["s"], 550); // 10 + 40 + 90 + 160 + 250
}

#[rstest]
fn global_sum_over_nested_product(mut testing_planner: TestingPlanner) {
    // A nested expression argument: the whole `(a * b) * c` is one computed
    // argument, compiled (recursively) into a single materialised column.
    let results = testing_planner
        .plan("SELECT SUM((a * b) * c) AS s FROM example_table")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);

    // (1*10)*100 + (2*20)*200 + (3*30)*300 + (4*40)*400 + (5*50)*500.
    assert_eq!(rows[0]["s"], 225_000);
}

#[rstest]
fn global_min_max_over_product(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .plan("SELECT MIN(a * b) AS lo, MAX(a * b) AS hi FROM example_table")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);

    assert_eq!(rows[0]["lo"], 10);
    assert_eq!(rows[0]["hi"], 250);
}

#[rstest]
fn grouped_sum_over_product(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .plan("SELECT name, SUM(a * b) AS s FROM example_table GROUP BY name")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);

    let alice = rows.iter().find(|r| r["name"] == "alice").unwrap();
    assert_eq!(alice["s"], 260); // 1*10 + 5*50
    let bob = rows.iter().find(|r| r["name"] == "bob").unwrap();
    assert_eq!(bob["s"], 40); // 2*20
}

#[rstest]
fn aggregate_argument_mixed_with_plain_aggregate(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .plan("SELECT SUM(a * b) AS s, SUM(c) AS t, COUNT(*) AS n FROM example_table")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);

    assert_eq!(rows[0]["s"], 550);
    assert_eq!(rows[0]["t"], 1500); // 100 + 200 + 300 + 400 + 500
    assert_eq!(rows[0]["n"], 5);
}

#[rstest]
fn count_distinct_over_product(mut testing_planner: TestingPlanner) {
    let results = testing_planner
        .plan("SELECT COUNT(DISTINCT a * b) AS d FROM example_table")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);

    assert_eq!(rows[0]["d"], 5); // {10, 40, 90, 160, 250} all distinct
}

#[rstest]
fn grouped_count_distinct_over_product(mut testing_planner: TestingPlanner) {
    // alice's two rows give a*b of 10 and 250 (two distinct values).
    let results = testing_planner
        .plan("SELECT name, COUNT(DISTINCT a * b) AS d FROM example_table GROUP BY name")
        .unwrap()
        .compile(
            testing_planner.dispatcher(),
            testing_planner.transaction().as_ref(),
        )
        .unwrap()
        .collect()
        .unwrap();

    let rows = batches_to_json(&results);

    let alice = rows.iter().find(|r| r["name"] == "alice").unwrap();
    assert_eq!(alice["d"], 2);
    let bob = rows.iter().find(|r| r["name"] == "bob").unwrap();
    assert_eq!(bob["d"], 1);
}

#[rstest]
fn aggregate_with_a_filter_clause_is_refused(mut testing_planner: TestingPlanner) {
    let error = testing_planner
        .plan("SELECT COUNT(*) FILTER (WHERE a > 2) AS n FROM example_table")
        .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("count_star with a FILTER clause"),
        "unexpected error: {error}"
    );
}

#[rstest]
fn computed_group_key_with_computed_aggregate_argument(mut testing_planner: TestingPlanner) {
    // Group by a computed key (a bucket) and aggregate a computed argument: both
    // are materialised into leading columns before the aggregate.
    let results = testing_planner
        .plan(
            "SELECT CASE WHEN a <= 3 THEN 1 ELSE 0 END AS bucket, SUM(a * b) AS s \
             FROM example_table GROUP BY CASE WHEN a <= 3 THEN 1 ELSE 0 END",
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

    // bucket 1 (a in 1,2,3): 10 + 40 + 90 = 140; bucket 0 (a in 4,5): 160 + 250 = 410.
    let low = rows.iter().find(|r| r["bucket"] == 1).unwrap();
    assert_eq!(low["s"], 140);
    let high = rows.iter().find(|r| r["bucket"] == 0).unwrap();
    assert_eq!(high["s"], 410);
}

#[rstest]
fn struct_field_read_by_name(mut testing_planner: TestingPlanner) {
    let rows = run(
        &mut testing_planner,
        "SELECT pair.label AS label, pair.amount AS amount \
         FROM (SELECT {'label': name, 'amount': b} AS pair FROM example_table WHERE a = 2)",
    );

    assert_eq!(
        rows,
        vec![serde_json::json!({"label": "bob", "amount": 20})]
    );
}

#[rstest]
fn tuple_field_read_by_position(mut testing_planner: TestingPlanner) {
    let rows = run(
        &mut testing_planner,
        "SELECT pair[1] AS label, pair[2] AS amount \
         FROM (SELECT (name, b) AS pair FROM example_table WHERE a = 2)",
    );

    assert_eq!(
        rows,
        vec![serde_json::json!({"label": "bob", "amount": 20})]
    );
}

#[rstest]
#[case::comparison(
    "SELECT a FROM example_table WHERE {'a': a} = {'a': 2}",
    "A comparison is not supported on a struct"
)]
#[case::in_list(
    "SELECT a FROM example_table WHERE {'a': a} IN ({'a': 2}, {'a': 3})",
    "IN is not supported on a struct"
)]
#[case::cast_to_text(
    "SELECT CAST({'a': a} AS VARCHAR) FROM example_table",
    "A cast to another type is not supported on a struct"
)]
#[case::returned_to_the_client(
    "SELECT {'a': a} AS pair FROM example_table",
    "A query cannot return a struct"
)]
fn whole_struct_operations_are_rejected_when_planning(
    mut testing_planner: TestingPlanner,
    #[case] sql: &str,
    #[case] expected_message: &str,
) {
    let error = testing_planner.plan(sql).unwrap_err();

    assert!(error.to_string().contains(expected_message), "{error}");
}
