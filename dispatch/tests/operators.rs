//! Single-operator integration tests, fed from in-memory `RecordBatch`es via
//! [`values_input`] rather than a Parquet scan: the batch is the source, so
//! these exercise operator behavior (count/filter/project/order-by/group-by)
//! without involving goose's reader.

mod common;

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Decimal128Type, Int64Type};
use arrow_array::{
    BooleanArray, Decimal128Array, Float64Array, Int64Array, RecordBatch, StringViewArray,
};
use arrow_buffer::BooleanBuffer;
use arrow_schema::{DataType, Field, Schema};

use common::*;
use dispatch::{
    AggregationKind, AggregationSlot, Compiled, Contains, CountSlot, Dynamic, IntKeyExtractor,
    OrderBy, RowKeyExtractor, RowKeySchema, StringKeyExtractor, values_input,
};

#[test]
fn count() {
    let dispatch = dispatch(1);
    let batch = strings_and_ints(&["a", "b", "c", "d", "e"], &[1, 2, 3, 4, 5]);

    let results = values_input(&dispatch, vec![batch])
        .record_batches()
        .aggregate::<i64>(vec![AggregationSlot::new(
            AggregationKind::CountStar,
            0,
            DataType::Int64,
        )])
        .collect()
        .unwrap();

    assert_eq!(extract_count(&results), 5);
}

#[test]
fn filter_string_contains() {
    let dispatch = dispatch(1);
    let batch = strings_and_ints(
        &["alice", "bob", "alice", "dave", "alice"],
        &[1, 2, 3, 4, 5],
    );

    let results = values_input(&dispatch, vec![batch])
        .record_batches()
        .filter(|| {
            let mut c = Contains::new("alice");
            move |batch: &RecordBatch,
                  _: &mut dispatch::memory::SlabAllocator,
                  scratch: &mut dispatch::SelectionScratch| {
                let mask = c.run(
                    batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<StringViewArray>()
                        .unwrap(),
                );
                dispatch::collect_selected_indices(&mask, &mut scratch.positions);
                dispatch::RowSelection::Indices
            }
        })
        .aggregate::<i64>(vec![AggregationSlot::new(
            AggregationKind::CountStar,
            0,
            DataType::Int64,
        )])
        .collect()
        .unwrap();

    assert_eq!(extract_count(&results), 3);
}

#[test]
fn filter_no_matches_returns_zero() {
    let dispatch = dispatch(1);
    let batch = strings_and_ints(&["alice", "bob", "carol"], &[1, 2, 3]);

    let results = values_input(&dispatch, vec![batch])
        .record_batches()
        .filter(|| {
            let mut c = Contains::new("zzz_no_match");
            move |batch: &RecordBatch,
                  _: &mut dispatch::memory::SlabAllocator,
                  scratch: &mut dispatch::SelectionScratch| {
                let mask = c.run(
                    batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<StringViewArray>()
                        .unwrap(),
                );
                dispatch::collect_selected_indices(&mask, &mut scratch.positions);
                dispatch::RowSelection::Indices
            }
        })
        .aggregate::<i64>(vec![AggregationSlot::new(
            AggregationKind::CountStar,
            0,
            DataType::Int64,
        )])
        .collect()
        .unwrap();

    assert_eq!(extract_count(&results), 0);
}

#[test]
fn filter_integer_column() {
    let dispatch = dispatch(1);
    let batch = strings_and_ints(&["a", "b", "c", "d", "e"], &[5, 15, 25, 35, 45]);

    let results = values_input(&dispatch, vec![batch])
        .record_batches()
        .filter(|| {
            move |batch: &RecordBatch,
                  _: &mut dispatch::memory::SlabAllocator,
                  scratch: &mut dispatch::SelectionScratch| {
                let col = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();
                let mask = BooleanArray::from(BooleanBuffer::collect_bool(col.len(), |i| {
                    col.value(i) > 20
                }));
                dispatch::collect_selected_indices(&mask, &mut scratch.positions);
                dispatch::RowSelection::Indices
            }
        })
        .aggregate::<i64>(vec![AggregationSlot::new(
            AggregationKind::CountStar,
            0,
            DataType::Int64,
        )])
        .collect()
        .unwrap();

    assert_eq!(extract_count(&results), 3);
}

#[test]
fn project_selects_single_column() {
    let dispatch = dispatch(1);
    let batch = strings_and_ints(&["a", "b"], &[10, 20]);

    let results = values_input(&dispatch, vec![batch])
        .record_batches()
        .project(|| {
            let idx = vec![1];
            move |batch: RecordBatch| batch.project(&idx).unwrap()
        })
        .collect()
        .unwrap();

    assert_eq!(results[0].num_columns(), 1);
    let mut vals = collect_i64s(&results, 0);
    vals.sort();
    assert_eq!(vals, vec![10, 20]);
}

#[test]
fn order_by_ascending_with_limit() {
    let dispatch = dispatch(1);
    let batch = strings_and_ints(&["e", "a", "d", "b", "c"], &[50, 10, 40, 20, 30]);

    let results = values_input(&dispatch, vec![batch])
        .record_batches()
        .order_by_limit(vec![OrderBy::new(1, false, false)], 3)
        .collect()
        .unwrap();

    assert_eq!(collect_i64s(&results, 1), vec![10, 20, 30]);
    assert_eq!(collect_strings(&results, 0), vec!["a", "b", "c"]);
}

#[test]
fn order_by_descending_with_limit() {
    let dispatch = dispatch(1);
    let batch = strings_and_ints(&["a", "b", "c", "d"], &[10, 40, 20, 30]);

    let results = values_input(&dispatch, vec![batch])
        .record_batches()
        .order_by_limit(vec![OrderBy::new(1, true, false)], 2)
        .collect()
        .unwrap();

    assert_eq!(collect_i64s(&results, 1), vec![40, 30]);
}

#[test]
fn order_by_limit_exceeds_row_count() {
    let dispatch = dispatch(1);
    let batch = strings_and_ints(&["c", "a", "b"], &[30, 10, 20]);

    let results = values_input(&dispatch, vec![batch])
        .record_batches()
        .order_by_limit(vec![OrderBy::new(1, false, false)], 100)
        .collect()
        .unwrap();

    assert_eq!(collect_i64s(&results, 1), vec![10, 20, 30]);
}

#[test]
fn order_by_without_limit_sorts_the_whole_stream() {
    let dispatch = dispatch(2);
    let batches = vec![
        strings_and_ints(&["nine", "one", "five"], &[9, 1, 5]),
        strings_and_ints(&["eight", "two", "four"], &[8, 2, 4]),
    ];

    let results = values_input(&dispatch, batches)
        .record_batches()
        .order_by(vec![OrderBy::new(1, false, false)])
        .collect()
        .unwrap();

    assert_eq!(collect_i64s(&results, 1), vec![1, 2, 4, 5, 8, 9]);
    assert_eq!(
        collect_strings(&results, 0),
        vec!["one", "two", "four", "five", "eight", "nine"]
    );
}

#[test]
fn ordered_batches_stay_ordered_through_projection_and_copy_out() {
    let dispatch = dispatch(1);
    let batches = vec![
        strings_and_ints(&["one", "two"], &[1, 2]),
        strings_and_ints(&["three", "four"], &[3, 4]),
    ];

    let results = values_input(&dispatch, batches)
        .record_batches()
        .order_by(vec![OrderBy::new(1, false, false)])
        .project(|| |batch| batch)
        .collect()
        .unwrap();

    assert_eq!(collect_i64s(&results, 1), vec![1, 2, 3, 4]);
}

#[test]
fn group_by_count_string_keys() {
    let dispatch = dispatch(1);
    let batch = strings_and_ints(
        &["alice", "bob", "alice", "carol", "bob", "alice"],
        &[1, 2, 3, 4, 5, 6],
    );

    let results = values_input(&dispatch, vec![batch])
        .record_batches()
        .group_by_aggregate::<StringKeyExtractor, Compiled<(CountSlot,)>>(
            vec![0],
            vec![AggregationSlot::new(
                AggregationKind::CountStar,
                0,
                DataType::Int64,
            )],
            None,
            (),
        )
        .order_by_limit(vec![OrderBy::new(1, true, false)], 10)
        .collect()
        .unwrap();

    assert_eq!(collect_strings(&results, 0), vec!["alice", "bob", "carol"]);
    assert_eq!(collect_i64s(&results, 1), vec![3, 2, 1]);
}

#[test]
fn group_by_count_int_keys() {
    let dispatch = dispatch(1);
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("cat", DataType::Int64, false)])),
        vec![Arc::new(Int64Array::from(vec![1, 2, 1, 3, 2, 1, 3, 2]))],
    )
    .unwrap();

    let results = values_input(&dispatch, vec![batch])
        .record_batches()
        .group_by_aggregate::<IntKeyExtractor<Int64Type>, Compiled<(CountSlot,)>>(
            vec![0],
            vec![AggregationSlot::new(
                AggregationKind::CountStar,
                0,
                DataType::Int64,
            )],
            None,
            (),
        )
        .order_by_limit(vec![OrderBy::new(1, true, false)], 10)
        .collect()
        .unwrap();

    assert_eq!(collect_i64s(&results, 1), vec![3, 3, 2]);
}

#[test]
fn group_by_sum_float_values() {
    let dispatch = dispatch(1);
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("cat", DataType::Int64, false),
            Field::new("v", DataType::Float64, false),
        ])),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 1, 2, 1])),
            Arc::new(Float64Array::from(vec![1.5, 2.0, 0.5, 3.0, 1.0])),
        ],
    )
    .unwrap();

    let results = values_input(&dispatch, vec![batch])
        .record_batches()
        .group_by_aggregate::<IntKeyExtractor<Int64Type>, Dynamic<i64, false>>(
            vec![0],
            vec![AggregationSlot::new(
                AggregationKind::Sum,
                1,
                DataType::Float64,
            )],
            None,
            (),
        )
        .order_by_limit(vec![OrderBy::new(0, false, false)], 10)
        .collect()
        .unwrap();

    // group 1: 1.5 + 0.5 + 1.0; group 2: 2.0 + 3.0.
    assert_eq!(collect_f64s(&results, 1), vec![3.0, 5.0]);
}

#[test]
fn group_by_sum_decimal128_does_not_clip_to_i64() {
    // A per-group SUM over a Decimal128 column (the wide re-read path of a two-level
    // COUNT(DISTINCT)) must keep the full i128, not truncate to i64.
    let dispatch = dispatch(1);
    let big = i64::MAX as i128 + 1000; // exceeds i64::MAX
    let values = Decimal128Array::from(vec![big, 5])
        .with_precision_and_scale(38, 0)
        .unwrap();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("g", DataType::Int64, false),
            Field::new("v", DataType::Decimal128(38, 0), false),
        ])),
        vec![Arc::new(Int64Array::from(vec![1i64, 1])), Arc::new(values)],
    )
    .unwrap();

    let results = values_input(&dispatch, vec![batch])
        .record_batches()
        .group_by_aggregate::<IntKeyExtractor<Int64Type>, Dynamic<i128, false>>(
            vec![0],
            vec![AggregationSlot::new(
                AggregationKind::Sum,
                1,
                DataType::Decimal128(38, 0),
            )],
            None,
            (),
        )
        .collect()
        .unwrap();

    let sum = results[0].column(1).as_primitive::<Decimal128Type>();
    assert_eq!(sum.value(0), big + 5);
}

// A plain `LIMIT` must abandon its upstream once satisfied, not drain the whole
// input. A counting filter sits *beneath* `LIMIT 1` while we feed 10k batches:
// with early termination the filter runs only a handful of times before the
// scan is torn down; without it, it would run once per batch.
#[test]
fn limit_abandons_upstream_after_reaching_count() {
    use std::sync::atomic::{AtomicUsize, Ordering};

    let dispatch = dispatch(1);
    let batches: Vec<RecordBatch> = (0..10_000)
        .map(|i| strings_and_ints(&["x"], &[i]))
        .collect();
    let consumed = Arc::new(AtomicUsize::new(0));

    let counter = consumed.clone();
    let results = values_input(&dispatch, batches)
        .record_batches()
        .filter(move || {
            let counter = counter.clone();
            move |_: &RecordBatch,
                  _: &mut dispatch::memory::SlabAllocator,
                  _: &mut dispatch::SelectionScratch| {
                counter.fetch_add(1, Ordering::Relaxed);
                dispatch::RowSelection::All
            }
        })
        .limit(1, 0)
        .collect()
        .unwrap();

    let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
    let filtered = consumed.load(Ordering::Relaxed);
    assert_eq!(total_rows, 1, "LIMIT 1 emits exactly one row");
    assert!(
        filtered < 100,
        "filter ran {filtered} times for LIMIT 1 over 10k batches: upstream was not abandoned",
    );
}

// Multi-worker LIMIT with far more workers than input batches: most workers get
// no row group and park while one produces the result and trips the limit. The
// worker that latches the limit must wake the parked peers so they abandon their
// own (now-unneeded) upstream and flush, or the assembler waits forever. Run
// across several worker counts to shake out that lost-wakeup.
#[test]
fn limit_with_idle_parked_workers_completes() {
    for workers in [2, 4, 8, 16] {
        let dispatch = dispatch(workers);
        let batches: Vec<RecordBatch> = (0..3).map(|i| strings_and_ints(&["x"], &[i])).collect();

        let results = values_input(&dispatch, batches)
            .record_batches()
            .limit(1, 0)
            .collect()
            .unwrap();

        let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
        assert_eq!(
            total_rows, 1,
            "LIMIT 1 with {workers} workers must emit one row"
        );
    }
}

#[test]
fn group_by_count_row_key_with_string_field() {
    let dispatch = dispatch(1);
    let batch = strings_and_ints(
        &["alice", "bob", "alice", "carol", "bob", "alice"],
        &[10, 20, 10, 40, 20, 10],
    );

    let results = values_input(&dispatch, vec![batch])
        .record_batches()
        .group_by_aggregate::<RowKeyExtractor, Compiled<(CountSlot,)>>(
            vec![0, 1],
            vec![AggregationSlot::new(
                AggregationKind::CountStar,
                0,
                DataType::Int64,
            )],
            None,
            RowKeySchema::new(
                vec![DataType::Utf8View, DataType::Int64],
                vec![false, false],
            ),
        )
        .order_by_limit(vec![OrderBy::new(2, true, false)], 10)
        .collect()
        .unwrap();

    assert_eq!(collect_strings(&results, 0), vec!["alice", "bob", "carol"]);
    assert_eq!(collect_i64s(&results, 1), vec![10, 20, 40]);
    assert_eq!(collect_i64s(&results, 2), vec![3, 2, 1]);
}
