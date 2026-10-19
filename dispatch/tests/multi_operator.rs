//! Multi-stage operator chains fed from in-memory `RecordBatch`es via
//! [`values_input`]: filter→project, filter→count, and
//! filter→group_by→order_by, all without a Parquet scan.

mod common;

use arrow_array::{Array, BooleanArray, RecordBatch, StringViewArray};
use arrow_buffer::BooleanBuffer;
use arrow_schema::DataType;

use common::*;
use dispatch::{
    AggregationKind, AggregationSlot, Compiled, Contains, CountSlot, OrderBy, StringKeyExtractor,
    values_input,
};

#[test]
fn filter_then_project() {
    let dispatch = dispatch(1);
    let batch = strings_and_ints(&["alice", "bob", "alice"], &[100, 200, 300]);

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
        .project(|| {
            let idx = vec![1];
            move |batch: RecordBatch| batch.project(&idx).unwrap()
        })
        .collect()
        .unwrap();

    let mut vals = collect_i64s(&results, 0);
    vals.sort();
    assert_eq!(vals, vec![100, 300]);
}

#[test]
fn filter_then_count() {
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
fn filter_then_group_by_then_order_by() {
    let dispatch = dispatch(1);
    let batch = strings_and_ints(
        &[
            "google.com",
            "apple.com",
            "google.com",
            "google.com",
            "meta.com",
            "apple.com",
            "google.com",
            "meta.com",
            "other.com",
        ],
        &[0, 1, 2, 3, 4, 5, 6, 7, 8],
    );

    let results = values_input(&dispatch, vec![batch])
        .record_batches()
        .filter(|| {
            move |batch: &RecordBatch,
                  _: &mut dispatch::memory::SlabAllocator,
                  scratch: &mut dispatch::SelectionScratch| {
                let col = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<StringViewArray>()
                    .unwrap();
                let mask = BooleanArray::from(BooleanBuffer::collect_bool(col.len(), |i| {
                    col.value(i) != "other.com"
                }));
                dispatch::collect_selected_indices(&mask, &mut scratch.positions);
                dispatch::RowSelection::Indices
            }
        })
        .group_by_aggregate::<StringKeyExtractor, Compiled<(CountSlot,), u8>>(
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

    let keys = collect_strings(&results, 0);
    let counts = collect_i64s(&results, 1);
    assert_eq!(keys[0], "google.com");
    assert_eq!(counts, vec![4, 2, 2]);
}
