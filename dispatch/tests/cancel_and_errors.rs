//! Error- and cancellation-path tests, fed from in-memory `RecordBatch`es via
//! [`values_input`] instead of a Parquet scan. The cancel test splits its rows
//! across many batches so each is a separate work item — cancellation is
//! checked between items, so a partially-drained query can still stop early.

mod common;

use arrow_array::{BooleanArray, Int64Array, RecordBatch};
use arrow_buffer::BooleanBuffer;
use arrow_schema::DataType;

use common::*;
use dispatch::{AggregationKind, AggregationSlot, values_input};

/// A panicking filter on every batch should surface as an error from
/// `.collect()`, not a silent empty result.
#[test]
fn panic_in_filter_returns_error() {
    let dispatch = dispatch(1);
    let names: Vec<&str> = (0..1000).map(|_| "x").collect();
    let values: Vec<i64> = (0..1000).collect();
    let batch = strings_and_ints(&names, &values);

    let result = values_input(&dispatch, vec![batch])
        .record_batches()
        .filter(|| {
            move |_: &RecordBatch,
                  _: &mut dispatch::memory::SlabAllocator,
                  _: &mut dispatch::SelectionScratch|
                  -> dispatch::RowSelection { panic!("intentional panic in filter") }
        })
        .aggregate::<i64>(vec![AggregationSlot::new(
            AggregationKind::CountStar,
            0,
            DataType::Int64,
        )])
        .collect();

    // Assert: collect() returned an error mentioning the panic.
    let err = result.expect_err("expected an error from a panicking filter");
    let msg = err.to_string();
    assert!(
        msg.contains("intentional panic in filter"),
        "error message should surface the panic; got: {msg}"
    );
}

/// Cancelling a query before draining its output should:
///   1. not hang `.collect()`,
///   2. not return an error (cancel is not a failure mode),
///   3. return at most the full result, often less.
#[test]
fn cancelled_query_returns_without_hanging() {
    // Setup: enough batches that finishing in one tick is unlikely. Each batch
    // is a separate work item, so cancellation can take effect between them.
    let dispatch = dispatch(1);
    let n: i64 = 200_000;
    let chunk = 8192;
    let batches: Vec<RecordBatch> = (0..n)
        .step_by(chunk)
        .map(|start| {
            let end = (start + chunk as i64).min(n);
            let names: Vec<&str> = (start..end).map(|_| "x").collect();
            let values: Vec<i64> = (start..end).collect();
            strings_and_ints(&names, &values)
        })
        .collect();

    // Execute: kick off the query, immediately cancel, then drain the handle.
    let handle = values_input(&dispatch, batches)
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
                let mask = BooleanArray::from(BooleanBuffer::collect_bool(col.len(), |_| true));
                dispatch::collect_selected_indices(&mask, &mut scratch.positions);
                dispatch::RowSelection::Indices
            }
        })
        .execute_as::<RecordBatch>();
    handle.cancel();
    let result = handle.collect();

    // Assert: collect succeeded and returned no more rows than we fed in.
    let batches = result.expect("cancelled query should not return an error");
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert!(
        total <= n as usize,
        "cancelled query returned more rows ({total}) than the input ({n})"
    );
}
