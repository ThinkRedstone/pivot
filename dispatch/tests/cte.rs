//! CTE integration tests: one chain produces rows, several read all of them.

mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};

use common::*;
use dispatch::{DataFlowDispatcher, JoinKind, JoinSpec, RecordBatchOperatorSpec, values_input};

/// The CTE index the planner would carry over from DuckDB.
const CTE: usize = 0;

fn int64_batch(values: &[i64]) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)])),
        vec![Arc::new(Int64Array::from(values.to_vec()))],
    )
    .unwrap()
}

fn inner_join_on_id() -> JoinSpec {
    JoinSpec {
        probe_key_indices: vec![0],
        build_key_indices: vec![0],
        probe_output_indices: vec![0],
        build_output_indices: vec![0],
        probe_fields: vec![Field::new("id", DataType::Int64, false)],
        build_fields: vec![Field::new("id", DataType::Int64, false)],
        kind: JoinKind::Inner,
        residual_filters: None,
        build_filters: Vec::new(),
    }
}

/// A CTE read by two sites, joined back to itself on its only column. Every id
/// matches exactly one id, so a site that saw a share of the rows rather than
/// all of them shows up as missing output rows.
fn self_join_over_cte(
    dispatcher: &DataFlowDispatcher,
    batches: Vec<RecordBatch>,
) -> Vec<RecordBatch> {
    let definition = values_input(dispatcher, batches).record_batches();

    let probe = RecordBatchOperatorSpec::cte_scan(dispatcher, CTE);
    let build = RecordBatchOperatorSpec::cte_scan(dispatcher, CTE);

    probe
        .join(build, &[DataType::Int64], inner_join_on_id())
        .with_cte(definition, CTE, 2)
        .collect()
        .unwrap()
}

#[test]
fn every_scan_site_reads_the_whole_cte() {
    let d = dispatch(4);
    let batches: Vec<_> = (0..8).map(|b| int64_batch(&[b * 2, b * 2 + 1])).collect();

    let results = self_join_over_cte(&d, batches);

    let mut ids = collect_i64s(&results, 0);
    ids.sort();
    assert_eq!(ids, (0..16).collect::<Vec<_>>());
}

/// A definition that produces nothing still has to release its sites, which
/// would otherwise wait for rows that are never coming.
#[test]
fn an_empty_cte_finishes_its_sites() {
    let d = dispatch(4);

    let results = self_join_over_cte(&d, vec![]);

    assert_eq!(collect_i64s(&results, 0), Vec::<i64>::new());
}

/// A definition slow enough that its sites are asked to finish while their
/// queues are empty and the rows are still coming. Answering on the queue alone
/// would let a site sign off there, and the join above it would run on the rows
/// that had made it through by then.
#[test]
fn a_site_waits_out_a_lull_in_the_definition() {
    let d = dispatch(4);
    let batches: Vec<_> = (0..8).map(|b| int64_batch(&[b])).collect();
    let definition = values_input(&d, batches).record_batches().project(|| {
        move |batch: RecordBatch| {
            std::thread::sleep(std::time::Duration::from_millis(20));
            Ok(batch)
        }
    });

    let results = RecordBatchOperatorSpec::cte_scan(&d, CTE)
        .join(
            RecordBatchOperatorSpec::cte_scan(&d, CTE),
            &[DataType::Int64],
            inner_join_on_id(),
        )
        .with_cte(definition, CTE, 2)
        .collect()
        .unwrap();

    let mut ids = collect_i64s(&results, 0);
    ids.sort();
    assert_eq!(ids, (0..8).collect::<Vec<_>>());
}

/// A satisfied `LIMIT` frees the stages upstream of it, and one of those is the
/// CTE, which is also feeding somebody else. The other site still has to see
/// every row: the `LIMIT` speaks for its own branch only.
#[test]
fn a_limit_over_one_site_spares_the_cte_its_other_readers_need() {
    let d = dispatch(4);
    let batches: Vec<_> = (0..64).map(|b| int64_batch(&[b])).collect();
    // Slow enough that the limit is satisfied, and asks for its upstream to be
    // torn down, while the definition is still producing.
    let definition = values_input(&d, batches).record_batches().project(|| {
        move |batch: RecordBatch| {
            std::thread::sleep(std::time::Duration::from_millis(20));
            Ok(batch)
        }
    });

    let seen = Arc::new(AtomicUsize::new(0));
    let counted = RecordBatchOperatorSpec::cte_scan(&d, CTE).project({
        let seen = seen.clone();
        move || {
            let seen = seen.clone();
            move |batch: RecordBatch| {
                seen.fetch_add(batch.num_rows(), Ordering::Relaxed);
                Ok(batch)
            }
        }
    });

    counted
        .join(
            RecordBatchOperatorSpec::cte_scan(&d, CTE).limit(2, 0),
            &[DataType::Int64],
            inner_join_on_id(),
        )
        .with_cte(definition, CTE, 2)
        .collect()
        .unwrap();

    assert_eq!(seen.load(Ordering::Relaxed), 64);
}
