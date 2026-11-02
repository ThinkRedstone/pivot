//! Tests for the in-memory values source ([`values_input`]) and the parallel
//! [`map_each`](dispatch::OperatorSpec::map_each) stage: a fixed `Vec<T>` fans
//! out across the worker pool and each item is transformed independently.

mod common;

use common::*;
use dispatch::values_input;

/// A worker count that exercises fan-out without exceeding the core count.
fn worker_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(4)
}

#[test]
fn map_each_transforms_every_item() {
    let dispatch = dispatch(worker_count());

    let mut doubled = values_input(&dispatch, 0..1000i64)
        .map_each(|x| Ok(x * 2))
        .collect()
        .unwrap();
    doubled.sort_unstable();

    assert_eq!(doubled, (0..1000i64).map(|x| x * 2).collect::<Vec<_>>());
}

#[test]
fn empty_input_produces_no_output() {
    let dispatch = dispatch(worker_count());

    let out: Vec<i64> = values_input(&dispatch, Vec::<i64>::new())
        .map_each(Ok)
        .collect()
        .unwrap();

    assert!(out.is_empty());
}
