//! Microbenchmark of the two ways a selection of a batch's rows reaches a
//! [`BatchAccumulator`]: listed as positions and gathered row by row, or
//! handed over as the mask the selection was computed as, which the
//! accumulator copies in runs where the rows come in runs.
//!
//! The selectivities span the shapes that decide which copy the mask append
//! picks: one and ten in a hundred kept at random, the selective filter, where
//! runs are single rows and both ways gather a few; half the rows, where runs
//! are a row or two long and both ways gather; three in four, where runs average four rows and the
//! choice is close; nine in ten and ninety-nine in a hundred, where the runs
//! grow long enough to copy as pieces; and ninety-nine in a hundred dropped
//! in one block, the shape of a deletion vector over rows written together.
//! The position form always pays the walk that lists the positions, so the
//! comparison is of the whole cost of each form, not of the copy alone.
//!
//! ```sh
//! cargo bench -p dispatch --features test-util --bench accumulator
//! ```

use std::sync::Arc;

use arrow_array::{Float64Array, Int32Array, Int64Array, RecordBatch, StringViewArray};
use arrow_buffer::BooleanBuffer;
use arrow_schema::{DataType, Field, Schema};
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use dispatch::RECORD_BATCH_SIZE;
use dispatch::arrays::accumulator::BatchAccumulator;
use dispatch::memory::{SlabAllocator, init_test_free_pool};

/// A deterministic bit source, so every run selects the same rows.
struct Xorshift(u64);

impl Xorshift {
    fn next_below(&mut self, bound: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % bound
    }
}

/// One batch of the ids, measures and strings a scan emits: wide enough for
/// the copy to dominate the walk over the mask.
fn mixed_batch(rows: usize) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("count", DataType::Int32, false),
            Field::new("measure", DataType::Float64, false),
            Field::new("code", DataType::Utf8View, false),
            Field::new("title", DataType::Utf8View, false),
        ])),
        vec![
            Arc::new(Int64Array::from_iter_values(0..rows as i64)),
            Arc::new(Int32Array::from_iter_values(
                (0..rows).map(|row| (row % 1000) as i32),
            )),
            Arc::new(Float64Array::from_iter_values(
                (0..rows).map(|row| row as f64 * 0.5),
            )),
            Arc::new(StringViewArray::from_iter_values(
                (0..rows).map(|row| format!("c{:06}", row % 5000)),
            )),
            Arc::new(StringViewArray::from_iter_values((0..rows).map(|row| {
                format!("a title long enough to live in a data buffer, {row}")
            }))),
        ],
    )
    .unwrap()
}

/// One narrow batch, where listing a position costs as much as copying the
/// row it names.
fn narrow_batch(rows: usize) -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)])),
        vec![Arc::new(Int64Array::from_iter_values(0..rows as i64))],
    )
    .unwrap()
}

/// A mask keeping each row with probability `kept_percent`.
fn random_mask(rows: usize, kept_percent: u64) -> BooleanBuffer {
    let mut bits = Xorshift(0x9E37_79B9_7F4A_7C15);
    BooleanBuffer::from_iter((0..rows).map(|_| bits.next_below(100) < kept_percent))
}

/// A mask dropping one block of rows, `dropped_percent` of the batch.
fn block_mask(rows: usize, dropped_percent: usize) -> BooleanBuffer {
    let dropped = rows * dropped_percent / 100;
    let start = rows / 2;
    BooleanBuffer::from_iter((0..rows).map(|row| !(start..start + dropped).contains(&row)))
}

fn bench_appends(c: &mut Criterion) {
    init_test_free_pool(64);
    let rows = RECORD_BATCH_SIZE;
    let masks = [
        ("random_1", random_mask(rows, 1)),
        ("random_10", random_mask(rows, 10)),
        ("random_50", random_mask(rows, 50)),
        ("random_75", random_mask(rows, 75)),
        ("random_90", random_mask(rows, 90)),
        ("random_99", random_mask(rows, 99)),
        ("block_99", block_mask(rows, 1)),
    ];
    for (shape, batch) in [("narrow", narrow_batch(rows)), ("mixed", mixed_batch(rows))] {
        let mut group = c.benchmark_group(format!("append/{shape}"));
        group.throughput(Throughput::Elements(rows as u64));
        for (name, mask) in &masks {
            let mut allocator = SlabAllocator::new(false);
            let mut accumulator = BatchAccumulator::new(batch.schema(), &mut allocator);
            let mut indices = Vec::with_capacity(rows);
            group.bench_with_input(BenchmarkId::new("indices", name), mask, |b, mask| {
                b.iter(|| {
                    indices.clear();
                    indices.extend(mask.set_indices_u32());
                    accumulator.append_batch_by_indices(&batch, &indices, &mut allocator);
                    accumulator.take_batch(&mut allocator).unwrap()
                })
            });
            group.bench_with_input(BenchmarkId::new("mask", name), mask, |b, mask| {
                b.iter(|| {
                    accumulator.append_batch_by_mask(&batch, mask, &mut allocator);
                    accumulator.take_batch(&mut allocator).unwrap()
                })
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench_appends);
criterion_main!(benches);
