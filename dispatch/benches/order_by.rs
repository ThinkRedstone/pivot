//! Benchmarks for the full ORDER BY operator (no LIMIT) over large inputs.
//!
//! The `order_by/*_limit` scenarios in `operators.rs` measure the bounded
//! top-N operator, whose working set is a few rows per worker. Here the whole
//! input is sorted and re-emitted, so run formation, the merge phases and the
//! per-column gather all run at input scale. Every scenario streams at least
//! `PIVOT_BENCH_SORT_MB` (default 1024, i.e. 1 GiB) of Arrow data through a
//! single `order_by` dataflow: large enough that the sort works far outside
//! cache and the merge output is written batch by batch across the ring.
//! Output batches are dropped on their producing workers (only row counts
//! leave the dataflow), so result delivery and the ring-to-heap output copy
//! are excluded: the numbers isolate the sort operator itself.
//!
//! # Scenarios
//!
//! * `i64_random` - an `i64` key drawn from the whole 64-bit range plus one
//!   `i64` payload (16 B/row). Key comparisons are single integer compares,
//!   so run sorting and merge traversal dominate.
//! * `i64_presorted` - the same shape with globally ascending keys. Batches
//!   arrive already ordered and extend runs untouched; this is the ordered
//!   input fast path and the cost floor of re-emitting the input.
//! * `i64_wide_payload` - the `i64` key with four `i64` payload columns and a
//!   ~32 B string payload column. Compare cost matches `i64_random`, but each
//!   reordered row drags ~88 B through the gather, so take/scatter dominates.
//! * `string_random` - a ~32 B `Utf8View` key (dictionary-shared views whose
//!   bytes diverge early) plus one `i64` payload; byte-wise key comparisons
//!   dominate.
//!
//! Scenario sizes count logical row bytes (fixed widths; views plus payload
//! bytes for strings) independent of dictionary sharing, and generation stops
//! at the first batch to cross the target, so reported bytes/s throughput is
//! against the target size and accurate to within one batch. The string
//! scenarios' input is dictionary-shared, so its resident footprint is below
//! the logical size; the merge gather still materializes every payload byte,
//! which is what the logical denominator tracks.
//!
//! # Running
//!
//! ```sh
//! # All scenarios:
//! cargo bench --bench order_by
//!
//! # A/B across two builds of the operator (run, rebuild, run):
//! cargo bench --bench order_by -- --save-baseline before
//! cargo bench --bench order_by -- --baseline before
//! ```
//!
//! Tunables (env): `PIVOT_BENCH_SORT_MB` (MiB of data per scenario, default
//! 1024), `PIVOT_BENCH_WORKERS` (default: all cores), `PIVOT_BENCH_BUFFERS`
//! (ring slots of 2 MiB each; defaults to 16x the scenario size, i.e. 8192
//! slots at the 1 GiB default).
//!
//! Sizing the ring: the sorted output and the intermediate merge runs live on
//! ring buffers, so the ring must hold several multiples of the input size.
//! An undersized ring does not degrade gracefully: a worker aborts with an
//! `Evicting` panic (8x the input has been seen to run out; the 16x default
//! has not). The default tracks `PIVOT_BENCH_SORT_MB`, so override
//! `PIVOT_BENCH_BUFFERS` only to probe headroom, and keep `buffers x 2 MiB`
//! (the ring is one pre-faulted mmap; the caches share its slots) plus the
//! generated input under box RAM.

use std::sync::Arc;

use arrow_array::builder::StringViewBuilder;
use arrow_array::{Array, ArrayRef, Int64Array, RecordBatch, StringViewArray};
use arrow_schema::{DataType, Field, Schema};
use criterion::measurement::WallTime;
use criterion::{BatchSize, BenchmarkGroup, Criterion, SamplingMode, Throughput};

use dispatch::{BUFFER_SIZE, DataFlowDispatcher, Dispatch, OrderBy, memory_ctx, values_input};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Rows per `RecordBatch`. Mirrors a scan morsel: many independent batches fan
/// out across workers through the work-stealing injector.
const BATCH_ROWS: usize = 64 * 1024;

/// Ring slots per byte of scenario data: the sorted output and the
/// intermediate merge runs both live on ring buffers, and 8x the input has
/// been seen to run out (an `Evicting` worker panic), so default to 16x.
const RING_HEADROOM: usize = 16;

/// Read an env override, falling back to `default` only when the variable is
/// unset. A set-but-unparseable value aborts rather than silently reverting,
/// so a typo cannot benchmark a different configuration than the one asked
/// for.
fn read_env_usize(key: &str, default: usize) -> usize {
    match std::env::var(key) {
        Ok(value) => value
            .parse()
            .unwrap_or_else(|_| panic!("{key} must be an integer, got {value:?}")),
        Err(_) => default,
    }
}

/// Bytes of Arrow data each scenario streams through the sort.
fn target_sort_bytes() -> usize {
    let mb = read_env_usize("PIVOT_BENCH_SORT_MB", 1024);
    assert!(mb > 0, "PIVOT_BENCH_SORT_MB must be at least 1");
    mb * (1 << 20)
}

fn worker_count() -> usize {
    read_env_usize(
        "PIVOT_BENCH_WORKERS",
        core_affinity::get_core_ids().map(|c| c.len()).unwrap_or(1),
    )
}

fn ring_buffers() -> usize {
    read_env_usize(
        "PIVOT_BENCH_BUFFERS",
        target_sort_bytes() * RING_HEADROOM / BUFFER_SIZE,
    )
}

// ---------------------------------------------------------------------------
// Deterministic PRNG (splitmix64), so both sides of an A/B sort identical data.
// ---------------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, n)`.
    #[inline]
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

// ---------------------------------------------------------------------------
// Column generators (one batch's worth at a time)
// ---------------------------------------------------------------------------

/// `i64` column of full-range random values (effectively all-distinct keys).
fn generate_i64_random(rng: &mut Rng, n: usize) -> ArrayRef {
    let values: Int64Array = (0..n).map(|_| rng.next_u64() as i64).collect();
    Arc::new(values)
}

/// `i64` column continuing a globally ascending sequence from `*next`.
fn generate_i64_ascending(next: &mut i64, n: usize) -> ArrayRef {
    let values: Int64Array = (0..n)
        .map(|_| {
            let key = *next;
            *next += 3;
            key
        })
        .collect();
    Arc::new(values)
}

/// Build `n_distinct` distinct strings averaging ~`avg_len` bytes whose bytes
/// diverge within the first few characters (a short shared lead, then a
/// per-entry token), so key comparisons short-circuit early like real ids and
/// URLs instead of scanning a long common prefix. All are >12 B so the views
/// are buffer-backed rather than inlined.
fn build_string_dict(n_distinct: usize, prefix: &str, avg_len: usize) -> Vec<String> {
    let lead: String = prefix.bytes().take(7).map(|b| b as char).collect();
    (0..n_distinct)
        .map(|i| {
            let token = i.wrapping_mul(2_654_435_761) % 100_000_000;
            let mut s = format!("{lead}{token}");
            let target = (avg_len / 2).max(8) + (i % avg_len.max(1));
            let mut k = i as u64 + 1;
            while s.len() < target {
                k = k.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                s.push('/');
                s.push_str(&(k % 100_000).to_string());
            }
            s.truncate(target.max(8));
            s
        })
        .collect()
}

/// A dictionary block: every distinct string stored once in a single backing
/// buffer, with the `(offset, len)` of each entry recorded. Columns then emit
/// dictionary-shared views into it, repeated rows pointing at the same bytes,
/// exactly as Parquet dictionary decoding produces.
struct DictBlock {
    buffer: arrow_buffer::Buffer,
    /// `(offset, len)` into `buffer` for each distinct dictionary entry.
    spans: Vec<(u32, u32)>,
}

fn build_dict_block(dict: &[String]) -> DictBlock {
    let mut data: Vec<u8> = Vec::new();
    let mut spans = Vec::with_capacity(dict.len());
    for s in dict {
        spans.push((data.len() as u32, s.len() as u32));
        data.extend_from_slice(s.as_bytes());
    }
    DictBlock {
        buffer: arrow_buffer::Buffer::from_vec(data),
        spans,
    }
}

/// An `n`-row `StringViewArray` of uniformly drawn dictionary-shared views.
fn generate_string_col(rng: &mut Rng, n: usize, block: &DictBlock) -> ArrayRef {
    let mut builder = StringViewBuilder::with_capacity(n);
    let block_id = builder.append_block(block.buffer.clone());
    for _ in 0..n {
        let (offset, len) = block.spans[rng.below(block.spans.len())];
        builder.try_append_view(block_id, offset, len).unwrap();
    }
    Arc::new(builder.finish())
}

/// Logical bytes of one batch: fixed widths for primitives, view plus payload
/// bytes for strings. Independent of buffer sharing, so it is a stable
/// denominator for bytes/s throughput.
fn count_logical_bytes(batch: &RecordBatch) -> usize {
    batch
        .columns()
        .iter()
        .map(|column| match column.data_type() {
            DataType::Int64 => 8 * column.len(),
            DataType::Utf8View => {
                let strings = column.as_any().downcast_ref::<StringViewArray>().unwrap();
                16 * strings.len()
                    + strings
                        .views()
                        .iter()
                        .map(|&view| (view as u32) as usize)
                        .sum::<usize>()
            }
            other => panic!("no logical size rule for bench column type {other}"),
        })
        .sum()
}

/// Build `BATCH_ROWS`-sized batches until their logical bytes reach `target`.
fn generate_until(
    target: usize,
    schema: &Arc<Schema>,
    mut make_cols: impl FnMut(usize) -> Vec<ArrayRef>,
) -> Vec<RecordBatch> {
    let mut total = 0;
    let mut batches = Vec::new();
    while total < target {
        let batch = RecordBatch::try_new(schema.clone(), make_cols(BATCH_ROWS)).unwrap();
        total += count_logical_bytes(&batch);
        batches.push(batch);
    }
    batches
}

// ---------------------------------------------------------------------------
// Run helper: stream the prebuilt batches through the pipeline and drain.
// `batches` is cloned per iteration (cheap Arc bumps); the engine is reused.
// ---------------------------------------------------------------------------

/// Sort `batches` through one dataflow, dropping every output batch on the
/// worker that produced it. Only per-batch row counts leave the dataflow, so
/// the measurement covers the sort alone: no ring-to-heap copy of the result
/// and no delivery funnel, and the slabs return to the ring where they were
/// filled. The row-count check still catches a sort that drops or duplicates
/// rows, which would otherwise read as a throughput change.
fn run_sort(d: &DataFlowDispatcher, batches: &[RecordBatch], order_by: &[OrderBy]) {
    let spec = values_input(d, batches.to_vec()).record_batches();
    let batch_rows = spec
        .order_by(order_by.to_vec())
        .map(|| |batch: RecordBatch| Ok(batch.num_rows()))
        .collect()
        .unwrap();
    let input_rows: usize = batches.iter().map(|batch| batch.num_rows()).sum();
    let output_rows: usize = batch_rows.iter().sum();
    assert_eq!(output_rows, input_rows, "sort did not re-emit every row");
}

/// Register one scenario. `make_data` runs only when the benchmark is selected
/// and its result is cached across iterations, so a filtered run builds just
/// the one scenario's dataset.
fn register<D>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    d: &DataFlowDispatcher,
    name: &str,
    make_data: D,
    keys: Vec<OrderBy>,
) where
    D: Fn() -> Vec<RecordBatch>,
{
    let mut cache: Option<Vec<RecordBatch>> = None;
    group.bench_function(name, move |b| {
        let batches = cache.get_or_insert_with(&make_data);
        // Per-iteration setup (UNTIMED): re-zero the buffers the previous
        // query dirtied, so allocation cost stays out of the measured window.
        b.iter_batched(
            || {
                d.run_on_workers(|| {
                    memory_ctx().zero_dirty_buffers();
                })
            },
            |_| run_sort(d, batches, &keys),
            BatchSize::PerIteration,
        );
    });
}

// ---------------------------------------------------------------------------
// Scenarios
// ---------------------------------------------------------------------------

fn bench_order_by_large(c: &mut Criterion, d: &DataFlowDispatcher) {
    let target = target_sort_bytes();

    // One group for all four scenarios, so criterion's group summary compares
    // them and the throughput/sampling config is stated once.
    let mut group = c.benchmark_group("order_by");
    group.throughput(Throughput::Bytes(target as u64));
    // Whole-input sorts run seconds per iteration: flat sampling keeps every
    // sample at one iteration instead of criterion's scaled linear ramp.
    group.sampling_mode(SamplingMode::Flat);

    register(
        &mut group,
        d,
        "i64_random",
        move || {
            let schema = Arc::new(Schema::new(vec![
                Field::new("k", DataType::Int64, false),
                Field::new("v", DataType::Int64, false),
            ]));
            let mut rng = Rng::new(1);
            generate_until(target, &schema, |n| {
                vec![
                    generate_i64_random(&mut rng, n),
                    generate_i64_random(&mut rng, n),
                ]
            })
        },
        vec![OrderBy::new(0, false, false)],
    );

    register(
        &mut group,
        d,
        "i64_presorted",
        move || {
            let schema = Arc::new(Schema::new(vec![
                Field::new("k", DataType::Int64, false),
                Field::new("v", DataType::Int64, false),
            ]));
            let mut rng = Rng::new(2);
            let mut next_key = 0;
            generate_until(target, &schema, |n| {
                vec![
                    generate_i64_ascending(&mut next_key, n),
                    generate_i64_random(&mut rng, n),
                ]
            })
        },
        vec![OrderBy::new(0, false, false)],
    );

    register(
        &mut group,
        d,
        "i64_wide_payload",
        move || {
            let schema = Arc::new(Schema::new(vec![
                Field::new("k", DataType::Int64, false),
                Field::new("v0", DataType::Int64, false),
                Field::new("v1", DataType::Int64, false),
                Field::new("v2", DataType::Int64, false),
                Field::new("v3", DataType::Int64, false),
                Field::new("s", DataType::Utf8View, false),
            ]));
            let block = build_dict_block(&build_string_dict(1_000_000, "payload ", 32));
            let mut rng = Rng::new(3);
            generate_until(target, &schema, |n| {
                vec![
                    generate_i64_random(&mut rng, n),
                    generate_i64_random(&mut rng, n),
                    generate_i64_random(&mut rng, n),
                    generate_i64_random(&mut rng, n),
                    generate_i64_random(&mut rng, n),
                    generate_string_col(&mut rng, n, &block),
                ]
            })
        },
        vec![OrderBy::new(0, false, false)],
    );

    register(
        &mut group,
        d,
        "string_random",
        move || {
            let schema = Arc::new(Schema::new(vec![
                Field::new("k", DataType::Utf8View, false),
                Field::new("v", DataType::Int64, false),
            ]));
            let block = build_dict_block(&build_string_dict(1_000_000, "series/", 32));
            let mut rng = Rng::new(4);
            generate_until(target, &schema, |n| {
                vec![
                    generate_string_col(&mut rng, n, &block),
                    generate_i64_random(&mut rng, n),
                ]
            })
        },
        vec![OrderBy::new(0, false, false)],
    );

    group.finish();
}

// ---------------------------------------------------------------------------
// Harness: one shared Dispatch (worker pool) for every scenario.
// ---------------------------------------------------------------------------

fn main() {
    let workers = worker_count();
    let buffers = ring_buffers();
    eprintln!(
        "order_by benches: {workers} workers, {buffers} ring buffers, {} MiB per scenario",
        target_sort_bytes() >> 20
    );

    let dispatch = Dispatch::spin_up(workers, buffers, None);
    let dispatcher = dispatch.dispatcher().clone();

    // Default to a handful of samples (each is a whole multi-second sort); a
    // `--sample-size` on the command line still overrides this.
    let mut c = Criterion::default().sample_size(10).configure_from_args();
    bench_order_by_large(&mut c, &dispatcher);
    c.final_summary();

    dispatch.exit();
}
