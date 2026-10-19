//! Microbenchmarks for individual dispatch operators (GROUP BY, filter/contains,
//! ORDER BY ... LIMIT) driven in **complete isolation** — no Parquet, no IO, no
//! catalog. Pre-built Arrow `RecordBatch`es are streamed straight into one
//! operator through `values_input`, exactly as a table scan would fan morsels
//! across the worker pool, and the result is drained with `.collect()`.
//!
//! # Why these scenarios
//!
//! Each scenario reproduces the *statistical shape* of a distinct analytical
//! workload we care about — the key/value column **types**, the **cardinality
//! ratio** (rows per group), string **length / buffer layout**, and filter
//! **selectivity**. Those four properties are what decide which monomorphised
//! code path runs and how time splits between hashing, probing, scatter and
//! merge — i.e. they decide the perf *sample distribution*. The goal is for a
//! few seconds of one of these benches to be a faithful, fast replicator of the
//! corresponding real query's operator profile, so we can iterate on (say) the
//! GROUP BY hot loop here instead of on a 14 GB dataset.
//!
//! The data shapes (not the queries) are described in comments next to each
//! scenario. We deliberately avoid coupling the bench to any external suite.
//!
//! # Running
//!
//! ```sh
//! # All scenarios, full statistical analysis:
//! cargo bench --bench operators
//!
//! # One scenario, looped for 20s with no analysis — ideal to attach perf to:
//! cargo bench --bench operators -- "group_by/string_highcard" --profile-time 20
//! ```
//!
//! Tunables (env): `PIVOT_BENCH_ROWS` (total input rows, default 8M),
//! `PIVOT_BENCH_WORKERS` (default: all cores — matches a real query),
//! `PIVOT_BENCH_BUFFERS` (ring slots of 2 MiB each, default 4096).
//!
//! Sizing the ring: two effects bound `PIVOT_BENCH_BUFFERS`.
//! * **Floor** — the high-cardinality integer GROUP BY scatters into 4096 radix
//!   partitions, each grabbing a ~64 KiB first chunk: a floor of ~128 buffers
//!   *per worker* the moment it switches, independent of row count. A 16-worker
//!   run thus needs ≳2k buffers just for scatter, plus the in-place tables and
//!   output. With no Parquet page cache to evict, an undersized ring panics
//!   (`Evicting`) — bump `PIVOT_BENCH_BUFFERS` if so.
//! * **Ceiling** — engine startup allocates a ring *and* a compressed cache, each
//!   sized to `buffers`, so the resident floor is ≈ `2 × buffers × 2 MiB`
//!   (≈16 GiB at 4096) before any data. Keep that under box RAM. (The file
//!   cache is unused here — no Parquet — but is still reserved.)
//!
//! A small smoke run can drop both (e.g. `PIVOT_BENCH_WORKERS=2
//! PIVOT_BENCH_BUFFERS=768`).
//!
//! NOTE: under `perf`, raise the locked-memory limit (`ulimit -l unlimited`),
//! otherwise per-worker io_uring setup hits `ENOMEM`.
//!
//! CAVEAT — high-cardinality *string* GROUP BY growth: re-running the
//! `string_*` scenarios many times on the one shared engine grows resident
//! memory (the per-worker key arena is not fully reclaimed between queries when
//! the engine is reused), so a long run at 8M rows can OOM. The integer paths
//! are steady. Until that is fixed engine-side, profile the string scenarios
//! with bounded iterations and/or fewer rows, e.g.
//! `PIVOT_BENCH_ROWS=4000000 … --bench "group_by/string_highcard" --sample-size 20`.
//!
//! VALIDATION — these scenarios were profiled against the matching real queries.
//! With the ORDER BY count DESC LIMIT 10 cap (below) the integer high-card GROUP BY
//! lines up with the real query: merge + consume dominate (~70%), `memset`/output
//! a few percent — matching the real profile. The cap matters: without it every
//! group reaches the output and the ring→heap `CopyOut` copy dominates instead of
//! the operator.

use std::sync::Arc;

use arrow_array::builder::StringViewBuilder;
use arrow_array::types::{Int16Type, Int32Type, Int64Type};
use arrow_array::{
    Array, ArrayRef, BooleanArray, Int16Array, Int32Array, Int64Array, RecordBatch, StringViewArray,
};
use arrow_schema::{DataType, Field, Schema};
use criterion::{BatchSize, Criterion, Throughput, black_box};

use dispatch::memory::SlabAllocator;
use dispatch::{
    AggregationKind, AggregationSlot, Compiled, Contains, CountSlot, DataFlowDispatcher, Dispatch,
    Distinct, Dynamic, GroupLimit, IntKeyExtractor, IntPairKeyExtractor, IntStrKeyExtractor,
    OrderBy, RecordBatchOperatorSpec, RowKeyExtractor, RowKeySchema, RowSelection,
    SelectionScratch, StringKeyExtractor, SumSlot, collect_selected_indices, memory_ctx,
    values_input,
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Total input rows per scenario. Large enough that the operator hot loop, not
/// fixed per-dataflow overhead, dominates the samples.
const DEFAULT_ROWS: usize = 8_000_000;

/// Rows per `RecordBatch`. Mirrors a scan morsel: many independent batches fan
/// out across workers through the work-stealing injector, so the consume phase
/// runs on every core just like a real query.
const BATCH_ROWS: usize = 64 * 1024;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn total_rows() -> usize {
    env_usize("PIVOT_BENCH_ROWS", DEFAULT_ROWS)
}

fn worker_count() -> usize {
    env_usize(
        "PIVOT_BENCH_WORKERS",
        core_affinity::get_core_ids().map(|c| c.len()).unwrap_or(1),
    )
}

fn ring_buffers() -> usize {
    env_usize("PIVOT_BENCH_BUFFERS", 4096)
}

/// Split `total` rows into `BATCH_ROWS`-sized chunks (last one may be smaller).
fn batch_sizes(total: usize) -> impl Iterator<Item = usize> {
    (0..total)
        .step_by(BATCH_ROWS)
        .map(move |start| BATCH_ROWS.min(total - start))
}

// ---------------------------------------------------------------------------
// Deterministic PRNG (splitmix64) — no external dependency, reproducible data.
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

    /// `true` with probability `p` (resolution 1/1_000_000).
    #[inline]
    fn chance(&mut self, p: f64) -> bool {
        self.below(1_000_000) < (p * 1_000_000.0) as usize
    }
}

// ---------------------------------------------------------------------------
// Column generators (one batch's worth at a time)
// ---------------------------------------------------------------------------

/// `i64` key column, `n_distinct` distinct values (id-like, spread out).
fn i64_keys(rng: &mut Rng, n: usize, n_distinct: usize) -> ArrayRef {
    // Spread the drawn ids across the 64-bit space so values look like real ids
    // (doesn't change the hash distribution, but keeps the data honest).
    let v: Int64Array = (0..n)
        .map(|_| (rng.below(n_distinct) as i64).wrapping_mul(0x100_0001))
        .collect();
    Arc::new(v)
}

/// `i32` key column, `n_distinct` distinct values.
fn i32_keys(rng: &mut Rng, n: usize, n_distinct: usize) -> ArrayRef {
    let v: Int32Array = (0..n).map(|_| rng.below(n_distinct) as i32).collect();
    Arc::new(v)
}

/// `i16` column, `n_distinct` distinct values starting at 1 with a configurable
/// fraction forced to 0 (an enum-like column dominated by a sentinel/"none").
fn i16_enum(rng: &mut Rng, n: usize, n_distinct: usize, zero_fraction: f64) -> ArrayRef {
    let v: Int16Array = (0..n)
        .map(|_| {
            if rng.chance(zero_fraction) {
                0
            } else {
                (1 + rng.below(n_distinct)) as i16
            }
        })
        .collect();
    Arc::new(v)
}

/// Small-range `i16` value column (e.g. a boolean-ish flag or a bounded width).
fn i16_value(rng: &mut Rng, n: usize, range: usize) -> ArrayRef {
    let v: Int16Array = (0..n).map(|_| rng.below(range) as i16).collect();
    Arc::new(v)
}

/// Build a dictionary of `n_distinct` distinct strings averaging ~`avg_len`
/// bytes. Two properties matter for matching a real string GROUP BY's profile:
/// keys **diverge within the first few bytes** (only a short shared lead from
/// `prefix`, then a per-entry token) so key comparisons short-circuit early like
/// real URLs/phrases instead of scanning a long common prefix — otherwise
/// `memcmp` is wildly over-weighted; and **lengths vary** (~0.5×–1.5× `avg_len`)
/// so hashing cost spreads like real data. All are >12 B (buffer-backed views).
fn string_dict(n_distinct: usize, prefix: &str, avg_len: usize) -> Vec<String> {
    // Short shared lead (e.g. "http://", "search ") — realistic but short, so the
    // distinguishing token starts within ~7 bytes.
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

/// A dictionary block: every distinct string stored **once** in a single
/// backing buffer, with the `(offset, len)` of each entry recorded. A column
/// then emits dictionary-*shared* views into it — repeated rows point at the
/// same bytes, exactly as Parquet dictionary decoding produces. This is what
/// makes the GROUP BY / `Contains` profile match a real scan: the key arena
/// copies each distinct value once (not per row), and `Contains` scans each
/// physical buffer once. Building views the naive way (append each value)
/// instead copies every row's bytes and wildly inflates `memcpy`.
struct DictBlock {
    buffer: arrow_buffer::Buffer,
    /// `(offset, len)` into `buffer` for each distinct dictionary entry.
    spans: Vec<(u32, u32)>,
}

fn dict_block(dict: &[String]) -> DictBlock {
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

/// Build an `n`-row [`StringViewArray`] of dictionary-shared views: `pick`
/// returns the dict index for each row, or `None` for an empty string.
fn dict_view_col(n: usize, block: &DictBlock, mut pick: impl FnMut() -> Option<usize>) -> ArrayRef {
    let mut b = StringViewBuilder::with_capacity(n);
    let blk = b.append_block(block.buffer.clone());
    for _ in 0..n {
        match pick() {
            Some(i) => {
                let (off, len) = block.spans[i];
                b.try_append_view(blk, off, len).unwrap();
            }
            None => b.append_value(""),
        }
    }
    Arc::new(b.finish())
}

/// Uniform dictionary column (every row a shared view into `block`).
fn string_col(rng: &mut Rng, n: usize, block: &DictBlock) -> ArrayRef {
    let len = block.spans.len();
    dict_view_col(n, block, || Some(rng.below(len)))
}

/// Empty with probability `empty_fraction`, otherwise a uniform dictionary
/// value — the "mostly-blank free-text" shape.
fn sparse_string_col(rng: &mut Rng, n: usize, block: &DictBlock, empty_fraction: f64) -> ArrayRef {
    let len = block.spans.len();
    dict_view_col(n, block, || {
        (!rng.chance(empty_fraction)).then(|| rng.below(len))
    })
}

fn schema(fields: Vec<Field>) -> Arc<Schema> {
    Arc::new(Schema::new(fields))
}

fn batch(schema: &Arc<Schema>, cols: Vec<ArrayRef>) -> RecordBatch {
    RecordBatch::try_new(schema.clone(), cols).unwrap()
}

// ---------------------------------------------------------------------------
// Filter predicate closures (representative, allocation-free per batch)
// ---------------------------------------------------------------------------

/// Keeps the rows where `column != 0` over an `i16` column.
fn keep_i16_nonzero(batch: &RecordBatch, col: usize, indices: &mut Vec<u32>) -> RowSelection {
    let a = batch
        .column(col)
        .as_any()
        .downcast_ref::<Int16Array>()
        .unwrap();
    let mask: BooleanArray = (0..a.len()).map(|i| Some(a.value(i) != 0)).collect();
    collect_selected_indices(&mask, indices);
    RowSelection::Indices
}

/// `column == target` over an `i64` column (point lookup).
fn keep_i64_eq(
    batch: &RecordBatch,
    col: usize,
    target: i64,
    indices: &mut Vec<u32>,
) -> RowSelection {
    let a = batch
        .column(col)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let mask: BooleanArray = (0..a.len()).map(|i| Some(a.value(i) == target)).collect();
    collect_selected_indices(&mask, indices);
    RowSelection::Indices
}

/// `column <> ''` over a string column.
fn keep_nonempty(batch: &RecordBatch, col: usize, indices: &mut Vec<u32>) -> RowSelection {
    let a = batch
        .column(col)
        .as_any()
        .downcast_ref::<StringViewArray>()
        .unwrap();
    // Length lives in the low 32 bits of each view, no UTF-8 decode needed.
    let mask: BooleanArray = a.views().iter().map(|&v| Some((v as u32) != 0)).collect();
    collect_selected_indices(&mask, indices);
    RowSelection::Indices
}

// ---------------------------------------------------------------------------
// Run helper: stream the prebuilt batches through the pipeline and drain.
// `batches` is cloned per iteration (cheap Arc bumps); the engine is reused.
// ---------------------------------------------------------------------------

fn run<F>(d: &DataFlowDispatcher, batches: &[RecordBatch], build: &F)
where
    F: Fn(RecordBatchOperatorSpec) -> RecordBatchOperatorSpec,
{
    let spec = values_input(d, batches.to_vec()).record_batches();
    let out = build(spec).collect().unwrap();
    black_box(out);
}

/// Register one scenario: a fresh group so per-scenario throughput (rows/s) is
/// reported. `make_data` is invoked **only when this benchmark is selected to
/// run** (Criterion calls the routine closure once per matching benchmark), so a
/// filtered run builds just the one scenario's dataset instead of all of them.
fn bench<D, F>(
    c: &mut Criterion,
    d: &DataFlowDispatcher,
    name: &str,
    rows: usize,
    make_data: D,
    build: F,
) where
    D: FnOnce() -> Vec<RecordBatch>,
    F: Fn(RecordBatchOperatorSpec) -> RecordBatchOperatorSpec,
{
    // Criterion drives the routine closure several times per benchmark (warm-up
    // then measurement), but never at all for a filtered-out benchmark — so we
    // build the dataset lazily on the first call and cache it, paying the build
    // once and only for the scenario actually selected.
    let make_data = std::cell::RefCell::new(Some(make_data));
    let cache: std::cell::RefCell<Option<Vec<RecordBatch>>> = std::cell::RefCell::new(None);
    let mut g = c.benchmark_group("dispatch");
    g.throughput(Throughput::Elements(rows as u64));
    g.bench_function(name, |b| {
        if cache.borrow().is_none() {
            let data = make_data.borrow_mut().take().expect("make_data missing")();
            *cache.borrow_mut() = Some(data);
        }
        let cached = cache.borrow();
        let batches = cached.as_ref().unwrap();
        // Per-iteration setup (UNTIMED): re-zero the buffers the previous query
        // dirtied, so this query's hash-table allocation pulls pre-zeroed buffers
        // and the inline `memset` (table init) stays out of the measured window —
        // it's allocation/setup, not group-by compute. Requires PerIteration so
        // setup runs before *every* timed call (a batched setup would let the
        // 2nd+ query in a batch hit dirty buffers again).
        b.iter_batched(
            || {
                d.run_on_workers(|| {
                    memory_ctx().zero_dirty_buffers();
                })
            },
            |_| run(d, batches, &build),
            BatchSize::PerIteration,
        );
    });
    g.finish();
}

// ---------------------------------------------------------------------------
// GROUP BY scenarios
// ---------------------------------------------------------------------------

fn bench_group_by(c: &mut Criterion, d: &DataFlowDispatcher) {
    let rows = total_rows();

    // (a) Low-cardinality i64 key over ALL rows. Few groups, every row probes a
    //     hot, cache-resident table; merge is trivial. Shape: a time-bucketed
    //     GROUP BY (e.g. truncate-to-minute) producing ~1.5k groups.
    bench(
        c,
        d,
        "group_by/int_lowcard_allrows",
        rows,
        || {
            let sch = schema(vec![Field::new("k", DataType::Int64, false)]);
            let mut rng = Rng::new(1);
            batch_sizes(rows)
                .map(|n| batch(&sch, vec![i64_keys(&mut rng, n, 1_500)]))
                .collect()
        },
        // ORDER BY count DESC LIMIT 10 — the near-universal shape of these
        // queries. Without it, every group flows to the output and the CopyOut
        // (ring→heap) copy dominates the profile instead of the group-by.
        |s| {
            s.group_by_aggregate::<IntKeyExtractor<Int64Type>, Compiled<(CountSlot,)>>(
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
        },
    );

    // (b) Enum-like i16 key dominated by 0, filtered to the non-zero rows before
    //     grouping. The group sees a small, dense slice (~tens of groups).
    bench(
        c,
        d,
        "group_by/int_lowcard_filtered",
        rows,
        || {
            let sch = schema(vec![Field::new("k", DataType::Int16, false)]);
            let mut rng = Rng::new(2);
            batch_sizes(rows)
                .map(|n| batch(&sch, vec![i16_enum(&mut rng, n, 50, 0.99)]))
                .collect()
        },
        |s| {
            s.filter(|| {
                move |b: &RecordBatch, _: &mut SlabAllocator, scratch: &mut SelectionScratch| {
                    keep_i16_nonzero(b, 0, &mut scratch.positions)
                }
            })
            .group_by_aggregate::<IntKeyExtractor<Int16Type>, Compiled<(CountSlot,)>>(
                vec![0],
                vec![AggregationSlot::new(
                    AggregationKind::CountStar,
                    0,
                    DataType::Int64,
                )],
                None,
                (),
            )
        },
    );

    // (c) High-cardinality i64 key (~rows/6 distinct → a few rows per group).
    //     Crosses the radix-switch threshold: scatter + partition-parallel merge
    //     dominate. The canonical bandwidth-bound GROUP BY.
    bench(
        c,
        d,
        "group_by/int_highcard",
        rows,
        || {
            let sch = schema(vec![Field::new("k", DataType::Int64, false)]);
            let distinct = (rows / 6).max(1);
            let mut rng = Rng::new(3);
            batch_sizes(rows)
                .map(|n| batch(&sch, vec![i64_keys(&mut rng, n, distinct)]))
                .collect()
        },
        // ORDER BY count DESC LIMIT 10 — the near-universal shape of these
        // queries. Without it, every group flows to the output and the CopyOut
        // (ring→heap) copy dominates the profile instead of the group-by.
        |s| {
            s.group_by_aggregate::<IntKeyExtractor<Int64Type>, Compiled<(CountSlot,)>>(
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
        },
    );

    // (d) High-cardinality string key (~rows/10 distinct). Always in-place (no
    //     radix for strings); cost is arena key copies + string hashing/compare.
    bench(
        c,
        d,
        "group_by/string_highcard",
        rows,
        || {
            let sch = schema(vec![Field::new("k", DataType::Utf8View, false)]);
            // Match the real URL column: ~18.3M distinct over ~100M rows
            // (≈5.5 rows/group), ~88-byte average length. Cardinality drives the
            // consume/memcmp balance — too few distinct means too many repeat
            // probes, each a full-length key compare, over-weighting memcmp;
            // length drives the hashing share.
            let dict = string_dict((rows * 18 / 100).max(1), "http://example.com/path", 88);
            let block = dict_block(&dict);
            let mut rng = Rng::new(4);
            batch_sizes(rows)
                .map(|n| batch(&sch, vec![string_col(&mut rng, n, &block)]))
                .collect()
        },
        |s| {
            s.group_by_aggregate::<StringKeyExtractor, Compiled<(CountSlot,)>>(
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
        },
    );

    // (e) Mostly-blank free-text string key, filtered to non-empty before
    //     grouping (~15% survive), medium cardinality among the survivors.
    bench(
        c,
        d,
        "group_by/string_filtered",
        rows,
        || {
            let sch = schema(vec![Field::new("k", DataType::Utf8View, false)]);
            let dict = string_dict((rows * 6 / 100).max(1), "search query ", 31);
            let block = dict_block(&dict);
            let mut rng = Rng::new(5);
            batch_sizes(rows)
                .map(|n| batch(&sch, vec![sparse_string_col(&mut rng, n, &block, 0.868)]))
                .collect()
        },
        |s| {
            s.filter(|| {
                move |b: &RecordBatch, _: &mut SlabAllocator, scratch: &mut SelectionScratch| {
                    keep_nonempty(b, 0, &mut scratch.positions)
                }
            })
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
        },
    );

    // (f) Heavily-selective predicate then a high-card string GROUP BY: only a
    //     few percent of rows reach the group. A second i16 "predicate" column
    //     gates the rows; the string column is the key.
    bench(
        c,
        d,
        "group_by/string_selective",
        rows,
        || {
            let sch = schema(vec![
                Field::new("pred", DataType::Int16, false),
                Field::new("k", DataType::Utf8View, false),
            ]);
            let dict = string_dict((rows / 10).max(1), "http://example.com/page", 88);
            let block = dict_block(&dict);
            let mut rng = Rng::new(6);
            batch_sizes(rows)
                .map(|n| {
                    // pred == 62 keeps ~3% of rows (single-value match).
                    let pred: Int16Array = (0..n)
                        .map(|_| if rng.chance(0.03) { 62 } else { 0 })
                        .collect();
                    batch(&sch, vec![Arc::new(pred), string_col(&mut rng, n, &block)])
                })
                .collect()
        },
        |s| {
            s.filter(|| {
                move |b: &RecordBatch, _: &mut SlabAllocator, scratch: &mut SelectionScratch| {
                    let a = b.column(0).as_any().downcast_ref::<Int16Array>().unwrap();
                    let mask: BooleanArray = (0..a.len()).map(|i| Some(a.value(i) == 62)).collect();
                    collect_selected_indices(&mask, &mut scratch.positions);
                    RowSelection::Indices
                }
            })
            .group_by_aggregate::<StringKeyExtractor, Compiled<(CountSlot,)>>(
                vec![1],
                vec![AggregationSlot::new(
                    AggregationKind::CountStar,
                    0,
                    DataType::Int64,
                )],
                None,
                (),
            )
            .order_by_limit(vec![OrderBy::new(1, true, false)], 10)
        },
    );

    // (g) Composite two-int key (i64, i32) with multiple aggregates and a
    //     top-k pushdown: COUNT(*), SUM, SUM, COUNT (the AVG lowering), ordered
    //     by the count slot DESC LIMIT 10. The compiled, straight-line value
    //     extractor — high cardinality, the heaviest grouped-aggregate path.
    {
        // [COUNT(*), SUM(v0), SUM(v1), COUNT(v1)] — matches the compiled
        // `(CountSlot, SumSlot<i16>, SumSlot<i16>, CountSlot)` value extractor.
        let slots = vec![
            AggregationSlot::new(AggregationKind::CountStar, 0, DataType::Int64),
            AggregationSlot::new(AggregationKind::Sum, 2, DataType::Decimal128(38, 0)),
            AggregationSlot::new(AggregationKind::Sum, 3, DataType::Decimal128(38, 0)),
            AggregationSlot::new(AggregationKind::Count, 3, DataType::Int64),
        ];
        type Value = Compiled<(CountSlot, SumSlot<Int16Type>, SumSlot<Int16Type>, CountSlot)>;
        bench(
            c,
            d,
            "group_by/pair_int_aggregates",
            rows,
            || {
                let sch = schema(vec![
                    Field::new("k0", DataType::Int64, false), // near-unique id
                    Field::new("k1", DataType::Int32, false), // high-card int
                    Field::new("v0", DataType::Int16, false), // boolean-ish flag
                    Field::new("v1", DataType::Int16, false), // bounded width
                ]);
                let mut rng = Rng::new(7);
                batch_sizes(rows)
                    .map(|n| {
                        batch(
                            &sch,
                            vec![
                                i64_keys(&mut rng, n, (rows / 2).max(1)),
                                i32_keys(&mut rng, n, 1_000_000),
                                i16_value(&mut rng, n, 2),
                                i16_value(&mut rng, n, 2560),
                            ],
                        )
                    })
                    .collect()
            },
            move |s| {
                s.group_by_aggregate::<IntPairKeyExtractor<Int64Type, Int32Type>, Value>(
                    vec![0, 1],
                    slots.clone(),
                    Some(GroupLimit::TopK {
                        slot: 0,
                        limit: 10,
                        with_ties: false,
                    }),
                    (),
                )
            },
        );
    }

    // (h) Composite low-card-ish key (i16, i32) with aggregates, behind a
    //     non-empty free-text filter — the lower-cardinality sibling of (g),
    //     using the runtime-signature `Dynamic` value. Its slots are all additive
    //     (COUNT/SUM), so it takes the `ONLY_ADDITIVE` form the planner routes an
    //     all-additive signature to: a branch-free additive fold, not the per-slot
    //     kind dispatch.
    {
        let slots = vec![
            AggregationSlot::new(AggregationKind::CountStar, 0, DataType::Int64),
            AggregationSlot::new(AggregationKind::Sum, 2, DataType::Decimal128(38, 0)),
            AggregationSlot::new(AggregationKind::Sum, 3, DataType::Decimal128(38, 0)),
            AggregationSlot::new(AggregationKind::Count, 3, DataType::Int64),
        ];
        bench(
            c,
            d,
            "group_by/pair_int_lowcard_agg",
            rows,
            || {
                let sch = schema(vec![
                    Field::new("k0", DataType::Int16, false),
                    Field::new("k1", DataType::Int32, false),
                    Field::new("v0", DataType::Int16, false),
                    Field::new("v1", DataType::Int16, false),
                    Field::new("free", DataType::Utf8View, false),
                ]);
                let dict = string_dict((rows * 6 / 100).max(1), "search query ", 31);
                let block = dict_block(&dict);
                let mut rng = Rng::new(8);
                batch_sizes(rows)
                    .map(|n| {
                        batch(
                            &sch,
                            vec![
                                i16_value(&mut rng, n, 32),
                                // ~2k distinct in the high-card column → with the
                                // i16 column ~64k group keys; after the non-empty
                                // filter (~15% of rows) that's tens of rows per
                                // group, like the real low-card composite — enough
                                // rows/group that consume (folding) balances merge,
                                // rather than every row being its own near-unique
                                // group (which over-weights merge).
                                i32_keys(&mut rng, n, 2_000),
                                i16_value(&mut rng, n, 2),
                                i16_value(&mut rng, n, 2560),
                                sparse_string_col(&mut rng, n, &block, 0.868),
                            ],
                        )
                    })
                    .collect()
            },
            move |s| {
                s.filter(|| move |b: &RecordBatch, _: &mut SlabAllocator, scratch: &mut SelectionScratch| keep_nonempty(b, 4, &mut scratch.positions))
                    .group_by_aggregate::<IntPairKeyExtractor<Int16Type, Int32Type>, Dynamic<i64, true>>(
                        vec![0, 1],
                        slots.clone(),
                        Some(GroupLimit::TopK { slot: 0, limit: 10, with_ties: false }),
                        (),
                    )
            },
        );
    }

    // (i) Two-level COUNT(DISTINCT): dedup (group, value) pairs with a keys-only
    //     two-int GROUP BY, then count the distinct rows per group. group = i32
    //     (~few thousand), value = i64 (high card).
    bench(
        c,
        d,
        "group_by/distinct_count",
        rows,
        || {
            let sch = schema(vec![
                Field::new("g", DataType::Int32, false),
                Field::new("x", DataType::Int64, false),
            ]);
            let mut rng = Rng::new(9);
            batch_sizes(rows)
                .map(|n| {
                    batch(
                        &sch,
                        vec![
                            i32_keys(&mut rng, n, 9_000),
                            i64_keys(&mut rng, n, (rows / 6).max(1)),
                        ],
                    )
                })
                .collect()
        },
        |s| {
            s.group_by_aggregate::<IntPairKeyExtractor<Int32Type, Int64Type>, Distinct>(
                vec![0, 1],
                Vec::new(),
                None,
                (),
            )
            .group_by_aggregate::<IntKeyExtractor<Int32Type>, Compiled<(CountSlot,)>>(
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
        },
    );

    // (j) Composite (i64 id, sparse free-text) key, high cardinality, most phrases
    //     blank so the key is dominated by the id. Models `GROUP BY id, phrase
    //     COUNT(*) LIMIT 10` (a plain `First` LIMIT pushdown keeps the output
    //     tiny so the group-by hot loop dominates rather than the CopyOut). Run
    //     two ways over identical data, so the pair is a direct A/B:
    //
    //     * `_rowkey` byte-encodes the `(Int64, Utf8View)` tuple with the generic
    //       `RowKeyExtractor` — its `encode_and_hash` shows up alongside the probe
    //       and merge.
    //     * `_pair` uses the dedicated `IntStrKeyExtractor`: the native int beside
    //       the string's arena handle, no row encode.
    let count_star = || {
        vec![AggregationSlot::new(
            AggregationKind::CountStar,
            0,
            DataType::Int64,
        )]
    };
    bench(
        c,
        d,
        "group_by/int_string_highcard_rowkey",
        rows,
        || int_string_dataset(rows),
        move |s| {
            s.group_by_aggregate::<RowKeyExtractor, Compiled<(CountSlot,)>>(
                vec![0, 1],
                count_star(),
                Some(GroupLimit::First { limit: 10 }),
                RowKeySchema::new(
                    vec![DataType::Int64, DataType::Utf8View],
                    vec![false, false],
                ),
            )
        },
    );
    bench(
        c,
        d,
        "group_by/int_string_highcard_pair",
        rows,
        || int_string_dataset(rows),
        move |s| {
            s.group_by_aggregate::<IntStrKeyExtractor<Int64Type>, Compiled<(CountSlot,)>>(
                vec![0, 1],
                count_star(),
                Some(GroupLimit::First { limit: 10 }),
                (),
            )
        },
    );
}

/// The `(i64 id, sparse free-text phrase)` dataset shared by the row-key and
/// dedicated-extractor variants of the mixed int+string GROUP BY, so the two are
/// a direct A/B over byte-for-byte identical input.
fn int_string_dataset(rows: usize) -> Vec<RecordBatch> {
    let sch = schema(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("phrase", DataType::Utf8View, false),
    ]);
    let dict = string_dict((rows * 2 / 100).max(1), "search query ", 20);
    let block = dict_block(&dict);
    let mut rng = Rng::new(10);
    batch_sizes(rows)
        .map(|n| {
            batch(
                &sch,
                vec![
                    i64_keys(&mut rng, n, (rows / 8).max(1)),
                    sparse_string_col(&mut rng, n, &block, 0.85),
                ],
            )
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Filter / Contains scenarios
// ---------------------------------------------------------------------------

fn bench_filter(c: &mut Criterion, d: &DataFlowDispatcher) {
    let rows = total_rows();

    // (a) Substring match (`LIKE '%needle%'`) over long, buffer-backed strings,
    //     then count survivors. Exercises the once-per-buffer needle scan.
    bench(
        c,
        d,
        "filter/contains_substring",
        rows,
        || {
            let sch = schema(vec![Field::new("s", DataType::Utf8View, false)]);
            // ~2% of the *distinct* URLs carry the needle (so ~2% of rows match)
            // — the matching strings are their own dict entries, exactly as a
            // real dictionary-decoded scan presents them to the substring scan.
            let mut dict = string_dict((rows / 10).max(1), "http://example.com/path", 88);
            let mut seed = Rng::new(201);
            for s in dict.iter_mut() {
                if seed.chance(0.02) {
                    let mid = s.len() / 2;
                    s.insert_str(mid, "google");
                }
            }
            let block = dict_block(&dict);
            let mut rng = Rng::new(20);
            batch_sizes(rows)
                .map(|n| batch(&sch, vec![string_col(&mut rng, n, &block)]))
                .collect()
        },
        |s| {
            s.filter(|| {
                let mut contains = Contains::new("google");
                move |b: &RecordBatch, _: &mut SlabAllocator, scratch: &mut SelectionScratch| {
                    let col = b
                        .column(0)
                        .as_any()
                        .downcast_ref::<StringViewArray>()
                        .unwrap();
                    let mask = contains.run(col);
                    collect_selected_indices(&mask, &mut scratch.positions);
                    RowSelection::Indices
                }
            })
            .aggregate::<i64>(vec![AggregationSlot::new(
                AggregationKind::CountStar,
                0,
                DataType::Int64,
            )])
        },
    );

    // (b) Non-empty string predicate (`<> ''`) over a mostly-blank column.
    bench(
        c,
        d,
        "filter/string_nonempty",
        rows,
        || {
            let sch = schema(vec![Field::new("s", DataType::Utf8View, false)]);
            let dict = string_dict((rows * 6 / 100).max(1), "search query ", 31);
            let block = dict_block(&dict);
            let mut rng = Rng::new(21);
            batch_sizes(rows)
                .map(|n| batch(&sch, vec![sparse_string_col(&mut rng, n, &block, 0.868)]))
                .collect()
        },
        |s| {
            s.filter(|| {
                move |b: &RecordBatch, _: &mut SlabAllocator, scratch: &mut SelectionScratch| {
                    keep_nonempty(b, 0, &mut scratch.positions)
                }
            })
            .aggregate::<i64>(vec![AggregationSlot::new(
                AggregationKind::CountStar,
                0,
                DataType::Int64,
            )])
        },
    );

    // (c) Integer point lookup (`= const`) — a single matching id in a sea of
    //     near-unique 64-bit ids.
    {
        // Target a value that exists in the id space (index 7 mapped the same way).
        let target = 7i64.wrapping_mul(0x100_0001);
        bench(
            c,
            d,
            "filter/int_equality",
            rows,
            || {
                let sch = schema(vec![Field::new("id", DataType::Int64, false)]);
                let distinct = (rows / 2).max(1);
                let mut rng = Rng::new(22);
                batch_sizes(rows)
                    .map(|n| batch(&sch, vec![i64_keys(&mut rng, n, distinct)]))
                    .collect()
            },
            move |s| {
                s.filter(move || {
                    move |b: &RecordBatch, _: &mut SlabAllocator, scratch: &mut SelectionScratch| {
                        keep_i64_eq(b, 0, target, &mut scratch.positions)
                    }
                })
                .aggregate::<i64>(vec![AggregationSlot::new(
                    AggregationKind::CountStar,
                    0,
                    DataType::Int64,
                )])
            },
        );
    }
}

// ---------------------------------------------------------------------------
// ORDER BY ... LIMIT scenarios
// ---------------------------------------------------------------------------

fn bench_order_by(c: &mut Criterion, d: &DataFlowDispatcher) {
    let rows = total_rows();

    // (a) ORDER BY <i64 timestamp> LIMIT 10, behind a non-empty filter. Each
    //     worker keeps a bounded top-N; cheap per-row compare dominates.
    bench(
        c,
        d,
        "order_by/int_limit",
        rows,
        || {
            let sch = schema(vec![
                Field::new("t", DataType::Int64, false),
                Field::new("s", DataType::Utf8View, false),
            ]);
            let dict = string_dict((rows * 6 / 100).max(1), "search query ", 31);
            let block = dict_block(&dict);
            let mut rng = Rng::new(30);
            batch_sizes(rows)
                .map(|n| {
                    batch(
                        &sch,
                        vec![
                            i64_keys(&mut rng, n, rows.max(1)),
                            sparse_string_col(&mut rng, n, &block, 0.868),
                        ],
                    )
                })
                .collect()
        },
        |s| {
            s.filter(|| {
                move |b: &RecordBatch, _: &mut SlabAllocator, scratch: &mut SelectionScratch| {
                    keep_nonempty(b, 1, &mut scratch.positions)
                }
            })
            .order_by_limit(vec![OrderBy::new(0, false, false)], 10)
        },
    );

    // (b) ORDER BY <string> LIMIT 10, behind the same filter — string compares
    //     in the top-N.
    bench(
        c,
        d,
        "order_by/string_limit",
        rows,
        || {
            let sch = schema(vec![Field::new("s", DataType::Utf8View, false)]);
            let dict = string_dict((rows * 6 / 100).max(1), "search query ", 31);
            let block = dict_block(&dict);
            let mut rng = Rng::new(31);
            batch_sizes(rows)
                .map(|n| batch(&sch, vec![sparse_string_col(&mut rng, n, &block, 0.868)]))
                .collect()
        },
        |s| {
            s.filter(|| {
                move |b: &RecordBatch, _: &mut SlabAllocator, scratch: &mut SelectionScratch| {
                    keep_nonempty(b, 0, &mut scratch.positions)
                }
            })
            .order_by_limit(vec![OrderBy::new(0, false, false)], 10)
        },
    );

    // (c) ORDER BY <i64>, <string> LIMIT 10 — multi-key top-N tie-breaking.
    bench(
        c,
        d,
        "order_by/multi_limit",
        rows,
        || {
            let sch = schema(vec![
                Field::new("t", DataType::Int64, false),
                Field::new("s", DataType::Utf8View, false),
            ]);
            let dict = string_dict((rows * 6 / 100).max(1), "search query ", 31);
            let block = dict_block(&dict);
            let mut rng = Rng::new(32);
            batch_sizes(rows)
                .map(|n| {
                    batch(
                        &sch,
                        vec![
                            i64_keys(&mut rng, n, rows.max(1)),
                            sparse_string_col(&mut rng, n, &block, 0.868),
                        ],
                    )
                })
                .collect()
        },
        |s| {
            s.filter(|| {
                move |b: &RecordBatch, _: &mut SlabAllocator, scratch: &mut SelectionScratch| {
                    keep_nonempty(b, 1, &mut scratch.positions)
                }
            })
            .order_by_limit(
                vec![OrderBy::new(0, false, false), OrderBy::new(1, false, false)],
                10,
            )
        },
    );
}

// ---------------------------------------------------------------------------
// GLOBAL aggregate scenarios (no GROUP BY) — the `aggregate` operator, a
// straight per-row reduction over a whole column with no hash table. A
// multi-aggregate global reduction (e.g. `SUM, COUNT(*), AVG`) lives here; its hot
// loop is a column reduction that must stay vectorised. A regression that turns
// the reduction scalar (e.g. a loop-carried runtime branch the autovectoriser
// won't lift) ~halves throughput but is invisible to the GROUP BY benches above,
// which exercise the hash-table path instead.
// ---------------------------------------------------------------------------

fn bench_aggregate(c: &mut Criterion, d: &DataFlowDispatcher) {
    let rows = total_rows();

    // SUM(v0), COUNT(*), and AVG(v1) — which DuckDB lowers to
    // SUM(v1) + COUNT(v1). Four slots, all reducing the same two i16 columns in
    // one pass, no grouping. `i64` accumulator (16-bit columns can't overflow it),
    // matching the planner's width choice for these columns.
    bench(
        c,
        d,
        "aggregate/global_multi",
        rows,
        || {
            let sch = schema(vec![
                Field::new("v0", DataType::Int16, false),
                Field::new("v1", DataType::Int16, false),
            ]);
            let mut rng = Rng::new(40);
            batch_sizes(rows)
                .map(|n| {
                    batch(
                        &sch,
                        vec![i16_value(&mut rng, n, 2_000), i16_value(&mut rng, n, 2_000)],
                    )
                })
                .collect()
        },
        |s| {
            let slots = vec![
                AggregationSlot::new(AggregationKind::Sum, 0, DataType::Decimal128(38, 0)),
                AggregationSlot::new(AggregationKind::CountStar, 0, DataType::Int64),
                AggregationSlot::new(AggregationKind::Sum, 1, DataType::Decimal128(38, 0)),
                AggregationSlot::new(AggregationKind::Count, 1, DataType::Int64),
            ];
            s.aggregate::<i64>(slots)
        },
    );
}

// ---------------------------------------------------------------------------
// Harness: one shared Dispatch (worker pool) for every scenario.
// ---------------------------------------------------------------------------

fn main() {
    let workers = worker_count();
    let buffers = ring_buffers();
    eprintln!(
        "dispatch operator benches: {workers} workers, {buffers} ring buffers, {} rows",
        total_rows()
    );

    let dispatch = Dispatch::spin_up(workers, buffers, None);
    let dispatcher = dispatch.dispatcher().clone();

    let mut c = Criterion::default().configure_from_args();
    bench_group_by(&mut c, &dispatcher);
    bench_filter(&mut c, &dispatcher);
    bench_order_by(&mut c, &dispatcher);
    bench_aggregate(&mut c, &dispatcher);
    c.final_summary();

    dispatch.exit();
}
