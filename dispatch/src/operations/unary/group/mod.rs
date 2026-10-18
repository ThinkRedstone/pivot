//! Parallel GROUP BY operator.
//!
//! # Architecture
//!
//! GROUP BY is a pipeline breaker: it must consume all input before producing
//! any output. The implementation has two phases, both fully parallel:
//!
//! ## Phase 1: Consume (parallel per worker, no coordination)
//!
//! Each worker owns a [`Group`] containing a local [`AggregatedTable`]. For
//! every input [`RecordBatch`], the worker:
//!
//! 1. Hashes all keys in the group column (all workers share the same
//!    [`RandomState`] so hashes are consistent).
//! 2. Inserts each key/value into its local hash table via
//!    [`AggregatedTable::consume_batch`]. Duplicate keys within the same
//!    worker are merged immediately (e.g. counts are summed).
//! 3. If the hash table exceeds its load threshold, a new, larger table is
//!    created and subsequent rows go there. The old table is kept — its
//!    entries will be merged in phase 2.  See [Why partitioned merging works](#why-partitioned-merging-works-across-different-table-sizes).
//!
//! For high-cardinality integer keys a worker instead **switches to radix**
//! partway through: rather than growing its in-place table further it scatters
//! later rows into [`RADIX_PARTITIONS`] per-partition buffers (no probing),
//! deferring their aggregation to a finer-grained, cache-resident phase-2 merge.
//! Strings and low-cardinality keys never switch and stay fully in-place.
//!
//! When consumption finishes, each worker sends its tables (plus any radix
//! buffers) to a shared gather barrier and transitions into a [`GroupOutputter`].
//!
//! ## Phase 2: Output / Merge (parallel via work-stealing)
//!
//! The final worker to reach the gather barrier collects every worker's tables
//! and publishes a power-of-two number of independent [`PartitionJob`]s to a
//! shared [`Injector`]. The count is chosen per query in
//! `GroupOutputter::create_partition_jobs` from the input volume and the
//! worker count. Each partition covers a disjoint range of hash values
//! (determined by the top `log2(num_partitions)` bits), so the jobs are
//! embarrassingly parallel.
//!
//! All workers then steal and execute jobs. Each [`PartitionJob`]:
//!
//! 1. Scans every source table for entries belonging to its partition.
//! 2. Merges them into a single result table (see `output::merge` for the
//!    batched, prefetched merge strategy).
//! 3. Converts the result table into an Arrow [`RecordBatch`] via the
//!    `output::accumulator` combinator (key columns from the
//!    [`KeyExtractor`], value columns from the [`AggregationValue`]) and
//!    sends it downstream.
//!
//! When any worker switched to radix, the same machinery runs at
//! [`RADIX_PARTITIONS`] granularity, and each job additionally aggregates its
//! partition's scatter buffers (where most of phase 1's aggregation was deferred)
//! alongside the in-place stacks.
//!
//! ## Why partitioned merging works across different table sizes
//!
//! The hash table uses **top-bit slot placement**: `slot = hash >> (64 - log2(capacity))`.
//! This means the top bits of the hash determine the slot, and the partition
//! (top 6 bits for 64 partitions) is always a prefix of the slot index.
//!
//! Because of this, entries for partition P occupy a predictable, contiguous
//! slot range in *any* power-of-2 table, regardless of its size:
//!
//! ```text
//! 128-slot table:  partition 0 = slots [0, 2)    partition 1 = slots [2, 4)   ...
//! 256-slot table:  partition 0 = slots [0, 4)    partition 1 = slots [4, 8)   ...
//! 1024-slot table: partition 0 = slots [0, 16)   partition 1 = slots [16, 32) ...
//! ```
//!
//! The ranges differ in width but always align — a 128-slot table's partition 0
//! range is a subset of a 256-slot table's partition 0 range (same top bits,
//! just fewer bits resolved). This lets the merge phase scan tables of mixed
//! sizes and route entries to the correct partition using only the hash value,
//! without needing to know the source table's capacity.
//!
//! ## Module layout
//!
//! - [`keys`] — the [`KeyExtractor`] trait and implementations
//!   ([`IntKeyExtractor`], [`StringKeyExtractor`]), each co-located with its key
//!   type (e.g. `keys::string` owns [`ArenaKey`])
//! - [`values`] — the [`AggregationValue`] trait and its container
//!   implementations (`Compiled`, `Dynamic`), the per-op folds (`Count`, `Sum`,
//!   …), plus [`AggregationKind`]/[`AggregationSlot`]
//! - [`hashtables`] — [`AggregatedTable`], [`HashTable`](hashtables::HashTable),
//!   and associated type machinery
//! - [`output`] — the merge phase: [`GroupOutputter`] drives it, one
//!   [`PartitionJob`] per hash partition merges its sources
//!   (`output::merge`), a pushed top-k prunes what cannot reach the result
//!   (`output::topk_pruning`), and `output::accumulator` packs merged groups
//!   into output batches
//! - [`arena`] — shared string storage backing string keys
//! - [`factory`] — [`GroupFactory`] for creating per-worker [`Group`] instances

pub(crate) mod arena;
mod factory;
mod hll;
mod keys;
mod output;
mod values;

use output::topk_pruning::{SharedBinTotals, TopKThreshold};
pub use output::{GroupOutputter, PartitionJob};

pub use factory::GroupFactory;
mod hashtables;

pub use keys::{
    ArenaKey, HashOnlyIntKeyExtractor, IntKeyExtractor, IntPairKeyExtractor, IntStrKeyExtractor,
    KeyExtractor, RowKeyExtractor, RowKeySchema, StringKeyExtractor,
};
pub(crate) use values::cast_value_column;
pub use values::{
    AggregationKind, AggregationSlot, AggregationValue, Cell, Compiled, Count, CountSlot,
    CountValidSlot, Distinct, Dynamic, F64Cell, F64Max, F64Min, F64Sum, Fold, IntCell, IntRead,
    Max, MaxSlot, Min, MinSlot, NoRead, OpTuple, Read, SharedContext, StrMax, StrMin, StrRead, Sum,
    SumSlot, U128Max, U128Min, U128Sum, WideCell, WideSum, WorkerContext,
};

use crate::gather_barrier::GatherBarrier;
use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::unary::group::hashtables::{
    AggregatedTable, AggregatedTableOutput, RadixConfig,
};
use ahash::RandomState;
use arena::SharedArena;
use arrow_array::RecordBatch;
use arrow_schema::{ArrowError, DataType};
use crossbeam_deque::Injector;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use thiserror::Error;
use unary::pipeline_breaker::Consumer;

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Arrow(#[from] ArrowError),
    #[error("{0}")]
    Channel(#[from] crate::operations::channels::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// A LIMIT pushed down into a grouped aggregate, so each partition emits only
/// the rows the downstream LIMIT can keep instead of every group.
///
/// Both forms are decomposable across the hash-disjoint partitions: the global
/// answer is recovered by re-applying the same LIMIT (and any ORDER BY) above.
#[derive(Clone, Copy, Debug)]
pub enum GroupLimit {
    /// `ORDER BY <slot> DESC LIMIT n`, ordering by `V::sort_key(value, slot)`.
    ///
    /// Without `with_ties`, each partition keeps only its top-`n` groups in a
    /// heap; groups tied at the boundary may be dropped, which is a valid
    /// answer when `slot` is the only order key. With `with_ties`
    /// (`ORDER BY <slot> DESC, <tiebreaks...> LIMIT n`), no group tied with
    /// the `n`-th best on `slot` may be dropped, since the downstream sort's
    /// tiebreaks decide among them: there is no per-partition heap, every
    /// merged group is emitted, and the merge only skips groups strictly
    /// dominated on `slot` by `n` others.
    TopK {
        slot: usize,
        limit: usize,
        with_ties: bool,
    },
    /// Plain `LIMIT n` with no ORDER BY: keep any `n` groups from this partition
    /// (SQL leaves which rows arbitrary, so the first `n` encountered suffice).
    First { limit: usize },
}

impl GroupLimit {
    /// The most rows this limit can let through, regardless of form — used to size
    /// the output builders.
    pub(crate) fn row_limit(self) -> usize {
        match self {
            // Keeping ties means every surviving group is emitted.
            GroupLimit::TopK {
                with_ties: true, ..
            } => usize::MAX,
            GroupLimit::TopK { limit, .. } | GroupLimit::First { limit } => limit,
        }
    }
}

/// Number of radix partitions for the scatter + merge of high-cardinality
/// (switched) workers. Far finer than the in-place merge's job count so each
/// radix target stays cache-resident at high group counts.
const RADIX_PARTITIONS: usize = 4096;

/// How many scatter streams the whole pool should aim to stay under. This is
/// a target, not a hard cap: the one-bucket-per-worker floor below may exceed
/// it on enormous pools.
///
/// Every (worker, bucket) pair is one small stream the merge later walks,
/// prefetch-warms, and tears down, so those fixed costs grow with
/// `workers x buckets` while the useful bytes per stream shrink. The scatter
/// side pays per stream too: each worker keeps one partly written row and one
/// buffer header hot per bucket, and past a few hundred buckets that frontier
/// no longer fits in L1. [`get_scatter_bucket_count_for_worker`] scales the
/// per-worker bucket count down as the pool grows to hold the total near this
/// target (chosen so pools of ~32 workers or fewer keep the full
/// [`RADIX_PARTITIONS`]; a 192-worker pool gets 512 buckets, where both sides'
/// per-stream costs stop throttling the memory bandwidth the merge can use).
const TARGET_SCATTER_STREAMS: usize = 1 << 17;

/// How many scatter buckets each worker should use when `total_workers`
/// workers share the pool.
///
/// Every worker keeps one output stream per bucket, so the pool as a whole
/// holds `total_workers * buckets` streams, and each stream costs a fixed
/// walk and teardown in the merge phase whether or not any rows landed in
/// it. Small pools can afford the full [`RADIX_PARTITIONS`]; large pools get
/// a smaller bucket count so the pool-wide stream total stays near
/// [`TARGET_SCATTER_STREAMS`]. The result is always a power of two (rows are
/// routed to buckets by hash bits) and never less than one bucket per
/// worker, because the merge phase creates one job per bucket and fewer
/// buckets than workers would leave cores idle.
fn get_scatter_bucket_count_for_worker(total_workers: usize) -> usize {
    let budget = (TARGET_SCATTER_STREAMS / total_workers.max(1)).max(1);
    // The largest power of two at or below the budget.
    (1usize << budget.ilog2())
        .max(total_workers.next_power_of_two())
        .min(RADIX_PARTITIONS)
}

/// Per-worker GROUP BY consumer.
///
/// During the consume phase, each worker owns a `Group` that hashes incoming
/// rows and inserts them into its local [`AggregatedTable`]. When consumption
/// finishes, the accumulated tables are published to a shared gather barrier,
/// and the `Group` transitions into a [`GroupOutputter`] for the merge phase.
pub struct Group<K: KeyExtractor, V: AggregationValue + ?Sized> {
    key_cols: Vec<usize>,
    /// Slots drive the per-batch value reader (which column / `COUNT` vs `SUM`).
    value_slots: Vec<AggregationSlot>,
    /// The value's shared, read-side context, built once from the slots + value
    /// arena (like `key_config`); resolves string keys during merge/output.
    shared_context: V::SharedContext,
    key_config: K::Config,

    aggregated_table: AggregatedTable<K, V>,
    gather: Arc<GatherBarrier<AggregatedTableOutput<K, V>>>,
    outputter: GroupOutputter<K, V>,
}

impl<K: KeyExtractor, V: AggregationValue + ?Sized> Group<K, V> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        key_arena: Arc<SharedArena>,
        value_arena: Arc<SharedArena>,
        state: RandomState,
        injectors: Arc<Vec<Injector<PartitionJob<K, V>>>>,
        key_cols: Vec<usize>,
        value_slots: Vec<AggregationSlot>,
        key_config: K::Config,
        output_limit: Option<GroupLimit>,
        count_only: bool,
        gather: Arc<GatherBarrier<AggregatedTableOutput<K, V>>>,
        partition_jobs_injected: Arc<AtomicBool>,
        radix: RadixConfig,
        topk_threshold: Arc<TopKThreshold>,
        shared_bin_totals: Arc<SharedBinTotals>,
    ) -> Self {
        let shared_context = <V::SharedContext as SharedContext>::build(&value_slots, &value_arena);
        let value_output_types: Arc<[DataType]> =
            value_slots.iter().map(|s| s.output_type.clone()).collect();
        // A string extreme persists its winner lazily during the in-place fold, so
        // it must not take the radix scatter path (which seeds — and thus
        // persists — every row's string before any comparison). Disable the switch
        // when any value slot is a string extreme; other signatures keep radix.
        let radix = if value_slots.iter().any(|s| s.is_string_extreme()) {
            radix.without_radix()
        } else {
            radix
        };
        // The per-worker write context is spawned from the shared one (a fresh
        // `WorkerArena` for a string value, `()` for numeric).
        let aggregated_table = AggregatedTable::new(
            state,
            key_arena.clone(),
            shared_context.clone(),
            shared_context.worker(),
            radix,
            output::topk_pruning::prunable_topk_slot(output_limit, &value_slots),
        );
        Self {
            key_cols,
            value_slots,
            shared_context: shared_context.clone(),
            key_config: key_config.clone(),
            outputter: GroupOutputter::new(
                key_arena,
                injectors,
                partition_jobs_injected,
                key_config,
                shared_context,
                value_output_types,
                output_limit,
                count_only,
                topk_threshold,
                shared_bin_totals,
            ),
            gather,
            aggregated_table,
        }
    }
}

impl<K: KeyExtractor, V: AggregationValue + ?Sized> Consumer<RecordBatch, RecordBatch>
    for Group<K, V>
{
    type Outputter = GroupOutputter<K, V>;

    fn consume(
        &mut self,
        batch: RecordBatch,
        _sender: &mut dyn Sender<RecordBatch>,
    ) -> unary::Result<()> {
        self.aggregated_table.consume_batch(
            &batch,
            &self.key_cols,
            &self.value_slots,
            &self.key_config,
            &self.shared_context,
        );
        Ok(())
    }

    fn into_outputter(mut self) -> unary::Result<Option<Self::Outputter>> {
        let (tables, topk_bin_totals) = self.aggregated_table.flush();
        if let Some(totals) = topk_bin_totals {
            self.outputter.add_bin_totals(totals);
        }
        self.gather.arrive(tables, |values| {
            self.outputter.create_partition_jobs(values)
        });
        Ok(Some(self.outputter))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;
    use crate::operations::unary::group::keys::{
        IntKeyExtractor, IntStrKeyExtractor, StringKeyExtractor,
    };
    use crate::operations::unary::test_utils::{CollectSender, run_consumers};
    use arrow_array::types::Int32Type;
    use arrow_array::{ArrayRef, Int32Array, RecordBatch, StringViewArray};
    use arrow_schema::{DataType, Field, Schema};

    type IntExtractor = IntKeyExtractor<Int32Type>;
    type CountValue = Compiled<(CountSlot,)>;
    type SumValue = Compiled<(SumSlot<Int32Type>,)>;

    #[test]
    fn scatter_buckets_shrink_as_the_pool_grows() {
        // Small pools keep the full per-worker resolution; large pools trade
        // it away to hold the total stream count near the budget.
        assert_eq!(get_scatter_bucket_count_for_worker(8), RADIX_PARTITIONS);
        assert_eq!(get_scatter_bucket_count_for_worker(32), RADIX_PARTITIONS);
        assert_eq!(get_scatter_bucket_count_for_worker(96), 1024);
        assert_eq!(get_scatter_bucket_count_for_worker(190), 512);
        // Never fewer buckets than workers, whatever the budget says.
        assert_eq!(get_scatter_bucket_count_for_worker(400), 512);
    }

    fn batch_with_column(values: &[i32]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("key", DataType::Int32, false)]));
        let col: ArrayRef = Arc::new(Int32Array::from(values.to_vec()));
        RecordBatch::try_new(schema, vec![col]).unwrap()
    }

    /// An `Int32` key column (0) plus an `Int32` value column (1), for `SUM`.
    fn keyed_i32_batch(keys: &[i32], vals: &[i32]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int32, false),
            Field::new("val", DataType::Int32, false),
        ]));
        let cols: Vec<ArrayRef> = vec![
            Arc::new(Int32Array::from(keys.to_vec())),
            Arc::new(Int32Array::from(vals.to_vec())),
        ];
        RecordBatch::try_new(schema, cols).unwrap()
    }

    /// A single `Utf8View` key column.
    fn string_key_batch(values: &[&str]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "key",
            DataType::Utf8View,
            false,
        )]));
        let col: ArrayRef = Arc::new(StringViewArray::from(values.to_vec()));
        RecordBatch::try_new(schema, vec![col]).unwrap()
    }

    fn count_slots() -> Vec<AggregationSlot> {
        vec![AggregationSlot::new(
            AggregationKind::CountStar,
            0,
            DataType::Int64,
        )]
    }

    /// Common case: `Int32` keys, `COUNT(*)`, key column 0, no LIMIT, default radix.
    fn run_group(worker_batches: Vec<Vec<RecordBatch>>) -> CollectSender {
        run_group_full::<IntExtractor, CountValue>(
            worker_batches,
            vec![0],
            count_slots(),
            None,
            RadixConfig::DEFAULT,
        )
    }

    #[test]
    fn batch_wider_than_record_batch_size() {
        // A single input batch wider than RECORD_BATCH_SIZE (8192) must be
        // consumed via windowing rather than overflowing the per-batch scratch.
        // This is the shape a GROUP BY feeding another GROUP BY produces.
        let values: Vec<i32> = (0..20_000).map(|i| i % 5_000).collect();
        let sender = run_group(vec![vec![batch_with_column(&values)]]);
        // 5_000 distinct keys, each appearing 4 times.
        assert_eq!(sender.total_rows(), 5_000);
    }

    /// Common case with a chosen [`RadixConfig`] — a small one forces the radix
    /// switch + scatter merge to run within the test slab pool.
    fn run_group_with_radix(
        worker_batches: Vec<Vec<RecordBatch>>,
        radix: RadixConfig,
    ) -> CollectSender {
        run_group_full::<IntExtractor, CountValue>(
            worker_batches,
            vec![0],
            count_slots(),
            None,
            radix,
        )
    }

    /// General harness: choose the key/value extractors, key columns, aggregates,
    /// LIMIT pushdown, and radix config. The common-case wrappers above cover
    /// `Int32` keys + `COUNT(*)`.
    fn run_group_full<K: KeyExtractor<Config: Default>, V: AggregationValue + ?Sized>(
        worker_batches: Vec<Vec<RecordBatch>>,
        key_cols: Vec<usize>,
        value_slots: Vec<AggregationSlot>,
        output_limit: Option<GroupLimit>,
        radix: RadixConfig,
    ) -> CollectSender {
        init_test_free_pool(64);
        let worker_count = worker_batches.len();
        let key_arena = SharedArena::new(64);
        let value_arena = SharedArena::new(64);
        let state = RandomState::new();
        let injector = Arc::new(vec![Injector::new()]);
        let partition_jobs_injected = Arc::new(AtomicBool::new(false));
        let gather = Arc::new(GatherBarrier::new(worker_count));
        let topk_threshold = Arc::new(TopKThreshold::for_limit(output_limit));
        let shared_bin_totals = Arc::new(SharedBinTotals::new(1));

        let groups: Vec<_> = (0..worker_count)
            .map(|_| {
                Group::<K, V>::new(
                    key_arena.clone(),
                    value_arena.clone(),
                    state.clone(),
                    injector.clone(),
                    key_cols.clone(),
                    value_slots.clone(),
                    K::Config::default(),
                    output_limit,
                    false,
                    gather.clone(),
                    partition_jobs_injected.clone(),
                    radix,
                    topk_threshold.clone(),
                    shared_bin_totals.clone(),
                )
            })
            .collect();

        run_consumers(groups, worker_batches)
    }

    #[test]
    fn single_worker_single_batch() {
        let sender = run_group(vec![vec![batch_with_column(&[1, 2, 3, 4, 5])]]);

        assert_eq!(sender.total_rows(), 5);
        assert_eq!(sender.sorted_i32_column(0), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn single_worker_duplicates_merged() {
        let sender = run_group(vec![vec![batch_with_column(&[1, 1, 2, 2, 3])]]);

        assert_eq!(sender.total_rows(), 3);
        assert_eq!(sender.sorted_i32_column(0), vec![1, 2, 3]);
    }

    #[test]
    fn single_worker_multiple_batches() {
        let sender = run_group(vec![vec![
            batch_with_column(&[1, 2]),
            batch_with_column(&[2, 3]),
            batch_with_column(&[3, 4]),
        ]]);

        assert_eq!(sender.total_rows(), 4);
        assert_eq!(sender.sorted_i32_column(0), vec![1, 2, 3, 4]);
    }

    #[test]
    fn two_workers_disjoint() {
        let sender = run_group(vec![
            vec![batch_with_column(&[1, 2, 3])],
            vec![batch_with_column(&[4, 5, 6])],
        ]);

        assert_eq!(sender.total_rows(), 6);
        assert_eq!(sender.sorted_i32_column(0), vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn two_workers_overlapping() {
        let sender = run_group(vec![
            vec![batch_with_column(&[1, 2, 3])],
            vec![batch_with_column(&[2, 3, 4])],
        ]);

        assert_eq!(sender.total_rows(), 4);
        assert_eq!(sender.sorted_i32_column(0), vec![1, 2, 3, 4]);
    }

    #[test]
    fn empty_input() {
        let sender = run_group(vec![vec![batch_with_column(&[])]]);

        assert_eq!(sender.total_rows(), 0);
    }

    #[test]
    fn many_distinct_keys() {
        let values: Vec<i32> = (0..1000).collect();
        let sender = run_group(vec![vec![batch_with_column(&values)]]);

        assert_eq!(sender.total_rows(), 1000);
    }

    #[test]
    fn four_workers_all_same_key() {
        let sender = run_group(vec![
            vec![batch_with_column(&[42, 42, 42])],
            vec![batch_with_column(&[42, 42])],
            vec![batch_with_column(&[42])],
            vec![batch_with_column(&[42, 42, 42, 42])],
        ]);

        assert_eq!(sender.total_rows(), 1);
    }

    /// (key, count) pairs from a finished `COUNT(*)` group-by, sorted by key.
    fn group_counts(sender: &CollectSender) -> Vec<(i32, i64)> {
        let mut pairs: Vec<_> = sender
            .i32_column(0)
            .into_iter()
            .zip(sender.i64_column(1))
            .collect();
        pairs.sort();
        pairs
    }

    #[test]
    fn counts_aggregate_within_and_across_workers() {
        let batches = vec![
            vec![batch_with_column(&[1, 2, 3])],
            vec![batch_with_column(&[2, 2, 3, 4])],
        ];

        let sender = run_group(batches);

        assert_eq!(group_counts(&sender), vec![(1, 1), (2, 3), (3, 2), (4, 1)]);
    }

    #[test]
    fn counts_survive_in_place_stack_growth() {
        let mut values: Vec<i32> = (0..3000).collect();
        values.extend(0..3000);

        let sender = run_group(vec![vec![batch_with_column(&values)]]);

        assert_eq!(
            group_counts(&sender),
            (0..3000).map(|k| (k, 2)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn radix_switch_single_worker_aggregates_counts() {
        // A small config makes a radix-eligible (integer) worker switch to scatter
        // after ~a couple hundred keys and scatter into 16 partitions, exercising
        // the whole radix path within the test pool.
        let radix = RadixConfig {
            switch_threshold: 256,
            partitions: 16,
            ..RadixConfig::DEFAULT
        };
        let mut values: Vec<i32> = (0..500).collect();
        values.extend(0..500);

        let sender = run_group_with_radix(vec![vec![batch_with_column(&values)]], radix);

        assert_eq!(
            group_counts(&sender),
            (0..500).map(|k| (k, 2)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn radix_switch_merges_across_workers() {
        // Two workers each switch to scatter; the radix merge has to combine both
        // workers' per-partition buffers and pre-switch stacks for every key.
        let radix = RadixConfig {
            switch_threshold: 256,
            partitions: 16,
            ..RadixConfig::DEFAULT
        };
        let worker = || vec![batch_with_column(&(0..500).collect::<Vec<_>>())];

        let sender = run_group_with_radix(vec![worker(), worker()], radix);

        assert_eq!(
            group_counts(&sender),
            (0..500).map(|k| (k, 2)).collect::<Vec<_>>()
        );
    }

    /// (string key, count) pairs from a finished string group-by, sorted by key.
    fn string_group_pairs(sender: &CollectSender) -> Vec<(String, i64)> {
        let mut pairs: Vec<_> = sender
            .string_column(0)
            .into_iter()
            .zip(sender.i64_column(1))
            .collect();
        pairs.sort();
        pairs
    }

    #[test]
    fn string_keys_dedup_and_count() {
        let batches = vec![vec![string_key_batch(&["a", "b", "a", "c", "b", "a"])]];

        let sender = run_group_full::<StringKeyExtractor, CountValue>(
            batches,
            vec![0],
            count_slots(),
            None,
            RadixConfig::DEFAULT,
        );

        assert_eq!(
            string_group_pairs(&sender),
            vec![
                ("a".to_string(), 3),
                ("b".to_string(), 2),
                ("c".to_string(), 1)
            ]
        );
    }

    #[test]
    fn sum_aggregates_value_per_group() {
        // key 1 -> 10+30, key 2 -> 20, key 3 -> 5+5.
        let batch = keyed_i32_batch(&[1, 2, 1, 3, 3], &[10, 20, 30, 5, 5]);
        let slots = vec![AggregationSlot::new(
            AggregationKind::Sum,
            1,
            DataType::Decimal128(38, 0),
        )];

        let sender = run_group_full::<IntExtractor, SumValue>(
            vec![vec![batch]],
            vec![0],
            slots,
            None,
            RadixConfig::DEFAULT,
        );

        // A SUM renders as a Decimal128 column.
        let mut pairs: Vec<(i32, i128)> = sender
            .i32_column(0)
            .into_iter()
            .zip(sender.decimal128_column(1))
            .collect();
        pairs.sort();
        assert_eq!(pairs, vec![(1, 40), (2, 20), (3, 10)]);
    }

    #[test]
    fn top_k_limits_output_keeping_partition_maxima() {
        // key 0 dominates (count 100); keys 1..1000 appear once. ORDER BY count
        // LIMIT 1 is applied per partition, collapsing ~1000 groups to at most one
        // row per partition — always the partition's largest, so key 0 survives.
        let mut values: Vec<i32> = vec![0; 100];
        values.extend(1..1000);

        let sender = run_group_full::<IntExtractor, CountValue>(
            vec![vec![batch_with_column(&values)]],
            vec![0],
            count_slots(),
            Some(GroupLimit::TopK {
                slot: 0,
                limit: 1,
                with_ties: false,
            }),
            RadixConfig::DEFAULT,
        );

        let pairs = group_counts(&sender);
        assert!(pairs.contains(&(0, 100)), "dominant group kept");
        assert!(
            pairs.iter().all(|&(k, c)| k == 0 || c == 1),
            "every survivor is its partition's max"
        );
    }

    #[test]
    fn first_limit_caps_each_partition_without_sorting() {
        // A plain LIMIT (no ORDER BY) keeps *any* `limit` groups per partition.
        // 1000 distinct keys, each once: with `limit = 1` each partition emits at
        // most one of its groups, so the output is capped at one row per partition
        // and every survivor is a real (key, count == 1) group.
        let values: Vec<i32> = (0..1000).collect();

        let sender = run_group_full::<IntExtractor, CountValue>(
            vec![vec![batch_with_column(&values)]],
            vec![0],
            count_slots(),
            Some(GroupLimit::First { limit: 1 }),
            RadixConfig::DEFAULT,
        );

        let pairs = group_counts(&sender);
        assert!(!pairs.is_empty(), "some groups kept");
        assert!(
            pairs.iter().all(|&(k, c)| (0..1000).contains(&k) && c == 1),
            "every survivor is a real group"
        );
    }

    // ---- Row-encoded multi-column keys (`RowKeyExtractor`) ----

    /// A mixed-key batch: an `Int64` key, a `Utf8View` key, and an `Int32` value.
    fn mixed_key_batch(ids: &[i64], names: &[&str], vals: &[i32]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8View, false),
            Field::new("val", DataType::Int32, false),
        ]));
        let id: ArrayRef = Arc::new(arrow_array::Int64Array::from(ids.to_vec()));
        let name: ArrayRef = Arc::new(StringViewArray::from(names.to_vec()));
        let val: ArrayRef = Arc::new(Int32Array::from(vals.to_vec()));
        RecordBatch::try_new(schema, vec![id, name, val]).unwrap()
    }

    fn run_row_key_group<V: AggregationValue + ?Sized>(
        worker_batches: Vec<Vec<RecordBatch>>,
        key_cols: Vec<usize>,
        schema: RowKeySchema,
        value_slots: Vec<AggregationSlot>,
    ) -> CollectSender {
        run_row_key_group_radix::<V>(
            worker_batches,
            key_cols,
            schema,
            value_slots,
            RadixConfig::DEFAULT,
        )
    }

    fn run_row_key_group_radix<V: AggregationValue + ?Sized>(
        worker_batches: Vec<Vec<RecordBatch>>,
        key_cols: Vec<usize>,
        schema: RowKeySchema,
        value_slots: Vec<AggregationSlot>,
        radix: RadixConfig,
    ) -> CollectSender {
        init_test_free_pool(64);
        let worker_count = worker_batches.len();
        let key_arena = SharedArena::new(64);
        let value_arena = SharedArena::new(64);
        let state = RandomState::new();
        let injector = Arc::new(vec![Injector::new()]);
        let partition_jobs_injected = Arc::new(AtomicBool::new(false));
        let gather = Arc::new(GatherBarrier::new(worker_count));
        let topk_threshold = Arc::new(TopKThreshold::for_limit(None));
        let shared_bin_totals = Arc::new(SharedBinTotals::new(1));
        let groups: Vec<_> = (0..worker_count)
            .map(|_| {
                Group::<RowKeyExtractor, V>::new(
                    key_arena.clone(),
                    value_arena.clone(),
                    state.clone(),
                    injector.clone(),
                    key_cols.clone(),
                    value_slots.clone(),
                    schema.clone(),
                    None,
                    false,
                    gather.clone(),
                    partition_jobs_injected.clone(),
                    radix,
                    topk_threshold.clone(),
                    shared_bin_totals.clone(),
                )
            })
            .collect();
        run_consumers(groups, worker_batches)
    }

    /// Read the `(id, name, value)` output rows of a row-key group-by, sorted.
    fn row_key_rows(sender: &CollectSender) -> Vec<(i64, String, i64)> {
        let mut rows: Vec<(i64, String, i64)> = sender
            .i64_column(0)
            .into_iter()
            .zip(sender.string_column(1))
            .zip(sender.i64_column(2))
            .map(|((id, name), v)| (id, name, v))
            .collect();
        rows.sort();
        rows
    }

    /// Read the `(id, name, sum)` output rows of a row-key group-by whose value is
    /// a `SUM` (a `Decimal128` column), sorted.
    fn row_key_sum_rows(sender: &CollectSender) -> Vec<(i64, String, i128)> {
        let mut rows: Vec<(i64, String, i128)> = sender
            .i64_column(0)
            .into_iter()
            .zip(sender.string_column(1))
            .zip(sender.decimal128_column(2))
            .map(|((id, name), v)| (id, name, v))
            .collect();
        rows.sort();
        rows
    }

    /// `GROUP BY (Int64, Utf8View)` — each distinct tuple is one group, counted.
    #[test]
    fn row_key_mixed_int_string_counts() {
        let batch = mixed_key_batch(&[1, 1, 1, 2, 2], &["a", "b", "a", "a", "a"], &[0; 5]);
        let schema = RowKeySchema::new(
            vec![DataType::Int64, DataType::Utf8View],
            vec![false, false],
        );
        let sender = run_row_key_group::<Compiled<(CountSlot,)>>(
            vec![vec![batch]],
            vec![0, 1],
            schema,
            count_slots(),
        );
        assert_eq!(
            row_key_rows(&sender),
            vec![
                (1, "a".to_string(), 2),
                (1, "b".to_string(), 1),
                (2, "a".to_string(), 2),
            ]
        );
    }

    /// `IntStrKeyExtractor`: `GROUP BY (Int64, Utf8View)` through the dedicated
    /// int+string extractor instead of the row encoder. Same groups as
    /// [`row_key_mixed_int_string_counts`], but keyed on the native int beside the
    /// string's arena handle.
    #[test]
    fn int_string_mixed_counts() {
        let batch = mixed_key_batch(&[1, 1, 1, 2, 2], &["a", "b", "a", "a", "a"], &[0; 5]);
        let sender = run_group_full::<
            IntStrKeyExtractor<arrow_array::types::Int64Type>,
            Compiled<(CountSlot,)>,
        >(
            vec![vec![batch]],
            vec![0, 1],
            count_slots(),
            None,
            RadixConfig::DEFAULT,
        );
        assert_eq!(
            row_key_rows(&sender),
            vec![
                (1, "a".to_string(), 2),
                (1, "b".to_string(), 1),
                (2, "a".to_string(), 2),
            ]
        );
    }

    /// `IntStrKeyExtractor` across two workers: the same `(int, string)` group is
    /// seen by both, so the merge must resolve each persisted key through the
    /// shared arena (the `IntStrResolvedKey` path) and combine. Includes a
    /// > 12-byte string so the string rides an arena blob, not the inline header.
    #[test]
    fn int_string_two_workers_merge() {
        let long = "a-string-well-over-twelve-bytes";
        let sender = run_group_full::<
            IntStrKeyExtractor<arrow_array::types::Int64Type>,
            Compiled<(CountSlot,)>,
        >(
            vec![
                vec![mixed_key_batch(&[1, 2], &[long, "a"], &[0; 2])],
                vec![mixed_key_batch(&[1, 1], &[long, "b"], &[0; 2])],
            ],
            vec![0, 1],
            count_slots(),
            None,
            RadixConfig::DEFAULT,
        );
        assert_eq!(
            row_key_rows(&sender),
            vec![
                (1, long.to_string(), 2),
                (1, "b".to_string(), 1),
                (2, "a".to_string(), 1),
            ]
        );
    }

    /// Row-key SUM: each distinct `(id, name)` sums its value column. Includes a
    /// > 12-byte string to exercise the arena (non-inline) path on output.
    #[test]
    fn row_key_sums_per_group() {
        let long = "a-string-well-over-twelve-bytes";
        let batch = mixed_key_batch(&[1, 2, 1, 2], &["x", long, "x", "y"], &[10, 5, 30, 7]);
        let schema = RowKeySchema::new(
            vec![DataType::Int64, DataType::Utf8View],
            vec![false, false],
        );
        let slots = vec![AggregationSlot::new(
            AggregationKind::Sum,
            2,
            DataType::Decimal128(38, 0),
        )];
        let sender = run_row_key_group::<Compiled<(SumSlot<Int32Type>,)>>(
            vec![vec![batch]],
            vec![0, 1],
            schema,
            slots,
        );
        assert_eq!(
            row_key_sum_rows(&sender),
            vec![
                (1, "x".to_string(), 40),
                (2, long.to_string(), 5),
                (2, "y".to_string(), 7),
            ]
        );
    }

    /// Keys split across two workers must merge into the same groups.
    #[test]
    fn row_key_two_workers_merge() {
        let schema = RowKeySchema::new(
            vec![DataType::Int64, DataType::Utf8View],
            vec![false, false],
        );
        let sender = run_row_key_group::<Compiled<(CountSlot,)>>(
            vec![
                vec![mixed_key_batch(&[1, 2], &["a", "a"], &[0; 2])],
                vec![mixed_key_batch(&[1, 1], &["a", "b"], &[0; 2])],
            ],
            vec![0, 1],
            schema,
            count_slots(),
        );
        assert_eq!(
            row_key_rows(&sender),
            vec![
                (1, "a".to_string(), 2),
                (1, "b".to_string(), 1),
                (2, "a".to_string(), 1),
            ]
        );
    }

    /// `GROUP BY (Utf8View, Int64)` — a string in a *non-trailing* position, so
    /// it keeps its `u32` length prefix while the trailing int does not. A string
    /// longer than 12 bytes forces an arena (non-inline) blob, exercising
    /// prefixed-string decode where a fixed-width field follows the string.
    #[test]
    fn row_key_string_not_trailing() {
        let long = "a-string-well-over-twelve-bytes";
        // GROUP BY (name, id), summing the value column.
        let batch = mixed_key_batch(&[1, 2, 1, 2], &[long, "y", long, "y"], &[10, 5, 30, 7]);
        let schema = RowKeySchema::new(
            vec![DataType::Utf8View, DataType::Int64],
            vec![false, false],
        );
        let slots = vec![AggregationSlot::new(
            AggregationKind::Sum,
            2,
            DataType::Decimal128(38, 0),
        )];
        let sender = run_row_key_group::<Compiled<(SumSlot<Int32Type>,)>>(
            vec![vec![batch]],
            vec![1, 0], // name (col 1) then id (col 0)
            schema,
            slots,
        );
        // Output columns: k0 = name (str), k1 = id (i64), agg = sum (Decimal128).
        let mut rows: Vec<(String, i64, i128)> = sender
            .string_column(0)
            .into_iter()
            .zip(sender.i64_column(1))
            .zip(sender.decimal128_column(2))
            .map(|((name, id), v)| (name, id, v))
            .collect();
        rows.sort();
        assert_eq!(
            rows,
            vec![(long.to_string(), 1, 40), ("y".to_string(), 2, 12)]
        );
    }

    /// One worker, several batches: the encode scratch is reused between batches
    /// and keys (inline and arena) are accumulated across them.
    #[test]
    fn row_key_scratch_reused_across_batches() {
        let long = "a-string-well-over-twelve-bytes";
        let b1 = mixed_key_batch(&[1, 1], &["a", long], &[10, 5]);
        let b2 = mixed_key_batch(&[1, 1], &["a", long], &[3, 7]);
        let schema = RowKeySchema::new(
            vec![DataType::Int64, DataType::Utf8View],
            vec![false, false],
        );
        let slots = vec![AggregationSlot::new(
            AggregationKind::Sum,
            2,
            DataType::Decimal128(38, 0),
        )];

        let sender = run_row_key_group::<Compiled<(SumSlot<Int32Type>,)>>(
            vec![vec![b1, b2]],
            vec![0, 1],
            schema,
            slots,
        );

        assert_eq!(
            row_key_sum_rows(&sender),
            vec![(1, "a".to_string(), 13), (1, long.to_string(), 12)],
        );
    }

    // ---- Mixed string + integer aggregates via the runtime `Dynamic` ----

    /// The same `MIN(name)` (string) + `MAX(v)` (int) mix, but folded by the
    /// runtime [`Dynamic`] instead of a `Compiled` tuple, the path the planner
    /// now takes for a heterogeneous string signature. The wide (`i128`) cell
    /// holds the string slot's `ArenaKey` and the int slot's value; the int
    /// extreme stores wide but its slot declares `Int64`, so the output phase
    /// casts it back to `Int64`.
    #[test]
    fn dynamic_string_min_int_max() {
        let batch = mixed_key_batch(
            &[1, 2, 1, 2, 1],
            &["cat", "fig", "ant", "bee", "dog"],
            &[10, 7, 30, 5, 20],
        );
        let slots = vec![
            AggregationSlot::new(AggregationKind::Min, 1, DataType::Utf8View), // MIN(name) — string
            AggregationSlot::new(AggregationKind::Max, 2, DataType::Int64),    // MAX(v)    — int
        ];
        type Mix = Dynamic<i128>;
        let sender = run_group_full::<IntKeyExtractor<arrow_array::types::Int64Type>, Mix>(
            vec![vec![batch]],
            vec![0],
            slots,
            None,
            RadixConfig::DEFAULT,
        );
        let mut rows: Vec<(i64, String, i64)> = sender
            .i64_column(0)
            .into_iter()
            .zip(sender.string_column(1))
            .zip(sender.i64_column(2))
            .map(|((id, name), v)| (id, name, v))
            .collect();
        rows.sort();
        assert_eq!(
            rows,
            vec![(1, "ant".to_string(), 30), (2, "bee".to_string(), 7)]
        );
    }

    /// `MIN(name)` and `MAX(name)` over the *same* string column — a string mix
    /// of opposite directions (rejected by the old `Compiled`-only routing). Each
    /// slot keeps its own `ArenaKey` cell and folds its own direction.
    #[test]
    fn dynamic_string_min_and_max() {
        let batch = mixed_key_batch(
            &[1, 2, 1, 2, 1],
            &["cat", "fig", "ant", "bee", "dog"],
            &[10, 7, 30, 5, 20],
        );
        let slots = vec![
            AggregationSlot::new(AggregationKind::Min, 1, DataType::Utf8View), // MIN(name)
            AggregationSlot::new(AggregationKind::Max, 1, DataType::Utf8View), // MAX(name)
        ];
        type Mix = Dynamic<i128>;
        let sender = run_group_full::<IntKeyExtractor<arrow_array::types::Int64Type>, Mix>(
            vec![vec![batch]],
            vec![0],
            slots,
            None,
            RadixConfig::DEFAULT,
        );
        let mut rows: Vec<(i64, String, String)> = sender
            .i64_column(0)
            .into_iter()
            .zip(sender.string_column(1))
            .zip(sender.string_column(2))
            .map(|((id, min), max)| (id, min, max))
            .collect();
        rows.sort();
        // id 1 -> names {cat, ant, dog}: min "ant", max "dog".
        // id 2 -> names {fig, bee}:      min "bee", max "fig".
        assert_eq!(
            rows,
            vec![
                (1, "ant".to_string(), "dog".to_string()),
                (2, "bee".to_string(), "fig".to_string()),
            ]
        );
    }

    /// Two workers each see part of every group, so the string extremes must
    /// survive the partition merge (which resolves both `ArenaKey`s through the
    /// shared value arena) and a long string (> 12 bytes, a non-inline view) must
    /// round-trip through the arena rather than the inline header.
    #[test]
    fn dynamic_string_extreme_merges_across_workers() {
        let long_a = "alpha-aardvark-antelope"; // > 12 bytes, non-inline view
        let long_z = "zeta-zebra-zephyr-zenith";
        let w0 = mixed_key_batch(&[1, 1], &["mango", long_z], &[1, 2]);
        let w1 = mixed_key_batch(&[1, 1], &[long_a, "mint"], &[3, 4]);
        let slots = vec![
            AggregationSlot::new(AggregationKind::Min, 1, DataType::Utf8View),
            AggregationSlot::new(AggregationKind::Max, 1, DataType::Utf8View),
        ];
        type Mix = Dynamic<i128>;
        let sender = run_group_full::<IntKeyExtractor<arrow_array::types::Int64Type>, Mix>(
            vec![vec![w0], vec![w1]],
            vec![0],
            slots,
            None,
            RadixConfig::DEFAULT,
        );
        let ids = sender.i64_column(0);
        let mins = sender.string_column(1);
        let maxes = sender.string_column(2);
        assert_eq!(ids, vec![1]);
        assert_eq!(mins, vec![long_a.to_string()]);
        assert_eq!(maxes, vec![long_z.to_string()]);
    }

    /// High cardinality with an aggressive `switch_threshold` a *numeric* value
    /// would cross and scatter on. A string value must stay on the in-place fold
    /// (radix is disabled for it, since scatter would eagerly persist every row's
    /// string), so this exercises in-place stack growth with string cells and
    /// confirms every group's `MIN`/`MAX(name)` is still correct.
    #[test]
    fn dynamic_string_high_cardinality_stays_correct() {
        const N: i64 = 400;
        let ids: Vec<i64> = (0..N).chain(0..N).collect();
        // Group k sees "a{k}" then "z{k}": MIN is the "a" form, MAX the "z" form.
        let lows: Vec<String> = (0..N).map(|k| format!("a{k:04}")).collect();
        let highs: Vec<String> = (0..N).map(|k| format!("z{k:04}")).collect();
        let names: Vec<&str> = lows
            .iter()
            .chain(highs.iter())
            .map(String::as_str)
            .collect();
        let vals: Vec<i32> = vec![0; (2 * N) as usize];
        let batch = mixed_key_batch(&ids, &names, &vals);

        let slots = vec![
            AggregationSlot::new(AggregationKind::Min, 1, DataType::Utf8View),
            AggregationSlot::new(AggregationKind::Max, 1, DataType::Utf8View),
        ];
        type Mix = Dynamic<i128>;
        // A threshold the in-place table crosses well before N groups — a numeric
        // value would switch to radix here; the string value's override must not.
        let radix = RadixConfig {
            switch_threshold: 256,
            partitions: 16,
            ..RadixConfig::DEFAULT
        };
        let sender = run_group_full::<IntKeyExtractor<arrow_array::types::Int64Type>, Mix>(
            vec![vec![batch]],
            vec![0],
            slots,
            None,
            radix,
        );

        let mut rows: Vec<(i64, String, String)> = sender
            .i64_column(0)
            .into_iter()
            .zip(sender.string_column(1))
            .zip(sender.string_column(2))
            .map(|((id, min), max)| (id, min, max))
            .collect();
        rows.sort();
        let expected: Vec<(i64, String, String)> = (0..N)
            .map(|k| (k, format!("a{k:04}"), format!("z{k:04}")))
            .collect();
        assert_eq!(rows, expected);
    }

    // ---- LIMIT pushdown (per-worker TopK / First) ----

    #[test]
    fn top_k_limit_zero_returns_no_rows() {
        let values: Vec<i32> = (0..100).collect();

        let sender = run_group_full::<IntExtractor, CountValue>(
            vec![vec![batch_with_column(&values)]],
            vec![0],
            count_slots(),
            Some(GroupLimit::TopK {
                slot: 0,
                limit: 0,
                with_ties: false,
            }),
            RadixConfig::DEFAULT,
        );

        assert_eq!(sender.total_rows(), 0);
    }

    #[test]
    fn first_limit_zero_returns_no_rows() {
        let values: Vec<i32> = (0..100).collect();

        let sender = run_group_full::<IntExtractor, CountValue>(
            vec![vec![batch_with_column(&values)]],
            vec![0],
            count_slots(),
            Some(GroupLimit::First { limit: 0 }),
            RadixConfig::DEFAULT,
        );

        assert_eq!(sender.total_rows(), 0);
    }

    #[test]
    fn top_k_limit_exceeding_group_count_returns_all_groups() {
        // A LIMIT far larger than the group count must not pre-size a giant heap;
        // every group survives, each with its true count.
        let values: Vec<i32> = vec![1, 2, 3, 4, 5, 1, 2, 3];

        let sender = run_group_full::<IntExtractor, CountValue>(
            vec![vec![batch_with_column(&values)]],
            vec![0],
            count_slots(),
            Some(GroupLimit::TopK {
                slot: 0,
                limit: 1_000_000,
                with_ties: false,
            }),
            RadixConfig::DEFAULT,
        );

        assert_eq!(
            group_counts(&sender),
            vec![(1, 2), (2, 2), (3, 2), (4, 1), (5, 1)]
        );
    }

    #[test]
    fn top_k_across_two_workers_keeps_the_global_max() {
        // Per-worker top-k: each output worker keeps its own top `limit`, so the
        // union still contains the global-maximum group for a downstream LIMIT.
        let mut dominant: Vec<i32> = vec![0; 100];
        dominant.extend(1..50);
        let other: Vec<i32> = (50..100).collect();

        let sender = run_group_full::<IntExtractor, CountValue>(
            vec![
                vec![batch_with_column(&dominant)],
                vec![batch_with_column(&other)],
            ],
            vec![0],
            count_slots(),
            Some(GroupLimit::TopK {
                slot: 0,
                limit: 1,
                with_ties: false,
            }),
            RadixConfig::DEFAULT,
        );

        let pairs = group_counts(&sender);
        assert!(pairs.contains(&(0, 100)), "global-max group survives");
        assert!(pairs.iter().all(|&(_, c)| c >= 1), "every survivor is real");
    }

    #[test]
    fn top_k_pruned_merge_keeps_exact_top_counts() {
        // Two workers cross the radix threshold with a pushed top-k, so the
        // merge runs in bound order and skips dominated partitions.
        // The two dominant groups must still come out with exact totals.
        let radix = RadixConfig {
            switch_threshold: 256,
            partitions: 16,
            ..RadixConfig::DEFAULT
        };
        let mut values: Vec<i32> = (0..600).collect();
        values.extend(std::iter::repeat_n(7, 500));
        values.extend(std::iter::repeat_n(11, 300));
        let worker = || vec![batch_with_column(&values)];

        let sender = run_group_full::<IntExtractor, CountValue>(
            vec![worker(), worker()],
            vec![0],
            count_slots(),
            Some(GroupLimit::TopK {
                slot: 0,
                limit: 2,
                with_ties: false,
            }),
            radix,
        );

        let pairs = group_counts(&sender);
        assert!(pairs.contains(&(7, 1002)), "top group exact: {pairs:?}");
        assert!(pairs.contains(&(11, 602)), "second group exact: {pairs:?}");
    }

    #[test]
    fn string_top_k_pruning_keeps_dominant_group() {
        // String keys stay on the in-place stacks; the bounds still order and
        // prunes their merge partitions under a pushed top-k.
        let owned: Vec<String> = (0..300).map(|k| format!("phrase{k:04}")).collect();
        let mut names: Vec<&str> = owned.iter().map(String::as_str).collect();
        names.extend(std::iter::repeat_n("dominant", 400));
        let worker = || vec![string_key_batch(&names)];

        let sender = run_group_full::<StringKeyExtractor, CountValue>(
            vec![worker(), worker()],
            vec![0],
            count_slots(),
            Some(GroupLimit::TopK {
                slot: 0,
                limit: 1,
                with_ties: false,
            }),
            RadixConfig::DEFAULT,
        );

        let pairs = string_group_pairs(&sender);
        assert!(pairs.contains(&("dominant".to_string(), 800)), "{pairs:?}");
    }

    #[test]
    fn top_k_with_ties_emits_exact_groups_containing_the_top() {
        // A multi-key sort's pushdown keeps ties: no per-worker heap, so more
        // than `limit` groups may come out, but every emitted group is exact
        // and the true top groups must be among them for the downstream sort.
        let radix = RadixConfig {
            switch_threshold: 256,
            partitions: 16,
            ..RadixConfig::DEFAULT
        };
        let mut values: Vec<i32> = (0..600).collect();
        values.extend(std::iter::repeat_n(7, 500));
        values.extend(std::iter::repeat_n(11, 300));
        let worker = || vec![batch_with_column(&values)];

        let sender = run_group_full::<IntExtractor, CountValue>(
            vec![worker(), worker()],
            vec![0],
            count_slots(),
            Some(GroupLimit::TopK {
                slot: 0,
                limit: 2,
                with_ties: true,
            }),
            radix,
        );

        let pairs = group_counts(&sender);
        assert!(pairs.contains(&(7, 1002)), "top group exact: {pairs:?}");
        assert!(pairs.contains(&(11, 602)), "second group exact: {pairs:?}");
        for &(key, count) in &pairs {
            let expected = match key {
                7 => 1002,
                11 => 602,
                _ => 2,
            };
            assert_eq!(count, expected, "group {key} must be exact");
        }
    }

    #[test]
    fn first_limit_skips_remaining_partitions_once_satisfied() {
        // A plain LIMIT stops the merge pool-wide once any `limit` groups have
        // been emitted; whatever comes out must still be real groups.
        let values: Vec<i32> = (0..2000).collect();
        let worker = || vec![batch_with_column(&values)];

        let sender = run_group_full::<IntExtractor, CountValue>(
            vec![worker(), worker()],
            vec![0],
            count_slots(),
            Some(GroupLimit::First { limit: 3 }),
            RadixConfig::DEFAULT,
        );

        let pairs = group_counts(&sender);
        assert!(pairs.len() >= 3, "at least the limit survives: {pairs:?}");
        assert!(
            pairs.iter().all(|&(k, c)| (0..2000).contains(&k) && c == 2),
            "every survivor is a real, exact group"
        );
    }

    // ---- Radix drain for blob (string / row) keys ----

    #[test]
    fn string_keys_switch_to_radix_and_count() {
        // Enough distinct strings to trip the small threshold makes the worker
        // drain the table, clear it, and keep probing; every key is still
        // counted once per occurrence across drained windows.
        let owned: Vec<String> = (0..500).map(|k| format!("phrase{k:04}")).collect();
        let names: Vec<&str> = owned
            .iter()
            .chain(owned.iter())
            .map(String::as_str)
            .collect();
        let radix = RadixConfig {
            switch_threshold: 256,
            partitions: 16,
            ..RadixConfig::DEFAULT
        };

        let sender = run_group_full::<StringKeyExtractor, CountValue>(
            vec![vec![string_key_batch(&names)]],
            vec![0],
            count_slots(),
            None,
            radix,
        );

        let pairs = string_group_pairs(&sender);
        assert_eq!(pairs.len(), 500);
        assert!(pairs.iter().all(|(_, c)| *c == 2));
    }

    #[test]
    fn string_keys_radix_merges_across_workers() {
        // Both workers cross the threshold and drain, so the merge must combine
        // both workers' drained partition buffers and surviving stacks per key.
        let owned: Vec<String> = (0..500).map(|k| format!("phrase{k:04}")).collect();
        let names: Vec<&str> = owned.iter().map(String::as_str).collect();
        let radix = RadixConfig {
            switch_threshold: 256,
            partitions: 16,
            ..RadixConfig::DEFAULT
        };
        let worker = || vec![string_key_batch(&names)];

        let sender = run_group_full::<StringKeyExtractor, CountValue>(
            vec![worker(), worker()],
            vec![0],
            count_slots(),
            None,
            radix,
        );

        let pairs = string_group_pairs(&sender);
        assert_eq!(pairs.len(), 500);
        assert!(pairs.iter().all(|(_, c)| *c == 2));
    }

    #[test]
    fn row_key_switch_to_radix_and_count() {
        // A multi-column row key past the threshold drains the same way: the
        // drained persisted handles must resolve back through the arena at
        // the merge and combine into the right groups.
        let ids: Vec<i64> = (0..500).chain(0..500).collect();
        let owned: Vec<String> = (0..500).map(|k| format!("name{k:04}")).collect();
        let names: Vec<&str> = owned
            .iter()
            .chain(owned.iter())
            .map(String::as_str)
            .collect();
        let vals = vec![0i32; 1000];
        let schema = RowKeySchema::new(
            vec![DataType::Int64, DataType::Utf8View],
            vec![false, false],
        );
        let radix = RadixConfig {
            switch_threshold: 256,
            partitions: 16,
            ..RadixConfig::DEFAULT
        };

        let sender = run_row_key_group_radix::<CountValue>(
            vec![vec![mixed_key_batch(&ids, &names, &vals)]],
            vec![0, 1],
            schema,
            count_slots(),
            radix,
        );

        let rows = row_key_rows(&sender);
        assert_eq!(rows.len(), 500);
        assert!(rows.iter().all(|(_, _, c)| *c == 2));
    }

    #[test]
    fn mixed_switch_and_in_place_workers_merge_correctly() {
        // One worker crosses the switch threshold while another stays in-place under
        // it. The in-place worker's keys are absent from the merged HLL, so the
        // merge-size estimate accounts for them separately; correctness here proves
        // the mixed switched/in-place merge still combines every key.
        let switching: Vec<i32> = (0..500).collect();
        let in_place: Vec<i32> = (0..5).collect();
        let radix = RadixConfig {
            switch_threshold: 256,
            partitions: 16,
            ..RadixConfig::DEFAULT
        };

        let sender = run_group_full::<IntExtractor, CountValue>(
            vec![
                vec![batch_with_column(&switching)],
                vec![batch_with_column(&in_place)],
            ],
            vec![0],
            count_slots(),
            None,
            radix,
        );

        let counts = group_counts(&sender);
        assert_eq!(counts.len(), 500);
        assert!(
            counts.iter().all(|&(k, c)| c == if k < 5 { 2 } else { 1 }),
            "keys 0..5 seen by both workers count 2, the rest count 1"
        );
    }
}
