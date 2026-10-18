//! Per-worker GROUP BY state with adaptive in-place and radix modes.
//!
//! Every worker starts by aggregating into hash tables. If the active table
//! grows beyond [`SWITCH_THRESHOLD`], eligible keys switch to radix scatter.
//! Remaining rows are partitioned by the high hash bits and aggregated during
//! merge.
//!
//! ```text
//! consume rows
//!     |
//!     v
//! in-place tables -- threshold crossed --> partitioned scatter rows
//!     |                                      |
//!     +------------------+-------------------+
//!                        v
//!                  partition merge
//! ```

use crate::RECORD_BATCH_SIZE;
use crate::memory::SlabAllocator;
use crate::operations::unary::group::RADIX_PARTITIONS;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use crate::operations::unary::group::hashtables::hash_table::HashTable;
use crate::operations::unary::group::hashtables::{
    AggregationValue, DEFAULT_CAPACITY, KeyExtractor, LiveKey, PersistedKey, StridedScatterRows,
};
use crate::operations::unary::group::hll::Hll;
use crate::operations::unary::group::output::topk_pruning::{
    HashBinTotals, MIN_BINNED_ENTRIES, TopKWeight,
};
use crate::operations::unary::group::values::{AggregationSlot, ArityBody, WorkerContext};
use ahash::RandomState;
use arrow_array::RecordBatch;
use std::cell::UnsafeCell;
use std::ops::Range;
use std::sync::Arc;

/// Table capacity at which eligible keys switch to radix scatter.
const SWITCH_THRESHOLD: usize = 32768;

/// Distinct-to-rows percentage at or above which a post-switch table fill is
/// judged to have deduplicated nothing worth the probe cost (see
/// [`RadixConfig::scatter_fallback_distinct_pct`]).
const SCATTER_FALLBACK_DISTINCT_PCT: usize = 90;

/// Configuration for the in-place to radix transition.
#[derive(Copy, Clone)]
pub struct RadixConfig {
    /// Table-slot count past which a radix-eligible worker switches to scatter.
    pub switch_threshold: usize,
    /// Number of scatter partitions (a power of two).
    pub partitions: usize,
    /// Post-switch table fills whose distinct-entry count reaches this percentage
    /// of the rows consumed make the worker fall back to raw scatter: when
    /// almost every row seeds a new group, probing and re-draining each entry
    /// costs more than appending the row directly.
    pub scatter_fallback_distinct_pct: usize,
}

impl RadixConfig {
    pub const DEFAULT: Self = Self {
        switch_threshold: SWITCH_THRESHOLD,
        partitions: RADIX_PARTITIONS,
        scatter_fallback_distinct_pct: SCATTER_FALLBACK_DISTINCT_PCT,
    };

    /// Disables radix scatter.
    ///
    /// String MIN and MAX use this mode so losing strings are compared before
    /// they are persisted in the value arena.
    pub const fn without_radix(self) -> Self {
        Self {
            switch_threshold: usize::MAX,
            ..self
        }
    }
}

/// One worker's scatter buffer for each radix partition, as the merge reads
/// them.
///
/// The partition job that merges a bucket drops it right after. The pool
/// holds workers times buckets of them, each owning its chunk list, and
/// dropping them all from whichever worker drops the shared merge state last
/// is a serial tail that grows with the worker count; released per partition,
/// the teardown spreads over the jobs.
pub struct PartitionBuffers<KP: PersistedKey, V: AggregationValue + ?Sized> {
    buckets: Vec<UnsafeCell<Option<StridedScatterRows<KP, V>>>>,
    /// Rows across every bucket, counted by the flushing worker so the merge
    /// coordinator sums one number per worker rather than walking the
    /// pool's buckets.
    total_rows: usize,
}

// SAFETY: the merge partitions tile the bucket range without overlap and each
// partition has one job per source set, so a bucket is read and then released
// by exactly one job; no two threads ever access the same cell.
unsafe impl<KP: PersistedKey, V: AggregationValue + ?Sized> Sync for PartitionBuffers<KP, V> {}

impl<KP: PersistedKey, V: AggregationValue + ?Sized> PartitionBuffers<KP, V> {
    pub fn new(buckets: Vec<StridedScatterRows<KP, V>>) -> Self {
        let total_rows = buckets.iter().map(|bucket| bucket.len()).sum();
        Self {
            buckets: buckets
                .into_iter()
                .map(|bucket| UnsafeCell::new(Some(bucket)))
                .collect(),
            total_rows,
        }
    }

    /// How many scatter buckets the worker holds.
    pub fn bucket_count(&self) -> usize {
        self.buckets.len()
    }

    /// Rows scattered across every bucket.
    pub fn total_rows(&self) -> usize {
        self.total_rows
    }

    /// The scatter buckets of `partition`, given the merge's partition count.
    /// Both counts are powers of two, so partitions cover equal, contiguous
    /// bucket ranges.
    pub fn bucket_range(&self, partition: usize, num_partitions: usize) -> Range<usize> {
        let buckets_per_partition = (self.buckets.len() / num_partitions).max(1);
        let first = partition * buckets_per_partition;
        first..first + buckets_per_partition
    }

    /// Runs `f` over the rows of bucket `index`. A bucket is released only by
    /// the job that merged it, so a bucket being read is still held.
    pub fn with_bucket<R>(
        &self,
        index: usize,
        f: impl FnOnce(&StridedScatterRows<KP, V>) -> R,
    ) -> R {
        // SAFETY: only the bucket's own partition job reaches this cell (see
        // the `Sync` impl), and it releases the bucket only after reading it.
        let bucket = unsafe { &*self.buckets[index].get() };
        f(bucket
            .as_ref()
            .expect("a scatter bucket is released only after its partition merged it"))
    }

    /// Drops the buckets of `partition` once its merge has read them.
    pub fn release_partition(&self, partition: usize, num_partitions: usize) {
        for index in self.bucket_range(partition, num_partitions) {
            // SAFETY: as in `with_bucket`; the releasing job is the reader,
            // and its read has completed.
            unsafe { *self.buckets[index].get() = None };
        }
    }
}

/// Tables, optional scatter buffers, and sizing data produced by one worker.
pub struct AggregatedTableOutput<K: KeyExtractor, V: AggregationValue + ?Sized> {
    /// The NUMA node whose worker flushed this output; the merge groups
    /// sources by it.
    pub node: usize,
    /// In-place table stack: the full result if the worker never switched,
    /// otherwise its pre-switch tables.
    pub tables: Vec<HashTable<K::Persisted, V>>,
    /// Per-partition scatter buffers, present only after a radix transition.
    pub buffers: Option<PartitionBuffers<K::Persisted, V>>,
    /// Distinct-count sketch over the worker's rows, for sizing the merge targets.
    pub hll: Hll,
    /// Whether this worker built top-k hash-bin totals (returned beside this output
    /// by [`AggregatedTable::flush`] and added into the pool-wide totals
    /// before the gather).
    pub has_bin_totals: bool,
    /// With bin totals and a raw-scatter fallback: this worker's raw row
    /// count per scatter bucket. Raw rows are not in the totals, and at
    /// worst one group absorbed a whole bucket of them, so the merge adds a
    /// bucket's count to the bound of every bin the bucket covers.
    ///
    /// This is most definitely a hack and a patch. It keeps the bounds valid
    /// without touching the per-row scatter path, at the price of loosening
    /// them by a whole bucket's rows. It will not be relevant once raw rows
    /// are summed into the totals properly.
    pub raw_scatter_rows: Option<Vec<u64>>,
    /// `K::DEDUP_BY_HASH` only: this worker saw the (single) key whose bijective
    /// hash is 0, which is excluded from the tables. Adds 1 to the distinct count.
    pub zero_hash_seen: bool,
}

/// Per-worker aggregation state.
pub struct AggregatedTable<K: KeyExtractor, V: AggregationValue + ?Sized> {
    hash_state: RandomState,
    /// Shared value state used to construct tables and scatter rows.
    shared_context: V::SharedContext,
    /// String key storage, separate from the value arena to keep borrows disjoint.
    key_arena: WorkerArena,
    /// Per-worker value state.
    worker_context: V::WorkerContext,
    allocator: SlabAllocator,
    /// Tables built before a radix transition.
    tables: Vec<HashTable<K::Persisted, V>>,
    /// Per-partition buffers allocated on the first radix transition.
    buffers: Option<Vec<StridedScatterRows<<K as KeyExtractor>::Persisted, V>>>,
    /// Distinct-count sketch used to size merge targets.
    hll: Hll,
    /// Index into the aggregation slots of the `ORDER BY <agg> DESC LIMIT k`
    /// aggregate of a pushed-down top-k, whose partial values are summed into
    /// the top-k hash-bin totals. Present only when that aggregate is a COUNT
    /// form: its partials are nonnegative and additive, so bin sums stay valid
    /// upper bounds (see [`prunable_topk_slot`](crate::operations::unary::group::output::topk_pruning::prunable_topk_slot)). `None` builds
    /// no totals, including for a pushed top-k on any other aggregate.
    /// Kept across the raw-scatter fallback: raw rows are not folded (each
    /// would cost a totals update), the flush reports their count per
    /// scatter bucket instead (see [`AggregatedTableOutput::raw_scatter_rows`]).
    topk_aggregation_slot: Option<usize>,
    /// Hash-bin totals of that slot's partials, allocated on first use.
    /// Every partial the merge will see feeds it: drained table entries as
    /// they scatter, surviving table entries at flush.
    topk_bin_totals: Option<HashBinTotals>,
    /// Whether this worker has entered radix mode.
    switched_to_radix: bool,
    /// Whether rows bypass the in-place table and scatter raw, set once a
    /// post-switch fill stopped deduplicating.
    scatter_raw: bool,
    /// Rows folded into the active table since it was last cleared, the
    /// denominator of the raw-scatter fallback's dedup check.
    rows_since_clear: usize,
    /// Radix threshold and partition count.
    radix_config: RadixConfig,
    /// Scratch buffer for the per-row hashes computed once per batch.
    hashes: Box<[u64; RECORD_BATCH_SIZE]>,
    /// Per-worker reusable key-extraction scratch (e.g. the row extractor's
    /// encode buffers). Lent to the reader each batch and reused, never
    /// reallocated. `()` for extractors that read columns directly.
    scratch: K::Scratch,
    /// `K::DEDUP_BY_HASH` only: a key whose bijective hash is exactly 0 collides
    /// with the empty-slot sentinel, so it is excluded from the table and recorded
    /// here. There is at most one such key (the hash is a bijection), so this
    /// boolean adds 0 or 1 to the distinct count at output.
    zero_hash_seen: bool,
}

impl<K: KeyExtractor, V: AggregationValue + ?Sized> AggregatedTable<K, V> {
    pub fn new(
        hash_state: RandomState,
        key_arena: Arc<SharedArena>,
        shared_context: V::SharedContext,
        worker_context: V::WorkerContext,
        radix_config: RadixConfig,
        topk_aggregation_slot: Option<usize>,
    ) -> Self {
        let mut allocator = SlabAllocator::new(true);
        let table = HashTable::new(&mut allocator, DEFAULT_CAPACITY, 0, &shared_context);
        Self {
            hash_state,
            shared_context,
            key_arena: WorkerArena::new(key_arena),
            worker_context,
            allocator,
            tables: vec![table],
            buffers: None,
            hll: Hll::new(),
            topk_aggregation_slot,
            topk_bin_totals: None,
            switched_to_radix: false,
            scatter_raw: false,
            rows_since_clear: 0,
            radix_config,
            hashes: vec![0u64; RECORD_BATCH_SIZE]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
            scratch: K::Scratch::default(),
            zero_hash_seen: false,
        }
    }

    /// Consumes a batch in windows that fit the fixed hash scratch buffer.
    pub fn consume_batch(
        &mut self,
        batch: &RecordBatch,
        key_columns: &[usize],
        value_slots: &[AggregationSlot],
        key_config: &K::Config,
        shared_context: &V::SharedContext,
    ) {
        let total = batch.num_rows();
        if total <= RECORD_BATCH_SIZE {
            self.consume_window(batch, key_columns, value_slots, key_config, shared_context);
            return;
        }
        let mut start = 0;
        while start < total {
            let len = (total - start).min(RECORD_BATCH_SIZE);
            self.consume_window(
                &batch.slice(start, len),
                key_columns,
                value_slots,
                key_config,
                shared_context,
            );
            start += len;
        }
    }

    fn consume_window(
        &mut self,
        batch: &RecordBatch,
        key_columns: &[usize],
        value_slots: &[AggregationSlot],
        key_config: &K::Config,
        shared_context: &V::SharedContext,
    ) {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: each feature check covers every feature on its clone.
            if crate::cpu_features::supports_icelake_kernels() {
                return unsafe {
                    self.consume_window_icelake(
                        batch,
                        key_columns,
                        value_slots,
                        key_config,
                        shared_context,
                    )
                };
            }
            if crate::cpu_features::supports_v4_kernels() {
                return unsafe {
                    self.consume_window_v4(
                        batch,
                        key_columns,
                        value_slots,
                        key_config,
                        shared_context,
                    )
                };
            }
        }
        self.consume_window_body::<false>(
            batch,
            key_columns,
            value_slots,
            key_config,
            shared_context,
        );
    }

    /// Ice Lake target-feature clone of `consume_window`. Its inlined hash,
    /// probe, and fold loops compile above the floor. Keep the attribute and
    /// runtime feature check aligned.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(
        enable = "avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,avx512bitalg,avx512vpopcntdq,bmi1,bmi2,lzcnt,movbe,fma"
    )]
    fn consume_window_icelake(
        &mut self,
        batch: &RecordBatch,
        key_columns: &[usize],
        value_slots: &[AggregationSlot],
        key_config: &K::Config,
        shared_context: &V::SharedContext,
    ) {
        self.consume_window_body::<true>(
            batch,
            key_columns,
            value_slots,
            key_config,
            shared_context,
        );
    }

    /// x86-64-v4 clone of `consume_window` (Skylake-SP / Cascade Lake). Also
    /// inlines probing: the AVX-512 register file that justifies it is part of
    /// this tier too.
    #[cfg(target_arch = "x86_64")]
    #[target_feature(
        enable = "avx512f,avx512bw,avx512cd,avx512dq,avx512vl,bmi1,bmi2,lzcnt,movbe,fma"
    )]
    fn consume_window_v4(
        &mut self,
        batch: &RecordBatch,
        key_columns: &[usize],
        value_slots: &[AggregationSlot],
        key_config: &K::Config,
        shared_context: &V::SharedContext,
    ) {
        self.consume_window_body::<true>(
            batch,
            key_columns,
            value_slots,
            key_config,
            shared_context,
        );
    }

    // Must stay `inline(always)` so the target-tier clones' features reach
    // these loops. Only the AVX-512 tiers inline probing; the floor keeps each
    // arity outlined for its own stack frame and register allocation.
    #[inline(always)]
    fn consume_window_body<const INLINE_PROBE: bool>(
        &mut self,
        batch: &RecordBatch,
        key_columns: &[usize],
        value_slots: &[AggregationSlot],
        key_config: &K::Config,
        shared_context: &V::SharedContext,
    ) {
        let length = batch.num_rows();

        // Move scratch out temporarily so the key reader does not borrow `self`
        // while the table is mutated.
        let mut scratch = std::mem::take(&mut self.scratch);
        {
            let mut key_reader = K::make_reader(batch, key_columns, key_config, &mut scratch);
            let value_reader = V::make_reader(batch, value_slots);

            // Hash and encode each key once before probing.
            K::prepare_and_hash(
                &mut key_reader,
                &self.hash_state,
                &mut self.hashes[..length],
            );

            // A worker whose fills stopped deduplicating bypasses the table;
            // everything else continues through the in-place table.
            if self.scatter_raw {
                self.scatter_range(0, length, &key_reader, &value_reader);
            } else {
                self.consume_in_place::<INLINE_PROBE>(
                    length,
                    &key_reader,
                    &value_reader,
                    shared_context,
                );
            }
        }
        self.scratch = scratch;
    }

    /// Probes rows until the batch ends or the active table crosses its limit.
    #[inline(always)]
    fn consume_in_place<'b, const INLINE_PROBE: bool>(
        &mut self,
        length: usize,
        key_reader: &K::Reader<'b>,
        value_reader: &V::Reader<'b>,
        shared_context: &V::SharedContext,
    ) {
        let metadata = V::storage_metadata(shared_context);
        let mut i = 0;
        while i < length {
            let window_start = i;
            let overflowed = {
                let Self {
                    tables,
                    key_arena,
                    worker_context,
                    hashes,
                    zero_hash_seen,
                    ..
                } = self;
                V::dispatch_arity(
                    metadata,
                    ProbeWindow::<K, V, INLINE_PROBE> {
                        table: tables.last_mut().unwrap(),
                        key_arena,
                        worker_context,
                        hashes,
                        zero_hash_seen,
                        next_row: &mut i,
                        length,
                        key_reader,
                        value_reader,
                        shared_context,
                    },
                )
            };
            self.rows_since_clear += i - window_start;
            // The row that crossed the limit has already been folded. Scatter
            // starts at the following row.
            if overflowed && self.grow_or_radix() {
                self.scatter_range(i, length, key_reader, value_reader);
                return;
            }
        }
    }

    /// Grows the in-place stack or enters radix mode.
    ///
    /// Past the switch threshold, drains the deduplicated table into the radix
    /// partitions and returns `false` so the caller continues probing with the
    /// cleared table. When the drained fill deduplicated almost nothing, the
    /// worker permanently falls back to raw scatter and returns `true`: the
    /// caller then scatters the remaining raw rows.
    #[inline(always)]
    fn grow_or_radix(&mut self) -> bool {
        let next_size = self.tables.last().unwrap().capacity() * 4;
        // Hash-only keys have no persisted key to scatter.
        if K::DEDUP_BY_HASH || next_size <= self.radix_config.switch_threshold {
            self.tables.push(HashTable::new(
                &mut self.allocator,
                next_size,
                0,
                &self.shared_context,
            ));
            return false;
        }
        // On fills after the switch (the table restarted from empty), a
        // distinct count near the fill's row count means almost nothing
        // deduplicated: fall back to raw scatter rather than keep paying
        // the probe (and drain) for it. The switch-time fill is exempt
        // because its rows accumulated across the growing stack.
        let almost_no_dedup = self.switched_to_radix
            && self.tables.last().unwrap().len() * 100
                >= self.rows_since_clear * self.radix_config.scatter_fallback_distinct_pct;
        self.switched_to_radix = true;
        self.rows_since_clear = 0;
        self.create_scatter_buffers();
        self.scatter_active_table_to_radix_partitions();
        self.scatter_raw = almost_no_dedup;
        almost_no_dedup
    }

    /// Allocates the per-partition scatter buffers if not already present.
    fn create_scatter_buffers(&mut self) {
        if self.buffers.is_none() {
            self.buffers = Some(
                (0..self.radix_config.partitions)
                    .map(|_| StridedScatterRows::new())
                    .collect(),
            );
        }
    }

    /// Scatters rows into radix partitions without deduplicating them. Only
    /// a worker that fell back to raw scatter gets here. The rows are not
    /// folded into the bin totals; the flush reports their count per bucket
    /// instead.
    #[inline(always)]
    fn scatter_range<'b>(
        &mut self,
        start: usize,
        end: usize,
        key_reader: &K::Reader<'b>,
        value_reader: &V::Reader<'b>,
    ) {
        let shift = u64::BITS - self.radix_config.partitions.trailing_zeros();
        let Self {
            key_arena,
            worker_context,
            allocator,
            buffers,
            hll,
            hashes,
            shared_context,
            ..
        } = self;
        V::dispatch_arity(
            V::storage_metadata(shared_context),
            ScatterWindow::<K, V> {
                buffers: buffers.as_mut().unwrap(),
                key_arena,
                worker_context,
                allocator,
                hll,
                hashes,
                shift,
                start,
                end,
                key_reader,
                value_reader,
                shared_context,
            },
        );
    }

    /// Moves the active table's partial groups into radix buffers and clears it.
    ///
    /// With a pushed top-k, the drained partials leave the worker's tables
    /// here, so they fold into the bin totals now to keep the per-bin upper
    /// bounds covering every partial the merge will see.
    #[inline(always)]
    fn scatter_active_table_to_radix_partitions(&mut self) {
        let shift = u64::BITS - self.radix_config.partitions.trailing_zeros();
        if self.topk_aggregation_slot.is_some() && self.topk_bin_totals.is_none() {
            self.topk_bin_totals = Some(HashBinTotals::new(&mut self.allocator));
        }
        let Self {
            tables,
            buffers,
            allocator,
            hll,
            shared_context,
            topk_bin_totals,
            topk_aggregation_slot,
            ..
        } = self;
        let mut totals =
            topk_aggregation_slot.map(|slot| (topk_bin_totals.as_mut().unwrap(), slot));
        let table = tables.last_mut().unwrap();
        let buffers = buffers.as_mut().unwrap();
        let scatter_layout =
            StridedScatterRows::<<K as KeyExtractor>::Persisted, V>::layout::<0>(shared_context);
        for entry in table.iter(0) {
            let hash = entry.hash;
            hll.add(hash);
            if let Some((totals, slot)) = &mut totals {
                totals.add(hash, entry.stored.sort_key(*slot).saturating_weight());
            }
            let partition = (hash >> shift) as usize;
            // Entries are already deduplicated within this table.
            buffers[partition].push_with(scatter_layout, allocator, hash, *entry.key, |stored| {
                stored.copy_from(entry.stored)
            });
        }
        table.clear();
    }

    /// Finishes worker state for the merge phase, returning the worker's
    /// top-k bin totals beside the output so the caller can fold them into the
    /// pool's shared accumulators before arriving at the gather barrier.
    pub fn flush(mut self) -> (AggregatedTableOutput<K, V>, Option<HashBinTotals>) {
        if self.switched_to_radix {
            // Add pre-transition groups to the scatter-side estimate.
            for table in &self.tables {
                for entry in table.iter(0) {
                    self.hll.add(entry.hash);
                }
            }
        }
        // Fold the entries still sitting in the in-place tables into the
        // top-k bin totals (drained entries were folded as they scattered). A
        // worker this small skips the totals (and thereby turns pruning off)
        // rather than pay the allocation on a query too small to prune.
        let table_entries: usize = self.tables.iter().map(|t| t.len()).sum();
        let allocator = &mut self.allocator;
        let taken_totals = self.topk_bin_totals.take();
        let topk_bin_totals = self.topk_aggregation_slot.and_then(|slot| {
            if taken_totals.is_none() && table_entries < MIN_BINNED_ENTRIES {
                return None;
            }
            let mut totals = taken_totals.unwrap_or_else(|| HashBinTotals::new(allocator));
            for table in &self.tables {
                for entry in table.iter(0) {
                    totals.add(entry.hash, entry.stored.sort_key(slot).saturating_weight());
                }
            }
            Some(totals)
        });
        // Raw-scattered rows never reached the totals. Report each bucket's
        // row count so the merge can widen its bounds by them (a hack, see
        // `AggregatedTableOutput::raw_scatter_rows`). The bucket counts also
        // include the drained partials scattered before the fallback, which
        // only loosens the bounds further.
        let raw_scatter_rows = match (&self.buffers, topk_bin_totals.is_some(), self.scatter_raw) {
            (Some(buffers), true, true) => {
                Some(buffers.iter().map(|bucket| bucket.len() as u64).collect())
            }
            _ => None,
        };
        self.key_arena.flush();
        self.worker_context.flush();
        let output = AggregatedTableOutput {
            node: crate::worker::current_node(),
            tables: self.tables,
            buffers: self.buffers.map(PartitionBuffers::new),
            hll: self.hll,
            has_bin_totals: topk_bin_totals.is_some(),
            raw_scatter_rows,
            zero_hash_seen: self.zero_hash_seen,
        };
        (output, topk_bin_totals)
    }
}

/// Arity-dispatched state for one consume window. `N == 0` uses runtime
/// metadata; other values specialize the layout and loops.
///
/// `INLINE_PROBE` is true only for the AVX-512 tiers. The floor keeps probing outlined
/// for separate stack frames and register allocation.
struct ProbeWindow<'a, 'b, K: KeyExtractor, V: AggregationValue + ?Sized, const INLINE_PROBE: bool>
{
    table: &'a mut HashTable<K::Persisted, V>,
    key_arena: &'a mut WorkerArena,
    worker_context: &'a mut V::WorkerContext,
    hashes: &'a [u64; RECORD_BATCH_SIZE],
    zero_hash_seen: &'a mut bool,
    next_row: &'a mut usize,
    length: usize,
    key_reader: &'a K::Reader<'b>,
    value_reader: &'a V::Reader<'b>,
    shared_context: &'a V::SharedContext,
}

impl<K: KeyExtractor, V: AggregationValue + ?Sized, const INLINE_PROBE: bool> ArityBody<bool>
    for ProbeWindow<'_, '_, K, V, INLINE_PROBE>
{
    #[inline(always)]
    fn run<const N: usize>(self) -> bool {
        let ProbeWindow {
            table,
            key_arena,
            worker_context,
            hashes,
            zero_hash_seen,
            next_row,
            length,
            key_reader,
            value_reader,
            shared_context,
        } = self;
        if INLINE_PROBE {
            // The AVX-512 tiers' larger register file makes inlining worthwhile.
            probe_rows_body::<N, K, V>(
                table,
                key_arena,
                worker_context,
                hashes,
                zero_hash_seen,
                next_row,
                length,
                key_reader,
                value_reader,
                shared_context,
            )
        } else {
            probe_rows_outlined::<N, K, V>(
                table,
                key_arena,
                worker_context,
                hashes,
                zero_hash_seen,
                next_row,
                length,
                key_reader,
                value_reader,
                shared_context,
            )
        }
    }
}

/// Floor-tier probe loop for one consume window.
///
/// Each arity stays outlined for its own stack frame and register allocation.
/// This costs one call per window, not per row. The AVX-512 tiers instead inline
/// [`probe_rows_body`] because its larger register file handles the pressure.
///
/// Separate reference parameters also preserve alias information across raw
/// table writes, allowing bound column pointers to remain outside the row loop.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn probe_rows_outlined<const N: usize, K: KeyExtractor, V: AggregationValue + ?Sized>(
    table: &mut HashTable<K::Persisted, V>,
    key_arena: &mut WorkerArena,
    worker_context: &mut V::WorkerContext,
    hashes: &[u64; RECORD_BATCH_SIZE],
    zero_hash_seen: &mut bool,
    next_row: &mut usize,
    length: usize,
    key_reader: &K::Reader<'_>,
    value_reader: &V::Reader<'_>,
    shared_context: &V::SharedContext,
) -> bool {
    probe_rows_body::<N, K, V>(
        table,
        key_arena,
        worker_context,
        hashes,
        zero_hash_seen,
        next_row,
        length,
        key_reader,
        value_reader,
        shared_context,
    )
}

// Must stay `inline(always)` so each caller compiles the loop for its feature
// tier. The floor wrapper itself stays outlined.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn probe_rows_body<const N: usize, K: KeyExtractor, V: AggregationValue + ?Sized>(
    table: &mut HashTable<K::Persisted, V>,
    key_arena: &mut WorkerArena,
    worker_context: &mut V::WorkerContext,
    hashes: &[u64; RECORD_BATCH_SIZE],
    zero_hash_seen: &mut bool,
    next_row: &mut usize,
    length: usize,
    key_reader: &K::Reader<'_>,
    value_reader: &V::Reader<'_>,
    shared_context: &V::SharedContext,
) -> bool {
    {
        const L1_DISTANCE: usize = 16;
        const L2_DISTANCE: usize = 48;
        // Reuse one probe layout for every row in the window.
        let mut prober = if N == 0 {
            table.prober()
        } else {
            table.prober_with_metadata(V::metadata_for_arity::<N>())
        };
        let mut row = *next_row;
        let mut overflowed = false;
        while row < length {
            let hash = hashes[row];
            // Exact hash-only distinct counting handles hash zero out of band
            // because zero is the table's empty sentinel.
            if K::DEDUP_BY_HASH && hash == 0 {
                *zero_hash_seen = true;
                row += 1;
                continue;
            }
            if row + L2_DISTANCE < length {
                prober.prefetch_l2(hashes[row + L2_DISTANCE]);
            }
            if row + L1_DISTANCE < length {
                prober.prefetch(hashes[row + L1_DISTANCE]);
            }
            // Probe once, then seed a new group or update the matching group.
            let key = K::live_key(key_reader, row, key_arena);
            prober.probe_fold::<false, _, _, _, _>(
                hash,
                key,
                &mut *worker_context,
                |worker_context, stored| stored.seed(value_reader, row, worker_context),
                |worker_context, stored| {
                    stored.update(value_reader, row, worker_context, shared_context)
                },
            );
            row += 1;
            if prober.undersized() {
                overflowed = true;
                break;
            }
        }
        *next_row = row;
        overflowed
    }
}

/// State passed through arity dispatch for one scatter range.
struct ScatterWindow<'a, 'b, K: KeyExtractor, V: AggregationValue + ?Sized> {
    buffers: &'a mut Vec<StridedScatterRows<<K as KeyExtractor>::Persisted, V>>,
    key_arena: &'a mut WorkerArena,
    worker_context: &'a mut V::WorkerContext,
    allocator: &'a mut SlabAllocator,
    hll: &'a mut Hll,
    hashes: &'a [u64; RECORD_BATCH_SIZE],
    shift: u32,
    start: usize,
    end: usize,
    key_reader: &'a K::Reader<'b>,
    value_reader: &'a V::Reader<'b>,
    shared_context: &'a V::SharedContext,
}

impl<K: KeyExtractor, V: AggregationValue + ?Sized> ArityBody<()> for ScatterWindow<'_, '_, K, V> {
    /// Scatter has little live state, so keeping it inline avoids a window call.
    #[inline(always)]
    fn run<const N: usize>(self) {
        let ScatterWindow {
            buffers,
            key_arena,
            worker_context,
            allocator,
            hll,
            hashes,
            shift,
            start,
            end,
            key_reader,
            value_reader,
            shared_context,
        } = self;
        // Every partition shares one row-layout snapshot.
        let scatter_layout =
            StridedScatterRows::<<K as KeyExtractor>::Persisted, V>::layout::<N>(shared_context);
        // Indexing rather than iterating: `i` is the absolute row, used to
        // read the hash, the key, and the value columns in parallel.
        #[allow(clippy::needless_range_loop)]
        for i in start..end {
            let hash = hashes[i];
            hll.add(hash);
            let partition = (hash >> shift) as usize;
            let key = K::live_key(key_reader, i, key_arena).persist();
            // Seed directly into the destination row.
            buffers[partition].push_with(scatter_layout, allocator, hash, key, |stored| {
                stored.seed(value_reader, i, worker_context)
            });
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;
    use crate::operations::unary::group::keys::IntKeyExtractor;
    use crate::operations::unary::group::values::{
        AggregationKind, AggregationSlot, Compiled, CountSlot,
    };
    use arrow_array::types::Int32Type;
    use arrow_array::{ArrayRef, Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    type IntExtractor = IntKeyExtractor<Int32Type>;
    type CountValue = Compiled<(CountSlot,), u8>;

    /// A config whose first in-place overflow already crosses the switch
    /// threshold, so radix behavior is reached within a few hundred rows.
    const SMALL_RADIX: RadixConfig = RadixConfig {
        switch_threshold: 256,
        partitions: 16,
        ..RadixConfig::DEFAULT
    };

    fn consume_all(table: &mut AggregatedTable<IntExtractor, CountValue>, values: &[i32]) {
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
        for chunk in values.chunks(RECORD_BATCH_SIZE) {
            let array: ArrayRef = Arc::new(Int32Array::from(chunk.to_vec()));
            let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
            table.consume_batch(
                &batch,
                &[0],
                &[AggregationSlot::new(
                    AggregationKind::CountStar,
                    0,
                    DataType::Int64,
                )],
                &(),
                &(),
            );
        }
    }

    fn scatter_row_count(output: &AggregatedTableOutput<IntExtractor, CountValue>) -> usize {
        let layout = StridedScatterRows::<i32, CountValue>::layout::<0>(&());
        let buffers = output.buffers.as_ref().expect("worker switched");
        let mut rows = 0;
        for bucket in 0..buffers.bucket_count() {
            buffers.with_bucket(bucket, |rows_in_bucket| {
                rows_in_bucket.for_each(layout, |_, _, _| rows += 1)
            });
        }
        assert_eq!(rows, buffers.total_rows());
        rows
    }

    #[test]
    fn unique_fills_fall_back_to_raw_scatter() {
        init_test_free_pool(64);
        let mut table = AggregatedTable::<IntExtractor, CountValue>::new(
            ahash::RandomState::new(),
            SharedArena::new(64),
            (),
            (),
            SMALL_RADIX,
            None,
        );

        // Unique keys make every post-switch fill 100% distinct, tripping the
        // fallback; the duplicate tail must then be scattered one row each.
        let mut values: Vec<i32> = (0..4000).collect();
        values.extend(std::iter::repeat_n(999_999, 1000));
        consume_all(&mut table, &values);
        let (output, _totals) = table.flush();

        assert!(
            scatter_row_count(&output) > 4500,
            "duplicates should be scattered raw after the fallback"
        );
    }

    #[test]
    fn deduplicating_fills_stay_on_the_dedup_route() {
        init_test_free_pool(64);
        let mut table = AggregatedTable::<IntExtractor, CountValue>::new(
            ahash::RandomState::new(),
            SharedArena::new(64),
            (),
            (),
            SMALL_RADIX,
            None,
        );

        // Each key repeats twice back to back, so every fill dedups half its
        // rows and stays comfortably under the fallback threshold.
        let values: Vec<i32> = (0..2000).flat_map(|k| [k, k]).collect();
        consume_all(&mut table, &values);
        let (output, _totals) = table.flush();

        assert!(
            scatter_row_count(&output) < 2500,
            "dedup before scatter should persist across repeat-heavy fills"
        );
    }

    #[test]
    fn top_k_drained_fills_feed_the_bin_totals() {
        init_test_free_pool(64);
        let mut table = AggregatedTable::<IntExtractor, CountValue>::new(
            ahash::RandomState::new(),
            SharedArena::new(64),
            (),
            (),
            SMALL_RADIX,
            Some(0),
        );

        // Deduplicating fills drain into buffers; every drained partial and
        // the final in-place stack must land in the bin totals, so their mass
        // equals the row count and the merge's bounds stay valid.
        let values: Vec<i32> = (0..6000).flat_map(|k| [k, k]).collect();
        consume_all(&mut table, &values);
        let (output, totals) = table.flush();

        assert!(output.buffers.is_some(), "fills drain into buffers");
        let totals = totals.expect("drained fills keep the bin totals fed");
        assert!(output.has_bin_totals);
        let mass: u64 = totals.into_bin_totals().iter().sum();
        assert_eq!(mass, 12000, "every partial is binned exactly once");
    }

    #[test]
    fn top_k_unique_fills_fall_back_to_raw_scatter() {
        init_test_free_pool(64);
        let mut table = AggregatedTable::<IntExtractor, CountValue>::new(
            ahash::RandomState::new(),
            SharedArena::new(64),
            (),
            (),
            SMALL_RADIX,
            Some(0),
        );

        // Unique keys make every post-switch fill 100% distinct: the worker
        // must fall back to raw scatter, keep the totals of what it drained
        // before, and report the raw row count per bucket for the merge to
        // widen its bounds with.
        let values: Vec<i32> = (0..6000).collect();
        consume_all(&mut table, &values);
        let (output, totals) = table.flush();

        assert!(output.buffers.is_some(), "fallback allocates buffers");
        assert!(scatter_row_count(&output) > 1000, "tail rows scatter raw");
        assert!(totals.is_some(), "drained fills keep the bin totals");
        assert!(output.has_bin_totals);
        let bucket_rows: u64 = output.raw_scatter_rows.as_ref().unwrap().iter().sum();
        assert_eq!(bucket_rows as usize, scatter_row_count(&output));
    }
}
