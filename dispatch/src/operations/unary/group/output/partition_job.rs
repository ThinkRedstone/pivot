//! One partition's merge work and the state every partition job shares.

use crate::memory::SlabAllocator;
use crate::operations::channels::Sender;
use crate::operations::unary::group::arena::SharedArena;
use crate::operations::unary::group::hashtables::{
    AggregationValue, HashTable, KeyExtractor, PartitionBuffers,
};
use crate::operations::unary::group::output::accumulator::{OutputAccumulator, emit_count};
use crate::operations::unary::group::output::merge;
use crate::operations::unary::group::output::topk_pruning::{TopKBounds, TopKThreshold};
use crate::operations::unary::group::{GroupLimit, Result};
use arrow_array::RecordBatch;
use arrow_buffer::Buffer;
use arrow_schema::DataType;
use crossbeam_deque::{Injector, Steal};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

/// One node's merge sources: its switched workers' scatter buffers and its
/// workers' in-place stacks.
pub(super) type NodeSources<K, V> = (
    Vec<PartitionBuffers<<K as KeyExtractor>::Persisted, V>>,
    Vec<HashTable<<K as KeyExtractor>::Persisted, V>>,
);

/// Everything the partition jobs of one merge share, built once by
/// `create_partition_jobs`: the sources they read, the sizing decided from
/// the gathered outputs, the handles output needs, and the progress and
/// rendezvous state the jobs write through.
///
/// Jobs go through work queues every worker drains, so each job carries one
/// `Arc` to this instead of cloning every shared field: thousands of jobs
/// each bumping and dropping many refcounts is cross-node cache-line traffic,
/// and under a pruned top-k most jobs are popped only to skip themselves.
pub(super) struct SharedMergeState<K: KeyExtractor, V: AggregationValue + ?Sized> {
    /// Per node: that node's switched workers' scatter buffers and the
    /// in-place stacks of its workers (switched workers' pre-switch tables
    /// and non-switched workers' full stacks). A direct merge holds a single
    /// entry with every node's sources.
    pub(super) sources: Vec<NodeSources<K, V>>,
    /// Hierarchical merge only: each partition's cross-node rendezvous,
    /// created by the first of its jobs to survive pruning (a fully pruned
    /// partition never allocates one).
    pub(super) cross_node: Option<Vec<OnceLock<CrossNodeMerge<K, V>>>>,
    pub(super) key_arena: Arc<SharedArena>,
    /// The arena's ring buffers wrapped as Arrow `Buffer`s, built once for the
    /// whole output phase. String-key output emits zero-copy views into this.
    pub(super) output_buffers: Arc<[Buffer]>,
    pub(super) partition_capacity: usize,
    /// Power-of-two count of hash partitions the merge runs at, chosen in
    /// `create_partition_jobs` from the input volume and the worker count.
    /// When a worker switched to radix it never exceeds the scatter bucket
    /// count (at most `RADIX_PARTITIONS`).
    pub(super) num_partitions: usize,
    pub(super) key_config: K::Config,
    /// The value's shared context, for the partition merge's entry fold + output.
    pub(super) shared_context: V::SharedContext,
    /// Declared output type per value column, in slot order; the output phase casts
    /// each finished value column to its type.
    pub(super) value_output_types: Arc<[DataType]>,
    pub(super) output_limit: Option<GroupLimit>,
    pub(super) count_only: bool,
    /// Top-k pruning state, present only when a prunable top-k was
    /// pushed: each partition's upper bound on any single group's final value,
    /// and the per-bin bounds a job's
    /// [`TopKCutoff`](crate::operations::unary::group::output::topk_pruning::TopKCutoff)
    /// admits groups by.
    pub(super) partition_bounds: Option<Vec<u64>>,
    pub(super) topk_bounds: Option<TopKBounds>,
    /// Shared pushed-limit progress: the k-th best weight the top-k bounds
    /// are checked against, and the emitted-group count a plain LIMIT skips
    /// against.
    pub(super) limit_progress: Arc<TopKThreshold>,
}

// The sources are read-only for the whole merge and every other field is
// either immutable or internally synchronized, so sharing this across worker
// threads is sound.
unsafe impl<K: KeyExtractor, V: AggregationValue + ?Sized> Send for SharedMergeState<K, V> {}
unsafe impl<K: KeyExtractor, V: AggregationValue + ?Sized> Sync for SharedMergeState<K, V> {}

/// One (NUMA node, partition) merge work unit: an index into the shared
/// [`SharedMergeState`], plus which node's sources to read on a hierarchical merge.
///
/// Created by the final worker to reach the gather barrier and pushed to the
/// node's [`Injector`] for work-stealing execution. Each job merges one node's
/// source tables for partition `index` into one result table. On a direct
/// merge (`SharedMergeState::cross_node` is `None`) that result is the partition's
/// final table and its rows are emitted directly. On a hierarchical merge,
/// each node's job merges only node-local memory and sends the resulting node
/// table into the partition's [`CrossNodeMerge`]; the last job to finish
/// receives every node's table and merges them into the final one. That last
/// merge is the only step that reads another node's memory.
pub struct PartitionJob<K: KeyExtractor, V: AggregationValue + ?Sized> {
    pub(super) shared: Arc<SharedMergeState<K, V>>,
    pub(super) index: usize,
    /// Which entry of [`SharedMergeState::sources`] this job merges (its node on a
    /// hierarchical merge; 0 on a direct merge).
    pub(super) node: usize,
}

unsafe impl<K: KeyExtractor, V: AggregationValue + ?Sized> Send for PartitionJob<K, V> {}

/// One partition's pending cross-node merge: each node's job sends the node's
/// merged aggregated table; the send that completes the set hands every
/// node's table back to that caller, electing it to run the final merge
/// ([`merge::merge_node_aggregated_tables`]) and emit. Senders never block
/// and no job ever waits: the mailbox is a lock-free queue and the election
/// is one atomic countdown.
pub(super) struct CrossNodeMerge<K: KeyExtractor, V: AggregationValue + ?Sized> {
    node_tables: Injector<HashTable<K::Persisted, V>>,
    /// Sends still outstanding; the sender that decrements this to zero is
    /// the receiver.
    pending_sends: AtomicUsize,
}

unsafe impl<K: KeyExtractor, V: AggregationValue + ?Sized> Send for CrossNodeMerge<K, V> {}
unsafe impl<K: KeyExtractor, V: AggregationValue + ?Sized> Sync for CrossNodeMerge<K, V> {}

impl<K: KeyExtractor, V: AggregationValue + ?Sized> CrossNodeMerge<K, V> {
    fn new(node_count: usize) -> Self {
        Self {
            node_tables: Injector::new(),
            pending_sends: AtomicUsize::new(node_count),
        }
    }

    /// Send this node's merged table. Exactly one call - the last - returns
    /// the full set; that caller must run the final merge. Each sender's push
    /// happens-before its countdown decrement, and the last sender's
    /// decrement observes all of them, so the drain below is guaranteed to
    /// see every node's table (and after the election nobody else touches
    /// the queue).
    fn send(&self, table: HashTable<K::Persisted, V>) -> Option<Vec<HashTable<K::Persisted, V>>> {
        self.node_tables.push(table);
        if self.pending_sends.fetch_sub(1, Ordering::AcqRel) != 1 {
            return None;
        }
        let mut all = Vec::new();
        loop {
            match self.node_tables.steal() {
                Steal::Success(table) => all.push(table),
                Steal::Empty => return Some(all),
                Steal::Retry => continue,
            }
        }
    }
}

impl<K: KeyExtractor, V: AggregationValue + ?Sized> PartitionJob<K, V> {
    /// Merge this partition's scatter buffers and in-place stacks into one result
    /// table, then feed its rows into the worker's shared `acc` (building columns
    /// into `allocator`). Accumulating across partition jobs, rather than emitting
    /// one batch per job, keeps the radix path's many small partitions from each
    /// producing a tiny `RecordBatch`.
    ///
    /// Whether it merged the partition or pruned it, the job then drops the
    /// partition's scatter buckets on its node. Each job frees its own share
    /// so the teardown runs on every worker instead of on the one that drops
    /// the shared state last.
    pub(super) fn run_into(
        self,
        acc: &mut Option<OutputAccumulator<K, V>>,
        sender: &mut dyn Sender<RecordBatch>,
        allocator: &mut SlabAllocator,
    ) -> Result<()> {
        let outcome = self.merge_and_emit(acc, sender, allocator);
        let (buffers, _) = &self.shared.sources[self.node];
        for worker_buffers in buffers {
            worker_buffers.release_partition(self.index, self.shared.num_partitions);
        }
        outcome
    }

    fn merge_and_emit(
        &self,
        acc: &mut Option<OutputAccumulator<K, V>>,
        sender: &mut dyn Sender<RecordBatch>,
        allocator: &mut SlabAllocator,
    ) -> Result<()> {
        let shared = &*self.shared;
        // Any `limit` groups satisfy a plain (unordered) pushed LIMIT, so once
        // that many are emitted the remaining partitions are unnecessary.
        if let Some(GroupLimit::First { limit }) = shared.output_limit
            && shared.limit_progress.emitted_groups() >= limit
        {
            return Ok(());
        }
        let mut cutoff = None;
        if let Some(bounds) = &shared.partition_bounds {
            let kth_best = shared.limit_progress.kth_best();
            if bounds[self.index] < kth_best {
                // No group in this partition can beat the k-th best exact group some
                // worker already holds, so merging it could not change the top-k.
                return Ok(());
            }
            let bin_bounds = shared
                .topk_bounds
                .as_ref()
                .expect("partition bounds only exist beside bin bounds");
            cutoff = bin_bounds.cutoff_for_partition(self.index, shared.num_partitions, kth_best);
        }
        let (buffers, tables) = &shared.sources[self.node];
        let result_map = merge::merge_combined::<K::Stored, V>(
            self.index,
            buffers,
            tables,
            shared.partition_capacity,
            shared.num_partitions,
            &shared.key_arena,
            &shared.shared_context,
            cutoff,
        );
        let result_map = match &shared.cross_node {
            None => result_map,
            Some(cells) => {
                // The rendezvous is created on demand by whichever of the
                // partition's jobs gets here first, so pruned partitions
                // never pay for one.
                let cross_node =
                    cells[self.index].get_or_init(|| CrossNodeMerge::new(shared.sources.len()));
                match cross_node.send(result_map) {
                    // Another node's job for this partition is still running;
                    // it will receive the tables and run the final merge.
                    None => return Ok(()),
                    Some(node_tables) => merge::merge_node_aggregated_tables::<K::Stored, V>(
                        node_tables,
                        shared.partition_capacity,
                        shared.num_partitions.trailing_zeros(),
                        &shared.key_arena,
                        &shared.shared_context,
                    ),
                }
            }
        };
        if result_map.len() == 0 {
            return Ok(());
        }
        // Global COUNT(DISTINCT) needs only each partition's distinct-key count,
        // not the keys, so it bypasses the accumulator and emits a single row.
        if shared.count_only {
            return emit_count(result_map.len(), sender);
        }
        let acc = acc.get_or_insert_with(|| {
            OutputAccumulator::new(
                allocator,
                shared.output_limit,
                shared.key_arena.clone(),
                shared.output_buffers.clone(),
                shared.key_config.clone(),
                shared.shared_context.clone(),
                shared.value_output_types.clone(),
                shared.limit_progress.clone(),
                shared.partition_bounds.is_some(),
            )
        });
        acc.extend_from_table(result_map, allocator, &mut *sender)
    }
}
