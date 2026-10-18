//! The per-worker driver of the merge phase: builds the partition jobs once
//! and runs whichever jobs this worker steals.

use crate::memory::SlabAllocator;
use crate::operations::channels::Sender;
use crate::operations::unary;
use crate::operations::unary::group::GroupLimit;
use crate::operations::unary::group::arena::SharedArena;
use crate::operations::unary::group::hashtables::{
    AggregatedTableOutput, AggregationValue, DEFAULT_CAPACITY, HashTable, KeyExtractor,
    MAX_LOAD_FACTOR, PartitionBuffers, entry_stride,
};
use crate::operations::unary::group::hll::Hll;
use crate::operations::unary::group::output::accumulator::{OutputAccumulator, emit_count};
use crate::operations::unary::group::output::partition_job::{PartitionJob, SharedMergeState};
use crate::operations::unary::group::output::topk_pruning::{
    HashBinTotals, SharedBinTotals, TopKBounds, TopKThreshold,
};
use crate::operations::unary::pipeline_breaker::Outputter;
use crate::worker::current_node;
use arrow_array::RecordBatch;
use arrow_buffer::Buffer;
use arrow_schema::DataType;
use crossbeam_deque::{Injector, Steal};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

/// The fewest partitions the merge phase ever runs with. The merge routes each
/// row by the top `log2(partitions)` bits of its hash, and a single partition
/// has zero such bits, which leaves no valid shift. Two near-empty jobs cost
/// nothing extra over one.
const MIN_MERGE_PARTITIONS: usize = 2;

/// Minimum merged distinct estimate for top-k partition pruning. Turning
/// the per-worker hash-bin totals into partition bounds costs a sum over
/// `workers x HASH_BINS` counters; below this many groups the whole merge
/// costs about the same, so pruning could only add overhead.
const MIN_TOPK_PRUNE_ESTIMATE: usize = 1 << 19;

/// Handles the output (merge) phase of a GROUP BY.
///
/// The last worker to reach the gather barrier collects all per-worker tables
/// and publishes one [`PartitionJob`] per hash partition to the shared
/// work-stealing [`Injector`]. All workers (including the one that injected)
/// then steal and execute jobs until the injector is empty.
pub struct GroupOutputter<K: KeyExtractor, V: AggregationValue + ?Sized> {
    /// String *key* storage; backs the leading key column(s) at output.
    key_arena: Arc<SharedArena>,
    /// One job queue per NUMA node. A node-level merge job reads that node's
    /// tables, so it is queued on (and preferentially run by) that node's
    /// workers; a worker falls back to other nodes' queues only when its own
    /// is empty, trading locality for tail balance.
    injectors: Arc<Vec<Injector<PartitionJob<K, V>>>>,
    /// This worker's node, i.e. which of `injectors` is local to it.
    node: usize,
    partition_jobs_injected: Arc<AtomicBool>,
    key_config: K::Config,
    /// The value's shared context, threaded into each [`PartitionJob`] so the merge
    /// folds existing entries via [`AggregationValue::merge_from`] and the output
    /// resolves string extremes (it carries the value arena).
    shared_context: V::SharedContext,
    /// The declared output type of each value column, in slot order. The output
    /// phase renders each accumulator at its storage width then casts it to this
    /// type (a `COUNT` in an `i128` cell down to `Int64`, a narrow `SUM` up to
    /// `Decimal128`); a no-op when the width already matches.
    value_output_types: Arc<[DataType]>,
    output_limit: Option<GroupLimit>,
    /// Global `COUNT(DISTINCT)`: emit each partition's distinct-key count instead
    /// of its keys (a downstream `SUM` totals them).
    count_only: bool,
    /// Shared lower bound on the pushed top-k's k-th best exact group value,
    /// fed by every merged group and read by partition jobs to skip
    /// partitions whose bound cannot reach it.
    topk_threshold: Arc<TopKThreshold>,
    /// Pool-wide bin totals every worker adds its own into on arrival; the
    /// final gather arrival takes the sum.
    shared_bin_totals: Arc<SharedBinTotals>,
    /// The final gather arrival saw the hash-only extractor's out-of-band zero
    /// hash and must emit its extra count row once.
    zero_hash_pending: bool,
    /// One allocator per worker for the output columns of every partition this
    /// worker handles, so small per-partition outputs pack into shared buffers
    /// instead of each grabbing a fresh 2MB one. Created lazily on the first job.
    output_allocator: Option<SlabAllocator>,
    /// Per-worker output accumulator: packs the rows from every partition job this
    /// worker runs into full output-chunk batches, so a high partition count (the
    /// radix path) doesn't emit one tiny batch per partition. Flushed when the queue
    /// drains.
    ///
    /// `Option` because it can't exist before the worker's first job: the
    /// per-output-phase buffers it needs are computed once (by the final worker
    /// at the gather barrier) and arrive attached to each [`PartitionJob`], so a
    /// worker only learns them from the first job it steals. A worker running only
    /// count-only `COUNT(DISTINCT)` jobs never builds one.
    output_accumulator: Option<OutputAccumulator<K, V>>,
}

impl<K: KeyExtractor, V: AggregationValue + ?Sized> GroupOutputter<K, V> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        key_arena: Arc<SharedArena>,
        injectors: Arc<Vec<Injector<PartitionJob<K, V>>>>,
        partition_jobs_injected: Arc<AtomicBool>,
        key_config: K::Config,
        shared_context: V::SharedContext,
        value_output_types: Arc<[DataType]>,
        output_limit: Option<GroupLimit>,
        count_only: bool,
        topk_threshold: Arc<TopKThreshold>,
        shared_bin_totals: Arc<SharedBinTotals>,
    ) -> Self {
        Self {
            key_arena,
            node: current_node(),
            injectors,
            partition_jobs_injected,
            key_config,
            shared_context,
            value_output_types,
            output_limit,
            count_only,
            topk_threshold,
            shared_bin_totals,
            zero_hash_pending: false,
            output_allocator: None,
            output_accumulator: None,
        }
    }

    /// Adds this worker's top-k bin totals to the pool-wide sum. Called before
    /// the worker arrives at the gather barrier.
    pub(crate) fn add_bin_totals(&self, totals: HashBinTotals) {
        self.shared_bin_totals.add(self.node, totals);
    }

    /// Combine every worker's gathered tables, size the merge from the merged
    /// distinct estimate, and publish one [`PartitionJob`] per merge partition to
    /// the shared injector. Run exactly once, by the last worker to reach the
    /// gather barrier.
    pub(crate) fn create_partition_jobs(&mut self, outputs: Vec<AggregatedTableOutput<K, V>>) {
        // We get worker count by outputs.len() since we don't have access here to the topology
        let worker_count = outputs.len();
        let node_count = self.injectors.len();
        let mut tables_by_node: Vec<Vec<HashTable<K::Persisted, V>>> =
            (0..node_count).map(|_| Vec::new()).collect();
        let mut buffers_by_node: Vec<Vec<PartitionBuffers<K::Persisted, V>>> =
            (0..node_count).map(|_| Vec::new()).collect();
        let mut hll = Hll::new();
        // A worker only folds its keys into the sketch once it switches to
        // radix; a worker that stayed in-place (`buffers.is_none()`) is absent
        // from the merged HLL. Add its exact distinct count (each table tracks
        // its own `len`, no scan) so the estimate below isn't skewed low when
        // some workers switched and others didn't. Counting a key here that a
        // switched worker also holds over-counts, which only over-sizes the
        // merge targets (safe); under-counting is what forces mid-merge resizes.
        let mut non_switched_groups = 0usize;
        // Bounds are only valid if every worker holding groups folded its
        // bin totals; a worker that skipped (too small, see the flush) makes them
        // undercount and disables pruning.
        let mut every_group_binned = true;
        // Pool-wide raw-scattered row count per scatter bucket, from the
        // workers that fell back to raw scatter; widens the bounds below.
        let mut raw_bucket_rows: Option<Vec<u64>> = None;
        for out in outputs {
            let table_entries: usize = out.tables.iter().map(|t| t.len()).sum();
            if out.buffers.is_none() {
                non_switched_groups += table_entries;
            }
            every_group_binned &=
                out.has_bin_totals || (table_entries == 0 && out.buffers.is_none());
            if let Some(rows) = &out.raw_scatter_rows {
                let sum = raw_bucket_rows.get_or_insert_with(|| vec![0; rows.len()]);
                for (bucket, count) in sum.iter_mut().zip(rows) {
                    *bucket += count;
                }
            }
            tables_by_node[out.node].extend(out.tables);
            if let Some(b) = out.buffers {
                buffers_by_node[out.node].push(b);
            }
            hll.merge(&out.hll);
            self.zero_hash_pending |= out.zero_hash_seen;
        }
        let any_switched = buffers_by_node.iter().any(|b| !b.is_empty());
        let total_in_place: usize = tables_by_node.iter().flatten().map(|t| t.len()).sum();
        // Estimate the global distinct count: the HLL covers switched
        // workers, and a non-switched worker's exact per-table count stands
        // in for its keys (over-counting keys a switched worker also holds,
        // which is safe).
        let estimate = hll.estimate() + non_switched_groups;
        let (num_partitions, partition_capacity) = if !any_switched {
            // How many entries we want in each table for a partition job
            const TARGET_SCANNED_ENTRIES_PER_JOB: usize = 16 * 1024;
            const MINIMUM_SCANNED_ENTRIES_PER_JOB: usize = 8 * 1024;

            let input_slots: usize = tables_by_node.iter().flatten().map(|t| t.capacity()).sum();
            let maximum_allowed_jobs = input_slots / MINIMUM_SCANNED_ENTRIES_PER_JOB;
            let partitions = (input_slots / TARGET_SCANNED_ENTRIES_PER_JOB)
                // Never fewer jobs than workers, so no core idles, but a job still has
                // to scan at least MINIMUM_SCANNED_ENTRIES_PER_JOB slots to be worth its
                // setup, so small inputs stay below that floor.
                .max(worker_count)
                .min(maximum_allowed_jobs)
                .max(MIN_MERGE_PARTITIONS)
                .next_power_of_two();
            (
                partitions,
                (total_in_place / partitions)
                    .next_power_of_two()
                    .max(DEFAULT_CAPACITY),
            )
        } else {
            // Decide how many independent merge jobs to run (`num_partitions`) and how
            // large each job's result table starts (`partition_capacity`). Every job
            // builds one hash table and probes it at random, so the win is keeping that
            // table inside a core's private cache: otherwise each probe is a DRAM trip.
            // `target_groups_per_partition` is that cache budget expressed as a group
            // count: (cache bytes) / (bytes per table entry). Because we divide by the
            // entry width, a wide multi-column key (fat entries) targets fewer groups
            // per job than a bare integer key.
            const TARGET_MERGE_PARTITION_BYTES: usize = 256 * 1024; // ~one core's L2
            // The exact bytes per table entry (hash + key + stored value with
            // their padding), as the tables themselves lay it out. Undersizing
            // this inflates the per-partition group target and produces fewer,
            // larger merge targets that fall out of cache.
            let entry_bytes = entry_stride::<K::Persisted, V>(&self.shared_context);
            let target_groups_per_partition =
                (TARGET_MERGE_PARTITION_BYTES / entry_bytes.max(1)).max(1);
            // A worker switched and scattered its groups into `scatter_buckets`
            // (at most RADIX_PARTITIONS) buckets. One merge job per bucket would be
            // thousands of near-empty jobs at moderate cardinality, so size the job
            // count to the now-exact distinct estimate instead: enough jobs that each
            // target holds ~target_groups_per_partition groups. Then bound it - at
            // least worker_count so every core has work, and never more than the
            // bucket count, since the merge can't be finer than the scatter (each job
            // folds a contiguous range of buckets, reaching one-bucket-per-job only
            // at very high cardinality).
            let scatter_buckets = buffers_by_node
                .iter()
                .flatten()
                .next()
                .expect("a worker switched, so some node has scatter buffers")
                .bucket_count();
            let merge_partitions = (estimate / target_groups_per_partition)
                .clamp(worker_count, scatter_buckets)
                .max(MIN_MERGE_PARTITIONS)
                .next_power_of_two();
            // Start each target big enough to hold its share at MAX_LOAD_FACTOR (the
            // merge's resize threshold), so it fills without resizing mid-merge.
            let per_partition = estimate as f64 / merge_partitions as f64;
            let capacity = ((per_partition / MAX_LOAD_FACTOR).ceil() as usize)
                .next_power_of_two()
                .max(DEFAULT_CAPACITY);
            (merge_partitions, capacity)
        };

        // Node-hierarchical or direct merge? Hierarchical (one job per node
        // and partition, then a final merge of the per-node tables) keeps
        // each job's reads node-local, but pays an extra materialization of
        // the node tables. That trade only wins when keys repeat across the
        // input enough that per-node aggregation shrinks what the cross-node
        // merge must touch. With near-unique keys (input close to the
        // distinct estimate) the node tables would be as large as the input,
        // so merge every node's sources directly and pay the remote reads
        // once. An all-in-place merge is always direct: its tables are
        // per-worker aggregated already and their estimate equals their
        // total, so the test is never met.
        let hierarchical = node_count > 1 && {
            let scatter_rows: usize = buffers_by_node
                .iter()
                .flatten()
                .map(|b| b.total_rows())
                .sum();
            total_in_place + scatter_rows > 2 * estimate
        };
        // Wrap the arena's ring buffers once for the whole output phase (consume
        // is done, so `next_idx` is final). Every partition job shares this one
        // `Arc<[Buffer]>` for zero-copy string output, so a batch attaches it with
        // a single refcount bump instead of re-wrapping every buffer per batch.
        // Held by the jobs (and the batches they emit), never by the arena, so
        // there is no `arena -> Buffer -> arena` cycle.
        let output_buffers: Arc<[Buffer]> = self.key_arena.to_arrow_buffers();
        // With bin totals and enough groups to make pruning pay, order the jobs
        // by their upper bound, best first: the heaviest partitions merge
        // first, fill the workers' top-k heaps, and raise the shared threshold
        // that lets later jobs skip themselves.
        // Tests prune unconditionally so the skip path stays exercised at
        // test-sized group counts.
        let topk_bounds = (every_group_binned
            && (cfg!(test) || estimate >= MIN_TOPK_PRUNE_ESTIMATE))
            .then(|| {
                self.shared_bin_totals.take_sum().map(|mut totals| {
                    if let Some(bucket_rows) = &raw_bucket_rows {
                        totals.add_raw_bucket_rows(bucket_rows);
                    }
                    TopKBounds::build(totals)
                })
            })
            .flatten();
        let partition_bounds = topk_bounds
            .as_ref()
            .map(|bounds| bounds.partition_bounds(num_partitions));
        // With bounds, lead with the heaviest partitions: the first wave fills
        // the top-k heaps and raises the threshold later jobs skip against.
        // Only the head is sorted; the tail keeps ascending partition order,
        // because consecutive partitions read consecutive memory in every
        // source, and on a query whose bounds are flat (nothing prunable) a
        // fully sorted order is a random permutation that costs the whole
        // merge its locality.
        const SORTED_HEAD_PARTITIONS: usize = 128;
        let partition_order: Vec<usize> = {
            let mut order: Vec<usize> = (0..num_partitions).collect();
            if let Some(bounds) = &partition_bounds {
                let head = SORTED_HEAD_PARTITIONS.min(num_partitions - 1);
                order.select_nth_unstable_by_key(head, |&i| std::cmp::Reverse(bounds[i]));
                order[..head].sort_unstable_by_key(|&i| std::cmp::Reverse(bounds[i]));
                order[head..].sort_unstable();
            }
            order
        };
        // Hierarchical keeps one sources entry per node (one job per node and
        // partition, queued on the owning node so the bulk of every merge
        // reads node-local memory); direct flattens every node's sources into
        // a single entry.
        let sources = if hierarchical {
            buffers_by_node.into_iter().zip(tables_by_node).collect()
        } else {
            vec![(
                buffers_by_node.into_iter().flatten().collect(),
                tables_by_node.into_iter().flatten().collect(),
            )]
        };
        let shared = Arc::new(SharedMergeState {
            sources,
            cross_node: hierarchical
                .then(|| (0..num_partitions).map(|_| OnceLock::new()).collect()),
            key_arena: self.key_arena.clone(),
            output_buffers,
            partition_capacity,
            num_partitions,
            key_config: self.key_config.clone(),
            shared_context: self.shared_context.clone(),
            value_output_types: self.value_output_types.clone(),
            output_limit: self.output_limit,
            count_only: self.count_only,
            partition_bounds,
            topk_bounds,
            limit_progress: self.topk_threshold.clone(),
        });
        // All jobs are pushed before the injected flag flips, so a drained
        // queue means a finished phase.
        if hierarchical {
            // Each node table starts at the full `partition_capacity`, not a
            // per-node share: this path is chosen exactly when keys repeat
            // across workers, and row groups are hash-assigned to nodes, so a
            // repeating key reaches every node and each node's table converges
            // toward the partition's full distinct count. A per-node share
            // would guarantee a mid-merge resize; the full size only costs
            // transient memory the final cross-node merge frees.
            for &i in &partition_order {
                for (node, injector) in self.injectors.iter().enumerate() {
                    injector.push(PartitionJob {
                        shared: shared.clone(),
                        index: i,
                        node,
                    });
                }
            }
        } else {
            // One job per partition over every node's sources, spread across
            // the node queues so all workers share the load.
            for (position, &i) in partition_order.iter().enumerate() {
                self.injectors[position % node_count].push(PartitionJob {
                    shared: shared.clone(),
                    index: i,
                    node: 0,
                });
            }
        }

        // Release pairs with the Acquire load in `output`: a worker that sees
        // the flag also sees every job pushed above it. With relaxed ordering
        // another core may observe the flag before the pushes and conclude
        // from a still-empty queue that the merge phase is over.
        self.partition_jobs_injected.store(true, Ordering::Release);
    }
}

impl<K: KeyExtractor, V: AggregationValue + ?Sized> Outputter<RecordBatch>
    for GroupOutputter<K, V>
{
    fn output(&mut self, sender: &mut dyn Sender<RecordBatch>) -> unary::Result<bool> {
        if self.zero_hash_pending && self.count_only {
            self.zero_hash_pending = false;
            emit_count(1, sender).map_err(unary::Error::from)?;
        }

        // Claim a job, trying the own node's queue first (its sources are
        // node-local), then other nodes' queues so a node that finished its
        // share helps with the tail instead of idling. The plain-load
        // `is_empty` pre-checks keep the drained-queue polls (every pass
        // until the dataflow finishes) from pinning a crossbeam epoch each
        // time.
        let queue_count = self.injectors.len();
        let mut steal = Steal::Empty;
        for offset in 0..queue_count {
            let queue = &self.injectors[(self.node + offset) % queue_count];
            if queue.is_empty() {
                continue;
            }
            match queue.steal() {
                Steal::Empty => {}
                s => {
                    steal = s;
                    break;
                }
            }
        }
        match steal {
            Steal::Success(job) => {
                let allocator = self
                    .output_allocator
                    .get_or_insert_with(|| SlabAllocator::new(false));
                job.run_into(&mut self.output_accumulator, sender, allocator)
                    .map_err(unary::Error::from)?;
            }
            Steal::Empty => {
                if self.partition_jobs_injected.load(Ordering::Acquire) {
                    // Queue drained: emit this worker's last partial batch.
                    if let (Some(acc), Some(allocator)) = (
                        self.output_accumulator.as_mut(),
                        self.output_allocator.as_mut(),
                    ) {
                        acc.flush(allocator, sender).map_err(unary::Error::from)?;
                    }
                    return Ok(true);
                }
            }
            Steal::Retry => {}
        }

        Ok(false)
    }
}
