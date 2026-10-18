//! Assembles a merged partition table into output [`RecordBatch`]es.
//!
//! This module combines key columns with aggregation columns. Neither side
//! needs to know the other's concrete representation.
//!
//! Rows from multiple partitions are accumulated into full output batches.

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch};
use arrow_buffer::Buffer;
use arrow_schema::{DataType, Field as ArrowField, Schema};

use crate::cpu_features::multitarget_kernel;
use crate::memory::{
    BUFFER_SIZE, HeapBuffer, MultiTopK, Ranked, SingleTopK, SlabAllocator, SlabTopK, slots_per_slab,
};
use crate::operations::channels::Sender;
use crate::operations::unary::group::arena::SharedArena;
use crate::operations::unary::group::keys::{KeyColumnBuilder, KeyExtractor};
use crate::operations::unary::group::output::topk_pruning::{TopKThreshold, TopKWeight};
use crate::operations::unary::group::values::{
    AggregationValue, ValueColumnBuilder, WorkerContext, cast_value_column,
};

use crate::operations::unary::group::hashtables::HashTable;
use crate::operations::unary::group::{GroupLimit, Result};

/// Maximum rows whose widest fixed-width column fits in one slab.
const OUTPUT_CHUNK_ROWS: usize = BUFFER_SIZE / 16;

/// Top-k heap shared across all partitions processed by one worker.
///
/// Limits that fit one slab use contiguous storage. Larger limits use
/// multi-slab storage. Selecting the representation once keeps row offers
/// statically dispatched.
enum TopKHeap<K: KeyExtractor, V: AggregationValue + ?Sized> {
    Single {
        slot: usize,
        heap: SingleTopK<V::SortKey, (K::Persisted, V::Owned)>,
    },
    Multi {
        slot: usize,
        heap: MultiTopK<V::SortKey, (K::Persisted, V::Owned)>,
    },
}

impl<K: KeyExtractor, V: AggregationValue + ?Sized> TopKHeap<K, V> {
    /// A heap that keeps the top `limit` groups by `sort_key(slot)`.
    fn new(allocator: &mut SlabAllocator, slot: usize, limit: usize) -> Self {
        if limit <= slots_per_slab::<V::SortKey, (K::Persisted, V::Owned)>() {
            Self::Single {
                slot,
                heap: SlabTopK::single(allocator, limit),
            }
        } else {
            Self::Multi {
                slot,
                heap: SlabTopK::multi(allocator, limit),
            }
        }
    }
}

/// Offers every group in a table to a top-k heap, and its weight to the
/// shared k-th best threshold when the merge is pruning by top-k bound.
/// These groups are fully merged, so each weight is a group's exact value.
///
/// The retention check precedes [`AggregationValue::to_owned`], so dynamic
/// cells are copied only for candidates the heap can keep.
fn offer_all<K, V, A>(
    heap: &mut SlabTopK<V::SortKey, (K::Persisted, V::Owned), A>,
    slot: usize,
    table: &HashTable<K::Persisted, V>,
    allocator: &mut SlabAllocator,
    context: &V::SharedContext,
    owned_context: &mut Option<V::WorkerContext>,
    topk_threshold: Option<&TopKThreshold>,
) where
    K: KeyExtractor,
    V: AggregationValue + ?Sized,
    A: HeapBuffer<Ranked<V::SortKey, (K::Persisted, V::Owned)>>,
{
    for entry in table.iter(0) {
        let sort_key = entry.stored.sort_key(slot);
        if let Some(threshold) = topk_threshold {
            threshold.offer(sort_key.saturating_weight());
        }
        if heap.would_retain(sort_key) {
            let value = entry.stored.to_owned(context, owned_context);
            heap.offer(allocator, sort_key, (*entry.key, value));
        }
    }
}

/// Per-worker output pruning mode.
enum OutputMode<K: KeyExtractor, V: AggregationValue + ?Sized> {
    /// Keeps the worker's best rows for ORDER BY and LIMIT.
    TopK(TopKHeap<K, V>),
    /// Top-k keeping ties: stream every merged group (the downstream sort
    /// applies the tiebreaks a per-worker heap would violate), feeding each
    /// group's weight to the shared k-th best so the merge can prune.
    TopKWithTies { slot: usize },
    /// Remaining row budget for an unordered LIMIT.
    First { remaining: usize },
    /// No pushdown: stream every group.
    Unlimited,
}

impl<K: KeyExtractor, V: AggregationValue + ?Sized> OutputMode<K, V> {
    fn new(allocator: &mut SlabAllocator, output_limit: Option<GroupLimit>) -> Self {
        match output_limit {
            Some(GroupLimit::TopK {
                slot,
                with_ties: true,
                ..
            }) => Self::TopKWithTies { slot },
            Some(GroupLimit::TopK { slot, limit, .. }) => {
                Self::TopK(TopKHeap::new(allocator, slot, limit))
            }
            Some(GroupLimit::First { limit }) => Self::First { remaining: limit },
            None => Self::Unlimited,
        }
    }
}

/// Accumulates groups from multiple partitions into output batches.
///
/// LIMIT state is worker-wide rather than partition-local, which allows many
/// small radix partitions to be pruned together.
pub(crate) struct OutputAccumulator<K: KeyExtractor, V: AggregationValue + ?Sized> {
    /// Key and value builders for the batch currently being filled.
    key_builder: K::ColumnBuilder,
    value_builder: V::ColumnBuilder,
    /// Allocated row capacity for each builder.
    builder_capacity: usize,
    len: usize,
    /// Which pushed LIMIT, if any, prunes this worker's output.
    mode: OutputMode<K, V>,
    key_arena: Arc<SharedArena>,
    output_buffers: Arc<[Buffer]>,
    key_config: K::Config,
    shared_context: V::SharedContext,
    /// Declared output type for each aggregation slot.
    value_output_types: Arc<[DataType]>,
    /// Lazily created arena context for dynamic values retained by top-k.
    owned_copy_context: Option<V::WorkerContext>,
    /// Shared pushed-limit progress: the k-th best tracker merged group
    /// weights are offered to, and the emitted-group count a plain LIMIT
    /// skips against.
    limit_progress: Arc<TopKThreshold>,
    /// Whether merged group weights feed the k-th best tracker (only when the
    /// merge phase is pruning by top-k bound; offers are useless otherwise).
    offer_weights: bool,
}

impl<K: KeyExtractor, V: AggregationValue + ?Sized> OutputAccumulator<K, V> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        allocator: &mut SlabAllocator,
        output_limit: Option<GroupLimit>,
        key_arena: Arc<SharedArena>,
        output_buffers: Arc<[Buffer]>,
        key_config: K::Config,
        shared_context: V::SharedContext,
        value_output_types: Arc<[DataType]>,
        limit_progress: Arc<TopKThreshold>,
        offer_weights: bool,
    ) -> Self {
        // A pushed LIMIT can reduce the required builder capacity.
        let builder_capacity = output_limit.map_or(OUTPUT_CHUNK_ROWS, |limit| {
            limit.row_limit().clamp(1, OUTPUT_CHUNK_ROWS)
        });
        Self {
            key_builder: K::ColumnBuilder::with_capacity(allocator, builder_capacity, &key_config),
            value_builder: V::ColumnBuilder::with_capacity(
                allocator,
                builder_capacity,
                &shared_context,
            ),
            builder_capacity,
            len: 0,
            mode: OutputMode::new(allocator, output_limit),
            key_arena,
            output_buffers,
            key_config,
            shared_context,
            value_output_types,
            owned_copy_context: None,
            limit_progress,
            offer_weights,
        }
    }

    /// Appends one group directly from a table entry.
    #[inline]
    fn push_entry(&mut self, key: &K::Persisted, stored: &V) {
        self.key_builder.push(key);
        self.value_builder.push_stored(stored);
        self.len += 1;
    }

    /// Appends one owned group from the top-k heap.
    #[inline]
    fn push_owned(&mut self, key: &K::Persisted, value: &V::Owned) {
        self.key_builder.push(key);
        self.value_builder.push(value);
        self.len += 1;
    }

    multitarget_kernel! {
        /// Appends a partition table with optional worker-wide pruning. x86
        /// selects the CPU tier at runtime.
        pub(crate) fn extend_from_table(
            &mut self,
            table: HashTable<K::Persisted, V>,
            allocator: &mut SlabAllocator,
            sender: &mut dyn Sender<RecordBatch>,
        ) -> Result<()> {
        // Match the top-k backing once per table, not once per row.
        {
            let Self {
                mode,
                shared_context,
                owned_copy_context,
                limit_progress,
                offer_weights,
                ..
            } = self;
            let threshold = offer_weights.then_some(&**limit_progress);
            match mode {
                OutputMode::TopK(TopKHeap::Single { slot, heap }) => {
                    offer_all::<K, V, _>(
                        heap,
                        *slot,
                        &table,
                        allocator,
                        shared_context,
                        owned_copy_context,
                        threshold,
                    );
                    return Ok(());
                }
                OutputMode::TopK(TopKHeap::Multi { slot, heap }) => {
                    offer_all::<K, V, _>(
                        heap,
                        *slot,
                        &table,
                        allocator,
                        shared_context,
                        owned_copy_context,
                        threshold,
                    );
                    return Ok(());
                }
                OutputMode::TopKWithTies { .. } | OutputMode::First { .. } | OutputMode::Unlimited => {
                }
            }
        }
        match &mut self.mode {
            // Stream every group, feeding the shared k-th best so the merge
            // prunes what the downstream tiebreaking sort could never keep.
            OutputMode::TopKWithTies { slot } => {
                let slot = *slot;
                let offer = self.offer_weights;
                for entry in table.iter(0) {
                    if offer {
                        self.limit_progress
                            .offer(entry.stored.sort_key(slot).saturating_weight());
                    }
                    self.push_entry(entry.key, entry.stored);
                    self.flush_if_full(allocator, sender)?;
                }
            }
            // Preserve the remaining budget for later partitions.
            OutputMode::First { remaining } => {
                let mut remaining = *remaining;
                let mut pushed = 0usize;
                for entry in table.iter(0) {
                    if remaining == 0 {
                        break;
                    }
                    remaining -= 1;
                    pushed += 1;
                    self.push_entry(entry.key, entry.stored);
                    self.flush_if_full(allocator, sender)?;
                }
                // Publish progress so merge jobs stop once any `limit` groups
                // exist pool-wide.
                self.limit_progress.add_emitted_groups(pushed);
                self.mode = OutputMode::First { remaining };
            }
            // Without pushdown, stream every group.
            OutputMode::Unlimited => {
                for entry in table.iter(0) {
                    self.push_entry(entry.key, entry.stored);
                    self.flush_if_full(allocator, sender)?;
                }
            }
            OutputMode::TopK(_) => unreachable!("top-k handled above"),
        }
        Ok(())
        }
    }

    #[inline]
    fn flush_if_full(
        &mut self,
        allocator: &mut SlabAllocator,
        sender: &mut dyn Sender<RecordBatch>,
    ) -> Result<()> {
        if self.len >= OUTPUT_CHUNK_ROWS {
            self.flush(allocator, sender)
        } else {
            Ok(())
        }
    }

    /// Emits accumulated rows and resets the builders.
    pub(crate) fn flush(
        &mut self,
        allocator: &mut SlabAllocator,
        sender: &mut dyn Sender<RecordBatch>,
    ) -> Result<()> {
        if let Some(top_k) = self.take_topk() {
            match top_k {
                TopKHeap::Single { mut heap, .. } => {
                    for (key, value) in heap.values() {
                        self.push_owned(&key, &value);
                        self.flush_if_full(allocator, sender)?;
                    }
                }
                TopKHeap::Multi { mut heap, .. } => {
                    for (key, value) in heap.values() {
                        self.push_owned(&key, &value);
                        self.flush_if_full(allocator, sender)?;
                    }
                }
            }
            // Owned values have been copied into the output columns.
            if let Some(worker_context) = self.owned_copy_context.take() {
                worker_context.flush();
            }
        }
        if self.len == 0 {
            return Ok(());
        }
        // Swap in fresh builders and emit the filled ones. (`emit` consumes the
        // builders to finish them, so they have to be replaced, not borrowed.)
        let key_builder = std::mem::replace(
            &mut self.key_builder,
            K::ColumnBuilder::with_capacity(allocator, self.builder_capacity, &self.key_config),
        );
        let value_builder = std::mem::replace(
            &mut self.value_builder,
            V::ColumnBuilder::with_capacity(allocator, self.builder_capacity, &self.shared_context),
        );
        self.len = 0;
        self.emit(key_builder, value_builder, allocator, sender)
    }

    /// Build the accumulated key and value columns into one `RecordBatch` and send
    /// it. Reads the per-output-phase state (`key_arena`, `output_buffers`,
    /// `shared_context`) straight off `self`; the caller hands over the filled
    /// builders to finish.
    fn emit(
        &self,
        key_builder: K::ColumnBuilder,
        value_builder: V::ColumnBuilder,
        allocator: &mut SlabAllocator,
        sender: &mut dyn Sender<RecordBatch>,
    ) -> Result<()> {
        let (mut fields, mut columns) =
            key_builder.finish(&self.key_arena, &self.output_buffers, allocator);
        let (value_fields, value_columns) = value_builder.finish(&self.shared_context);
        // The accumulator renders each value at its storage width; cast it to the
        // slot's declared output type (zero-cost when they already match).
        for ((field, column), output_type) in value_fields
            .into_iter()
            .zip(value_columns)
            .zip(self.value_output_types.iter())
        {
            let (field, column) = cast_value_column(field, column, output_type);
            fields.push(field);
            columns.push(column);
        }
        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)?;
        sender.send(batch)?;
        Ok(())
    }

    /// Take the top-k heap if in `TopK` mode, switching to `Unlimited` so the heap
    /// is drained exactly once. Leaves `First`/`Unlimited` modes untouched (a
    /// mid-stream flush there keeps its budget).
    fn take_topk(&mut self) -> Option<TopKHeap<K, V>> {
        if matches!(self.mode, OutputMode::TopK(_)) {
            match std::mem::replace(&mut self.mode, OutputMode::Unlimited) {
                OutputMode::TopK(heap) => Some(heap),
                _ => unreachable!(),
            }
        } else {
            None
        }
    }
}

/// Global `COUNT(DISTINCT)`: emit a partition's distinct-key count as a single
/// `Int64` row instead of materialising every key column (hundreds of MB of pure
/// waste at high cardinality); a downstream `SUM` over the per-partition counts
/// gives the total (partitions are hash-disjoint). Callers skip `count == 0`
/// partitions, so every emitted count is non-zero.
pub(crate) fn emit_count(count: usize, sender: &mut dyn Sender<RecordBatch>) -> Result<()> {
    let arr = Arc::new(Int64Array::from(vec![count as i64]));
    let field = ArrowField::new("v0", DataType::Int64, false);
    let batch = RecordBatch::try_new(Arc::new(Schema::new(vec![field])), vec![arr])?;
    sender.send(batch)?;
    Ok(())
}
