//! Partition-parallel merge of GROUP BY partial results.
//!
//! High hash bits select both the merge partition and the source-table slot
//! range. Each merge job therefore reads only its portion of every source:
//!
//! ```text
//! worker tables ----+
//!                   +--> partition target --> final node merge
//! scatter buffers --+
//! ```
//!
//! Tables with equal capacity are scanned together. Linear-probe entries that
//! crossed the nominal partition boundary are handled by a short overflow scan.

use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::SharedArena;
use crate::operations::unary::group::hashtables::{
    AggregationValue, DEFAULT_CAPACITY, MAX_LOAD_FACTOR, PersistedKey, Prober, StridedScatterRows,
};
use crate::operations::unary::group::hashtables::{HashTable, PartitionBuffers};
use crate::operations::unary::group::keys::StoredKey;
use crate::operations::unary::group::output::topk_pruning::TopKCutoff;
use crate::operations::unary::group::values::ArityBody;

/// Tables grouped by the slot count they were built at. Equal-sized tables share
/// partition boundaries, so the merge scans them together.
type TablesByCapacity<'a, S, V> = Vec<(usize, Vec<&'a HashTable<<S as StoredKey>::Persisted, V>>)>;

/// Collision-to-entry ratio at which we double the target table.
///
/// For linear probing, the cumulative collision ratio is
/// `alpha / (2 * (1 - alpha))`. A threshold of two allows dense result tables
/// without letting probe chains grow without bound.
const RESIZE_COLLISION_RATIO: f64 = 2.0;

/// Slots processed from each equal-sized source before moving to the next.
const SCAN_BATCH_SIZE: usize = 100;

const PREFETCH_DISTANCE: usize = 8;

/// Try to resize the target if cumulative collision pressure is too high.
#[inline]
fn resize_if_needed<KP: PersistedKey, V: AggregationValue + ?Sized>(
    allocator: &mut SlabAllocator,
    target: &mut Prober<'_, KP, V>,
) {
    // Use the integer form of `collisions / len > ratio`.
    target.resize_on_collisions(allocator, RESIZE_COLLISION_RATIO as usize);
}

/// Scan the expected slot range `[start, end)` across all source `tables`,
/// merging entries that belong to `partition` into `target`.
///
/// Equal-sized source tables are scanned in small batches so their target
/// region remains local.
#[allow(clippy::too_many_arguments)]
fn merge_within_partition_bounds<const N: usize, S: StoredKey, V: AggregationValue + ?Sized>(
    allocator: &mut SlabAllocator,
    arena: &SharedArena,
    partition: usize,
    slot_count: usize,
    tables: &[&HashTable<S::Persisted, V>],
    target: &mut Prober<'_, S::Persisted, V>,
    partition_bits: u32,
    context: &V::SharedContext,
    cutoff: Option<TopKCutoff<'_>>,
) {
    let partition_shift = u64::BITS - partition_bits;
    let start = (partition * slot_count) >> partition_bits;
    let end = ((partition + 1) * slot_count) >> partition_bits;
    let mut i = start;

    while i < end {
        let range = std::cmp::min(SCAN_BATCH_SIZE, end - i);

        for table in tables {
            // Reuse one source reader for this range.
            let source_reader = table.reader::<N>();
            for j in i..i + range {
                if j + PREFETCH_DISTANCE < slot_count {
                    // Prefetching hash zero is harmless and avoids a branch.
                    target.prefetch(source_reader.hash_at(j + PREFETCH_DISTANCE));
                }
                let entry = source_reader.view_at(j);
                let hash = entry.hash;
                // A non-short-circuiting AND combines both data-dependent tests.
                let take = (hash != 0) & ((hash >> partition_shift) as usize == partition);
                if take && cutoff.is_none_or(|cutoff| cutoff.admits(hash)) {
                    target.merge_from::<true, _>(
                        hash,
                        S::resolve_persisted(arena, *entry.key),
                        entry.stored,
                        context,
                    );
                    resize_if_needed::<S::Persisted, V>(allocator, target);
                }
            }
        }

        i += range;
    }
}

/// Scans the probe-chain overflow beyond a partition's nominal slot range.
#[allow(clippy::too_many_arguments)]
fn merge_past_partition_bounds<const N: usize, S: StoredKey, V: AggregationValue + ?Sized>(
    allocator: &mut SlabAllocator,
    arena: &SharedArena,
    partition: usize,
    slot_count: usize,
    tables: &[&HashTable<S::Persisted, V>],
    target: &mut Prober<'_, S::Persisted, V>,
    partition_bits: u32,
    context: &V::SharedContext,
    cutoff: Option<TopKCutoff<'_>>,
) {
    let partition_shift = u64::BITS - partition_bits;
    let end = ((partition + 1) * slot_count) >> partition_bits;
    let mask = slot_count - 1;

    for table in tables {
        let source_reader = table.reader::<N>();
        let mut i = end & mask;
        loop {
            let entry = source_reader.view_at(i);
            let hash = entry.hash;
            if hash == 0 {
                break;
            }
            if (hash >> partition_shift) as usize == partition
                && cutoff.is_none_or(|cutoff| cutoff.admits(hash))
            {
                target.merge_from::<true, _>(
                    hash,
                    S::resolve_persisted(arena, *entry.key),
                    entry.stored,
                    context,
                );
                resize_if_needed::<S::Persisted, V>(allocator, target);
            }
            i = (i + 1) & mask;
        }
    }
}

/// Merge entries from `tables` that belong to `partition` into `target`.
#[allow(clippy::too_many_arguments)]
fn merge_into_partition<const N: usize, S: StoredKey, V: AggregationValue + ?Sized>(
    allocator: &mut SlabAllocator,
    arena: &SharedArena,
    partition: usize,
    slot_count: usize,
    tables: Vec<&HashTable<S::Persisted, V>>,
    target: &mut Prober<'_, S::Persisted, V>,
    partition_bits: u32,
    context: &V::SharedContext,
    cutoff: Option<TopKCutoff<'_>>,
) {
    merge_within_partition_bounds::<N, S, V>(
        allocator,
        arena,
        partition,
        slot_count,
        &tables,
        target,
        partition_bits,
        context,
        cutoff,
    );
    merge_past_partition_bounds::<N, S, V>(
        allocator,
        arena,
        partition,
        slot_count,
        &tables,
        target,
        partition_bits,
        context,
        cutoff,
    );
}

/// Merges one partition's scatter rows and in-place tables.
///
/// Both sources use the same high hash bits, so no preliminary repartitioning
/// is required.
#[allow(clippy::too_many_arguments)]
pub(super) fn merge_combined<S: StoredKey, V: AggregationValue + ?Sized>(
    partition: usize,
    buffers: &[PartitionBuffers<S::Persisted, V>],
    tables: &[HashTable<S::Persisted, V>],
    partition_capacity: usize,
    num_partitions: usize,
    key_arena: &SharedArena,
    context: &V::SharedContext,
    cutoff: Option<TopKCutoff<'_>>,
) -> HashTable<S::Persisted, V> {
    // Dispatch once so every loop in this job shares the specialized arity.
    V::dispatch_arity(
        V::storage_metadata(context),
        MergeCombined::<S, V> {
            partition,
            buffers,
            tables,
            partition_capacity,
            num_partitions,
            key_arena,
            context,
            cutoff,
        },
    )
}

/// State passed through arity dispatch for one partition merge.
struct MergeCombined<'a, S: StoredKey, V: AggregationValue + ?Sized> {
    partition: usize,
    buffers: &'a [PartitionBuffers<S::Persisted, V>],
    tables: &'a [HashTable<S::Persisted, V>],
    partition_capacity: usize,
    num_partitions: usize,
    key_arena: &'a SharedArena,
    context: &'a V::SharedContext,
    cutoff: Option<TopKCutoff<'a>>,
}

impl<S: StoredKey, V: AggregationValue + ?Sized> ArityBody<HashTable<S::Persisted, V>>
    for MergeCombined<'_, S, V>
{
    #[inline(always)]
    fn run<const N: usize>(self) -> HashTable<S::Persisted, V> {
        let MergeCombined {
            partition,
            buffers,
            tables,
            partition_capacity,
            num_partitions,
            key_arena,
            context,
            cutoff,
        } = self;
        merge_combined_rows::<N, S, V>(
            partition,
            buffers,
            tables,
            partition_capacity,
            num_partitions,
            key_arena,
            context,
            cutoff,
        )
    }
}

/// Merges one partition after arity and runtime CPU dispatch.
///
/// Separate reference parameters preserve alias information across raw target
/// writes, keeping shared state outside the inner loops.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn merge_combined_rows<const N: usize, S: StoredKey, V: AggregationValue + ?Sized>(
    partition: usize,
    buffers: &[PartitionBuffers<S::Persisted, V>],
    tables: &[HashTable<S::Persisted, V>],
    partition_capacity: usize,
    num_partitions: usize,
    key_arena: &SharedArena,
    context: &V::SharedContext,
    cutoff: Option<TopKCutoff<'_>>,
) -> HashTable<S::Persisted, V> {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: each feature check covers every feature on its clone.
        if crate::cpu_features::supports_icelake_kernels() {
            return unsafe {
                merge_combined_rows_icelake::<N, S, V>(
                    partition,
                    buffers,
                    tables,
                    partition_capacity,
                    num_partitions,
                    key_arena,
                    context,
                    cutoff,
                )
            };
        }
        if crate::cpu_features::supports_v4_kernels() {
            return unsafe {
                merge_combined_rows_v4::<N, S, V>(
                    partition,
                    buffers,
                    tables,
                    partition_capacity,
                    num_partitions,
                    key_arena,
                    context,
                    cutoff,
                )
            };
        }
    }
    merge_combined_rows_body::<N, S, V>(
        partition,
        buffers,
        tables,
        partition_capacity,
        num_partitions,
        key_arena,
        context,
        cutoff,
    )
}

/// Ice Lake target-feature clone. Keep the attribute and runtime check aligned.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[target_feature(
    enable = "avx512f,avx512bw,avx512cd,avx512dq,avx512vl,avx512vbmi,avx512vbmi2,avx512vnni,avx512bitalg,avx512vpopcntdq,bmi1,bmi2,lzcnt,movbe,fma"
)]
fn merge_combined_rows_icelake<const N: usize, S: StoredKey, V: AggregationValue + ?Sized>(
    partition: usize,
    buffers: &[PartitionBuffers<S::Persisted, V>],
    tables: &[HashTable<S::Persisted, V>],
    partition_capacity: usize,
    num_partitions: usize,
    key_arena: &SharedArena,
    context: &V::SharedContext,
    cutoff: Option<TopKCutoff<'_>>,
) -> HashTable<S::Persisted, V> {
    merge_combined_rows_body::<N, S, V>(
        partition,
        buffers,
        tables,
        partition_capacity,
        num_partitions,
        key_arena,
        context,
        cutoff,
    )
}

/// x86-64-v4 target-feature clone (Skylake-SP / Cascade Lake). Keep the
/// attribute and runtime check aligned.
#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx512f,avx512bw,avx512cd,avx512dq,avx512vl,bmi1,bmi2,lzcnt,movbe,fma")]
fn merge_combined_rows_v4<const N: usize, S: StoredKey, V: AggregationValue + ?Sized>(
    partition: usize,
    buffers: &[PartitionBuffers<S::Persisted, V>],
    tables: &[HashTable<S::Persisted, V>],
    partition_capacity: usize,
    num_partitions: usize,
    key_arena: &SharedArena,
    context: &V::SharedContext,
    cutoff: Option<TopKCutoff<'_>>,
) -> HashTable<S::Persisted, V> {
    merge_combined_rows_body::<N, S, V>(
        partition,
        buffers,
        tables,
        partition_capacity,
        num_partitions,
        key_arena,
        context,
        cutoff,
    )
}

// Must stay `inline(always)` so each CPU tier compiles the merge loops with
// its own target features.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn merge_combined_rows_body<const N: usize, S: StoredKey, V: AggregationValue + ?Sized>(
    partition: usize,
    buffers: &[PartitionBuffers<S::Persisted, V>],
    tables: &[HashTable<S::Persisted, V>],
    partition_capacity: usize,
    num_partitions: usize,
    key_arena: &SharedArena,
    context: &V::SharedContext,
    cutoff: Option<TopKCutoff<'_>>,
) -> HashTable<S::Persisted, V> {
    {
        let partition_bits = num_partitions.trailing_zeros();
        let mut allocator = SlabAllocator::new(true);
        let mut capacity = partition_capacity.max(DEFAULT_CAPACITY);
        let mut result: HashTable<S::Persisted, V> =
            <HashTable<S::Persisted, V>>::new(&mut allocator, capacity, partition_bits, context);
        // Reuse one target probe layout across the partition.
        let mut target = if N == 0 {
            result.prober()
        } else {
            result.prober_with_metadata(V::metadata_for_arity::<N>())
        };

        // Scatter rows were not aggregated during consume, so each is inserted
        // once here.
        const SCATTER_PREFETCH_AHEAD: usize = 16;
        // Prefetch strings unconditionally and fixed-width keys only when the
        // target is large enough for probes to be cold.
        const TARGET_PREFETCH_MIN_SLOTS: usize = 16384;
        let merge_prefetch =
            <S::Persisted as PersistedKey>::HAS_BLOB || capacity > TARGET_PREFETCH_MIN_SLOTS;
        // A merge partition may own several consecutive scatter buckets. Both
        // counts are powers of two, so the ranges divide evenly; the bucket
        // release relies on no two partitions sharing a bucket.
        if let Some(first) = buffers.first() {
            let scatter_bucket_count = first.bucket_count();
            assert!(
                scatter_bucket_count.is_multiple_of(num_partitions),
                "merge partitions ({num_partitions}) must evenly divide scatter buckets ({scatter_bucket_count})"
            );
        }
        // All scatter buckets for this signature share one row layout.
        let scatter_layout = StridedScatterRows::<S::Persisted, V>::layout::<N>(context);
        for worker_buffers in buffers {
            for bucket in worker_buffers.bucket_range(partition, num_partitions) {
                if merge_prefetch {
                    worker_buffers.with_bucket(bucket, |rows| {
                        rows.for_each_prefetched::<SCATTER_PREFETCH_AHEAD>(
                            scatter_layout,
                            |hash, key, stored, ahead| {
                                if !cutoff.is_none_or(|cutoff| cutoff.admits(hash)) {
                                    return;
                                }
                                if let Some((ahead_hash, ahead_key)) = ahead {
                                    ahead_key.prefetch_blob(key_arena);
                                    target.prefetch(ahead_hash);
                                }
                                target.grow_if_full(&mut allocator, &mut capacity);
                                let live_key = S::resolve_persisted(key_arena, *key);
                                target.merge_from::<false, _>(hash, live_key, stored, context);
                            },
                        )
                    });
                } else {
                    worker_buffers.with_bucket(bucket, |rows| {
                        rows.for_each(scatter_layout, |hash, key, stored| {
                            if !cutoff.is_none_or(|cutoff| cutoff.admits(hash)) {
                                return;
                            }
                            target.grow_if_full(&mut allocator, &mut capacity);
                            let live_key = S::resolve_persisted(key_arena, *key);
                            target.merge_from::<false, _>(hash, live_key, stored, context);
                        })
                    });
                }
            }
        }

        // Scan equal-sized tables together so they share partition boundaries.
        let mut tables_by_capacity: TablesByCapacity<'_, S, V> = Vec::new();
        for table in tables {
            let slot_count = table.capacity();
            if let Some((_, same_size_tables)) = tables_by_capacity
                .iter_mut()
                .find(|(group_slot_count, _)| *group_slot_count == slot_count)
            {
                same_size_tables.push(table);
            } else {
                tables_by_capacity.push((slot_count, vec![table]));
            }
        }
        for (slot_count, same_size_tables) in tables_by_capacity {
            merge_into_partition::<N, S, V>(
                &mut allocator,
                key_arena,
                partition,
                slot_count,
                same_size_tables,
                &mut target,
                partition_bits,
                context,
                cutoff,
            );
        }
        result
    }
}
/// Merges one partition's node-local results.
///
/// The largest input is reused when it can hold the combined upper bound.
/// Otherwise a new table is allocated at the estimated partition capacity.
pub(super) fn merge_node_aggregated_tables<S: StoredKey, V: AggregationValue + ?Sized>(
    mut node_tables: Vec<HashTable<S::Persisted, V>>,
    partition_capacity: usize,
    partition_bits: u32,
    key_arena: &SharedArena,
    context: &V::SharedContext,
) -> HashTable<S::Persisted, V> {
    let mut allocator = SlabAllocator::new(true);
    let total: usize = node_tables.iter().map(|table| table.len()).sum();
    let largest_table_index = node_tables
        .iter()
        .enumerate()
        .max_by_key(|(_, table)| table.len())
        .map(|(index, _)| index)
        .expect("a partition always has at least one node_table");
    let mut target =
        if total as f64 <= node_tables[largest_table_index].capacity() as f64 * MAX_LOAD_FACTOR {
            node_tables.swap_remove(largest_table_index)
        } else {
            let capacity = partition_capacity.max(DEFAULT_CAPACITY);
            <HashTable<S::Persisted, V>>::new(&mut allocator, capacity, partition_bits, context)
        };
    let mut prober = target.prober();
    for node_table in &node_tables {
        for entry in node_table.iter(0) {
            prober.merge_from::<true, _>(
                entry.hash,
                S::resolve_persisted(key_arena, *entry.key),
                entry.stored,
                context,
            );
            resize_if_needed::<S::Persisted, V>(&mut allocator, &mut prober);
        }
    }
    // The prober's borrow of `target` ends at its last use above.
    target
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RECORD_BATCH_SIZE;
    use crate::memory::init_test_free_pool;
    use crate::operations::unary::group::arena::SharedArena;
    use crate::operations::unary::group::hashtables::{
        AggregatedTable, AggregatedTableOutput, RadixConfig,
    };
    use crate::operations::unary::group::keys::{InlineKey, IntKeyExtractor};
    use crate::operations::unary::group::values::{
        AggregationKind, AggregationSlot, AggregationValue, Compiled, CountSlot,
    };
    use ahash::RandomState;
    use arrow_array::types::Int32Type;
    use arrow_array::{ArrayRef, Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    /// The fan-out these tests merge at. Production sizes its partition count
    /// per query; the merge itself only needs a power of two.
    const PARTITIONS: usize = 64;

    type IntExtractor = IntKeyExtractor<Int32Type>;
    // How `IntExtractor` stores its keys. The merge is generic over this rather
    // than over the extractor, so this is what selects it.
    type IntStored = InlineKey<i32>;
    // A single `COUNT` slot. `Compiled` is numeric-only, so its `SharedContext`
    // is concretely `()`.
    type CountValue = Compiled<(CountSlot,), u8>;
    const COUNT_CFG: <CountValue as AggregationValue>::SharedContext = ();

    fn make_worker_tables(
        state: &RandomState,
        arena: &Arc<SharedArena>,
        values: &[i32],
    ) -> Vec<HashTable<i32, CountValue>> {
        let mut agg = AggregatedTable::<IntExtractor, CountValue>::new(
            state.clone(),
            arena.clone(),
            (),
            (),
            RadixConfig::DEFAULT,
            None,
        );
        let array: ArrayRef = Arc::new(Int32Array::from(values.to_vec()));
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
        let batch = RecordBatch::try_new(schema, vec![array]).unwrap();
        agg.consume_batch(
            &batch,
            &[0],
            &[AggregationSlot::new(
                AggregationKind::CountStar,
                0,
                DataType::Int64,
            )],
            &(),
            &COUNT_CFG,
        );
        // The test data is low-cardinality, so the worker never switches to radix:
        // `buffers` is None and the full result is in the in-place stack.
        let out = agg.flush().0;
        assert!(out.buffers.is_none(), "low-cardinality test stays in-place");
        out.tables
    }

    fn merge_all_partitions(
        tables: &[HashTable<i32, CountValue>],
        arena: &SharedArena,
    ) -> Vec<(i32, usize)> {
        let total_cap: usize = tables.iter().map(|t| t.capacity()).sum::<usize>() / 2;
        let partition_cap = (total_cap / PARTITIONS).max(1).next_power_of_two();

        let mut all_entries = vec![];
        for p in 0..PARTITIONS {
            let result = merge_combined::<IntStored, CountValue>(
                p,
                &[],
                tables,
                partition_cap,
                PARTITIONS,
                arena,
                &COUNT_CFG,
                None,
            );
            for entry in result.iter(0) {
                all_entries.push((*entry.key, entry.stored.sort_key(0) as usize));
            }
        }
        all_entries.sort_by_key(|(k, _)| *k);
        all_entries
    }

    #[test]
    fn single_worker_all_entries_preserved() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let tables = make_worker_tables(&state, &arena, &[1, 2, 3, 4, 5]);

        let entries = merge_all_partitions(&tables, &arena);

        assert_eq!(entries.len(), 5);
        for (key, count) in &entries {
            assert_eq!(*count, 1, "key {} should have count 1", key);
        }
    }

    #[test]
    fn two_workers_disjoint_keys() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let mut tables = make_worker_tables(&state, &arena, &[1, 2, 3]);
        tables.extend(make_worker_tables(&state, &arena, &[4, 5, 6]));

        let entries = merge_all_partitions(&tables, &arena);

        assert_eq!(entries.len(), 6);
    }

    #[test]
    fn two_workers_overlapping_keys_merged() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let mut tables = make_worker_tables(&state, &arena, &[1, 2, 3]);
        tables.extend(make_worker_tables(&state, &arena, &[2, 3, 4]));

        let entries = merge_all_partitions(&tables, &arena);

        assert_eq!(entries.len(), 4);
        let count_for = |k: i32| entries.iter().find(|(key, _)| *key == k).unwrap().1;
        assert_eq!(count_for(1), 1);
        assert_eq!(count_for(2), 2);
        assert_eq!(count_for(3), 2);
        assert_eq!(count_for(4), 1);
    }

    #[test]
    fn empty_tables_produce_no_entries() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let tables = make_worker_tables(&state, &arena, &[]);

        let entries = merge_all_partitions(&tables, &arena);

        assert_eq!(entries.len(), 0);
    }

    #[test]
    fn many_workers_large_overlap() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let mut tables = vec![];
        for _ in 0..8 {
            tables.extend(make_worker_tables(&state, &arena, &[10, 20, 30]));
        }

        let entries = merge_all_partitions(&tables, &arena);

        assert_eq!(entries.len(), 3);
        for (_, count) in &entries {
            assert_eq!(*count, 8);
        }
    }

    #[test]
    fn partitions_are_disjoint() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let values: Vec<i32> = (0..200).collect();
        let tables = make_worker_tables(&state, &arena, &values);

        let total_cap: usize = tables.iter().map(|t| t.capacity()).sum::<usize>() / 2;
        let partition_cap = (total_cap / PARTITIONS).max(1).next_power_of_two();
        let mut total = 0;
        for p in 0..PARTITIONS {
            let result = merge_combined::<IntStored, CountValue>(
                p,
                &[],
                &tables,
                partition_cap,
                PARTITIONS,
                &arena,
                &COUNT_CFG,
                None,
            );
            total += result.iter(0).count();
        }

        assert_eq!(total, 200);
    }

    #[test]
    fn mixed_size_tables_merge_correctly() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let small = make_worker_tables(&state, &arena, &[1, 2]);
        let big_values: Vec<i32> = (0..500).collect();
        let big = make_worker_tables(&state, &arena, &big_values);
        let mut tables = small;
        tables.extend(big);

        let entries = merge_all_partitions(&tables, &arena);

        assert_eq!(entries.len(), 500);
        let count_for = |k: i32| entries.iter().find(|(key, _)| *key == k).unwrap().1;
        assert_eq!(count_for(1), 2);
        assert_eq!(count_for(2), 2);
        assert_eq!(count_for(499), 1);
    }

    #[test]
    fn duplicates_within_single_worker_carry_through_merge() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let tables = make_worker_tables(&state, &arena, &[5, 5, 5, 5, 5]);

        let entries = merge_all_partitions(&tables, &arena);

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0], (5, 5));
    }

    /// Node-hierarchical merges combine per-node aggregated tables (each a
    /// `merge_combined` result for the same partition) into one final table;
    /// keys shared across nodes must combine, node-disjoint keys must all
    /// survive.
    #[test]
    fn node_tables_merge_across_nodes() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let node_a = make_worker_tables(&state, &arena, &[1, 2, 3]);
        let node_b = make_worker_tables(&state, &arena, &[2, 3, 4]);

        let mut entries = vec![];
        for p in 0..PARTITIONS {
            let node_tables = vec![
                merge_combined::<IntStored, CountValue>(
                    p,
                    &[],
                    &node_a,
                    4,
                    PARTITIONS,
                    &arena,
                    &COUNT_CFG,
                    None,
                ),
                merge_combined::<IntStored, CountValue>(
                    p,
                    &[],
                    &node_b,
                    4,
                    PARTITIONS,
                    &arena,
                    &COUNT_CFG,
                    None,
                ),
            ];
            let folded = merge_node_aggregated_tables::<IntStored, CountValue>(
                node_tables,
                DEFAULT_CAPACITY,
                PARTITIONS.trailing_zeros(),
                &arena,
                &COUNT_CFG,
            );
            for entry in folded.iter(0) {
                entries.push((*entry.key, entry.stored.sort_key(0) as usize));
            }
        }

        entries.sort();
        assert_eq!(entries, vec![(1, 1), (2, 2), (3, 2), (4, 1)]);
    }

    /// Feed `values` through a worker in batch-sized chunks and hand back its
    /// finalized output (in-place stack + scatter buffers if it switched).
    fn make_worker_output(
        state: &RandomState,
        arena: &Arc<SharedArena>,
        values: &[i32],
    ) -> AggregatedTableOutput<IntExtractor, CountValue> {
        let mut agg = AggregatedTable::<IntExtractor, CountValue>::new(
            state.clone(),
            arena.clone(),
            (),
            (),
            RadixConfig::DEFAULT,
            None,
        );
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, false)]));
        // Consume one batch at a time because hash scratch has a fixed size.
        for chunk in values.chunks(RECORD_BATCH_SIZE) {
            let array: ArrayRef = Arc::new(Int32Array::from(chunk.to_vec()));
            let batch = RecordBatch::try_new(schema.clone(), vec![array]).unwrap();
            agg.consume_batch(
                &batch,
                &[0],
                &[AggregationSlot::new(
                    AggregationKind::CountStar,
                    0,
                    DataType::Int64,
                )],
                &(),
                &COUNT_CFG,
            );
        }
        agg.flush().0
    }

    /// A worker with enough distinct keys to switch to radix produces scatter
    /// buffers *and* a pre-switch in-place stack. The radix merge must combine
    /// both at RADIX_PARTITIONS granularity, landing every key in exactly one
    /// partition with the right count, losing and double-counting nothing.
    /// The radix path slot-range-merges in-place stacks at RADIX_PARTITIONS, a
    /// granularity that can exceed a stack table's slot count, the regime the
    /// 64-way merge never hits. Exercise it directly: 200 distinct keys (stack
    /// tables of 128 and 512 slots) merged at `num_partitions = 256`, finer than
    /// the 128-slot table. Each key must still land in exactly one partition with
    /// the right count (the degenerate within-range plus probe-chain spillover in
    /// `merge_past_partition_bounds` has to lose and duplicate nothing).
    #[test]
    fn slot_range_merge_finer_than_table() {
        init_test_free_pool(64);
        let arena = SharedArena::new(64);
        let state = RandomState::new();
        let n = 200i32;
        let values: Vec<i32> = (0..n).chain(0..n).collect(); // each key twice
        let out = make_worker_output(&state, &arena, &values);
        assert!(out.buffers.is_none(), "200 keys stay in-place");

        let num_partitions = 256;
        let mut occurrences = vec![0usize; n as usize];
        let mut counts = vec![0usize; n as usize];
        for p in 0..num_partitions {
            let result = merge_combined::<IntStored, CountValue>(
                p,
                &[],
                &out.tables,
                DEFAULT_CAPACITY,
                num_partitions,
                &arena,
                &COUNT_CFG,
                None,
            );
            for entry in result.iter(0) {
                occurrences[*entry.key as usize] += 1;
                counts[*entry.key as usize] += entry.stored.sort_key(0) as usize;
            }
        }

        for k in 0..n as usize {
            assert_eq!(
                occurrences[k], 1,
                "key {k} must be in exactly one partition"
            );
            assert_eq!(counts[k], 2, "key {k} has the wrong count");
        }
    }
}
