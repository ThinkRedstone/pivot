//! Streaming Parquet file construction for INSERT and compaction.
//!
//! INSERT prepares files as follows:
//!
//! 1. [`partition_sorter`] splits each input batch by table partition, sorts
//!    each piece, and copies it into independent memory. Each piece is one
//!    sorted run located on the NUMA node that produced it.
//! 2. [`file_collector`] chooses the runs belonging to each output file and
//!    groups them by source node. Every worker collects the runs it sorted
//!    into pending runs all workers share, and the byte target applies to the
//!    retained Arrow data each partition has pending.
//!
//! Compaction replaces those two stages: its selected files already share one
//! known partition, and every decoded batch retains its source file's sort
//! order. It wraps each batch as a run without partitioning, sorting, or copying,
//! then collects all selected runs into one file without a memory-size cutoff.
//!
//! Both paths share the remainder:
//!
//! 3. The local planner in [`file_merge`] reads the sort keys of one node's runs
//!    and divides their k-way merge into independent output slices. It does not
//!    copy rows. A node-local work queue keeps this planning beside the input
//!    data and distributes the resulting slice jobs among workers on that node.
//! 4. The local executors heap-merge each slice, gather every column, and
//!    produce one sorted run for the node.
//! 5. The global planner waits for all participating nodes, then divides the
//!    k-way merge of their node-level runs into global output slices. A shared
//!    queue is appropriate here because the inputs already span nodes.
//! 6. The global executors materialize those slices on their selected NUMA
//!    nodes. Together, in slice order, they are the fully sorted file input.
//! 7. [`row_group_planner`] divides the ordered file into column jobs.
//! 8. [`shredder`] splits each column job into its primitive leaves, shredding
//!    a variant column a record batch of rows per worker turn.
//! 9. [`encoder`] materializes and encodes each leaf as Parquet pages.
//! 10. [`assembler`] returns encoded leaf chunks to one worker per file and
//!     produces the final file bytes and footer metadata.
//!
//! In outline, the two merge levels are:
//!
//! ```text
//! node 0 runs ── local k-way merge ── node 0 run ─┐
//! node 1 runs ── local k-way merge ── node 1 run ─┼─ global k-way merge ─ file
//! node N runs ── local k-way merge ── node N run ─┘
//! ```
//!
//! A merge level with one run is a zero-copy identity. Files without sort
//! columns pass through all merge stages without planning or copying rows.
//!
//! An unsorted file skips the merge work. Files remain on dispatch workers
//! through assembly and upload because their bytes occupy worker-owned ring
//! memory.

mod assembler;
mod compression;
pub(crate) mod encoder;
pub(crate) mod error;
mod file_collector;
mod file_merge;
mod leaves;
mod partition_sorter;
mod presorted_run;
mod row_group_planner;
mod shredder;
mod shredding;
mod stats;
pub use stats::aggregate_file_stats;
mod types;

pub use types::AssembledFile;

use std::sync::Arc;

use arrow_array::RecordBatch;
use arrow_schema::SchemaRef;
use dispatch::memory::SlabAllocator;
use dispatch::{
    DefaultUnaryFactory, OperatorFactory, OperatorSpec, OrderBy, RecordBatchOperatorSpec,
    node_work_queue, return_to_worker_mpsc, shared_work_queue, stealable, to_single_worker_mpsc,
};

use partition_sorter::{PartitionSorterFactory, SortedPartitionRun};
use presorted_run::PresortedRun;
use types::{
    ColumnChunkJob, EncodedLeafChunk, FileOrderInput, GlobalMergeJob, LeafChunkJob, LocalMergeJob,
    LocalMergeResult, ReadyFile,
};

/// Appends Parquet construction to an existing record-batch dataflow.
///
/// The retained-memory target cuts each partition's INSERT rows into output
/// files. Callers first restore shredded variants with
/// [`unshred_batches_spec`].
pub fn encode_record_batches_spec(
    spec: RecordBatchOperatorSpec,
    schema: SchemaRef,
    partition_column_names: Arc<[String]>,
    sort_column_names: Arc<[String]>,
    target_rows_per_group: usize,
    target_in_memory_bytes_per_file: usize,
) -> OperatorSpec<AssembledFile, impl OperatorFactory<AssembledFile> + 'static> {
    let partition_column_indices: Vec<usize> = partition_column_names
        .iter()
        .map(|name| {
            schema
                .index_of(name)
                .expect("partition columns name declared columns")
        })
        .collect();
    let order_by = sort_order(&schema, &sort_column_names);

    let (dispatcher, heads) = spec.into_parts();
    let input_batches = OperatorSpec::new(dispatcher, heads.into_iter().collect::<Vec<_>>());
    let topology = input_batches.dispatcher().topology();
    let files = input_batches
        .chain(
            stealable::<RecordBatch>(topology).into_iter().collect(),
            PartitionSorterFactory::create_for_workers(
                partition_column_indices,
                order_by.clone(),
                topology,
            ),
        )
        // Each worker collects the runs it sorted: the collectors share their
        // pending runs, so no run queues behind a single collecting worker.
        .chain(
            stealable::<SortedPartitionRun>(topology)
                .into_iter()
                .collect(),
            file_collector::factories(
                partition_column_names,
                order_by.into(),
                target_rows_per_group,
                target_in_memory_bytes_per_file,
                None,
                topology,
            ),
        );
    encode_files_spec(files)
}

/// Appends Parquet construction to a compaction scan. Every input batch already
/// belongs to the selected partition and is ordered as it was in its source
/// file, so the INSERT-only partition/sort/copy stage is skipped. All batches
/// form one ordered output stream, split at target-sized row-group boundaries;
/// sorted tables still use the downstream k-way merge to combine
/// overlapping source-file runs.
pub fn encode_compaction_batches_spec(
    spec: RecordBatchOperatorSpec,
    schema: SchemaRef,
    partition: Option<crate::PartitionValues>,
    partition_column_names: Arc<[String]>,
    sort_column_names: Arc<[String]>,
    target_rows_per_group: usize,
    max_file_size: usize,
) -> OperatorSpec<AssembledFile, impl OperatorFactory<AssembledFile> + 'static> {
    let spec = unshred_batches_spec(spec);
    let order_by = sort_order(&schema, &sort_column_names);
    let (dispatcher, heads) = spec.into_parts();
    let worker_count = heads.len();
    let file_collector_worker = dispatcher.next_worker();
    let input_batches = OperatorSpec::new(dispatcher, heads.into_iter().collect::<Vec<_>>());
    let topology = input_batches.dispatcher().topology();
    let presorted_runs =
        PresortedRun::create_for_workers(partition.as_ref(), &partition_column_names, topology);
    let files = input_batches
        .chain(
            stealable::<RecordBatch>(topology).into_iter().collect(),
            presorted_runs,
        )
        .chain(
            to_single_worker_mpsc::<SortedPartitionRun>(worker_count, file_collector_worker)
                .into_iter()
                .collect(),
            file_collector::factories(
                partition_column_names,
                order_by.into(),
                target_rows_per_group,
                usize::MAX,
                Some(max_file_size),
                topology,
            ),
        );
    encode_files_spec(files)
}

pub fn unshred_batches_spec(spec: RecordBatchOperatorSpec) -> RecordBatchOperatorSpec {
    spec.project(|| {
        // Built on the first batch rather than here: `SlabAllocator::new` takes
        // a write buffer from the worker's memory context, which exists only
        // once the worker is running. One allocator per worker, reused across
        // batches so its part-filled slab carries over.
        let mut allocator: Option<SlabAllocator> = None;
        move |batch| {
            let allocator = allocator.get_or_insert_with(|| SlabAllocator::new(false));
            Ok(shredding::unshred_batch(batch, allocator)?)
        }
    })
}

fn sort_order(schema: &SchemaRef, sort_column_names: &[String]) -> Vec<OrderBy> {
    sort_column_names
        .iter()
        .map(|name| {
            let column = schema
                .index_of(name)
                .expect("sort columns name declared columns");
            OrderBy::new(column, false, true)
        })
        .collect()
}

/// Shared half of INSERT and compaction: merge the already-sorted runs for each
/// output file, divide the result into row groups, encode its columns, and
/// assemble the Parquet bytes.
fn encode_files_spec<OF>(
    files: OperatorSpec<FileOrderInput, OF>,
) -> OperatorSpec<AssembledFile, impl OperatorFactory<AssembledFile> + 'static>
where
    OF: OperatorFactory<FileOrderInput> + 'static,
{
    let worker_count = files.dispatcher().worker_count();
    let topology = files.dispatcher().topology();
    files
        .chain(
            node_work_queue::<FileOrderInput>(topology)
                .into_iter()
                .collect(),
            DefaultUnaryFactory::<file_merge::LocalMergePlanner>::create_for_workers(worker_count),
        )
        .chain(
            node_work_queue::<LocalMergeJob>(topology)
                .into_iter()
                .collect(),
            DefaultUnaryFactory::<file_merge::LocalMergeExecutor>::create_for_workers(worker_count),
        )
        .chain(
            shared_work_queue::<LocalMergeResult>(worker_count)
                .into_iter()
                .collect(),
            DefaultUnaryFactory::<file_merge::GlobalMergePlanner>::create_for_workers(worker_count),
        )
        .chain(
            node_work_queue::<GlobalMergeJob>(topology)
                .into_iter()
                .collect(),
            DefaultUnaryFactory::<file_merge::GlobalMergeExecutor>::create_for_workers(
                worker_count,
            ),
        )
        .chain(
            node_work_queue::<ReadyFile>(topology).into_iter().collect(),
            row_group_planner::factories(topology),
        )
        .chain(
            node_work_queue::<ColumnChunkJob>(topology)
                .into_iter()
                .collect(),
            shredder::factories(worker_count),
        )
        .chain(
            node_work_queue::<LeafChunkJob>(topology)
                .into_iter()
                .collect(),
            encoder::factories(worker_count),
        )
        .chain(
            return_to_worker_mpsc::<EncodedLeafChunk>(worker_count)
                .into_iter()
                .collect(),
            assembler::factories(worker_count),
        )
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod type_tests;
