//! Parallel **footer metadata** fetch — the table-load pipeline, run once at
//! `CREATE`/`ATTACH` (the reading pipeline scans the row groups it produces).
//! A small work-stealing dataflow with one module per stage, carrying one
//! [`FileRowGroups`] (a file's [`FileRef`] plus the row groups read from its
//! footer) per file end to end. This layer is Parquet-specific and knows nothing
//! of any table log: [`load_file_row_groups`] returns one result per input file,
//! in input order, so the owning table format pairs each with its file entry by
//! position.
//!
//! - [`FileInjectorFactory`] (from `object_storage::file_injector`) — the source: hands
//!   out the input files.
//! - [`fetcher`] — reads each file's footer (over the io_uring ring, through the
//!   compressed cache) on whatever worker steals it, emitting one [`FileRowGroups`].
//! - [`writer`] — the terminal fan-in sink: gathers the [`FileRowGroups`] on one
//!   worker and hands them to the transaction-staging closure.
//!
//! It is consumed two ways, both over the same fetch core
//! ([`fetch_file_row_group_factories`]): [`load_file_row_groups`] collects the
//! [`FileRowGroups`] on the coordinator and returns them as a **value** (the
//! `ParquetTable::from_*` constructors, which flatten them into a table), while
//! [`create_load_and_stage_spec`] returns a `RecordBatchOperatorSpec` ending
//! in the [`writer`] sink that hands the `Vec<FileRowGroups>` to a staging closure
//! for the `CREATE TABLE` transaction.

mod fetcher;
mod writer;

use crate::types::columns::TableColumns;
use crate::types::metadata::RowGroupMetadata;
use dispatch::{
    DataFlowDispatcher, OperatorSpec, RecordBatchOperatorSpec, RootUnaryOperatorFactory,
    UnaryFactory, fan_in,
};
use fetcher::FileRowGroupsFetcher;
use object_storage::file_injector::FileInjectorFactory;
use object_storage::{DataFile, DataFileLocation, FileRef, ObjectPath};
use roaring::RoaringTreemap;
use std::collections::HashMap;
use std::ops::{Range, RangeInclusive};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use writer::FileRowGroupsSinkFactory;

/// A file's footer metadata: its store identity ([`FileRef`]) and the row groups
/// read from its footer, in file-local order. The Parquet-level result of the
/// metadata-fetch pipeline; a table format can join it with its own file entry.
pub struct FileRowGroups {
    pub file: FileRef,
    pub row_groups: Vec<Arc<RowGroupMetadata>>,
}

impl FileRowGroups {
    /// Record the rows the file's row-level deletes removed, `deleted` by
    /// position in the file, on the row groups that may hold them, so a scan
    /// drops them. Each shares the bitmap as decoded, so no position is
    /// visited. Only for row groups fresh from the footer read, which nothing
    /// shares yet.
    pub fn mark_deleted_rows(&mut self, deleted: RoaringTreemap) {
        let deleted = Arc::new(deleted);
        for row_group in &mut self.row_groups {
            if deleted.range_cardinality(containers_covering(&row_group.file_positions())) == 0 {
                continue;
            }
            Arc::get_mut(row_group)
                .expect("row groups fresh from the footer read are not shared")
                .deleted_rows = Some(deleted.clone());
        }
    }
}

/// The range to count the bitmap over to learn whether any of `positions`
/// MIGHT be deleted: a roaring bitmap keeps a container per block of
/// `1 << 16` positions, each with its cardinality, and a count whose bounds
/// sit on container boundaries sums those cardinalities and reads no bits,
/// where a bound inside a container has its bits up to the bound counted.
/// The containers at the edges are shared with the neighbouring row groups,
/// whose deleted rows count too, so only a zero is exact; a row group handed
/// deleted rows may have none. The range ends on the first position of the
/// container past the last, inclusive, as the bitmap takes an exclusive end
/// as the last position inside a container.
fn containers_covering(positions: &Range<u64>) -> RangeInclusive<u64> {
    const CONTAINER_POSITIONS: u64 = 1 << 16;
    let first = positions.start / CONTAINER_POSITIONS * CONTAINER_POSITIONS;
    let past_last = positions.end.div_ceil(CONTAINER_POSITIONS) * CONTAINER_POSITIONS;
    first..=past_last
}

/// Builds each worker's [`FileRowGroupsFetcher`], carrying the table's
/// declared columns so every parsed footer is reconciled with them.
struct MetadataFetcherFactory {
    table_columns: TableColumns,
}

impl UnaryFactory<DataFile, FileRowGroups> for MetadataFetcherFactory {
    type Unary = FileRowGroupsFetcher;

    fn build_unary(self) -> Self::Unary {
        FileRowGroupsFetcher::new(self.table_columns)
    }
}

/// Build the scan metadata for a file a worker has just uploaded, from the footer
/// metadata the writer already produced. This skips both scheduling a nested
/// metadata dataflow from a worker and re-parsing a footer we just wrote; only
/// the open_file has to be bound now, since it names the stored file that future
/// scans read and exists only once the upload has landed.
pub fn file_row_groups_from_metadata(
    file: FileRef,
    source: DataFileLocation,
    metadata: crate::thrift::footer::FileMetaData,
    table_columns: &TableColumns,
) -> Result<FileRowGroups, crate::ParquetTableError> {
    let open_file = source
        .open_read(file.size)
        .map_err(crate::ParquetTableError::IO)?;
    let row_groups =
        crate::types::table::row_groups_from_metadata(metadata, open_file, table_columns)?
            .into_iter()
            .map(Arc::new)
            .collect();
    Ok(FileRowGroups { file, row_groups })
}

/// The per-worker source→fetch factories: each worker steals files from a shared
/// injector and reads their footers, emitting one [`FileRowGroups`] per file (the
/// file's [`FileRef`] rides along on the [`DataFile`] and lands on the result).
fn fetch_file_row_group_factories(
    files: &[DataFile],
    workers: usize,
    table_columns: TableColumns,
) -> Vec<
    RootUnaryOperatorFactory<DataFile, FileRowGroups, MetadataFetcherFactory, FileInjectorFactory>,
> {
    let injector = FileInjectorFactory::new(files);
    let siblings = Arc::new(AtomicUsize::new(workers));
    (0..workers)
        .map(|_| {
            RootUnaryOperatorFactory::new(
                MetadataFetcherFactory {
                    table_columns: table_columns.clone(),
                },
                injector.clone(),
                siblings.clone(),
            )
        })
        .collect()
}

/// Read every file's footer in parallel and collect the resulting
/// [`FileRowGroups`] on the coordinator: one per file, in the order of `files`,
/// whose paths must be distinct. The fetch stage already emits `FileRowGroups`, so the dataflow's typed
/// `collect` drains them directly — no terminal sink. Drives the dataflow, so it
/// must run on the **coordinator**, not inside a `run_on_worker` closure.
pub fn load_file_row_groups(
    dispatcher: &DataFlowDispatcher,
    files: &[DataFile],
    table_columns: TableColumns,
) -> Result<Vec<FileRowGroups>, dispatch::DataFlowError> {
    // When there is nothing to fetch, skip the dataflow round-trip entirely.
    // Every query's compile resolves its table through here, and a warm
    // catalog has no missing footers, so this is the common case.
    if files.is_empty() {
        return Ok(Vec::new());
    }
    let loaded: Vec<FileRowGroups> = OperatorSpec::new(
        dispatcher.clone(),
        fetch_file_row_group_factories(files, dispatcher.worker_count(), table_columns),
    )
    .collect()?;
    // Workers finish in any order: hand the footers back in the order of `files`.
    let mut footers: HashMap<ObjectPath, FileRowGroups> = loaded
        .into_iter()
        .map(|footer| (footer.file.path.clone(), footer))
        .collect();
    Ok(files
        .iter()
        .map(|file| {
            footers
                .remove(&file.file.path)
                .expect("every distinct file's footer is read once")
        })
        .collect())
}

/// A `RecordBatchOperatorSpec` that, when executed, reads every file's footer in
/// parallel and — at its terminal stage — regroups the [`FileRowGroups`] and
/// hands them to `stage` once on the terminal worker, in whichever order the
/// workers finished. Emits no rows. The stage
/// closure must only enqueue the loaded metadata; durable table creation belongs
/// to the transaction's commit path, outside the dispatch pool.
pub fn create_load_and_stage_spec<C>(
    dispatcher: &DataFlowDispatcher,
    files: &[DataFile],
    table_columns: TableColumns,
    stage: C,
) -> RecordBatchOperatorSpec
where
    C: FnOnce(Vec<FileRowGroups>) + Send + 'static,
{
    let fetch = OperatorSpec::new(
        dispatcher.clone(),
        fetch_file_row_group_factories(files, dispatcher.worker_count(), table_columns),
    );

    // One worker receives every row group and the staging closure; the channel
    // closing is the "all fetched" signal. Rotate that role across creations.
    let worker_count = dispatcher.worker_count();
    let target = dispatcher.next_worker();
    let mut stage = Some(stage);
    let sinks: Vec<_> = (0..worker_count)
        .map(|worker| {
            FileRowGroupsSinkFactory::new(if worker == target { stage.take() } else { None })
        })
        .collect();
    RecordBatchOperatorSpec::from_spec(
        fetch.chain(fan_in::<FileRowGroups>(worker_count, target), sinks),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::dummy_row_group;

    /// A file of row groups holding `rows` rows each.
    fn file_of_row_groups(rows: &[i64]) -> FileRowGroups {
        let dummy = dummy_row_group();
        let mut file_row_offset = 0;
        FileRowGroups {
            file: FileRef {
                path: ObjectPath::new("data.parquet"),
                size: 0,
            },
            row_groups: rows
                .iter()
                .enumerate()
                .map(|(index, rows)| {
                    let row_group = Arc::new(RowGroupMetadata {
                        num_rows: *rows,
                        file_row_offset,
                        file_row_group_idx: index,
                        ..(*dummy).clone()
                    });
                    file_row_offset += *rows as u64;
                    row_group
                })
                .collect(),
        }
    }

    /// The deleted rows of each of `file`'s row groups, by position in the
    /// row group; `None` for a row group handed none.
    fn deleted_per_row_group(file: &FileRowGroups) -> Vec<Option<Vec<u32>>> {
        file.row_groups
            .iter()
            .map(|row_group| {
                row_group.deleted_rows.as_ref().map(|_| {
                    row_group
                        .deleted_rows_in(0..row_group.num_rows as u32)
                        .collect()
                })
            })
            .collect()
    }

    #[test]
    fn each_row_group_sees_the_deleted_rows_among_its_own() {
        let container = 1 << 16;
        let mut file = file_of_row_groups(&[container, container, container]);

        file.mark_deleted_rows([1, 2 * container as u64 + 3].into_iter().collect());

        assert_eq!(
            deleted_per_row_group(&file),
            vec![Some(vec![1]), None, Some(vec![3])]
        );
    }

    #[test]
    fn a_row_group_sharing_a_container_with_a_deleted_row_is_handed_none_of_its_own() {
        let container = 1 << 16;
        let mut file = file_of_row_groups(&[container + 10, container]);

        file.mark_deleted_rows([container as u64 + 5].into_iter().collect());

        assert_eq!(
            deleted_per_row_group(&file),
            vec![Some(vec![container as u32 + 5]), Some(vec![])]
        );
    }
}
