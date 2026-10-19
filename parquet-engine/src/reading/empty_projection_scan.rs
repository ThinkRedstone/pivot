//! Scan for a projection with no data columns.
//!
//! The page-driven read pipeline is column-driven: with zero projected columns
//! it fetches nothing, produces no pages, and emits nothing. This source instead
//! emits one batch per row group straight from each row group's `num_rows` — no
//! IO, no decode — chunked to [`RECORD_BATCH_SIZE`] so a downstream `LIMIT` can
//! reject most of them after the first chunk.
//!
//! Row-group/row-index metadata columns are appended only when requested: a
//! late-materialized plain `LIMIT` (whose narrow scan has no sort/filter key)
//! needs them for the downstream `Materialize`; any other zero-column scan (a
//! row count) gets plain empty-schema rows.

use std::sync::Arc;

use arrow_array::{RecordBatch, RecordBatchOptions, UInt32Array};
use arrow_schema::Schema;
use crossbeam_deque::{Injector, Steal};
use dispatch::{
    DataFlowDispatcher, Nullary, NullaryFactory, NullaryResult, RECORD_BATCH_SIZE,
    RecordBatchOperatorSpec, Sender, WorkStatus,
};

use crate::RowGroupFilter;
use crate::reading::record_batch_metadata::{with_row_group_metadata, with_row_group_positions};
use crate::types::metadata::{QueryRowGroupMetadata, RowSelection};
use crate::types::table::ParquetTable;

/// Build a scan that emits the rows of an empty (zero-data-column) projection.
pub(crate) fn empty_projection_scan(
    dispatcher: &DataFlowDispatcher,
    table: &Arc<ParquetTable>,
    filter: Option<RowGroupFilter>,
    add_row_group_metadata: bool,
) -> RecordBatchOperatorSpec {
    let n = dispatcher.worker_count();
    let row_groups = Arc::new(Injector::new());
    for idx in 0..table.row_groups.len() {
        row_groups.push(QueryRowGroupMetadata::new(table, idx, RowSelection::All));
    }
    let factories = (0..n).map(|_| EmptyProjectionScanFactory {
        row_groups: row_groups.clone(),
        filter: filter.clone(),
        add_row_group_metadata,
    });
    RecordBatchOperatorSpec::from_nullary(dispatcher, factories)
}

/// Per-worker factory for [`EmptyProjectionScan`]. Workers share one
/// work-stealing queue of row groups, so each row group is emitted exactly once.
struct EmptyProjectionScanFactory {
    row_groups: Arc<Injector<QueryRowGroupMetadata>>,
    filter: Option<RowGroupFilter>,
    add_row_group_metadata: bool,
}

impl NullaryFactory<RecordBatch> for EmptyProjectionScanFactory {
    type Nullary = EmptyProjectionScan;
    fn build_nullary(self) -> EmptyProjectionScan {
        EmptyProjectionScan {
            row_groups: self.row_groups,
            filter: self.filter,
            add_row_group_metadata: self.add_row_group_metadata,
        }
    }
}

struct EmptyProjectionScan {
    row_groups: Arc<Injector<QueryRowGroupMetadata>>,
    filter: Option<RowGroupFilter>,
    add_row_group_metadata: bool,
}

impl Nullary<RecordBatch> for EmptyProjectionScan {
    fn run(
        &mut self,
        sender: &mut dyn Sender<RecordBatch>,
        _io: &mut dispatch::OperatorIO,
    ) -> NullaryResult<WorkStatus> {
        let rg = match self.row_groups.steal() {
            Steal::Success(rg) => rg,
            Steal::Empty | Steal::Retry => return Ok(WorkStatus::Pending),
        };
        // A pruned row group contributes no rows; skip without emitting.
        if let Some(filter) = &self.filter
            && !filter(rg.get_metadata())
        {
            return Ok(WorkStatus::Ran);
        }
        // Zero data columns + an explicit row count; never fails.
        let empty_rows = |rows: usize| {
            RecordBatch::try_new_with_options(
                Arc::new(Schema::empty()),
                vec![],
                &RecordBatchOptions::new().with_row_count(Some(rows)),
            )
            .expect("empty batch")
        };
        let metadata = rg.get_metadata();
        match (&metadata.deleted_rows, self.add_row_group_metadata) {
            // The surviving rows by position, since the deleted ones leave gaps.
            (Some(_), true) => {
                let positions: Vec<u32> = (0..metadata.num_rows as u32)
                    .filter(|row| !metadata.is_deleted(*row))
                    .collect();
                for chunk in positions.chunks(RECORD_BATCH_SIZE) {
                    let batch = with_row_group_positions(
                        empty_rows(chunk.len()),
                        rg.index(),
                        UInt32Array::from(chunk.to_vec()),
                    );
                    sender.send(batch)?;
                }
            }
            (deleted, add_row_group_metadata) => {
                let total = metadata.live_rows() as usize;
                let mut offset = 0;
                while offset < total {
                    let chunk = (total - offset).min(RECORD_BATCH_SIZE);
                    let empty = empty_rows(chunk);
                    let batch = if add_row_group_metadata {
                        debug_assert!(deleted.is_none(), "deleted rows leave no consecutive run");
                        with_row_group_metadata(empty, rg.index(), offset)
                    } else {
                        empty
                    };
                    sender.send(batch)?;
                    offset += chunk;
                }
            }
        }
        Ok(WorkStatus::Ran)
    }

    fn finish(&mut self, _sender: &mut dyn Sender<RecordBatch>) -> NullaryResult<bool> {
        Ok(self.row_groups.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reading::record_batch_metadata::row_index;
    use crate::test_utils::{deleted_rows, dummy_row_group};
    use crate::types::metadata::RowGroupMetadata;
    use dispatch::Dispatch;

    /// A table of row groups holding `rows` rows each, with the `deleted`
    /// rows of each marked deleted.
    fn table_of_row_groups(rows: &[i64], deleted: &[&[u32]]) -> Arc<ParquetTable> {
        let dummy = dummy_row_group();
        let row_groups = rows
            .iter()
            .zip(deleted)
            .enumerate()
            .map(|(index, (rows, deleted))| {
                Arc::new(RowGroupMetadata {
                    num_rows: *rows,
                    deleted_rows: deleted_rows(0, deleted),
                    file_row_group_idx: index,
                    ..(*dummy).clone()
                })
            })
            .collect();
        Arc::new(ParquetTable::new(row_groups))
    }

    #[test]
    fn deleted_rows_are_left_out_of_the_count() {
        let dispatch = Dispatch::spin_up(1, 32, None);
        let table = table_of_row_groups(&[5, 4], &[&[1, 3], &[]]);

        let batches = empty_projection_scan(dispatch.dispatcher(), &table, None, false)
            .collect()
            .unwrap();

        let rows: usize = batches.iter().map(|batch| batch.num_rows()).sum();
        assert_eq!(rows, 7);
    }

    #[test]
    fn deleted_rows_leave_gaps_in_the_row_indices() {
        let dispatch = Dispatch::spin_up(1, 32, None);
        let table = table_of_row_groups(&[5], &[&[1, 3]]);

        let batches = empty_projection_scan(dispatch.dispatcher(), &table, None, true)
            .collect()
            .unwrap();

        let positions: Vec<u32> = batches
            .iter()
            .flat_map(|batch| row_index(batch).values().to_vec())
            .collect();
        assert_eq!(positions, vec![0, 2, 4]);
    }
}
