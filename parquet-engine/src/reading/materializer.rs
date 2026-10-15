//! Materializes further projections later on in the dataflow.
//!
//! After upstream operators evaluate predicates on lightweight index columns,
//! the surviving rows arrive here as `RecordBatch`es that still carry metadata
//! columns (global row-group ID and per-row index). The [`Materializer`]
//! accumulates those indices, groups them by row group, and on `finish` emits
//! one [`QueryRowGroupMetadata`] per row group with sorted indices so that the
//! downstream Parquet reader can fetch only the rows that passed a filter,
//! reading them sequentially within each row group for optimal IO.

use crate::ParquetTable;
use crate::reading::record_batch_metadata::{global_row_group, row_index};
use crate::types::metadata::{QueryRowGroupMetadata, RowSelection};
use ahash::HashMap;
use arrow_array::{Array, RecordBatch, UInt32Array};
use dispatch::Sender;
use dispatch::{Unary, UnaryFactory};
use std::mem;
use std::sync::Arc;

/// Factory for creating [`Materializer`] instances within the unary pipeline.
///
/// Captures the [`ParquetTable`] handle that the materializer will later
/// reference when building each row group's [`QueryRowGroupMetadata`].
/// Consumed once via [`UnaryFactory::build_unary`].
pub struct MaterializerFactory {
    table: Arc<ParquetTable>,
}

impl MaterializerFactory {
    /// Creates a new factory.
    ///
    /// * `table` – shared handle to the Parquet table metadata.
    pub fn new(table: Arc<ParquetTable>) -> Self {
        Self { table }
    }
}

impl UnaryFactory<RecordBatch, QueryRowGroupMetadata> for MaterializerFactory {
    type Unary = Materializer;

    fn build_unary(self) -> Materializer {
        Materializer::new(self.table)
    }
}

/// Collects filtered row indices from upstream `RecordBatch`es and emits one
/// [`QueryRowGroupMetadata`] per row group for the fetcher to read the matching
/// rows from disk.
///
/// During `consume`, incoming batches carry metadata columns (global row group
/// ID and per-row index). The materializer groups row indices by their row
/// group. On `finish`, it emits one row group with sorted indices, allowing
/// downstream IO to read only the needed rows.
pub struct Materializer {
    /// Row group ID -> collected row indices within that group.
    pending_row_groups: HashMap<u32, Vec<u32>>,
    table: Arc<ParquetTable>,
}

impl Materializer {
    /// Creates a new materializer.
    ///
    /// * `table` – shared Parquet table metadata used to build each row
    ///   group's metadata.
    pub fn new(table: Arc<ParquetTable>) -> Self {
        Self {
            pending_row_groups: HashMap::default(),
            table,
        }
    }
}

impl Unary<RecordBatch, QueryRowGroupMetadata> for Materializer {
    /// Extracts row group IDs and row indices from the batch metadata columns,
    /// accumulating them into `pending_row_groups` keyed by row group.
    fn consume(
        &mut self,
        batch: RecordBatch,
        _sender: &mut dyn Sender<QueryRowGroupMetadata>,
        _io: &mut dispatch::OperatorIO,
    ) -> dispatch::UnaryResult<()> {
        // The row-group column is a `RunArray` (consecutive same-group rows = one
        // run). The batch may be a logical slice (e.g. a `LIMIT` above the scan),
        // so walk the *logical* runs via `RunEndBuffer::sliced_values()` — run
        // ends already adjusted by the slice offset and capped at the slice
        // length — and map each run to its physical group value from
        // `get_start_physical_index()`. Reading the raw (physical) `run_ends`
        // would treat them as logical bounds and overrun the (shorter) sliced
        // `row_indices`.
        let groups = global_row_group(&batch);
        let run_ends = groups.run_ends();
        let physical_start = run_ends.get_start_physical_index();
        let group_values = groups
            .values()
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let row_indices = row_index(&batch);

        let mut logical = 0usize;
        for (run_offset, logical_end) in run_ends.sliced_values().enumerate() {
            let logical_end = logical_end as usize;
            let group = group_values.value(physical_start + run_offset);
            let entries = self.pending_row_groups.entry(group).or_default();
            // `row_indices` is logically indexed, so `value(logical)` accounts
            // for any slice offset.
            while logical < logical_end {
                entries.push(row_indices.value(logical));
                logical += 1;
            }
        }
        Ok(())
    }

    /// Drains accumulated row groups and sends each one. Indices are sorted so
    /// downstream IO can read them sequentially.
    fn finish(
        &mut self,
        sender: &mut dyn Sender<QueryRowGroupMetadata>,
    ) -> dispatch::UnaryResult<bool> {
        let pending_row_groups = mem::take(&mut self.pending_row_groups);
        for (group, mut indices) in pending_row_groups {
            indices.sort_unstable();
            sender.send(QueryRowGroupMetadata::new(
                &self.table,
                group as usize,
                RowSelection::Indices(indices.into()),
            ))?;
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::Materializer;
    use crate::ParquetTable;
    use crate::reading::record_batch_metadata::with_row_group_metadata;
    use crate::types::metadata::QueryRowGroupMetadata;
    use ahash::HashMap;
    use arrow::compute::concat_batches;
    use arrow_array::{RecordBatch, RecordBatchOptions};
    use arrow_schema::Schema;
    use dispatch::test_utils::CollectSender;
    use std::sync::Arc;

    /// A metadata-only batch (no data columns) tagging `rows` rows with row group
    /// `group` and row indices `0..rows` — the shape a metadata scan emits.
    fn meta_batch(group: usize, rows: usize) -> RecordBatch {
        let empty = RecordBatch::try_new_with_options(
            Arc::new(Schema::empty()),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(rows)),
        )
        .unwrap();
        with_row_group_metadata(empty, group, 0)
    }

    /// Run `batch` through a `Materializer`'s `consume` and return the row indices
    /// it grouped by row group. The table is unused by `consume` (only `finish`
    /// reads it), so a dummy empty one suffices.
    fn consume(batch: RecordBatch) -> HashMap<u32, Vec<u32>> {
        let mut materializer = Materializer::new(Arc::new(ParquetTable::new(vec![])));
        let mut sink = CollectSender::<QueryRowGroupMetadata>::default();
        dispatch::Unary::consume(
            &mut materializer,
            batch,
            &mut sink,
            &mut dispatch::TestOperatorIO::default().io(),
        )
        .unwrap();
        materializer.pending_row_groups
    }

    // Two row groups concatenated → a 2-run RunArray; the materializer recovers
    // every (group, row_idx) pair.
    #[test]
    fn collects_indices_grouped_by_row_group() {
        let a = meta_batch(5, 6);
        let batch = concat_batches(&a.schema(), &[a.clone(), meta_batch(8, 6)]).unwrap();

        let out = consume(batch);

        assert_eq!(out[&5], vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(out[&8], vec![0, 1, 2, 3, 4, 5]);
    }

    // Regression: a logically *sliced* metadata batch (what a LIMIT above the scan
    // produces) must be decoded by logical run bounds, not the unchanged physical
    // run_ends — the latter overran the shorter sliced row_idx. The slice spans
    // into the second run, so it also exercises `get_start_physical_index() > 0`.
    #[test]
    fn collects_indices_from_sliced_batch() {
        let a = meta_batch(5, 6);
        let batch = concat_batches(&a.schema(), &[a.clone(), meta_batch(8, 6)]).unwrap();

        let sliced = batch.slice(8, 3); // logical rows 8,9,10 → group 8, row_idx 2,3,4
        let out = consume(sliced);

        assert_eq!(out.len(), 1);
        assert_eq!(out[&8], vec![2, 3, 4]);
    }
}
