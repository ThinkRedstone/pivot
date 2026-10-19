//! Accumulating any column the other strategies do not cover, by keeping the
//! appended rows as zero-copy slices and concatenating them on emit.
//!
//! This is what makes the accumulator total: booleans, offset strings,
//! dictionaries, lists and anything else the engine produces still work, they
//! just go through Arrow's `concat` instead of a slab. The other strategies are
//! the fast paths.
//!
//! Two consequences follow from holding slices, and a caller that cares about
//! either should know which of its columns land here. The rows are held as
//! references to the batches they came from, so an accumulation keeps every
//! input alive for these columns however few rows it took from it; and the
//! concatenation allocates through Arrow's own allocator, so their values leave
//! the ring.

use arrow::compute::kernels::concat::concat;
use arrow_array::{Array, ArrayRef};
use arrow_buffer::BooleanBuffer;
use arrow_schema::ArrowError;

use arrow_array::make_array;

use super::column::{ChunkedColumn, ColumnAccumulator, EncodedRowIds, split_encoded_id};
use crate::memory::SlabAllocator;

/// A column held as the arrays it was appended from, concatenated on emit.
pub(super) struct ConcatenatedColumn {
    arrays: Vec<ArrayRef>,
}

impl ConcatenatedColumn {
    pub(super) fn new() -> Self {
        Self { arrays: Vec::new() }
    }
}

impl ConcatenatedColumn {
    fn append_ids<I: Copy + Into<u64>>(&mut self, column: &ChunkedColumn, ids: &[I], shift: u32) {
        // Consecutive same-batch rows become one slice, as in the indexed
        // append.
        let mut push = |batch: usize, start: usize, len: usize| {
            self.arrays
                .push(make_array(column.data[batch].slice(start, len)));
        };
        let mut run: Option<(usize, usize, usize)> = None;
        for &id in ids {
            let (batch, row) = split_encoded_id(id, shift);
            match run {
                Some((run_batch, start, end)) if run_batch == batch && row == end => {
                    run = Some((run_batch, start, end + 1));
                }
                Some((run_batch, start, end)) => {
                    push(run_batch, start, end - start);
                    run = Some((batch, row, row + 1));
                }
                None => run = Some((batch, row, row + 1)),
            }
        }
        if let Some((run_batch, start, end)) = run {
            push(run_batch, start, end - start);
        }
    }
}

impl ColumnAccumulator for ConcatenatedColumn {
    fn append_from_indices(
        &mut self,
        column: &ArrayRef,
        indices: &[u32],
        _destination_start: usize,
        _allocator: &mut SlabAllocator,
    ) {
        // Consecutive positions become one slice, so a selection that keeps a
        // run of rows costs one array rather than one per row.
        let mut run: Option<(usize, usize)> = None;
        for &row in indices {
            let row = row as usize;
            match run {
                Some((start, end)) if row == end => run = Some((start, end + 1)),
                Some((start, end)) => {
                    self.arrays.push(column.slice(start, end - start));
                    run = Some((row, row + 1));
                }
                None => run = Some((row, row + 1)),
            }
        }
        if let Some((start, end)) = run {
            self.arrays.push(column.slice(start, end - start));
        }
    }

    fn append_from_range(
        &mut self,
        column: &ArrayRef,
        start: usize,
        len: usize,
        _destination_start: usize,
        _allocator: &mut SlabAllocator,
    ) {
        self.arrays.push(column.slice(start, len));
    }

    fn append_from_mask(
        &mut self,
        column: &ArrayRef,
        mask: &BooleanBuffer,
        _selected_count: usize,
        _destination_start: usize,
        _allocator: &mut SlabAllocator,
    ) {
        for (start, end) in mask.set_slices() {
            self.arrays.push(column.slice(start, end - start));
        }
    }

    fn append_from_batches(
        &mut self,
        column: &ChunkedColumn,
        ids: EncodedRowIds<'_>,
        shift: u32,
        _destination_start: usize,
        _allocator: &mut SlabAllocator,
    ) {
        match ids {
            EncodedRowIds::Narrow(ids) => self.append_ids(column, ids, shift),
            EncodedRowIds::Wide(ids) => self.append_ids(column, ids, shift),
        }
    }

    fn take_array(
        &mut self,
        _len: usize,
        _allocator: &mut SlabAllocator,
    ) -> Result<ArrayRef, ArrowError> {
        let refs: Vec<&dyn Array> = self.arrays.iter().map(|array| array.as_ref()).collect();
        let out = concat(&refs)?;
        self.arrays.clear();
        Ok(out)
    }
}
