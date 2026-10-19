//! Accumulating an Arrow struct column.
//!
//! A struct holds no values of its own: its children do, and it contributes
//! only the validity saying which rows have a struct at all. So this
//! accumulates through one child accumulator per field, exactly as Parquet
//! stores a group, and a child gets whichever strategy its own type calls for.
//!
//! Slicing a struct slices its children with it, so a child's rows are at the
//! same positions as the struct's own and the same indices or range go straight
//! down.

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, StructArray};
use arrow_buffer::BooleanBuffer;
use arrow_schema::{ArrowError, Fields};

use super::column::{ChunkedColumn, ColumnAccumulator, EncodedRowIds};
use super::create_column_accumulator;
use super::validity::ValidityMask;
use crate::memory::SlabAllocator;

/// A struct column, accumulated as its children plus its own validity.
pub(super) struct StructColumn {
    fields: Fields,
    children: Vec<Box<dyn ColumnAccumulator>>,
    validity: ValidityMask,
}

impl StructColumn {
    pub(super) fn new(fields: &Fields, capacity: usize, allocator: &mut SlabAllocator) -> Self {
        Self {
            fields: fields.clone(),
            children: fields
                .iter()
                .map(|field| create_column_accumulator(field.data_type(), capacity, allocator))
                .collect(),
            validity: ValidityMask::new(capacity),
        }
    }
}

impl ColumnAccumulator for StructColumn {
    fn append_from_indices(
        &mut self,
        column: &ArrayRef,
        indices: &[u32],
        destination_start: usize,
        allocator: &mut SlabAllocator,
    ) {
        let column = column.as_struct();
        self.validity
            .append_indices(column.nulls(), indices, destination_start);
        for (child, values) in self.children.iter_mut().zip(column.columns()) {
            child.append_from_indices(values, indices, destination_start, allocator);
        }
    }

    fn append_from_range(
        &mut self,
        column: &ArrayRef,
        start: usize,
        len: usize,
        destination_start: usize,
        allocator: &mut SlabAllocator,
    ) {
        let column = column.as_struct();
        self.validity
            .append_range(column.nulls(), start, len, destination_start);
        for (child, values) in self.children.iter_mut().zip(column.columns()) {
            child.append_from_range(values, start, len, destination_start, allocator);
        }
    }

    fn append_from_mask(
        &mut self,
        column: &ArrayRef,
        mask: &BooleanBuffer,
        selected_count: usize,
        destination_start: usize,
        allocator: &mut SlabAllocator,
    ) {
        let column = column.as_struct();
        self.validity
            .append_mask(column.nulls(), mask, destination_start);
        for (child, values) in self.children.iter_mut().zip(column.columns()) {
            child.append_from_mask(values, mask, selected_count, destination_start, allocator);
        }
    }

    fn append_from_batches(
        &mut self,
        column: &ChunkedColumn,
        ids: EncodedRowIds<'_>,
        shift: u32,
        destination_start: usize,
        allocator: &mut SlabAllocator,
    ) {
        let batch_nulls = |batch: usize| {
            column.data[batch]
                .nulls()
                .filter(|nulls| nulls.null_count() > 0)
        };
        match ids {
            EncodedRowIds::Narrow(ids) => {
                self.validity
                    .append_by_ids(ids, shift, destination_start, batch_nulls)
            }
            EncodedRowIds::Wide(ids) => {
                self.validity
                    .append_by_ids(ids, shift, destination_start, batch_nulls)
            }
        }
        // Each child gets its own prepared column. Materializing it here clones
        // one Arc-backed ArrayData per batch per append, which is fine for the
        // rare struct-typed column.
        for (k, child) in self.children.iter_mut().enumerate() {
            let child_column = ChunkedColumn::new(
                column
                    .data
                    .iter()
                    .map(|data| data.child_data()[k].clone())
                    .collect(),
            );
            child.append_from_batches(&child_column, ids, shift, destination_start, allocator);
        }
    }

    fn take_array(
        &mut self,
        len: usize,
        allocator: &mut SlabAllocator,
    ) -> Result<ArrayRef, ArrowError> {
        let nulls = self.validity.take(len);
        let columns = self
            .children
            .iter_mut()
            .map(|child| child.take_array(len, allocator))
            .collect::<Result<Vec<_>, _>>()?;
        if columns.is_empty() {
            // A struct with no fields has no child to take its length from.
            return Ok(Arc::new(StructArray::new_empty_fields(len, nulls)));
        }
        Ok(Arc::new(StructArray::new(
            self.fields.clone(),
            columns,
            nulls,
        )))
    }
}
