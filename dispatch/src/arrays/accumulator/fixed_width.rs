//! Accumulating a column whose values are all the same number of bytes.
//!
//! The simplest case: the values are the array's one buffer, so appending is a
//! copy of raw bytes and emitting is that buffer with the accumulated validity
//! beside it. Nothing here reads a value, so one implementation covers every
//! width Arrow packs this way, from a boolean-free 1-byte type up to a 16-byte
//! decimal.

use arrow::array::ArrayData;
use arrow_array::{ArrayRef, make_array};
use arrow_buffer::BooleanBuffer;
use arrow_schema::{ArrowError, DataType};

use super::column::{ChunkedColumn, ColumnAccumulator, EncodedRowIds, split_encoded_id};
use super::validity::ValidityMask;
use crate::memory::{SlabAllocator, SlabBuffer};

pub(super) struct FixedWidthColumn {
    data_type: DataType,
    width: usize,
    capacity: usize,
    /// The values, u128-backed so the buffer start is aligned for any
    /// fixed-width value type Arrow reads through it.
    slab: SlabBuffer<u128>,
    validity: ValidityMask,
}

impl FixedWidthColumn {
    pub(super) fn new(
        data_type: &DataType,
        width: usize,
        capacity: usize,
        allocator: &mut SlabAllocator,
    ) -> Self {
        Self {
            data_type: data_type.clone(),
            width,
            capacity,
            slab: allocate_values_slab(capacity, width, allocator),
            validity: ValidityMask::new(capacity),
        }
    }
}

impl ColumnAccumulator for FixedWidthColumn {
    fn append_from_indices(
        &mut self,
        column: &ArrayRef,
        indices: &[u32],
        destination_start: usize,
        _allocator: &mut SlabAllocator,
    ) {
        let data = column.to_data();
        self.validity
            .append_indices(data.nulls(), indices, destination_start);
        let width = self.width;
        // SAFETY: every index is an in-bounds source position and the slab has
        // capacity for the appended rows (checked by the caller).
        unsafe {
            let src = data.buffers()[0].as_ptr().add(data.offset() * width);
            let dst = (self.slab.ptr_at_index(0) as *mut u8).add(destination_start * width);
            match width {
                1 => gather_fixed_width::<u8>(src, dst, indices),
                2 => gather_fixed_width::<u16>(src, dst, indices),
                4 => gather_fixed_width::<u32>(src, dst, indices),
                8 => gather_fixed_width::<u64>(src, dst, indices),
                16 => gather_fixed_width::<u128>(src, dst, indices),
                _ => unreachable!("built only for the widths above"),
            }
        }
    }

    fn append_from_range(
        &mut self,
        column: &ArrayRef,
        start: usize,
        len: usize,
        destination_start: usize,
        _allocator: &mut SlabAllocator,
    ) {
        let data = column.to_data();
        self.validity
            .append_range(data.nulls(), start, len, destination_start);
        let width = self.width;
        // SAFETY: the range is in bounds and the destination has room for it.
        unsafe {
            let src = data.buffers()[0]
                .as_ptr()
                .add((data.offset() + start) * width);
            let dst = (self.slab.ptr_at_index(0) as *mut u8).add(destination_start * width);
            std::ptr::copy_nonoverlapping(src, dst, len * width);
        }
    }

    fn append_from_mask(
        &mut self,
        column: &ArrayRef,
        mask: &BooleanBuffer,
        _selected_count: usize,
        destination_start: usize,
        _allocator: &mut SlabAllocator,
    ) {
        let data = column.to_data();
        self.validity
            .append_mask(data.nulls(), mask, destination_start);
        let width = self.width;
        // SAFETY: the mask is as long as the column and the slab has capacity
        // for the appended rows (checked by the caller).
        unsafe {
            let src = data.buffers()[0].as_ptr().add(data.offset() * width);
            let mut dst = (self.slab.ptr_at_index(0) as *mut u8).add(destination_start * width);
            for (start, end) in mask.set_slices() {
                let bytes = (end - start) * width;
                std::ptr::copy_nonoverlapping(src.add(start * width), dst, bytes);
                dst = dst.add(bytes);
            }
        }
    }

    fn append_from_batches(
        &mut self,
        column: &ChunkedColumn,
        ids: EncodedRowIds<'_>,
        shift: u32,
        destination_start: usize,
        _allocator: &mut SlabAllocator,
    ) {
        match ids {
            EncodedRowIds::Narrow(ids) => self.append_ids(column, ids, shift, destination_start),
            EncodedRowIds::Wide(ids) => self.append_ids(column, ids, shift, destination_start),
        }
    }

    fn take_array(
        &mut self,
        len: usize,
        allocator: &mut SlabAllocator,
    ) -> Result<ArrayRef, ArrowError> {
        let fresh = allocate_values_slab(self.capacity, self.width, allocator);
        let slab = std::mem::replace(&mut self.slab, fresh);
        let buffer = slab.into_buffer(len * self.width);
        let out = ArrayData::builder(self.data_type.clone())
            .len(len)
            .add_buffer(buffer)
            .nulls(self.validity.take(len));
        // SAFETY: a fixed-width array is a single values buffer plus optional
        // validity, and both were copied verbatim.
        Ok(make_array(unsafe { out.build_unchecked() }))
    }
}

/// A slab holding `capacity` values of `width` bytes, as the `u128` elements the
/// buffer is aligned by.
fn allocate_values_slab(
    capacity: usize,
    width: usize,
    allocator: &mut SlabAllocator,
) -> SlabBuffer<u128> {
    allocator.create_slab_buffer((capacity * width).div_ceil(size_of::<u128>()), false)
}

/// Gather `indices` rows from `src` to `dst`, as elements of `T`. A flat loop
/// with independent iterations, so the CPU overlaps the source cache misses of
/// many gathers. Shared with [`view`](super::view), whose views are gathered the
/// same way when they need no rebasing.
///
/// Four rows are copied per step rather than left to the compiler to unroll. How
/// far it unrolls this loop depends on the size of the function the loop ends up
/// in, so left implicit the cost of a copy moves with the shape of the code
/// around it rather than staying a property of the copy.
///
/// # Safety
/// `src` must hold every indexed row, `dst` must have room for `indices.len()`
/// elements, and both must be valid for unaligned `T` access.
pub(super) unsafe fn gather_fixed_width<T: Copy>(src: *const u8, dst: *mut u8, indices: &[u32]) {
    let src = src as *const T;
    let dst = dst as *mut T;
    /// Rows copied per step of the loop below.
    const STEP: usize = 4;
    unsafe {
        let mut destination = 0;
        let (steps, remainder) = indices.as_chunks::<STEP>();
        for step in steps {
            for (offset, &row) in step.iter().enumerate() {
                dst.add(destination + offset)
                    .write_unaligned(src.add(row as usize).read_unaligned());
            }
            destination += STEP;
        }
        for (offset, &row) in remainder.iter().enumerate() {
            dst.add(destination + offset)
                .write_unaligned(src.add(row as usize).read_unaligned());
        }
    }
}

/// Gather the rows at the encoded `ids` through one resolved values pointer
/// per batch. The two-level read stays a flat loop with independent iterations,
/// like [`gather_fixed_width`].
///
/// # Safety
/// Every id must name an in-bounds row of an in-bounds batch, `dst` must have
/// room for `ids.len()` elements, and all pointers must be valid for unaligned
/// `T` access.
unsafe fn gather_batches<T: Copy, I: Copy + Into<u64>>(
    values: &[*const u8],
    ids: &[I],
    shift: u32,
    dst: *mut u8,
) {
    let dst = dst as *mut T;
    unsafe {
        for (out_idx, &id) in ids.iter().enumerate() {
            let (batch, row) = split_encoded_id(id, shift);
            let src = (*values.get_unchecked(batch) as *const T).add(row);
            dst.add(out_idx).write_unaligned(src.read_unaligned());
        }
    }
}

impl FixedWidthColumn {
    fn append_ids<I: Copy + Into<u64>>(
        &mut self,
        column: &ChunkedColumn,
        ids: &[I],
        shift: u32,
        destination_start: usize,
    ) {
        self.validity
            .append_by_ids(ids, shift, destination_start, |batch| {
                column.data[batch]
                    .nulls()
                    .filter(|nulls| nulls.null_count() > 0)
            });
        // SAFETY: each id names an in-bounds row of its batch, and the slab has
        // capacity for `destination_start` plus the appended rows (checked by
        // the caller).
        unsafe {
            let dst = (self.slab.ptr_at_index(0) as *mut u8).add(destination_start * self.width);
            match self.width {
                1 => gather_batches::<u8, I>(&column.values, ids, shift, dst),
                2 => gather_batches::<u16, I>(&column.values, ids, shift, dst),
                4 => gather_batches::<u32, I>(&column.values, ids, shift, dst),
                8 => gather_batches::<u64, I>(&column.values, ids, shift, dst),
                16 => gather_batches::<u128, I>(&column.values, ids, shift, dst),
                _ => unreachable!("built only for the widths above"),
            }
        }
    }
}
