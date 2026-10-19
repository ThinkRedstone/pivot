//! The contract every column of a [`BatchAccumulator`](super::BatchAccumulator)
//! is accumulated through. The module docs of [`accumulator`](super) map out
//! which implementation a column type gets and why.

use arrow::array::ArrayData;
use arrow_array::ArrayRef;
use arrow_buffer::BooleanBuffer;
use arrow_schema::{ArrowError, DataType};

use crate::memory::SlabAllocator;

/// A column stored across many batches, prepared for gathering rows by
/// encoded id.
///
/// The batches own the buffers addressed by `values`; resolving those pointers
/// once keeps the gather loop from walking through `ArrayData` for every row.
pub struct ChunkedColumn {
    pub(super) data: Vec<ArrayData>,
    pub(super) values: Vec<*const u8>,
}

impl ChunkedColumn {
    pub fn new(data: Vec<ArrayData>) -> Self {
        let values = data.iter().map(resolve_values_pointer).collect();
        Self { data, values }
    }
}

fn resolve_values_pointer(data: &ArrayData) -> *const u8 {
    let width = match data.data_type() {
        DataType::Utf8View | DataType::BinaryView => size_of::<u128>(),
        other => match other.primitive_width() {
            Some(width) => width,
            None => return std::ptr::null(),
        },
    };
    // SAFETY: a fixed-stride array's first buffer holds `offset + len`
    // elements of `width` bytes.
    unsafe { data.buffers()[0].as_ptr().add(data.offset() * width) }
}

// SAFETY: the pointers address buffers `data` keeps alive through their Arcs,
// and are only used for read-only access.
unsafe impl Send for ChunkedColumn {}
unsafe impl Sync for ChunkedColumn {}

/// Rows to gather from a [`ChunkedColumn`], each encoded as
/// `batch << shift | row`, in whichever width the caller keeps its ids. The
/// accumulators are trait objects, so the width reaches them as data and each
/// gather runs in the width it was handed.
#[derive(Clone, Copy)]
pub enum EncodedRowIds<'a> {
    Narrow(&'a [u32]),
    Wide(&'a [u64]),
}

impl EncodedRowIds<'_> {
    pub fn len(&self) -> usize {
        match self {
            Self::Narrow(ids) => ids.len(),
            Self::Wide(ids) => ids.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Split an encoded row id into its batch index and the row within it.
#[inline(always)]
pub(super) fn split_encoded_id<I: Copy + Into<u64>>(id: I, shift: u32) -> (usize, usize) {
    let id: u64 = id.into();
    (
        (id >> shift) as usize,
        (id & ((1u64 << shift) - 1)) as usize,
    )
}

/// One column of an accumulation. Rows are copied in with the append methods
/// and handed to Arrow with [`take_array`](ColumnAccumulator::take_array).
///
/// An implementation is built for one Arrow type and one capacity and keeps
/// both, so neither is passed back in on every call. It must tolerate being
/// taken from and appended to again: an accumulator is reused for as many
/// batches as its owner emits.
pub(super) trait ColumnAccumulator {
    /// Append indexed rows from one batch's column.
    fn append_from_indices(
        &mut self,
        column: &ArrayRef,
        indices: &[u32],
        destination_start: usize,
        allocator: &mut SlabAllocator,
    );

    /// Append a contiguous range from one batch's column.
    fn append_from_range(
        &mut self,
        column: &ArrayRef,
        start: usize,
        len: usize,
        destination_start: usize,
        allocator: &mut SlabAllocator,
    );

    /// Append the rows `mask` selects from one batch's column, `selected_count` in
    /// number, walking the mask as runs of consecutive selected rows. A run
    /// is contiguous in the source and lands contiguously in the
    /// accumulation, so it copies as one piece however long it is. The mask
    /// must be as long as the column.
    fn append_from_mask(
        &mut self,
        column: &ArrayRef,
        mask: &BooleanBuffer,
        selected_count: usize,
        destination_start: usize,
        allocator: &mut SlabAllocator,
    );

    /// Append rows from a column stored across batches. Each `id` encodes
    /// `batch << shift | row`.
    fn append_from_batches(
        &mut self,
        column: &ChunkedColumn,
        ids: EncodedRowIds<'_>,
        shift: u32,
        destination_start: usize,
        allocator: &mut SlabAllocator,
    );

    /// Hand the `len` accumulated rows to Arrow and reset for the next batch.
    /// `len` is the accumulation's row count, which every column shares.
    fn take_array(
        &mut self,
        len: usize,
        allocator: &mut SlabAllocator,
    ) -> Result<ArrayRef, ArrowError>;
}
