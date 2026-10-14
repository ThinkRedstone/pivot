//! Accumulating a byte-view column (`Utf8View` or `BinaryView`).
//!
//! A view column is two things: 16-byte views, one per row, and the data
//! buffers the values too long to inline live in. The views always accumulate
//! into a slab of their own. Where the bytes behind them come from is decided
//! per append, by how much of the source the append takes:
//!
//! - An append taking at least [`RETAIN_SHARE_PERCENT`] of the source's rows
//!   copies no bytes. The source's data buffers join the emitted array's buffer
//!   list and each view's buffer index is rebased onto it, so the emitted array
//!   keeps that source alive. A source whose buffers are all already on the
//!   list is rebased onto the entries it has, whatever share the append takes,
//!   since referencing them again keeps nothing extra alive.
//! - A smaller append copies each value into blocks the column owns, so the
//!   emitted array names nothing of that source. Retaining a source for a few
//!   rows would pin its whole page (a decompressed page is a ring slot) for as
//!   long as the accumulation and then the emitted batch live, which under a
//!   selective filter is one pinned page per source batch for the whole scan.
//!
//! A value of up to [`INLINE_VIEW_LEN`] bytes needs neither: it lives inside its
//! own view, and both paths copy that view across as it lies.

use super::BUFFER_SIZE;
use super::column::{ChunkedColumn, ColumnAccumulator, EncodedRowIds, split_encoded_id};
use super::fixed_width::gather_fixed_width;
use super::validity::ValidityMask;
use crate::memory::{SlabAllocator, SlabBuffer};
use arrow::array::ArrayData;
use arrow_array::builder::make_view;
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, make_array};
use arrow_buffer::{Buffer, NullBuffer};
use arrow_schema::{ArrowError, DataType};

/// Bytes a view column takes for copied values at a time. Well under a slab, so
/// blocks pack into the buffers the allocator is bumping through rather than
/// taking one apiece and leaving the rest unused.
const DATA_BLOCK_SIZE: usize = 256 * 1024;

/// A byte-view value up to this length lives inside its view, with no data
/// buffer behind it.
pub(in crate::arrays) const INLINE_VIEW_LEN: u32 = 12;

/// The share of a source's rows an append must take for the source's data
/// buffers to be retained rather than its values copied (see the module docs).
const RETAIN_SHARE_PERCENT: usize = 15;

/// A column of byte views, with the bytes behind them held as the module docs say.
pub(super) struct ViewColumn {
    data_type: DataType,
    capacity: usize,
    views: SlabBuffer<u128>,
    /// The emitted array's data buffers, in the order the views' buffer
    /// indices name them: retained source buffers and owned blocks, as the
    /// appends that added them came.
    buffers: Vec<DataBlock>,
    /// The position on `buffers` of the block copied values are going into.
    /// Retained buffers can follow it on the list, so it need not be the last
    /// entry.
    open_block_idx: Option<usize>,
    validity: ValidityMask,
    /// Per-batch rebase cache for multi-batch appends: the buffer-list position
    /// a batch's data buffers were registered at. An entry is live only while
    /// its generation matches `generation`, which
    /// [`take_array`](ColumnAccumulator::take_array) bumps because the buffer
    /// list resets with every emitted batch.
    batch_bases: Vec<(u64, u32)>,
    generation: u64,
}

/// One data buffer of an accumulating view column, shared with the chunked
/// take ([`crate::arrays::take_chunked`]), whose view gather copies
/// values the same way.
///
/// A copied value lands in slab memory, except for the one case the ring cannot
/// serve: a value's bytes must be one contiguous run, and the ring hands out
/// runs of at most [`BUFFER_SIZE`]. A single value larger than that gets a heap
/// buffer of its own, one allocation of exactly that value's size. Retaining
/// the source buffer it came from would copy nothing, but it would pin however
/// much else that buffer holds, which copying exists to avoid.
pub(in crate::arrays) enum DataBlock {
    /// A source's data buffer, referenced as it is.
    Retained(Buffer),
    /// A block this column fills with copied values.
    Slab {
        slab: SlabBuffer<u8>,
        /// Bytes the block can hold.
        size: usize,
        /// Bytes written so far.
        used: usize,
    },
    /// A single copied value too large for any slab.
    OversizedValue(Vec<u8>),
}

impl DataBlock {
    /// Hand the block's bytes to Arrow. A slab block rides in the buffer's
    /// allocation `Arc`, so it returns to the pool with the last reference to
    /// the array.
    pub(in crate::arrays) fn into_buffer(self) -> Buffer {
        match self {
            Self::Retained(buffer) => buffer,
            Self::Slab { slab, used, .. } => slab.into_buffer(used),
            Self::OversizedValue(bytes) => Buffer::from_vec(bytes),
        }
    }
}

/// Where an append's values go, decided by [`ViewColumn::place_source_buffer`].
enum Placement {
    /// Rebase the views onto the source's buffers, which sit at this position
    /// of the buffer list.
    Rebase(u128),
    /// Copy the values into owned blocks.
    Copy,
}

impl ViewColumn {
    pub(super) fn new(
        data_type: &DataType,
        capacity: usize,
        allocator: &mut SlabAllocator,
    ) -> Self {
        Self {
            data_type: data_type.clone(),
            capacity,
            views: allocator.create_slab_buffer(capacity, false),
            buffers: Vec::new(),
            open_block_idx: None,
            validity: ValidityMask::new(capacity),
            batch_bases: Vec::new(),
            generation: 1,
        }
    }

    /// Where `source`'s buffers already sit on the list as one run of retained
    /// entries, if they do. A source is retained as a run, so its buffers are
    /// found together or not at all.
    ///
    /// Comparing by address is sound because the list holds a clone of every
    /// buffer it retains: those clones keep the allocations alive, so an address
    /// that still matches cannot have been freed and handed to a different buffer
    /// in the meantime.
    fn find_retained_run(&self, source: &[Buffer]) -> Option<u128> {
        (0..(self.buffers.len() + 1).saturating_sub(source.len()))
            .find(|&start| {
                self.buffers[start..start + source.len()]
                    .iter()
                    .zip(source)
                    .all(|(held, incoming)| {
                        if let DataBlock::Retained(buffer) = held {
                            buffer.as_ptr() == incoming.as_ptr() && buffer.len() == incoming.len()
                        } else {
                            false
                        }
                    })
            })
            .map(|start| start as u128)
    }

    /// Decide where an append of `selected` of a source's `source_rows` rows
    /// puts its values, retaining the source's buffers when it does.
    fn place_source_buffer(
        &mut self,
        source_buffers: &[Buffer],
        selected: usize,
        source_rows: usize,
    ) -> Placement {
        if source_buffers.is_empty() {
            // Every value of the source inlines, so no view will be rebased.
            return Placement::Rebase(0);
        }
        if let Some(base) = self.find_retained_run(source_buffers) {
            return Placement::Rebase(base);
        }
        if selected * 100 < source_rows * RETAIN_SHARE_PERCENT {
            return Placement::Copy;
        }
        let base = self.buffers.len() as u128;
        self.buffers
            .extend(source_buffers.iter().cloned().map(DataBlock::Retained));
        Placement::Rebase(base)
    }

    /// Read the view at `row` through a batch's resolved values pointer, whose
    /// array offset is already applied.
    #[inline(always)]
    fn view_at(values: *const u8, row: usize) -> u128 {
        // SAFETY: `row` is an in-bounds position of the batch the pointer
        // resolves, which holds one 16-byte view per element.
        unsafe { (values as *const u128).add(row).read_unaligned() }
    }

    /// Append rows gathered across the chunks of a [`ChunkedColumn`]. The
    /// chunks are retained whatever share of each the append takes: a caller
    /// holding rows as chunks keeps them alive for as long as it gathers from
    /// them, so referencing them pins nothing the caller does not pin already.
    fn append_batches<I: Copy + Into<u64>>(
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
        if self.batch_bases.len() < column.data.len() {
            self.batch_bases.resize(column.data.len(), (0, 0));
        }
        let Self {
            views,
            buffers,
            batch_bases,
            generation,
            ..
        } = self;
        // SAFETY: the views slab has capacity for `destination_start` plus the
        // appended rows (checked by the caller).
        unsafe {
            let mut dst = views.ptr_at_index(destination_start);
            for &id in ids {
                let (batch_idx, row) = split_encoded_id(id, shift);
                let mut view = Self::view_at(*column.values.get_unchecked(batch_idx), row);
                // A view longer than the inline limit points into its batch's
                // data buffers (buffers 1 onward); register those once per
                // emitted batch and rebase the buffer index onto the list.
                if view as u32 > INLINE_VIEW_LEN {
                    let entry = batch_bases.get_unchecked_mut(batch_idx);
                    if entry.0 != *generation {
                        *entry = (*generation, buffers.len() as u32);
                        buffers.extend(
                            column.data[batch_idx].buffers()[1..]
                                .iter()
                                .cloned()
                                .map(DataBlock::Retained),
                        );
                    }
                    view += (entry.1 as u128) << 64;
                }
                dst.write(view);
                dst = dst.add(1);
            }
        }
    }

    /// Append indexed views that keep pointing at the source's data buffers,
    /// which sit at `base` of the buffer list.
    fn append_rebasing_indices(
        &mut self,
        base: u128,
        views: &[u128],
        indices: &[u32],
        destination_start: usize,
    ) {
        unsafe {
            let src = views.as_ptr();
            let dst = self.views.ptr_at_index(destination_start);
            if base == 0 {
                gather_fixed_width::<u128>(src.cast(), dst.cast(), indices);
            } else {
                for (destination, &row) in indices.iter().enumerate() {
                    let mut view = src.add(row as usize).read_unaligned();
                    if view as u32 > INLINE_VIEW_LEN {
                        view += base << 64;
                    }
                    dst.add(destination).write(view);
                }
            }
        }
    }

    /// Append a contiguous range of views that keep pointing at the source's data
    /// buffers, which sit at `base` of the buffer list.
    fn append_rebasing_range(
        &mut self,
        base: u128,
        views: &[u128],
        start: usize,
        len: usize,
        destination_start: usize,
    ) {
        unsafe {
            let src = views.as_ptr();
            let dst = self.views.ptr_at_index(destination_start);
            if base == 0 {
                std::ptr::copy_nonoverlapping(src.add(start), dst, len);
            } else {
                for (destination, row) in (start..start + len).enumerate() {
                    let mut view = src.add(row).read_unaligned();
                    if view as u32 > INLINE_VIEW_LEN {
                        view += base << 64;
                    }
                    dst.add(destination).write(view);
                }
            }
        }
    }

    /// Append views over bytes copied into this column's own blocks, so the emitted
    /// array holds nothing of the source. An inlined value is its own view and comes
    /// over as it lies; a longer one's bytes are copied and re-viewed against the
    /// block they land in.
    fn append_copying_values<I>(
        &mut self,
        views: &[u128],
        source_buffers: &[Buffer],
        rows: I,
        destination_start: usize,
        allocator: &mut SlabAllocator,
    ) where
        I: IntoIterator<Item = usize>,
    {
        for (destination, row) in (destination_start..).zip(rows) {
            let view = views[row];
            let length = view as u32;
            let copied = if length <= INLINE_VIEW_LEN {
                view
            } else {
                // The bytes a long view names: its buffer, at its offset.
                let buffer = (view >> 64) as u32 as usize;
                let offset = (view >> 96) as u32 as usize;
                let value = &source_buffers[buffer][offset..offset + length as usize];
                copy_value(
                    &mut self.buffers,
                    &mut self.open_block_idx,
                    value,
                    allocator,
                )
            };
            // SAFETY: the slab has room for `destination_start` plus the appended
            // rows, which the caller checked against the capacity.
            unsafe { *self.views.ptr_at_index(destination) = copied };
        }
    }
}

impl ColumnAccumulator for ViewColumn {
    fn append_from_indices(
        &mut self,
        column: &ArrayRef,
        indices: &[u32],
        destination_start: usize,
        allocator: &mut SlabAllocator,
    ) {
        if indices.is_empty() {
            return;
        }
        let (views, source_buffers, nulls) = source_parts(column);
        self.validity
            .append_indices(nulls, indices, destination_start);
        match self.place_source_buffer(source_buffers, indices.len(), column.len()) {
            Placement::Rebase(base) => {
                self.append_rebasing_indices(base, views, indices, destination_start)
            }
            Placement::Copy => self.append_copying_values(
                views,
                source_buffers,
                indices.iter().map(|&row| row as usize),
                destination_start,
                allocator,
            ),
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
        if len == 0 {
            return;
        }
        let (views, source_buffers, nulls) = source_parts(column);
        self.validity
            .append_range(nulls, start, len, destination_start);
        match self.place_source_buffer(source_buffers, len, column.len()) {
            Placement::Rebase(base) => {
                self.append_rebasing_range(base, views, start, len, destination_start)
            }
            Placement::Copy => self.append_copying_values(
                views,
                source_buffers,
                start..start + len,
                destination_start,
                allocator,
            ),
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
            EncodedRowIds::Narrow(ids) => {
                self.append_batches(column, ids, shift, destination_start)
            }
            EncodedRowIds::Wide(ids) => self.append_batches(column, ids, shift, destination_start),
        }
    }

    fn take_array(
        &mut self,
        len: usize,
        allocator: &mut SlabAllocator,
    ) -> Result<ArrayRef, ArrowError> {
        self.generation += 1;
        let fresh = allocator.create_slab_buffer(self.capacity, false);
        let views = std::mem::replace(&mut self.views, fresh).into_buffer(len * size_of::<u128>());
        self.open_block_idx = None;
        let buffers: Vec<Buffer> = std::mem::take(&mut self.buffers)
            .into_iter()
            .map(DataBlock::into_buffer)
            .collect();
        let out = ArrayData::builder(self.data_type.clone())
            .len(len)
            .add_buffer(views)
            .add_buffers(buffers)
            .nulls(self.validity.take(len));
        // SAFETY: every appended view was rebased onto the buffer list emitted
        // with it, or built over the block its bytes were copied into, so all
        // view references stay valid. Views of null rows hold whatever bytes the
        // source held, which the validity marks as not a value.
        Ok(make_array(unsafe { out.build_unchecked() }))
    }
}

/// Borrow view metadata through the concrete array type. `to_data` would clone
/// and drop an Arc per data buffer on every append.
fn source_parts(source: &ArrayRef) -> (&[u128], &[Buffer], Option<&NullBuffer>) {
    match source.data_type() {
        DataType::Utf8View => {
            let array = source.as_string_view();
            (array.views(), array.data_buffers(), array.nulls())
        }
        DataType::BinaryView => {
            let array = source.as_binary_view();
            (array.views(), array.data_buffers(), array.nulls())
        }
        _ => unreachable!("built only for view types"),
    }
}

/// Copy `value` into the block being filled, the one at `open_block_idx`,
/// opening one (or a larger one for an outsized value) as needed, and return
/// the view naming where it landed.
pub fn copy_value(
    blocks: &mut Vec<DataBlock>,
    open_block_idx: &mut Option<usize>,
    value: &[u8],
    allocator: &mut SlabAllocator,
) -> u128 {
    if value.len() > BUFFER_SIZE {
        blocks.push(DataBlock::OversizedValue(value.to_vec()));
        return make_view(value, blocks.len() as u32 - 1, 0);
    }
    let free = match open_block_idx.map(|block| &blocks[block]) {
        Some(DataBlock::Slab { size, used, .. }) => size - used,
        Some(_) => unreachable!("the block being filled is always a slab block"),
        None => 0,
    };
    if free < value.len() {
        // A value too large for a standard block takes a block sized to it,
        // which still fits one slab (anything larger took the branch above).
        let size = DATA_BLOCK_SIZE.max(value.len());
        blocks.push(DataBlock::Slab {
            slab: allocator.create_slab_buffer(size, false),
            size,
            used: 0,
        });
        *open_block_idx = Some(blocks.len() - 1);
    }
    let block = open_block_idx.expect("a block was just ensured");
    let DataBlock::Slab { slab, used, .. } = &mut blocks[block] else {
        unreachable!("the block being filled is always a slab block")
    };
    let offset = *used;
    // SAFETY: the block holds `size` bytes and `offset + value.len()` is within
    // it, which is what the free-space check above establishes.
    unsafe {
        std::ptr::copy_nonoverlapping(value.as_ptr(), slab.ptr_at_index(offset), value.len());
    }
    *used += value.len();
    make_view(value, block as u32, offset as u32)
}
