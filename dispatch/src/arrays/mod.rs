//! Building Arrow arrays directly over engine-owned ([`WriteBuffer`](crate::memory::WriteBuffer))
//! memory.
//!
//! The [`builders`] accumulate values into a [`SlabAllocator`]-backed
//! buffer and produce the finished Arrow array with zero copies: the slab is
//! handed to Arrow via `Buffer::from_custom_allocation`, and its
//! `Arc<WriteBuffer>` keeps the memory pinned until the array is dropped (at
//! which point the buffer returns to the free pool). This keeps output
//! allocations on our pre-faulted, huge-paged, accounted memory instead of the
//! global allocator. [`SlabColumn`] is the single-slab column they build on.
//!
//! Used both by the parquet decoders (decoding pages into arrays) and the GROUP BY
//! output (building result columns).
//!
//! The kernels here build slab-backed arrays from selections of a column:
//! [`take`] picks rows of one source array, [`take_chunked`] rows scattered
//! across several chunks, and [`concat_chunks`] appends whole chunks back to
//! back.

pub mod accumulator;
pub mod builders;
mod int_bits;
mod slab_column;
pub mod variant;

use std::sync::Arc;

use arrow::array::ArrayData;
use arrow::compute::kernels::interleave::interleave;
use arrow::compute::take as arrow_take;
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, StructArray, UInt32Array, make_array};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer};
use arrow_schema::{ArrowError, DataType};

use crate::arrays::accumulator::{DataBlock, INLINE_VIEW_LEN, copy_value};
use crate::memory::SlabAllocator;
pub use builders::{ArrayBuilder, PrimitiveBuilder, ValidityBuilder, ViewBuilder};
pub use int_bits::IntBits;
pub use slab_column::SlabColumn;

/// Collect the ascending positions of the set bits of `bits` into `out`,
/// replacing its contents.
///
/// Extending a `Vec` straight from the bit iterator pays a capacity check
/// per element (the iterator has no usable size hint); reserving the exact
/// popcount up front and writing raw skips that.
pub fn collect_set_positions(bits: &BooleanBuffer, out: &mut Vec<u32>) {
    out.clear();
    let set = bits.count_set_bits();
    out.reserve(set);
    // SAFETY: `reserve(set)` guarantees capacity for every set bit, and the
    // iterator yields exactly `set` positions.
    unsafe {
        let ptr = out.as_mut_ptr();
        let mut len = 0;
        for position in bits.set_indices_u32() {
            ptr.add(len).write(position);
            len += 1;
        }
        out.set_len(len);
    }
}

/// Take the rows of `column` at `indices` (ascending) into a new array.
///
/// Fixed-width columns gather raw values into a slab buffer, the same
/// ring-owned memory the rest of the dataflow runs on; view columns gather the
/// view structs and share the underlying data buffers. A nullable column's
/// validity gathers into a slab-backed bitmap alongside (null rows copy
/// whatever value bytes they hold, which the bitmap marks dead). Anything else
/// (dictionaries, nested types) goes through Arrow's take kernel instead.
pub fn take(
    allocator: &mut SlabAllocator,
    column: &ArrayRef,
    indices: &[u32],
) -> Result<ArrayRef, ArrowError> {
    let data = column.to_data();
    match data.data_type() {
        DataType::Utf8View | DataType::BinaryView => {
            let views = gather_indices::<u128>(allocator, &data, indices);
            let out = ArrayData::builder(data.data_type().clone())
                .len(indices.len())
                .add_buffer(views)
                .add_buffers(data.buffers()[1..].to_vec())
                .nulls(taken_validity(allocator, &data, indices));
            // SAFETY: the views are copied verbatim and reference the same
            // data buffers as the source array, so every view stays valid.
            Ok(make_array(unsafe { out.build_unchecked() }))
        }
        dt => {
            let values = match dt.primitive_width() {
                Some(1) => gather_indices::<u8>(allocator, &data, indices),
                Some(2) => gather_indices::<u16>(allocator, &data, indices),
                Some(4) => gather_indices::<u32>(allocator, &data, indices),
                Some(8) => gather_indices::<u64>(allocator, &data, indices),
                Some(16) => gather_indices::<u128>(allocator, &data, indices),
                _ => return take_via_arrow(column, indices),
            };
            let out = ArrayData::builder(dt.clone())
                .len(indices.len())
                .add_buffer(values)
                .nulls(taken_validity(allocator, &data, indices));
            // SAFETY: a fixed-width array is a single values buffer plus its
            // validity, and both gather verbatim per taken row.
            Ok(make_array(unsafe { out.build_unchecked() }))
        }
    }
}

/// The taken rows' validity as a slab-backed bitmap, `None` when the source
/// rows are all valid (so a null-free column costs nothing here).
fn taken_validity(
    allocator: &mut SlabAllocator,
    data: &ArrayData,
    indices: &[u32],
) -> Option<NullBuffer> {
    let source_validity = data.nulls()?;
    if source_validity.null_count() == 0 {
        return None;
    }
    let mut bitmap = ValidityBuilder::with_capacity(allocator, indices.len());
    for &row in indices {
        bitmap.append_n(1, source_validity.is_valid(row as usize));
    }
    Some(NullBuffer::new(BooleanBuffer::new(
        bitmap.into_buffer(),
        0,
        indices.len(),
    )))
}

fn gather_indices<T: Copy>(
    allocator: &mut SlabAllocator,
    data: &ArrayData,
    indices: &[u32],
) -> Buffer {
    let slab = allocator.create_slab_buffer::<T>(indices.len(), false);
    // SAFETY: buffer 0 holds `offset + len` values of `T`, every index is a
    // valid row of the source array, and the slab holds `indices.len()`
    // slots. Unaligned reads tolerate sliced offsets.
    unsafe {
        let src = (data.buffers()[0].as_ptr() as *const T).add(data.offset());
        let dst = slab.ptr_at_index(0);
        for (out_idx, &row) in indices.iter().enumerate() {
            dst.add(out_idx)
                .write(src.add(row as usize).read_unaligned());
        }
    }
    slab.into_buffer(indices.len() * size_of::<T>())
}

fn take_via_arrow(column: &ArrayRef, indices: &[u32]) -> Result<ArrayRef, ArrowError> {
    arrow_take(column, &UInt32Array::from(indices.to_vec()), None)
}

/// Take rows of a column chunked across `chunks` into a new array, one row per
/// `(chunk, row within that chunk)` pair of `mapping`, in mapping order.
///
/// The same fast paths as [`take`], generalised to many sources: fixed-width
/// columns gather raw values into a slab buffer through one base pointer per
/// chunk, and view columns gather the view structs into a slab while copying
/// each out-of-line value's bytes into blocks of their own, in gather order —
/// the output shares nothing with the chunks, so their memory is free to drop
/// and a consumer walking the rows reads the bytes sequentially however
/// scattered the mapping was. A nullable column's validity gathers into a
/// slab-backed bitmap alongside. A struct column gathers each of its children
/// this same way and its own validity alongside, so a nested column costs no
/// more heap than its leaves would as flat columns. Anything else
/// (dictionaries, lists) goes through Arrow's interleave kernel instead. The
/// chunks must share one data type.
pub fn take_chunked(
    allocator: &mut SlabAllocator,
    chunks: &[ArrayRef],
    mapping: &[(u32, u32)],
) -> Result<ArrayRef, ArrowError> {
    let data: Vec<ArrayData> = chunks.iter().map(|chunk| chunk.to_data()).collect();
    let data_type = data[0].data_type();
    match data_type {
        DataType::Utf8View | DataType::BinaryView => {
            let (views, data_buffers) = gather_chunked_views(allocator, &data, mapping);
            let out = ArrayData::builder(data_type.clone())
                .len(mapping.len())
                .add_buffer(views)
                .add_buffers(data_buffers)
                .nulls(taken_validity_chunked(allocator, &data, mapping));
            // SAFETY: each view is either inline, rebuilt over the block its
            // value's bytes were just copied into, or the empty view of a row
            // the bitmap marks null.
            Ok(make_array(unsafe { out.build_unchecked() }))
        }
        DataType::Struct(fields) => {
            let children = struct_children(chunks, fields.len());
            let gathered = children
                .iter()
                .map(|child_chunks| take_chunked(allocator, child_chunks, mapping))
                .collect::<Result<Vec<_>, _>>()?;
            let nulls = taken_validity_chunked(allocator, &data, mapping);
            Ok(Arc::new(StructArray::try_new(
                fields.clone(),
                gathered,
                nulls,
            )?))
        }
        dt => {
            let values = match dt.primitive_width() {
                Some(1) => gather_chunked::<u8>(allocator, &data, mapping),
                Some(2) => gather_chunked::<u16>(allocator, &data, mapping),
                Some(4) => gather_chunked::<u32>(allocator, &data, mapping),
                Some(8) => gather_chunked::<u64>(allocator, &data, mapping),
                Some(16) => gather_chunked::<u128>(allocator, &data, mapping),
                _ => return interleave_via_arrow(chunks, mapping),
            };
            let out = ArrayData::builder(dt.clone())
                .len(mapping.len())
                .add_buffer(values)
                .nulls(taken_validity_chunked(allocator, &data, mapping));
            // SAFETY: a fixed-width array is a single values buffer plus its
            // validity, and both gather verbatim per taken row.
            Ok(make_array(unsafe { out.build_unchecked() }))
        }
    }
}

/// Concatenate a column's `chunks` into one slab-backed array, rows in chunk
/// order — what [`take_chunked`] degenerates to when the order is already
/// right, without its per-row indirection: fixed-width values copy in one run
/// per chunk, and a view chunk's structs are walked without any mapping (each
/// out-of-line value's bytes still copy one value at a time, into blocks of
/// the output's own, so nothing of the chunks stays pinned). A struct column
/// concatenates each child this way and its own validity alongside. Anything
/// without a fast path (dictionaries, lists) goes through Arrow's concat
/// kernel instead. The chunks must share one data type.
pub fn concat_chunks(
    allocator: &mut SlabAllocator,
    chunks: &[ArrayRef],
) -> Result<ArrayRef, ArrowError> {
    let data: Vec<ArrayData> = chunks.iter().map(|chunk| chunk.to_data()).collect();
    let rows: usize = data.iter().map(ArrayData::len).sum();
    let data_type = data[0].data_type();
    match data_type {
        DataType::Utf8View | DataType::BinaryView => {
            let (views, data_buffers) = concat_view_chunks(allocator, &data, rows);
            let out = ArrayData::builder(data_type.clone())
                .len(rows)
                .add_buffer(views)
                .add_buffers(data_buffers)
                .nulls(concatenated_validity(allocator, &data, rows));
            // SAFETY: each view is either inline, rebuilt over the block its
            // value's bytes were just copied into, or the empty view of a row
            // the bitmap marks null.
            Ok(make_array(unsafe { out.build_unchecked() }))
        }
        DataType::Struct(fields) => {
            let children = struct_children(chunks, fields.len());
            let concatenated = children
                .iter()
                .map(|child_chunks| concat_chunks(allocator, child_chunks))
                .collect::<Result<Vec<_>, _>>()?;
            let nulls = concatenated_validity(allocator, &data, rows);
            Ok(Arc::new(StructArray::try_new(
                fields.clone(),
                concatenated,
                nulls,
            )?))
        }
        dt => {
            let values = match dt.primitive_width() {
                Some(1) => concat_fixed_width_chunks::<u8>(allocator, &data, rows),
                Some(2) => concat_fixed_width_chunks::<u16>(allocator, &data, rows),
                Some(4) => concat_fixed_width_chunks::<u32>(allocator, &data, rows),
                Some(8) => concat_fixed_width_chunks::<u64>(allocator, &data, rows),
                Some(16) => concat_fixed_width_chunks::<u128>(allocator, &data, rows),
                _ => return concat_via_arrow(chunks),
            };
            let out = ArrayData::builder(dt.clone())
                .len(rows)
                .add_buffer(values)
                .nulls(concatenated_validity(allocator, &data, rows));
            // SAFETY: a fixed-width array is a single values buffer plus its
            // validity, and both copy verbatim chunk by chunk.
            Ok(make_array(unsafe { out.build_unchecked() }))
        }
    }
}

/// The chunks of each child column of a struct column's `chunks`, child by
/// child: `children[i]` holds child `i` of every chunk, in chunk order. A
/// sliced struct chunk hands out children already sliced with it, so a row
/// number of the chunk is the same row of each child.
fn struct_children(chunks: &[ArrayRef], field_count: usize) -> Vec<Vec<ArrayRef>> {
    (0..field_count)
        .map(|child| {
            chunks
                .iter()
                .map(|chunk| chunk.as_struct().column(child).clone())
                .collect()
        })
        .collect()
}

fn concat_fixed_width_chunks<T: Copy>(
    allocator: &mut SlabAllocator,
    chunks: &[ArrayData],
    rows: usize,
) -> Buffer {
    // The slab is typed so its pointer carries the value type's alignment;
    // the copies themselves are byte-wise, indifferent to sliced offsets.
    let slab = allocator.create_slab_buffer::<T>(rows, false);
    let mut written = 0;
    for chunk in chunks {
        // SAFETY: buffer 0 holds `offset + len` values, and the slab holds
        // every chunk's rows.
        unsafe {
            let values = (chunk.buffers()[0].as_ptr() as *const T).add(chunk.offset());
            std::ptr::copy_nonoverlapping(
                values as *const u8,
                slab.ptr_at_index(written) as *mut u8,
                chunk.len() * size_of::<T>(),
            );
        }
        written += chunk.len();
    }
    slab.into_buffer(rows * size_of::<T>())
}

fn concat_view_chunks(
    allocator: &mut SlabAllocator,
    chunks: &[ArrayData],
    rows: usize,
) -> (Buffer, Vec<Buffer>) {
    let slab = allocator.create_slab_buffer::<u128>(rows, false);
    let mut blocks: Vec<DataBlock> = Vec::new();
    let mut open_block_idx = None;
    let mut out_row = 0;
    for chunk in chunks {
        // SAFETY (of the `add`): buffer 0 of a view array holds `offset + len`
        // views, so the offset stays within its allocation.
        let views = unsafe { (chunk.buffers()[0].as_ptr() as *const u128).add(chunk.offset()) };
        let data_buffers = &chunk.buffers()[1..];
        let validity = chunk.nulls();
        for row in 0..chunk.len() {
            let valid = validity.is_none_or(|validity| validity.is_valid(row));
            // SAFETY: the row is in bounds; unaligned reads tolerate sliced
            // offsets. A null row's view is never read — its bytes are dead.
            let view = if valid {
                unsafe { views.add(row).read_unaligned() }
            } else {
                0
            };
            let length = view as u32;
            let copied = if length <= INLINE_VIEW_LEN {
                view
            } else {
                // The bytes the view names: its buffer, at its offset.
                let buffer = (view >> 64) as u32 as usize;
                let offset = (view >> 96) as u32 as usize;
                let value = &data_buffers[buffer][offset..offset + length as usize];
                copy_value(&mut blocks, &mut open_block_idx, value, allocator)
            };
            // SAFETY: the slab holds `rows` slots.
            unsafe { slab.ptr_at_index(out_row).write(copied) };
            out_row += 1;
        }
    }
    (
        slab.into_buffer(rows * size_of::<u128>()),
        blocks.into_iter().map(DataBlock::into_buffer).collect(),
    )
}

/// The concatenated rows' validity as a slab-backed bitmap, `None` when every
/// chunk is all-valid; all-valid chunks append as one run.
fn concatenated_validity(
    allocator: &mut SlabAllocator,
    chunks: &[ArrayData],
    rows: usize,
) -> Option<NullBuffer> {
    if chunks.iter().all(|chunk| chunk.null_count() == 0) {
        return None;
    }
    let mut bitmap = ValidityBuilder::with_capacity(allocator, rows);
    for chunk in chunks {
        match chunk.nulls() {
            None => bitmap.append_n(chunk.len(), true),
            Some(validity) => {
                for row in 0..chunk.len() {
                    bitmap.append_n(1, validity.is_valid(row));
                }
            }
        }
    }
    Some(NullBuffer::new(BooleanBuffer::new(
        bitmap.into_buffer(),
        0,
        rows,
    )))
}

fn concat_via_arrow(chunks: &[ArrayRef]) -> Result<ArrayRef, ArrowError> {
    let sources: Vec<&dyn Array> = chunks.iter().map(|chunk| chunk.as_ref()).collect();
    arrow::compute::concat(&sources)
}

/// The gathered rows' validity as a slab-backed bitmap, `None` when every
/// source chunk is all-valid (so null-free columns cost nothing here).
fn taken_validity_chunked(
    allocator: &mut SlabAllocator,
    chunks: &[ArrayData],
    mapping: &[(u32, u32)],
) -> Option<NullBuffer> {
    if chunks.iter().all(|chunk| chunk.null_count() == 0) {
        return None;
    }
    let chunk_validity: Vec<Option<&NullBuffer>> =
        chunks.iter().map(|chunk| chunk.nulls()).collect();
    let mut bitmap = ValidityBuilder::with_capacity(allocator, mapping.len());
    for &(chunk, row) in mapping {
        let valid =
            chunk_validity[chunk as usize].is_none_or(|validity| validity.is_valid(row as usize));
        bitmap.append_n(1, valid);
    }
    Some(NullBuffer::new(BooleanBuffer::new(
        bitmap.into_buffer(),
        0,
        mapping.len(),
    )))
}

fn gather_chunked<T: Copy>(
    allocator: &mut SlabAllocator,
    chunks: &[ArrayData],
    mapping: &[(u32, u32)],
) -> Buffer {
    let slab = allocator.create_slab_buffer::<T>(mapping.len(), false);
    let sources: Vec<*const T> = chunks
        .iter()
        // SAFETY (of the `add`): buffer 0 of each chunk holds `offset + len`
        // values of `T`, so the offset stays within its allocation.
        .map(|chunk| unsafe { (chunk.buffers()[0].as_ptr() as *const T).add(chunk.offset()) })
        .collect();
    // SAFETY: every mapping pair names a valid row of the chunk it points at,
    // and the slab holds `mapping.len()` slots. Unaligned reads tolerate
    // sliced offsets.
    unsafe {
        let dst = slab.ptr_at_index(0);
        for (out_idx, &(chunk, row)) in mapping.iter().enumerate() {
            dst.add(out_idx)
                .write(sources[chunk as usize].add(row as usize).read_unaligned());
        }
    }
    slab.into_buffer(mapping.len() * size_of::<T>())
}

/// Gather the 16-byte view structs at `mapping` into a slab, copying each
/// out-of-line value's bytes into fresh [`DataBlock`]s (the same blocks a
/// copying accumulator fills) and re-viewing it against the block it landed
/// in; a view of an inline value (12 bytes or fewer) is copied as is, and a
/// null row becomes the empty view — its bytes are dead, so its source view
/// is never even read. Returns the views buffer and the blocks as the output
/// array's data buffers.
fn gather_chunked_views(
    allocator: &mut SlabAllocator,
    chunks: &[ArrayData],
    mapping: &[(u32, u32)],
) -> (Buffer, Vec<Buffer>) {
    let slab = allocator.create_slab_buffer::<u128>(mapping.len(), false);
    let mut blocks: Vec<DataBlock> = Vec::new();
    let mut open_block_idx = None;
    let sources: Vec<(*const u128, &[Buffer], Option<&NullBuffer>)> = chunks
        .iter()
        // SAFETY (of the `add`): buffer 0 of a view array holds `offset + len`
        // views, so the offset stays within its allocation.
        .map(|chunk| {
            let views = unsafe { (chunk.buffers()[0].as_ptr() as *const u128).add(chunk.offset()) };
            (views, &chunk.buffers()[1..], chunk.nulls())
        })
        .collect();
    for (out_idx, &(chunk, row)) in mapping.iter().enumerate() {
        let (views, data_buffers, validity) = sources[chunk as usize];
        let valid = validity.is_none_or(|validity| validity.is_valid(row as usize));
        // SAFETY: every mapping pair names a valid row of the chunk it points
        // at. Unaligned reads tolerate sliced offsets.
        let view = if valid {
            unsafe { views.add(row as usize).read_unaligned() }
        } else {
            0
        };
        let length = view as u32;
        let copied = if length <= INLINE_VIEW_LEN {
            view
        } else {
            // The bytes the view names: its buffer, at its offset.
            let buffer = (view >> 64) as u32 as usize;
            let offset = (view >> 96) as u32 as usize;
            let value = &data_buffers[buffer][offset..offset + length as usize];
            copy_value(&mut blocks, &mut open_block_idx, value, allocator)
        };
        // SAFETY: the slab holds `mapping.len()` slots.
        unsafe { slab.ptr_at_index(out_idx).write(copied) };
    }
    (
        slab.into_buffer(mapping.len() * size_of::<u128>()),
        blocks.into_iter().map(DataBlock::into_buffer).collect(),
    )
}

fn interleave_via_arrow(
    chunks: &[ArrayRef],
    mapping: &[(u32, u32)],
) -> Result<ArrayRef, ArrowError> {
    let sources: Vec<&dyn Array> = chunks.iter().map(|chunk| chunk.as_ref()).collect();
    let indices: Vec<(usize, usize)> = mapping
        .iter()
        .map(|&(chunk, row)| (chunk as usize, row as usize))
        .collect();
    interleave(&sources, &indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;
    use arrow::compute::filter_record_batch;
    use arrow_array::{BooleanArray, Int64Array, RecordBatch, StringViewArray};
    use arrow_schema::{Field, Schema};
    use std::sync::Arc;

    #[test]
    fn take_matches_arrow_filter() {
        init_test_free_pool(4);
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("name", DataType::Utf8View, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5])),
                Arc::new(StringViewArray::from(vec![
                    "aa",
                    "a longer string that spills out of the view",
                    "cc",
                    "dd",
                    "ee",
                ])),
            ],
        )
        .unwrap();
        let mask = BooleanArray::from(vec![true, false, true, false, true]);
        let mut allocator = SlabAllocator::new(false);

        let taken = take(&mut allocator, batch.column(1), &[0, 2, 4]).unwrap();

        let expected = filter_record_batch(&batch, &mask).unwrap();
        assert_eq!(&taken, expected.column(1));
    }

    #[test]
    fn take_chunked_gathers_a_permutation_across_chunks() {
        init_test_free_pool(4);
        let chunks: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(vec![10, 11, 12])),
            Arc::new(Int64Array::from(vec![20, 21])),
        ];
        let mapping = [(1u32, 1u32), (0, 2), (1, 0), (0, 0), (0, 1)];
        let mut allocator = SlabAllocator::new(false);

        let taken = take_chunked(&mut allocator, &chunks, &mapping).unwrap();

        let expected = Int64Array::from(vec![21, 12, 20, 10, 11]);
        assert_eq!(taken.as_ref(), &expected);
    }

    #[test]
    fn take_chunked_copies_view_bytes_into_its_own_blocks() {
        use arrow_array::cast::AsArray;

        init_test_free_pool(4);
        let chunks: Vec<ArrayRef> = vec![
            Arc::new(StringViewArray::from(vec![
                "short",
                "a long value that lives in the first chunk's data buffer",
            ])),
            Arc::new(StringViewArray::from(vec![
                "a long value that lives in the second chunk's data buffer",
                "tiny",
            ])),
        ];
        let mapping = [(1u32, 0u32), (0, 1), (1, 1), (0, 0)];
        let mut allocator = SlabAllocator::new(false);

        let taken = take_chunked(&mut allocator, &chunks, &mapping).unwrap();

        let expected = StringViewArray::from(vec![
            "a long value that lives in the second chunk's data buffer",
            "a long value that lives in the first chunk's data buffer",
            "tiny",
            "short",
        ]);
        assert_eq!(taken.as_ref(), &expected);
        let source_buffers: Vec<*const u8> = chunks
            .iter()
            .flat_map(|chunk| chunk.as_string_view().data_buffers())
            .map(|buffer| buffer.as_ptr())
            .collect();
        assert!(
            taken
                .as_string_view()
                .data_buffers()
                .iter()
                .all(|buffer| !source_buffers.contains(&buffer.as_ptr())),
            "the gather must not hold a source data buffer"
        );
    }

    #[test]
    fn take_chunked_gathers_nulls_into_its_own_bitmap() {
        init_test_free_pool(4);
        let chunks: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(vec![Some(1), None])),
            Arc::new(Int64Array::from(vec![None, Some(4)])),
        ];
        let mapping = [(1u32, 1u32), (0, 1), (1, 0), (0, 0)];
        let mut allocator = SlabAllocator::new(false);

        let taken = take_chunked(&mut allocator, &chunks, &mapping).unwrap();

        let expected = Int64Array::from(vec![Some(4), None, None, Some(1)]);
        assert_eq!(taken.as_ref(), &expected);
    }

    /// A nullable view column gathers natively too: values follow their rows,
    /// null rows come out null, and no source data buffer is retained.
    #[test]
    fn take_chunked_gathers_nullable_views() {
        use arrow_array::cast::AsArray;

        init_test_free_pool(4);
        let chunks: Vec<ArrayRef> = vec![
            Arc::new(StringViewArray::from(vec![
                Some("a long value that lives in the first chunk's data buffer"),
                None,
            ])),
            Arc::new(StringViewArray::from(vec![Some("short"), None])),
        ];
        let mapping = [(0u32, 1u32), (1, 0), (0, 0), (1, 1)];
        let mut allocator = SlabAllocator::new(false);

        let taken = take_chunked(&mut allocator, &chunks, &mapping).unwrap();

        let expected = StringViewArray::from(vec![
            None,
            Some("short"),
            Some("a long value that lives in the first chunk's data buffer"),
            None,
        ]);
        assert_eq!(taken.as_ref(), &expected);
        let source_buffers: Vec<*const u8> = chunks
            .iter()
            .flat_map(|chunk| chunk.as_string_view().data_buffers())
            .map(|buffer| buffer.as_ptr())
            .collect();
        assert!(
            taken
                .as_string_view()
                .data_buffers()
                .iter()
                .all(|buffer| !source_buffers.contains(&buffer.as_ptr()))
        );
    }

    /// Two struct chunks with a long view in one child, a null in the other,
    /// and a null struct row, so every part of a struct gather is exercised.
    fn struct_chunks() -> Vec<ArrayRef> {
        use arrow_array::BinaryViewArray;
        use arrow_schema::Fields;

        let fields = Fields::from(vec![
            Field::new("metadata", DataType::BinaryView, false),
            Field::new("value", DataType::Int64, true),
        ]);
        vec![
            Arc::new(StructArray::new(
                fields.clone(),
                vec![
                    Arc::new(BinaryViewArray::from(vec![
                        b"m0".as_slice(),
                        b"a metadata value longer than twelve bytes",
                    ])),
                    Arc::new(Int64Array::from(vec![Some(10), None])),
                ],
                None,
            )),
            Arc::new(StructArray::new(
                fields,
                vec![
                    Arc::new(BinaryViewArray::from(vec![b"m2".as_slice()])),
                    Arc::new(Int64Array::from(vec![Some(20)])),
                ],
                Some(NullBuffer::from(vec![false])),
            )),
        ]
    }

    #[test]
    fn take_chunked_gathers_a_struct_child_by_child() {
        init_test_free_pool(4);
        let chunks = struct_chunks();
        let mapping = [(1u32, 0u32), (0, 1), (0, 0)];
        let mut allocator = SlabAllocator::new(false);

        let taken = take_chunked(&mut allocator, &chunks, &mapping).unwrap();

        let sources: Vec<&dyn Array> = chunks.iter().map(|chunk| chunk.as_ref()).collect();
        let expected = interleave(&sources, &[(1, 0), (0, 1), (0, 0)]).unwrap();
        assert_eq!(&taken, &expected);
        let source_buffers: Vec<*const u8> = chunks
            .iter()
            .flat_map(|chunk| chunk.as_struct().column(0).as_binary_view().data_buffers())
            .map(|buffer| buffer.as_ptr())
            .collect();
        assert!(
            taken
                .as_struct()
                .column(0)
                .as_binary_view()
                .data_buffers()
                .iter()
                .all(|buffer| !source_buffers.contains(&buffer.as_ptr())),
            "a struct gather must not hold a source data buffer either"
        );
    }

    #[test]
    fn concat_chunks_concatenates_a_struct_child_by_child() {
        init_test_free_pool(4);
        let chunks = struct_chunks();
        let mut allocator = SlabAllocator::new(false);

        let concatenated = concat_chunks(&mut allocator, &chunks).unwrap();

        let sources: Vec<&dyn Array> = chunks.iter().map(|chunk| chunk.as_ref()).collect();
        let expected = arrow::compute::concat(&sources).unwrap();
        assert_eq!(&concatenated, &expected);
    }

    /// Chunk concatenation matches Arrow's concat, nulls and out-of-line
    /// string bytes included, and retains no source data buffer.
    #[test]
    fn concat_chunks_matches_arrows_concat() {
        use arrow_array::cast::AsArray;

        init_test_free_pool(4);
        let int_chunks: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(vec![Some(1), None])),
            (Arc::new(Int64Array::from(vec![Some(3), None])) as ArrayRef).slice(0, 1),
            Arc::new(Int64Array::from(vec![None, Some(5)])),
        ];
        let string_chunks: Vec<ArrayRef> = vec![
            Arc::new(StringViewArray::from(vec![
                Some("short"),
                Some("a long value that lives in the first chunk's data buffer"),
            ])),
            Arc::new(StringViewArray::from(vec![None, Some("tiny")])),
        ];
        let mut allocator = SlabAllocator::new(false);

        let ints = concat_chunks(&mut allocator, &int_chunks).unwrap();
        let strings = concat_chunks(&mut allocator, &string_chunks).unwrap();

        let int_sources: Vec<&dyn Array> = int_chunks.iter().map(|c| c.as_ref()).collect();
        assert_eq!(&ints, &arrow::compute::concat(&int_sources).unwrap());
        let string_sources: Vec<&dyn Array> = string_chunks.iter().map(|c| c.as_ref()).collect();
        assert_eq!(&strings, &arrow::compute::concat(&string_sources).unwrap());
        let source_buffers: Vec<*const u8> = string_chunks
            .iter()
            .flat_map(|chunk| chunk.as_string_view().data_buffers())
            .map(|buffer| buffer.as_ptr())
            .collect();
        assert!(
            strings
                .as_string_view()
                .data_buffers()
                .iter()
                .all(|buffer| !source_buffers.contains(&buffer.as_ptr()))
        );
    }

    /// Single-source take of a nullable column: same rows and nulls as
    /// filtering with Arrow.
    #[test]
    fn take_gathers_a_nullable_column() {
        init_test_free_pool(4);
        let column: ArrayRef = Arc::new(Int64Array::from(vec![
            Some(1),
            None,
            Some(3),
            None,
            Some(5),
        ]));
        let mut allocator = SlabAllocator::new(false);

        let taken = take(&mut allocator, &column, &[4, 1, 3, 0]).unwrap();

        let expected = Int64Array::from(vec![Some(5), None, None, Some(1)]);
        assert_eq!(taken.as_ref(), &expected);
    }
}
