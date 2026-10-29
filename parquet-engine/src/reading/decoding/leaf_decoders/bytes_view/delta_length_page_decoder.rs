//! [`DeltaLengthPageDecoder`] — reads `DELTA_LENGTH_BYTE_ARRAY` pages into a
//! [`ViewBuilder`].
//!
//! The page is a `DELTA_BINARY_PACKED` block holding every value's length,
//! followed by all the value bytes back to back. Splitting it that way is why
//! a writer picks this encoding for strings: the lengths pack into a couple of
//! bits each, and the bytes lose the four-byte prefix that plain encoding
//! spends per value.
//!
//! So the lengths are decoded up front, through the same decoder a delta-packed
//! integer column uses, and reading a value is then only a matter of walking
//! the byte section. That walk is where the encoding pays off again: values sit
//! next to each other with nothing in between, so a run of them inside one
//! buffer becomes a run of views over that buffer with no copying.

use arrow_array::builder::make_view;
use arrow_array::types::{ByteViewType, Int32Type};
use arrow_buffer::Buffer;
use bytes::Bytes;
use dispatch::env::MAX_INLINE_STRING_VIEW;
use dispatch::memory::{MultiBufferReader, ReaderPosition};
use std::marker::PhantomData;

use crate::reading::decoding::leaf_decoders::delta_binary_packed::DeltaDecoder;
use crate::reading::decoding::leaf_decoders::{ArrayBuilder, DecodeEncoding};
use dispatch::arrays::ViewBuilder;

/// Reads `DELTA_LENGTH_BYTE_ARRAY` pages, producing views for a string or
/// binary [`ViewBuilder`] (per `V`).
pub struct DeltaLengthPageDecoder<V: ByteViewType> {
    data: Vec<Bytes>,
    /// Arrow `Buffer` wrappers over `data`, registered as view blocks.
    buffers: Vec<Buffer>,
    /// Cursor into the value bytes, which follow the lengths.
    position: ReaderPosition,
    /// Where each value ends, measured from the start of the byte section and
    /// carrying a leading zero, so value `i` occupies `ends[i]..ends[i + 1]`.
    /// Held this way rather than as lengths because a run of values is then
    /// bounded by a search over the ends instead of a walk that adds them up.
    ends: Vec<u32>,
    /// The length of the next value to hand out.
    next: usize,
    phantom: PhantomData<V>,
}

impl<V: ByteViewType> DecodeEncoding for DeltaLengthPageDecoder<V> {
    type Builder = ViewBuilder<V>;

    fn new(data: Vec<Bytes>, position: ReaderPosition) -> Option<Self> {
        // The lengths are a delta-packed `INT32` page in their own right, so
        // they are read by the decoder that handles those, which also reports
        // where it stopped: that is where the value bytes begin.
        let mut length_decoder = DeltaDecoder::<Int32Type>::new(data.clone(), position)?;
        let mut lengths = vec![0i32; length_decoder.remaining()];
        length_decoder.fill(&mut lengths);

        let mut ends = Vec::with_capacity(lengths.len() + 1);
        let mut end = 0u32;
        ends.push(end);
        for len in &lengths {
            end += *len as u32;
            ends.push(end);
        }

        Some(Self {
            buffers: data.iter().map(|b| Buffer::from(b.clone())).collect(),
            data,
            position: length_decoder.position(),
            ends,
            next: 0,
            phantom: PhantomData,
        })
    }

    fn read(&mut self, builder: &mut ViewBuilder<V>, size: usize) {
        let mut left = size.min(self.values_left());
        while left > 0 {
            if self.position.offset >= self.data[self.position.buffer_index].len()
                && self.position.buffer_index + 1 < self.data.len()
            {
                self.position.buffer_index += 1;
                self.position.offset = 0;
            }
            let taken = self.read_from_current_buffer(builder, left);
            left -= taken;
            if left > 0 {
                // The value straddles a buffer boundary, so it has to be
                // gathered before it can be viewed.
                self.append_view_across_buffers(builder);
                left -= 1;
            }
        }
    }

    fn skip(&mut self, size: usize) {
        let size = size.min(self.values_left());
        let skipped = (self.ends[self.next + size] - self.ends[self.next]) as usize;
        self.next += size;
        let mut reader = MultiBufferReader::new(&self.data, &mut self.position);
        reader.skip(skipped);
    }
}

impl<V: ByteViewType> DeltaLengthPageDecoder<V> {
    /// Append views for as many of the next `size` values as sit whole inside
    /// the current buffer, returning how many that was. Stops one short when
    /// the next value crosses into the following buffer.
    ///
    /// Because the values are laid end to end, how many of them fit is a
    /// question about the ends alone, which `ends` answers with one search
    /// rather than a walk. The run is then written as a run: the block is
    /// looked up once and the views go straight into the builder's slots,
    /// instead of both being redone per value.
    fn read_from_current_buffer(&mut self, builder: &mut ViewBuilder<V>, size: usize) -> usize {
        let buffer = &self.data[self.position.buffer_index];
        let available = buffer.len();
        if self.position.offset >= available {
            return 0;
        }

        // Ends are measured from the start of the byte section, so rebase them
        // on where this buffer's run starts.
        let start = self.ends[self.next];
        let budget = (available - self.position.offset) as u32;
        let candidates = &self.ends[self.next + 1..(self.next + 1 + size).min(self.ends.len())];
        let fitting = candidates.partition_point(|end| end - start <= budget);
        if fitting == 0 {
            return 0;
        }

        // The block is registered only when a value in the run is too long
        // to inline, so a run of short values names no page and keeps none
        // alive. A view over an inlined value ignores the block it is given.
        let mut previous = start;
        let needs_block = candidates[..fitting].iter().any(|&end| {
            let long = end - previous > MAX_INLINE_STRING_VIEW as u32;
            previous = end;
            long
        });
        let block = if needs_block {
            builder.append_block(self.buffers[self.position.buffer_index].clone())
        } else {
            0
        };
        let bytes: &[u8] = buffer.as_ref();
        let base = self.position.offset as u32;
        // Walk the run's ends into views over the block registered above.
        // Building a view reads the start of each value, since its first four
        // bytes become the view's prefix and a short value is copied in whole.
        // Prefetching those reads a few values ahead was measured and did not
        // pay: the page is already resident by the time this walks it.
        let out = builder.spare_mut(fitting);
        let mut offset = base;
        for (slot, end) in out.iter_mut().zip(&candidates[..fitting]) {
            let value_end = base + (end - start);
            // SAFETY: `fitting` counted only values whose end is inside this
            // buffer, so the slice is in bounds.
            let value = unsafe { bytes.get_unchecked(offset as usize..value_end as usize) };
            *slot = make_view(value, block, offset);
            offset = value_end;
        }
        self.position.offset = offset as usize;
        self.next += fitting;
        fitting
    }

    /// Values of the page not handed out yet.
    fn values_left(&self) -> usize {
        self.ends.len() - 1 - self.next
    }

    /// Gather one value that spans a buffer boundary and append its view.
    fn append_view_across_buffers(&mut self, builder: &mut ViewBuilder<V>) {
        let len = (self.ends[self.next + 1] - self.ends[self.next]) as usize;
        self.next += 1;
        let mut reader = MultiBufferReader::new(&self.data, &mut self.position);
        let bytes = reader.read_bytes(len);
        if len > MAX_INLINE_STRING_VIEW {
            let block = builder.append_block(Buffer::from(bytes));
            // SAFETY: the block holds exactly this value.
            unsafe { builder.append_view_unchecked(block, 0, len as u32) };
        } else {
            // SAFETY: a value this short is stored inside the view itself.
            unsafe { builder.append_raw_view_unchecked(&make_view(bytes.as_ref(), 0, 0)) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reading::decoding::leaf_decoders::{ArrayBuilder, DecodeEncoding};
    use crate::writing::encoder::delta::encode_length_byte_array;
    use arrow_array::types::StringViewType;
    use arrow_array::{Array, StringViewArray};
    use dispatch::memory::{SlabAllocator, init_test_free_pool};

    fn encode_page(values: &[&str]) -> Vec<Bytes> {
        let mut page = Vec::new();
        encode_length_byte_array(&StringViewArray::from(values.to_vec()), &mut page).unwrap();
        vec![Bytes::from(page)]
    }

    fn decode_in_runs(values: &[&str], run: usize) -> StringViewArray {
        let mut allocator = SlabAllocator::new(true);
        let mut decoder = DeltaLengthPageDecoder::<StringViewType>::new(
            encode_page(values),
            ReaderPosition::default(),
        )
        .unwrap();
        let mut builder = ViewBuilder::with_capacity(&mut allocator, values.len());
        for _ in 0..values.len().div_ceil(run) {
            decoder.read(&mut builder, run);
        }
        builder
            .into_array(None)
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap()
            .clone()
    }

    /// A page consumed as many runs (how nulls and pushed-down filter masks
    /// drive the decoder) registers its buffer once, not once per run.
    #[test]
    fn reading_a_page_in_runs_registers_its_buffer_once() {
        init_test_free_pool(4);
        let values = ["a value longer than twelve bytes"; 6];

        let array = decode_in_runs(&values, 1);

        assert_eq!(array.data_buffers().len(), 1);
        assert_eq!(array.iter().flatten().collect::<Vec<_>>(), values);
    }

    /// Values that inline into their views need no data block, so a page of
    /// short values leaves the array with no buffers to keep alive.
    #[test]
    fn short_values_register_no_buffer() {
        init_test_free_pool(4);
        let values = ["short", "values", "only"];

        let array = decode_in_runs(&values, 3);

        assert_eq!(array.data_buffers().len(), 0);
        assert_eq!(array.iter().flatten().collect::<Vec<_>>(), values);
    }
}
