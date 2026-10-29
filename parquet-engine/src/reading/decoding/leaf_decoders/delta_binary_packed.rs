//! Decoder for Parquet's `DELTA_BINARY_PACKED` encoding.
//!
//! The encoding stores a column of integers as a starting value plus the
//! differences between consecutive values, bit-packed to the narrowest width
//! each group of them needs. A page is laid out as:
//!
//! ```text
//! header: block_size  miniblocks_per_block  total_values  first_value
//! block:  min_delta  bit_width[miniblocks_per_block]  miniblock*
//! ```
//!
//! where the header numbers are LEB128 varints, `first_value` and `min_delta`
//! are zigzag varints, and each miniblock holds `block_size /
//! miniblocks_per_block` deltas packed at its own width. A value is
//! reconstructed as `previous + min_delta + packed_delta`, so decoding is a
//! bit-unpack followed by a running sum.
//!
//! Two properties of the format keep the reader simple: the miniblock value
//! count is a multiple of 32, so every miniblock starts on a byte boundary,
//! and a block's widths are all known before its data, so a miniblock's byte
//! length is known before reading it.
//!
//! The encoding only exists for the `INT32` and `INT64` physical types, so
//! [`DeltaPacking`] pairs the integer natives with [`DeltaDecoder`] and the
//! floats with [`Unsupported`]. A header the decoder cannot make sense of
//! also declines, which surfaces as the same "unsupported encoding" error.

use std::marker::PhantomData;

use arrow_array::types::ArrowPrimitiveType;
use arrow_buffer::ArrowNativeType;
use bytes::Bytes;
use dispatch::arrays::ArrayBuilder;
use dispatch::memory::{MultiBufferReader, ReaderPosition};

use crate::reading::decoding::leaf_decoders::primitive::PrimitiveBuilder;
use crate::reading::decoding::leaf_decoders::{DecodeEncoding, Unsupported};

/// Widest block layout accepted. Writers use four miniblocks of 32 values;
/// this admits far more while keeping the per-block widths on the stack, and
/// rejects absurd headers instead of allocating from them.
const MAX_MINIBLOCKS: usize = 64;

/// Slack kept past the end of a miniblock's bytes so the unpacking loop can
/// always load a full 128-bit window without a bounds check. A value is at
/// most 64 bits and starts at most 7 bits into a byte, so 16 bytes covers
/// every load the last value can make.
const WINDOW_SLACK: usize = 16;

/// A column value the decoder can produce from the `i64` it accumulates in.
///
/// The encoding stores whole numbers, so every integer has one of these and
/// the floats do not. A decimal's raw unscaled integer is delta packed like
/// any other, so its carrier has one too.
pub trait FromDelta: Copy + Default {
    /// Narrow a value the decoder accumulated in `i64` to this type.
    fn from_delta(value: i64) -> Self;
}

macro_rules! integer_from_delta {
    ($($ty:ty),*) => {
        $(impl FromDelta for $ty {
            #[inline(always)]
            fn from_delta(value: i64) -> Self {
                value as $ty
            }
        })*
    };
}
integer_from_delta!(i8, u8, i16, u16, i32, u32, i64, u64, i128);

/// How a primitive column with this native type reads a `DELTA_BINARY_PACKED`
/// page: through [`DeltaDecoder`] for the integers, which the encoding is
/// defined for, and not at all for the floats.
pub trait DeltaPacking: ArrowNativeType {
    type Decoder<T: ArrowPrimitiveType<Native = Self>>: DecodeEncoding<
        Builder = PrimitiveBuilder<T>,
    >;
}

macro_rules! integer_delta_packing {
    ($($ty:ty),*) => {
        $(impl DeltaPacking for $ty {
            type Decoder<T: ArrowPrimitiveType<Native = $ty>> = DeltaDecoder<T>;
        })*
    };
}
integer_delta_packing!(i8, u8, i16, u16, i32, u32, i64, u64, i128);

macro_rules! float_delta_packing {
    ($($ty:ty),*) => {
        $(impl DeltaPacking for $ty {
            type Decoder<T: ArrowPrimitiveType<Native = $ty>> = Unsupported<PrimitiveBuilder<T>>;
        })*
    };
}
float_delta_packing!(f32, f64);

/// Reads `DELTA_BINARY_PACKED` pages for a fixed-width primitive column.
pub struct DeltaDecoder<T: ArrowPrimitiveType>
where
    T::Native: FromDelta,
{
    data: Vec<Bytes>,
    position: ReaderPosition,
    /// Deltas per miniblock, from the page header.
    values_per_miniblock: usize,
    /// Miniblocks per block, from the page header.
    miniblocks_per_block: usize,
    /// The current block's per-miniblock widths, read with its header.
    bit_widths: [u8; MAX_MINIBLOCKS],
    /// Which miniblock of the current block comes next.
    miniblock: usize,
    /// The current block's minimum delta, added back to every delta in it.
    min_delta: i64,
    /// The last value handed out, which the next delta builds on.
    last: i64,
    /// Values of the page not yet handed out, the header's count counting down.
    remaining: usize,
    /// The header's first value, held until the first read consumes it.
    first: Option<i64>,
    /// Values unpacked from the current miniblock and not yet handed out.
    ready: Vec<T::Native>,
    ready_pos: usize,
    phantom: PhantomData<T>,
}

impl<T: ArrowPrimitiveType> DecodeEncoding for DeltaDecoder<T>
where
    T::Native: FromDelta,
{
    type Builder = PrimitiveBuilder<T>;

    fn new(data: Vec<Bytes>, mut position: ReaderPosition) -> Option<Self> {
        let (block_size, miniblocks_per_block, total_values, first_value) = {
            let mut reader = MultiBufferReader::new(&data, &mut position);
            (
                read_uvarint(&mut reader) as usize,
                read_uvarint(&mut reader) as usize,
                read_uvarint(&mut reader) as usize,
                read_zigzag(&mut reader),
            )
        };

        if miniblocks_per_block == 0
            || miniblocks_per_block > MAX_MINIBLOCKS
            || block_size == 0
            || !block_size.is_multiple_of(miniblocks_per_block)
        {
            return None;
        }
        let values_per_miniblock = block_size / miniblocks_per_block;

        Some(Self {
            data,
            position,
            values_per_miniblock,
            miniblocks_per_block,
            bit_widths: [0; MAX_MINIBLOCKS],
            // Forces the first fill to read a block header.
            miniblock: miniblocks_per_block,
            min_delta: 0,
            last: first_value,
            remaining: total_values,
            first: (total_values > 0).then_some(first_value),
            // Empty, so the first read fills it rather than handing out slots
            // that hold nothing yet.
            ready: Vec::with_capacity(values_per_miniblock),
            ready_pos: 0,
            phantom: PhantomData,
        })
    }

    fn read(&mut self, builder: &mut PrimitiveBuilder<T>, size: usize) {
        let size = size.min(self.remaining);
        if size == 0 {
            return;
        }
        let out = builder.spare_mut(size);
        self.fill(out);
    }

    fn skip(&mut self, size: usize) {
        let mut left = size.min(self.remaining);
        if self.first.take().is_some() {
            left -= 1;
            self.remaining -= 1;
        }
        // Deltas are cumulative, so a skipped run still has to be decoded to
        // know where the next one starts; only the writing out is skipped.
        while left > 0 {
            if self.ready_pos < self.ready.len() {
                let take = (self.ready.len() - self.ready_pos).min(left);
                self.ready_pos += take;
                left -= take;
                self.remaining -= take;
                continue;
            }
            if self.next_miniblock_len() == 0 {
                break;
            }
            self.fill_ready();
        }
    }
}

impl<T: ArrowPrimitiveType> DeltaDecoder<T>
where
    T::Native: FromDelta,
{
    /// Values of the page not handed out yet.
    pub(super) fn remaining(&self) -> usize {
        self.remaining
    }

    /// Where the page's bytes end, once the whole page has been decoded. The
    /// `DELTA_LENGTH_BYTE_ARRAY` decoder reads its lengths through this decoder
    /// and then starts reading value bytes here.
    pub(super) fn position(&self) -> ReaderPosition {
        self.position
    }

    /// Decode the next `out.len()` values into `out`.
    pub(super) fn fill(&mut self, out: &mut [T::Native]) {
        let size = out.len();
        let mut written = 0;

        if let Some(first) = self.first.take() {
            out[0] = T::Native::from_delta(first);
            written = 1;
            self.remaining -= 1;
        }

        while written < size {
            // Whatever an earlier partial read left buffered goes out first.
            if self.ready_pos < self.ready.len() {
                let take = (self.ready.len() - self.ready_pos).min(size - written);
                out[written..written + take]
                    .copy_from_slice(&self.ready[self.ready_pos..self.ready_pos + take]);
                self.ready_pos += take;
                written += take;
                self.remaining -= take;
                continue;
            }
            let count = self.next_miniblock_len();
            if count == 0 {
                break;
            }
            if count <= size - written {
                // The read covers a whole miniblock, so unpack it straight into
                // the output and leave the buffer out of it.
                self.decode_miniblock(&mut out[written..written + count]);
                written += count;
                self.remaining -= count;
            } else {
                self.fill_ready();
            }
        }

        // A page that ends early leaves the tail of the reserved slots
        // untouched; fill them so no slot is read uninitialised.
        for slot in out[written..].iter_mut() {
            *slot = T::Native::default();
        }
    }

    /// How many values the next miniblock holds, `0` at the end of the page.
    fn next_miniblock_len(&self) -> usize {
        self.values_per_miniblock.min(self.remaining)
    }

    /// Unpack the next miniblock straight into `out`, which must have room for
    /// exactly [`next_miniblock_len`](Self::next_miniblock_len) values. Reads a
    /// block header first when the previous block is used up.
    fn decode_miniblock(&mut self, out: &mut [T::Native]) {
        if self.miniblock == self.miniblocks_per_block {
            self.read_block_header();
        }
        let bit_width = self.bit_widths[self.miniblock];
        self.miniblock += 1;

        // The writer emits every miniblock at full width, so the cursor moves
        // by the full length even when the block's tail holds no values, while
        // only the bytes covering `out` have to be readable.
        let full_bytes = self.values_per_miniblock * bit_width as usize / 8;
        let needed_bytes = (out.len() * bit_width as usize).div_ceil(8);

        let (min_delta, last) = (self.min_delta, self.last);
        let mut reader = MultiBufferReader::new(&self.data, &mut self.position);
        if reader.remaining_in_cur() >= needed_bytes + WINDOW_SLACK {
            // The miniblock, and the slack the loop over-reads past it, both
            // sit in the current buffer: unpack straight out of it.
            self.last =
                unpack_and_accumulate(reader.current_slice(), bit_width, min_delta, last, out);
            reader.skip(full_bytes);
        } else {
            // The miniblock straddles two buffers, or ends the page with no
            // slack behind it. Gather it into a padded scratch so the loop
            // keeps its unchecked loads. One copy per buffer boundary.
            let mut bytes = reader.read_bytes(needed_bytes);
            bytes.resize(needed_bytes + WINDOW_SLACK, 0);
            self.last = unpack_and_accumulate(&bytes, bit_width, min_delta, last, out);
            reader.skip(full_bytes - needed_bytes);
        }
    }

    /// Unpack the next miniblock into `ready`, for a read that wants fewer
    /// values than the miniblock holds. The rest waits there for the next call.
    fn fill_ready(&mut self) {
        let count = self.next_miniblock_len();
        self.ready.clear();
        self.ready_pos = 0;
        if count == 0 {
            return;
        }
        self.ready.resize(count, T::Native::default());
        let mut ready = std::mem::take(&mut self.ready);
        self.decode_miniblock(&mut ready);
        self.ready = ready;
    }

    /// Read a block header: its minimum delta and one width per miniblock.
    fn read_block_header(&mut self) {
        let mut reader = MultiBufferReader::new(&self.data, &mut self.position);
        self.min_delta = read_zigzag(&mut reader);
        for i in 0..self.miniblocks_per_block {
            self.bit_widths[i] = reader.read_u8();
        }
        self.miniblock = 0;
    }
}

/// Unpack `out.len()` deltas of `bit_width` bits from `src`, add `min_delta`
/// back to each, and run them into values starting from `last`. Returns the
/// final value, which the next miniblock continues from.
///
/// `src` must hold [`WINDOW_SLACK`] readable bytes past the packed deltas: each
/// value is read by loading a window at its starting byte and shifting down, so
/// the last value of a miniblock reads past the deltas themselves.
fn unpack_and_accumulate<N: FromDelta>(
    src: &[u8],
    bit_width: u8,
    min_delta: i64,
    last: i64,
    out: &mut [N],
) -> i64 {
    // Specialising on the width makes every shift and mask below a compile-time
    // constant, and lets the widths that fit a 64-bit window skip the 128-bit
    // one. The dispatch happens once per miniblock, not once per value.
    macro_rules! widths {
        ($($w:literal),* $(,)?) => {
            match bit_width {
                0 => run_of_equal_deltas(min_delta, last, out),
                $($w => unpack_width::<$w, N>(src, min_delta, last, out),)*
                _ => unreachable!("a delta bit width is at most 64"),
            }
        };
    }
    widths!(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,
        49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
    )
}

/// A miniblock whose width is zero: every delta is exactly `min_delta`, so the
/// values are an arithmetic run and no bytes are read at all.
fn run_of_equal_deltas<N: FromDelta>(min_delta: i64, last: i64, out: &mut [N]) -> i64 {
    let mut running = last;
    for slot in out.iter_mut() {
        running = running.wrapping_add(min_delta);
        *slot = N::from_delta(running);
    }
    running
}

/// The unpacking loop for one width, monomorphised so the mask and every shift
/// is a constant.
///
/// Values are taken eight at a time because eight of them occupy `8 * W` bits,
/// exactly `W` whole bytes: every group starts on a byte boundary, so a group's
/// eight bit offsets are the same constants no matter where in the miniblock it
/// sits, and the loop needs no running bit cursor. Extracting the eight is
/// independent work the processor can overlap; only the running sum that
/// follows is serial.
///
/// A value of `W` bits starting up to 7 bits into a byte ends within
/// `(W + 7).div_ceil(8)` bytes of that byte, so widths up to 57 are covered by
/// a 64-bit window and only the widest ones need a 128-bit one.
#[inline]
fn unpack_width<const W: usize, N: FromDelta>(
    src: &[u8],
    min_delta: i64,
    last: i64,
    out: &mut [N],
) -> i64 {
    debug_assert!(src.len() >= (out.len() * W).div_ceil(8) + WINDOW_SLACK);
    let mask: u64 = if W == 64 { u64::MAX } else { (1u64 << W) - 1 };
    let base = src.as_ptr();
    let mut running = last;
    let mut byte = 0usize;
    let (groups, remainder) = out.as_chunks_mut::<8>();
    for group in groups {
        let mut deltas = [0u64; 8];
        for (k, delta) in deltas.iter_mut().enumerate() {
            let bit = k * W;
            // SAFETY: the caller guarantees WINDOW_SLACK readable bytes past
            // the packed deltas, and this reads at most the byte the group's
            // last value starts in, so every window load stays inside `src`.
            *delta = unsafe { load_delta::<W>(base, byte + (bit >> 3), bit & 7, mask) };
        }

        // Then run them into values. The two adds here are the only serial
        // work, and leaving them be measured fastest. Three ways of shortening
        // the chain were slower: folding `min_delta` into the unpacking above,
        // lifting it out as a per-position constant, and summing each group in
        // log-many rounds. The next group's loads already overlap this chain,
        // so every one of them moved work into the phase doing the overlapping.
        for (slot, delta) in group.iter_mut().zip(deltas) {
            running = running.wrapping_add(min_delta).wrapping_add(delta as i64);
            *slot = N::from_delta(running);
        }
        byte += W;
    }

    // A miniblock holds a multiple of 32 values, so only a page's last one can
    // leave a partial group.
    let mut bit = byte * 8;
    for slot in remainder {
        // SAFETY: as above.
        let delta = unsafe { load_delta::<W>(base, bit >> 3, bit & 7, mask) };
        running = running.wrapping_add(min_delta).wrapping_add(delta as i64);
        *slot = N::from_delta(running);
        bit += W;
    }
    running
}

/// Read one `W`-bit value starting `shift` bits into the byte at `offset`.
///
/// # Safety
///
/// `offset` plus a 64-bit window (128-bit for the widest values) must lie
/// within the allocation `base` points into.
#[inline(always)]
unsafe fn load_delta<const W: usize>(
    base: *const u8,
    offset: usize,
    shift: usize,
    mask: u64,
) -> u64 {
    unsafe {
        if W <= 57 {
            let window = (base.add(offset) as *const u64).read_unaligned();
            (u64::from_le(window) >> shift) & mask
        } else {
            let window = (base.add(offset) as *const u128).read_unaligned();
            ((u128::from_le(window) >> shift) as u64) & mask
        }
    }
}

/// Read an unsigned LEB128 varint.
fn read_uvarint(reader: &mut MultiBufferReader) -> u64 {
    let mut value = 0u64;
    let mut shift = 0;
    loop {
        let byte = reader.read_u8();
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 || shift >= 63 {
            return value;
        }
        shift += 7;
    }
}

/// Read a zigzag-encoded LEB128 varint, where the sign rides in the low bit.
fn read_zigzag(reader: &mut MultiBufferReader) -> i64 {
    let raw = read_uvarint(reader);
    ((raw >> 1) as i64) ^ -((raw & 1) as i64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::types::{Decimal64Type, Decimal128Type, Float64Type, Int32Type, Int64Type};
    use dispatch::memory::{SlabAllocator, init_test_free_pool};

    /// Encode `values` the way a writer does, so the tests drive the decoder
    /// from bytes rather than from its own internals.
    fn encode(values: &[i64], values_per_miniblock: usize, miniblocks: usize) -> Vec<u8> {
        let mut out = Vec::new();
        put_uvarint(&mut out, (values_per_miniblock * miniblocks) as u64);
        put_uvarint(&mut out, miniblocks as u64);
        put_uvarint(&mut out, values.len() as u64);
        put_zigzag(&mut out, values[0]);

        let deltas: Vec<i64> = values.windows(2).map(|w| w[1].wrapping_sub(w[0])).collect();
        for block in deltas.chunks(values_per_miniblock * miniblocks) {
            let min_delta = *block.iter().min().unwrap();
            put_zigzag(&mut out, min_delta);
            let widths: Vec<u8> = (0..miniblocks)
                .map(|i| {
                    let start = i * values_per_miniblock;
                    let chunk = block.get(start..).unwrap_or(&[]);
                    let chunk = &chunk[..chunk.len().min(values_per_miniblock)];
                    chunk
                        .iter()
                        .map(|d| 64 - d.wrapping_sub(min_delta).leading_zeros() as u8)
                        .max()
                        .unwrap_or(0)
                })
                .collect();
            out.extend_from_slice(&widths);
            for (i, width) in widths.iter().enumerate() {
                let start = i * values_per_miniblock;
                let chunk = block.get(start..).unwrap_or(&[]);
                let chunk = &chunk[..chunk.len().min(values_per_miniblock)];
                let mut bits = vec![0u8; values_per_miniblock * *width as usize / 8];
                for (j, delta) in chunk.iter().enumerate() {
                    let packed = delta.wrapping_sub(min_delta) as u64;
                    for bit in 0..*width as usize {
                        if packed >> bit & 1 == 1 {
                            let pos = j * *width as usize + bit;
                            bits[pos / 8] |= 1 << (pos % 8);
                        }
                    }
                }
                out.extend_from_slice(&bits);
            }
        }
        out
    }

    fn put_uvarint(out: &mut Vec<u8>, mut value: u64) {
        loop {
            let byte = (value & 0x7f) as u8;
            value >>= 7;
            out.push(if value == 0 { byte } else { byte | 0x80 });
            if value == 0 {
                return;
            }
        }
    }

    fn put_zigzag(out: &mut Vec<u8>, value: i64) {
        put_uvarint(out, ((value << 1) ^ (value >> 63)) as u64);
    }

    /// Decode a page split into `buffers` pieces, exercising the straddle path
    /// when there is more than one.
    fn decode_i64(page: &[u8], count: usize, buffers: usize) -> Vec<i64> {
        let chunk = page.len().div_ceil(buffers);
        let data: Vec<Bytes> = page.chunks(chunk).map(Bytes::copy_from_slice).collect();
        init_test_free_pool(8);
        let mut allocator = SlabAllocator::new(true);
        let mut builder = PrimitiveBuilder::<Int64Type>::with_capacity(&mut allocator, count);
        let mut decoder = DeltaDecoder::<Int64Type>::new(data, ReaderPosition::default()).unwrap();

        decoder.read(&mut builder, count);

        let array = builder.into_array(None);
        let values = array
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .unwrap();
        (0..count).map(|i| values.value(i)).collect()
    }

    #[test]
    fn decodes_a_run_of_ascending_values() {
        let values: Vec<i64> = (0..300).map(|i| 1_000 + i * 7).collect();
        let page = encode(&values, 32, 4);

        let decoded = decode_i64(&page, values.len(), 1);

        assert_eq!(decoded, values);
    }

    #[test]
    fn decodes_values_that_fall_and_rise() {
        let values: Vec<i64> = (0..256)
            .map(|i: i64| if i % 3 == 0 { -i * 1_000 } else { i * 17 })
            .collect();
        let page = encode(&values, 32, 4);

        let decoded = decode_i64(&page, values.len(), 1);

        assert_eq!(decoded, values);
    }

    #[test]
    fn decodes_wide_deltas_at_full_width() {
        let values: Vec<i64> = (0..128)
            .map(|i| {
                if i % 2 == 0 {
                    i64::MIN / 2
                } else {
                    i64::MAX / 2
                }
            })
            .collect();
        let page = encode(&values, 32, 4);

        let decoded = decode_i64(&page, values.len(), 1);

        assert_eq!(decoded, values);
    }

    #[test]
    fn decodes_a_page_scattered_across_buffers() {
        let values: Vec<i64> = (0..1_000).map(|i| 5_000_000 - i * 13).collect();
        let page = encode(&values, 32, 4);

        let decoded = decode_i64(&page, values.len(), 7);

        assert_eq!(decoded, values);
    }

    #[test]
    fn reads_a_page_in_several_calls() {
        let values: Vec<i64> = (0..200).map(|i| i * 3).collect();
        let page = encode(&values, 32, 4);
        let data = vec![Bytes::copy_from_slice(&page)];
        init_test_free_pool(8);
        let mut allocator = SlabAllocator::new(true);
        let mut builder = PrimitiveBuilder::<Int64Type>::with_capacity(&mut allocator, 200);
        let mut decoder = DeltaDecoder::<Int64Type>::new(data, ReaderPosition::default()).unwrap();

        for _ in 0..4 {
            decoder.read(&mut builder, 50);
        }

        let array = builder.into_array(None);
        let got = array
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .unwrap();
        assert_eq!((0..200).map(|i| got.value(i)).collect::<Vec<_>>(), values);
    }

    #[test]
    fn skipping_leaves_the_following_values_correct() {
        let values: Vec<i64> = (0..200).map(|i| 900 + i * 11).collect();
        let page = encode(&values, 32, 4);
        let data = vec![Bytes::copy_from_slice(&page)];
        init_test_free_pool(8);
        let mut allocator = SlabAllocator::new(true);
        let mut builder = PrimitiveBuilder::<Int64Type>::with_capacity(&mut allocator, 50);
        let mut decoder = DeltaDecoder::<Int64Type>::new(data, ReaderPosition::default()).unwrap();

        decoder.skip(150);
        decoder.read(&mut builder, 50);

        let array = builder.into_array(None);
        let got = array
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .unwrap();
        assert_eq!(
            (0..50).map(|i| got.value(i)).collect::<Vec<_>>(),
            values[150..]
        );
    }

    #[test]
    fn decodes_an_int32_column() {
        let values: Vec<i64> = (0..128).map(|i| 70_000 + i * 5).collect();
        let page = encode(&values, 32, 4);
        let data = vec![Bytes::copy_from_slice(&page)];
        init_test_free_pool(8);
        let mut allocator = SlabAllocator::new(true);
        let mut builder = PrimitiveBuilder::<Int32Type>::with_capacity(&mut allocator, 128);
        let mut decoder = DeltaDecoder::<Int32Type>::new(data, ReaderPosition::default()).unwrap();

        decoder.read(&mut builder, 128);

        let array = builder.into_array(None);
        let got = array
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .unwrap();
        assert_eq!(
            (0..128).map(|i| got.value(i) as i64).collect::<Vec<_>>(),
            values
        );
    }

    #[test]
    fn decodes_values_that_are_all_negative() {
        let values: Vec<i64> = (0..200).map(|i| -5_000_000 - i * 37).collect();
        let page = encode(&values, 32, 4);

        let decoded = decode_i64(&page, values.len(), 1);

        assert_eq!(decoded, values);
    }

    #[test]
    fn decodes_a_page_whose_first_value_is_negative() {
        // The first value rides the header as a zigzag varint of its own, so a
        // negative one exercises a path no delta does.
        let values: Vec<i64> = (0..64).map(|i| -42 + i).collect();
        let page = encode(&values, 32, 4);

        let decoded = decode_i64(&page, values.len(), 1);

        assert_eq!(decoded[0], -42);
        assert_eq!(decoded, values);
    }

    #[test]
    fn decodes_values_that_step_by_a_constant() {
        // Equal deltas leave nothing to pack: the miniblock's width is zero and
        // its values come from `min_delta` alone, reading no bytes at all.
        let values: Vec<i64> = (0..128).map(|i| 900 - i * 3).collect();
        let page = encode(&values, 32, 4);

        let decoded = decode_i64(&page, values.len(), 1);

        assert_eq!(decoded, values);
    }

    #[test]
    fn decodes_a_page_holding_one_value() {
        // Only the header: a lone value has no delta to follow it.
        let values = vec![-7i64];
        let page = encode(&values, 32, 4);

        let decoded = decode_i64(&page, 1, 1);

        assert_eq!(decoded, values);
    }

    #[test]
    fn decodes_a_page_that_ends_mid_miniblock() {
        // 100 values leave the last miniblock part full, and the last group of
        // it part full too, so both tail paths run.
        let values: Vec<i64> = (0..100).map(|i| 17 * i - 500).collect();
        let page = encode(&values, 32, 4);

        let decoded = decode_i64(&page, values.len(), 1);

        assert_eq!(decoded, values);
    }

    #[test]
    fn decodes_the_extremes_of_an_int32_column() {
        let values: Vec<i64> = vec![i32::MIN as i64, 0, i32::MAX as i64, -1, 1];
        let page = encode(&values, 32, 4);
        let data = vec![Bytes::copy_from_slice(&page)];
        init_test_free_pool(8);
        let mut allocator = SlabAllocator::new(true);
        let mut builder =
            PrimitiveBuilder::<Int32Type>::with_capacity(&mut allocator, values.len());
        let mut decoder = DeltaDecoder::<Int32Type>::new(data, ReaderPosition::default()).unwrap();

        decoder.read(&mut builder, values.len());

        let array = builder.into_array(None);
        let got = array
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .unwrap();
        assert_eq!(
            (0..values.len())
                .map(|i| got.value(i) as i64)
                .collect::<Vec<_>>(),
            values
        );
    }

    #[test]
    fn decodes_a_decimal_column_carried_at_both_widths() {
        // A decimal is packed as its raw unscaled integer, so the same page
        // decodes under either carrier through the integer decoder.
        let values: Vec<i64> = (0..64).map(|i| -12_345 + i * 100).collect();
        let page = encode(&values, 32, 4);
        init_test_free_pool(8);
        let mut allocator = SlabAllocator::new(true);
        let mut narrow =
            PrimitiveBuilder::<Decimal64Type>::with_capacity(&mut allocator, values.len());
        let mut wide =
            PrimitiveBuilder::<Decimal128Type>::with_capacity(&mut allocator, values.len());

        DeltaDecoder::<Decimal64Type>::new(
            vec![Bytes::copy_from_slice(&page)],
            ReaderPosition::default(),
        )
        .unwrap()
        .read(&mut narrow, values.len());
        DeltaDecoder::<Decimal128Type>::new(
            vec![Bytes::copy_from_slice(&page)],
            ReaderPosition::default(),
        )
        .unwrap()
        .read(&mut wide, values.len());

        let narrow = narrow.into_array(None);
        let narrow = narrow
            .as_any()
            .downcast_ref::<arrow_array::Decimal64Array>()
            .unwrap();
        let wide = wide.into_array(None);
        let wide = wide
            .as_any()
            .downcast_ref::<arrow_array::Decimal128Array>()
            .unwrap();
        assert_eq!(
            (0..values.len())
                .map(|i| narrow.value(i))
                .collect::<Vec<_>>(),
            values
        );
        assert_eq!(
            (0..values.len())
                .map(|i| wide.value(i) as i64)
                .collect::<Vec<_>>(),
            values
        );
    }

    #[test]
    fn a_float_column_has_no_delta_form() {
        // The encoding stores whole numbers, so a float page cannot be one and
        // the float's decoder declines rather than reading the bytes as
        // something else.
        let page = encode(&[1i64, 2, 3], 32, 4);

        let decoder = <f64 as DeltaPacking>::Decoder::<Float64Type>::new(
            vec![Bytes::copy_from_slice(&page)],
            ReaderPosition::default(),
        );

        assert!(decoder.is_none());
    }

    #[test]
    fn skipping_the_whole_page_leaves_nothing_to_read() {
        let values: Vec<i64> = (0..64).map(|i| i * 5).collect();
        let page = encode(&values, 32, 4);
        let data = vec![Bytes::copy_from_slice(&page)];
        init_test_free_pool(8);
        let mut allocator = SlabAllocator::new(true);
        let mut builder = PrimitiveBuilder::<Int64Type>::with_capacity(&mut allocator, 64);
        let mut decoder = DeltaDecoder::<Int64Type>::new(data, ReaderPosition::default()).unwrap();

        decoder.skip(values.len());
        decoder.read(&mut builder, 10);

        assert_eq!(builder.len(), 0);
    }
}
