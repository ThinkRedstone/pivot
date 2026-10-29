use std::sync::Arc;

use arrow_array::{ArrayRef, BooleanArray};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer};

use super::ArrayBuilder;
use crate::memory::{SlabAllocator, SlabBuffer};

/// [`ArrayBuilder`] for boolean arrays, holding its values as the bitmap
/// arrow wants from the start.
///
/// A Parquet PLAIN boolean page is that same bitmap, so a decoder appends a
/// page's bits with [`append_bits`](Self::append_bits), a copy of whole bytes
/// when both sides are byte-aligned, and a run of one value is a bit fill.
/// Only [`spare_mut`](ArrayBuilder::spare_mut), which hands out element
/// slots to be filled after the call, cannot write bits directly: it stages
/// the slots in a small vector that the next call packs. For booleans that
/// path serves the dictionary-index gather alone, which no writer produces
/// for the type.
pub struct BooleanBuilder {
    /// The bitmap, `capacity.div_ceil(8)` zeroed bytes. Every bit at or past
    /// `len` is zero, so appending only ever sets bits.
    bits: SlabBuffer<u8>,
    /// Values packed into `bits`.
    len: usize,
    /// Values handed out by `spare_mut` and not yet packed. They follow the
    /// packed ones, so the builder's length counts them too.
    staged: Vec<bool>,
}

impl BooleanBuilder {
    /// Appends `count` bits of `src`, starting at bit `src_bit` of it, least
    /// significant bit first.
    pub fn append_bits(&mut self, src: &[u8], src_bit: usize, count: usize) {
        self.pack_staged();
        let mut dst_bit = self.len;
        let mut src_bit = src_bit;
        let mut left = count;
        // The bits up to the destination's next byte boundary go in with one
        // mask, so the bulk below writes whole destination bytes.
        let head = (8 - dst_bit % 8) % 8;
        if head > 0 && left > 0 {
            let take = head.min(left);
            let byte = bits_from(src, src_bit, take) << (dst_bit % 8);
            unsafe { *self.bits.ptr_at_index(dst_bit / 8) |= byte };
            dst_bit += take;
            src_bit += take;
            left -= take;
        }
        let whole_bytes = left / 8;
        let shift = src_bit % 8;
        let src_byte = src_bit / 8;
        let dst_byte = dst_bit / 8;
        if shift == 0 {
            // SAFETY: the bitmap holds the builder's capacity, which the
            // caller never exceeds; the source holds `count` bits past
            // `src_bit`.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    src.as_ptr().add(src_byte),
                    self.bits.ptr_at_index(dst_byte),
                    whole_bytes,
                );
            }
        } else {
            // Each destination byte is the tail of one source byte and the
            // head of the next.
            for i in 0..whole_bytes {
                let byte = (src[src_byte + i] >> shift) | (src[src_byte + i + 1] << (8 - shift));
                unsafe { *self.bits.ptr_at_index(dst_byte + i) = byte };
            }
        }
        dst_bit += whole_bytes * 8;
        src_bit += whole_bytes * 8;
        let tail = left % 8;
        if tail > 0 {
            unsafe { *self.bits.ptr_at_index(dst_bit / 8) |= bits_from(src, src_bit, tail) };
        }
        self.len += count;
    }

    /// Packs the values `spare_mut` handed out, if any, onto the bitmap.
    fn pack_staged(&mut self) {
        if self.staged.is_empty() {
            return;
        }
        let staged = std::mem::take(&mut self.staged);
        for (dst_bit, &value) in (self.len..).zip(&staged) {
            self.set_bit(dst_bit, value);
        }
        self.len += staged.len();
        self.staged = staged;
        self.staged.clear();
    }

    #[inline(always)]
    fn set_bit(&mut self, bit: usize, value: bool) {
        unsafe { *self.bits.ptr_at_index(bit / 8) |= (value as u8) << (bit % 8) };
    }

    /// Sets `count` bits from `start` on, which are zero, to one: a mask on
    /// the first and last bytes touched, whole bytes between.
    fn set_bits(&mut self, start: usize, count: usize) {
        if count == 0 {
            return;
        }
        let end = start + count;
        let first = start / 8;
        let last = (end - 1) / 8;
        let head = 0xffu8 << (start % 8);
        let tail = low_mask(end - last * 8);
        unsafe {
            if first == last {
                *self.bits.ptr_at_index(first) |= head & tail;
                return;
            }
            *self.bits.ptr_at_index(first) |= head;
            std::ptr::write_bytes(self.bits.ptr_at_index(first + 1), 0xff, last - first - 1);
            *self.bits.ptr_at_index(last) |= tail;
        }
    }
}

/// A byte whose low `n` bits are set, `n` at most 8.
#[inline(always)]
fn low_mask(n: usize) -> u8 {
    ((1u16 << n) - 1) as u8
}

/// The `n` bits of `bytes` from bit `bit` on, `n` at most 8, in the low bits
/// of a byte.
#[inline(always)]
fn bits_from(bytes: &[u8], bit: usize, n: usize) -> u8 {
    let shift = bit % 8;
    let mut word = bytes[bit / 8] as u16;
    if shift + n > 8 {
        word |= (bytes[bit / 8 + 1] as u16) << 8;
    }
    (word >> shift) as u8 & low_mask(n)
}

impl ArrayBuilder for BooleanBuilder {
    type Element = bool;

    fn with_capacity(allocator: &mut SlabAllocator, capacity: usize) -> Self {
        Self {
            bits: allocator.create_slab_buffer(capacity.div_ceil(8), true),
            len: 0,
            staged: Vec::new(),
        }
    }

    #[inline]
    fn len(&self) -> usize {
        self.len + self.staged.len()
    }

    #[inline]
    fn push(&mut self, element: &bool, amount: usize) {
        self.pack_staged();
        if *element {
            self.set_bits(self.len, amount);
        }
        self.len += amount;
    }

    fn spare_mut(&mut self, count: usize) -> &mut [bool] {
        self.pack_staged();
        self.staged.resize(count, false);
        &mut self.staged
    }

    fn into_array(mut self, null_buffer: Option<Buffer>) -> ArrayRef {
        self.pack_staged();
        let len = self.len;
        let values = BooleanBuffer::new(self.bits.into_buffer(len.div_ceil(8)), 0, len);
        let nulls = null_buffer
            .map(|b| NullBuffer::new(BooleanBuffer::new(b, 0, len)))
            .filter(|n| n.null_count() != 0);
        Arc::new(BooleanArray::new(values, nulls))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;
    use arrow_array::Array;

    fn values_of(array: &ArrayRef) -> Vec<bool> {
        array
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap()
            .iter()
            .flatten()
            .collect()
    }

    fn pattern(count: usize) -> Vec<bool> {
        (0..count).map(|i| i % 3 == 0 || i % 7 == 0).collect()
    }

    fn pack(values: &[bool]) -> Vec<u8> {
        let mut bytes = vec![0u8; values.len().div_ceil(8)];
        for (i, &value) in values.iter().enumerate() {
            bytes[i / 8] |= (value as u8) << (i % 8);
        }
        bytes
    }

    #[test]
    fn packs_pushed_and_staged_values_in_order() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut builder = BooleanBuilder::with_capacity(&mut allocator, 20);

        builder.push(&true, 3);
        builder.push(&false, 7);
        builder.spare_mut(10).copy_from_slice(&[
            true, false, true, false, true, false, true, false, true, true,
        ]);
        let array = builder.into_array(None);

        let mut want = vec![
            true, true, true, false, false, false, false, false, false, false,
        ];
        want.extend([
            true, false, true, false, true, false, true, false, true, true,
        ]);
        assert_eq!(values_of(&array), want);
        assert_eq!(array.null_count(), 0);
    }

    #[test]
    fn appends_bits_at_every_alignment() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let values = pattern(100);
        let packed = pack(&values);
        let mut builder = BooleanBuilder::with_capacity(&mut allocator, 100);

        // Pieces whose lengths leave the destination and the source at
        // different offsets within a byte.
        let mut at = 0;
        for piece in [3, 8, 13, 1, 24, 5, 46] {
            builder.append_bits(&packed[at / 8..], at % 8, piece);
            at += piece;
        }
        let array = builder.into_array(None);

        assert_eq!(values_of(&array), values);
    }

    #[test]
    fn a_run_of_true_fills_whole_bytes() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut builder = BooleanBuilder::with_capacity(&mut allocator, 40);

        builder.push(&false, 5);
        builder.push(&true, 30);
        builder.push(&false, 5);
        let array = builder.into_array(None);

        let mut want = vec![false; 5];
        want.extend([true; 30]);
        want.extend([false; 5]);
        assert_eq!(values_of(&array), want);
    }

    #[test]
    fn applies_the_null_buffer() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut builder = BooleanBuilder::with_capacity(&mut allocator, 3);
        builder.push(&true, 3);

        let array = builder.into_array(Some(Buffer::from(vec![0b101u8])));

        let array = array.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(
            array.iter().collect::<Vec<_>>(),
            vec![Some(true), None, Some(true)]
        );
    }
}
