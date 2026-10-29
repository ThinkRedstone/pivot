//! Column decoder for BOOLEAN leaves.
//!
//! A PLAIN boolean page stores one bit per value, least significant bit first,
//! which is arrow's own bitmap layout and the layout the [`BooleanBuilder`]
//! holds, so decoding a page is copying its bits. An RLE boolean page stores a
//! 4-byte little-endian length followed by runs: a header whose low bit says
//! whether the run is one value repeated, which becomes a bit fill, or groups
//! of eight bit-packed values, which at a bit width of one are laid out as
//! PLAIN and are copied the same way.
//!
//! [`BooleanLeaf`] names these as the column's [`LeafType`], and
//! [`BooleanLeafDecoder`] is the ready-to-use [`TypedLeafDecoder`] over it.

use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, Datum, Scalar};
use bytes::Bytes;
use dispatch::arrays::BooleanBuilder;
use dispatch::memory::{MultiBufferReader, ReaderPosition, SlabAllocator};

use crate::reading::decoding::leaf_decoders::{
    ArrayBuilder, DecodeEncoding, DecodePlain, Dict, DictFromBytes, DictFromVecBytes, LeafType,
    TypedLeafDecoder, Unsupported,
};

/// [`DecodePlain`] implementation for booleans: one bit per value, least
/// significant bit first, copied into the builder's bitmap.
///
/// The cursor is the byte holding the next value's bit and which bit of it
/// that is, so a read or skip may stop anywhere within a byte.
pub struct BooleanPlainDecoder {
    data: Vec<Bytes>,
    position: ReaderPosition,
    /// The bit of the byte at `position` the next value sits in.
    bit: usize,
}

impl BooleanPlainDecoder {
    /// Appends `size` values to `builder`, one buffer of the page at a time.
    fn append_to(&mut self, builder: &mut BooleanBuilder, size: usize) {
        let mut left = size;
        while left > 0 {
            let buffer = &self.data[self.position.buffer_index];
            let bits_left = (buffer.len() - self.position.offset) * 8 - self.bit;
            if bits_left == 0 {
                self.position.buffer_index += 1;
                self.position.offset = 0;
                self.bit = 0;
                continue;
            }
            let take = left.min(bits_left);
            builder.append_bits(&buffer[self.position.offset..], self.bit, take);
            self.advance(take);
            left -= take;
        }
    }

    /// Moves the cursor `count` values on. It may come to rest at the end of
    /// a buffer, which the next read steps past.
    fn advance(&mut self, count: usize) {
        let bit = self.bit + count;
        self.position.offset += bit / 8;
        self.bit = bit % 8;
        while self.position.buffer_index < self.data.len() {
            let len = self.data[self.position.buffer_index].len();
            let past_end =
                self.position.offset > len || (self.position.offset == len && self.bit > 0);
            if !past_end {
                break;
            }
            self.position.offset -= len;
            self.position.buffer_index += 1;
        }
    }
}

impl DecodePlain for BooleanPlainDecoder {
    type Builder = BooleanBuilder;

    fn new(data: Vec<Bytes>, position: ReaderPosition) -> Self {
        Self {
            data,
            position,
            bit: 0,
        }
    }

    fn read(&mut self, builder: &mut BooleanBuilder, size: usize) {
        self.append_to(builder, size);
    }

    fn skip(&mut self, size: usize) {
        self.advance(size);
    }
}

/// The run an RLE page's cursor is in.
enum Run {
    /// None: the next run's header is at the cursor.
    Header,
    /// One value repeated, `left` times more.
    Repeated { value: bool, left: usize },
    /// Bit-packed values, `left` of them still at the cursor as PLAIN bits.
    Packed { left: usize },
}

/// [`DecodeEncoding`] implementation for an RLE boolean page.
pub struct BooleanRleDecoder {
    /// The cursor over the page's bytes, which reads a packed run's bits.
    bits: BooleanPlainDecoder,
    run: Run,
}

impl BooleanRleDecoder {
    /// Parses the run header at the cursor, which a packed run leaves on a
    /// byte boundary since its groups are whole bytes.
    ///
    /// A header is at most a five-byte varint and a value byte. One the
    /// current buffer holds whole is decoded from the buffer in place; one at
    /// a buffer's end goes through the reader, which crosses into the next.
    fn parse_run(&mut self) -> Run {
        let buffer = &self.bits.data[self.bits.position.buffer_index];
        let bytes = &buffer[self.bits.position.offset..];
        if bytes.len() >= 6 {
            let (header, used) = decode_varint(bytes);
            let run = run_of(header, || bytes[used]);
            self.bits.position.offset += used + matches!(run, Run::Repeated { .. }) as usize;
            run
        } else {
            let mut reader = MultiBufferReader::new(&self.bits.data, &mut self.bits.position);
            let header = reader.read_varint() as usize;
            run_of(header, || reader.read_u8())
        }
    }
}

/// The run a header announces. `value` fetches the byte holding a repeated
/// run's value, the one byte a width of one needs.
fn run_of(header: usize, value: impl FnOnce() -> u8) -> Run {
    if header & 1 == 0 {
        Run::Repeated {
            value: value() & 1 == 1,
            left: header >> 1,
        }
    } else {
        Run::Packed {
            left: (header >> 1) * 8,
        }
    }
}

/// Decodes the LEB128 varint at the start of `bytes`: its value and how many
/// bytes it took.
fn decode_varint(bytes: &[u8]) -> (usize, usize) {
    let mut value = 0;
    let mut used = 0;
    loop {
        let byte = bytes[used];
        value |= ((byte & 0x7f) as usize) << (7 * used);
        used += 1;
        if byte & 0x80 == 0 {
            return (value, used);
        }
    }
}

impl DecodeEncoding for BooleanRleDecoder {
    type Builder = BooleanBuilder;

    fn new(data: Vec<Bytes>, mut position: ReaderPosition) -> Option<Self> {
        // The stream's byte length leads it. The page's value count already
        // bounds the reads, so the length is stepped over.
        MultiBufferReader::new(&data, &mut position).read_u32_le();
        Some(Self {
            bits: BooleanPlainDecoder::new(data, position),
            run: Run::Header,
        })
    }

    fn read(&mut self, builder: &mut BooleanBuilder, size: usize) {
        let mut left = size;
        while left > 0 {
            if matches!(self.run, Run::Header) {
                self.run = self.parse_run();
            }
            let run_left = match &mut self.run {
                Run::Header => unreachable!("a run was just parsed"),
                Run::Repeated {
                    value,
                    left: run_left,
                } => {
                    let take = left.min(*run_left);
                    builder.push(value, take);
                    *run_left -= take;
                    left -= take;
                    *run_left
                }
                Run::Packed { left: run_left } => {
                    let take = left.min(*run_left);
                    self.bits.append_to(builder, take);
                    *run_left -= take;
                    left -= take;
                    *run_left
                }
            };
            if run_left == 0 {
                self.run = Run::Header;
            }
        }
    }

    fn skip(&mut self, size: usize) {
        let mut left = size;
        while left > 0 {
            if matches!(self.run, Run::Header) {
                self.run = self.parse_run();
            }
            let run_left = match &mut self.run {
                Run::Header => unreachable!("a run was just parsed"),
                Run::Repeated { left: run_left, .. } => {
                    let take = left.min(*run_left);
                    *run_left -= take;
                    left -= take;
                    *run_left
                }
                Run::Packed { left: run_left } => {
                    let take = left.min(*run_left);
                    self.bits.advance(take);
                    *run_left -= take;
                    left -= take;
                    *run_left
                }
            };
            if run_left == 0 {
                self.run = Run::Header;
            }
        }
    }
}

/// [`Dict`] implementation for booleans.
///
/// No writer dictionary-encodes a boolean column (the format reserves the
/// encoding for the other types), but a dictionary page of PLAIN booleans is
/// a well-defined thing to read, and reading it keeps this decoder on the
/// same page machinery as every other leaf. Such a dictionary holds two
/// entries at most, so it lives on the heap rather than on a slab.
pub struct BooleanDict {
    entries: Vec<bool>,
}

impl Dict for BooleanDict {
    type Builder = BooleanBuilder;
    type Item = bool;
    type EqConstant = bool;

    fn eq_constant_from_scalar(scalar: &Scalar<ArrayRef>) -> Option<bool> {
        let (array, _) = scalar.get();
        let booleans = array.as_boolean_opt()?;
        (booleans.len() == 1 && !booleans.is_null(0)).then(|| booleans.value(0))
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    #[inline(always)]
    fn entry(&self, idx: usize) -> bool {
        self.entries[idx]
    }
}

impl BooleanDict {
    fn from_plain_page(data: Vec<Bytes>, size: usize) -> Self {
        let mut position = ReaderPosition::default();
        let mut reader = MultiBufferReader::new(&data, &mut position);
        let mut entries = Vec::with_capacity(size);
        while entries.len() < size {
            let byte = reader.read_u8();
            for bit in 0..8.min(size - entries.len()) {
                entries.push((byte >> bit) & 1 == 1);
            }
        }
        Self { entries }
    }
}

impl DictFromBytes for BooleanDict {
    fn new_from_bytes(data: Bytes, size: usize, _allocator: &mut SlabAllocator) -> Self {
        Self::from_plain_page(vec![data], size)
    }
}

impl DictFromVecBytes for BooleanDict {
    fn new_from_vec_bytes(data: Vec<Bytes>, size: usize, _allocator: &mut SlabAllocator) -> Self {
        Self::from_plain_page(data, size)
    }
}

/// The [`LeafType`] of a boolean column: its pages are PLAIN bits, runs, or
/// dictionary indices.
pub struct BooleanLeaf;

impl LeafType for BooleanLeaf {
    type Builder = BooleanBuilder;

    type Plain = BooleanPlainDecoder;
    type DeltaBinaryPacked = Unsupported<BooleanBuilder>;
    type DeltaLengthByteArray = Unsupported<BooleanBuilder>;
    type Rle = BooleanRleDecoder;

    type ContiguousDict = BooleanDict;
    type ScatteredDict = BooleanDict;
}

/// Ready-to-use column decoder for BOOLEAN leaves.
pub type BooleanLeafDecoder = TypedLeafDecoder<BooleanLeaf>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reading::decoding::leaf_decoders::LeafDecoder;
    use crate::test_utils::dummy_metadata;
    use crate::thrift::general::Encoding;
    use crate::thrift::headers::PageHeader;
    use crate::types::filter_mask::FilterMask;
    use crate::types::metadata::RowSelection;
    use crate::types::page::{DataPage, DecompressedPage, DecompressedPageType};
    use arrow_array::BooleanArray;
    use dispatch::memory::init_test_free_pool;

    fn data_page(
        data: Vec<Bytes>,
        num_values: usize,
        encoding: Encoding,
        filter_mask: Option<FilterMask>,
    ) -> DecompressedPage {
        let header = PageHeader::for_data_page(num_values as i32, encoding);
        DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: dummy_metadata(RowSelection::All),
            column_idx: 0,
            idx: 0,
            first_row: 0,
            data: DecompressedPageType::Data(DataPage {
                header: header.data_page_header.unwrap(),
                data,
                filter_mask,
            }),
        }
    }

    /// PLAIN booleans: one bit per value, least significant bit first.
    fn pack_bits(values: &[bool]) -> Vec<u8> {
        let mut bytes = vec![0u8; values.len().div_ceil(8)];
        for (i, &value) in values.iter().enumerate() {
            bytes[i / 8] |= (value as u8) << (i % 8);
        }
        bytes
    }

    fn pattern(count: usize) -> Vec<bool> {
        (0..count).map(|i| i % 3 == 0 || i % 7 == 0).collect()
    }

    fn values_of(array: &ArrayRef) -> Vec<bool> {
        array
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap()
            .iter()
            .map(|value| value.unwrap())
            .collect()
    }

    #[test]
    fn plain_page_reads_in_pieces_that_split_bytes() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let values = pattern(45);
        let mut decoder = BooleanLeafDecoder::new(0);
        decoder.insert_page(data_page(
            vec![Bytes::from(pack_bits(&values))],
            45,
            Encoding::PLAIN,
            None,
        ));

        let first = values_of(&decoder.read(&mut allocator, 5).unwrap());
        let second = values_of(&decoder.read(&mut allocator, 21).unwrap());
        let third = values_of(&decoder.read(&mut allocator, 19).unwrap());

        assert_eq!([first, second, third].concat(), values);
    }

    #[test]
    fn plain_page_split_across_buffers_reads_as_one() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let values = pattern(80);
        let packed = pack_bits(&values);
        let mut decoder = BooleanLeafDecoder::new(0);
        decoder.insert_page(data_page(
            vec![
                Bytes::copy_from_slice(&packed[..3]),
                Bytes::copy_from_slice(&packed[3..7]),
                Bytes::copy_from_slice(&packed[7..]),
            ],
            80,
            Encoding::PLAIN,
            None,
        ));

        let first = values_of(&decoder.read(&mut allocator, 30).unwrap());
        let second = values_of(&decoder.read(&mut allocator, 50).unwrap());

        assert_eq!([first, second].concat(), values);
    }

    #[test]
    fn plain_page_skips_filtered_rows_mid_byte() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let values = pattern(40);
        let kept = [3u32, 4, 5, 13, 30, 31, 39];
        let mask = FilterMask::new(0, 40, &kept);
        let mut decoder = BooleanLeafDecoder::new(0);
        decoder.insert_page(data_page(
            vec![Bytes::from(pack_bits(&values))],
            40,
            Encoding::PLAIN,
            Some(mask),
        ));

        let read = values_of(&decoder.read(&mut allocator, kept.len()).unwrap());

        let expected: Vec<bool> = kept.iter().map(|&row| values[row as usize]).collect();
        assert_eq!(read, expected);
    }

    #[test]
    fn rle_page_reads_runs_and_packed_groups() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        // A bit-packed group of eight values, then a run of five `true`s, then
        // two more packed groups, behind the stream's byte length.
        let first_group = pattern(8);
        let last_groups = pattern(16);
        let mut stream = vec![(1 << 1) | 1];
        stream.extend(pack_bits(&first_group));
        stream.extend([5 << 1, 1]);
        stream.push((2 << 1) | 1);
        stream.extend(pack_bits(&last_groups));
        let mut page = (stream.len() as u32).to_le_bytes().to_vec();
        page.extend(stream);
        let mut decoder = BooleanLeafDecoder::new(0);
        decoder.insert_page(data_page(vec![Bytes::from(page)], 29, Encoding::RLE, None));

        let first = values_of(&decoder.read(&mut allocator, 11).unwrap());
        let second = values_of(&decoder.read(&mut allocator, 18).unwrap());

        let expected = [first_group, vec![true; 5], last_groups].concat();
        assert_eq!([first, second].concat(), expected);
    }

    #[test]
    fn rle_page_skips_filtered_rows_inside_runs() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        // Eight packed values, nine `false`s, eight packed values.
        let packed = pattern(8);
        let mut stream = vec![(1 << 1) | 1];
        stream.extend(pack_bits(&packed));
        stream.extend([9 << 1, 0]);
        stream.push((1 << 1) | 1);
        stream.extend(pack_bits(&packed));
        let mut page = (stream.len() as u32).to_le_bytes().to_vec();
        page.extend(stream);
        let values = [packed.clone(), vec![false; 9], packed].concat();
        let kept = [2u32, 7, 8, 16, 17, 24];
        let mask = FilterMask::new(0, 25, &kept);
        let mut decoder = BooleanLeafDecoder::new(0);
        decoder.insert_page(data_page(
            vec![Bytes::from(page)],
            25,
            Encoding::RLE,
            Some(mask),
        ));

        let read = values_of(&decoder.read(&mut allocator, kept.len()).unwrap());

        let expected: Vec<bool> = kept.iter().map(|&row| values[row as usize]).collect();
        assert_eq!(read, expected);
    }
}
