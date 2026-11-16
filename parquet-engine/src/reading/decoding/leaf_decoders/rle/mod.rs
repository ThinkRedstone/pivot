//! RLE/bit-packed hybrid decoder for Parquet dictionary indices.
//!
//! Dictionary-encoded columns store row values as integer indices into a
//! dictionary. These indices are encoded with Parquet's RLE/bit-packed hybrid
//! scheme: a stream of *runs*, each either:
//!
//! - **RLE** — a single value repeated `count` times.
//! - **Bit-packed** — `num_groups × 8` values packed at `bit_width` bits each,
//!   decoded by [`BitPackDecoder`].
//!
//! [`RleDecoder`] is the stateful entry point: it lazily parses run headers and
//! delegates to [`Run::read_into`] / [`Run::skip`]. Partially consumed runs are
//! saved and resumed on the next call, so the decoder can be driven in
//! arbitrarily sized chunks.

use crate::reading::decoding::leaf_decoders::{ArrayBuilder, Dict};
use bytes::Bytes;
use dispatch::memory::{MultiBufferReader, ReaderPosition};

mod bit_pack_decoder;
pub use bit_pack_decoder::{BitDecoderOverflow, BitPackDecoder};

/// Writes `dict`'s entry for each key into `dest` (same length).
///
/// Kept out of line: with its own small frame the lookup loop unrolls,
/// while inlined into the decode body it competes for registers with
/// everything else and stays scalar.
///
/// # Safety
///
/// Every key must be `< dict.len()`.
#[inline(never)]
unsafe fn gather_entries<D: Dict>(dict: &D, keys: &[u32], dest: &mut [D::Item]) {
    for (d, &k) in dest.iter_mut().zip(keys) {
        *d = unsafe { dict.entry_unchecked(k as usize) };
    }
}

/// Appends `dict`'s entry for each of `keys` to `builder`.
///
/// Keys are bounded by the bit width; when the dictionary is at least that
/// large (`keys_in_range`) no key can be out of range. Otherwise one max-scan
/// (auto-vectorized) checks all of `keys` up front: slab-backed dictionaries
/// index with no per-element bounds check, so out-of-range keys must be
/// impossible by the time the lookup loop runs.
#[inline(always)]
fn push_entries_for_keys<I, B: ArrayBuilder<Element = I>, D: Dict<Builder = B, Item = I>>(
    builder: &mut B,
    dict: &D,
    keys: &[u32],
    keys_in_range: bool,
) {
    if !keys_in_range {
        // Folding from zero, rather than from the first key, leaves a count
        // that is a whole number of vectors whenever the key count is.
        let max = keys.iter().copied().fold(0, u32::max);
        assert!(
            (max as usize) < dict.len(),
            "dictionary key {max} out of range ({} entries)",
            dict.len()
        );
    }
    let dest = builder.spare_mut(keys.len());
    // SAFETY: every key is < dict.len() - by the bit-width bound or the
    // max-scan assert above.
    unsafe { gather_entries(dict, keys, dest) };
}

/// Bytes a run takes up to the end of its repeated value when its header
/// fits one byte: the header, then at most four bytes of value.
const SHORT_RUN_PREFIX: usize = 1 + size_of::<u32>();

/// Slots written at a time by [`fill_run`]. Writers rarely emit a repeated
/// run shorter than this, so nearly every run is at least one block long.
const FILL_BLOCK: usize = 8;

/// Sets every slot of `dest` to `value`.
///
/// Runs are short, so the fill is shaped for them: whole blocks from the
/// front, and one block flush with the end that overlaps the block before it
/// instead of finishing slot by slot.
#[inline(always)]
fn fill_run<T: Copy>(dest: &mut [T], value: T) {
    let block = [value; FILL_BLOCK];
    let Some(last) = dest.last_chunk_mut::<FILL_BLOCK>() else {
        dest.fill(value);
        return;
    };
    *last = block;
    for chunk in dest.as_chunks_mut::<FILL_BLOCK>().0 {
        *chunk = block;
    }
}

/// A single run in the RLE/bit-packed stream.
pub enum Run {
    Rle {
        value: u32,
        count: usize,
    },
    BitPacked {
        remaining_in_run: usize,
        partial: [u32; 7],
        partial_count: u8,
    },
}

/// Reads one group of 8 bit-packed values that straddles a buffer boundary.
///
/// Uses [`MultiBufferReader`] to gather `bit_width` bytes across buffers,
/// decodes the full group, and returns all 8 values.
#[inline(always)]
fn decode_cross_boundary_group(
    data: &[Bytes],
    bit_width: u8,
    position: &mut ReaderPosition,
) -> [u32; 8] {
    let mut reader = MultiBufferReader::new(data, position);
    let mut group_bytes = [0u8; 32];
    for b in group_bytes.iter_mut().take(bit_width as usize) {
        *b = reader.read_u8();
    }
    let mut group = [0u32; 8];
    BitPackDecoder::new(&group_bytes[..bit_width as usize], 0, bit_width, [0; 7], 0)
        .decode(&mut group)
        .unwrap();
    group
}

/// Builds partial state from the unused tail of a decoded group.
///
/// `partial[7-count..7]` holds the values, matching the layout expected by
/// [`BitPackDecoder`].
#[inline(always)]
fn partial_from_group(group: &[u32; 8], used: usize) -> ([u32; 7], u8) {
    let count = (8 - used) as u8;
    let mut partial = [0u32; 7];
    for j in 0..count as usize {
        partial[7 - count as usize + j] = group[used + j];
    }
    (partial, count)
}

/// Decodes up to `limit` values from a bit-packed run, processing them in
/// chunks of up to 1024 via the `on_chunk` callback.
///
/// Handles data spanning multiple [`Bytes`] buffers: when the current buffer
/// runs out mid-group, [`decode_cross_boundary_group`] stitches the group
/// across the boundary. Any unused values from that group are carried forward
/// as partial state on the new [`BitPackDecoder`].
///
/// # Returns
///
/// `(values_read, leftover_run)` — the count of values consumed and an
/// optional [`Run::BitPacked`] for the unconsumed tail of the run.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn decode_bitpacked(
    remaining_in_run: usize,
    partial: [u32; 7],
    partial_count: u8,
    scratch: &mut [u32; 1024],
    data: &[Bytes],
    bit_width: u8,
    position: &mut ReaderPosition,
    limit: usize,
    mut on_chunk: impl FnMut(&mut [u32; 1024], usize),
) -> (usize, Option<Run>) {
    let to_read = remaining_in_run.min(limit);
    let mut remaining = to_read;

    let mut decoder = BitPackDecoder::new(
        &data[position.buffer_index],
        position.offset,
        bit_width,
        partial,
        partial_count,
    );

    while remaining > 0 {
        let chunk = remaining.min(1024);
        let count = match decoder.decode(&mut scratch[..chunk]) {
            Ok(n) => n,
            Err(BitDecoderOverflow(n)) => {
                position.offset = decoder.pos();
                let group = decode_cross_boundary_group(data, bit_width, position);

                let take = (chunk - n).min(8);
                scratch[n..n + take].copy_from_slice(&group[..take]);

                // Carry unused group values as partial state so they're
                // emitted on the next iteration.
                let (new_partial, new_partial_count) = partial_from_group(&group, take);
                decoder = BitPackDecoder::new(
                    &data[position.buffer_index],
                    position.offset,
                    bit_width,
                    new_partial,
                    new_partial_count,
                );
                n + take
            }
        };

        on_chunk(scratch, count);
        remaining -= count;
    }

    position.offset = decoder.pos();

    let leftover = if remaining_in_run > limit {
        Some(Run::BitPacked {
            remaining_in_run: remaining_in_run - limit,
            partial: *decoder.partial_values(),
            partial_count: decoder.partial_count(),
        })
    } else {
        None
    };

    (to_read, leftover)
}

impl Run {
    /// Parses the next run header from `reader` and returns the run.
    ///
    /// The header's LSB distinguishes RLE (0) from bit-packed (1). For RLE
    /// runs the repeated value follows; for bit-packed runs only the group
    /// count is recorded — actual decoding happens in [`read_into`](Self::read_into).
    ///
    /// Always inlined, because it runs once per run inside the decode loop and
    /// the compiler otherwise makes it a call in some builds.
    #[inline(always)]
    pub fn parse_from_header(bit_width: u8, reader: &mut MultiBufferReader) -> Self {
        let header = reader.read_varint();
        if header & 1 == 0 {
            // RLE run
            let count = (header >> 1) as usize;
            let byte_width = byte_width(bit_width);
            let mut value: u32 = 0;
            for i in 0..byte_width {
                value |= (reader.read_u8() as u32) << (i * 8);
            }
            Self::Rle { value, count }
        } else {
            // Bit-packed run
            let num_groups = (header >> 1) as usize;
            Self::BitPacked {
                remaining_in_run: num_groups * 8,
                partial: [0; 7],
                partial_count: 0,
            }
        }
    }

    /// Decodes up to `limit` values from this run, looking up each index in
    /// `dict` and pushing the result into `builder`.
    ///
    /// Returns `Some(run)` with the remaining portion if the run was not
    /// fully consumed, or `None` if it's exhausted.
    #[allow(clippy::too_many_arguments)]
    pub fn read_into<I, B: ArrayBuilder<Element = I>, D: Dict<Builder = B, Item = I>>(
        self,
        builder: &mut B,
        scratch: &mut [u32; 1024],
        data: &[Bytes],
        bit_width: u8,
        position: &mut ReaderPosition,
        dict: &D,
        limit: usize,
    ) -> Option<Self> {
        match self {
            Run::Rle { value, mut count } => {
                let emit = count.min(limit);
                // A run's value fills whole bytes, so it can exceed both the
                // bit width and the dictionary, and slab-backed dictionaries
                // index with no bounds check of their own.
                assert!(
                    (value as usize) < dict.len(),
                    "dictionary key {value} out of range ({} entries)",
                    dict.len()
                );
                let entry = dict.entry(value as usize);
                builder.push(&entry, emit);
                count -= emit;
                if count > 0 {
                    Some(Run::Rle { value, count })
                } else {
                    None
                }
            }
            Run::BitPacked {
                remaining_in_run,
                partial,
                partial_count,
            } => {
                let keys_in_range = (1u64 << bit_width) as usize <= dict.len();
                let (_, leftover) = decode_bitpacked(
                    remaining_in_run,
                    partial,
                    partial_count,
                    scratch,
                    data,
                    bit_width,
                    position,
                    limit,
                    |scratch, count| {
                        push_entries_for_keys(builder, dict, &scratch[..count], keys_in_range)
                    },
                );
                leftover
            }
        }
    }

    /// Advances past up to `amount` values without writing them.
    ///
    /// Returns `(skipped, remaining_run)` — the number of values actually
    /// skipped and the leftover run (if any).
    pub fn skip(
        self,
        data: &[Bytes],
        bit_width: u8,
        position: &mut ReaderPosition,
        amount: usize,
    ) -> (usize, Option<Run>) {
        match self {
            Run::Rle { value, mut count } => {
                let emit = count.min(amount);
                count -= emit;
                if count > 0 {
                    (emit, Some(Run::Rle { value, count }))
                } else {
                    (emit, None)
                }
            }
            Run::BitPacked {
                remaining_in_run,
                partial,
                mut partial_count,
            } => {
                let skipped = amount.min(remaining_in_run);
                let mut left = skipped;
                // Values already unpacked into the partial group sit at the
                // end of the array, so consuming them from the front only
                // shortens the count.
                let from_partial = (partial_count as usize).min(left);
                partial_count -= from_partial as u8;
                left -= from_partial;
                // Whole groups are `bit_width` bytes each and need no
                // unpacking to step over.
                let groups = left / 8;
                if groups > 0 {
                    MultiBufferReader::new(data, position).skip(groups * bit_width as usize);
                    left -= groups * 8;
                }
                // The values past the last whole group come from one more
                // group, unpacked so its unconsumed tail carries over.
                let (partial, partial_count) = if left > 0 {
                    let group = decode_cross_boundary_group(data, bit_width, position);
                    partial_from_group(&group, left)
                } else {
                    (partial, partial_count)
                };
                let leftover = (remaining_in_run > skipped).then_some(Run::BitPacked {
                    remaining_in_run: remaining_in_run - skipped,
                    partial,
                    partial_count,
                });
                (skipped, leftover)
            }
        }
    }
}

/// Minimum number of bytes needed to hold a value of `bit_width` bits.
const fn byte_width(bit_width: u8) -> u8 {
    bit_width.div_ceil(8)
}

/// Stateful decoder for Parquet's RLE/bit-packed hybrid encoding.
///
/// Lazily parses [`Run`] headers on demand and supports incremental
/// consumption — partially consumed runs are stashed in `run` and resumed on
/// the next `read` or `skip` call.
pub struct RleDecoder {
    position: ReaderPosition,
    bit_width: u8,
    data: Vec<Bytes>,
    /// Scratch buffer for bit-pack decoding (avoids per-call allocation).
    buffer: Box<[u32; 1024]>,
    /// The in-progress run, if any.
    run: Option<Run>,
}

impl RleDecoder {
    pub fn new(data: Vec<Bytes>, position: ReaderPosition, bit_width: u8) -> Self {
        Self {
            position,
            bit_width,
            data,
            buffer: Box::new([0; 1024]),
            run: None,
        }
    }

    /// Returns the current in-progress run, or parses the next one from the
    /// stream.
    #[inline(always)]
    fn get_or_set_next_run(&mut self) -> Run {
        match self.run.take() {
            Some(r) => r,
            None => Run::parse_from_header(
                self.bit_width,
                &mut MultiBufferReader::new(&self.data, &mut self.position),
            ),
        }
    }

    /// Advances past `size` values without producing output.
    pub fn skip(&mut self, mut size: usize) {
        while size > 0 {
            let run = self.get_or_set_next_run();
            let (skipped, run) = run.skip(&self.data, self.bit_width, &mut self.position, size);
            self.run = run;
            size -= skipped;
        }
    }

    /// Decodes runs straight out of the current buffer for as long as each
    /// one is short and whole: its header fits one byte, all of it lies in
    /// this buffer, and it holds no more values than `builder` still needs
    /// to reach `target` elements.
    ///
    /// Most runs of a page are like that and a page holds very many of them,
    /// so this loop keeps the read offset and the count of values still
    /// wanted in locals and writes the decoder's position once, on the way
    /// out. It stops at the first run that is anything else, leaving the
    /// position on that run's header.
    ///
    /// No run may be pending when this is called.
    fn read_short_runs<B: ArrayBuilder, D: Dict<Builder = B, Item = B::Element>>(
        &mut self,
        builder: &mut B,
        dict: &D,
        target: usize,
    ) {
        // A position past the last buffer has no run to offer here.
        let Some(buffer) = self.data.get(self.position.buffer_index) else {
            return;
        };
        let bytes: &[u8] = buffer;
        let value_bytes = byte_width(self.bit_width) as usize;
        let value_mask = ((1u64 << (8 * value_bytes)) - 1) as u32;
        let entries = dict.len();
        let keys_in_range = (1u64 << self.bit_width) as usize <= entries;
        let mut offset = self.position.offset;
        let mut wanted = target - builder.len();

        while wanted > 0 {
            // The header and the widest repeated value are read together, so
            // a run within a few bytes of the buffer's end is not taken here
            // even when its own bytes all fit.
            let Some(&[header, value @ ..]) = bytes
                .get(offset..)
                .and_then(<[u8]>::first_chunk::<SHORT_RUN_PREFIX>)
            else {
                break;
            };
            if header & 0x80 != 0 {
                break;
            }
            let count = usize::from(header >> 1);
            if header & 1 == 0 {
                if count > wanted {
                    break;
                }
                let key = u32::from_le_bytes(value) & value_mask;
                // The value fills whole bytes, so it can exceed both the bit
                // width and the dictionary, and slab-backed dictionaries
                // index with no bounds check of their own.
                assert!(
                    (key as usize) < entries,
                    "dictionary key {key} out of range ({entries} entries)"
                );
                fill_run(builder.spare_mut(count), dict.entry(key as usize));
                offset += 1 + value_bytes;
                wanted -= count;
            } else {
                let values = count * 8;
                let packed_end = offset + 1 + count * usize::from(self.bit_width);
                if values > wanted || packed_end > bytes.len() {
                    break;
                }
                let keys = &mut self.buffer[..values];
                BitPackDecoder::new(bytes, offset + 1, self.bit_width, [0; 7], 0)
                    .decode(keys)
                    .expect("every group of the run lies inside the buffer");
                push_entries_for_keys(builder, dict, keys, keys_in_range);
                offset = packed_end;
                wanted -= values;
            }
        }

        self.position.offset = offset;
    }

    /// Decodes `size` values, looking up each index in `dict` and pushing
    /// the result into `builder`.
    pub fn read<I, B: ArrayBuilder<Element = I>, D: Dict<Builder = B, Item = I>>(
        &mut self,
        builder: &mut B,
        dict: &D,
        size: usize,
    ) {
        let target = builder.len() + size;
        while builder.len() < target {
            if self.run.is_none() {
                self.read_short_runs(builder, dict, target);
                if builder.len() == target {
                    break;
                }
            }
            let size = target - builder.len();
            let run = self.get_or_set_next_run();
            self.run = run.read_into(
                builder,
                &mut self.buffer,
                &self.data,
                self.bit_width,
                &mut self.position,
                dict,
                size,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reading::decoding::leaf_decoders::ArrayBuilder;
    use crate::reading::decoding::leaf_decoders::Dict;
    use crate::reading::decoding::leaf_decoders::bytes_view::dict::{DictFactory, ViewDict};
    use arrow_array::types::StringViewType;
    use arrow_array::{Array, StringViewArray};
    use bytes::Bytes;
    use dispatch::arrays::ViewBuilder;
    use dispatch::memory::SlabAllocator;
    use dispatch::memory::init_test_free_pool;

    fn encode_plain_strings(strings: &[&str]) -> Vec<u8> {
        let mut data = Vec::new();
        for s in strings {
            data.extend_from_slice(&(s.len() as u32).to_le_bytes());
            data.extend_from_slice(s.as_bytes());
        }
        data
    }

    fn make_dict(entries: &[&str]) -> ViewDict<StringViewType> {
        let data = vec![Bytes::from(encode_plain_strings(entries))];
        DictFactory::new(data, entries.len()).create_dict()
    }

    fn make_data(data: Vec<u8>) -> Vec<Bytes> {
        vec![Bytes::from(data)]
    }

    fn make_data_multi_buffer(buffers: Vec<Vec<u8>>) -> Vec<Bytes> {
        buffers.into_iter().map(Bytes::from).collect()
    }

    fn extract_strings(builder: ViewBuilder<StringViewType>) -> Vec<String> {
        let array = builder.into_array(None);
        let sv = array.as_any().downcast_ref::<StringViewArray>().unwrap();
        (0..sv.len()).map(|i| sv.value(i).to_string()).collect()
    }

    fn new_decoder(data: Vec<u8>, bit_width: u8) -> RleDecoder {
        RleDecoder::new(make_data(data), ReaderPosition::default(), bit_width)
    }

    fn push_all(
        allocator: &mut SlabAllocator,
        decoder: &mut RleDecoder,
        dict: &ViewDict<StringViewType>,
        size: usize,
    ) -> Vec<String> {
        let mut buf = ViewBuilder::with_capacity(allocator, size);
        dict.register_onto(&mut buf);
        decoder.read(&mut buf, dict, size);
        extract_strings(buf)
    }

    #[test]
    fn test_rle_run() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let dict = make_dict(&["AA", "BB", "CC"]);
        let mut dec = new_decoder(vec![10, 2], 2);

        let result = push_all(&mut allocator, &mut dec, &dict, 5);

        assert_eq!(result, vec!["CC"; 5]);
    }

    #[test]
    fn test_rle_incremental() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let dict = make_dict(&["X", "Y"]);
        let mut dec = new_decoder(vec![12, 1], 1);

        let r1 = push_all(&mut allocator, &mut dec, &dict, 2);
        let r2 = push_all(&mut allocator, &mut dec, &dict, 2);
        let r3 = push_all(&mut allocator, &mut dec, &dict, 2);

        assert_eq!(r1, vec!["Y"; 2]);
        assert_eq!(r2, vec!["Y"; 2]);
        assert_eq!(r3, vec!["Y"; 2]);
    }

    #[test]
    fn test_bitpacked_single_group() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let dict = make_dict(&["AA", "BB", "CC", "DD"]);
        let mut dec = new_decoder(vec![3, 0xE4, 0xE4], 2);

        let result = push_all(&mut allocator, &mut dec, &dict, 8);

        assert_eq!(result, vec!["AA", "BB", "CC", "DD", "AA", "BB", "CC", "DD"]);
    }

    #[test]
    fn test_bitpacked_incremental() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let dict = make_dict(&["AA", "BB", "CC", "DD"]);
        let mut dec = new_decoder(vec![3, 0xE4, 0xE4], 2);

        let r1 = push_all(&mut allocator, &mut dec, &dict, 3);
        let r2 = push_all(&mut allocator, &mut dec, &dict, 5);

        assert_eq!(r1, vec!["AA", "BB", "CC"]);
        assert_eq!(r2, vec!["DD", "AA", "BB", "CC", "DD"]);
    }

    #[test]
    fn test_rle_then_bitpacked() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let dict = make_dict(&["AA", "BB", "CC", "DD"]);
        let mut dec = new_decoder(vec![4, 0, 3, 0xE4, 0xE4], 2);

        let r1 = push_all(&mut allocator, &mut dec, &dict, 2);
        assert_eq!(r1, vec!["AA", "AA"]);

        let r2 = push_all(&mut allocator, &mut dec, &dict, 8);
        assert_eq!(r2, vec!["AA", "BB", "CC", "DD", "AA", "BB", "CC", "DD"]);
    }

    #[test]
    fn test_bitpacked_cross_buffer_boundary() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let dict = make_dict(&["AA", "BB", "CC", "DD"]);
        let data = make_data_multi_buffer(vec![vec![3], vec![0xE4, 0xE4]]);
        let mut dec = RleDecoder::new(data, ReaderPosition::default(), 2);

        let result = push_all(&mut allocator, &mut dec, &dict, 8);

        assert_eq!(result, vec!["AA", "BB", "CC", "DD", "AA", "BB", "CC", "DD"]);
    }

    /// Repeated and bit-packed runs: 5 x CC, one group, 3 x BB, one group,
    /// 2 x AA. Every run but the last has room after it in the buffer.
    const MIXED_SHORT_RUNS: [u8; 12] = [10, 2, 3, 0xE4, 0xE4, 6, 1, 3, 0xE4, 0xE4, 4, 0];

    fn mixed_short_runs_values() -> Vec<&'static str> {
        let group = ["AA", "BB", "CC", "DD", "AA", "BB", "CC", "DD"];
        [&["CC"; 5][..], &group, &["BB"; 3], &group, &["AA"; 2]].concat()
    }

    #[test]
    fn reads_consecutive_short_runs() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let dict = make_dict(&["AA", "BB", "CC", "DD"]);
        let mut dec = new_decoder(MIXED_SHORT_RUNS.to_vec(), 2);

        let result = push_all(&mut allocator, &mut dec, &dict, 26);

        assert_eq!(result, mixed_short_runs_values());
    }

    /// Reads that end inside a repeated run and inside a bit-packed group
    /// pick up where they stopped.
    #[test]
    fn short_runs_split_across_reads() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let dict = make_dict(&["AA", "BB", "CC", "DD"]);
        let mut dec = new_decoder(MIXED_SHORT_RUNS.to_vec(), 2);

        let mut result = Vec::new();
        for size in [3, 4, 9, 10] {
            result.extend(push_all(&mut allocator, &mut dec, &dict, size));
        }

        assert_eq!(result, mixed_short_runs_values());
    }

    /// Repeated runs longer than one fill block, neither a whole number of
    /// blocks: 13 x BB then 37 x DD.
    #[test]
    fn fills_repeated_runs_longer_than_a_block() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let dict = make_dict(&["AA", "BB", "CC", "DD"]);
        let mut dec = new_decoder(vec![26, 1, 74, 3, 3, 0xE4, 0xE4], 2);

        let result = push_all(&mut allocator, &mut dec, &dict, 58);

        let group = ["AA", "BB", "CC", "DD", "AA", "BB", "CC", "DD"];
        let expected = [&["BB"; 13][..], &["DD"; 37], &group].concat();
        assert_eq!(result, expected);
    }

    /// A run whose header takes two bytes (70 x BB) sits between short runs.
    #[test]
    fn long_run_between_short_runs() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let dict = make_dict(&["AA", "BB", "CC", "DD"]);
        let data = vec![4, 3, 0x8C, 0x01, 1, 3, 0xE4, 0xE4, 6, 2, 3, 0xE4, 0xE4];
        let mut dec = new_decoder(data, 2);

        let result = push_all(&mut allocator, &mut dec, &dict, 91);

        let group = ["AA", "BB", "CC", "DD", "AA", "BB", "CC", "DD"];
        let expected = [&["DD"; 2][..], &["BB"; 70], &group, &["CC"; 3], &group].concat();
        assert_eq!(result, expected);
    }

    /// A bit-packed run of two groups whose last byte is the buffer's last.
    #[test]
    fn reads_bitpacked_run_ending_at_buffer_end() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let dict = make_dict(&["AA", "BB", "CC", "DD"]);
        let mut dec = new_decoder(vec![5, 0xE4, 0xE4, 0xE4, 0xE4], 2);

        let result = push_all(&mut allocator, &mut dec, &dict, 16);

        let expected: Vec<&str> = ["AA", "BB", "CC", "DD"]
            .iter()
            .copied()
            .cycle()
            .take(16)
            .collect();
        assert_eq!(result, expected);
    }

    /// Short runs in one buffer, a bit-packed run of two groups that starts
    /// in it and ends in the next, then short runs in that next buffer.
    #[test]
    fn short_runs_around_a_run_split_across_buffers() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let dict = make_dict(&["AA", "BB", "CC", "DD"]);
        let data = make_data_multi_buffer(vec![
            vec![10, 2, 3, 0xE4, 0xE4, 6, 1, 5, 0xE4],
            vec![0xE4, 0xE4, 0xE4, 4, 0, 3, 0xE4, 0xE4, 6, 3, 4, 1],
        ]);
        let mut dec = RleDecoder::new(data, ReaderPosition::default(), 2);

        let result = push_all(&mut allocator, &mut dec, &dict, 47);

        let group = ["AA", "BB", "CC", "DD", "AA", "BB", "CC", "DD"];
        let expected = [
            &["CC"; 5][..],
            &group,
            &["BB"; 3],
            &group,
            &group,
            &["AA"; 2],
            &group,
            &["DD"; 3],
            &["BB"; 2],
        ]
        .concat();
        assert_eq!(result, expected);
    }

    /// At a bit width of 12 a repeated run's value takes two bytes: keys 258
    /// and 299 need both, and each value is followed directly by the next
    /// run's header.
    #[test]
    fn reads_repeated_runs_with_two_byte_values() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let names: Vec<String> = (0..300).map(|key| format!("v{key:03}")).collect();
        let dict = make_dict(&names.iter().map(String::as_str).collect::<Vec<_>>());
        let data = vec![10, 0x02, 0x01, 6, 0x2B, 0x01, 4, 0, 0, 2, 5, 0];
        let mut dec = new_decoder(data, 12);

        let result = push_all(&mut allocator, &mut dec, &dict, 11);

        let expected = [&["v258"; 5][..], &["v299"; 3], &["v000"; 2], &["v005"; 1]].concat();
        assert_eq!(result, expected);
    }

    /// A repeated run's value is a whole byte even at a bit width of 2, so it
    /// can name a key the dictionary does not hold. Here the run has room
    /// after it in the buffer.
    #[test]
    #[should_panic(expected = "dictionary key 255 out of range (4 entries)")]
    fn rejects_repeated_run_key_past_dictionary() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let dict = make_dict(&["AA", "BB", "CC", "DD"]);
        let mut dec = new_decoder(vec![10, 0xFF, 4, 0, 0], 2);

        push_all(&mut allocator, &mut dec, &dict, 5);
    }

    /// The same out-of-range key in a run that ends the buffer.
    #[test]
    #[should_panic(expected = "dictionary key 255 out of range (4 entries)")]
    fn rejects_repeated_run_key_past_dictionary_at_buffer_end() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let dict = make_dict(&["AA", "BB", "CC", "DD"]);
        let mut dec = new_decoder(vec![10, 0xFF], 2);

        push_all(&mut allocator, &mut dec, &dict, 5);
    }

    /// Skipping into the middle of a bit-packed run leaves the decoder at the
    /// right value, whether the skip ends inside a group or on a group edge,
    /// and whether it starts with unpacked values pending.
    #[test]
    fn skip_lands_inside_bitpacked_run() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let dict = make_dict(&["AA", "BB", "CC", "DD"]);
        // Header 0x07 = 3 bit-packed groups of AA BB CC DD AA BB CC DD.
        let mut dec = new_decoder(vec![0x07, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4, 0xE4], 2);

        let head = push_all(&mut allocator, &mut dec, &dict, 3);
        dec.skip(11);
        let middle = push_all(&mut allocator, &mut dec, &dict, 2);
        dec.skip(4);
        let tail = push_all(&mut allocator, &mut dec, &dict, 4);

        assert_eq!(head, vec!["AA", "BB", "CC"]);
        assert_eq!(middle, vec!["CC", "DD"]);
        assert_eq!(tail, vec!["AA", "BB", "CC", "DD"]);
    }

    /// Regression: after handling a buffer-boundary overflow, the decoder must
    /// not re-enter the overflow path on subsequent successful decodes.
    #[test]
    fn test_bitpacked_cross_buffer_with_remaining_groups() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let dict = make_dict(&["AA", "BB", "CC", "DD"]);
        // Header 0x07 = 3 bit-packed groups (24 values). Split so group 1
        // straddles the buffer boundary, forcing an overflow. Groups 2-3
        // are fully in buffer 1.
        let data =
            make_data_multi_buffer(vec![vec![0x07, 0xE4], vec![0xE4, 0xE4, 0xE4, 0xE4, 0xE4]]);
        let mut dec = RleDecoder::new(data, ReaderPosition::default(), 2);

        let result = push_all(&mut allocator, &mut dec, &dict, 24);

        let expected: Vec<&str> = ["AA", "BB", "CC", "DD"]
            .iter()
            .copied()
            .cycle()
            .take(24)
            .collect();
        assert_eq!(result, expected);
    }
}
