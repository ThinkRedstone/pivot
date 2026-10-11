//! Column decoder for fixed-width primitive types (integers, floats).
//!
//! Parquet stores primitives as little-endian bytes with a fixed physical
//! width. Some Arrow types are narrower than their Parquet physical type
//! (e.g. `Int16` is stored as 4-byte `INT32`), so [`ReadLeBytes`] abstracts
//! over the on-disk width vs. the in-memory width.
//!
//! The public type alias [`PrimitiveLeafDecoder`] wires everything together
//! into a ready-to-use [`TypedLeafDecoder`].

use std::marker::PhantomData;
use std::mem;
use std::ops::Index;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{ArrowPrimitiveType, TimestampMicrosecondType};
use arrow_array::{ArrayRef, RecordBatch, Scalar, TimestampMicrosecondArray};
use arrow_buffer::ArrowNativeType;

use super::{BuiltDictionary, SharedDictionary};
use crate::reading::decoding::leaf_decoders::{
    DecodePlain, DeltaDecoder, Dict, DictFromBytes, DictFromVecBytes, FromDelta, LeafDecoder,
    Result, TypedLeafDecoder,
};
use crate::thrift::headers::DictionaryPageHeader;
use crate::types::page::DecompressedPage;
use bytes::Bytes;
use dispatch::memory::{
    MultiBufferReader, MultiSlabBuffer, ReaderPosition, SlabAllocator, SlabBuffer,
};

/// Reads a single value from a [`MultiBufferReader`] in little-endian byte
/// order.
///
/// `PHYSICAL_SIZE` is the number of bytes the value occupies on disk (may
/// differ from `size_of::<Self>()` for narrowed types like `i16`).
pub trait ReadLeBytes: ArrowNativeType {
    const PHYSICAL_SIZE: usize;

    fn read_le(reader: &mut MultiBufferReader) -> Self;
}

impl ReadLeBytes for i8 {
    const PHYSICAL_SIZE: usize = 4;

    #[inline(always)]
    fn read_le(reader: &mut MultiBufferReader) -> Self {
        reader.read_u32_le() as i8
    }
}

impl ReadLeBytes for u8 {
    const PHYSICAL_SIZE: usize = 4;

    #[inline(always)]
    fn read_le(reader: &mut MultiBufferReader) -> Self {
        reader.read_u32_le() as u8
    }
}

impl ReadLeBytes for i16 {
    const PHYSICAL_SIZE: usize = 4;

    #[inline(always)]
    fn read_le(reader: &mut MultiBufferReader) -> Self {
        reader.read_u32_le() as i16
    }
}

impl ReadLeBytes for u16 {
    const PHYSICAL_SIZE: usize = 4;

    #[inline(always)]
    fn read_le(reader: &mut MultiBufferReader) -> Self {
        reader.read_u32_le() as u16
    }
}

impl ReadLeBytes for i32 {
    const PHYSICAL_SIZE: usize = 4;

    #[inline(always)]
    fn read_le(reader: &mut MultiBufferReader) -> Self {
        reader.read_i32_le()
    }
}

impl ReadLeBytes for u32 {
    const PHYSICAL_SIZE: usize = 4;

    #[inline(always)]
    fn read_le(reader: &mut MultiBufferReader) -> Self {
        reader.read_u32_le()
    }
}

impl ReadLeBytes for i64 {
    const PHYSICAL_SIZE: usize = 8;

    #[inline(always)]
    fn read_le(reader: &mut MultiBufferReader) -> Self {
        reader.read_i64_le()
    }
}

/// An unsigned 64-bit value is stored as the INT64 holding its bits, so the
/// read is the signed one reinterpreted rather than a conversion.
impl ReadLeBytes for u64 {
    const PHYSICAL_SIZE: usize = 8;

    #[inline(always)]
    fn read_le(reader: &mut MultiBufferReader) -> Self {
        reader.read_i64_le() as u64
    }
}

impl ReadLeBytes for f32 {
    const PHYSICAL_SIZE: usize = 4;

    #[inline(always)]
    fn read_le(reader: &mut MultiBufferReader) -> Self {
        let bytes = reader.read_bytes(Self::PHYSICAL_SIZE);
        f32::from_le_bytes(bytes.try_into().unwrap())
    }
}

impl ReadLeBytes for f64 {
    const PHYSICAL_SIZE: usize = 8;

    #[inline(always)]
    fn read_le(reader: &mut MultiBufferReader) -> Self {
        let bytes = reader.read_bytes(Self::PHYSICAL_SIZE);
        f64::from_le_bytes(bytes.try_into().unwrap())
    }
}

/// Lets bulk decode loops ([`read_primitives`] here, the decimal decoder's
/// `read_decimals`) write into either backing without a per-element slab
/// lookup in the common case: the single [`SlabBuffer`] of a [`PrimitiveBuilder`]
/// or the [`MultiSlabBuffer`] of a dictionary.
pub(super) trait ElemPtr<T> {
    fn elem_ptr(&self, index: usize) -> *mut T;

    /// How many elements from `index` onwards are laid out contiguously after
    /// `elem_ptr(index)`: the most a single bulk copy may write there. Zero
    /// means the buffer is full.
    fn contiguous_elements_from(&self, index: usize) -> usize;
}

impl<T> ElemPtr<T> for SlabBuffer<T> {
    #[inline(always)]
    fn elem_ptr(&self, index: usize) -> *mut T {
        self.ptr_at_index(index)
    }

    #[inline(always)]
    fn contiguous_elements_from(&self, index: usize) -> usize {
        self.elements_left(index)
    }
}

impl<T> ElemPtr<T> for MultiSlabBuffer<T> {
    #[inline(always)]
    fn elem_ptr(&self, index: usize) -> *mut T {
        self.ptr_at_index(index)
    }

    #[inline(always)]
    fn contiguous_elements_from(&self, index: usize) -> usize {
        self.elements_left_in_slab(index)
    }
}

/// Reads `count` fixed-width LE primitive values from scattered buffers into
/// `output` (a single-slab builder buffer or a multi-slab dictionary buffer).
///
/// Fast path: when many values fit within the current buffer, copies them
/// in bulk via memcpy (no per-value overhead). A copy never crosses the end
/// of the output's current slab: a multi-slab dictionary's slabs are separate
/// allocations, so a copy sized by the input alone would run off one slab
/// into unrelated memory and leave the next slab's values unwritten.
/// Slow path: when a value straddles a buffer boundary, falls back to
/// `MultiBufferReader` for that single value, then resumes the fast path.
#[inline]
fn read_primitives<N: ReadLeBytes, B: ElemPtr<N>>(
    data: &[Bytes],
    position: &mut ReaderPosition,
    output: &mut B,
    output_len: &mut usize,
    count: usize,
) {
    let byte_width = N::PHYSICAL_SIZE;
    let target = *output_len + count;

    while *output_len < target {
        let remaining = target - *output_len;
        let buf = &data[position.buffer_index];
        let available_bytes = buf.len() - position.offset;
        let fit = available_bytes / byte_width;

        if fit > 0 && byte_width == mem::size_of::<N>() {
            // Fast path: bulk-copy all values that fit in the current input
            // buffer and the current output slab
            let to_read = remaining
                .min(fit)
                .min(output.contiguous_elements_from(*output_len));
            assert!(
                to_read > 0,
                "output buffer is full after {output_len} elements, {remaining} more requested"
            );
            let byte_count = to_read * byte_width;
            let src = &buf[position.offset..position.offset + byte_count];
            unsafe {
                let dst = output.elem_ptr(*output_len) as *mut u8;
                std::ptr::copy_nonoverlapping(src.as_ptr(), dst, byte_count);
            }
            *output_len += to_read;
            position.offset += byte_count;
        } else if fit > 0 {
            let to_read = remaining.min(fit);
            let mut reader = MultiBufferReader::new(data, position);
            for _ in 0..to_read {
                unsafe { *output.elem_ptr(*output_len) = N::read_le(&mut reader) };
                *output_len += 1;
            }
        } else if available_bytes > 0 {
            // Slow path: value straddles buffer boundary
            let mut reader = MultiBufferReader::new(data, position);
            unsafe { *output.elem_ptr(*output_len) = N::read_le(&mut reader) };
            *output_len += 1;
        } else {
            // Buffer fully consumed, advance to next
            position.buffer_index += 1;
            position.offset = 0;
        }
    }
}

// The primitive [`ArrayBuilder`] now lives in `dispatch::arrays` (shared with the
// GROUP BY output); re-exported so this module's decoders keep using it.
pub use dispatch::arrays::PrimitiveBuilder;

/// [`DecodePlain`] implementation for fixed-width primitives.
///
/// Delegates to [`read_primitives`] for bulk decoding with its fast-path /
/// slow-path strategy.
pub struct PrimitivePlainDecoder<T: ArrowPrimitiveType>
where
    T::Native: ReadLeBytes + FromDelta,
{
    data: Vec<Bytes>,
    position: ReaderPosition,
    _phantom: PhantomData<T>,
}

impl<T: ArrowPrimitiveType> DecodePlain for PrimitivePlainDecoder<T>
where
    T::Native: ReadLeBytes + FromDelta,
{
    type Builder = PrimitiveBuilder<T>;
    type Delta = DeltaDecoder<T>;

    fn new(data: Vec<Bytes>, position: ReaderPosition) -> Self {
        Self {
            data,
            position,
            _phantom: PhantomData,
        }
    }

    fn read(&mut self, builder: &mut PrimitiveBuilder<T>, size: usize) {
        read_primitives::<T::Native, _>(
            &self.data,
            &mut self.position,
            &mut builder.col.values,
            &mut builder.col.len,
            size,
        );
    }

    fn skip(&mut self, size: usize) {
        let mut reader = MultiBufferReader::new(&self.data, &mut self.position);
        reader.skip(size * T::Native::PHYSICAL_SIZE);
    }
}

/// [`Dict`] implementation for fixed-width primitives, generic over the
/// buffer holding the entries.
///
/// A dictionary page that arrives in one contiguous buffer fits a single
/// [`SlabBuffer`] (the buffer is at most one 2MB ring slot and the native
/// width never exceeds the physical width), where every lookup is plain
/// pointer arithmetic. A scattered page falls back to a [`MultiSlabBuffer`]
/// and its per-element slab addressing.
pub struct PrimitiveDict<T: ArrowPrimitiveType, B>
where
    T::Native: ReadLeBytes,
{
    entries: B,
    len: usize,
    phantom: PhantomData<fn() -> T>,
}

impl<T: ArrowPrimitiveType, B: Index<usize, Output = T::Native> + Send + Sync + 'static> Dict
    for PrimitiveDict<T, B>
where
    T::Native: ReadLeBytes,
{
    type Builder = PrimitiveBuilder<T>;
    type Item = T::Native;
    type EqConstant = T::Native;

    /// The scalar matches when its array is a single-element
    /// `PrimitiveArray<T>`; a logical/physical type mismatch yields `None`.
    fn eq_constant_from_scalar(scalar: &Scalar<ArrayRef>) -> Option<T::Native> {
        let (arr, _) = arrow_array::Datum::get(scalar);
        let primitive = arr.as_primitive_opt::<T>()?;
        (primitive.len() == 1).then(|| primitive.value(0))
    }

    /// Scans the raw dictionary for `needle` without allocating or copying — so a
    /// row group whose dictionary excludes a pushed-down equality constant is
    /// pruned without building the dictionary.
    ///
    /// Fast path (the common case): a single contiguous, native-width,
    /// `T::Native`-aligned buffer is reinterpreted as `&[T::Native]` and scanned
    /// with the auto-vectorized [`slice::contains`] — the same scan the built
    /// dictionary would get, minus the copy. Anything else (unaligned, split
    /// across buffers, or a narrowed physical type) falls back to the scalar
    /// reader.
    fn maybe_contains(data: &[Bytes], size: usize, needle: &T::Native) -> bool {
        let width = T::Native::PHYSICAL_SIZE;
        if data.len() == 1 && width == mem::size_of::<T::Native>() && data[0].len() >= size * width
        {
            // SAFETY: `align_to` only reinterprets bytes; we only read. We trust
            // `mid` only when `head` is empty — i.e. the buffer is `T::Native`-
            // aligned so `mid` covers the `size` values exactly.
            let (head, mid, _) = unsafe { data[0][..size * width].align_to::<T::Native>() };
            if head.is_empty() {
                return mid.contains(needle);
            }
        }
        let mut position = ReaderPosition::default();
        let mut reader = MultiBufferReader::new(data, &mut position);
        (0..size).any(|_| T::Native::read_le(&mut reader) == *needle)
    }

    fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    fn entry(&self, idx: usize) -> T::Native {
        self.entries[idx]
    }
}

impl<T: ArrowPrimitiveType> DictFromBytes for PrimitiveDict<T, SlabBuffer<T::Native>>
where
    T::Native: ReadLeBytes,
{
    fn new_from_bytes(data: Bytes, size: usize, allocator: &mut SlabAllocator) -> Self {
        let mut entries = allocator.create_slab_buffer(size, false);
        let mut position = ReaderPosition::default();
        let mut len = 0;
        read_primitives::<T::Native, _>(&[data], &mut position, &mut entries, &mut len, size);
        Self {
            entries,
            len: size,
            phantom: PhantomData,
        }
    }
}

impl<T: ArrowPrimitiveType> DictFromVecBytes for PrimitiveDict<T, MultiSlabBuffer<T::Native>>
where
    T::Native: ReadLeBytes,
{
    fn new_from_vec_bytes(data: Vec<Bytes>, size: usize, allocator: &mut SlabAllocator) -> Self {
        let mut entries = allocator.create_multi_slab_buffer(size, false);
        let mut position = ReaderPosition::default();
        let mut len = 0;
        read_primitives::<T::Native, _>(&data, &mut position, &mut entries, &mut len, size);
        Self {
            entries,
            len: size,
            phantom: PhantomData,
        }
    }
}

/// Ready-to-use column decoder for any [`ArrowPrimitiveType`] whose native
/// type implements [`ReadLeBytes`].
pub type PrimitiveLeafDecoder<T> = TypedLeafDecoder<
    PrimitiveDict<T, SlabBuffer<<T as ArrowPrimitiveType>::Native>>,
    PrimitiveDict<T, MultiSlabBuffer<<T as ArrowPrimitiveType>::Native>>,
    PrimitiveBuilder<T>,
    PrimitivePlainDecoder<T>,
>;

/// Decodes a Parquet `INT64` timestamp and attaches its Arrow timezone
/// metadata without touching the decoded microsecond values.
///
/// This mirrors arrow-rs' Parquet reader: decode the physical values first,
/// then use [`TimestampMicrosecondArray::with_timezone_opt`] to annotate the
/// finished array. Cloning the primitive array only clones its buffer handles.
pub struct TimestampMicrosecondLeafDecoder {
    inner: PrimitiveLeafDecoder<TimestampMicrosecondType>,
    timezone: Option<Arc<str>>,
}

impl TimestampMicrosecondLeafDecoder {
    pub fn new(max_def_level: i16, timezone: Option<Arc<str>>) -> Self {
        Self {
            inner: PrimitiveLeafDecoder::new(max_def_level),
            timezone,
        }
    }
}

fn attach_timestamp_timezone(
    array: ArrayRef,
    timezone: Option<Arc<str>>,
) -> TimestampMicrosecondArray {
    array
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .expect("timestamp decoder must produce a microsecond timestamp array")
        .clone()
        .with_timezone_opt(timezone)
}

impl LeafDecoder for TimestampMicrosecondLeafDecoder {
    fn available(&self) -> usize {
        self.inner.available()
    }

    fn insert_page(&mut self, page: DecompressedPage) {
        self.inner.insert_page(page);
    }

    fn build_dictionary(
        &self,
        header: DictionaryPageHeader,
        data: Vec<Bytes>,
        allocator: &mut SlabAllocator,
    ) -> BuiltDictionary {
        self.inner.build_dictionary(header, data, allocator)
    }

    fn adopt_dictionary(&mut self, dictionary: SharedDictionary) {
        self.inner.adopt_dictionary(dictionary);
    }

    fn restart_at_page(&mut self, page_idx: usize) {
        self.inner.restart_at_page(page_idx);
    }

    fn read(&mut self, allocator: &mut SlabAllocator, size: usize) -> Result<ArrayRef> {
        let array = self.inner.read(allocator, size)?;
        Ok(Arc::new(attach_timestamp_timezone(
            array,
            self.timezone.clone(),
        )))
    }

    fn set_eq_constant(&mut self, value: &Scalar<ArrayRef>) {
        self.inner.set_eq_constant(value);
    }

    fn fast_filter_record_batch(&self, batch: RecordBatch, column: usize) -> RecordBatch {
        self.inner.fast_filter_record_batch(batch, column)
    }
}
//
#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::thrift::general::Encoding;
    use crate::thrift::headers::PageHeader;
    use arrow_array::types::{Float32Type, Int16Type, Int32Type, Int64Type, UInt16Type};
    use arrow_array::{
        Array, ArrayRef, Float32Array, Int16Array, Int32Array, Int64Array,
        TimestampMicrosecondArray, UInt16Array,
    };
    use bytes::Bytes;

    use super::{
        Dict, PrimitiveDict, PrimitiveLeafDecoder, ReaderPosition, attach_timestamp_timezone,
        read_primitives,
    };
    use dispatch::memory::{BUFFER_SIZE, MultiSlabBuffer, SlabBuffer};
    use std::mem;

    /// `maybe_contains` scans raw page bytes, so the buffer flavour is
    /// irrelevant; any instantiation works.
    type Int64Dict = PrimitiveDict<Int64Type, SlabBuffer<i64>>;
    use crate::reading::decoding::leaf_decoders::{BuiltDictionary, LeafDecoder};
    use crate::test_utils::dummy_metadata;
    use crate::types::metadata::RowSelection;
    use dispatch::memory::SlabAllocator;
    use dispatch::memory::init_test_free_pool;

    #[test]
    fn attaching_a_timestamp_timezone_preserves_the_value_buffer() {
        let input = TimestampMicrosecondArray::from(vec![42]);
        let values = input.values().as_ptr();

        let output = attach_timestamp_timezone(Arc::new(input), Some("UTC".into()));

        assert_eq!(output.timezone(), Some("UTC"));
        assert_eq!(output.values().as_ptr(), values);
        assert_eq!(output.value(0), 42);
    }

    #[test]
    fn maybe_contains_finds_present_and_rejects_absent() {
        init_test_free_pool(4);
        // A contiguous, aligned i64 dictionary (the vectorized fast path).
        let data = vec![Bytes::from(encode_i64s(&[10, 20, 30, 40, 50]))];
        assert!(Int64Dict::maybe_contains(&data, 5, &10));
        assert!(Int64Dict::maybe_contains(&data, 5, &50));
        assert!(Int64Dict::maybe_contains(&data, 5, &30));
        assert!(!Int64Dict::maybe_contains(&data, 5, &35));
        assert!(!Int64Dict::maybe_contains(&data, 5, &0));
    }

    #[test]
    fn bulk_read_into_two_slabs_stops_at_the_slab_boundary() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        // Two output slabs with a canary slab between them, as the pool may
        // hand them out in production, and one slab of i64s plus some to read
        // from a single input buffer: a copy sized by the input alone would
        // run past the first slab into the canary.
        let first = allocator.get_slab_of_size(BUFFER_SIZE, true);
        let canary = allocator.get_slab_of_size(BUFFER_SIZE, true);
        let second = allocator.get_slab_of_size(BUFFER_SIZE, true);
        let mut output = MultiSlabBuffer::<i64>::new(vec![first, second]);
        let per_slab = BUFFER_SIZE / mem::size_of::<i64>();
        let values: Vec<i64> = (0..per_slab as i64 + 1000).map(|i| i * 7 - 3).collect();
        let data = [Bytes::from(encode_i64s(&values))];
        let mut position = ReaderPosition::default();
        let mut len = 0;

        read_primitives::<i64, _>(&data, &mut position, &mut output, &mut len, values.len());

        assert_eq!(len, values.len());
        for (idx, expected) in values.iter().enumerate() {
            assert_eq!(output[idx], *expected, "entry {idx}");
        }
        assert!(canary.as_slice().iter().all(|byte| *byte == 0));
    }

    #[test]
    #[should_panic(expected = "output buffer is full after 100 elements")]
    fn bulk_read_refuses_to_overrun_a_single_slab_output() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut output: SlabBuffer<i64> = allocator.create_slab_buffer(100, true);
        let values: Vec<i64> = (0..101).collect();
        let data = [Bytes::from(encode_i64s(&values))];
        let mut position = ReaderPosition::default();
        let mut len = 0;

        read_primitives::<i64, _>(&data, &mut position, &mut output, &mut len, values.len());
    }

    #[test]
    fn maybe_contains_finds_value_straddling_a_buffer_boundary() {
        init_test_free_pool(4);
        // Three i64 values split so the middle one straddles the seam — this
        // takes the scalar-reader fallback (data.len() != 1).
        let bytes = encode_i64s(&[10, 20, 30]);
        let data = vec![
            Bytes::from(bytes[..12].to_vec()), // v0 + first half of v1
            Bytes::from(bytes[12..].to_vec()), // second half of v1 + v2
        ];
        assert!(Int64Dict::maybe_contains(&data, 3, &10));
        assert!(Int64Dict::maybe_contains(&data, 3, &20)); // straddles seam
        assert!(Int64Dict::maybe_contains(&data, 3, &30));
        assert!(!Int64Dict::maybe_contains(&data, 3, &99));
    }

    fn make_data_page(
        data: Vec<u8>,
        num_values: usize,
        encoding: Encoding,
        idx: usize,
    ) -> DecompressedPage {
        let header = PageHeader::for_data_page(num_values as i32, encoding);
        DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: dummy_metadata(RowSelection::All),
            column_idx: 0,
            idx,
            first_row: 0,
            data: DecompressedPageType::Data(DataPage {
                header: header.data_page_header.unwrap(),
                data: vec![Bytes::from(data)],
                filter_mask: None,
            }),
        }
    }

    fn make_data_page_multi_buffer(
        buffers: Vec<Vec<u8>>,
        num_values: usize,
        encoding: Encoding,
        idx: usize,
    ) -> DecompressedPage {
        let header = PageHeader::for_data_page(num_values as i32, encoding);
        DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: dummy_metadata(RowSelection::All),
            column_idx: 0,
            idx,
            first_row: 0,
            data: DecompressedPageType::Data(DataPage {
                header: header.data_page_header.unwrap(),
                data: buffers.into_iter().map(Bytes::from).collect(),
                filter_mask: None,
            }),
        }
    }

    fn load_dictionary(
        dec: &mut impl LeafDecoder,
        data: Vec<u8>,
        num_values: usize,
        allocator: &mut SlabAllocator,
    ) {
        let header = PageHeader::for_dict_page(num_values as i32)
            .dictionary_page_header
            .unwrap();
        match dec.build_dictionary(header, vec![Bytes::from(data)], allocator) {
            BuiltDictionary::Built(dictionary) => dec.adopt_dictionary(dictionary),
            BuiltDictionary::Pruned => panic!("no constant is set"),
        }
    }

    fn encode_i32s(values: &[i32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn encode_i64s(values: &[i64]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn encode_f32s(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn extract_i32s(arr: &ArrayRef) -> Vec<i32> {
        let a = arr.as_any().downcast_ref::<Int32Array>().unwrap();
        (0..a.len()).map(|i| a.value(i)).collect()
    }

    fn extract_i64s(arr: &ArrayRef) -> Vec<i64> {
        let a = arr.as_any().downcast_ref::<Int64Array>().unwrap();
        (0..a.len()).map(|i| a.value(i)).collect()
    }

    fn extract_f32s(arr: &ArrayRef) -> Vec<f32> {
        let a = arr.as_any().downcast_ref::<Float32Array>().unwrap();
        (0..a.len()).map(|i| a.value(i)).collect()
    }

    // -- PLAIN single-page tests --

    #[test]
    fn test_i32_single_page() {
        let values = vec![1i32, 2, 3, 100, 0];
        let page = make_data_page(encode_i32s(&values), values.len(), Encoding::PLAIN, 0);

        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut dec = PrimitiveLeafDecoder::<Int32Type>::new(0);
        dec.insert_page(page);

        assert_eq!(dec.available(), 5);
        let result = dec.read(&mut allocator, 5).unwrap();
        assert_eq!(extract_i32s(&result), values);
    }

    #[test]
    fn test_i64_single_page() {
        let values = vec![10i64, 20, 30, i64::MAX, i64::MIN];
        let page = make_data_page(encode_i64s(&values), values.len(), Encoding::PLAIN, 0);

        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut dec = PrimitiveLeafDecoder::<Int64Type>::new(0);
        dec.insert_page(page);

        assert_eq!(dec.available(), 5);
        let result = dec.read(&mut allocator, 5).unwrap();
        assert_eq!(extract_i64s(&result), values);
    }

    #[test]
    fn test_f32_single_page() {
        let values = vec![1.5f32, -2.25, 0.0, f32::MAX];
        let page = make_data_page(encode_f32s(&values), values.len(), Encoding::PLAIN, 0);

        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut dec = PrimitiveLeafDecoder::<Float32Type>::new(0);
        dec.insert_page(page);

        assert_eq!(dec.available(), 4);
        let result = dec.read(&mut allocator, 4).unwrap();
        assert_eq!(extract_f32s(&result), values);
    }

    // -- Incremental & multi-page --

    #[test]
    fn test_i32_incremental() {
        let values = vec![10i32, 20, 30, 40, 50];
        let page = make_data_page(encode_i32s(&values), values.len(), Encoding::PLAIN, 0);

        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut dec = PrimitiveLeafDecoder::<Int32Type>::new(0);
        dec.insert_page(page);

        let r1 = dec.read(&mut allocator, 2).unwrap();
        assert_eq!(extract_i32s(&r1), vec![10, 20]);

        let r2 = dec.read(&mut allocator, 3).unwrap();
        assert_eq!(extract_i32s(&r2), vec![30, 40, 50]);
    }

    #[test]
    fn test_i32_multiple_pages() {
        let page0 = make_data_page(encode_i32s(&[1, 2, 3]), 3, Encoding::PLAIN, 0);
        let page1 = make_data_page(encode_i32s(&[4, 5]), 2, Encoding::PLAIN, 1);

        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut dec = PrimitiveLeafDecoder::<Int32Type>::new(0);
        dec.insert_page(page0);
        dec.insert_page(page1);

        assert_eq!(dec.available(), 5);
        let result = dec.read(&mut allocator, 5).unwrap();
        assert_eq!(extract_i32s(&result), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_i32_negative_values() {
        let values = vec![-1i32, -100, -i32::MAX, i32::MIN, 0, 42];
        let page = make_data_page(encode_i32s(&values), values.len(), Encoding::PLAIN, 0);

        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut dec = PrimitiveLeafDecoder::<Int32Type>::new(0);
        dec.insert_page(page);

        let result = dec.read(&mut allocator, 6).unwrap();
        assert_eq!(extract_i32s(&result), values);
    }

    // -- Cross-buffer boundary (slow path) --

    #[test]
    fn test_i32_cross_buffer_boundary() {
        // 3 i32 values = 12 bytes. Split at byte 6 (middle of 2nd value).
        let data = encode_i32s(&[10, 20, 30]);
        let page = make_data_page_multi_buffer(
            vec![data[..6].to_vec(), data[6..].to_vec()],
            3,
            Encoding::PLAIN,
            0,
        );

        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut dec = PrimitiveLeafDecoder::<Int32Type>::new(0);
        dec.insert_page(page);

        let result = dec.read(&mut allocator, 3).unwrap();
        assert_eq!(extract_i32s(&result), vec![10, 20, 30]);
    }

    // -- Dict-encoded --

    #[test]
    fn test_i32_dict_encoded() {
        // Dictionary entries: [100, 200, 300]

        // RLE data page: bit_width=2, 1 group of 8 values
        // Indices: [0, 1, 2, 0, 0, 1, 2, 0]
        // 2-bit packed LSB-first:
        //   byte0: idx[0]=0(00) | idx[1]=1(01)<<2 | idx[2]=2(10)<<4 | idx[3]=0(00)<<6 = 0x24
        //   byte1: idx[4]=0(00) | idx[5]=1(01)<<2 | idx[6]=2(10)<<4 | idx[7]=0(00)<<6 = 0x24
        let mut rle_data = vec![2u8]; // bit_width
        rle_data.extend_from_slice(&[3, 0x24, 0x24]); // header=(1<<1)|1=3, packed bytes
        let data_page = make_data_page(rle_data, 8, Encoding::RLE_DICTIONARY, 0);

        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut dec = PrimitiveLeafDecoder::<Int32Type>::new(0);
        load_dictionary(&mut dec, encode_i32s(&[100, 200, 300]), 3, &mut allocator);
        dec.insert_page(data_page);

        assert_eq!(dec.available(), 8);
        let result = dec.read(&mut allocator, 8).unwrap();
        assert_eq!(
            extract_i32s(&result),
            vec![100, 200, 300, 100, 100, 200, 300, 100]
        );
    }

    // -- Filter mask integration --

    use crate::types::filter_mask::FilterMask;
    use crate::types::page::{DataPage, DecompressedPage, DecompressedPageType};

    fn make_filtered_data_page(
        data: Vec<u8>,
        num_values: usize,
        encoding: Encoding,
        idx: usize,
        filter_mask: FilterMask,
    ) -> DecompressedPage {
        let header = PageHeader::for_data_page(num_values as i32, encoding);
        DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: dummy_metadata(RowSelection::All),
            column_idx: 0,
            idx,
            first_row: 0,
            data: DecompressedPageType::Data(DataPage {
                header: header.data_page_header.unwrap(),
                data: vec![Bytes::from(data)],
                filter_mask: Some(filter_mask),
            }),
        }
    }

    /// Page [10, 20, 30, 40, 50], keep indices [1, 3] → [20, 40].
    #[test]
    fn test_i32_filter_skip_first_and_middle() {
        let page = make_filtered_data_page(
            encode_i32s(&[10, 20, 30, 40, 50]),
            5,
            Encoding::PLAIN,
            0,
            FilterMask::new(0, 5, &[1, 3]),
        );

        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut dec = PrimitiveLeafDecoder::<Int32Type>::new(0);
        dec.insert_page(page);

        assert_eq!(dec.available(), 2);
        let result = dec.read(&mut allocator, 2).unwrap();
        assert_eq!(extract_i32s(&result), vec![20, 40]);
    }

    /// Page [10, 20, 30, 40, 50], keep only last [4] → [50].
    #[test]
    fn test_i32_filter_keep_last_only() {
        let page = make_filtered_data_page(
            encode_i32s(&[10, 20, 30, 40, 50]),
            5,
            Encoding::PLAIN,
            0,
            FilterMask::new(0, 5, &[4]),
        );

        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut dec = PrimitiveLeafDecoder::<Int32Type>::new(0);
        dec.insert_page(page);

        assert_eq!(dec.available(), 1);
        let result = dec.read(&mut allocator, 1).unwrap();
        assert_eq!(extract_i32s(&result), vec![50]);
    }

    /// Page [10, 20, 30], keep all [0,1,2] → [10, 20, 30].
    #[test]
    fn test_i32_filter_keep_all() {
        let page = make_filtered_data_page(
            encode_i32s(&[10, 20, 30]),
            3,
            Encoding::PLAIN,
            0,
            FilterMask::new(0, 3, &[0, 1, 2]),
        );

        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut dec = PrimitiveLeafDecoder::<Int32Type>::new(0);
        dec.insert_page(page);

        assert_eq!(dec.available(), 3);
        let result = dec.read(&mut allocator, 3).unwrap();
        assert_eq!(extract_i32s(&result), vec![10, 20, 30]);
    }

    /// Page [10, 20, 30, 40, 50], keep none → 0 available.
    #[test]
    fn test_i32_filter_keep_none() {
        let page = make_filtered_data_page(
            encode_i32s(&[10, 20, 30, 40, 50]),
            5,
            Encoding::PLAIN,
            0,
            FilterMask::new(0, 5, &[]),
        );

        init_test_free_pool(4);
        let mut dec = PrimitiveLeafDecoder::<Int32Type>::new(0);
        dec.insert_page(page);

        assert_eq!(dec.available(), 0);
    }

    /// Dict-encoded page, 8 values [100,200,300,400,100,200,300,400],
    /// keep [0, 2, 5] → [100, 300, 200].
    #[test]
    fn test_i32_filter_dict_encoded() {
        // bit_width=2, 1 group of 8 values, indices [0,1,2,3,0,1,2,3]
        let mut rle_data = vec![2u8];
        rle_data.extend_from_slice(&[3, 0xE4, 0xE4]);

        let data_page = make_filtered_data_page(
            rle_data,
            8,
            Encoding::RLE_DICTIONARY,
            0,
            FilterMask::new(0, 8, &[0, 2, 5]),
        );

        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut dec = PrimitiveLeafDecoder::<Int32Type>::new(0);
        load_dictionary(
            &mut dec,
            encode_i32s(&[100, 200, 300, 400]),
            4,
            &mut allocator,
        );
        dec.insert_page(data_page);

        assert_eq!(dec.available(), 3);
        let result = dec.read(&mut allocator, 3).unwrap();
        assert_eq!(extract_i32s(&result), vec![100, 300, 200]);
    }

    // -- i16/u16 tests (Parquet stores these as 4-byte INT32) --

    /// Encode i16 values as Parquet INT32 (4 bytes LE, sign-extended).
    fn encode_i16_as_parquet(values: &[i16]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&v| (v as i32).to_le_bytes())
            .collect()
    }

    /// Encode u16 values as Parquet INT32 (4 bytes LE, zero-extended).
    fn encode_u16_as_parquet(values: &[u16]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&v| (v as i32).to_le_bytes())
            .collect()
    }

    fn extract_i16s(arr: &ArrayRef) -> Vec<i16> {
        let a = arr.as_any().downcast_ref::<Int16Array>().unwrap();
        (0..a.len()).map(|i| a.value(i)).collect()
    }

    fn extract_u16s(arr: &ArrayRef) -> Vec<u16> {
        let a = arr.as_any().downcast_ref::<UInt16Array>().unwrap();
        (0..a.len()).map(|i| a.value(i)).collect()
    }

    #[test]
    fn test_i16_single_value() {
        let values = vec![42i16];
        let page = make_data_page(encode_i16_as_parquet(&values), 1, Encoding::PLAIN, 0);

        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut dec = PrimitiveLeafDecoder::<Int16Type>::new(0);
        dec.insert_page(page);

        assert_eq!(dec.available(), 1);
        let result = dec.read(&mut allocator, 1).unwrap();
        assert_eq!(extract_i16s(&result), vec![42]);
    }

    /// Multiple i16 values - exposes fast-path byte_width mismatch.
    #[test]
    fn test_i16_multiple_values() {
        let values = vec![1i16, 2, 3, 4, 5];
        let page = make_data_page(encode_i16_as_parquet(&values), 5, Encoding::PLAIN, 0);

        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut dec = PrimitiveLeafDecoder::<Int16Type>::new(0);
        dec.insert_page(page);

        assert_eq!(dec.available(), 5);
        let result = dec.read(&mut allocator, 5).unwrap();
        assert_eq!(extract_i16s(&result), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_i16_negative_values() {
        let values = vec![-1i16, -100, i16::MIN, i16::MAX, 0];
        let page = make_data_page(encode_i16_as_parquet(&values), 5, Encoding::PLAIN, 0);

        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut dec = PrimitiveLeafDecoder::<Int16Type>::new(0);
        dec.insert_page(page);

        assert_eq!(dec.available(), 5);
        let result = dec.read(&mut allocator, 5).unwrap();
        assert_eq!(extract_i16s(&result), vec![-1, -100, i16::MIN, i16::MAX, 0]);
    }

    #[test]
    fn test_u16_multiple_values() {
        let values = vec![0u16, 1, 1000, u16::MAX, 42];
        let page = make_data_page(encode_u16_as_parquet(&values), 5, Encoding::PLAIN, 0);

        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut dec = PrimitiveLeafDecoder::<UInt16Type>::new(0);
        dec.insert_page(page);

        assert_eq!(dec.available(), 5);
        let result = dec.read(&mut allocator, 5).unwrap();
        assert_eq!(extract_u16s(&result), vec![0, 1, 1000, u16::MAX, 42]);
    }

    /// Incremental reads for i16 - read 2, then 3.
    #[test]
    fn test_i16_incremental_read() {
        let values = vec![10i16, 20, 30, 40, 50];
        let page = make_data_page(encode_i16_as_parquet(&values), 5, Encoding::PLAIN, 0);

        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut dec = PrimitiveLeafDecoder::<Int16Type>::new(0);
        dec.insert_page(page);

        let r1 = dec.read(&mut allocator, 2).unwrap();
        assert_eq!(extract_i16s(&r1), vec![10, 20]);

        let r2 = dec.read(&mut allocator, 3).unwrap();
        assert_eq!(extract_i16s(&r2), vec![30, 40, 50]);
    }

    /// i16 with filter mask - keep indices [1, 3] from [10, 20, 30, 40, 50] → [20, 40].
    #[test]
    fn test_i16_filter() {
        let page = make_filtered_data_page(
            encode_i16_as_parquet(&[10, 20, 30, 40, 50]),
            5,
            Encoding::PLAIN,
            0,
            FilterMask::new(0, 5, &[1, 3]),
        );

        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut dec = PrimitiveLeafDecoder::<Int16Type>::new(0);
        dec.insert_page(page);

        assert_eq!(dec.available(), 2);
        let result = dec.read(&mut allocator, 2).unwrap();
        assert_eq!(extract_i16s(&result), vec![20, 40]);
    }

    /// i16 across two pages.
    #[test]
    fn test_i16_two_pages() {
        let page0 = make_data_page(encode_i16_as_parquet(&[1, 2, 3]), 3, Encoding::PLAIN, 0);
        let page1 = make_data_page(encode_i16_as_parquet(&[4, 5]), 2, Encoding::PLAIN, 1);

        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut dec = PrimitiveLeafDecoder::<Int16Type>::new(0);
        dec.insert_page(page0);
        dec.insert_page(page1);

        assert_eq!(dec.available(), 5);
        let result = dec.read(&mut allocator, 5).unwrap();
        assert_eq!(extract_i16s(&result), vec![1, 2, 3, 4, 5]);
    }
}
