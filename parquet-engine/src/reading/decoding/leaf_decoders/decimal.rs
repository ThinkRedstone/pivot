//! Column decoder for DECIMAL columns.
//!
//! Parquet allows three on-disk storages for a decimal's unscaled integer:
//! INT32 and INT64 hold it little-endian, FIXED_LEN_BYTE_ARRAY holds it
//! big-endian two's complement in the schema's declared length.
//! [`DecimalStorage`] abstracts the storage; every storage decodes through an
//! `i128`, which [`DecimalCarrier`] then narrows to the column's in-memory
//! carrier: `Decimal64` (i64) for a declared precision up to 18 digits,
//! `Decimal128` (i128) beyond.
//!
//! Decoding is chunked like the primitive decoder's: whole values are decoded
//! straight off each buffer's contiguous slice, and only a value straddling a
//! buffer boundary goes through a [`MultiBufferReader`].
//!
//! [`DecimalLeafDecoder`] pairs the page decoder with the column's declared
//! precision and scale, so every array it emits carries the declared shape.

use std::marker::PhantomData;
use std::ops::Index;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Decimal64Type, Decimal128Type, DecimalType};
use arrow_array::{ArrayRef, RecordBatch, Scalar};
use bytes::Bytes;

use super::{BuiltDictionary, SharedDictionary};
use crate::reading::decoding::leaf_decoders::primitive::ElemPtr;
use crate::reading::decoding::leaf_decoders::{
    DecodeEncoding, DecodePlain, DeltaDecoder, Dict, DictFromBytes, DictFromVecBytes, Error,
    FromDelta, LeafDecoder, LeafType, Result, TypedLeafDecoder, Unsupported,
};
use crate::thrift::headers::DictionaryPageHeader;
use crate::types::metadata::ColumnChunkMeta;
use crate::types::page::DecompressedPage;
use dispatch::arrays::PrimitiveBuilder;
use dispatch::memory::{
    MultiBufferReader, MultiSlabBuffer, ReaderPosition, SlabAllocator, SlabBuffer,
};

/// How a decimal column's unscaled integer is stored on disk. Each storage
/// decodes through the same sign-extended `i128`.
pub trait DecimalStorage: 'static {
    /// Bytes one value occupies on disk.
    const PHYSICAL_SIZE: usize;

    /// The decoder of a `DELTA_BINARY_PACKED` page in this storage. The
    /// encoding is defined for `INT32` and `INT64`, whose unscaled integers
    /// pack like any other integer's, and not for fixed-length bytes.
    type DeltaBinaryPacked<T: DecimalCarrier>: DecodeEncoding<Builder = PrimitiveBuilder<T>>;

    /// Decodes one value from its exactly-`PHYSICAL_SIZE`-byte slice,
    /// sign-extending to 128 bits.
    fn decode(bytes: &[u8]) -> i128;

    /// Reads one value through the reader, for a value that straddles a
    /// buffer boundary.
    fn read(reader: &mut MultiBufferReader) -> i128;
}

/// INT32 storage: 4 little-endian bytes (precision up to 9).
pub struct DecimalFromInt32;

impl DecimalStorage for DecimalFromInt32 {
    const PHYSICAL_SIZE: usize = 4;
    type DeltaBinaryPacked<T: DecimalCarrier> = DeltaDecoder<T>;

    #[inline(always)]
    fn decode(bytes: &[u8]) -> i128 {
        i32::from_le_bytes(bytes.try_into().unwrap()) as i128
    }

    #[inline(always)]
    fn read(reader: &mut MultiBufferReader) -> i128 {
        reader.read_i32_le() as i128
    }
}

/// INT64 storage: 8 little-endian bytes (precision up to 18).
pub struct DecimalFromInt64;

impl DecimalStorage for DecimalFromInt64 {
    const PHYSICAL_SIZE: usize = 8;
    type DeltaBinaryPacked<T: DecimalCarrier> = DeltaDecoder<T>;

    #[inline(always)]
    fn decode(bytes: &[u8]) -> i128 {
        i64::from_le_bytes(bytes.try_into().unwrap()) as i128
    }

    #[inline(always)]
    fn read(reader: &mut MultiBufferReader) -> i128 {
        reader.read_i64_le() as i128
    }
}

/// FIXED_LEN_BYTE_ARRAY storage: `LEN` big-endian two's-complement bytes,
/// where `LEN` is the schema element's `type_length` (1 through 16).
pub struct DecimalFromFixedLen<const LEN: usize>;

impl<const LEN: usize> DecimalStorage for DecimalFromFixedLen<LEN> {
    const PHYSICAL_SIZE: usize = LEN;
    type DeltaBinaryPacked<T: DecimalCarrier> = Unsupported<PrimitiveBuilder<T>>;

    #[inline(always)]
    fn decode(bytes: &[u8]) -> i128 {
        let fill = if bytes[0] & 0x80 != 0 { 0xff } else { 0 };
        let mut wide = [fill; 16];
        wide[16 - LEN..].copy_from_slice(bytes);
        i128::from_be_bytes(wide)
    }

    #[inline(always)]
    fn read(reader: &mut MultiBufferReader) -> i128 {
        let bytes: [u8; LEN] = reader.read_fixed_slice::<LEN>();
        Self::decode(&bytes)
    }
}

/// The in-memory column a decimal column decodes into: [`Decimal64Type`]
/// (i64) for a declared precision up to 18 digits, [`Decimal128Type`] (i128)
/// beyond.
pub trait DecimalCarrier: DecimalType<Native: FromDelta> {
    /// Narrows the storage's sign-extended `i128` to the carrier's native
    /// integer. The declared precision guarantees the value fits.
    fn narrow(value: i128) -> Self::Native;
}

impl DecimalCarrier for Decimal64Type {
    #[inline(always)]
    fn narrow(value: i128) -> i64 {
        debug_assert!(
            i64::try_from(value).is_ok(),
            "a decimal declared at 18 digits or fewer fits in i64"
        );
        value as i64
    }
}

impl DecimalCarrier for Decimal128Type {
    #[inline(always)]
    fn narrow(value: i128) -> i128 {
        value
    }
}

/// Reads `count` decimal values from scattered buffers into `output` (a
/// single-slab builder buffer or a multi-slab dictionary buffer).
///
/// Fast path: all values that fit within the current buffer are decoded
/// straight off its contiguous slice, one `PHYSICAL_SIZE` chunk at a time.
/// Slow path: a value that straddles a buffer boundary falls back to a
/// [`MultiBufferReader`] for that single value, then the fast path resumes.
#[inline]
// `as_chunks` wants the width as a const-generic argument, which an associated
// const of a generic parameter cannot be.
#[allow(clippy::chunks_exact_to_as_chunks)]
fn read_decimals<T: DecimalCarrier, S: DecimalStorage, B: ElemPtr<T::Native>>(
    data: &[Bytes],
    position: &mut ReaderPosition,
    output: &mut B,
    output_len: &mut usize,
    count: usize,
) {
    let target = *output_len + count;

    while *output_len < target {
        let remaining = target - *output_len;
        let buf = &data[position.buffer_index];
        let available_bytes = buf.len() - position.offset;
        let fit = available_bytes / S::PHYSICAL_SIZE;

        if fit > 0 {
            // Fast path: decode all values that fit in the current buffer.
            let to_read = remaining.min(fit);
            let byte_count = to_read * S::PHYSICAL_SIZE;
            let src = &buf[position.offset..position.offset + byte_count];
            for chunk in src.chunks_exact(S::PHYSICAL_SIZE) {
                unsafe { *output.elem_ptr(*output_len) = T::narrow(S::decode(chunk)) };
                *output_len += 1;
            }
            position.offset += byte_count;
        } else if available_bytes > 0 {
            // Slow path: value straddles a buffer boundary.
            let mut reader = MultiBufferReader::new(data, position);
            unsafe { *output.elem_ptr(*output_len) = T::narrow(S::read(&mut reader)) };
            *output_len += 1;
        } else {
            // Buffer fully consumed, advance to next.
            position.buffer_index += 1;
            position.offset = 0;
        }
    }
}

/// [`DecodePlain`] implementation for decimals, delegating to
/// [`read_decimals`] for chunked decoding.
pub struct DecimalPlainDecoder<T: DecimalCarrier, S: DecimalStorage> {
    data: Vec<Bytes>,
    position: ReaderPosition,
    _marker: PhantomData<(T, S)>,
}

impl<T: DecimalCarrier, S: DecimalStorage> DecodePlain for DecimalPlainDecoder<T, S> {
    type Builder = PrimitiveBuilder<T>;

    fn new(data: Vec<Bytes>, position: ReaderPosition) -> Self {
        Self {
            data,
            position,
            _marker: PhantomData,
        }
    }

    fn read(&mut self, builder: &mut PrimitiveBuilder<T>, size: usize) {
        read_decimals::<T, S, _>(
            &self.data,
            &mut self.position,
            &mut builder.col.values,
            &mut builder.col.len,
            size,
        );
    }

    fn skip(&mut self, size: usize) {
        let mut reader = MultiBufferReader::new(&self.data, &mut self.position);
        reader.skip(size * S::PHYSICAL_SIZE);
    }
}

/// [`Dict`] implementation for decimals, generic over the buffer holding the
/// carrier-width entries (see [`super::primitive::PrimitiveDict`] for why
/// there are two buffer flavours).
pub struct DecimalDict<T: DecimalCarrier, S: DecimalStorage, B> {
    entries: B,
    len: usize,
    _marker: PhantomData<fn() -> (T, S)>,
}

impl<T, S, B> Dict for DecimalDict<T, S, B>
where
    T: DecimalCarrier,
    S: DecimalStorage,
    B: Index<usize, Output = T::Native> + Send + Sync + 'static,
{
    type Builder = PrimitiveBuilder<T>;
    type Item = T::Native;
    type EqConstant = T::Native;

    /// The unscaled integer of a single-element decimal scalar of the
    /// carrier's width. The pushed constant is bound by DuckDB to the
    /// column's own type, so its scale matches the column's and the raw
    /// integers compare directly.
    fn eq_constant_from_scalar(scalar: &Scalar<ArrayRef>) -> Option<T::Native> {
        let (arr, _) = arrow_array::Datum::get(scalar);
        let decimal = arr.as_primitive_opt::<T>()?;
        (decimal.len() == 1).then(|| decimal.value(0))
    }

    /// Scans the raw dictionary bytes for `needle` without building the
    /// dictionary, chunked like [`read_decimals`]: each buffer's whole values
    /// are decoded off its contiguous slice, and only a value straddling a
    /// buffer boundary goes through a reader.
    // `as_chunks` wants the width as a const-generic argument, which an
    // associated const of a generic parameter cannot be.
    #[allow(clippy::chunks_exact_to_as_chunks)]
    fn maybe_contains(data: &[Bytes], size: usize, needle: &T::Native) -> bool {
        let mut position = ReaderPosition::default();
        let mut remaining = size;
        while remaining > 0 {
            let buf = &data[position.buffer_index];
            let available_bytes = buf.len() - position.offset;
            let fit = available_bytes / S::PHYSICAL_SIZE;
            if fit > 0 {
                let to_scan = remaining.min(fit);
                let byte_count = to_scan * S::PHYSICAL_SIZE;
                let src = &buf[position.offset..position.offset + byte_count];
                if src
                    .chunks_exact(S::PHYSICAL_SIZE)
                    .any(|chunk| T::narrow(S::decode(chunk)) == *needle)
                {
                    return true;
                }
                position.offset += byte_count;
                remaining -= to_scan;
            } else if available_bytes > 0 {
                let mut reader = MultiBufferReader::new(data, &mut position);
                if T::narrow(S::read(&mut reader)) == *needle {
                    return true;
                }
                remaining -= 1;
            } else {
                position.buffer_index += 1;
                position.offset = 0;
            }
        }
        false
    }

    fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    fn entry(&self, idx: usize) -> T::Native {
        self.entries[idx]
    }
}

impl<T: DecimalCarrier, S: DecimalStorage> DictFromBytes
    for DecimalDict<T, S, SlabBuffer<T::Native>>
{
    fn new_from_bytes(data: Bytes, size: usize, allocator: &mut SlabAllocator) -> Self {
        let mut entries: SlabBuffer<T::Native> = allocator.create_slab_buffer(size, false);
        let mut position = ReaderPosition::default();
        let mut len = 0;
        read_decimals::<T, S, _>(&[data], &mut position, &mut entries, &mut len, size);
        Self {
            entries,
            len: size,
            _marker: PhantomData,
        }
    }
}

impl<T: DecimalCarrier, S: DecimalStorage> DictFromVecBytes
    for DecimalDict<T, S, MultiSlabBuffer<T::Native>>
{
    fn new_from_vec_bytes(data: Vec<Bytes>, size: usize, allocator: &mut SlabAllocator) -> Self {
        let mut entries: MultiSlabBuffer<T::Native> =
            allocator.create_multi_slab_buffer(size, false);
        let mut position = ReaderPosition::default();
        let mut len = 0;
        read_decimals::<T, S, _>(&data, &mut position, &mut entries, &mut len, size);
        Self {
            entries,
            len: size,
            _marker: PhantomData,
        }
    }
}

/// The [`LeafType`] of a decimal column of carrier `T` and storage `S`.
pub struct DecimalLeaf<T, S>(PhantomData<(T, S)>);

impl<T: DecimalCarrier, S: DecimalStorage> LeafType for DecimalLeaf<T, S> {
    type Builder = PrimitiveBuilder<T>;

    type Plain = DecimalPlainDecoder<T, S>;
    type DeltaBinaryPacked = S::DeltaBinaryPacked<T>;
    type DeltaLengthByteArray = Unsupported<PrimitiveBuilder<T>>;
    type Rle = Unsupported<PrimitiveBuilder<T>>;

    type ContiguousDict = DecimalDict<T, S, SlabBuffer<T::Native>>;
    type ScatteredDict = DecimalDict<T, S, MultiSlabBuffer<T::Native>>;
}

/// The generic page decoder a decimal column decodes through, before its
/// arrays are given the column's declared shape.
type DecimalPageDecoder<T, S> = TypedLeafDecoder<DecimalLeaf<T, S>>;

/// Ready-to-use column decoder for a decimal column of carrier `T` and
/// storage `S`: the page decoder plus the column's declared precision and
/// scale, which every array it reads is restamped to (the page decoder alone
/// emits the carrier's default shape, since a builder cannot know the
/// column's).
///
/// The restamp is metadata-only: the values buffer is shared, never rescaled.
/// An arrow `cast` between decimal types would rescale the stored integers,
/// so it must never be used here.
pub struct DecimalLeafDecoder<T: DecimalCarrier, S: DecimalStorage> {
    inner: DecimalPageDecoder<T, S>,
    precision: u8,
    scale: i8,
}

impl<T: DecimalCarrier, S: DecimalStorage> DecimalLeafDecoder<T, S> {
    pub fn new(max_def_level: i16, precision: u8, scale: i8) -> Self {
        Self {
            inner: DecimalPageDecoder::new(max_def_level),
            precision,
            scale,
        }
    }
}

impl<T: DecimalCarrier, S: DecimalStorage> LeafDecoder for DecimalLeafDecoder<T, S> {
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
        let restamped = array
            .as_primitive::<T>()
            .clone()
            .with_precision_and_scale(self.precision, self.scale)
            .expect("the declared decimal shape was validated at footer parse");
        Ok(Arc::new(restamped))
    }

    fn set_eq_constant(&mut self, value: &Scalar<ArrayRef>) {
        self.inner.set_eq_constant(value);
    }

    fn fast_filter_record_batch(&self, batch: RecordBatch, column: usize) -> RecordBatch {
        self.inner.fast_filter_record_batch(batch, column)
    }
}

/// Get the decimal column decoder matching the chunk's physical storage
/// (INT32/INT64 hold the unscaled integer little-endian, FIXED_LEN_BYTE_ARRAY
/// big-endian in the schema's declared byte width). The carrier `T` is chosen
/// by the caller from the leaf's arrow type; the leaf's declared precision
/// and scale complete the decoder.
pub fn decimal_decoder<T: DecimalCarrier>(
    chunk: &ColumnChunkMeta,
    precision: u8,
    scale: i8,
) -> Result<Box<dyn LeafDecoder>> {
    use crate::thrift::general::Type as PhysicalType;
    if chunk.physical_type == PhysicalType::INT32 as i32 {
        return Ok(Box::new(DecimalLeafDecoder::<T, DecimalFromInt32>::new(
            chunk.max_def_level,
            precision,
            scale,
        )));
    }
    if chunk.physical_type == PhysicalType::INT64 as i32 {
        return Ok(Box::new(DecimalLeafDecoder::<T, DecimalFromInt64>::new(
            chunk.max_def_level,
            precision,
            scale,
        )));
    }
    if chunk.physical_type == PhysicalType::FIXED_LEN_BYTE_ARRAY as i32 {
        macro_rules! fixed_len {
            ($($len:literal),+) => {
                match chunk.fixed_len_byte_width {
                    $(Some($len) => {
                        return Ok(Box::new(
                            DecimalLeafDecoder::<T, DecimalFromFixedLen<$len>>::new(
                                chunk.max_def_level,
                                precision,
                                scale,
                            ),
                        ));
                    })+
                    _ => {}
                }
            };
        }
        fixed_len!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
    }
    Err(Error::UnsupportedDecimalStorage {
        data_type: T::TYPE_CONSTRUCTOR(precision, scale),
        physical_type: chunk.physical_type,
        type_length: chunk.fixed_len_byte_width,
    })
}
