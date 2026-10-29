//! The PLAIN encode path: cut a leaf into pages and PLAIN-encode each.
//!
//! A page's body is its definition levels (for a leaf that has any) followed by
//! the stored values back to back — fixed-width values little-endian, booleans
//! one bit each, BYTE_ARRAY values a 4-byte LE length prefix then the bytes.
//! [`encode_into`] is also reused by [`dictionary`](super::dictionary) to encode
//! a dictionary page's distinct values.

use crate::thrift::general::{CompressionCodec, Encoding};
use arrow_array::{
    Array, BinaryViewArray, BooleanArray, Date32Array, Decimal64Array, Decimal128Array,
    Float32Array, Float64Array, Int8Array, Int16Array, Int32Array, Int64Array, StringArray,
    StringViewArray, TimestampMicrosecondArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, TimeUnit};

use dispatch::memory::SlabAllocator;

use crate::DecimalWriteStorage;

use super::super::error::{WriteError, WriteResult};
use super::super::leaves::Leaf;
use super::super::types::EncodedPage;
use super::pages::{self, PageKind, PageRange};

/// PLAIN-encode a leaf: cut it into pages and encode each.
pub(super) fn encode_chunk(
    leaf: &Leaf,
    codec: CompressionCodec,
    allocator: &mut SlabAllocator,
) -> WriteResult<Vec<EncodedPage>> {
    pages::page_ranges(leaf)?
        .into_iter()
        .map(|range| encode_data_page(leaf, range, codec, allocator))
        .collect()
}

/// Encode one PLAIN data page: the page's rows' definition levels, then its
/// stored values. A page counts its rows, not its values — the absent rows have
/// a level but nothing in the value stream.
fn encode_data_page(
    leaf: &Leaf,
    range: PageRange,
    codec: CompressionCodec,
    allocator: &mut SlabAllocator,
) -> WriteResult<EncodedPage> {
    let num_rows = range.rows.len();
    let values = leaf.values.slice(range.values.start, range.values.len());
    let mut encoded = Vec::new();
    encode_into(values.as_ref(), &mut encoded)?;

    let levels = leaf.def_levels.as_ref().map(|levels| &levels[range.rows]);
    let raw = pages::data_page_body(levels, leaf.max_def_level, encoded);
    pages::assemble_page(
        num_rows as i64,
        raw,
        PageKind::Data {
            num_values: num_rows,
            encoding: Encoding::PLAIN,
        },
        codec,
        allocator,
    )
}

/// Append `array`'s PLAIN-encoded values to `out`: fixed-width values
/// little-endian (except wide decimals, whose fixed-length bytes are
/// big-endian), booleans one bit each, BYTE_ARRAY values a length-prefixed
/// copy.
///
/// Parquet stores only the values that are present, so a leaf's absent rows are
/// already dropped by the time they reach here (see
/// [`leaves`](super::super::leaves)) — a null left in the array would mean a field
/// nullable in the data but required in the schema, which would silently write
/// the wrong values, so it errors instead.
pub(super) fn encode_into(array: &dyn Array, out: &mut Vec<u8>) -> WriteResult<()> {
    if array.null_count() > 0 {
        return Err(WriteError::NullsInRequiredColumn {
            nulls: array.null_count(),
        });
    }
    let len = array.len();
    // Fixed-width values: their native little-endian bytes, back to back.
    macro_rules! fixed {
        ($arr:ty) => {{
            let a = downcast::<$arr>(array)?;
            (0..len).for_each(|i| out.extend_from_slice(&a.value(i).to_le_bytes()));
        }};
    }
    // Values narrower than their Parquet physical type: extended to the
    // physical width, then little-endian. Each is widened in its own
    // signedness (a signed value sign-extended, an unsigned one zero-extended),
    // so the physical integer holds the bits the annotation tells a reader to
    // read back at the column's true width.
    macro_rules! widened {
        ($arr:ty, $physical:ty) => {{
            let a = downcast::<$arr>(array)?;
            (0..len).for_each(|i| out.extend_from_slice(&(a.value(i) as $physical).to_le_bytes()));
        }};
    }
    // BYTE_ARRAY values: a 4-byte LE length prefix then the bytes.
    macro_rules! byte_array {
        ($arr:ty, |$value:ident| $bytes:expr) => {{
            let a = downcast::<$arr>(array)?;
            (0..len).for_each(|i| {
                let $value = a.value(i);
                let bytes: &[u8] = $bytes;
                out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
                out.extend_from_slice(bytes);
            });
        }};
    }
    match array.data_type() {
        // A boolean is one bit, least significant bit first, which is the
        // layout arrow already holds the values in; `sliced` is the bitmap
        // realigned to start at bit zero, as it does unless the array is a
        // slice of a larger one.
        DataType::Boolean => {
            let a = downcast::<BooleanArray>(array)?;
            out.extend_from_slice(&a.values().sliced()[..len.div_ceil(8)]);
        }
        // The two narrow signed widths widen to the INT32 that is Parquet's
        // narrowest integer.
        DataType::Int8 => widened!(Int8Array, i32),
        DataType::Int16 => widened!(Int16Array, i32),
        DataType::Int32 => fixed!(Int32Array),
        DataType::Int64 => fixed!(Int64Array),
        // An unsigned value writes into the signed physical type of its width,
        // widening the two narrow ones the same way.
        DataType::UInt8 => widened!(UInt8Array, u32),
        DataType::UInt16 => widened!(UInt16Array, u32),
        DataType::UInt32 => fixed!(UInt32Array),
        DataType::UInt64 => fixed!(UInt64Array),
        // A date writes the day count its INT32 storage holds.
        DataType::Date32 => fixed!(Date32Array),
        DataType::Timestamp(TimeUnit::Microsecond, _) => fixed!(TimestampMicrosecondArray),
        DataType::Float32 => fixed!(Float32Array),
        DataType::Float64 => fixed!(Float64Array),
        // A decimal writes the narrowest storage its precision allows (see
        // `decimal_write_storage`): INT32/INT64 little-endian like the
        // primitives above, or the wide form's 16 big-endian two's-complement
        // bytes. Every value fits the narrowed integer because the column's
        // precision bounds it.
        DataType::Decimal64(precision, _) => {
            let a = downcast::<Decimal64Array>(array)?;
            match crate::decimal_write_storage(*precision) {
                DecimalWriteStorage::Int32 => {
                    (0..len).for_each(|i| out.extend_from_slice(&(a.value(i) as i32).to_le_bytes()))
                }
                DecimalWriteStorage::Int64 => {
                    (0..len).for_each(|i| out.extend_from_slice(&a.value(i).to_le_bytes()))
                }
                DecimalWriteStorage::FixedLen => (0..len)
                    .for_each(|i| out.extend_from_slice(&(a.value(i) as i128).to_be_bytes())),
            }
        }
        DataType::Decimal128(precision, _) => {
            let a = downcast::<Decimal128Array>(array)?;
            match crate::decimal_write_storage(*precision) {
                DecimalWriteStorage::Int32 => {
                    (0..len).for_each(|i| out.extend_from_slice(&(a.value(i) as i32).to_le_bytes()))
                }
                DecimalWriteStorage::Int64 => {
                    (0..len).for_each(|i| out.extend_from_slice(&(a.value(i) as i64).to_le_bytes()))
                }
                DecimalWriteStorage::FixedLen => {
                    (0..len).for_each(|i| out.extend_from_slice(&a.value(i).to_be_bytes()))
                }
            }
        }
        DataType::Utf8 => byte_array!(StringArray, |value| value.as_bytes()),
        DataType::Utf8View => byte_array!(StringViewArray, |value| value.as_bytes()),
        DataType::BinaryView => byte_array!(BinaryViewArray, |value| value),
        other => return Err(WriteError::UnsupportedType(other.clone())),
    }
    Ok(())
}

fn downcast<A: 'static>(array: &dyn Array) -> WriteResult<&A> {
    array
        .as_any()
        .downcast_ref::<A>()
        .ok_or(WriteError::Downcast {
            expected: std::any::type_name::<A>(),
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A string-view value longer than 12 bytes lives in an external buffer, not
    /// inline in the view; the encoder must follow the view to it and write the
    /// full bytes (length prefix + data), not the inline prefix.
    #[test]
    fn encodes_long_string_view_from_its_buffer() {
        let short = "ab";
        let long = "this string is well past twelve bytes";
        let array = StringViewArray::from(vec![short, long]);

        let mut out = Vec::new();
        encode_into(&array, &mut out).unwrap();

        let mut expected = Vec::new();
        for s in [short, long] {
            expected.extend_from_slice(&(s.len() as u32).to_le_bytes());
            expected.extend_from_slice(s.as_bytes());
        }
        assert_eq!(out, expected);
    }

    /// A wide decimal value encodes as its unscaled integer's 16 big-endian
    /// two's-complement bytes, with no length prefix; a negative value is
    /// sign-filled from the left.
    #[test]
    fn encodes_wide_decimals_as_16_big_endian_bytes() {
        let array = Decimal128Array::from(vec![12345_i128, -2_i128])
            .with_precision_and_scale(20, 2)
            .unwrap();

        let mut out = Vec::new();
        encode_into(&array, &mut out).unwrap();

        // 12345 = 0x3039 right-aligned in 16 bytes; -2 = 0xff..fe.
        let mut expected = vec![0u8; 14];
        expected.extend_from_slice(&[0x30, 0x39]);
        expected.extend_from_slice(&[0xff; 15]);
        expected.push(0xfe);
        assert_eq!(out, expected);
    }

    /// A narrow decimal value encodes little-endian in the integer width its
    /// precision selects: 8 bytes up to 18 digits, 4 bytes up to 9.
    #[test]
    fn encodes_narrow_decimals_as_little_endian_ints() {
        let int64_form = Decimal64Array::from(vec![12345_i64, -2_i64])
            .with_precision_and_scale(10, 2)
            .unwrap();
        let int32_form = Decimal64Array::from(vec![12345_i64, -2_i64])
            .with_precision_and_scale(5, 2)
            .unwrap();

        let mut int64_out = Vec::new();
        encode_into(&int64_form, &mut int64_out).unwrap();
        let mut int32_out = Vec::new();
        encode_into(&int32_form, &mut int32_out).unwrap();

        let mut expected64 = 12345_i64.to_le_bytes().to_vec();
        expected64.extend_from_slice(&(-2_i64).to_le_bytes());
        assert_eq!(int64_out, expected64);
        let mut expected32 = 12345_i32.to_le_bytes().to_vec();
        expected32.extend_from_slice(&(-2_i32).to_le_bytes());
        assert_eq!(int32_out, expected32);
    }

    /// The precision picks the storage, not the carrier: a `Decimal128` array
    /// at a narrow precision writes the same bytes as the `Decimal64` form.
    #[test]
    fn precision_decides_the_decimal_storage_regardless_of_carrier() {
        let narrow = Decimal64Array::from(vec![12345_i64, -2_i64])
            .with_precision_and_scale(10, 2)
            .unwrap();
        let wide = Decimal128Array::from(vec![12345_i128, -2_i128])
            .with_precision_and_scale(10, 2)
            .unwrap();

        let mut narrow_out = Vec::new();
        encode_into(&narrow, &mut narrow_out).unwrap();
        let mut wide_out = Vec::new();
        encode_into(&wide, &mut wide_out).unwrap();

        assert_eq!(narrow_out, wide_out);
    }

    /// A variant's `metadata`/`value` leaves are binary, and encode as
    /// length-prefixed bytes just like strings do.
    #[test]
    fn encodes_binary_view_values() {
        let array = BinaryViewArray::from(vec![b"\x01\x02".as_slice(), b"\xff".as_slice()]);

        let mut out = Vec::new();
        encode_into(&array, &mut out).unwrap();

        assert_eq!(out, vec![2, 0, 0, 0, 1, 2, 1, 0, 0, 0, 255]);
    }
}
