//! The one mapping between Parquet's on-disk types and the executor's arrow
//! [`DataType`].
//!
//! The read path ([`parquet_to_arrow`], used when loading a file's footer) and
//! the write path ([`arrow_to_parquet_physical`], used when stamping a Parquet
//! footer) are inverses kept side by side, so a type added to one is added to
//! the other instead of the two drifting apart.
//!
//! ```text
//!   arrow DataType    parquet physical   annotation              direction
//!   --------------    ----------------   ----------------------  ---------
//!   Boolean           BOOLEAN            -                       read+write
//!   Int8              INT32              Integer{ 8, signed}     read+write
//!   UInt8             INT32              Integer{ 8, unsigned}   read+write
//!   Int16             INT32              Integer{16, signed}     read+write
//!   UInt16            INT32              Integer{16, unsigned}   read+write
//!   Int32             INT32              -                       read+write
//!   UInt32            INT32              Integer{32, unsigned}   read+write
//!   Int64             INT64              -                       read+write
//!   UInt64            INT64              Integer{64, unsigned}   read+write
//!   Float32           FLOAT              -                       read+write
//!   Float64           DOUBLE             -                       read+write
//!   Utf8 / Utf8View   BYTE_ARRAY         converted UTF8 / String read+write
//!   Date32            INT32              Date                    read+write
//!   Timestamp(Micro)  INT64              Timestamp{MICROS}       read+write
//!   Timestamp(Micro,  INT64              Timestamp{MICROS, UTC}  read+write
//!     "UTC")
//!   BinaryView        BYTE_ARRAY         unannotated             read+write
//!   Decimal64(p,s)    INT32              Decimal (p <= 9)        read+write
//!   Decimal64(p,s)    INT64              Decimal (p <= 18)       read+write
//!   Decimal64/128     FIXED_LEN_B_A(n)   Decimal (p <= 38)       read, write p > 18
//! ```
//!
//! A decimal's arrow carrier follows its declared precision: `Decimal64` up
//! to 18 digits, `Decimal128` beyond, matching the planner's
//! `physical_arrow_type` so a file-derived schema and a declared schema agree.
//!
//! Pivot's own writer only emits the column set its encoder supports (booleans,
//! Int32/Int64/the unsigned widths/Date32/Timestamp/Float32/Float64/strings/
//! binary/decimals), so [`arrow_to_parquet_physical`] errors on the rest.
//! `writing::type_tests` round trips every one of them.
//!
//! The write path is two halves: [`arrow_to_parquet_physical`] for the physical
//! type, [`arrow_to_annotation`] for everything the schema element says on top
//! of it.

use arrow_schema::{DataType, TimeUnit};

use super::table::{Error, Result};
use crate::thrift::footer::{LogicalType, SchemaElement};
use crate::thrift::general::{TimeUnit as ParquetTimeUnit, Type};

// Parquet physical type ids, named off the same thrift enum the writer emits,
// so read and write reference one definition rather than bare integers.
const BOOLEAN: i32 = Type::BOOLEAN as i32;
const INT32: i32 = Type::INT32 as i32;
const INT64: i32 = Type::INT64 as i32;
const FLOAT: i32 = Type::FLOAT as i32;
const DOUBLE: i32 = Type::DOUBLE as i32;
const BYTE_ARRAY: i32 = Type::BYTE_ARRAY as i32;
const FIXED_LEN_BYTE_ARRAY: i32 = Type::FIXED_LEN_BYTE_ARRAY as i32;

/// The fixed-length array width pivot writes a wide decimal with: the full 16
/// bytes of the unscaled 128-bit integer, big-endian, per the Parquet spec.
pub const DECIMAL_FIXED_LEN: i32 = 16;

/// The physical storage pivot writes a decimal column with, chosen by its
/// declared precision: the narrowest of the spec's three decimal storages
/// that holds every value of that precision.
///
/// INT32 and INT64 store the unscaled integer little-endian like every other
/// primitive (and decode through the same contiguous fast paths); only the
/// wide FIXED_LEN_BYTE_ARRAY form is big-endian.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DecimalWriteStorage {
    /// Up to 9 digits: 4 little-endian bytes.
    Int32,
    /// Up to 18 digits: 8 little-endian bytes.
    Int64,
    /// Beyond 18 digits: [`DECIMAL_FIXED_LEN`] big-endian bytes.
    FixedLen,
}

/// The largest decimal precision an INT32-stored unscaled integer holds.
pub const DECIMAL_INT32_MAX_PRECISION: u8 = 9;
/// The largest decimal precision an INT64-stored unscaled integer holds.
pub const DECIMAL_INT64_MAX_PRECISION: u8 = 18;

/// The storage for a decimal of `precision` digits (see [`DecimalWriteStorage`]).
pub fn decimal_write_storage(precision: u8) -> DecimalWriteStorage {
    if precision <= DECIMAL_INT32_MAX_PRECISION {
        DecimalWriteStorage::Int32
    } else if precision <= DECIMAL_INT64_MAX_PRECISION {
        DecimalWriteStorage::Int64
    } else {
        DecimalWriteStorage::FixedLen
    }
}

impl DecimalWriteStorage {
    /// The Parquet physical type id this storage writes.
    pub fn physical_type(self) -> i32 {
        match self {
            DecimalWriteStorage::Int32 => INT32,
            DecimalWriteStorage::Int64 => INT64,
            DecimalWriteStorage::FixedLen => FIXED_LEN_BYTE_ARRAY,
        }
    }

    /// Bytes one value occupies on disk.
    pub fn byte_width(self) -> usize {
        match self {
            DecimalWriteStorage::Int32 => 4,
            DecimalWriteStorage::Int64 => 8,
            DecimalWriteStorage::FixedLen => DECIMAL_FIXED_LEN as usize,
        }
    }

    /// The schema element's `type_length`, set only for the fixed-length form.
    pub fn type_length(self) -> Option<i32> {
        match self {
            DecimalWriteStorage::FixedLen => Some(DECIMAL_FIXED_LEN),
            DecimalWriteStorage::Int32 | DecimalWriteStorage::Int64 => None,
        }
    }
}

// Parquet `ConvertedType` ids: the legacy width/temporal annotation, read
// alongside the modern `LogicalType`.
pub(crate) const CONVERTED_UTF8: i32 = 0;
pub(crate) const CONVERTED_UINT_8: i32 = 11;
pub(crate) const CONVERTED_UINT_16: i32 = 12;
pub(crate) const CONVERTED_UINT_32: i32 = 13;
pub(crate) const CONVERTED_UINT_64: i32 = 14;
pub(crate) const CONVERTED_INT_8: i32 = 15;
pub(crate) const CONVERTED_INT_16: i32 = 16;
pub(crate) const CONVERTED_INT_32: i32 = 17;
/// Legacy temporal annotations (superseded by `LogicalType::Date`/`Timestamp`).
pub(crate) const CONVERTED_DATE: i32 = 6;
pub(crate) const CONVERTED_TIMESTAMP_MILLIS: i32 = 9;
pub(crate) const CONVERTED_TIMESTAMP_MICROS: i32 = 10;
/// Legacy decimal annotation (superseded by `LogicalType::Decimal`), read
/// together with the schema element's own `precision`/`scale` fields.
pub(crate) const CONVERTED_DECIMAL: i32 = 5;

/// Read path: a Parquet leaf's physical type (plus optional logical/converted
/// annotations) -> the arrow [`DataType`] the executor decodes it into.
///
/// A BYTE_ARRAY is a string only when the file says so (a UTF8/String
/// annotation); an unannotated one is binary, exactly as the spec defines it.
/// Files in the wild do store text without the annotation, but that fact isn't
/// in the file: it comes from the table's declared schema, and
/// `apply_declared_types` (in `table.rs`) retypes such columns to string when
/// the table declares them VARCHAR.
pub fn parquet_to_arrow(elem: &SchemaElement) -> Result<DataType> {
    let pt = elem.physical_type.ok_or_else(|| {
        Error::UnsupportedType("leaf schema element missing physical type".to_string())
    })?;
    let converted_type = elem.converted_type;
    let logical_type = elem.logical_type.as_ref();

    if let Some((precision, scale)) = decimal_annotation(elem)? {
        return match pt {
            INT32 => decimal_arrow(precision, scale, DECIMAL_INT32_MAX_PRECISION as i32),
            INT64 => decimal_arrow(precision, scale, DECIMAL_INT64_MAX_PRECISION as i32),
            FIXED_LEN_BYTE_ARRAY => {
                decimal_arrow(precision, scale, flba_decimal_max_precision(elem)?)
            }
            _ => Err(Error::UnsupportedType(format!(
                "DECIMAL on parquet physical type {pt}"
            ))),
        };
    }

    match pt {
        BOOLEAN => Ok(DataType::Boolean),
        INT32 => int32_arrow(converted_type, logical_type),
        INT64 => int64_arrow(converted_type, logical_type),
        FLOAT => Ok(DataType::Float32),
        DOUBLE => Ok(DataType::Float64),
        BYTE_ARRAY => {
            let is_string = converted_type == Some(CONVERTED_UTF8)
                || matches!(logical_type, Some(LogicalType::String));
            if is_string {
                Ok(DataType::Utf8View)
            } else {
                Ok(DataType::BinaryView)
            }
        }
        _ => Err(Error::UnsupportedType(format!(
            "Parquet physical type {pt}"
        ))),
    }
}

/// The precision and scale of a leaf's DECIMAL annotation, or `None` when the
/// leaf is not a decimal. The modern `LogicalType` wins; the legacy
/// `ConvertedType` reads the schema element's own precision/scale fields.
fn decimal_annotation(elem: &SchemaElement) -> Result<Option<(i32, i32)>> {
    if let Some(LogicalType::Decimal { scale, precision }) = elem.logical_type {
        return Ok(Some((precision, scale)));
    }
    if elem.converted_type == Some(CONVERTED_DECIMAL) {
        let (Some(precision), Some(scale)) = (elem.precision, elem.scale) else {
            return Err(Error::UnsupportedType(
                "DECIMAL converted type without precision/scale".to_string(),
            ));
        };
        return Ok(Some((precision, scale)));
    }
    Ok(None)
}

/// The arrow type for a decimal leaf, after checking the annotation fits its
/// physical storage (`max_precision` decimal digits). The carrier follows
/// the declared precision: `Decimal64` up to 18 digits, `Decimal128` beyond.
fn decimal_arrow(precision: i32, scale: i32, max_precision: i32) -> Result<DataType> {
    if precision < 1 || precision > max_precision || scale < 0 || scale > precision {
        return Err(Error::UnsupportedType(format!(
            "DECIMAL({precision},{scale}) does not fit its physical storage \
             (max precision {max_precision})"
        )));
    }
    if precision <= planner::types::MAX_DECIMAL64_PRECISION as i32 {
        Ok(DataType::Decimal64(precision as u8, scale as i8))
    } else {
        Ok(DataType::Decimal128(precision as u8, scale as i8))
    }
}

/// The largest decimal precision a FIXED_LEN_BYTE_ARRAY of the element's
/// `type_length` bytes can hold (the spec's `floor(log10(2^(8*len - 1) - 1))`
/// table). Lengths beyond 16 bytes exceed the 128-bit unscaled integer the
/// executor carries and are rejected.
fn flba_decimal_max_precision(elem: &SchemaElement) -> Result<i32> {
    const MAX_PRECISION_BY_LEN: [i32; 16] =
        [2, 4, 6, 9, 11, 14, 16, 18, 21, 23, 26, 28, 31, 33, 36, 38];
    match elem.type_length {
        Some(len @ 1..=16) => Ok(MAX_PRECISION_BY_LEN[(len - 1) as usize]),
        Some(len) => Err(Error::UnsupportedType(format!(
            "DECIMAL with FIXED_LEN_BYTE_ARRAY length {len}"
        ))),
        None => Err(Error::UnsupportedType(
            "FIXED_LEN_BYTE_ARRAY without a type_length".to_string(),
        )),
    }
}

/// The arrow type of an INT32 leaf, resolved from its annotation: a `DATE` is
/// `Date32` (an INT32 day count), the integer annotations give the narrow widths,
/// and a bare INT32 is `Int32`. The modern `LogicalType` wins; the legacy
/// `ConvertedType` is the fallback. Decimals are resolved before this is
/// reached. Errors on an annotation we don't support (e.g. a TIME width)
/// rather than silently reading raw `Int32`.
fn int32_arrow(
    converted_type: Option<i32>,
    logical_type: Option<&LogicalType>,
) -> Result<DataType> {
    match logical_type {
        Some(LogicalType::Date) => Ok(DataType::Date32),
        Some(LogicalType::Integer {
            bit_width,
            is_signed,
        }) => match (*bit_width, *is_signed) {
            (8, true) => Ok(DataType::Int8),
            (8, false) => Ok(DataType::UInt8),
            (16, true) => Ok(DataType::Int16),
            (16, false) => Ok(DataType::UInt16),
            (32, true) => Ok(DataType::Int32),
            (32, false) => Ok(DataType::UInt32),
            _ => Err(Error::UnsupportedType(format!(
                "INT32 column with integer width {bit_width} (signed: {is_signed})"
            ))),
        },
        // No (recognized) LogicalType: fall back to the legacy ConvertedType.
        _ => match converted_type {
            Some(CONVERTED_DATE) => Ok(DataType::Date32),
            Some(CONVERTED_INT_8) => Ok(DataType::Int8),
            Some(CONVERTED_UINT_8) => Ok(DataType::UInt8),
            Some(CONVERTED_INT_16) => Ok(DataType::Int16),
            Some(CONVERTED_UINT_16) => Ok(DataType::UInt16),
            Some(CONVERTED_INT_32) => Ok(DataType::Int32),
            Some(CONVERTED_UINT_32) => Ok(DataType::UInt32),
            // A bare INT32 is a plain signed 32-bit int.
            None => Ok(DataType::Int32),
            Some(other) => Err(Error::UnsupportedType(format!(
                "INT32 converted type {other}"
            ))),
        },
    }
}

/// The arrow type of an INT64 leaf: a microsecond `TIMESTAMP` is
/// `Timestamp(Microsecond)`, the one resolution a pivot timestamp counts in;
/// otherwise a plain `Int64`.
///
/// A file that declares any other unit errors rather than being read at a
/// resolution it was not written in, which would misplace every value by the
/// ratio between the two units.
///
/// The annotation's `is_adjusted_to_utc` flag decides zone-ness: an adjusted
/// column holds UTC instants (`TIMESTAMP WITH TIME ZONE`), an unadjusted one
/// zone-less wall times. The legacy `TIMESTAMP_MICROS` spelling carries no
/// flag and the format defines it as the adjusted one, so a legacy-only file
/// reads as instants too.
fn int64_arrow(
    converted_type: Option<i32>,
    logical_type: Option<&LogicalType>,
) -> Result<DataType> {
    match logical_type {
        Some(LogicalType::Timestamp {
            unit,
            is_adjusted_to_utc,
        }) => match unit {
            ParquetTimeUnit::MICROS => Ok(if *is_adjusted_to_utc {
                DataType::Timestamp(
                    TimeUnit::Microsecond,
                    Some(planner::types::UTC_TIMEZONE.into()),
                )
            } else {
                DataType::Timestamp(TimeUnit::Microsecond, None)
            }),
            other => Err(Error::UnsupportedType(format!(
                "TIMESTAMP column in {other:?}; pivot reads microsecond timestamps"
            ))),
        },
        Some(LogicalType::Integer {
            bit_width,
            is_signed,
        }) => match (*bit_width, *is_signed) {
            (64, true) => Ok(DataType::Int64),
            (64, false) => Ok(DataType::UInt64),
            _ => Err(Error::UnsupportedType(format!(
                "INT64 column with integer width {bit_width} (signed: {is_signed})"
            ))),
        },
        // No (recognized) LogicalType: fall back to the legacy ConvertedType.
        _ => match converted_type {
            Some(CONVERTED_TIMESTAMP_MICROS) => Ok(DataType::Timestamp(
                TimeUnit::Microsecond,
                Some(planner::types::UTC_TIMEZONE.into()),
            )),
            Some(CONVERTED_TIMESTAMP_MILLIS) => Err(Error::UnsupportedType(
                "TIMESTAMP_MILLIS column; pivot reads microsecond timestamps".to_string(),
            )),
            Some(CONVERTED_UINT_64) => Ok(DataType::UInt64),
            _ => Ok(DataType::Int64),
        },
    }
}

/// Everything a leaf's schema element says about its type beyond the physical
/// type [`arrow_to_parquet_physical`] gives it, which is what
/// [`parquet_to_arrow`] reads back.
///
/// Each annotation is written in both spellings: the modern `LogicalType` and
/// the legacy `ConvertedType`, so a reader that only knows the older one
/// resolves the column too.
#[derive(Debug, Default, PartialEq)]
pub struct LeafAnnotation {
    pub logical_type: Option<LogicalType>,
    pub converted_type: Option<i32>,
    /// Set only by the fixed-length decimal storage, whose declared length is
    /// how a reader knows how many bytes a value takes.
    pub type_length: Option<i32>,
    pub precision: Option<i32>,
    pub scale: Option<i32>,
}

/// Write path: how `data_type` is annotated in its schema element. A type whose
/// physical type describes it on its own — a signed integer, a float, a binary
/// leaf — needs no annotation and takes the empty default.
pub fn arrow_to_annotation(data_type: &DataType) -> LeafAnnotation {
    match data_type {
        // A BYTE_ARRAY is binary by default, so text says so.
        DataType::Utf8 | DataType::Utf8View => LeafAnnotation {
            converted_type: Some(CONVERTED_UTF8),
            ..LeafAnnotation::default()
        },
        // Without this a date reads back as the plain INT32 day count it is
        // stored as.
        DataType::Date32 => LeafAnnotation {
            logical_type: Some(LogicalType::Date),
            converted_type: Some(CONVERTED_DATE),
            ..LeafAnnotation::default()
        },
        // `is_adjusted_to_utc` carries the column's zone-ness: true for a
        // `TIMESTAMP WITH TIME ZONE` column of UTC instants, false for a
        // zone-less one, whose values a reader must not shift into its
        // session's zone. The legacy spelling has no such flag and the format
        // defines it as the adjusted one, so a reader old enough to resolve
        // only that one reads both as UTC; it is written anyway because every
        // writer in the ecosystem writes it for a microsecond timestamp, and
        // leaving it off would have that reader see a bare INT64 instead.
        DataType::Timestamp(TimeUnit::Microsecond, zone) => LeafAnnotation {
            logical_type: Some(LogicalType::Timestamp {
                unit: ParquetTimeUnit::MICROS,
                is_adjusted_to_utc: zone.is_some(),
            }),
            converted_type: Some(CONVERTED_TIMESTAMP_MICROS),
            ..LeafAnnotation::default()
        },
        // A decimal carries its full description: the precision and scale that
        // place the point, and the fixed length its widest storage declares.
        DataType::Decimal64(precision, scale) | DataType::Decimal128(precision, scale) => {
            let (precision, scale) = (*precision as i32, *scale as i32);
            LeafAnnotation {
                logical_type: Some(LogicalType::Decimal { scale, precision }),
                converted_type: Some(CONVERTED_DECIMAL),
                type_length: decimal_write_storage(precision as u8).type_length(),
                precision: Some(precision),
                scale: Some(scale),
            }
        }
        // A leaf narrower than the physical type it stores its bits in says so
        // here, which is what keeps a signed value from reading back as the
        // wider type it was widened into. For an unsigned leaf the annotation
        // also carries the signedness, without which a value past the signed
        // maximum reads back negative.
        DataType::Int8 => integer_annotation(8, true, CONVERTED_INT_8),
        DataType::Int16 => integer_annotation(16, true, CONVERTED_INT_16),
        DataType::UInt8 => integer_annotation(8, false, CONVERTED_UINT_8),
        DataType::UInt16 => integer_annotation(16, false, CONVERTED_UINT_16),
        DataType::UInt32 => integer_annotation(32, false, CONVERTED_UINT_32),
        DataType::UInt64 => integer_annotation(64, false, CONVERTED_UINT_64),
        _ => LeafAnnotation::default(),
    }
}

/// The INTEGER annotation naming a leaf's true width and signedness.
fn integer_annotation(bit_width: i8, is_signed: bool, converted_type: i32) -> LeafAnnotation {
    LeafAnnotation {
        logical_type: Some(LogicalType::Integer {
            bit_width,
            is_signed,
        }),
        converted_type: Some(converted_type),
        ..LeafAnnotation::default()
    }
}

/// Write path: the Parquet physical type id for an arrow column. The inverse of
/// [`parquet_to_arrow`] over the writable subset; errors on a type the encoder
/// doesn't emit.
///
/// A `BinaryView` writes as an unannotated BYTE_ARRAY, the spec's own
/// definition of binary, and what a variant's `metadata`/`value` leaves are.
/// Leaving off the UTF8 annotation is what keeps [`parquet_to_arrow`] reading it
/// back as binary rather than as text.
///
/// A decimal writes with the narrowest storage its precision allows
/// ([`decimal_write_storage`]); the schema element's `type_length`, decimal
/// annotation, and precision/scale complete the description on the writer
/// side.
pub fn arrow_to_parquet_physical(data_type: &DataType) -> Result<i32> {
    Ok(match data_type {
        DataType::Boolean => BOOLEAN,
        DataType::Int32 => INT32,
        DataType::Int64 => INT64,
        // Parquet has no unsigned physical type and no integer narrower than
        // INT32, so a column of either kind stores its bits in the signed
        // physical type of the next width up (sign-extended for a narrow signed
        // one, zero-extended for a narrow unsigned one) and the INTEGER
        // annotation ([`arrow_to_annotation`], stamped alongside it) tells a
        // reader the width and signedness to read those bits back at. A
        // `UInt32`/`UInt64` value above the signed maximum stores as a negative
        // physical value, which is exactly what the spec prescribes.
        DataType::Int8
        | DataType::Int16
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32 => INT32,
        DataType::UInt64 => INT64,
        // A date is its day count, stored as the INT32 the DATE annotation
        // (stamped alongside it) tells a reader to interpret.
        DataType::Date32 => INT32,
        DataType::Timestamp(TimeUnit::Microsecond, _) => INT64,
        DataType::Float32 => FLOAT,
        DataType::Float64 => DOUBLE,
        DataType::Utf8 | DataType::Utf8View | DataType::BinaryView => BYTE_ARRAY,
        DataType::Decimal64(precision, _) | DataType::Decimal128(precision, _) => {
            decimal_write_storage(*precision).physical_type()
        }
        other => {
            return Err(Error::UnsupportedType(format!("arrow type {other:?}")));
        }
    })
}
