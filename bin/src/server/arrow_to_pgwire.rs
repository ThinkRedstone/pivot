//! Encode an Arrow [`RecordBatch`] into pgwire `FieldInfo` schemas and
//! `DataRow` streams.
//!
//! Rows are encoded in either wire format, fixed by the batch type: a
//! [`PGTextRowBatch`] carries pgwire's [`FieldFormat::Text`], what `psql` and
//! the simple protocol use, and a [`PGBinaryRowBatch`] carries
//! [`FieldFormat::Binary`], what a client binding binary result columns over
//! the extended protocol asks for. Each Arrow array kind has a dedicated
//! `encode` arm that pushes its native Rust type into [`DataRowEncoder`];
//! anything we can't represent precisely is shipped as `text` so at least
//! the value arrives.

use std::sync::Arc;

use arrow_array::{
    Array, BooleanArray, Date32Array, Decimal64Array, Decimal128Array, Float32Array, Float64Array,
    Int8Array, Int16Array, Int32Array, Int64Array, IntervalMonthDayNanoArray, RecordBatch,
    StringArray, StringViewArray, TimestampMicrosecondArray, UInt8Array, UInt16Array, UInt32Array,
    UInt64Array,
};
use arrow_schema::{DataType, SchemaRef};
use pg_interval_2::Interval;
use planner::types::render_interval;

use pgwire::api::Type;
use pgwire::api::results::{DataRowEncoder, FieldFormat, FieldInfo};
use pgwire::messages::data::DataRow;

/// Build a pgwire row schema from an Arrow [`SchemaRef`]. Each Arrow column
/// becomes one [`FieldInfo`] with the closest matching Postgres [`Type`].
pub fn build_field_info(schema: &SchemaRef, format: FieldFormat) -> Arc<Vec<FieldInfo>> {
    let fields = schema
        .fields()
        .iter()
        .map(|f| {
            FieldInfo::new(
                f.name().clone(),
                None,
                None,
                pg_type_for_arrow(f.data_type()),
                format,
            )
        })
        .collect();
    Arc::new(fields)
}

pub struct PGRowBatch {
    pub rows: Vec<DataRow>,
    pub fields: Arc<Vec<FieldInfo>>,
}

/// A row batch whose type fixes the wire format its cells are encoded in.
pub trait PGRowEncoding: dispatch::OutputBatch + Into<PGRowBatch> {
    const FORMAT: FieldFormat;
}

/// Rows encoded in the text wire format.
pub struct PGTextRowBatch(PGRowBatch);

/// Rows encoded in the binary wire format.
pub struct PGBinaryRowBatch(PGRowBatch);

macro_rules! row_encodings {
    ($( ($batch:ident, $format:ident) ),+ $(,)?) => {$(
        impl PGRowEncoding for $batch {
            const FORMAT: FieldFormat = FieldFormat::$format;
        }

        impl dispatch::OutputBatch for $batch {
            fn from_record_batch(batch: RecordBatch) -> Self {
                Self(encode_batch(&batch, Self::FORMAT))
            }
        }

        impl From<$batch> for PGRowBatch {
            fn from(batch: $batch) -> Self {
                batch.0
            }
        }
    )+};
}

row_encodings! {
    (PGTextRowBatch, Text),
    (PGBinaryRowBatch, Binary),
}

fn encode_batch(batch: &RecordBatch, format: FieldFormat) -> PGRowBatch {
    let fields = build_field_info(&batch.schema(), format);
    let mut encoder = DataRowEncoder::new(fields.clone());
    let mut rows = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        for col in 0..batch.num_columns() {
            encode_cell(&mut encoder, batch.column(col).as_ref(), row, format);
        }
        rows.push(encoder.take_row());
    }
    PGRowBatch { rows, fields }
}

/// Generates [`pg_type_for_arrow`] (schema: arrow type -> Postgres OID) and
/// [`encode_cell`] (data: one cell -> pgwire field) from one table, so the OID a
/// column advertises and the value its rows carry are declared together and
/// can't drift.
///
/// Two sections, because Postgres has no unsigned or one-byte integers:
///
/// - `direct`: value encoded straight from `array.value(row)`, OID one-to-one.
///   Row shape `(arrow DataType, arrow array type, Postgres Type)`.
/// - `widened`: an int that widens to a wider signed type on the wire.
///   Row adds the signed cast target: `(DataType, array type, Postgres Type, signed)`.
///
/// The arms that fit neither shape stay spelled out, because their text and
/// binary encodings come from different values:
///
/// - `UInt64`: no signed type fits, so it ships as TEXT via `to_string`.
/// - `Decimal64`/`Decimal128`: NUMERIC, as `value_as_string` in text and via
///   [`encode_numeric_binary`] in binary.
/// - `Date32`/`Timestamp`/`Interval`: rendered as a Postgres server prints
///   them in text, and handed to the encoder as chrono / [`Interval`] values
///   in binary.
/// - The advertise-only `Binary`/`LargeUtf8`, which fall through to the
///   best-effort text fallback in `encode_cell`.
macro_rules! arrow_pg_types {
    (
        direct: [ $( ($dt:ident, $array:ty, $pg:ident) ),+ $(,)? ],
        widened: [ $( ($udt:ident, $uarray:ty, $upg:ident, $signed:ty) ),+ $(,)? ] $(,)?
    ) => {
        /// Map an Arrow [`DataType`] to its closest Postgres [`Type`]. Values we
        /// can't represent precisely fall back to [`Type::TEXT`].
        fn pg_type_for_arrow(dt: &DataType) -> Type {
            match dt {
                $( DataType::$dt => Type::$pg, )+
                $( DataType::$udt => Type::$upg, )+
                // UInt64 can't fit any signed type, so it ships as TEXT.
                DataType::UInt64 => Type::TEXT,
                DataType::Decimal64(_, _) | DataType::Decimal128(_, _) => Type::NUMERIC,
                // Temporal types reinterpreted from the executor's int columns at
                // the output boundary (see `encode_cell`). A zone-carrying
                // timestamp is Postgres's TIMESTAMPTZ; a zone-less one its
                // TIMESTAMP.
                DataType::Date32 => Type::DATE,
                DataType::Timestamp(_, None) => Type::TIMESTAMP,
                DataType::Timestamp(_, Some(_)) => Type::TIMESTAMPTZ,
                DataType::Interval(_) => Type::INTERVAL,
                // Advertised but encoded via the text fallback below.
                DataType::LargeUtf8 => Type::TEXT,
                DataType::Binary | DataType::LargeBinary | DataType::BinaryView => Type::BYTEA,
                _ => Type::TEXT,
            }
        }

        /// Encode a single cell. We always feed the encoder a typed Rust value
        /// (or `Option::None` for SQL NULL) so pgwire's `ToSqlText` impl handles
        /// the formatting, with no manual `to_string()` round-trips. The binary
        /// path writes the value's `ToSql` bytes without checking them against
        /// the advertised type, so every value handed to it must already be
        /// that type's binary representation.
        fn encode_cell(encoder: &mut DataRowEncoder, arr: &dyn Array, row: usize, format: FieldFormat) {
            if arr.is_null(row) {
                // Type is irrelevant for null encoding: the encoder writes -1 length.
                let _ = encoder.encode_field::<Option<&str>>(&None);
                return;
            }
            let _ = match arr.data_type() {
                $( DataType::$dt => encoder
                    .encode_field(&arr.as_any().downcast_ref::<$array>().unwrap().value(row)), )+
                // Ints widen to a Postgres integer type the value fits.
                $( DataType::$udt => encoder
                    .encode_field(&(arr.as_any().downcast_ref::<$uarray>().unwrap().value(row) as $signed)), )+
                DataType::UInt64 => encoder
                    .encode_field(&arr.as_any().downcast_ref::<UInt64Array>().unwrap().value(row).to_string()),
                // Decimals in either carrier width (Decimal128 is e.g. the SUM
                // aggregate output). `value_as_string` renders the
                // integer/decimal with its scale applied; scale 0 yields a
                // plain integer like "12345".
                // In binary they are built from the stored integer and scale.
                DataType::Decimal64(_, scale) => {
                    let decimals = arr.as_any().downcast_ref::<Decimal64Array>().unwrap();
                    match format {
                        FieldFormat::Text => encoder.encode_field(&decimals.value_as_string(row)),
                        FieldFormat::Binary => encoder
                            .encode_field(&encode_numeric_binary(i128::from(decimals.value(row)), *scale)),
                    }
                }
                DataType::Decimal128(_, scale) => {
                    let decimals = arr.as_any().downcast_ref::<Decimal128Array>().unwrap();
                    match format {
                        FieldFormat::Text => encoder.encode_field(&decimals.value_as_string(row)),
                        FieldFormat::Binary => {
                            encoder.encode_field(&encode_numeric_binary(decimals.value(row), *scale))
                        }
                    }
                }
                // Temporal columns render as their ISO string, Postgres's text
                // wire form for DATE/TIMESTAMP. A timestamp's fraction goes
                // through [`trim_fraction`] to match what a server prints.
                // Handing the encoder the chrono value itself would also
                // encode, but pgwire's `ToSqlText` fixes the fraction at six
                // digits, so every whole second would arrive as `.000000`.
                // In binary the chrono value is handed over as is.
                DataType::Date32 => {
                    let date = arr.as_any().downcast_ref::<Date32Array>().unwrap().value_as_date(row);
                    match format {
                        FieldFormat::Text => encoder.encode_field(&date.map(|d| d.to_string())),
                        FieldFormat::Binary => encoder.encode_field(&date),
                    }
                }
                DataType::Timestamp(_, None) => {
                    let timestamp = arr.as_any().downcast_ref::<TimestampMicrosecondArray>().unwrap()
                        .value_as_datetime(row);
                    match format {
                        FieldFormat::Text => encoder.encode_field(&timestamp.map(|t| trim_fraction(t.to_string()))),
                        FieldFormat::Binary => encoder.encode_field(&timestamp),
                    }
                }
                // A TIMESTAMPTZ renders in the session time zone, which is
                // fixed to UTC, with the offset suffix a Postgres server
                // prints: `2020-01-01 00:00:00+00`. `value_as_datetime` reads
                // the stored count as the UTC instant it is.
                DataType::Timestamp(_, Some(_)) => {
                    let timestamp = arr.as_any().downcast_ref::<TimestampMicrosecondArray>().unwrap()
                        .value_as_datetime(row);
                    match format {
                        FieldFormat::Text => encoder
                            .encode_field(&timestamp.map(|t| format!("{}+00", trim_fraction(t.to_string())))),
                        FieldFormat::Binary => encoder.encode_field(&timestamp.map(|t| t.and_utc())),
                    }
                }
                // A time span renders as the text a Postgres server prints for
                // an INTERVAL. In binary it is Postgres's microsecond
                // resolution, dropping any sub-microsecond part.
                DataType::Interval(_) => {
                    let interval =
                        arr.as_any().downcast_ref::<IntervalMonthDayNanoArray>().unwrap().value(row);
                    match format {
                        FieldFormat::Text => encoder.encode_field(&render_interval(interval)),
                        FieldFormat::Binary => encoder.encode_field(&Interval::new(
                            interval.months,
                            interval.days,
                            interval.nanoseconds / 1_000,
                        )),
                    }
                }
                // Best-effort fallback: stringify and ship as text.
                _ => encoder.encode_field(&format!("{:?}", arr.slice(row, 1))),
            };
        }
    };
}

/// Encode the decimal `unscaled × 10^-scale` (an Arrow decimal's stored
/// integer and scale) in Postgres's binary NUMERIC format.
///
/// NUMERIC writes the value in groups of four decimal digits, so essentially
/// in base 10,000, with the groups aligned on the decimal point. The format,
/// every field big-endian:
///
/// 1. `ndigits: i16`: how many groups follow.
/// 2. `weight: i16`: how far shifted the first group is (the power of 10,000
///    it is multiplied by).
/// 3. `sign: u16`: positive (`0x0000`) or negative (`0x4000`).
/// 4. `dscale: u16`: how many digits to display after the dot.
/// 5. `digits: [i16]`: the groups, most significant first, each the value of
///    four decimal digits (`0..10000`). Zero groups at either end are left
///    out, so zero has none.
///
/// For example, `123.45` (unscaled `12345`, scale 2) has weight 0, dscale 2
/// and the groups `[123, 4500]`: `123 × 10000^0 + 4500 × 10000^-1`.
///
/// When the scale isn't a multiple of four, the lowest group needs zeros
/// appended to fill its four digits (the `45` above becomes `4500`). Only that
/// group is multiplied up, not the whole value, since multiplying an `i128`
/// near its maximum would overflow.
fn encode_numeric_binary(unscaled: i128, scale: i8) -> Vec<u8> {
    const POSITIVE: u16 = 0x0000;
    const NEGATIVE: u16 = 0x4000;
    const BASE: u128 = 10_000;
    let scale = i32::from(scale);
    let pad = (-scale).rem_euclid(4) as u32;
    // Groups below the decimal point; negative when a negative scale puts
    // whole zero groups below the lowest stored digit.
    let fraction_groups = (scale + pad as i32) / 4;

    // Least significant first.
    let mut groups: Vec<i16> = Vec::new();
    let mut rest = unscaled.unsigned_abs();
    if pad > 0 {
        let low_digits = 10u128.pow(4 - pad);
        groups.push((rest % low_digits * 10u128.pow(pad)) as i16);
        rest /= low_digits;
    }
    while rest > 0 {
        groups.push((rest % BASE) as i16);
        rest /= BASE;
    }
    let weight = groups.len() as i32 - 1 - fraction_groups;
    let trailing_zeros = groups.iter().take_while(|&&group| group == 0).count();
    groups.drain(..trailing_zeros);
    groups.reverse();
    // Zero carries no digits; Postgres writes it with weight 0 and a positive
    // sign.
    let (weight, sign) = match (groups.is_empty(), unscaled < 0) {
        (true, _) => (0, POSITIVE),
        (false, true) => (weight, NEGATIVE),
        (false, false) => (weight, POSITIVE),
    };

    let mut bytes = Vec::with_capacity(8 + 2 * groups.len());
    bytes.extend_from_slice(&(groups.len() as i16).to_be_bytes());
    bytes.extend_from_slice(&(weight as i16).to_be_bytes());
    bytes.extend_from_slice(&sign.to_be_bytes());
    bytes.extend_from_slice(&(scale.max(0) as u16).to_be_bytes());
    for group in groups {
        bytes.extend_from_slice(&group.to_be_bytes());
    }
    bytes
}

/// Drop the trailing zeros from a rendered timestamp's fractional second, and
/// the fraction itself when nothing is left, which is how a Postgres server
/// prints one: `00:00:00.5`, not `00:00:00.500`. Chrono pads the fraction to
/// three, six or nine digits, so its rendering needs this to read the same as a
/// real server's.
fn trim_fraction(rendered: String) -> String {
    let Some((instant, fraction)) = rendered.split_once('.') else {
        return rendered;
    };
    let fraction = fraction.trim_end_matches('0');
    if fraction.is_empty() {
        instant.to_string()
    } else {
        format!("{instant}.{fraction}")
    }
}

arrow_pg_types! {
    direct: [
        (Boolean,  BooleanArray,    BOOL),
        (Int16,    Int16Array,      INT2),
        (Int32,    Int32Array,      INT4),
        (Int64,    Int64Array,      INT8),
        (Float32,  Float32Array,    FLOAT4),
        (Float64,  Float64Array,    FLOAT8),
        (Utf8,     StringArray,     TEXT),
        (Utf8View, StringViewArray, TEXT),
    ],
    widened: [
        (Int8,   Int8Array,   INT2, i16),
        (UInt8,  UInt8Array,  INT2, i16),
        (UInt16, UInt16Array, INT4, i32),
        (UInt32, UInt32Array, INT8, i64),
    ],
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::ArrayRef;
    use arrow_schema::{Field, Schema};
    use dispatch::OutputBatch;

    fn schema(fields: Vec<(&str, DataType)>) -> SchemaRef {
        Arc::new(Schema::new(
            fields
                .into_iter()
                .map(|(n, dt)| Field::new(n, dt, true))
                .collect::<Vec<_>>(),
        ))
    }

    fn batch(fields: Vec<(&str, DataType)>, columns: Vec<ArrayRef>) -> RecordBatch {
        RecordBatch::try_new(schema(fields), columns).unwrap()
    }

    /// Decode a text-format `DataRow`'s wire bytes into `Vec<Option<String>>`.
    /// Each field is a big-endian i32 length followed by that many UTF-8 bytes;
    /// length -1 means SQL NULL.
    fn decode_text_row(row: &DataRow) -> Vec<Option<String>> {
        let bytes: &[u8] = &row.data;
        let mut out = Vec::with_capacity(row.field_count as usize);
        let mut i = 0;
        for _ in 0..row.field_count {
            let len = i32::from_be_bytes(bytes[i..i + 4].try_into().unwrap());
            i += 4;
            if len < 0 {
                out.push(None);
            } else {
                let n = len as usize;
                out.push(Some(String::from_utf8(bytes[i..i + n].to_vec()).unwrap()));
                i += n;
            }
        }
        out
    }

    fn rows(batch: &RecordBatch) -> Vec<Vec<Option<String>>> {
        let pg: PGRowBatch = PGTextRowBatch::from_record_batch(batch.clone()).into();
        pg.rows.iter().map(decode_text_row).collect()
    }

    /// Decode a binary-format `DataRow` into each field's raw bytes, `None`
    /// for SQL NULL.
    fn decode_binary_row(row: &DataRow) -> Vec<Option<Vec<u8>>> {
        let bytes: &[u8] = &row.data;
        let mut out = Vec::with_capacity(row.field_count as usize);
        let mut i = 0;
        for _ in 0..row.field_count {
            let len = i32::from_be_bytes(bytes[i..i + 4].try_into().unwrap());
            i += 4;
            if len < 0 {
                out.push(None);
            } else {
                let n = len as usize;
                out.push(Some(bytes[i..i + n].to_vec()));
                i += n;
            }
        }
        out
    }

    /// The single column of each row of `batch`, binary-encoded.
    fn binary_cells(batch: &RecordBatch) -> Vec<Option<Vec<u8>>> {
        let pg: PGRowBatch = PGBinaryRowBatch::from_record_batch(batch.clone()).into();
        pg.rows
            .iter()
            .map(|row| decode_binary_row(row).remove(0))
            .collect()
    }

    /// NUMERIC binary bytes from header fields and base-10000 digits.
    fn numeric(weight: i16, sign: u16, scale: u16, digits: &[i16]) -> Vec<u8> {
        let mut bytes = (digits.len() as i16).to_be_bytes().to_vec();
        bytes.extend_from_slice(&weight.to_be_bytes());
        bytes.extend_from_slice(&sign.to_be_bytes());
        bytes.extend_from_slice(&scale.to_be_bytes());
        for digit in digits {
            bytes.extend_from_slice(&digit.to_be_bytes());
        }
        bytes
    }

    #[test]
    fn build_field_info_maps_names_and_types() {
        let schema = schema(vec![
            ("flag", DataType::Boolean),
            ("small", DataType::Int16),
            ("big", DataType::Int64),
            ("u64", DataType::UInt64),
            ("f", DataType::Float64),
            ("s", DataType::Utf8),
        ]);

        let fields = build_field_info(&schema, FieldFormat::Text);

        assert_eq!(fields.len(), 6);
        assert_eq!(fields[0].name(), "flag");
        assert_eq!(fields[0].datatype(), &Type::BOOL);
        assert_eq!(fields[0].format(), FieldFormat::Text);
        assert_eq!(fields[1].datatype(), &Type::INT2);
        assert_eq!(fields[2].datatype(), &Type::INT8);
        assert_eq!(fields[3].datatype(), &Type::TEXT);
        assert_eq!(fields[4].datatype(), &Type::FLOAT8);
        assert_eq!(fields[5].datatype(), &Type::TEXT);
    }

    #[test]
    fn build_field_info_widens_unsigned_types() {
        let schema = schema(vec![
            ("a", DataType::UInt8),
            ("b", DataType::UInt16),
            ("c", DataType::UInt32),
        ]);

        let fields = build_field_info(&schema, FieldFormat::Text);

        assert_eq!(fields[0].datatype(), &Type::INT2);
        assert_eq!(fields[1].datatype(), &Type::INT4);
        assert_eq!(fields[2].datatype(), &Type::INT8);
    }

    #[test]
    fn encode_batch_row_count_matches_input() {
        let col: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3, 4]));
        let b = batch(vec![("v", DataType::Int32)], vec![col]);

        let decoded = rows(&b);

        assert_eq!(decoded.len(), 4);
    }

    #[test]
    fn encode_batch_empty_produces_no_rows() {
        let col: ArrayRef = Arc::new(Int32Array::from(Vec::<i32>::new()));
        let b = batch(vec![("v", DataType::Int32)], vec![col]);

        let decoded = rows(&b);

        assert_eq!(decoded.len(), 0);
    }

    #[test]
    fn encode_batch_encodes_integers_as_text() {
        let col: ArrayRef = Arc::new(Int32Array::from(vec![0, -7, 12345]));
        let b = batch(vec![("v", DataType::Int32)], vec![col]);

        let decoded = rows(&b);

        assert_eq!(decoded[0], vec![Some("0".to_string())]);
        assert_eq!(decoded[1], vec![Some("-7".to_string())]);
        assert_eq!(decoded[2], vec![Some("12345".to_string())]);
    }

    #[test]
    fn encode_batch_encodes_booleans_as_t_and_f() {
        let col: ArrayRef = Arc::new(BooleanArray::from(vec![true, false]));
        let b = batch(vec![("v", DataType::Boolean)], vec![col]);

        let decoded = rows(&b);

        assert_eq!(decoded[0], vec![Some("t".to_string())]);
        assert_eq!(decoded[1], vec![Some("f".to_string())]);
    }

    #[test]
    fn encode_batch_encodes_strings_verbatim() {
        let col: ArrayRef = Arc::new(StringArray::from(vec!["hello", "", "héllo"]));
        let b = batch(vec![("v", DataType::Utf8)], vec![col]);

        let decoded = rows(&b);

        assert_eq!(decoded[0], vec![Some("hello".to_string())]);
        assert_eq!(decoded[1], vec![Some("".to_string())]);
        assert_eq!(decoded[2], vec![Some("héllo".to_string())]);
    }

    #[test]
    fn encode_batch_encodes_floats() {
        let col: ArrayRef = Arc::new(Float64Array::from(vec![1.5, -0.25]));
        let b = batch(vec![("v", DataType::Float64)], vec![col]);

        let decoded = rows(&b);

        assert_eq!(decoded[0], vec![Some("1.5".to_string())]);
        assert_eq!(decoded[1], vec![Some("-0.25".to_string())]);
    }

    #[test]
    fn encode_batch_encodes_nulls() {
        let col: ArrayRef = Arc::new(Int32Array::from(vec![Some(1), None, Some(3)]));
        let b = batch(vec![("v", DataType::Int32)], vec![col]);

        let decoded = rows(&b);

        assert_eq!(decoded[0], vec![Some("1".to_string())]);
        assert_eq!(decoded[1], vec![None]);
        assert_eq!(decoded[2], vec![Some("3".to_string())]);
    }

    #[test]
    fn encode_batch_encodes_decimal128_as_integer() {
        let col: ArrayRef = Arc::new(
            Decimal128Array::from(vec![15_i128, 9_223_372_036_854_775_807_i128 * 3])
                .with_precision_and_scale(38, 0)
                .unwrap(),
        );
        let b = batch(vec![("sum", DataType::Decimal128(38, 0))], vec![col]);

        let decoded = rows(&b);

        assert_eq!(decoded[0], vec![Some("15".to_string())]);
        assert_eq!(decoded[1], vec![Some("27670116110564327421".to_string())]);
    }

    #[test]
    fn decimal128_maps_to_numeric() {
        let fields = build_field_info(
            &schema(vec![("s", DataType::Decimal128(38, 0))]),
            FieldFormat::Text,
        );
        assert_eq!(fields[0].datatype(), &Type::NUMERIC);
    }

    #[test]
    fn encode_batch_encodes_decimal64_with_scale_applied() {
        let col: ArrayRef = Arc::new(
            Decimal64Array::from(vec![12345_i64, -250])
                .with_precision_and_scale(10, 2)
                .unwrap(),
        );
        let b = batch(vec![("v", DataType::Decimal64(10, 2))], vec![col]);

        let decoded = rows(&b);

        assert_eq!(decoded[0], vec![Some("123.45".to_string())]);
        assert_eq!(decoded[1], vec![Some("-2.50".to_string())]);
    }

    #[test]
    fn decimal64_maps_to_numeric() {
        let fields = build_field_info(
            &schema(vec![("v", DataType::Decimal64(10, 2))]),
            FieldFormat::Text,
        );
        assert_eq!(fields[0].datatype(), &Type::NUMERIC);
    }

    #[test]
    fn date32_maps_to_date_and_renders_iso() {
        let col: ArrayRef = Arc::new(Date32Array::from(vec![0, 7]));
        let b = batch(vec![("d", DataType::Date32)], vec![col]);

        let fields = build_field_info(&b.schema(), FieldFormat::Text);
        let decoded = rows(&b);

        assert_eq!(fields[0].datatype(), &Type::DATE);
        assert_eq!(decoded[0], vec![Some("1970-01-01".to_string())]);
        assert_eq!(decoded[1], vec![Some("1970-01-08".to_string())]);
    }

    /// A fractional second renders with its trailing zeros dropped, and a
    /// pre-epoch instant keeps the fraction of the second it falls in. Every
    /// expectation here is what a PostgreSQL 16 server prints for the same
    /// value.
    #[test]
    fn fractional_timestamps_render_as_a_postgres_server_prints_them() {
        let col: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![
            1_577_836_800_500_000,
            1_577_836_800_120_000,
            1_577_836_800_000_001,
            -1_500_000,
        ]));
        let b = batch(
            vec![(
                "t",
                DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None),
            )],
            vec![col],
        );

        let decoded = rows(&b);

        assert_eq!(decoded[0], vec![Some("2020-01-01 00:00:00.5".to_string())]);
        assert_eq!(decoded[1], vec![Some("2020-01-01 00:00:00.12".to_string())]);
        assert_eq!(
            decoded[2],
            vec![Some("2020-01-01 00:00:00.000001".to_string())]
        );
        assert_eq!(decoded[3], vec![Some("1969-12-31 23:59:58.5".to_string())]);
    }

    #[test]
    fn interval_maps_to_interval_and_renders_dates_and_a_clock() {
        let col: ArrayRef = Arc::new(IntervalMonthDayNanoArray::from(vec![
            arrow_buffer::IntervalMonthDayNano::new(0, 2, 11_045_500_000_000),
            arrow_buffer::IntervalMonthDayNano::new(0, 0, -5_000_000_000),
            arrow_buffer::IntervalMonthDayNano::new(14, 0, 0),
        ]));
        let b = batch(
            vec![(
                "age",
                DataType::Interval(arrow_schema::IntervalUnit::MonthDayNano),
            )],
            vec![col],
        );

        let fields = build_field_info(&b.schema(), FieldFormat::Text);
        let decoded = rows(&b);

        assert_eq!(fields[0].datatype(), &Type::INTERVAL);
        assert_eq!(decoded[0], vec![Some("2 days 03:04:05.5".to_string())]);
        assert_eq!(decoded[1], vec![Some("-00:00:05".to_string())]);
        assert_eq!(decoded[2], vec![Some("1 year 2 mons".to_string())]);
    }

    #[test]
    fn timestamp_maps_to_timestamp_and_renders_iso() {
        let col: ArrayRef = Arc::new(TimestampMicrosecondArray::from(vec![
            0,
            90_000_000,
            1_700_000_000_123_456,
        ]));
        let b = batch(
            vec![(
                "t",
                DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, None),
            )],
            vec![col],
        );

        let fields = build_field_info(&b.schema(), FieldFormat::Text);
        let decoded = rows(&b);

        assert_eq!(fields[0].datatype(), &Type::TIMESTAMP);
        assert_eq!(decoded[0], vec![Some("1970-01-01 00:00:00".to_string())]);
        assert_eq!(decoded[1], vec![Some("1970-01-01 00:01:30".to_string())]);
        // A sub-second part renders with the timestamp and a whole second
        // renders without one, which is how Postgres prints them.
        assert_eq!(
            decoded[2],
            vec![Some("2023-11-14 22:13:20.123456".to_string())]
        );
    }

    /// A zone-carrying timestamp advertises TIMESTAMPTZ and renders in the UTC
    /// session with the offset suffix a Postgres server prints, fraction
    /// trimming included.
    #[test]
    fn timestamptz_maps_to_timestamptz_and_renders_with_a_utc_offset() {
        let col: ArrayRef = Arc::new(
            TimestampMicrosecondArray::from(vec![0, 1_577_836_800_500_000]).with_timezone("UTC"),
        );
        let b = batch(
            vec![(
                "t",
                DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some("UTC".into())),
            )],
            vec![col],
        );

        let fields = build_field_info(&b.schema(), FieldFormat::Text);
        let decoded = rows(&b);

        assert_eq!(fields[0].datatype(), &Type::TIMESTAMPTZ);
        assert_eq!(decoded[0], vec![Some("1970-01-01 00:00:00+00".to_string())]);
        assert_eq!(
            decoded[1],
            vec![Some("2020-01-01 00:00:00.5+00".to_string())]
        );
    }

    #[test]
    fn encode_batch_encodes_uint64_as_text() {
        let col: ArrayRef = Arc::new(UInt64Array::from(vec![u64::MAX, 0]));
        let b = batch(vec![("v", DataType::UInt64)], vec![col]);

        let decoded = rows(&b);

        assert_eq!(decoded[0], vec![Some(u64::MAX.to_string())]);
        assert_eq!(decoded[1], vec![Some("0".to_string())]);
    }

    #[test]
    fn encode_batch_handles_multiple_columns_per_row() {
        let id: ArrayRef = Arc::new(Int32Array::from(vec![1, 2]));
        let name: ArrayRef = Arc::new(StringArray::from(vec!["a", "b"]));
        let flag: ArrayRef = Arc::new(BooleanArray::from(vec![true, false]));
        let b = batch(
            vec![
                ("id", DataType::Int32),
                ("name", DataType::Utf8),
                ("flag", DataType::Boolean),
            ],
            vec![id, name, flag],
        );

        let decoded = rows(&b);

        assert_eq!(
            decoded[0],
            vec![
                Some("1".to_string()),
                Some("a".to_string()),
                Some("t".to_string()),
            ],
        );
        assert_eq!(
            decoded[1],
            vec![
                Some("2".to_string()),
                Some("b".to_string()),
                Some("f".to_string()),
            ],
        );
    }

    #[test]
    fn binary_batches_advertise_the_binary_format() {
        let col: ArrayRef = Arc::new(Int32Array::from(vec![1]));
        let b = batch(vec![("v", DataType::Int32)], vec![col]);

        let pg: PGRowBatch = PGBinaryRowBatch::from_record_batch(b).into();

        assert_eq!(pg.fields[0].format(), FieldFormat::Binary);
    }

    #[test]
    fn binary_encodes_integers_big_endian_and_nulls_as_absent() {
        let ints: ArrayRef = Arc::new(Int32Array::from(vec![Some(-2), None]));
        let tiny: ArrayRef = Arc::new(Int8Array::from(vec![Some(7), None]));
        let b = batch(
            vec![("i", DataType::Int32), ("t", DataType::Int8)],
            vec![ints, tiny],
        );

        let pg: PGRowBatch = PGBinaryRowBatch::from_record_batch(b).into();
        let decoded: Vec<_> = pg.rows.iter().map(decode_binary_row).collect();

        assert_eq!(
            decoded[0],
            vec![
                Some((-2i32).to_be_bytes().to_vec()),
                Some(7i16.to_be_bytes().to_vec())
            ]
        );
        assert_eq!(decoded[1], vec![None, None]);
    }

    #[test]
    fn binary_encodes_dates_and_timestamps_from_the_2000_epoch() {
        let dates: ArrayRef = Arc::new(Date32Array::from(vec![10_957, 0]));
        let timestamps: ArrayRef = Arc::new(
            TimestampMicrosecondArray::from(vec![946_684_800_000_001, 0]).with_timezone("UTC"),
        );
        let b = batch(
            vec![
                ("d", DataType::Date32),
                (
                    "t",
                    DataType::Timestamp(arrow_schema::TimeUnit::Microsecond, Some("UTC".into())),
                ),
            ],
            vec![dates, timestamps],
        );

        let pg: PGRowBatch = PGBinaryRowBatch::from_record_batch(b).into();
        let decoded: Vec<_> = pg.rows.iter().map(decode_binary_row).collect();

        assert_eq!(
            decoded[0],
            vec![
                Some(0i32.to_be_bytes().to_vec()),
                Some(1i64.to_be_bytes().to_vec())
            ]
        );
        assert_eq!(
            decoded[1],
            vec![
                Some((-10_957i32).to_be_bytes().to_vec()),
                Some((-946_684_800_000_000i64).to_be_bytes().to_vec()),
            ]
        );
    }

    #[test]
    fn binary_encodes_intervals_as_micros_days_months() {
        let col: ArrayRef = Arc::new(IntervalMonthDayNanoArray::from(vec![
            arrow_buffer::IntervalMonthDayNano::new(1, 2, 3_500_000_999),
        ]));
        let b = batch(
            vec![(
                "age",
                DataType::Interval(arrow_schema::IntervalUnit::MonthDayNano),
            )],
            vec![col],
        );

        let cells = binary_cells(&b);

        let mut expected = 3_500_000i64.to_be_bytes().to_vec();
        expected.extend_from_slice(&2i32.to_be_bytes());
        expected.extend_from_slice(&1i32.to_be_bytes());
        assert_eq!(cells[0], Some(expected));
    }

    #[test]
    fn binary_encodes_decimals_as_base_10000_numerics() {
        let col: ArrayRef = Arc::new(
            Decimal64Array::from(vec![12345_i64, -250, 0, 1_000_000, 5])
                .with_precision_and_scale(10, 2)
                .unwrap(),
        );
        let b = batch(vec![("v", DataType::Decimal64(10, 2))], vec![col]);

        let cells = binary_cells(&b);

        assert_eq!(cells[0], Some(numeric(0, 0x0000, 2, &[123, 4500])));
        assert_eq!(cells[1], Some(numeric(0, 0x4000, 2, &[2, 5000])));
        assert_eq!(cells[2], Some(numeric(0, 0x0000, 2, &[])));
        assert_eq!(cells[3], Some(numeric(1, 0x0000, 2, &[1])));
        assert_eq!(cells[4], Some(numeric(-1, 0x0000, 2, &[500])));
    }

    /// A value with more digits than its declared precision still arrives
    /// whole; only the stored integer and scale decide the encoding.
    #[test]
    fn binary_encodes_decimals_beyond_their_precision() {
        let col: ArrayRef = Arc::new(
            Decimal128Array::from(vec![i128::MAX, i128::MIN + 1])
                .with_precision_and_scale(38, 0)
                .unwrap(),
        );
        let b = batch(vec![("sum", DataType::Decimal128(38, 0))], vec![col]);

        let cells = binary_cells(&b);

        let digits = [170, 1411, 8346, 469, 2317, 3168, 7303, 7158, 8410, 5727];
        assert_eq!(cells[0], Some(numeric(9, 0x0000, 0, &digits)));
        assert_eq!(cells[1], Some(numeric(9, 0x4000, 0, &digits)));
    }

    #[test]
    fn binary_encodes_negative_scale_decimals_as_integers() {
        let col: ArrayRef = Arc::new(
            Decimal64Array::from(vec![5_i64, 123])
                .with_precision_and_scale(10, -5)
                .unwrap(),
        );
        let b = batch(vec![("v", DataType::Decimal64(10, -5))], vec![col]);

        let cells = binary_cells(&b);

        assert_eq!(cells[0], Some(numeric(1, 0x0000, 0, &[50])));
        assert_eq!(cells[1], Some(numeric(1, 0x0000, 0, &[1230])));
    }

    #[test]
    fn binary_encodes_wide_decimal128_integers() {
        let col: ArrayRef = Arc::new(
            Decimal128Array::from(vec![27_670_116_110_564_327_421_i128])
                .with_precision_and_scale(38, 0)
                .unwrap(),
        );
        let b = batch(vec![("sum", DataType::Decimal128(38, 0))], vec![col]);

        let cells = binary_cells(&b);

        assert_eq!(
            cells[0],
            Some(numeric(4, 0x0000, 0, &[2767, 116, 1105, 6432, 7421]))
        );
    }
}
