//! Encode an Arrow [`RecordBatch`] into pgwire `FieldInfo` schemas and
//! `DataRow` streams.
//!
//! Only **text format** is emitted (pgwire's [`FieldFormat::Text`]). It is the
//! universally supported representation, requires no per-type binary encoder,
//! and matches what `psql` prints. Each Arrow array kind has a dedicated
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
use planner::types::render_interval;

use pgwire::api::Type;
use pgwire::api::results::{DataRowEncoder, FieldFormat, FieldInfo};
use pgwire::messages::data::DataRow;

/// Build a pgwire row schema from an Arrow [`SchemaRef`]. Each Arrow column
/// becomes one [`FieldInfo`] with the closest matching Postgres [`Type`].
pub fn build_field_info(schema: &SchemaRef) -> Arc<Vec<FieldInfo>> {
    let fields = schema
        .fields()
        .iter()
        .map(|f| {
            FieldInfo::new(
                f.name().clone(),
                None,
                None,
                pg_type_for_arrow(f.data_type()),
                FieldFormat::Text,
            )
        })
        .collect();
    Arc::new(fields)
}

pub struct PGRowBatch {
    pub rows: Vec<DataRow>,
    pub fields: Arc<Vec<FieldInfo>>,
}

impl From<RecordBatch> for PGRowBatch {
    fn from(batch: RecordBatch) -> Self {
        let fields = build_field_info(&batch.schema());
        let mut encoder = DataRowEncoder::new(fields.clone());
        let mut rows = Vec::with_capacity(batch.num_rows());
        for row in 0..batch.num_rows() {
            for col in 0..batch.num_columns() {
                encode_cell(&mut encoder, batch.column(col).as_ref(), row);
            }
            rows.push(encoder.take_row());
        }
        Self { rows, fields }
    }
}

impl dispatch::OutputBatch for PGRowBatch {
    fn from_record_batch(batch: RecordBatch) -> dispatch::UnaryResult<Self> {
        Ok(batch.into())
    }
}

/// Generates [`pg_type_for_arrow`] (schema: arrow type -> Postgres OID) and
/// [`encode_cell`] (data: one cell -> pgwire field) from one table, so the OID a
/// column advertises and the value its rows carry are declared together and
/// can't drift.
///
/// Two sections, because Postgres has no unsigned types:
///
/// - `direct`: value encoded straight from `array.value(row)`, OID one-to-one.
///   Row shape `(arrow DataType, arrow array type, Postgres Type)`.
/// - `widened`: an unsigned int that widens to the next signed type on the wire.
///   Row adds the signed cast target: `(DataType, array type, Postgres Type, signed)`.
///
/// The arms that fit neither shape stay spelled out: `UInt64` (no signed type
/// fits, so it ships as TEXT via `to_string`), `Decimal128` (NUMERIC via
/// `value_as_string`), and the advertise-only `Binary`/`LargeUtf8` that fall
/// through to the best-effort text fallback in `encode_cell`.
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
        /// the formatting, with no manual `to_string()` round-trips.
        fn encode_cell(encoder: &mut DataRowEncoder, arr: &dyn Array, row: usize) {
            if arr.is_null(row) {
                // Type is irrelevant for null encoding: the encoder writes -1 length.
                let _ = encoder.encode_field::<Option<&str>>(&None);
                return;
            }
            let _ = match arr.data_type() {
                $( DataType::$dt => encoder
                    .encode_field(&arr.as_any().downcast_ref::<$array>().unwrap().value(row)), )+
                // Unsigned ints widen to the next signed type so the value fits.
                $( DataType::$udt => encoder
                    .encode_field(&(arr.as_any().downcast_ref::<$uarray>().unwrap().value(row) as $signed)), )+
                DataType::UInt64 => encoder
                    .encode_field(&arr.as_any().downcast_ref::<UInt64Array>().unwrap().value(row).to_string()),
                // Decimals in either carrier width (Decimal128 is e.g. the SUM
                // aggregate output). `value_as_string` renders the
                // integer/decimal with its scale applied; scale 0 yields a
                // plain integer like "12345".
                DataType::Decimal64(_, _) => encoder.encode_field(
                    &arr.as_any().downcast_ref::<Decimal64Array>().unwrap().value_as_string(row),
                ),
                DataType::Decimal128(_, _) => encoder.encode_field(
                    &arr.as_any().downcast_ref::<Decimal128Array>().unwrap().value_as_string(row),
                ),
                // Temporal columns render as their ISO string, Postgres's text
                // wire form for DATE/TIMESTAMP. A timestamp's fraction goes
                // through [`trim_fraction`] to match what a server prints.
                // Handing the encoder the chrono value itself would also
                // encode, but pgwire's `ToSqlText` fixes the fraction at six
                // digits, so every whole second would arrive as `.000000`.
                // Only text format is emitted (see the module doc), so nothing
                // here rests on the encoder's binary path.
                DataType::Date32 => encoder.encode_field(
                    &arr.as_any().downcast_ref::<Date32Array>().unwrap()
                        .value_as_date(row).map(|d| d.to_string()),
                ),
                DataType::Timestamp(_, None) => encoder.encode_field(
                    &arr.as_any().downcast_ref::<TimestampMicrosecondArray>().unwrap()
                        .value_as_datetime(row).map(|t| trim_fraction(t.to_string())),
                ),
                // A TIMESTAMPTZ renders in the session time zone, which is
                // fixed to UTC, with the offset suffix a Postgres server
                // prints: `2020-01-01 00:00:00+00`. `value_as_datetime` reads
                // the stored count as the UTC instant it is.
                DataType::Timestamp(_, Some(_)) => encoder.encode_field(
                    &arr.as_any().downcast_ref::<TimestampMicrosecondArray>().unwrap()
                        .value_as_datetime(row)
                        .map(|t| format!("{}+00", trim_fraction(t.to_string()))),
                ),
                // A time span renders as the text a Postgres server prints for
                // an INTERVAL.
                DataType::Interval(_) => encoder.encode_field(&render_interval(
                    arr.as_any().downcast_ref::<IntervalMonthDayNanoArray>().unwrap().value(row),
                )),
                // Best-effort fallback: stringify and ship as text.
                _ => encoder.encode_field(&format!("{:?}", arr.slice(row, 1))),
            };
        }
    };
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
        (Int8,     Int8Array,       INT2),
        (Int16,    Int16Array,      INT2),
        (Int32,    Int32Array,      INT4),
        (Int64,    Int64Array,      INT8),
        (Float32,  Float32Array,    FLOAT4),
        (Float64,  Float64Array,    FLOAT8),
        (Utf8,     StringArray,     TEXT),
        (Utf8View, StringViewArray, TEXT),
    ],
    widened: [
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
        let pg: PGRowBatch = batch.clone().into();
        pg.rows.iter().map(decode_text_row).collect()
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

        let fields = build_field_info(&schema);

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

        let fields = build_field_info(&schema);

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
        let fields = build_field_info(&schema(vec![("s", DataType::Decimal128(38, 0))]));
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
        let fields = build_field_info(&schema(vec![("v", DataType::Decimal64(10, 2))]));
        assert_eq!(fields[0].datatype(), &Type::NUMERIC);
    }

    #[test]
    fn date32_maps_to_date_and_renders_iso() {
        let col: ArrayRef = Arc::new(Date32Array::from(vec![0, 7]));
        let b = batch(vec![("d", DataType::Date32)], vec![col]);

        let fields = build_field_info(&b.schema());
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

        let fields = build_field_info(&b.schema());
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

        let fields = build_field_info(&b.schema());
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

        let fields = build_field_info(&b.schema());
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
}
