//! Every arrow type the writer accepts, written through the real write pipeline
//! and read back both ways: through pivot's own reader, and through arrow-rs's
//! strict reader as the "another engine can read this" oracle. A type only
//! counts as supported when both agree with what went in.
//!
//! Each type appears in three shapes, because the encoder picks its encoding
//! from the values rather than from the type: a column that repeats itself takes
//! the dictionary, one whose values are all distinct is packed as differences (or
//! left plain where the type has no delta form), and one with nulls exercises
//! level assembly on top of either. The [`type_tests!`] table at the bottom turns
//! each into its own test, named `<type>::<shape>`.

use std::sync::Arc;

use crate::{ParquetTable, table_input};
use arrow::compute::cast;
use arrow_array::types::{
    Date32Type, Decimal64Type, Decimal128Type, Float32Type, Float64Type, Int8Type, Int16Type,
    Int32Type, Int64Type, TimestampMicrosecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{
    Array, ArrayRef, ArrowPrimitiveType, BinaryViewArray, BooleanArray, Date32Array,
    PrimitiveArray, RecordBatch, StringArray, StringViewArray, TimestampMicrosecondArray,
};
use arrow_schema::{DataType, Field, Schema, TimeUnit};
use dispatch::{Dispatch, Projection, values_input};

use super::{AssembledFile, encode_record_batches_spec};
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::Encoding;
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

const ROWS: usize = 5_000;

/// Distinct values a repeating column cycles through, well under the share of
/// the rows at which the encoder gives up on a dictionary.
const REPEATED_DISTINCT: usize = 50;

/// Ring buffers a test asks for. Each is faulted in at spin-up, so this is the
/// bulk of a test's cost: a handful of columns needs very few.
const RING_BUFFERS: usize = 16;

const DICTIONARY: &str = "RLE_DICTIONARY";
const PACKED: &str = "DELTA_BINARY_PACKED";
const PACKED_LENGTHS: &str = "DELTA_LENGTH_BYTE_ARRAY";
const PLAIN: &str = "PLAIN";

/// The three ways a column's values can fall, which is what the encoder chooses
/// its encoding from.
#[derive(Clone, Copy)]
enum Shape {
    Repeated,
    Distinct,
    Nullable,
}

/// How to build one type's column, and the encodings its repeated and its
/// distinct values take.
struct TypeSpec {
    repeated_encoding: &'static str,
    distinct_encoding: &'static str,
    build: Box<dyn Fn(Shape) -> ArrayRef>,
}

/// One column to round trip: the values that go in, and the encoding they should
/// come out under.
struct Case {
    name: String,
    values: ArrayRef,
    encoding: &'static str,
}

impl TypeSpec {
    fn case(&self, type_name: &str, shape: Shape, shape_name: &str) -> Case {
        Case {
            name: format!("{type_name}_{shape_name}"),
            values: (self.build)(shape),
            encoding: match shape {
                Shape::Repeated => self.repeated_encoding,
                Shape::Distinct | Shape::Nullable => self.distinct_encoding,
            },
        }
    }
}

/// A repeating column's value index for a row: long stretches of one value,
/// broken up by scattered ones, so the dictionary's index stream carries both
/// the runs and the bit-packed groups it can hold.
fn repeated_index(row: usize) -> usize {
    if row % 200 < 100 {
        7
    } else {
        row % REPEATED_DISTINCT
    }
}

/// Which rows a nullable shape leaves empty: both edges, since that is where
/// level assembly goes wrong, and a scattering in between.
fn is_null(row: usize) -> bool {
    row == 0 || row == ROWS - 1 || row % 7 == 3
}

/// A numeric type, from a function giving a row's value and one stamping the
/// finished array (which is where a decimal's precision and scale go on).
fn primitive<T: ArrowPrimitiveType>(
    distinct_encoding: &'static str,
    value: impl Fn(usize) -> T::Native + 'static,
    stamp: impl Fn(PrimitiveArray<T>) -> ArrayRef + 'static,
) -> TypeSpec {
    TypeSpec {
        repeated_encoding: DICTIONARY,
        distinct_encoding,
        build: Box::new(move |shape| {
            let array: PrimitiveArray<T> = (0..ROWS)
                .map(|row| match shape {
                    Shape::Repeated => Some(value(repeated_index(row))),
                    Shape::Distinct => Some(value(row)),
                    Shape::Nullable => (!is_null(row)).then(|| value(row)),
                })
                .collect();
            stamp(array)
        }),
    }
}

/// A byte-array type, whose values come from a row's string rather than from a
/// number. Their lengths vary, so the packed lengths are worth something.
fn bytes(build: impl Fn(Vec<Option<String>>) -> ArrayRef + 'static) -> TypeSpec {
    let value = |row: usize| format!("{row}-{}", "value".repeat(row % 17));
    TypeSpec {
        repeated_encoding: DICTIONARY,
        distinct_encoding: PACKED_LENGTHS,
        build: Box::new(move |shape| {
            let rows = (0..ROWS)
                .map(|row| match shape {
                    Shape::Repeated => Some(value(repeated_index(row))),
                    Shape::Distinct => Some(value(row)),
                    Shape::Nullable => (!is_null(row)).then(|| value(row)),
                })
                .collect();
            build(rows)
        }),
    }
}

/// The stamp for a type that needs none: the array is already what it is.
fn as_is<T: ArrowPrimitiveType>(array: PrimitiveArray<T>) -> ArrayRef {
    Arc::new(array) as ArrayRef
}

/// A boolean column, which has neither a dictionary form (the format reserves
/// it for the other types) nor a delta form: it stays PLAIN, one bit per value,
/// whatever its values do. Two values is all the type has, so the shapes differ
/// only in how long the runs of each are.
fn boolean() -> TypeSpec {
    TypeSpec {
        repeated_encoding: PLAIN,
        distinct_encoding: PLAIN,
        build: Box::new(|shape| {
            let value = |row: usize| row.is_multiple_of(3);
            Arc::new(BooleanArray::from_iter((0..ROWS).map(|row| match shape {
                Shape::Repeated => Some(value(repeated_index(row))),
                Shape::Distinct => Some(value(row)),
                Shape::Nullable => (!is_null(row)).then(|| value(row)),
            }))) as ArrayRef
        }),
    }
}

/// Write `cases` as the columns of one file, through the pipeline a catalog
/// INSERT uses, and return the file's bytes for the readers to check.
fn write_cases(dispatch: &Dispatch, cases: &[Case]) -> Vec<u8> {
    let fields: Vec<Field> = cases
        .iter()
        .map(|case| {
            Field::new(
                &case.name,
                case.values.data_type().clone(),
                case.values.null_count() > 0,
            )
        })
        .collect();
    let columns = cases.iter().map(|case| case.values.clone()).collect();
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();

    let schema = batch.schema();
    let spec = values_input(dispatch.dispatcher(), vec![batch]).record_batches();
    // A file's bytes live in ring memory the assembling worker owns, so they are
    // copied out there rather than followed to this thread.
    let mut files: Vec<Vec<u8>> =
        encode_record_batches_spec(spec, schema, Arc::from([]), Arc::from([]), ROWS, usize::MAX)
            .map_each(|file: AssembledFile| {
                file.bytes.runs().flatten().copied().collect::<Vec<u8>>()
            })
            .execute()
            .collect()
            .unwrap();
    assert_eq!(files.len(), 1, "the rows fit one file");
    files.pop().unwrap()
}

/// Read a file back through pivot's own reader.
fn read_with_pivot(dispatch: &Dispatch, bytes: Vec<u8>, columns: usize) -> RecordBatch {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("data.parquet");
    std::fs::write(&path, bytes).unwrap();

    let table = Arc::new(ParquetTable::from_files(dispatch.dispatcher(), &[path], &[]).unwrap());
    let batches = table_input(
        dispatch.dispatcher(),
        &table,
        Projection::all(columns),
        false,
    )
    .collect()
    .unwrap();
    arrow_select::concat::concat_batches(&batches[0].schema(), &batches).unwrap()
}

/// Read a file back through arrow-rs's strict reader, as another engine would.
fn read_with_arrow(bytes: Vec<u8>) -> RecordBatch {
    let reader = ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::from(bytes))
        .unwrap()
        .build()
        .unwrap();
    let batches: Vec<RecordBatch> = reader.map(|batch| batch.unwrap()).collect();
    arrow_select::concat::concat_batches(&batches[0].schema(), &batches).unwrap()
}

/// Compare a column read back against the values that went in. The two readers
/// resolve a type to whichever arrow carrier they prefer — a byte array comes
/// back as a view from one and as a string from the other — so the values are
/// compared at the width they were written, not at whatever carrier arrived.
fn assert_column_matches(case: &Case, read: &ArrayRef, reader: &str) {
    let cast_back = cast(read, case.values.data_type())
        .unwrap_or_else(|e| panic!("{}: cannot compare {read:?}: {e}", case.name));
    assert_eq!(
        cast_back.to_data(),
        case.values.to_data(),
        "{} did not survive {reader}",
        case.name
    );
}

/// The encodings a file's column chunk reports.
fn encodings(bytes: Vec<u8>, column: usize) -> Vec<String> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::from(bytes)).unwrap();
    reader
        .metadata()
        .row_group(0)
        .column(column)
        .encodings()
        .map(|encoding| encoding.to_string())
        .collect()
}

/// Write one column and check everything about it: both readers give back what
/// went in, and the encoder chose the encoding the values call for.
fn assert_round_trips(cases: Vec<Case>) {
    let dispatch = Dispatch::spin_up(2, RING_BUFFERS, None);
    let bytes = write_cases(&dispatch, &cases);

    let pivot = read_with_pivot(&dispatch, bytes.clone(), cases.len());
    let arrow = read_with_arrow(bytes.clone());

    for (column, case) in cases.iter().enumerate() {
        assert_eq!(pivot.num_rows(), ROWS);
        assert_column_matches(case, pivot.column(column), "pivot's reader");
        assert_column_matches(case, arrow.column(column), "arrow-rs's reader");
        let took = encodings(bytes.clone(), column);
        assert!(
            took.contains(&case.encoding.to_string()),
            "{} took {took:?}, not {}",
            case.name,
            case.encoding
        );
        // A dictionary page is itself PLAIN, so the assertion above cannot tell
        // a plain column from a dictionary-encoded one on its own.
        assert_eq!(
            case.encoding == DICTIONARY,
            took.contains(&DICTIONARY.to_string()),
            "{} took {took:?}",
            case.name
        );
    }
}

/// The annotation an annotated leaf is written under, as another engine
/// resolves it. `assert_round_trips` cannot see this: it casts each column back
/// to the type that went in before comparing, and an integer casts to its
/// temporal type losslessly, so a leaf that lost its annotation would still
/// compare equal. Reading the schema instead is what pins the annotation, since
/// an unannotated leaf resolves to the bare integer it is stored as.
fn assert_reads_back_as(values: ArrayRef, expected: &DataType) {
    let dispatch = Dispatch::spin_up(2, RING_BUFFERS, None);
    let case = Case {
        name: "annotated".to_string(),
        values,
        encoding: PLAIN,
    };

    let bytes = write_cases(&dispatch, &[case]);
    let schema = ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::from(bytes))
        .unwrap()
        .schema()
        .clone();

    assert_eq!(schema.field(0).data_type(), expected);
}

/// A timestamp column says so on the leaf, so another engine reads microseconds
/// rather than the count's bare integer.
#[test]
fn timestamp_leaf_carries_its_unit() {
    let values: ArrayRef = Arc::new(TimestampMicrosecondArray::from(
        (0..ROWS as i64)
            .map(|row| 795_225_600_000_000 + row)
            .collect::<Vec<_>>(),
    ));

    assert_reads_back_as(values, &DataType::Timestamp(TimeUnit::Microsecond, None));
}

/// A zone-carrying timestamp column stamps `isAdjustedToUTC`, so another
/// engine reads UTC instants rather than zone-less wall times.
#[test]
fn timestamp_tz_leaf_carries_the_utc_flag() {
    let values: ArrayRef = Arc::new(
        TimestampMicrosecondArray::from(
            (0..ROWS as i64)
                .map(|row| 795_225_600_000_000 + row)
                .collect::<Vec<_>>(),
        )
        .with_timezone(planner::types::UTC_TIMEZONE),
    );

    assert_reads_back_as(
        values,
        &DataType::Timestamp(
            TimeUnit::Microsecond,
            Some(planner::types::UTC_TIMEZONE.into()),
        ),
    );
}

/// The same for a date, whose day count is likewise an integer on disk.
#[test]
fn date_leaf_carries_its_annotation() {
    let values: ArrayRef = Arc::new(Date32Array::from(
        (0..ROWS as i32).map(|row| 9_204 + row).collect::<Vec<_>>(),
    ));

    assert_reads_back_as(values, &DataType::Date32);
}

/// Another engine may store a boolean column as runs rather than bits (the
/// format's RLE encoding), which pivot's writer never produces, so the file
/// comes from arrow-rs's writer told to encode the column that way.
#[test]
fn booleans_stored_as_runs_by_another_engine_read_back() {
    let dispatch = Dispatch::spin_up(2, RING_BUFFERS, None);
    let values = BooleanArray::from_iter(
        (0..ROWS).map(|row| (row % 11 != 1).then_some(row % 100 < 60 || row % 2 == 0)),
    );
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new(
            "flag",
            DataType::Boolean,
            true,
        )])),
        vec![Arc::new(values.clone()) as ArrayRef],
    )
    .unwrap();
    let properties = WriterProperties::builder()
        .set_dictionary_enabled(false)
        .set_encoding(Encoding::RLE)
        .build();
    let mut writer = ArrowWriter::try_new(Vec::new(), batch.schema(), Some(properties)).unwrap();
    writer.write(&batch).unwrap();
    let bytes = writer.into_inner().unwrap();
    assert_eq!(encodings(bytes.clone(), 0), vec!["RLE".to_string()]);

    let read = read_with_pivot(&dispatch, bytes, 1);

    assert_eq!(read.column(0).as_ref(), &values);
}

/// Every type the writer accepts, one test per type and shape. A type added to
/// the writer belongs here: this table is the only list of them, so a new line
/// adds its three tests and its columns to the mixed-file test below.
macro_rules! type_tests {
    ($($type_name:ident => $spec:expr;)*) => {
        $(
            mod $type_name {
                use super::*;

                #[test]
                fn repeated() {
                    let spec = $spec;
                    assert_round_trips(vec![spec.case(
                        stringify!($type_name),
                        Shape::Repeated,
                        "repeated",
                    )]);
                }

                #[test]
                fn distinct() {
                    let spec = $spec;
                    assert_round_trips(vec![spec.case(
                        stringify!($type_name),
                        Shape::Distinct,
                        "distinct",
                    )]);
                }

                #[test]
                fn nullable() {
                    let spec = $spec;
                    assert_round_trips(vec![spec.case(
                        stringify!($type_name),
                        Shape::Nullable,
                        "nullable",
                    )]);
                }
            }
        )*

        /// Every type's every shape as one column of one file, which is the shape
        /// a real table has: the footer's schema elements, leaf order and column
        /// chunk offsets all have to line up across columns of different physical
        /// types, and no single-column file exercises that.
        #[test]
        fn every_type_shares_one_file() {
            let mut cases = Vec::new();
            $(
                let spec = $spec;
                for (shape, shape_name) in [
                    (Shape::Repeated, "repeated"),
                    (Shape::Distinct, "distinct"),
                    (Shape::Nullable, "nullable"),
                ] {
                    cases.push(spec.case(stringify!($type_name), shape, shape_name));
                }
            )*
            assert_round_trips(cases);
        }
    };
}

type_tests! {
    boolean => boolean();
    // The narrow signed widths store their bits in an INT32, and read back as
    // themselves only if the file kept the column's width. Their deltas fit
    // that width, so distinct values pack like any other whole number.
    //
    // An `i8` column is the exception that cannot leave the dictionary: 256 is
    // every value the type has, well under the share of the rows at which the
    // encoder gives up, so all three of its shapes take the dictionary.
    int8 => primitive::<Int8Type>(DICTIONARY, |row| (row % 256) as i8, as_is);
    int16 => primitive::<Int16Type>(PACKED, |row| -20_000 + row as i16, as_is);
    int32 => primitive::<Int32Type>(PACKED, |row| row as i32 * 7_919, as_is);
    int64 => primitive::<Int64Type>(PACKED, |row| row as i64 * 7_919_483, as_is);
    // Unsigned columns store their bits in a signed physical type, so each of
    // these holds values past that type's maximum: they read back as themselves
    // only if the file kept the column's width and signedness. None has a delta
    // form (its deltas would not fit the physical type's declared width), so
    // distinct values stay plain.
    //
    // A `u8` column is the exception that cannot leave the dictionary: 256 is
    // every value the type has, well under the share of the rows at which the
    // encoder gives up, so all three of its shapes take the dictionary.
    uint8 => primitive::<UInt8Type>(DICTIONARY, |row| (row % 256) as u8, as_is);
    uint16 => primitive::<UInt16Type>(PLAIN, |row| 40_000u16.wrapping_add(row as u16), as_is);
    uint32 => primitive::<UInt32Type>(PLAIN, |row| 4_000_000_000 + row as u32, as_is);
    uint64 => primitive::<UInt64Type>(
        PLAIN,
        |row| 18_000_000_000_000_000_000 + row as u64,
        as_is
    );
    // 9204 days after the epoch is 1995-03-15.
    date32 => primitive::<Date32Type>(PACKED, |row| 9_204 + row as i32, as_is);
    // A timestamp counts microseconds, so these run from 1995-03-15 00:00:00 in
    // steps that keep a sub-second part.
    timestamp => primitive::<TimestampMicrosecondType>(
        PACKED,
        |row| 795_225_600_000_000 + row as i64 * 1_500_007,
        as_is
    );
    // A zone-carrying timestamp stores the same microsecond count; only the
    // leaf's UTC flag differs, and it must survive the round trip.
    timestamp_tz => primitive::<TimestampMicrosecondType>(
        PACKED,
        |row| 795_225_600_000_000 + row as i64 * 1_500_007,
        |array| Arc::new(array.with_timezone(planner::types::UTC_TIMEZONE)) as ArrayRef
    );
    // A float is not a whole number, so it has no delta form and no dictionary
    // cast: it stays plain whatever its values do.
    float32 => primitive::<Float32Type>(PLAIN, |row| row as f32 * 1.5, as_is);
    float64 => primitive::<Float64Type>(PLAIN, |row| row as f64 * 1.25, as_is);
    // A decimal narrow enough is stored as an int of the matching width, and one
    // too wide for that as big-endian bytes, which have no delta form. Precision
    // alone decides which, so each width appears here.
    decimal64_as_int32 => primitive::<Decimal64Type>(PACKED, |row| row as i64 * 3, |array| {
        Arc::new(array.with_precision_and_scale(9, 2).unwrap()) as ArrayRef
    });
    decimal64_as_int64 => primitive::<Decimal64Type>(PACKED, |row| row as i64 * 1_000_003, |array| {
        Arc::new(array.with_precision_and_scale(18, 4).unwrap()) as ArrayRef
    });
    decimal128_as_bytes => primitive::<Decimal128Type>(PLAIN, |row| row as i128 * 10_000_000_019, |array| {
        Arc::new(array.with_precision_and_scale(20, 2).unwrap()) as ArrayRef
    });
    decimal128_widest => primitive::<Decimal128Type>(PLAIN, |row| row as i128 * 170_141_183_460_469_231, |array| {
        Arc::new(array.with_precision_and_scale(38, 10).unwrap()) as ArrayRef
    });
    utf8 => bytes(|rows| Arc::new(StringArray::from(rows)) as ArrayRef);
    utf8_view => bytes(|rows| Arc::new(StringViewArray::from(rows)) as ArrayRef);
    binary_view => bytes(|rows| {
        let values: Vec<Option<Vec<u8>>> = rows
            .into_iter()
            // A trailing byte no UTF-8 string can hold, so this is binary and
            // not a string that happens to be readable.
            .map(|row| row.map(|row| [row.as_bytes(), &[0xff]].concat()))
            .collect();
        Arc::new(BinaryViewArray::from_iter(values)) as ArrayRef
    });
}
