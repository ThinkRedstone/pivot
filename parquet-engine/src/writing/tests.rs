//! End-to-end tests for writing JSON as a shredded variant column: drive the
//! whole write pipeline ([`encode_record_batches`]) and read the finished files
//! back with arrow-rs's strict reader — the oracle for "another engine can
//! read this".
//!
//! The two batch shapes that write variants both come through here: unshredded
//! documents ([`JsonItem`]), and batches read back out of files that each
//! shredded differently ([`ShreddedItem`], as compaction feeds the pipeline).
//! The pipeline is the same either way, which is the point of normalizing on the
//! way in.

use std::sync::Arc;

use arrow_array::{
    ArrayRef, BinaryViewArray, Date32Array, Float64Array, Int64Array, RecordBatch, StringArray,
    UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use dispatch::{BUFFER_SIZE, Dispatch, values_input};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::basic::ColumnOrder;
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet_variant_compute::{
    ShreddedSchemaBuilder, VariantArray, json_to_variant, shred_variant, unshred_variant,
};
use parquet_variant_json::VariantToJson;

use super::{
    AssembledFile, encode_compaction_batches_spec, encode_record_batches_spec, unshred_batches_spec,
};

/// A 64 MiB file-cache ring, as the other in-crate write tests use.
const RING_BUFFERS: usize = 64 * 1024 * 1024 / BUFFER_SIZE;

/// The column name every test writes its documents under.
const COLUMN: &str = "attrs";

/// A batch of input rows for the pipeline, one per test shape below.
trait IntoBatch {
    fn into_batch(self) -> Result<RecordBatch, ArrowError>;
}

/// One batch's worth of JSON documents, converted to an unshredded variant
/// column — what a producer feeding JSON into the pipeline looks like.
struct JsonItem(Vec<String>);

impl IntoBatch for JsonItem {
    fn into_batch(self) -> Result<RecordBatch, ArrowError> {
        variant_batch(&self.0, None)
    }
}

/// Documents already shredded on `paths` — what compaction reads back out of a
/// file that made its own shredding choice.
struct ShreddedItem {
    rows: Vec<String>,
    paths: Vec<&'static str>,
}

impl IntoBatch for ShreddedItem {
    fn into_batch(self) -> Result<RecordBatch, ArrowError> {
        variant_batch(&self.rows, Some(&self.paths))
    }
}

/// Documents under an integer tag column, so a test can steer rows into
/// separate partitions — and so separate files.
struct TaggedJsonItem {
    file_tag: i64,
    rows: Vec<String>,
}

impl IntoBatch for TaggedJsonItem {
    fn into_batch(self) -> Result<RecordBatch, ArrowError> {
        let variant = variant_batch(&self.rows, None)?;
        let schema = Schema::new(vec![
            Field::new("file_tag", DataType::Int64, false),
            variant.schema().field(0).clone(),
        ]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Int64Array::from(vec![self.file_tag; variant.num_rows()])),
                variant.column(0).clone(),
            ],
        )
    }
}

/// A one-column batch of `rows` as a variant, shredded on `paths` if given.
fn variant_batch(rows: &[String], paths: Option<&[&str]>) -> Result<RecordBatch, ArrowError> {
    let json: ArrayRef = Arc::new(StringArray::from(rows.to_vec()));
    let mut variants = json_to_variant(&json)?;
    if let Some(paths) = paths {
        let mut builder = ShreddedSchemaBuilder::new();
        for path in paths {
            builder = builder.with_path(*path, &arrow_schema::DataType::Int64)?;
        }
        variants = shred_variant(&variants, &builder.build())?;
    }
    let schema = Schema::new(vec![variants.field(COLUMN)]);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(variants.into_inner())])
}

/// Run the write pipeline over `items` (one unpartitioned, unsorted stream —
/// so one output file) and return the finished files' bytes. `rows_per_group`
/// caps a row group's rows, so a test can force a file of several groups.
fn write<T: IntoBatch>(items: Vec<T>, rows_per_group: usize) -> Vec<Vec<u8>> {
    write_grouped(items, &[], &[], rows_per_group)
}

/// Writes files ordered by the named columns.
fn write_sorted<T: IntoBatch>(
    items: Vec<T>,
    sort_column_names: &[&str],
    rows_per_group: usize,
) -> Vec<Vec<u8>> {
    write_grouped(items, &[], sort_column_names, rows_per_group)
}

/// Writes one ordered file for each partition represented by the input.
fn write_grouped<T: IntoBatch>(
    items: Vec<T>,
    partition_column_names: &[&str],
    sort_column_names: &[&str],
    rows_per_group: usize,
) -> Vec<Vec<u8>> {
    let dispatch = Dispatch::spin_up(2, RING_BUFFERS, None);
    let batches: Vec<RecordBatch> = items
        .into_iter()
        .map(|item| item.into_batch().unwrap())
        .collect();
    let schema = batches[0].schema();
    let spec = unshred_batches_spec(values_input(dispatch.dispatcher(), batches).record_batches());
    let partition_column_names: Arc<[String]> = partition_column_names
        .iter()
        .map(|name| name.to_string())
        .collect();
    let sort_column_names: Arc<[String]> = sort_column_names
        .iter()
        .map(|name| name.to_string())
        .collect();
    // Copy worker-owned ring memory before returning the bytes to the test
    // thread.
    let files: Vec<Vec<u8>> = encode_record_batches_spec(
        spec,
        schema,
        partition_column_names,
        sort_column_names,
        rows_per_group,
        usize::MAX,
    )
    .map_each(|file: AssembledFile| Ok(file.bytes.runs().flatten().copied().collect::<Vec<u8>>()))
    .collect()
    .unwrap();
    dispatch.exit();
    files
}

/// Read a file back through arrow-rs's strict reader, as another engine would.
fn read_back(bytes: &[u8]) -> RecordBatch {
    let reader = ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::copy_from_slice(bytes))
        .unwrap()
        .build()
        .unwrap();
    let batches: Vec<RecordBatch> = reader.map(|b| b.unwrap()).collect();
    arrow_select::concat::concat_batches(&batches[0].schema(), &batches).unwrap()
}

/// A file's first row group's statistics per leaf, keyed by dotted path.
fn leaf_stats(bytes: &[u8]) -> Vec<(String, Option<u64>, bool)> {
    let reader =
        ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::copy_from_slice(bytes)).unwrap();
    let row_group = reader.metadata().row_group(0);
    (0..row_group.num_columns())
        .map(|i| {
            let column = row_group.column(i);
            let stats = column.statistics();
            (
                column.column_path().string(),
                stats.and_then(|s| s.null_count_opt()),
                stats.is_some_and(|s| s.min_bytes_opt().is_some()),
            )
        })
        .collect()
}

/// A batch of plain columns, for the tests about which encoding a column's
/// values take.
struct ColumnsItem(Vec<(&'static str, ArrayRef)>);

impl IntoBatch for ColumnsItem {
    fn into_batch(self) -> Result<RecordBatch, ArrowError> {
        let fields: Vec<_> = self
            .0
            .iter()
            .map(|(name, array)| {
                arrow_schema::Field::new(*name, array.data_type().clone(), array.null_count() != 0)
            })
            .collect();
        let arrays = self.0.into_iter().map(|(_, array)| array).collect();
        RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays)
    }
}

/// The encodings a file's footer reports per leaf, keyed by column name.
fn leaf_encodings(bytes: &[u8]) -> Vec<(String, Vec<String>)> {
    let reader =
        ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::copy_from_slice(bytes)).unwrap();
    let row_group = reader.metadata().row_group(0);
    (0..row_group.num_columns())
        .map(|i| {
            let column = row_group.column(i);
            (
                column.column_path().string(),
                column
                    .encodings()
                    .map(|e| e.to_string())
                    .collect::<Vec<String>>(),
            )
        })
        .collect()
}

/// A file's leaf column paths, dotted — the shape the writer actually stamped
/// into the footer.
fn leaf_paths(bytes: &[u8]) -> Vec<String> {
    let reader =
        ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::copy_from_slice(bytes)).unwrap();
    reader
        .metadata()
        .file_metadata()
        .schema_descr()
        .columns()
        .iter()
        .map(|c| c.path().string())
        .collect()
}

/// Render the variant column back to one JSON string per row, exactly as a query
/// does: fold the typed leaves back into the binary value, then render each row.
/// The documents must survive shredding whole.
fn documents(batch: &RecordBatch) -> Vec<String> {
    let variants = VariantArray::try_new(batch.column(0).as_ref()).unwrap();
    let variants = unshred_variant(&variants).unwrap();
    (0..variants.len())
        .map(|row| {
            let mut text = Vec::new();
            variants.value(row).to_json(&mut text).unwrap();
            String::from_utf8(text).unwrap()
        })
        .collect()
}

fn rows(docs: &[&str]) -> Vec<String> {
    docs.iter().map(|d| d.to_string()).collect()
}

/// JSON fed in as a variant column comes out as a real shredded VARIANT: the
/// footer carries a typed leaf per agreed path beside the binary fallback, and a
/// strict reader accepts the file.
#[test]
fn json_writes_as_a_shredded_variant_column() {
    let files = write(
        vec![JsonItem(rows(&[
            r#"{"id": 1, "name": "a"}"#,
            r#"{"id": 2, "name": "b"}"#,
        ]))],
        128 * 1024,
    );

    assert_eq!(files.len(), 1);
    assert_eq!(
        leaf_paths(&files[0]),
        vec![
            "attrs.metadata",
            "attrs.value",
            "attrs.typed_value.id.value",
            "attrs.typed_value.id.typed_value",
            "attrs.typed_value.name.value",
            "attrs.typed_value.name.typed_value",
        ]
    );
}

/// Shredding never loses a document: the rows read back are the rows written,
/// both the shredded fields and the ones left in the binary fallback.
#[test]
fn a_shredded_document_reads_back_whole() {
    let files = write(
        vec![JsonItem(rows(&[
            r#"{"id": 1, "extra": true}"#,
            r#"{"id": 2, "extra": false}"#,
        ]))],
        128 * 1024,
    );

    let batch = read_back(&files[0]);

    // `id` shredded into a typed leaf, `extra` (a boolean, which has no leaf
    // type) stayed in `value` — and both come back.
    assert_eq!(
        documents(&batch),
        vec![r#"{"extra":true,"id":1}"#, r#"{"extra":false,"id":2}"#]
    );
}

/// A row whose value at a shredded path is of another type keeps that value in
/// the binary fallback rather than losing it — which is why one file can shred a
/// path its rows only mostly agree on.
#[test]
fn a_row_that_disagrees_on_a_shredded_types_falls_back() {
    let files = write(
        vec![JsonItem(rows(&[
            r#"{"v": 1}"#,
            r#"{"v": 2}"#,
            r#"{"v": "not a number"}"#,
        ]))],
        128 * 1024,
    );

    let batch = read_back(&files[0]);

    assert!(leaf_paths(&files[0]).contains(&"attrs.typed_value.v.typed_value".to_string()));
    assert_eq!(
        documents(&batch),
        vec![r#"{"v":1}"#, r#"{"v":2}"#, r#"{"v":"not a number"}"#]
    );
}

/// Each file shreds to its own rows: two partitions' files whose documents
/// disagree get different layouts, which the read path resolves per file.
#[test]
fn each_file_shreds_to_its_own_rows() {
    // One file per partition, so each tag's documents shred on their own.
    let files = write_grouped(
        vec![
            TaggedJsonItem {
                file_tag: 1,
                rows: rows(&[r#"{"id": 1}"#, r#"{"id": 2}"#]),
            },
            TaggedJsonItem {
                file_tag: 2,
                rows: rows(&[r#"{"host": "a"}"#, r#"{"host": "b"}"#]),
            },
        ],
        &["file_tag"],
        &[],
        128 * 1024,
    );

    assert_eq!(files.len(), 2);
    let layouts: Vec<Vec<String>> = files.iter().map(|f| leaf_paths(f)).collect();
    assert!(
        layouts
            .iter()
            .any(|l| l.contains(&"attrs.typed_value.id.typed_value".to_string()))
    );
    assert!(
        layouts
            .iter()
            .any(|l| l.contains(&"attrs.typed_value.host.typed_value".to_string()))
    );
    assert_ne!(layouts[0], layouts[1]);
}

/// Documents with no path worth shredding stay as the plain `{metadata, value}`
/// pair — a legal variant column, and still readable.
#[test]
fn a_column_with_nothing_worth_shredding_stays_unshredded() {
    let files = write(
        vec![JsonItem(rows(&[r#"{"ok": true}"#, r#"{"ok": false}"#]))],
        128 * 1024,
    );

    assert_eq!(leaf_paths(&files[0]), vec!["attrs.metadata", "attrs.value"]);
    assert_eq!(
        documents(&read_back(&files[0])),
        vec![r#"{"ok":true}"#, r#"{"ok":false}"#]
    );
}

/// The re-shredding compaction depends on: batches that arrive already shredded
/// — each on a different path, as separate files would be — merge into one file
/// that shreds to what its own rows favour, not to what any input file chose.
#[test]
fn already_shredded_input_re_shreds_to_the_merged_rows() {
    let files = write(
        vec![
            ShreddedItem {
                rows: rows(&[r#"{"id": 1, "n": 10}"#, r#"{"id": 2, "n": 20}"#]),
                paths: vec!["id"],
            },
            ShreddedItem {
                rows: rows(&[r#"{"id": 3, "n": 30}"#, r#"{"id": 4, "n": 40}"#]),
                paths: vec!["n"],
            },
        ],
        128 * 1024,
    );

    // One file, and it shreds *both* paths — neither input's layout survived as
    // such; the output decided afresh from all four rows.
    assert_eq!(files.len(), 1);
    assert_eq!(
        leaf_paths(&files[0]),
        vec![
            "attrs.metadata",
            "attrs.value",
            "attrs.typed_value.id.value",
            "attrs.typed_value.id.typed_value",
            "attrs.typed_value.n.value",
            "attrs.typed_value.n.typed_value",
        ]
    );

    let mut got = documents(&read_back(&files[0]));
    got.sort();
    assert_eq!(
        got,
        vec![
            r#"{"id":1,"n":10}"#,
            r#"{"id":2,"n":20}"#,
            r#"{"id":3,"n":30}"#,
            r#"{"id":4,"n":40}"#,
        ]
    );
}

/// Every leaf carries footer statistics, shredded or not, so a reader can prune
/// row groups by any of them. Typed leaves get a range; binary variant leaves
/// always get a null count and get a range only for canonical JSON nulls.
#[test]
fn every_leaf_carries_footer_statistics() {
    let files = write(
        vec![JsonItem(rows(&[r#"{"id": 1}"#, r#"{"id": 5}"#]))],
        128 * 1024,
    );

    let stats = leaf_stats(&files[0]);

    assert!(
        stats.iter().all(|(_, null_count, _)| null_count.is_some()),
        "every leaf needs a null count: {stats:?}"
    );
    let typed = stats
        .iter()
        .find(|(path, ..)| path == "attrs.typed_value.id.typed_value")
        .unwrap();
    assert_eq!(typed.1, Some(0), "every row shredded into the typed leaf");
    assert!(typed.2, "a typed leaf carries a min/max to prune on");
}

/// A reader following the spec ignores every chunk's `min_value`/`max_value`
/// unless the footer names an order for each leaf column, so the writer has to
/// stamp one per leaf.
#[test]
fn the_footer_names_a_type_defined_order_for_every_leaf() {
    let files = write(
        vec![JsonItem(rows(&[r#"{"id": 1, "name": "a"}"#]))],
        128 * 1024,
    );

    let reader = SerializedFileReader::new(bytes::Bytes::copy_from_slice(&files[0])).unwrap();
    let file_metadata = reader.metadata().file_metadata();
    let column_orders = file_metadata.column_orders().unwrap();

    assert_eq!(
        column_orders.len(),
        file_metadata.schema_descr().num_columns()
    );
    assert!(
        column_orders
            .iter()
            .all(|order| matches!(order, ColumnOrder::TYPE_DEFINED_ORDER(_))),
        "{column_orders:?}"
    );
}

/// A shredded path may only be pruned by its typed leaf when the `value`
/// fallback beside it holds nothing, and a reader establishes that from the
/// fallback's null count against the row count. So an unused fallback has to
/// record one, even though it is all-null and has no range to report.
#[test]
fn an_unused_fallback_leaf_records_a_full_null_count() {
    let documents = rows(&[r#"{"id": 1}"#, r#"{"id": 5}"#, r#"{"id": 9}"#]);
    let files = write(vec![JsonItem(documents.clone())], 128 * 1024);

    let stats = leaf_stats(&files[0]);

    let fallback = stats
        .iter()
        .find(|(path, ..)| path == "attrs.typed_value.id.value")
        .unwrap();
    assert_eq!(
        fallback.1,
        Some(documents.len() as u64),
        "every row shredded, so the fallback is null throughout: {stats:?}"
    );
    assert!(!fallback.2, "an all-null leaf has no min/max to report");
}

#[test]
fn a_json_null_only_binary_leaf_records_its_canonical_bounds() {
    // Setup
    let nulls: ArrayRef = Arc::new(BinaryViewArray::from(vec![
        Some(&[0_u8][..]),
        None,
        Some(&[0_u8][..]),
    ]));
    let mixed: ArrayRef = Arc::new(BinaryViewArray::from(vec![
        &[0_u8][..],
        &[1_u8][..],
        &[0_u8][..],
    ]));

    // Execute
    let files = write(
        vec![ColumnsItem(vec![("nulls", nulls), ("mixed", mixed)])],
        100_000,
    );
    let reader =
        ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::copy_from_slice(&files[0])).unwrap();
    let row_group = reader.metadata().row_group(0);
    let null_stats = row_group.column(0).statistics();
    let mixed_stats = row_group.column(1).statistics();

    // Assert
    assert_eq!(
        null_stats.and_then(|stats| stats.min_bytes_opt()),
        Some(&[0_u8][..])
    );
    assert_eq!(
        null_stats.and_then(|stats| stats.max_bytes_opt()),
        Some(&[0_u8][..])
    );
    assert_eq!(mixed_stats.and_then(|stats| stats.min_bytes_opt()), None);
    assert_eq!(mixed_stats.and_then(|stats| stats.max_bytes_opt()), None);
}

/// A nested path shreds into nested groups, and the document still reads back
/// whole through them.
#[test]
fn a_nested_path_shreds_and_reads_back() {
    let files = write(
        vec![JsonItem(rows(&[
            r#"{"user": {"id": 1}}"#,
            r#"{"user": {"id": 2}}"#,
        ]))],
        128 * 1024,
    );

    assert!(
        leaf_paths(&files[0])
            .contains(&"attrs.typed_value.user.typed_value.id.typed_value".to_string())
    );
    assert_eq!(
        documents(&read_back(&files[0])),
        vec![r#"{"user":{"id":1}}"#, r#"{"user":{"id":2}}"#]
    );
}

/// A column with too many distinct values to dictionary-encode used to fall to
/// PLAIN and pay the full width per value. It now packs its differences, and
/// another implementation reads the values back unchanged.
#[test]
fn a_high_cardinality_integer_column_packs_its_differences() {
    let values: Vec<i64> = (0..50_000).map(|i| 1_000_000 + i * 7919).collect();
    let column: ArrayRef = Arc::new(Int64Array::from(values.clone()));

    let files = write(vec![ColumnsItem(vec![("key", column)])], 128 * 1024);

    assert_eq!(
        leaf_encodings(&files[0]),
        vec![("key".to_string(), vec!["DELTA_BINARY_PACKED".to_string()])]
    );
    let batch = read_back(&files[0]);
    let read: Vec<i64> = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .values()
        .to_vec();
    assert_eq!(read, values);
}

/// Values that fall as well as rise, and that span far enough for a difference
/// to need every bit of its width.
#[test]
fn packed_differences_survive_falling_and_wide_values() {
    let values: Vec<i64> = (0..200_000)
        .map(|i: i64| match i % 3 {
            0 => -i * 1_000_003,
            1 => i64::MAX / 2 - i,
            _ => i64::MIN / 2 + i,
        })
        .collect();
    let column: ArrayRef = Arc::new(Int64Array::from(values.clone()));

    let files = write(vec![ColumnsItem(vec![("key", column)])], 128 * 1024);

    let batch = read_back(&files[0]);
    let read: Vec<i64> = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .values()
        .to_vec();
    assert_eq!(read, values);
}

/// A string column past the dictionary drops the four-byte length it used to
/// spend per value, packing the lengths instead and laying the bytes end to end.
#[test]
fn a_high_cardinality_string_column_packs_its_lengths() {
    let values: Vec<String> = (0..40_000)
        .map(|i: usize| {
            if i.is_multiple_of(7) {
                String::new()
            } else {
                format!("value {i} with enough tail to not inline")
            }
        })
        .collect();
    let column: ArrayRef = Arc::new(StringArray::from(values.clone()));

    let files = write(vec![ColumnsItem(vec![("name", column)])], 100_000);

    assert_eq!(
        leaf_encodings(&files[0]),
        vec![(
            "name".to_string(),
            vec!["DELTA_LENGTH_BYTE_ARRAY".to_string()]
        )]
    );
    let batch = read_back(&files[0]);
    let read: Vec<String> = batch
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .map(|v| v.unwrap().to_string())
        .collect();
    assert_eq!(read, values);
}

/// Few enough distinct values and the dictionary still wins, which is what the
/// delta forms are a fallback from, not a replacement for.
#[test]
fn a_repeating_column_still_dictionary_encodes() {
    let values: Vec<String> = (0..40_000).map(|i| format!("colour {}", i % 8)).collect();
    let column: ArrayRef = Arc::new(StringArray::from(values));

    let files = write(vec![ColumnsItem(vec![("colour", column)])], 100_000);

    let (_, encodings) = leaf_encodings(&files[0]).pop().unwrap();
    assert!(
        encodings.contains(&"RLE_DICTIONARY".to_string()),
        "expected a dictionary, got {encodings:?}"
    );
}

/// Floats are not whole numbers and have no delta form, so they keep taking
/// PLAIN rather than being packed as something they are not.
#[test]
fn a_float_column_stays_plain() {
    let values: Vec<f64> = (0..200_000).map(|i| i as f64 * 1.5).collect();
    let column: ArrayRef = Arc::new(Float64Array::from(values.clone()));

    let files = write(vec![ColumnsItem(vec![("measure", column)])], 128 * 1024);

    assert_eq!(
        leaf_encodings(&files[0]),
        vec![("measure".to_string(), vec!["PLAIN".to_string()])]
    );
    let batch = read_back(&files[0]);
    let read: Vec<f64> = batch
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .values()
        .to_vec();
    assert_eq!(read, values);
}

/// What the fallback buys, measured rather than assumed. A key scattered over
/// twenty million values has no run to exploit, so every difference still needs
/// most of its width; the packing is what takes the column from the eight bytes
/// PLAIN spends per value to a little over three.
#[test]
fn packing_a_scattered_key_costs_a_third_of_writing_it_whole() {
    let mut seed = 12_345u64;
    let mut next = || {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (seed >> 33) as i64
    };
    let values: Vec<i64> = (0..200_000).map(|_| 1 + next() % 20_000_000).collect();
    let column: ArrayRef = Arc::new(Int64Array::from(values.clone()));

    let bytes = write(vec![ColumnsItem(vec![("key", column)])], 128 * 1024)[0].len();

    let per_value = bytes as f64 / values.len() as f64;
    assert!(per_value < 4.0, "{per_value} bytes a value");
}

/// Rows written under a sort key read back in key order however they arrived,
/// each row's other columns still on it — across row groups, since the file is
/// forced to hold several.
#[test]
fn a_sort_key_orders_a_files_rows() {
    let keys: Vec<i64> = (0..10_000).map(|i| (i * 7919) % 10_000).collect();
    let names: Vec<String> = keys.iter().map(|key| format!("row {key}")).collect();
    let item = |range: std::ops::Range<usize>| {
        ColumnsItem(vec![
            (
                "key",
                Arc::new(Int64Array::from(keys[range.clone()].to_vec())) as ArrayRef,
            ),
            ("name", Arc::new(StringArray::from(names[range].to_vec()))),
        ])
    };

    let files = write_sorted(vec![item(0..5_000), item(5_000..10_000)], &["key"], 1_000);

    assert_eq!(files.len(), 1);
    let batch = read_back(&files[0]);
    let read_keys: Vec<i64> = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .values()
        .to_vec();
    assert_eq!(read_keys, (0..10_000).collect::<Vec<i64>>());
    let read_names = batch
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    for (row, key) in read_keys.iter().enumerate() {
        assert_eq!(read_names.value(row), format!("row {key}"));
    }
}

/// Compaction receives runs that are already sorted within their source files.
/// It must merge overlapping runs globally, but must not need the INSERT stage
/// that sorts each incoming batch first.
#[test]
fn compaction_merges_presorted_source_runs_into_one_sorted_file() {
    let dispatch = Dispatch::spin_up(2, RING_BUFFERS, None);
    let batches = [vec![1, 3, 5, 7], vec![2, 4, 6, 8]]
        .into_iter()
        .map(|keys| {
            ColumnsItem(vec![("key", Arc::new(Int64Array::from(keys)) as ArrayRef)])
                .into_batch()
                .unwrap()
        })
        .collect::<Vec<_>>();
    let schema = batches[0].schema();
    let input = values_input(dispatch.dispatcher(), batches).record_batches();

    let files: Vec<Vec<u8>> = encode_compaction_batches_spec(
        input,
        schema,
        None,
        Arc::from([]),
        Arc::from(["key".to_string()]),
        3,
        usize::MAX,
    )
    .map_each(|file: AssembledFile| Ok(file.bytes.runs().flatten().copied().collect::<Vec<u8>>()))
    .collect()
    .unwrap();
    dispatch.exit();

    assert_eq!(files.len(), 1);
    let batch = read_back(&files[0]);
    let keys = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .values();
    assert_eq!(keys, &[1, 2, 3, 4, 5, 6, 7, 8]);
}

/// A small in-memory target produces several internally ordered files without
/// dropping or duplicating rows. Parallel batch arrival makes the exact file
/// count nondeterministic.
#[test]
fn a_stream_past_the_file_target_cuts_several_sorted_files() {
    let keys: Vec<i64> = (0..3_000).map(|i| (i * 7_919) % 3_000).collect();
    let items: Vec<ColumnsItem> = keys
        .chunks(500)
        .map(|chunk| {
            ColumnsItem(vec![(
                "key",
                Arc::new(Int64Array::from(chunk.to_vec())) as ArrayRef,
            )])
        })
        .collect();

    let dispatch = Dispatch::spin_up(2, RING_BUFFERS, None);
    let batches: Vec<RecordBatch> = items
        .into_iter()
        .map(|item| item.into_batch().unwrap())
        .collect();
    let schema = batches[0].schema();
    let spec = values_input(dispatch.dispatcher(), batches).record_batches();
    let files: Vec<Vec<u8>> = encode_record_batches_spec(
        spec,
        schema,
        Arc::from([]),
        Arc::from(["key".to_string()]),
        500,
        8 * 1024,
    )
    .map_each(|file: AssembledFile| Ok(file.bytes.runs().flatten().copied().collect::<Vec<u8>>()))
    .collect()
    .unwrap();
    dispatch.exit();

    assert!(files.len() >= 2, "24KB of rows against an 8KB file target");
    let mut all_keys = Vec::new();
    for file in &files {
        let batch = read_back(file);
        let file_keys: Vec<i64> = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values()
            .to_vec();
        assert!(file_keys.windows(2).all(|pair| pair[0] <= pair[1]));
        all_keys.extend(file_keys);
    }
    all_keys.sort();
    assert_eq!(all_keys, (0..3_000).collect::<Vec<i64>>());
}

/// A partitioned, sorted write cuts one file per partition tuple, each file
/// single-partition with its rows in key order.
#[test]
fn a_partitioned_sorted_write_cuts_one_sorted_file_per_partition() {
    let parts: Vec<i64> = (0..2_000).map(|i| i % 2).collect();
    let keys: Vec<i64> = (0..2_000).map(|i| (i * 7919) % 2_000).collect();
    let files = write_grouped(
        vec![ColumnsItem(vec![
            ("part", Arc::new(Int64Array::from(parts)) as ArrayRef),
            ("key", Arc::new(Int64Array::from(keys))),
        ])],
        &["part"],
        &["key"],
        500,
    );

    assert_eq!(files.len(), 2);
    for file in &files {
        let batch = read_back(file);
        let parts = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert!(parts.values().iter().all(|&part| part == parts.value(0)));
        let keys = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values();
        assert!(keys.windows(2).all(|pair| pair[0] <= pair[1]));
        assert_eq!(keys.len(), 1_000);
    }
}

/// A string sort key orders a file's rows too, through the general sort path
/// rather than the fixed-width one.
#[test]
fn a_string_sort_key_orders_a_files_rows() {
    let names: Vec<String> = (0..2_000)
        .map(|i| format!("name {:04}", (i * 7919) % 2_000))
        .collect();
    let keys: Vec<i64> = (0..2_000).collect();
    let files = write_sorted(
        vec![ColumnsItem(vec![
            (
                "name",
                Arc::new(StringArray::from(names.clone())) as ArrayRef,
            ),
            ("key", Arc::new(Int64Array::from(keys))),
        ])],
        &["name"],
        1_000,
    );

    assert_eq!(files.len(), 1);
    let batch = read_back(&files[0]);
    let read_names: Vec<String> = batch
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .map(|value| value.unwrap().to_string())
        .collect();
    let mut expected = names;
    expected.sort();
    assert_eq!(read_names, expected);
}

/// A string-and-integer key sorts lexicographically: rows tied on the string
/// order by the integer.
#[test]
fn a_string_and_int_key_sort_ties_by_the_int() {
    let names: Vec<String> = (0..2_000).map(|i| format!("name {}", i % 4)).collect();
    let keys: Vec<i64> = (0..2_000).map(|i| (i * 7919) % 2_000).collect();
    let files = write_sorted(
        vec![ColumnsItem(vec![
            (
                "name",
                Arc::new(StringArray::from(names.clone())) as ArrayRef,
            ),
            ("key", Arc::new(Int64Array::from(keys.clone()))),
        ])],
        &["name", "key"],
        1_000,
    );

    let batch = read_back(&files[0]);
    let read: Vec<(String, i64)> = batch
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .zip(
            batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values(),
        )
        .map(|(name, &key)| (name.unwrap().to_string(), key))
        .collect();
    let mut expected: Vec<(String, i64)> = names.into_iter().zip(keys).collect();
    expected.sort();
    assert_eq!(read, expected);
}

/// A date column writes as the day count its INT32 storage holds, annotated so
/// a reader knows it is a date and not a plain integer. Arrow's reader reads it
/// back as a date, which is what a table with a date column needs to round trip
/// at all.
#[test]
fn a_date_column_reads_back_as_a_date() {
    // 9204 days after the epoch is 1995-03-15.
    let days: Vec<i32> = (0..50_000).map(|i| 9_204 + i % 2_557).collect();
    let column: ArrayRef = Arc::new(Date32Array::from(days.clone()));

    let files = write(vec![ColumnsItem(vec![("shipdate", column)])], 100_000);

    let batch = read_back(&files[0]);
    assert_eq!(batch.schema().field(0).data_type(), &DataType::Date32);
    let read: Vec<i32> = batch
        .column(0)
        .as_any()
        .downcast_ref::<Date32Array>()
        .unwrap()
        .values()
        .to_vec();
    assert_eq!(read, days);
}

/// The four unsigned columns, each holding values past the signed maximum of
/// the physical type it is stored in — the values a file that lost the column's
/// width or signedness reads back wrong.
fn unsigned_columns() -> Vec<(&'static str, ArrayRef)> {
    let bytes: Vec<u8> = (0..2_000).map(|i| (128 + i % 128) as u8).collect();
    let shorts: Vec<u16> = (0..2_000).map(|i| 40_000 + i as u16).collect();
    let ints: Vec<u32> = (0..2_000).map(|i| 4_000_000_000 + i as u32).collect();
    let longs: Vec<u64> = (0..2_000)
        .map(|i| 18_000_000_000_000_000_000 + i as u64)
        .collect();
    vec![
        ("u8", Arc::new(UInt8Array::from(bytes)) as ArrayRef),
        ("u16", Arc::new(UInt16Array::from(shorts))),
        ("u32", Arc::new(UInt32Array::from(ints))),
        ("u64", Arc::new(UInt64Array::from(longs))),
    ]
}

/// An unsigned column stores its bits in a signed physical type, so a value past
/// that type's maximum comes back as a negative number unless the file says how
/// wide the column really is and that it is unsigned. Arrow-rs reading these
/// back as their own unsigned types is what says the annotation landed.
#[test]
fn unsigned_columns_read_back_unsigned() {
    let columns = unsigned_columns();

    let files = write(vec![ColumnsItem(columns.clone())], 100_000);

    let batch = read_back(&files[0]);
    for (i, (name, written)) in columns.iter().enumerate() {
        assert_eq!(batch.schema().field(i).name(), name);
        assert_eq!(batch.column(i), written, "column {name} round trips");
    }
}

/// An unsigned column that repeats itself takes the dictionary, whose page
/// stores the distinct values through the same encoder a PLAIN page uses. Both
/// paths write the values, so both are covered: this one and the all-distinct
/// column below.
#[test]
fn a_repeating_unsigned_column_round_trips_through_the_dictionary() {
    let values: Vec<u8> = (0..2_000).map(|i| (128 + i % 64) as u8).collect();
    let column: ArrayRef = Arc::new(UInt8Array::from(values.clone()));

    let files = write(vec![ColumnsItem(vec![("u8", column)])], 128 * 1024);

    let (_, encodings) = leaf_encodings(&files[0]).pop().unwrap();
    assert!(
        encodings.contains(&"RLE_DICTIONARY".to_string()),
        "expected a dictionary, got {encodings:?}"
    );
    let batch = read_back(&files[0]);
    assert_eq!(batch.column(0).as_ref(), &UInt8Array::from(values));
}

/// An all-distinct unsigned column falls out of the dictionary and takes PLAIN,
/// since the delta form the signed integers take is not open to it (its values
/// would not fit the width the physical type declares).
#[test]
fn an_all_distinct_unsigned_column_round_trips_through_plain() {
    let values: Vec<u64> = (0..2_000)
        .map(|i| 18_000_000_000_000_000_000 + i as u64)
        .collect();
    let column: ArrayRef = Arc::new(UInt64Array::from(values.clone()));

    let files = write(vec![ColumnsItem(vec![("u64", column)])], 128 * 1024);

    let (_, encodings) = leaf_encodings(&files[0]).pop().unwrap();
    assert_eq!(encodings, vec!["PLAIN".to_string()]);
    let batch = read_back(&files[0]);
    assert_eq!(batch.column(0).as_ref(), &UInt64Array::from(values));
}

/// An unsigned column's footer bounds have to be ordered as unsigned, or a
/// reader pruning row groups by them drops live rows: every value here is past
/// the signed maximum, so signed bounds would order the whole column below zero.
#[test]
fn unsigned_column_bounds_are_ordered_unsigned() {
    let values: Vec<u32> = vec![4_000_000_000, 7, 2_147_483_648, 4_294_967_295];
    let column: ArrayRef = Arc::new(UInt32Array::from(values));

    let files = write(vec![ColumnsItem(vec![("u32", column)])], 100_000);

    let reader =
        ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::copy_from_slice(&files[0])).unwrap();
    let stats = reader.metadata().row_group(0).column(0).statistics();
    let bounds = stats.map(|s| {
        (
            s.min_bytes_opt().map(<[u8]>::to_vec),
            s.max_bytes_opt().map(<[u8]>::to_vec),
        )
    });
    assert_eq!(
        bounds,
        Some((
            Some(7u32.to_le_bytes().to_vec()),
            Some(4_294_967_295u32.to_le_bytes().to_vec())
        ))
    );
}

/// The dictionary is left behind on how many distinct values a column has, not
/// on how large they are. This column's dictionary is only 400 KB, well inside
/// the size guard, but every value is distinct, so the dictionary would store
/// each one once and then spend an index per row on top. It packs instead.
#[test]
fn a_column_of_distinct_values_packs_rather_than_dictionary_encodes() {
    let values: Vec<i64> = (0..50_000).map(|i| 7_000_000 + i * 13).collect();
    let column: ArrayRef = Arc::new(Int64Array::from(values.clone()));

    let files = write(vec![ColumnsItem(vec![("key", column)])], 128 * 1024);

    assert_eq!(
        leaf_encodings(&files[0]),
        vec![("key".to_string(), vec!["DELTA_BINARY_PACKED".to_string()])]
    );
}

/// A column that repeats itself often enough still takes the dictionary, even
/// where the values themselves are long: it is the repetition that pays, and
/// this is what keeps a low-cardinality string column dictionary-encoded, where
/// a reader can prune a row group by the dictionary alone.
#[test]
fn a_column_that_repeats_takes_the_dictionary_however_long_its_values() {
    let values: Vec<String> = (0..50_000)
        .map(|i: usize| format!("a fairly long repeated value number {}", i % 100))
        .collect();
    let column: ArrayRef = Arc::new(StringArray::from(values));

    let files = write(vec![ColumnsItem(vec![("label", column)])], 128 * 1024);

    let (_, encodings) = leaf_encodings(&files[0]).pop().unwrap();
    assert!(
        encodings.contains(&"RLE_DICTIONARY".to_string()),
        "expected a dictionary, got {encodings:?}"
    );
}

/// A date column that falls out of the dictionary still writes: every step of
/// the encode path has to know the type, not just the value encoders, and a
/// small row group is enough to tip a column of few distinct dates out.
#[test]
fn a_date_column_writes_whichever_encoding_it_takes() {
    let days: Vec<i32> = (0..2_000).map(|i| 9_204 + i).collect();
    let column: ArrayRef = Arc::new(Date32Array::from(days.clone()));

    let files = write(vec![ColumnsItem(vec![("shipdate", column)])], 128 * 1024);

    let (_, encodings) = leaf_encodings(&files[0]).pop().unwrap();
    assert_eq!(encodings, vec!["DELTA_BINARY_PACKED".to_string()]);
    let batch = read_back(&files[0]);
    let read: Vec<i32> = batch
        .column(0)
        .as_any()
        .downcast_ref::<Date32Array>()
        .unwrap()
        .values()
        .to_vec();
    assert_eq!(read, days);
}
