use std::collections::HashMap;
use std::sync::Arc;

use super::*;
use arrow_array::{
    ArrayRef, BooleanArray, Date32Array, Decimal128Array, Float32Array, Float64Array, Int32Array,
    Int64Array, Scalar, StringViewArray, TimestampMicrosecondArray,
};
use iceberg::spec::{
    DataContentType, DataFileBuilder, DataFileFormat, FieldSummary, FormatVersion, Literal,
    ManifestContentType, ManifestEntry, ManifestMetadata, ManifestStatus, NestedField,
    PartitionSpec, PrimitiveType, Schema, Struct, Transform as IcebergTransform, Type,
};
use planner::expression::{
    Between, Compare, CompareType, Conjunction, ConjunctionOp, InList, IsNull, Ref,
};
use planner::types::{Type as PivotType, type_from_physical};

fn schema(id: i32, name: &str, primitive: PrimitiveType) -> SchemaRef {
    Arc::new(
        Schema::builder()
            .with_fields(vec![Arc::new(NestedField::optional(
                id,
                name,
                Type::Primitive(primitive),
            ))])
            .build()
            .unwrap(),
    )
}

fn predicate(compare_type: CompareType, value: ArrayRef) -> Vec<Expression> {
    let constant_type = type_from_physical(value.data_type()).unwrap();
    vec![Expression::Compare(Compare {
        left: Box::new(Expression::Ref(Ref {
            column_idx: 0,
            return_type: constant_type,
            name: None,
        })),
        right: Box::new(Expression::Constant(Scalar::new(value))),
        compare_type,
        return_type: PivotType::Boolean,
    })]
}

fn long_predicate(compare_type: CompareType, value: i64) -> Vec<Expression> {
    predicate(compare_type, Arc::new(Int64Array::from(vec![value])))
}

fn double_predicate(compare_type: CompareType, value: f64) -> Vec<Expression> {
    predicate(compare_type, Arc::new(Float64Array::from(vec![value])))
}

/// A one-row data file named `path` whose metrics describe field 1.
fn file(path: &str) -> DataFileBuilder {
    let mut builder = DataFileBuilder::default();
    builder
        .content(DataContentType::Data)
        .file_path(path.into())
        .file_format(DataFileFormat::Parquet)
        .file_size_in_bytes(1)
        .record_count(1);
    builder
}

/// A file whose field 1 holds exactly `value`, without NULLs.
fn file_holding(path: &str, value: Datum) -> DataFile {
    file(path)
        .null_value_counts(HashMap::from([(1, 0)]))
        .lower_bounds(HashMap::from([(1, value.clone())]))
        .upper_bounds(HashMap::from([(1, value)]))
        .build()
        .unwrap()
}

/// A manifest written under `stored` and `spec`, listing `files` as added.
fn manifest(stored: &SchemaRef, spec: PartitionSpec, files: Vec<DataFile>) -> Manifest {
    let entries = files.into_iter().map(|file| (ManifestStatus::Added, file));
    manifest_of_entries(stored, spec, entries.collect())
}

fn manifest_of_entries(
    stored: &SchemaRef,
    spec: PartitionSpec,
    entries: Vec<(ManifestStatus, DataFile)>,
) -> Manifest {
    Manifest::new(
        ManifestMetadata {
            schema: stored.clone(),
            schema_id: stored.schema_id(),
            partition_spec: spec,
            format_version: FormatVersion::V2,
            content: ManifestContentType::Data,
        },
        entries
            .into_iter()
            .map(|(status, file)| {
                ManifestEntry::builder()
                    .status(status)
                    .data_file(file)
                    .build()
            })
            .collect(),
    )
}

fn unpartitioned(schema: &SchemaRef) -> PartitionSpec {
    PartitionSpec::builder(schema.clone()).build().unwrap()
}

fn partitioned_by(schema: &SchemaRef, spec_id: i32, transform: IcebergTransform) -> PartitionSpec {
    let column = schema.as_struct().fields()[0].name.clone();
    PartitionSpec::builder(schema.clone())
        .with_spec_id(spec_id)
        .add_partition_field(column, "partition", transform)
        .unwrap()
        .build()
        .unwrap()
}

/// The data files of `manifests`, in a table of `current` whose partition
/// specs are the ones the manifests were written under.
fn data_files(current: &SchemaRef, manifests: &[Manifest]) -> DataFiles {
    DataFiles {
        manifests: manifests.to_vec(),
        schema: current.clone(),
        partition_specs: manifests
            .iter()
            .map(|manifest| Arc::new(manifest.metadata().partition_spec.clone()))
            .collect(),
    }
}

/// The paths of the files of `manifests` that `filters` keep.
fn selected(current: &SchemaRef, manifests: &[Manifest], filters: &[Expression]) -> Vec<String> {
    data_files(current, manifests)
        .select(filters)
        .unwrap()
        .into_iter()
        .map(|file| file.path)
        .collect()
}

#[test]
fn renamed_and_promoted_fields_compare_old_metrics_in_the_current_type() {
    let current = schema(1, "renamed", PrimitiveType::Long);
    let stored = schema(1, "original", PrimitiveType::Int);
    let files = vec![
        file_holding("three", Datum::int(3)),
        file_holding("five", Datum::int(5)),
    ];
    let manifests = [manifest(&stored, unpartitioned(&stored), files)];

    let above_four = selected(
        &current,
        &manifests,
        &long_predicate(CompareType::Greater, 4),
    );
    let beyond_int = selected(
        &current,
        &manifests,
        &long_predicate(CompareType::Greater, i64::from(i32::MAX) + 1),
    );

    assert_eq!(above_four, ["five"]);
    assert!(beyond_int.is_empty());
}

#[test]
fn missing_metrics_are_unknown_and_an_all_null_column_matches_nothing() {
    let schema = schema(1, "id", PrimitiveType::Long);
    let unknown = file("unknown").build().unwrap();
    let all_null = file("all null")
        .null_value_counts(HashMap::from([(1, 1)]))
        .build()
        .unwrap();
    let manifests = [manifest(
        &schema,
        unpartitioned(&schema),
        vec![unknown, all_null],
    )];

    let kept = selected(&schema, &manifests, &long_predicate(CompareType::Equal, 5));

    assert_eq!(kept, ["unknown"]);
}

#[test]
fn a_column_added_after_a_manifest_was_written_is_null() {
    let current = schema(2, "added", PrimitiveType::Long);
    let stored = schema(1, "original", PrimitiveType::Long);
    let files = vec![file_holding("older", Datum::long(5))];
    let manifests = [manifest(&stored, unpartitioned(&stored), files)];

    let kept = selected(&current, &manifests, &long_predicate(CompareType::Equal, 5));

    assert!(kept.is_empty());
}

#[test]
fn only_live_non_empty_files_are_selected_in_manifest_order() {
    let schema = schema(1, "id", PrimitiveType::Long);
    let empty = file("empty").record_count(0).build().unwrap();
    let entries = vec![
        (ManifestStatus::Added, file_holding("added", Datum::long(1))),
        (
            ManifestStatus::Deleted,
            file_holding("deleted", Datum::long(1)),
        ),
        (ManifestStatus::Added, empty),
        (
            ManifestStatus::Existing,
            file_holding("existing", Datum::long(1)),
        ),
        (
            ManifestStatus::Existing,
            file_holding("other", Datum::long(2)),
        ),
    ];
    let manifests = [manifest_of_entries(
        &schema,
        unpartitioned(&schema),
        entries,
    )];

    let unfiltered = selected(&schema, &manifests, &[]);
    let ones = selected(&schema, &manifests, &long_predicate(CompareType::Equal, 1));

    assert_eq!(unfiltered, ["added", "existing", "other"]);
    assert_eq!(ones, ["added", "existing"]);
}

#[test]
fn a_file_is_pruned_by_the_partition_value_of_its_own_spec() {
    let stored = schema(1, "id", PrimitiveType::Int);
    let current = schema(1, "id", PrimitiveType::Long);
    let larger = i64::from(i32::MAX) + 1;
    let in_partition = |path: &str, value: Datum| {
        file(path)
            .partition(Struct::from_iter([Some(Literal::from(value))]))
            .build()
            .unwrap()
    };
    let manifests = [
        // Written while the column was an INT.
        manifest(
            &stored,
            partitioned_by(&stored, 0, IcebergTransform::Identity),
            vec![in_partition("five", Datum::int(5))],
        ),
        manifest(
            &current,
            partitioned_by(&current, 1, IcebergTransform::Identity),
            vec![in_partition("larger", Datum::long(larger))],
        ),
        manifest(
            &current,
            partitioned_by(&current, 2, IcebergTransform::Truncate(10)),
            vec![in_partition("fifties", Datum::long(50))],
        ),
        manifest(
            &current,
            PartitionSpec::builder(current.clone())
                .with_spec_id(3)
                .build()
                .unwrap(),
            vec![file("unpartitioned").build().unwrap()],
        ),
    ];

    let five = selected(&current, &manifests, &long_predicate(CompareType::Equal, 5));
    let fifty_five = selected(
        &current,
        &manifests,
        &long_predicate(CompareType::Equal, 55),
    );
    let large = selected(
        &current,
        &manifests,
        &long_predicate(CompareType::Equal, larger),
    );

    assert_eq!(five, ["five", "unpartitioned"]);
    assert_eq!(fifty_five, ["fifties", "unpartitioned"]);
    assert_eq!(large, ["larger", "unpartitioned"]);
}

#[test]
fn a_promoted_float_partition_value_is_compared_exactly() {
    let current = schema(1, "value", PrimitiveType::Double);
    let stored = schema(1, "value", PrimitiveType::Float);
    let one = file("one")
        .partition(Struct::from_iter([Some(Literal::from(Datum::float(
            1.0_f32,
        )))]))
        .build()
        .unwrap();
    let spec = partitioned_by(&stored, 0, IcebergTransform::Identity);
    let manifests = [manifest(&stored, spec, vec![one])];

    // 1.00000005 rounds to 1.0 as a FLOAT, but the stored 1.0 differs from it.
    let differs = selected(
        &current,
        &manifests,
        &double_predicate(CompareType::NotEqual, 1.00000005),
    );
    let equals = selected(
        &current,
        &manifests,
        &double_predicate(CompareType::Equal, 1.00000005),
    );

    assert_eq!(differs, ["one"]);
    assert!(equals.is_empty());
}

fn id_column() -> Box<Expression> {
    Box::new(Expression::Ref(Ref {
        column_idx: 0,
        return_type: PivotType::Int64,
        name: None,
    }))
}

fn long_constant(value: Option<i64>) -> Expression {
    Expression::Constant(Scalar::new(
        Arc::new(Int64Array::from(vec![value])) as ArrayRef
    ))
}

fn equals(value: i64) -> Expression {
    long_predicate(CompareType::Equal, value).remove(0)
}

/// A table of one BIGINT column `id`, and a manifest of three files holding
/// 10, 20 and 30.
fn ten_twenty_thirty() -> (SchemaRef, [Manifest; 1]) {
    let schema = schema(1, "id", PrimitiveType::Long);
    let files = vec![
        file_holding("ten", Datum::long(10)),
        file_holding("twenty", Datum::long(20)),
        file_holding("thirty", Datum::long(30)),
    ];
    let manifests = [manifest(&schema, unpartitioned(&schema), files)];
    (schema, manifests)
}

#[test]
fn an_or_keeps_the_files_either_side_may_match() {
    let (schema, manifests) = ten_twenty_thirty();
    let ten_or_thirty = Expression::Conjunction(Conjunction {
        op: ConjunctionOp::Or,
        children: vec![equals(10), equals(30)],
    });

    let kept = selected(&schema, &manifests, &[ten_or_thirty]);

    assert_eq!(kept, ["ten", "thirty"]);
}

#[test]
fn an_or_with_a_side_metadata_cannot_answer_keeps_every_file() {
    let (schema, manifests) = ten_twenty_thirty();
    let is_null = Expression::IsNull(IsNull {
        negated: false,
        input: id_column(),
    });
    let ten_or_null = Expression::Conjunction(Conjunction {
        op: ConjunctionOp::Or,
        children: vec![equals(10), is_null],
    });

    let kept = selected(&schema, &manifests, &[ten_or_null]);

    assert_eq!(kept, ["ten", "twenty", "thirty"]);
}

#[test]
fn an_in_list_keeps_the_files_that_may_hold_one_of_its_values() {
    let (schema, manifests) = ten_twenty_thirty();
    let in_ten_or_thirty = Expression::InList(InList {
        input: id_column(),
        values: vec![long_constant(Some(10)), long_constant(Some(30))],
    });

    let kept = selected(&schema, &manifests, &[in_ten_or_thirty]);

    assert_eq!(kept, ["ten", "thirty"]);
}

#[test]
fn a_between_keeps_the_files_that_overlap_its_range() {
    let (schema, manifests) = ten_twenty_thirty();
    let from_fifteen_to_thirty = Expression::Between(Between {
        input: id_column(),
        lower: Box::new(long_constant(Some(15))),
        upper: Box::new(long_constant(Some(30))),
        lower_inclusive: true,
        upper_inclusive: false,
    });

    let kept = selected(&schema, &manifests, &[from_fifteen_to_thirty]);

    assert_eq!(kept, ["twenty"]);
}

#[test]
fn every_filter_has_to_keep_a_file() {
    let (schema, manifests) = ten_twenty_thirty();
    let above_ten = long_predicate(CompareType::Greater, 10).remove(0);
    let below_thirty = long_predicate(CompareType::Less, 30).remove(0);

    let kept = selected(&schema, &manifests, &[above_ten, below_thirty]);

    assert_eq!(kept, ["twenty"]);
}

#[test]
fn a_comparison_with_null_matches_no_file() {
    let (schema, manifests) = ten_twenty_thirty();
    let equals_null = Expression::Compare(Compare {
        left: id_column(),
        right: Box::new(long_constant(None)),
        compare_type: CompareType::Equal,
        return_type: PivotType::Boolean,
    });

    let kept = selected(&schema, &manifests, &[equals_null]);

    assert!(kept.is_empty());
}

fn manifest_entry(spec_id: i32, row: usize, partitions: Option<Vec<FieldSummary>>) -> ManifestFile {
    ManifestFile {
        manifest_path: format!("manifest-{row}.avro"),
        manifest_length: 100 + row as i64,
        partition_spec_id: spec_id,
        content: ManifestContentType::Data,
        sequence_number: 1,
        min_sequence_number: 1,
        added_snapshot_id: 1,
        added_files_count: Some(1),
        existing_files_count: Some(0),
        deleted_files_count: Some(0),
        added_rows_count: Some(row as u64 + 1),
        existing_rows_count: Some(2),
        deleted_rows_count: Some(0),
        partitions,
        key_metadata: None,
        first_row_id: None,
    }
}

/// The summary of a partition field holding one value, or only NULLs.
fn summary(bytes: Option<Vec<u8>>) -> FieldSummary {
    FieldSummary {
        contains_null: bytes.is_none(),
        contains_nan: Some(false),
        lower_bound: bytes.clone().map(Into::into),
        upper_bound: bytes.map(Into::into),
    }
}

/// A table of `at` and `region` whose spec changed three times: by region and
/// month, by day, by month and region, and unpartitioned.
fn evolved_table() -> TableMetadata {
    use iceberg::spec::{SortOrder, TableMetadataBuilder, UnboundPartitionSpec};
    let schema = Schema::builder()
        .with_fields(vec![
            Arc::new(NestedField::optional(
                1,
                "at",
                Type::Primitive(PrimitiveType::Timestamp),
            )),
            Arc::new(NestedField::optional(
                2,
                "region",
                Type::Primitive(PrimitiveType::String),
            )),
        ])
        .build()
        .unwrap();
    let spec = |fields: &[(i32, &str, IcebergTransform)]| {
        let mut spec = UnboundPartitionSpec::builder();
        for (source_id, name, transform) in fields {
            spec = spec
                .add_partition_field(*source_id, name.to_string(), *transform)
                .unwrap();
        }
        spec.build()
    };
    let by_region_and_month = spec(&[
        (2, "region", IcebergTransform::Identity),
        (1, "at_month", IcebergTransform::Month),
    ]);
    TableMetadataBuilder::new(
        schema,
        by_region_and_month,
        SortOrder::unsorted_order(),
        "s3://test/table".into(),
        FormatVersion::V2,
        HashMap::new(),
    )
    .unwrap()
    .add_partition_spec(spec(&[(1, "at_day", IcebergTransform::Day)]))
    .unwrap()
    .add_partition_spec(spec(&[
        (1, "at_month", IcebergTransform::Month),
        (2, "region", IcebergTransform::Identity),
    ]))
    .unwrap()
    .add_partition_spec(spec(&[]))
    .unwrap()
    .build()
    .unwrap()
    .metadata
}

#[test]
fn a_manifest_is_pruned_by_the_partition_summaries_of_its_own_spec() {
    let metadata = evolved_table();
    let integer = |value: i32| summary(Some(value.to_le_bytes().to_vec()));
    let region = || summary(Some(b"west".to_vec()));
    // Day 18263 is 2020-01-02, in month 600.
    let manifests = vec![
        manifest_entry(0, 0, Some(vec![region(), integer(600)])),
        manifest_entry(1, 1, Some(vec![integer(18_263)])),
        manifest_entry(2, 2, Some(vec![integer(601), region()])),
        manifest_entry(1, 3, Some(vec![integer(18_262)])),
        manifest_entry(3, 4, None),
        manifest_entry(0, 5, None),
        manifest_entry(0, 6, Some(vec![region(), summary(None)])),
    ];
    let list = ManifestList::new(&metadata, manifests).unwrap();
    let before_day_18263 = predicate(
        CompareType::Less,
        Arc::new(TimestampMicrosecondArray::from(vec![
            18_263 * 86_400_000_000_i64,
        ])),
    );

    let selected = list.select(&before_day_18263).unwrap();

    let paths: Vec<_> = selected
        .data
        .iter()
        .map(|manifest| &manifest.manifest_path)
        .collect();
    let lengths: Vec<_> = selected
        .data
        .iter()
        .map(|manifest| manifest.manifest_length)
        .collect();
    assert_eq!(
        paths,
        [
            "manifest-0.avro",
            "manifest-3.avro",
            "manifest-4.avro",
            "manifest-5.avro"
        ]
    );
    assert_eq!(lengths, [100, 103, 104, 105]);
    assert_eq!(list.select(&[]).unwrap().data.len(), 7);
}

#[test]
fn delete_manifests_are_listed_apart_and_pruned_like_data_manifests() {
    let metadata = evolved_table();
    let integer = |value: i32| summary(Some(value.to_le_bytes().to_vec()));
    let mut deletes_of_day_18263 = manifest_entry(1, 1, Some(vec![integer(18_263)]));
    deletes_of_day_18263.content = ManifestContentType::Deletes;
    let mut deletes_of_day_18262 = manifest_entry(1, 3, Some(vec![integer(18_262)]));
    deletes_of_day_18262.content = ManifestContentType::Deletes;
    let manifests = vec![
        manifest_entry(3, 0, None),
        deletes_of_day_18263,
        deletes_of_day_18262,
    ];
    let list = ManifestList::new(&metadata, manifests).unwrap();
    let before_day_18263 = predicate(
        CompareType::Less,
        Arc::new(TimestampMicrosecondArray::from(vec![
            18_263 * 86_400_000_000_i64,
        ])),
    );

    let selected = list.select(&before_day_18263).unwrap();

    let paths = |manifests: &[&ManifestFile]| -> Vec<String> {
        manifests
            .iter()
            .map(|manifest| manifest.manifest_path.clone())
            .collect()
    };
    assert_eq!(paths(&selected.data), ["manifest-0.avro"]);
    assert_eq!(paths(&selected.deletes), ["manifest-3.avro"]);
    assert_eq!(list.data_manifests().count(), 1);
    assert_eq!(list.delete_manifests().count(), 2);
    assert_eq!(list.row_count(), Some(3));
}

#[test]
fn the_row_count_is_known_only_when_every_manifest_counts_its_rows() {
    let metadata = evolved_table();
    let counted = vec![manifest_entry(3, 0, None), manifest_entry(3, 1, None)];
    let mut uncounted = counted.clone();
    uncounted[0].added_rows_count = None;
    let mut overflowing = counted.clone();
    overflowing[0].added_rows_count = Some(u64::MAX);

    let row_count = |manifests: &[ManifestFile]| {
        ManifestList::new(&metadata, manifests.to_vec())
            .unwrap()
            .row_count()
    };

    assert_eq!(row_count(&counted), Some(7));
    assert_eq!(row_count(&uncounted), None);
    assert_eq!(row_count(&overflowing), None);
    assert_eq!(row_count(&[]), Some(0));
}

#[test]
fn the_estimated_row_count_discounts_the_delete_manifests_row_totals() {
    let metadata = evolved_table();
    let mut deletes = manifest_entry(1, 1, None);
    deletes.content = ManifestContentType::Deletes;
    let mut too_many_deletes = deletes.clone();
    too_many_deletes.added_rows_count = Some(100);
    let data = manifest_entry(1, 6, None);

    let estimate = |manifests: Vec<ManifestFile>| {
        ManifestList::new(&metadata, manifests)
            .unwrap()
            .estimated_row_count()
    };

    assert_eq!(estimate(vec![data.clone(), deletes]), Some(5));
    assert_eq!(estimate(vec![data.clone(), too_many_deletes]), Some(0));
    assert_eq!(estimate(vec![data]), Some(9));
}

#[test]
fn a_partition_requirement_iceberg_cannot_express_keeps_the_file() {
    let price = PrimitiveType::Decimal {
        precision: 10,
        scale: 2,
    };
    let schema = schema(1, "price", price);
    let truncated = Datum::decimal_from_str("12.30").unwrap();
    let priced = file("priced")
        .partition(Struct::from_iter([Some(Literal::from(truncated))]))
        .build()
        .unwrap();
    let spec = partitioned_by(&schema, 0, IcebergTransform::Truncate(10));
    let manifests = [manifest(&schema, spec, vec![priced])];
    let below_fifty = Decimal128Array::from(vec![5000])
        .with_precision_and_scale(10, 2)
        .unwrap();

    let kept = selected(
        &schema,
        &manifests,
        &predicate(CompareType::Less, Arc::new(below_fifty)),
    );

    assert_eq!(kept, ["priced"]);
}

/// A file named "bounded" whose field 1 lies between `lower` and `upper`,
/// without NULLs.
fn file_between(lower: Datum, upper: Datum) -> DataFile {
    file("bounded")
        .null_value_counts(HashMap::from([(1, 0)]))
        .lower_bounds(HashMap::from([(1, lower)]))
        .upper_bounds(HashMap::from([(1, upper)]))
        .build()
        .unwrap()
}

#[test]
fn a_file_is_kept_exactly_when_a_value_within_its_bounds_would_match() {
    type Matches = fn(i64, i64) -> bool;
    let comparisons: [(CompareType, Matches); 6] = [
        (CompareType::Equal, |value, constant| value == constant),
        (CompareType::NotEqual, |value, constant| value != constant),
        (CompareType::Less, |value, constant| value < constant),
        (CompareType::LessEqual, |value, constant| value <= constant),
        (CompareType::Greater, |value, constant| value > constant),
        (CompareType::GreaterEqual, |value, constant| {
            value >= constant
        }),
    ];
    let schema = schema(1, "id", PrimitiveType::Long);

    for (lower, upper) in [(10, 20), (15, 15)] {
        let bounded = file_between(Datum::long(lower), Datum::long(upper));
        let manifests = [manifest(&schema, unpartitioned(&schema), vec![bounded])];
        for (compare_type, matches) in comparisons {
            for constant in lower - 1..=upper + 1 {
                let kept = selected(&schema, &manifests, &long_predicate(compare_type, constant));

                let may_match = (lower..=upper).any(|value| matches(value, constant));
                assert_eq!(
                    !kept.is_empty(),
                    may_match,
                    "[{lower}, {upper}] {compare_type:?} {constant}"
                );
            }
        }
    }
}

#[test]
fn bounds_of_every_column_type_compare_with_constants_of_that_type() {
    let decimal = PrimitiveType::Decimal {
        precision: 10,
        scale: 2,
    };
    let utc = |micros: i64| TimestampMicrosecondArray::from(vec![micros]).with_timezone("UTC");
    let decimal_constant = |unscaled: i128| {
        Decimal128Array::from(vec![unscaled])
            .with_precision_and_scale(10, 2)
            .unwrap()
    };
    // Each column holds values from `lower` to `upper`; `inside` is one of
    // them and `outside` is not.
    let cases: Vec<(PrimitiveType, Datum, Datum, ArrayRef, ArrayRef)> = vec![
        (
            PrimitiveType::Boolean,
            Datum::bool(false),
            Datum::bool(false),
            Arc::new(BooleanArray::from(vec![false])),
            Arc::new(BooleanArray::from(vec![true])),
        ),
        (
            PrimitiveType::Int,
            Datum::int(10),
            Datum::int(20),
            Arc::new(Int32Array::from(vec![15])),
            Arc::new(Int32Array::from(vec![25])),
        ),
        (
            PrimitiveType::Float,
            Datum::float(1.0_f32),
            Datum::float(2.0_f32),
            Arc::new(Float32Array::from(vec![1.5])),
            Arc::new(Float32Array::from(vec![2.5])),
        ),
        (
            PrimitiveType::Double,
            Datum::double(1.0),
            Datum::double(2.0),
            Arc::new(Float64Array::from(vec![1.5])),
            Arc::new(Float64Array::from(vec![2.5])),
        ),
        (
            PrimitiveType::Date,
            Datum::date(10),
            Datum::date(20),
            Arc::new(Date32Array::from(vec![15])),
            Arc::new(Date32Array::from(vec![25])),
        ),
        (
            PrimitiveType::Timestamp,
            Datum::timestamp_micros(10),
            Datum::timestamp_micros(20),
            Arc::new(TimestampMicrosecondArray::from(vec![15])),
            Arc::new(TimestampMicrosecondArray::from(vec![25])),
        ),
        (
            PrimitiveType::Timestamptz,
            Datum::timestamptz_micros(10),
            Datum::timestamptz_micros(20),
            Arc::new(utc(15)),
            Arc::new(utc(25)),
        ),
        (
            decimal,
            Datum::decimal_from_str("1.00").unwrap(),
            Datum::decimal_from_str("2.00").unwrap(),
            Arc::new(decimal_constant(150)),
            Arc::new(decimal_constant(250)),
        ),
        (
            PrimitiveType::String,
            Datum::string("b"),
            Datum::string("d"),
            Arc::new(StringViewArray::from(vec!["c"])),
            Arc::new(StringViewArray::from(vec!["e"])),
        ),
    ];

    for (column_type, lower, upper, inside, outside) in cases {
        let schema = schema(1, "value", column_type.clone());
        let manifests = [manifest(
            &schema,
            unpartitioned(&schema),
            vec![file_between(lower, upper)],
        )];

        let inside = selected(&schema, &manifests, &predicate(CompareType::Equal, inside));
        let outside = selected(&schema, &manifests, &predicate(CompareType::Equal, outside));

        assert_eq!(inside, ["bounded"], "{column_type}");
        assert!(outside.is_empty(), "{column_type}");
    }
}

#[test]
fn an_in_list_longer_than_the_limit_keeps_every_file() {
    let (schema, manifests) = ten_twenty_thirty();
    let in_many_other_values = Expression::InList(InList {
        input: id_column(),
        values: (100..301).map(|value| long_constant(Some(value))).collect(),
    });

    let kept = selected(&schema, &manifests, &[in_many_other_values]);

    assert_eq!(kept, ["ten", "twenty", "thirty"]);
}

#[test]
fn an_in_list_within_the_limit_prunes_by_each_of_its_values() {
    let (schema, manifests) = ten_twenty_thirty();
    // The odd numbers from 15 to 413, and 30.
    let mut values: Vec<_> = (0..199)
        .map(|step| long_constant(Some(15 + 2 * step)))
        .collect();
    values.push(long_constant(Some(30)));
    let in_many_odd_values_and_thirty = Expression::InList(InList {
        input: id_column(),
        values,
    });

    let kept = selected(&schema, &manifests, &[in_many_odd_values_and_thirty]);

    assert_eq!(kept, ["thirty"]);
}

#[test]
fn a_constant_on_the_left_keeps_every_file() {
    let (schema, manifests) = ten_twenty_thirty();
    let hundred_equals_id = Expression::Compare(Compare {
        left: Box::new(long_constant(Some(100))),
        right: id_column(),
        compare_type: CompareType::Equal,
        return_type: PivotType::Boolean,
    });

    let kept = selected(&schema, &manifests, &[hundred_equals_id]);

    assert_eq!(kept, ["ten", "twenty", "thirty"]);
}

#[test]
fn a_filter_the_column_type_cannot_read_leaves_the_others_to_prune() {
    let (schema, manifests) = ten_twenty_thirty();
    let equals_text = predicate(
        CompareType::Equal,
        Arc::new(StringViewArray::from(vec!["ten"])),
    )
    .remove(0);
    let above_fifteen = long_predicate(CompareType::Greater, 15).remove(0);

    let kept = selected(&schema, &manifests, &[equals_text, above_fifteen]);

    assert_eq!(kept, ["twenty", "thirty"]);
}

#[test]
fn a_constant_beyond_the_column_type_matches_every_value_or_none() {
    let schema = schema(1, "id", PrimitiveType::Int);
    let files = vec![file_holding("five", Datum::int(5))];
    let manifests = [manifest(&schema, unpartitioned(&schema), files)];
    let beyond_int = i64::from(i32::MAX) + 1;

    let above = selected(
        &schema,
        &manifests,
        &long_predicate(CompareType::Greater, beyond_int),
    );
    let below = selected(
        &schema,
        &manifests,
        &long_predicate(CompareType::Less, beyond_int),
    );

    assert!(above.is_empty());
    assert_eq!(below, ["five"]);
}

#[test]
fn a_void_partition_field_prunes_nothing() {
    let schema = schema(1, "id", PrimitiveType::Long);
    let voided = file("voided")
        .partition(Struct::from_iter([None::<Literal>]))
        .null_value_counts(HashMap::from([(1, 0)]))
        .lower_bounds(HashMap::from([(1, Datum::long(5))]))
        .upper_bounds(HashMap::from([(1, Datum::long(5))]))
        .build()
        .unwrap();
    let spec = partitioned_by(&schema, 0, IcebergTransform::Void);
    let manifests = [manifest(&schema, spec, vec![voided])];

    let five = selected(&schema, &manifests, &long_predicate(CompareType::Equal, 5));
    let six = selected(&schema, &manifests, &long_predicate(CompareType::Equal, 6));

    assert_eq!(five, ["voided"]);
    assert!(six.is_empty());
}

#[test]
fn a_date_before_the_epoch_also_keeps_the_month_after_its_own() {
    let schema = schema(1, "day", PrimitiveType::Date);
    let in_month = |path: &str, months_since_epoch: i32| {
        file(path)
            .partition(Struct::from_iter([Some(Literal::int(months_since_epoch))]))
            .build()
            .unwrap()
    };
    let files = vec![
        in_month("october 1969", -3),
        in_month("november 1969", -2),
        in_month("december 1969", -1),
        in_month("january 1970", 0),
        in_month("february 1970", 1),
    ];
    let spec = partitioned_by(&schema, 0, IcebergTransform::Month);
    let manifests = [manifest(&schema, spec, files)];
    // 1969-12-15 is 17 days before the epoch. Iceberg projects a date before
    // the epoch onto its own month and the next: some writers put it there.
    let mid_december = predicate(CompareType::Equal, Arc::new(Date32Array::from(vec![-17])));

    let kept = selected(&schema, &manifests, &mid_december);

    assert_eq!(kept, ["december 1969", "january 1970"]);
}

#[test]
fn a_manifest_of_a_spec_the_table_does_not_list_is_kept() {
    let metadata = evolved_table();
    let day = |value: i32| summary(Some(value.to_le_bytes().to_vec()));
    let manifests = vec![
        manifest_entry(1, 0, Some(vec![day(18_264)])),
        manifest_entry(99, 1, Some(vec![day(18_264)])),
    ];
    let list = ManifestList::new(&metadata, manifests).unwrap();
    let before_day_18263 = predicate(
        CompareType::Less,
        Arc::new(TimestampMicrosecondArray::from(vec![
            18_263 * 86_400_000_000_i64,
        ])),
    );

    let selected = list.select(&before_day_18263).unwrap();

    let paths: Vec<_> = selected
        .data
        .iter()
        .map(|manifest| &manifest.manifest_path)
        .collect();
    assert_eq!(paths, ["manifest-1.avro"]);
}

#[test]
fn a_summary_that_does_not_match_its_spec_is_an_error() {
    let metadata = evolved_table();
    let region_only = vec![summary(Some(b"west".to_vec()))];
    let manifests = vec![manifest_entry(0, 0, Some(region_only))];

    let list = ManifestList::new(&metadata, manifests);

    assert!(list.is_err());
}

#[test]
fn a_partition_tuple_that_does_not_match_its_spec_is_an_error() {
    let schema = schema(1, "id", PrimitiveType::Long);
    let without_a_value = file("short").build().unwrap();
    let spec = partitioned_by(&schema, 0, IcebergTransform::Identity);
    let manifests = [manifest(&schema, spec, vec![without_a_value])];

    let selected = data_files(&schema, &manifests).select(&long_predicate(CompareType::Equal, 5));

    assert!(selected.is_err());
}

#[test]
fn a_wide_or_is_checked_like_a_narrow_one() {
    let (schema, manifests) = ten_twenty_thirty();
    let mut children: Vec<_> = (1_000..11_000).map(equals).collect();
    children.push(equals(20));
    let twenty_or_many_others = Expression::Conjunction(Conjunction {
        op: ConjunctionOp::Or,
        children,
    });

    let kept = selected(&schema, &manifests, &[twenty_or_many_others]);

    assert_eq!(kept, ["twenty"]);
}
