//! SQL over a variant (JSON) column stored in real parquet files: the planner
//! projects the `doc` column, the catalog scan reassembles its leaves (however
//! the file shredded them), and the `->`/cast expressions read the paths.

use std::sync::Arc;

use arrow_array::{ArrayRef, RecordBatch, StringArray};
use arrow_schema::{DataType, Schema};
use parquet_variant_compute::{ShreddedSchemaBuilder, json_to_variant, shred_variant};

use crate::common::*;
use planner::catalog::Column;
use planner::types::Type;
use rstest::rstest;

/// Register a parquet-backed `docs(d VARIANT)` table holding `rows` as JSON.
fn docs_table(planner: &mut TestingPlanner, rows: Vec<&str>) {
    let json: ArrayRef = Arc::new(StringArray::from(rows));
    let docs = json_to_variant(&json).unwrap().into_inner();
    planner.add_table("docs", &[("d", Type::Variant, Arc::new(docs) as ArrayRef)]);
}

/// Like [`docs_table`], but with `path` shredded into a typed leaf, so the file
/// carries the extra `typed_value` chunks a real ingest would write.
fn shredded_docs_table(planner: &mut TestingPlanner, rows: Vec<&str>, path: &str, ty: &DataType) {
    let json: ArrayRef = Arc::new(StringArray::from(rows));
    let shred = ShreddedSchemaBuilder::new()
        .with_path(path, ty)
        .unwrap()
        .build();
    let shredded = shred_variant(&json_to_variant(&json).unwrap(), &shred).unwrap();
    let docs = shredded.into_inner();
    planner.add_table("docs", &[("d", Type::Variant, Arc::new(docs) as ArrayRef)]);
}

#[rstest]
fn casts_a_path_from_parquet(mut testing_planner: TestingPlanner) {
    docs_table(&mut testing_planner, vec![r#"{"age":30}"#, r#"{"age":25}"#]);

    let rows = run(
        &mut testing_planner,
        "SELECT CAST(d->'age' AS BIGINT) AS a FROM docs ORDER BY a",
    );

    let got: Vec<i64> = rows
        .iter()
        .map(|r| only_column(r).as_i64().unwrap())
        .collect();
    assert_eq!(got, vec![25, 30]);
}

#[rstest]
fn casts_a_shredded_path_from_parquet(mut testing_planner: TestingPlanner) {
    shredded_docs_table(
        &mut testing_planner,
        vec![r#"{"age":30}"#, r#"{"name":"bob"}"#],
        "age",
        &DataType::Int64,
    );

    let rows = run(
        &mut testing_planner,
        "SELECT CAST(d->'age' AS BIGINT) AS a FROM docs",
    );

    // The JSON writer omits null fields, so the missing row has no "a" key
    // (indexing yields JSON null).
    let mut got: Vec<Option<i64>> = rows.iter().map(|r| r["a"].as_i64()).collect();
    got.sort();
    assert_eq!(got, vec![None, Some(30)]);
}

/// A typed cast of a path the file provably lacks: every row is perfectly
/// shredded, so the untyped fallback is all NULL and the path resolves to SQL
/// NULL everywhere instead of failing the scan.
#[rstest]
fn casts_an_absent_path_from_a_fully_shredded_parquet(mut testing_planner: TestingPlanner) {
    shredded_docs_table(
        &mut testing_planner,
        vec![r#"{"age":30}"#, r#"{"age":25}"#],
        "age",
        &DataType::Int64,
    );

    let rows = run(
        &mut testing_planner,
        "SELECT max(CAST(d->'latency' AS FLOAT)) AS m FROM docs",
    );

    assert_eq!(rows[0]["m"].as_f64(), None);
}

/// A single-column `docs(d)` batch holding `rows`, shredding `path` into a
/// typed leaf when given.
fn docs_batch_shredded_as(rows: Vec<&str>, shred: Option<(&str, &DataType)>) -> RecordBatch {
    let json: ArrayRef = Arc::new(StringArray::from(rows));
    let variant = json_to_variant(&json).unwrap();
    let (field, array) = match shred {
        Some((path, ty)) => {
            let shred = ShreddedSchemaBuilder::new()
                .with_path(path, ty)
                .unwrap()
                .build();
            let shredded = shred_variant(&variant, &shred).unwrap();
            (shredded.field("d"), shredded.into_inner())
        }
        None => (variant.field("d"), variant.into_inner()),
    };
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![field])),
        vec![Arc::new(array) as _],
    )
    .unwrap()
}

/// A single-column `docs(d)` batch holding `rows`, with `shred_age` deciding
/// whether the file gets a typed `age` leaf or stays unshredded.
fn docs_batch(rows: Vec<&str>, shred_age: bool) -> RecordBatch {
    docs_batch_shredded_as(rows, shred_age.then_some(("age", &DataType::Int64)))
}

/// Register `docs(d VARIANT)` over several parquet files, one per batch.
fn docs_table_files(planner: &mut TestingPlanner, batches: &[RecordBatch]) {
    planner.add_table_files(
        "docs",
        vec![Column {
            name: "d".to_string(),
            col_type: Type::Variant,
        }],
        batches,
    );
}

/// `SELECT d.age` across two files where only one shreds `age`: each file's
/// leaves resolve on their own, and both render as the same JSON text.
#[rstest]
fn dot_access_across_mixed_shredding(mut testing_planner: TestingPlanner) {
    let shredded = docs_batch(vec![r#"{"age":30}"#, r#"{"age":25}"#], true);
    let unshredded = docs_batch(vec![r#"{"age":40}"#, r#"{"name":"bob"}"#], false);
    docs_table_files(&mut testing_planner, &[shredded, unshredded]);

    let rows = run(&mut testing_planner, "SELECT d.age AS a FROM docs");

    let mut got: Vec<Option<String>> = rows
        .iter()
        .map(|r| r["a"].as_str().map(str::to_string))
        .collect();
    got.sort();
    assert_eq!(
        got,
        vec![
            None,
            Some("25".to_string()),
            Some("30".to_string()),
            Some("40".to_string()),
        ]
    );
}

/// `SELECT d.age` across a file that provably lacks `age` (fully shredded on
/// another field) and one that shreds it: the lacking file's rows come back
/// SQL NULL and both files' layouts merge into one column.
#[rstest]
fn dot_access_across_a_file_provably_lacking_the_path(mut testing_planner: TestingPlanner) {
    let lacking = docs_batch_shredded_as(
        vec![r#"{"name":"bob"}"#, r#"{"name":"amy"}"#],
        Some(("name", &DataType::Utf8View)),
    );
    let shredded = docs_batch(vec![r#"{"age":30}"#], true);
    docs_table_files(&mut testing_planner, &[lacking, shredded]);

    let rows = run(&mut testing_planner, "SELECT d.age AS a FROM docs");

    let mut got: Vec<Option<String>> = rows
        .iter()
        .map(|r| r["a"].as_str().map(str::to_string))
        .collect();
    got.sort();
    assert_eq!(got, vec![None, None, Some("30".to_string())]);
}

/// The `d.age` dot syntax over a real shredded parquet file.
#[rstest]
fn dot_access_over_shredded_parquet(mut testing_planner: TestingPlanner) {
    shredded_docs_table(
        &mut testing_planner,
        vec![r#"{"age":30}"#, r#"{"name":"bob"}"#],
        "age",
        &DataType::Int64,
    );

    let rows = run(
        &mut testing_planner,
        "SELECT CAST(d.age AS BIGINT) AS a FROM docs",
    );

    let mut got: Vec<Option<i64>> = rows.iter().map(|r| r["a"].as_i64()).collect();
    got.sort();
    assert_eq!(got, vec![None, Some(30)]);
}

#[rstest]
fn groups_by_a_cast_path_from_parquet(mut testing_planner: TestingPlanner) {
    shredded_docs_table(
        &mut testing_planner,
        vec![r#"{"age":30}"#, r#"{"age":30}"#, r#"{"age":25}"#],
        "age",
        &DataType::Int64,
    );

    let rows = run(
        &mut testing_planner,
        "SELECT CAST(d->'age' AS BIGINT) AS a, count(*) AS n FROM docs GROUP BY a ORDER BY a",
    );

    let got: Vec<(i64, i64)> = rows
        .iter()
        .map(|r| (r["a"].as_i64().unwrap(), r["n"].as_i64().unwrap()))
        .collect();
    assert_eq!(got, vec![(25, 1), (30, 2)]);
}

#[rstest]
fn filters_on_a_cast_path_from_parquet(mut testing_planner: TestingPlanner) {
    docs_table(
        &mut testing_planner,
        vec![r#"{"age":30}"#, r#"{"age":25}"#, r#"{"name":"bob"}"#],
    );

    let rows = run(
        &mut testing_planner,
        "SELECT CAST(d->'age' AS BIGINT) AS a FROM docs \
         WHERE CAST(d->'age' AS BIGINT) > 27",
    );

    let got: Vec<i64> = rows
        .iter()
        .map(|r| only_column(r).as_i64().unwrap())
        .collect();
    assert_eq!(got, vec![30]);
}

/// A typed cast across shredded and unshredded files: one file reads its
/// typed leaf, the other decodes binary blobs, and the outputs unify.
#[rstest]
fn casts_a_path_across_mixed_shredding(mut testing_planner: TestingPlanner) {
    let shredded = docs_batch(vec![r#"{"age":30}"#, r#"{"age":25}"#], true);
    let unshredded = docs_batch(vec![r#"{"age":40}"#, r#"{"name":"bob"}"#], false);
    docs_table_files(&mut testing_planner, &[shredded, unshredded]);

    let rows = run(
        &mut testing_planner,
        "SELECT CAST(d->'age' AS BIGINT) AS a FROM docs",
    );

    let mut got: Vec<Option<i64>> = rows.iter().map(|r| r["a"].as_i64()).collect();
    got.sort();
    assert_eq!(got, vec![None, Some(25), Some(30), Some(40)]);
}

#[rstest]
fn casts_a_path_shredded_as_different_types_per_file(mut testing_planner: TestingPlanner) {
    // Setup
    let ints = docs_batch_shredded_as(
        vec![r#"{"age":30}"#, r#"{"age":25}"#],
        Some(("age", &DataType::Int64)),
    );
    let texts = docs_batch_shredded_as(vec![r#"{"age":"forty"}"#], Some(("age", &DataType::Utf8)));
    docs_table_files(&mut testing_planner, &[ints, texts]);

    // Execute
    let rows = run(
        &mut testing_planner,
        "SELECT CAST(d->'age' AS VARCHAR) AS a FROM docs",
    );

    // Assert
    let mut got: Vec<&str> = rows.iter().filter_map(|r| r["a"].as_str()).collect();
    got.sort();
    assert_eq!(got, vec!["25", "30", "forty"]);
}

#[rstest]
fn rejects_an_invalid_cast_from_a_shredded_leaf(mut testing_planner: TestingPlanner) {
    // Setup
    shredded_docs_table(
        &mut testing_planner,
        vec![r#"{"age":"forty"}"#],
        "age",
        &DataType::Utf8,
    );

    // Execute
    let error = run_expecting_error(
        &mut testing_planner,
        "SELECT CAST(d->'age' AS BIGINT) FROM docs",
    );

    // Assert
    assert!(error.contains("cannot cast variant value"));
}

#[rstest]
fn casts_typed_and_fallback_values_in_one_shredded_file(mut testing_planner: TestingPlanner) {
    // Setup
    shredded_docs_table(
        &mut testing_planner,
        vec![r#"{"x":30}"#, r#"{"x":"thirty"}"#, r#"{"x":null}"#, r#"{}"#],
        "x",
        &DataType::Int64,
    );

    // Execute
    let rows = run(
        &mut testing_planner,
        "SELECT CAST(d->'x' AS VARCHAR) AS x FROM docs",
    );

    // Assert
    let got: Vec<Option<&str>> = rows.iter().map(|row| row["x"].as_str()).collect();
    assert_eq!(got, vec![Some("30"), Some("thirty"), Some("null"), None]);
}

#[rstest]
fn distinguishes_a_text_cast_from_variant_output(mut testing_planner: TestingPlanner) {
    // Setup
    docs_table(&mut testing_planner, vec![r#""bob""#]);

    // Execute
    let rows = run(
        &mut testing_planner,
        "SELECT CAST(d AS VARCHAR) AS cast, d AS output FROM docs",
    );

    // Assert
    assert_eq!(rows[0]["cast"], "bob");
    assert_eq!(rows[0]["output"], r#""bob""#);
}

/// A nested path (`user.id`) shredded two levels deep reads its typed leaf.
#[rstest]
fn casts_a_nested_shredded_path(mut testing_planner: TestingPlanner) {
    shredded_docs_table(
        &mut testing_planner,
        vec![
            r#"{"user":{"id":7}}"#,
            r#"{"user":{"name":"b"}}"#,
            r#"{"name":"x"}"#,
        ],
        "user.id",
        &DataType::Int64,
    );

    let rows = run(
        &mut testing_planner,
        "SELECT CAST(d->'user'->'id' AS BIGINT) AS a FROM docs",
    );

    let mut got: Vec<Option<i64>> = rows.iter().map(|r| r["a"].as_i64()).collect();
    got.sort();
    assert_eq!(got, vec![None, None, Some(7)]);
}

#[rstest]
fn filters_on_a_nested_shredded_path(mut testing_planner: TestingPlanner) {
    shredded_docs_table(
        &mut testing_planner,
        vec![r#"{"user":{"id":7}}"#, r#"{"user":{"id":8}}"#],
        "user.id",
        &DataType::Int64,
    );

    let rows = run(
        &mut testing_planner,
        "SELECT CAST(d->'user'->'id' AS BIGINT) AS a FROM docs \
         WHERE CAST(d->'user'->'id' AS BIGINT) = 7",
    );

    let got: Vec<Option<i64>> = rows.iter().map(|r| r["a"].as_i64()).collect();
    assert_eq!(got, vec![Some(7)]);
}

/// A path that is NOT shredded but lives inside a shredded object: `user.id`
/// has a typed leaf, `user.name` falls back to the object's value blob.
#[rstest]
fn reads_an_unshredded_path_inside_a_shredded_object(mut testing_planner: TestingPlanner) {
    shredded_docs_table(
        &mut testing_planner,
        vec![r#"{"user":{"id":7,"name":"ann"}}"#, r#"{"user":{"id":8}}"#],
        "user.id",
        &DataType::Int64,
    );

    let rows = run(
        &mut testing_planner,
        "SELECT CAST(d->'user'->'name' AS VARCHAR) AS n FROM docs",
    );

    let mut got: Vec<Option<&str>> = rows.iter().map(|r| r["n"].as_str()).collect();
    got.sort();
    assert_eq!(got, vec![None, Some("ann")]);
}

#[rstest]
fn filters_out_a_missing_imperfectly_shredded_text_path(mut testing_planner: TestingPlanner) {
    shredded_docs_table(
        &mut testing_planner,
        vec![r#"{"stat":"UnaryStats"}"#, r#"{"event.name":"document"}"#],
        "stat",
        &DataType::Int64,
    );

    let rows = run(
        &mut testing_planner,
        "SELECT d FROM docs WHERE CAST(d->'stat' AS VARCHAR) != ''",
    );

    assert_eq!(rows.len(), 1);
    assert_eq!(
        only_column(&rows[0]).as_str(),
        Some(r#"{"stat":"UnaryStats"}"#)
    );
}

/// The three shapes of "no value" behave distinctly: an explicit JSON null
/// renders as `null` text, an absent path and a NULL document yield SQL NULL.
#[rstest]
fn distinguishes_json_null_absent_path_and_null_document(mut testing_planner: TestingPlanner) {
    let json: ArrayRef = Arc::new(StringArray::from(vec![
        Some(r#"{"age":null}"#),
        Some("{}"),
        None,
        Some(r#"{"age":30}"#),
    ]));
    let variant = json_to_variant(&json).unwrap();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![variant.field("d")])),
        vec![Arc::new(variant.into_inner()) as _],
    )
    .unwrap();
    docs_table_files(&mut testing_planner, &[batch]);

    let rows = run(
        &mut testing_planner,
        "SELECT CAST(d->'age' AS BIGINT) AS a, d.age AS j FROM docs",
    );

    let mut got: Vec<(Option<i64>, Option<String>)> = rows
        .iter()
        .map(|r| (r["a"].as_i64(), r["j"].as_str().map(str::to_string)))
        .collect();
    got.sort();
    assert_eq!(
        got,
        vec![
            (None, None),                     // {} : absent path
            (None, None),                     // NULL document
            (None, Some("null".to_string())), // {"age":null}
            (Some(30), Some("30".to_string())),
        ]
    );
}

/// `SELECT d` (the whole document) renders the same JSON text whether the
/// file shredded the column or not.
#[rstest]
fn selects_the_whole_document_across_mixed_shredding(mut testing_planner: TestingPlanner) {
    let shredded = docs_batch(vec![r#"{"age":30}"#], true);
    let unshredded = docs_batch(vec![r#"{"age":30}"#], false);
    docs_table_files(&mut testing_planner, &[shredded, unshredded]);

    let rows = run(&mut testing_planner, "SELECT d FROM docs");

    let got: Vec<Option<&str>> = rows.iter().map(|r| only_column(r).as_str()).collect();
    assert_eq!(got, vec![Some(r#"{"age":30}"#); 2]);
}

/// Global aggregates over a typed path read.
#[rstest]
fn aggregates_a_cast_path(mut testing_planner: TestingPlanner) {
    shredded_docs_table(
        &mut testing_planner,
        vec![r#"{"age":30}"#, r#"{"age":30}"#, r#"{"age":25}"#],
        "age",
        &DataType::Int64,
    );

    let rows = run(
        &mut testing_planner,
        "SELECT SUM(CAST(d->'age' AS BIGINT)) AS s, \
                MIN(CAST(d->'age' AS BIGINT)) AS lo, \
                MAX(CAST(d->'age' AS BIGINT)) AS hi FROM docs",
    );

    assert_eq!(rows[0]["s"].as_i64(), Some(85));
    assert_eq!(rows[0]["lo"].as_i64(), Some(25));
    assert_eq!(rows[0]["hi"].as_i64(), Some(30));
}

/// Two paths read out of the same document in one query.
#[rstest]
fn reads_two_paths_from_one_document(mut testing_planner: TestingPlanner) {
    shredded_docs_table(
        &mut testing_planner,
        vec![r#"{"age":30,"name":"ann"}"#, r#"{"age":25,"name":"bob"}"#],
        "age",
        &DataType::Int64,
    );

    let rows = run(
        &mut testing_planner,
        "SELECT CAST(d->'age' AS BIGINT) AS a, CAST(d->'name' AS VARCHAR) AS n \
         FROM docs ORDER BY a",
    );

    let got: Vec<(Option<i64>, Option<&str>)> = rows
        .iter()
        .map(|r| (r["a"].as_i64(), r["n"].as_str()))
        .collect();
    assert_eq!(got, vec![(Some(25), Some("bob")), (Some(30), Some("ann"))]);
}

/// Top-N over a typed path read.
#[rstest]
fn orders_and_limits_by_a_cast_path(mut testing_planner: TestingPlanner) {
    shredded_docs_table(
        &mut testing_planner,
        vec![
            r#"{"age":30}"#,
            r#"{"age":25}"#,
            r#"{"age":40}"#,
            r#"{"age":10}"#,
        ],
        "age",
        &DataType::Int64,
    );

    let rows = run(
        &mut testing_planner,
        "SELECT CAST(d->'age' AS BIGINT) AS a FROM docs ORDER BY a LIMIT 2",
    );

    let got: Vec<Option<i64>> = rows.iter().map(|r| r["a"].as_i64()).collect();
    assert_eq!(got, vec![Some(10), Some(25)]);
}

/// The whole document renders as JSON when the file shreds a nested path, so
/// the typed leaves sit under an object typed_value with its own object child.
#[rstest]
fn selects_the_whole_document_with_nested_shredding(mut testing_planner: TestingPlanner) {
    shredded_docs_table(
        &mut testing_planner,
        vec![r#"{"user":{"id":7}}"#],
        "user.id",
        &DataType::Int64,
    );

    let rows = run(&mut testing_planner, "SELECT d FROM docs");

    let got: Vec<Option<&str>> = rows.iter().map(|r| only_column(r).as_str()).collect();
    assert_eq!(got, vec![Some(r#"{"user":{"id":7}}"#)]);
}

/// A bare extraction of a shredded object renders the sub-variant as JSON.
#[rstest]
fn bare_extraction_of_a_shredded_object_renders_json(mut testing_planner: TestingPlanner) {
    shredded_docs_table(
        &mut testing_planner,
        vec![r#"{"user":{"id":7}}"#],
        "user.id",
        &DataType::Int64,
    );

    let rows = run(&mut testing_planner, "SELECT d->'user' AS u FROM docs");

    let got: Vec<Option<&str>> = rows.iter().map(|r| r["u"].as_str()).collect();
    assert_eq!(got, vec![Some(r#"{"id":7}"#)]);
}

/// A filtered LIMIT over whole documents spanning differently-shredded files:
/// the limit gathers raw variant batches whose physical struct layouts differ
/// per file, which no operator may try to concatenate.
#[rstest]
fn limits_filtered_whole_documents_across_mixed_shredding(mut testing_planner: TestingPlanner) {
    let shredded = docs_batch(vec![r#"{"age":30}"#, r#"{"age":31}"#], true);
    let unshredded = docs_batch(vec![r#"{"age":32}"#, r#"{"age":33}"#], false);
    docs_table_files(&mut testing_planner, &[shredded, unshredded]);

    let rows = run(
        &mut testing_planner,
        "SELECT d FROM docs WHERE CAST(d->'age' AS BIGINT) >= 30 LIMIT 3",
    );

    let mut got: Vec<Option<&str>> = rows.iter().map(|r| only_column(r).as_str()).collect();
    got.sort();
    assert_eq!(got.len(), 3);
    for doc in got {
        let doc = doc.expect("each document renders as JSON text");
        assert!(doc.starts_with(r#"{"age":3"#), "got: {doc}");
    }
}

/// Top-N whose payload is the whole document, spanning differently-shredded
/// files: the sorted gather must not concatenate the raw variant batches.
#[rstest]
fn orders_whole_documents_across_mixed_shredding(mut testing_planner: TestingPlanner) {
    let shredded = docs_batch(vec![r#"{"age":31}"#, r#"{"age":33}"#], true);
    let unshredded = docs_batch(vec![r#"{"age":32}"#, r#"{"age":30}"#], false);
    docs_table_files(&mut testing_planner, &[shredded, unshredded]);

    let rows = run(
        &mut testing_planner,
        "SELECT d FROM docs ORDER BY CAST(d->'age' AS BIGINT) LIMIT 3",
    );

    let got: Vec<Option<&str>> = rows.iter().map(|r| only_column(r).as_str()).collect();
    assert_eq!(
        got,
        vec![
            Some(r#"{"age":30}"#),
            Some(r#"{"age":31}"#),
            Some(r#"{"age":32}"#),
        ]
    );
}

/// A full sort (no LIMIT) whose payload is the whole document, spanning
/// differently-shredded files with interleaving keys: the merge gathers rows
/// across raw variant batches with per-file layouts.
#[rstest]
fn fully_sorts_whole_documents_across_mixed_shredding(mut testing_planner: TestingPlanner) {
    let odd_docs: Vec<String> = (0..100)
        .map(|i| format!(r#"{{"age":{}}}"#, 2 * i + 1))
        .collect();
    let even_docs: Vec<String> = (0..100)
        .map(|i| format!(r#"{{"age":{}}}"#, 2 * i))
        .collect();
    let shredded = docs_batch(odd_docs.iter().map(String::as_str).collect(), true);
    let unshredded = docs_batch(even_docs.iter().map(String::as_str).collect(), false);
    docs_table_files(&mut testing_planner, &[shredded, unshredded]);

    let rows = run(
        &mut testing_planner,
        "SELECT d FROM docs ORDER BY CAST(d->'age' AS BIGINT)",
    );

    let got: Vec<String> = rows
        .iter()
        .map(|r| only_column(r).as_str().unwrap().to_string())
        .collect();
    let expected: Vec<String> = (0..200).map(|age| format!(r#"{{"age":{age}}}"#)).collect();
    assert_eq!(got, expected);
}

/// A join whose probe side carries the whole document, spanning
/// differently-shredded files: the match outputter splices probe rows from
/// raw variant batches with per-file layouts.
#[rstest]
fn joins_whole_documents_across_mixed_shredding(mut testing_planner: TestingPlanner) {
    let shredded = docs_batch(vec![r#"{"age":31}"#, r#"{"age":33}"#], true);
    let unshredded = docs_batch(vec![r#"{"age":32}"#, r#"{"age":30}"#], false);
    docs_table_files(&mut testing_planner, &[shredded, unshredded]);
    testing_planner.add_table(
        "wanted",
        &[(
            "age",
            Type::Int64,
            Arc::new(arrow_array::Int64Array::from(vec![30, 31])) as ArrayRef,
        )],
    );

    let rows = run(
        &mut testing_planner,
        "SELECT d FROM docs JOIN wanted ON CAST(d->'age' AS BIGINT) = wanted.age",
    );

    let mut got: Vec<Option<String>> = rows
        .iter()
        .map(|r| only_column(r).as_str().map(str::to_string))
        .collect();
    got.sort();
    assert_eq!(
        got,
        vec![
            Some(r#"{"age":30}"#.to_string()),
            Some(r#"{"age":31}"#.to_string()),
        ]
    );
}

/// The mirror of [`joins_whole_documents_across_mixed_shredding`]: the variant
/// table is the smaller (build) side, whose stored rows are gathered across
/// its differently-shredded batches when matches emit.
#[rstest]
fn joins_whole_documents_on_the_build_side(mut testing_planner: TestingPlanner) {
    let shredded = docs_batch(vec![r#"{"age":31}"#, r#"{"age":33}"#], true);
    let unshredded = docs_batch(vec![r#"{"age":32}"#, r#"{"age":30}"#], false);
    docs_table_files(&mut testing_planner, &[shredded, unshredded]);
    testing_planner.add_table(
        "wanted",
        &[(
            "age",
            Type::Int64,
            Arc::new(arrow_array::Int64Array::from(vec![
                30, 31, 34, 35, 36, 37, 38, 39, 40, 41,
            ])) as ArrayRef,
        )],
    );

    let rows = run(
        &mut testing_planner,
        "SELECT d FROM wanted JOIN docs ON CAST(d->'age' AS BIGINT) = wanted.age",
    );

    let mut got: Vec<Option<String>> = rows
        .iter()
        .map(|r| only_column(r).as_str().map(str::to_string))
        .collect();
    got.sort();
    assert_eq!(
        got,
        vec![
            Some(r#"{"age":30}"#.to_string()),
            Some(r#"{"age":31}"#.to_string()),
        ]
    );
}

/// Files that all shred alike keep their layout through the join: the same
/// query as the mixed tests, but with nothing to unify.
#[rstest]
fn joins_whole_documents_with_uniform_shredding(mut testing_planner: TestingPlanner) {
    let first = docs_batch(vec![r#"{"age":31}"#, r#"{"age":33}"#], true);
    let second = docs_batch(vec![r#"{"age":32}"#, r#"{"age":30}"#], true);
    docs_table_files(&mut testing_planner, &[first, second]);
    testing_planner.add_table(
        "wanted",
        &[(
            "age",
            Type::Int64,
            Arc::new(arrow_array::Int64Array::from(vec![30, 31])) as ArrayRef,
        )],
    );

    let rows = run(
        &mut testing_planner,
        "SELECT d FROM docs JOIN wanted ON CAST(d->'age' AS BIGINT) = wanted.age",
    );

    let mut got: Vec<Option<String>> = rows
        .iter()
        .map(|r| only_column(r).as_str().map(str::to_string))
        .collect();
    got.sort();
    assert_eq!(
        got,
        vec![
            Some(r#"{"age":30}"#.to_string()),
            Some(r#"{"age":31}"#.to_string()),
        ]
    );
}

/// An outer join whose unmatched probe rows are whole documents from
/// differently-shredded files: the padding accumulator crosses the layout
/// change too.
#[rstest]
fn outer_joins_whole_documents_across_mixed_shredding(mut testing_planner: TestingPlanner) {
    let shredded = docs_batch(vec![r#"{"age":31}"#, r#"{"age":33}"#], true);
    let unshredded = docs_batch(vec![r#"{"age":32}"#, r#"{"age":30}"#], false);
    docs_table_files(&mut testing_planner, &[shredded, unshredded]);
    testing_planner.add_table(
        "wanted",
        &[(
            "age",
            Type::Int64,
            Arc::new(arrow_array::Int64Array::from(vec![30, 31])) as ArrayRef,
        )],
    );

    let rows = run(
        &mut testing_planner,
        "SELECT d, wanted.age AS w FROM docs \
         LEFT JOIN wanted ON CAST(d->'age' AS BIGINT) = wanted.age",
    );

    let mut got: Vec<(String, Option<i64>)> = rows
        .iter()
        .map(|r| (r["d"].as_str().unwrap().to_string(), r["w"].as_i64()))
        .collect();
    got.sort();
    assert_eq!(
        got,
        vec![
            (r#"{"age":30}"#.to_string(), Some(30)),
            (r#"{"age":31}"#.to_string(), Some(31)),
            (r#"{"age":32}"#.to_string(), None),
            (r#"{"age":33}"#.to_string(), None),
        ]
    );
}

/// Casting the whole document to VARCHAR over a shredded file renders it as
/// JSON, folding the typed leaves back into the text.
#[rstest]
fn casts_the_whole_shredded_document_to_string(mut testing_planner: TestingPlanner) {
    shredded_docs_table(
        &mut testing_planner,
        vec![r#"{"user":{"id":7}}"#],
        "user.id",
        &DataType::Int64,
    );

    let rows = run(
        &mut testing_planner,
        "SELECT CAST(d AS VARCHAR) AS j FROM docs",
    );

    let got: Vec<Option<&str>> = rows.iter().map(|r| r["j"].as_str()).collect();
    assert_eq!(got, vec![Some(r#"{"user":{"id":7}}"#)]);
}

/// A cast path under a plain LIMIT with OFFSET: DuckDB late-materializes the
/// scan into a narrow row-id pipeline plus a fetch of the surviving rows, and
/// the fetch must keep the pushed field extract rather than hand back the
/// whole document column.
#[rstest]
fn casts_a_path_under_late_materialization(mut testing_planner: TestingPlanner) {
    shredded_docs_table(
        &mut testing_planner,
        vec![
            r#"{"age":30}"#,
            r#"{"age":25}"#,
            r#"{"age":40}"#,
            r#"{"age":10}"#,
        ],
        "age",
        &DataType::Int64,
    );

    let rows = run(
        &mut testing_planner,
        "SELECT CAST(d->'age' AS BIGINT) AS a FROM docs LIMIT 2 OFFSET 1",
    );

    let got: Vec<Option<i64>> = rows.iter().map(|r| r["a"].as_i64()).collect();
    assert_eq!(got, vec![Some(25), Some(40)]);
}

/// A bare extraction under a late-materialized LIMIT renders the sub-variant
/// as JSON text, exactly as it does without the limit.
#[rstest]
fn bare_extraction_under_late_materialization(mut testing_planner: TestingPlanner) {
    shredded_docs_table(
        &mut testing_planner,
        vec![r#"{"age":30}"#, r#"{"age":25}"#, r#"{"age":40}"#],
        "age",
        &DataType::Int64,
    );

    let rows = run(
        &mut testing_planner,
        "SELECT d->'age' AS a FROM docs LIMIT 2 OFFSET 1",
    );

    let got: Vec<Option<&str>> = rows.iter().map(|r| r["a"].as_str()).collect();
    assert_eq!(got, vec![Some("25"), Some("40")]);
}

#[rstest]
fn struct_with_a_variant_field_is_rejected_when_planning(mut testing_planner: TestingPlanner) {
    docs_table(&mut testing_planner, vec![r#"{"age": 1}"#]);

    let error = testing_planner
        .plan("SELECT pair.doc FROM (SELECT {'doc': d} AS pair FROM docs)")
        .unwrap_err();

    assert!(
        error.to_string().contains("a struct with a VARIANT field"),
        "{error}"
    );
}
