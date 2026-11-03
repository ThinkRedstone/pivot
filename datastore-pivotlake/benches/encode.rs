//! Benchmarks for the Parquet **write** pipeline: rows in, finished files out.
//!
//! The mirror of `decode.rs`, which measures the read pipeline the same way.
//! Each case runs an INSERT, which is the whole thing: rows are split by
//! partition and accumulated into row groups, the shredder splits each column
//! chunk into leaves, each leaf is dictionary, delta or plain encoded, its
//! pages are compressed, and the assembler lays the file out and writes the
//! footer.
//!
//! What a change inside an encoder is worth is what it is worth to a write, so
//! that is what these measure. The shapes that lean on one encoder are named for
//! it rather than driven on their own.
//!
//! The files land on tmpfs, so a write is a memcpy and what is measured is the
//! pipeline rather than the disk under it.
//!
//! The shapes are the ones the write path behaves differently on:
//!
//! - `delta_sorted`, a clustered key whose differences pack into a few bits, and
//!   `delta_scattered`, keys spread over a wide range so every difference needs
//!   most of its width. Between them they are where `DELTA_BINARY_PACKED` does
//!   its most and least work per value.
//! - `dictionary`, a column of few distinct values, which is the other encoding
//!   an integer column can take.
//! - `strings`, the byte-view case: values are copied into the blocks a row
//!   group owns, and their lengths pack as deltas.
//! - `mixed`, the shape of a fact table (keys, money, dates and text), which is
//!   what a table load actually looks like.
//! - `mixed_shuffled_key`, the same fact table with its lead key scattered, and
//!   the sort-key cases over it: `sort_shuffled`, where every file's rows are
//!   reordered by the integer key as the file is cut, and `sort_presorted`,
//!   the same sort key over rows fed in order. `sort_shuffled` against
//!   `mixed_shuffled_key` is what the reorder costs on scattered input;
//!   `sort_presorted` against `mixed` is its floor on ordered input.
//! - `sort_string` and `sort_string_int`, the same rows sorted by the comment
//!   strings (alone, then with the integer key breaking ties). A string key
//!   has no radix form, so these are what the comparison-sort path costs.
//! - `variant`, a column of JSON documents, the one shape that carries a struct
//!   of leaves through accumulation and picks a shredding layout per file.
//!
//! # Running
//!
//! ```sh
//! cargo bench --bench encode
//! cargo bench --bench encode -- --save-baseline main   # then compare with --baseline main
//! ```
//!
//! Env: `PIVOT_BENCH_ROWS` (default 2M), `PIVOT_BENCH_WORKERS` (default all
//! cores), `PIVOT_BENCH_BUFFERS` (ring slots of 2 MiB, default 1024). The
//! variant case builds its documents through a JSON parser, which costs more to
//! set up than it does to encode, so it takes a fraction of the rows.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::builder::StringViewBuilder;
use arrow_array::{
    Array, ArrayRef, Date32Array, Decimal64Array, Int64Array, RecordBatch, StringArray, StructArray,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use criterion::{BatchSize, Criterion, Throughput, black_box};
use parquet_variant_compute::{VariantArray, json_to_variant};

use catalog::metastore::{DEFAULT_USER_NAME, Metastore, UserAuth};
use catalog::{DEFAULT_DATASTORE_NAME, Datastore, PivotCatalog};
use datastore_pivotlake::PivotlakeDatastore;
use dispatch::{Dispatch, RECORD_BATCH_SIZE, values_input};
use planner::catalog::{Column, CreateTableRequest, SchemaQualifiedTableName, TableReference};
use planner::types::{Type, physical_arrow_type};
use tempfile::TempDir;

/// Where the files land. tmpfs, so a write is a memcpy and the numbers are the
/// pipeline's rather than the disk's.
const OUTPUT_ROOT: &str = "/dev/shm";

/// The table each case inserts into.
const TABLE: &str = "written";

const DEFAULT_ROWS: usize = 2_000_000;
/// Rows the variant case takes, as a fraction of the rest. Its documents are
/// built by parsing JSON, which dominates setup at the full row count.
const VARIANT_ROW_SHARE: usize = 8;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}
fn total_rows() -> usize {
    env_usize("PIVOT_BENCH_ROWS", DEFAULT_ROWS)
}
fn worker_count() -> usize {
    env_usize(
        "PIVOT_BENCH_WORKERS",
        core_affinity::get_core_ids().map(|c| c.len()).unwrap_or(1),
    )
}
fn ring_buffers() -> usize {
    env_usize("PIVOT_BENCH_BUFFERS", 1024)
}

/// Deterministic splitmix64, so every run encodes the same bytes.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }
    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    #[inline]
    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Distinct strings that share a short lead and then diverge, with lengths
/// spread around `avg_len`. Long enough to leave the view and land in a data
/// block, which is the case the accumulator copies.
fn string_dict(n_distinct: usize, prefix: &str, avg_len: usize) -> Vec<String> {
    (0..n_distinct.max(1))
        .map(|i| {
            let mut value = format!("{prefix}{}", i.wrapping_mul(2_654_435_761) % 100_000_000);
            let target = (avg_len / 2).max(8) + (i % avg_len.max(1));
            while value.len() < target {
                value.push_str("-filler");
            }
            value.truncate(target.max(8));
            value
        })
        .collect()
}

fn keys(rng: &mut Rng, rows: usize, range: i64) -> ArrayRef {
    Arc::new(Int64Array::from(
        (0..rows)
            .map(|_| rng.below(range as usize) as i64)
            .collect::<Vec<_>>(),
    ))
}

/// A key that climbs with small gaps, which is what a delta encoding packs down
/// to a few bits per value.
fn sorted_keys(rows: usize, start: i64) -> ArrayRef {
    let mut value = start;
    Arc::new(Int64Array::from(
        (0..rows)
            .map(|i| {
                value += 1 + (i % 3) as i64;
                value
            })
            .collect::<Vec<_>>(),
    ))
}

fn text(rng: &mut Rng, rows: usize, dict: &[String]) -> ArrayRef {
    let mut builder = StringViewBuilder::with_capacity(rows);
    for _ in 0..rows {
        builder.append_value(&dict[rng.below(dict.len())]);
    }
    Arc::new(builder.finish())
}

fn money(rng: &mut Rng, rows: usize) -> ArrayRef {
    let values = Decimal64Array::from(
        (0..rows)
            .map(|_| rng.below(10_000_000) as i64)
            .collect::<Vec<_>>(),
    )
    .with_precision_and_scale(12, 2)
    .unwrap();
    Arc::new(values)
}

/// Ship dates spread over a few years, which the writer stores as the INT32 day
/// count behind a DATE annotation.
fn dates(rng: &mut Rng, rows: usize) -> ArrayRef {
    Arc::new(Date32Array::from(
        (0..rows)
            .map(|_| 19_000 + rng.below(2_000) as i32)
            .collect::<Vec<_>>(),
    ))
}

/// Split `rows` into the batch sizes the pipeline is fed in production, and
/// build each with `column`.
fn batches(
    schema: &SchemaRef,
    rows: usize,
    mut column: impl FnMut(usize, usize) -> Vec<ArrayRef>,
) -> Vec<RecordBatch> {
    let mut built = Vec::new();
    let mut done = 0;
    while done < rows {
        let n = RECORD_BATCH_SIZE.min(rows - done);
        built.push(RecordBatch::try_new(schema.clone(), column(done, n)).unwrap());
        done += n;
    }
    built
}

/// Keys that climb with small gaps, so their differences pack into a few bits
/// and the packing loop does the most work per byte written.
fn delta_sorted_batches(rows: usize) -> Vec<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("order_key", DataType::Int64, false),
        Field::new("line_key", DataType::Int64, false),
    ]));
    batches(&schema, rows, |done, n| {
        vec![
            sorted_keys(n, done as i64 * 4),
            sorted_keys(n, done as i64 * 9),
        ]
    })
}

/// Keys spread over a wide range, so every difference needs most of its width
/// and the same encoder writes its widest miniblocks.
fn delta_scattered_batches(rows: usize) -> Vec<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("part_key", DataType::Int64, false),
        Field::new("supplier_key", DataType::Int64, false),
    ]));
    let mut rng = Rng::new(1);
    batches(&schema, rows, |_, n| {
        vec![keys(&mut rng, n, 20_000_000), keys(&mut rng, n, 10_000_000)]
    })
}

/// Few distinct values, which is what makes a column worth a dictionary: the
/// encoder builds one and the column becomes RLE indices into it.
fn dictionary_batches(rows: usize) -> Vec<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("quantity", DataType::Int64, false),
        Field::new("line_number", DataType::Int64, false),
    ]));
    let mut rng = Rng::new(2);
    batches(&schema, rows, |_, n| {
        vec![keys(&mut rng, n, 50), keys(&mut rng, n, 7)]
    })
}

fn string_batches(rows: usize) -> Vec<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("comment", DataType::Utf8View, false),
        Field::new("status", DataType::Utf8View, false),
    ]));
    // A comment is nearly unique and too long to inline; a status is one of a
    // handful of short values, which is what the dictionary encoder is for.
    let comments = string_dict((rows / 4).max(1), "a line of commentary ", 44);
    let statuses = string_dict(8, "STATUS", 10);
    let mut rng = Rng::new(2);
    batches(&schema, rows, |_, n| {
        vec![text(&mut rng, n, &comments), text(&mut rng, n, &statuses)]
    })
}

fn mixed_batches(rows: usize) -> Vec<RecordBatch> {
    mixed_batches_keyed(rows, true)
}

/// The mixed shape with its lead key scattered instead of climbing: on its own
/// a control with no sort key, and under `sort_by` the input every file must
/// reorder.
fn mixed_shuffled_key_batches(rows: usize) -> Vec<RecordBatch> {
    mixed_batches_keyed(rows, false)
}

fn mixed_batches_keyed(rows: usize, key_in_order: bool) -> Vec<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("order_key", DataType::Int64, false),
        Field::new("part_key", DataType::Int64, false),
        Field::new("price", DataType::Decimal64(12, 2), false),
        Field::new("ship_date", DataType::Date32, false),
        Field::new("comment", DataType::Utf8View, false),
    ]));
    let comments = string_dict((rows / 4).max(1), "a line of commentary ", 44);
    let mut rng = Rng::new(3);
    batches(&schema, rows, |done, n| {
        let order_key = if key_in_order {
            sorted_keys(n, done as i64 * 4)
        } else {
            keys(&mut rng, n, 20_000_000)
        };
        vec![
            order_key,
            keys(&mut rng, n, 20_000_000),
            money(&mut rng, n),
            dates(&mut rng, n),
            text(&mut rng, n, &comments),
        ]
    })
}

/// Documents shaped like an event stream: a few fields every row carries, so
/// inference finds paths worth shredding, and one that only some rows do.
fn variant_batches(rows: usize) -> Vec<RecordBatch> {
    let mut rng = Rng::new(4);
    let mut built = Vec::new();
    let mut done = 0;
    while done < rows {
        let n = RECORD_BATCH_SIZE.min(rows - done);
        let documents: Vec<String> = (0..n)
            .map(|i| {
                let id = done + i;
                let session = rng.below(1_000_000);
                if id % 5 == 0 {
                    format!(
                        r#"{{"user":{{"id":{id},"name":"user {id}"}},"session":{session},"retry":{}}}"#,
                        id % 7
                    )
                } else {
                    format!(r#"{{"user":{{"id":{id},"name":"user {id}"}},"session":{session}}}"#)
                }
            })
            .collect();
        let json: ArrayRef = Arc::new(StringArray::from(documents));
        let column = as_declared_variant(json_to_variant(&json).unwrap());
        let schema = Arc::new(Schema::new(vec![
            Field::new("attrs", column.data_type().clone(), true)
                .with_metadata(parquet_engine::variant_extension_metadata()),
        ]));
        built.push(RecordBatch::try_new(schema, vec![column]).unwrap());
        done += n;
    }
    built
}

/// `variants` as the column type a VARIANT declares. `json_to_variant` marks the
/// `value` child non-nullable when every document has one, while the declared
/// column always allows a null there, and an INSERT checks the two agree.
fn as_declared_variant(variants: VariantArray) -> ArrayRef {
    let DataType::Struct(declared) = physical_arrow_type(&Type::Variant) else {
        unreachable!("a variant is declared as a struct")
    };
    let array = variants.into_inner();
    let (_, columns, nulls) = array.into_parts();
    Arc::new(StructArray::new(declared, columns, nulls))
}

/// One INSERT of `input`: the write pipeline end to end, ending where the files
/// land. The transaction is rolled back rather than committed, since the
/// benchmark measures the writing and not the publishing.
fn insert(dispatch: &Dispatch, catalog: &PivotCatalog, input: Vec<RecordBatch>) {
    let transaction = catalog.begin_transaction();
    let table = transaction
        .bind_table(&TableReference {
            datastore: DEFAULT_DATASTORE_NAME.to_string(),
            schema: planner::DEFAULT_SCHEMA_NAME.to_string(),
            table: TABLE.to_string(),
        })
        .expect("table lookup succeeds")
        .expect("the table was created");
    let rows = values_input(dispatch.dispatcher(), input).record_batches();
    table
        .compile_insert(rows, dispatch.dispatcher())
        .unwrap()
        .collect()
        .unwrap();
    transaction.rollback();
}

/// The `mixed` fact-table columns, shared by the plain and sort-key cases.
fn mixed_columns() -> Vec<Column> {
    vec![
        column("order_key", Type::Int64),
        column("part_key", Type::Int64),
        column(
            "price",
            Type::Decimal {
                precision: 12,
                scale: 2,
            },
        ),
        column("ship_date", Type::Date),
        column("comment", Type::Utf8),
    ]
}

/// One bench case: a table shape, the batches written into it, and the sort
/// key the write orders by (empty for an unsorted table).
struct EncodeCase {
    name: &'static str,
    rows: usize,
    columns: Vec<Column>,
    sort_by: &'static [&'static str],
    input: Vec<RecordBatch>,
}

fn encode_case(
    name: &'static str,
    rows: usize,
    columns: Vec<Column>,
    sort_by: &'static [&'static str],
    input: Vec<RecordBatch>,
) -> EncodeCase {
    EncodeCase {
        name,
        rows,
        columns,
        sort_by,
        input,
    }
}

fn bench_encode(c: &mut Criterion, dispatch: &Dispatch, rows: usize) {
    let cases: Vec<EncodeCase> = vec![
        encode_case(
            "delta_sorted",
            rows,
            columns(&["order_key", "line_key"], Type::Int64),
            &[],
            delta_sorted_batches(rows),
        ),
        encode_case(
            "delta_scattered",
            rows,
            columns(&["part_key", "supplier_key"], Type::Int64),
            &[],
            delta_scattered_batches(rows),
        ),
        encode_case(
            "dictionary",
            rows,
            columns(&["quantity", "line_number"], Type::Int64),
            &[],
            dictionary_batches(rows),
        ),
        encode_case(
            "strings",
            rows,
            columns(&["comment", "status"], Type::Utf8),
            &[],
            string_batches(rows),
        ),
        encode_case("mixed", rows, mixed_columns(), &[], mixed_batches(rows)),
        encode_case(
            "mixed_shuffled_key",
            rows,
            mixed_columns(),
            &[],
            mixed_shuffled_key_batches(rows),
        ),
        encode_case(
            "sort_shuffled",
            rows,
            mixed_columns(),
            &["order_key"],
            mixed_shuffled_key_batches(rows),
        ),
        encode_case(
            "sort_presorted",
            rows,
            mixed_columns(),
            &["order_key"],
            mixed_batches(rows),
        ),
        encode_case(
            "sort_string",
            rows,
            mixed_columns(),
            &["comment"],
            mixed_shuffled_key_batches(rows),
        ),
        encode_case(
            "sort_string_int",
            rows,
            mixed_columns(),
            &["comment", "order_key"],
            mixed_shuffled_key_batches(rows),
        ),
        encode_case(
            "variant",
            rows / VARIANT_ROW_SHARE,
            columns(&["attrs"], Type::Variant),
            &[],
            variant_batches(rows / VARIANT_ROW_SHARE),
        ),
    ];

    let mut group = c.benchmark_group("encode");
    for case in &cases {
        // Each case writes into its own table on tmpfs, so the measurement is
        // the pipeline rather than the disk under it. The rows are rolled back
        // instead of committed, and the files they left behind are cleared
        // between iterations, so the directory holds one run's output at a time.
        let dir = TempDir::new_in(OUTPUT_ROOT).expect("a writable tmpfs directory");
        let (catalog, table_dir) =
            table_over(dispatch, dir.path(), case.columns.clone(), case.sort_by);
        group.throughput(Throughput::Elements(case.rows as u64));
        group.bench_function(case.name, |b| {
            b.iter_batched(
                || {
                    clear_files(&table_dir);
                    // The batches are Arc-backed, so a clone hands the pipeline
                    // its own handles without copying any values.
                    case.input.clone()
                },
                |batches| insert(dispatch, &catalog, black_box(batches)),
                BatchSize::PerIteration,
            );
        });
    }
    group.finish();
}

/// One column of `name` and `col_type`.
fn column(name: &str, col_type: Type) -> Column {
    Column {
        name: name.to_string(),
        col_type,
    }
}

/// Columns that share a type.
fn columns(names: &[&str], col_type: Type) -> Vec<Column> {
    names
        .iter()
        .map(|name| column(name, col_type.clone()))
        .collect()
}

/// A catalog holding one table of `columns` in a database rooted at `dir`,
/// sorted by `sort_by` where one is given, and the directory that table keeps
/// its own storage in (where the files an insert writes land).
/// A metastore serving no datastores and only the built-in trusted user: the
/// catalog here gets its datastore handed in directly.
fn trust_metastore() -> Arc<dyn Metastore> {
    #[derive(Debug)]
    struct TrustMetastore;

    impl Metastore for TrustMetastore {
        fn open_datastores(
            &self,
            _dispatcher: &dispatch::DataFlowDispatcher,
        ) -> catalog::metastore::Result<HashMap<String, Arc<dyn Datastore>>> {
            Ok(HashMap::new())
        }

        fn default_datastore_name(&self) -> &str {
            DEFAULT_DATASTORE_NAME
        }

        fn user_auth(&self, username: &str) -> Option<UserAuth> {
            (username == DEFAULT_USER_NAME).then_some(UserAuth::Trust)
        }
    }

    Arc::new(TrustMetastore)
}

fn table_over(
    dispatch: &Dispatch,
    dir: &Path,
    columns: Vec<Column>,
    sort_by: &[&str],
) -> (PivotCatalog, PathBuf) {
    let datastore =
        PivotlakeDatastore::open(&dir.to_string_lossy(), dispatch.dispatcher()).unwrap();
    let catalog = PivotCatalog::new(
        HashMap::from([(
            DEFAULT_DATASTORE_NAME.to_string(),
            datastore.clone() as Arc<dyn Datastore>,
        )]),
        DEFAULT_DATASTORE_NAME.to_string(),
        trust_metastore(),
    )
    .unwrap();

    let mut options = HashMap::new();
    if !sort_by.is_empty() {
        options.insert("sort_by".to_string(), sort_by.join(", "));
    }
    let creation = catalog.begin_transaction();
    creation
        .bind_create_table(CreateTableRequest {
            datastore_name: None,
            schema_name: None,
            name: TABLE.to_string(),
            columns,
            options,
            if_not_exists: false,
        })
        .unwrap()
        .compile(dispatch.dispatcher())
        .unwrap()
        .collect()
        .unwrap();
    // The table is only visible to the transactions that insert into it once its
    // creation is committed, and committing is async.
    tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(creation.commit())
        .unwrap();

    // The table's own directory, named for its identity: what an insert writes
    // into, and so what each iteration clears.
    let table_dir = dir.join(
        datastore
            .table_handle(&SchemaQualifiedTableName::in_default_schema(TABLE))
            .expect("the table was created")
            .location(),
    );
    (catalog, table_dir)
}

/// Remove the files a previous iteration wrote, leaving the table's log alone.
fn clear_files(dir: &Path) {
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let path = entry.path();
        if path
            .extension()
            .is_some_and(|extension| extension == "parquet")
        {
            std::fs::remove_file(path).unwrap();
        }
    }
}

fn main() {
    let workers = worker_count();
    let buffers = ring_buffers();
    let rows = total_rows();
    eprintln!("catalog encode benches: {workers} workers, {buffers} buffers, {rows} rows");

    let dispatch = Dispatch::spin_up(workers, buffers, None);
    let mut criterion = Criterion::default().configure_from_args();
    bench_encode(&mut criterion, &dispatch, rows);
    criterion.final_summary();
    dispatch.exit();
}
