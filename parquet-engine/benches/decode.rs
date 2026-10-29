//! Microbenchmarks for the Parquet **decompression + decoding** path — the work
//! that dominates "decode-bound" queries. Profiling the real analytical
//! queries showed that for many of them the dispatch *operator* is a sliver and
//! the wall-time is spent in the scan: snappy-decompressing pages and decoding
//! them into Arrow. For example, a `LIKE '%x%'` scan over
//! the long URL column spent ~73% in `snap::decompress`, and an `ORDER BY
//! <string/time> LIMIT 10` spent 33-46%, with `RleDecoder::read`, the
//! `bytes_view` plain/dict decoders, and `PrimitiveDict` making up most of the
//! rest. None of that is exercised by the `dispatch` operator benches (which run
//! on pre-built in-memory batches), so this bench covers the other half: the
//! column reader.
//!
//! It generates its own SNAPPY + dictionary Parquet file in a tempdir (no
//! dependency on a 14 GB dataset), shaped to the measured statistics of
//! the real columns so the encoding the writer picks — and therefore the decode
//! path the reader takes matches. The `url` column is long (~88 B avg),
//! high-cardinality strings, with too many distinct values per row group to
//! dictionary-encode, so it lands in PLAIN byte-array pages (the long-URL scan
//! shape). `search_phrase` is ~87% empty and otherwise ~31 B, with low distinct
//! per row group, so DICTIONARY + RLE (the search-phrase scan shape).
//! `event_time` is `i64` (the timestamp columns). The boolean columns cover
//! the three ways a flag column arrives: PLAIN bits, RLE runs (what other
//! writers emit for a column with long runs), and PLAIN bits with nulls.
//!
//! Then it scans one column at a time with `table_input(..).collect()`, driving
//! fetch → snappy-decompress → decode → Arrow materialize. Every scenario is a
//! whole scan: what a change inside a decoder is worth is what it is worth to
//! one, which is the number this reports. The delta scenarios are here for the
//! same reason, at both packed widths, since that is where the decode loop does
//! the most work per byte read.
//!
//! # Running
//!
//! ```sh
//! cargo bench --bench decode
//! cargo bench --bench decode -- "decode/url" --profile-time 20   # attach perf
//! ```
//!
//! Env: `PIVOT_BENCH_ROWS` (default 4M), `PIVOT_BENCH_WORKERS` (default all
//! cores), `PIVOT_BENCH_BUFFERS` (ring/file-cache slots of 2 MiB, default 1024).
//! Iteration 1 is a cold read; the rest decode from the warm compressed cache (so the
//! steady-state samples are decompress+decode, not disk IO). Under `perf`, raise
//! the locked-memory limit (`ulimit -l unlimited`) for io_uring setup.

use std::fs::File;
use std::sync::Arc;

use arrow_array::builder::StringViewBuilder;
use arrow_array::{ArrayRef, BooleanArray, Int64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use criterion::{Criterion, Throughput, black_box};
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, Encoding};
use parquet::file::properties::WriterProperties;
use tempfile::TempDir;

use dispatch::{Dispatch, Projection};
use parquet_engine::{ParquetTable, table_input};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

const DEFAULT_ROWS: usize = 4_000_000;
/// Rows per row group — many row groups so the scan fans across all workers,
/// like a real multi-row-group file.
const ROW_GROUP_ROWS: usize = 256 * 1024;

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

// ---------------------------------------------------------------------------
// Deterministic PRNG (splitmix64)
// ---------------------------------------------------------------------------

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
    #[inline]
    fn chance(&mut self, p: f64) -> bool {
        self.below(1_000_000) < (p * 1_000_000.0) as usize
    }
}

/// Dictionary of distinct strings: a short shared lead then a per-entry token so
/// they diverge early, with lengths varying around `avg_len` — the realistic
/// shape (see the dispatch operator bench for why this matters for the decoder).
fn string_dict(n_distinct: usize, prefix: &str, avg_len: usize) -> Vec<String> {
    let lead: String = prefix.bytes().take(7).map(|b| b as char).collect();
    (0..n_distinct.max(1))
        .map(|i| {
            let token = i.wrapping_mul(2_654_435_761) % 100_000_000;
            let mut s = format!("{lead}{token}");
            let target = (avg_len / 2).max(8) + (i % avg_len.max(1));
            let mut k = i as u64 + 1;
            while s.len() < target {
                k = k.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                s.push('/');
                s.push_str(&(k % 100_000).to_string());
            }
            s.truncate(target.max(8));
            s
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Synthetic Parquet generation
// ---------------------------------------------------------------------------

fn string_col(rng: &mut Rng, n: usize, dict: &[String]) -> ArrayRef {
    let mut b = StringViewBuilder::with_capacity(n);
    for _ in 0..n {
        b.append_value(&dict[rng.below(dict.len())]);
    }
    Arc::new(b.finish())
}

fn sparse_string_col(rng: &mut Rng, n: usize, dict: &[String], empty_fraction: f64) -> ArrayRef {
    let mut b = StringViewBuilder::with_capacity(n);
    for _ in 0..n {
        if rng.chance(empty_fraction) {
            b.append_value("");
        } else {
            b.append_value(&dict[rng.below(dict.len())]);
        }
    }
    Arc::new(b.finish())
}

/// An ascending column starting at `start`, stepping by a small random gap:
/// the shape of a key a table is clustered on.
fn sorted_i64_col(rng: &mut Rng, n: usize, start: i64) -> ArrayRef {
    let mut value = start;
    let values: Vec<i64> = (0..n)
        .map(|_| {
            value += 1 + rng.below(7) as i64;
            value
        })
        .collect();
    Arc::new(Int64Array::from(values)) as ArrayRef
}

fn i64_col(rng: &mut Rng, n: usize, base: i64, span: usize) -> ArrayRef {
    let v: Int64Array = (0..n).map(|_| base + rng.below(span) as i64).collect();
    Arc::new(v)
}

/// A flag that is true for about half the rows, with no run structure.
fn bool_col(rng: &mut Rng, n: usize) -> ArrayRef {
    Arc::new(BooleanArray::from_iter(
        (0..n).map(|_| Some(rng.chance(0.5))),
    ))
}

/// A flag that flips about once every `run` rows: the shape a status column
/// has in a table clustered on time, and what an RLE page stores as runs.
fn bool_runs_col(rng: &mut Rng, n: usize, run: usize) -> ArrayRef {
    let mut value = false;
    Arc::new(BooleanArray::from_iter((0..n).map(|_| {
        if rng.below(run) == 0 {
            value = !value;
        }
        Some(value)
    })))
}

/// A half-true flag with a null in about one row in ten.
fn nullable_bool_col(rng: &mut Rng, n: usize) -> ArrayRef {
    Arc::new(BooleanArray::from_iter(
        (0..n).map(|_| (!rng.chance(0.1)).then(|| rng.chance(0.5))),
    ))
}

/// Write a SNAPPY + dictionary Parquet file (one column per decode shape) into
/// `dir`, `rows` total across `ROW_GROUP_ROWS`-sized row groups. The schema
/// column order: 0=url, 1=search_phrase, 2=event_time, 3=scattered_key,
/// 4=sorted_key, 5=flag, 6=flag_runs, 7=flag_nullable, 8=flag_packed.
fn write_parquet(dir: &TempDir, rows: usize) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("url", DataType::Utf8View, false),
        Field::new("search_phrase", DataType::Utf8View, false),
        Field::new("event_time", DataType::Int64, false),
        Field::new("scattered_key", DataType::Int64, false),
        Field::new("sorted_key", DataType::Int64, false),
        Field::new("flag", DataType::Boolean, false),
        Field::new("flag_runs", DataType::Boolean, false),
        Field::new("flag_nullable", DataType::Boolean, true),
        Field::new("flag_packed", DataType::Boolean, false),
    ]));
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .set_dictionary_enabled(true) // writer still falls back to PLAIN when a
        // row group has too many distinct values (the URL case) — exactly what
        // the real file does.
        .set_max_row_group_row_count(Some(ROW_GROUP_ROWS))
        // The two key columns carry the delta encoding. Dictionary encoding
        // wins the race for a column unless it is turned off per column, and a
        // key column with millions of distinct values is exactly where a
        // dictionary stops paying, which is why these are written delta packed.
        .set_column_dictionary_enabled("scattered_key".into(), false)
        .set_column_encoding("scattered_key".into(), Encoding::DELTA_BINARY_PACKED)
        .set_column_dictionary_enabled("sorted_key".into(), false)
        .set_column_encoding("sorted_key".into(), Encoding::DELTA_BINARY_PACKED)
        // A boolean column is PLAIN bits unless told otherwise; the run-shaped
        // one is stored as runs, and so is a run-less one, which the RLE
        // encoder then stores almost entirely as bit-packed groups.
        .set_column_dictionary_enabled("flag_runs".into(), false)
        .set_column_encoding("flag_runs".into(), Encoding::RLE)
        .set_column_dictionary_enabled("flag_packed".into(), false)
        .set_column_encoding("flag_packed".into(), Encoding::RLE)
        .build();
    let mut writer = ArrowWriter::try_new(
        File::create(dir.path().join("data.parquet")).unwrap(),
        schema.clone(),
        Some(props),
    )
    .unwrap();

    // Real-data stats: URL ≈ rows×18/100 distinct
    // (~5.5 rows/group), ~88 B; search_phrase ≈ rows×6/100 distinct, ~87% empty,
    // ~31 B non-empty.
    let url_dict = string_dict((rows * 18 / 100).max(1), "http://example.com/path", 88);
    let phrase_dict = string_dict((rows * 6 / 100).max(1), "search query ", 31);
    let mut rng = Rng::new(1);

    let mut written = 0;
    while written < rows {
        let n = ROW_GROUP_ROWS.min(rows - written);
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                string_col(&mut rng, n, &url_dict),
                sparse_string_col(&mut rng, n, &phrase_dict, 0.868),
                i64_col(&mut rng, n, 1_700_000_000_000, 86_400_000),
                // A foreign key: distinct values spread over a wide range, so
                // every delta needs its full width (the `l_partkey` shape).
                i64_col(&mut rng, n, 1, 20_000_000),
                // A clustered key: ascending with small gaps, so the deltas
                // pack into a handful of bits (the `l_orderkey` shape).
                sorted_i64_col(&mut rng, n, written as i64 * 4),
                bool_col(&mut rng, n),
                bool_runs_col(&mut rng, n, 64),
                nullable_bool_col(&mut rng, n),
                bool_col(&mut rng, n),
            ],
        )
        .unwrap();
        writer.write(&batch).unwrap();
        written += n;
    }
    writer.close().unwrap();
}

// ---------------------------------------------------------------------------
// DELTA_BINARY_PACKED page decoding, on its own
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

fn bench_decode(c: &mut Criterion, dispatch: &Dispatch, table: &Arc<ParquetTable>, rows: usize) {
    let d = dispatch.dispatcher();
    let mut scenario = |name: &str, proj: Projection| {
        let mut g = c.benchmark_group("decode");
        g.throughput(Throughput::Elements(rows as u64));
        g.bench_function(name, |b| {
            b.iter(|| {
                let out = table_input(d, table, proj.clone(), false)
                    .collect()
                    .unwrap();
                black_box(out);
            });
        });
        g.finish();
    };

    // Long high-cardinality strings → PLAIN byte-array pages + snappy (the URL /
    // `LIKE` scan: decompress-dominated).
    scenario("decode/url", Projection::columns([0]));
    // Mostly-empty, low-distinct strings → DICTIONARY + RLE + snappy (the
    // search_phrase / order-by scan).
    scenario("decode/search_phrase", Projection::columns([1]));
    // `i64` timestamps (the event_time column).
    scenario("decode/event_time", Projection::columns([2]));
    // DELTA_BINARY_PACKED at full width: scattered keys, so every delta uses
    // all the bits its miniblock allows.
    scenario("decode/delta_scattered", Projection::columns([3]));
    // DELTA_BINARY_PACKED at a narrow width: a clustered key whose deltas fit
    // in a few bits, so the unpacking loop does the most work per byte read.
    scenario("decode/delta_sorted", Projection::columns([4]));
    // Booleans: PLAIN bits, RLE runs, PLAIN bits with nulls, and RLE over
    // run-less values (bit-packed groups behind run headers). Next to
    // `event_time`, which is the same number of values as eight-byte
    // integers, they show what a value costs to decode in each form.
    scenario("decode/flag", Projection::columns([5]));
    scenario("decode/flag_runs", Projection::columns([6]));
    scenario("decode/flag_nullable", Projection::columns([7]));
    scenario("decode/flag_packed", Projection::columns([8]));
    // Every non-boolean column at once — a fuller scan.
    scenario("decode/all", Projection::columns([0, 1, 2, 3, 4]));
}

fn main() {
    let workers = worker_count();
    let buffers = ring_buffers();
    let rows = total_rows();
    eprintln!("catalog decode benches: {workers} workers, {buffers} buffers, {rows} rows");

    let dispatch = Dispatch::spin_up(workers, buffers, None);

    // Generate the synthetic parquet once, then load the table (the metadata
    // fetch runs on the coordinator, like the test helper).
    let dir = TempDir::new().unwrap();
    write_parquet(&dir, rows);
    let table = Arc::new(
        ParquetTable::from_files(
            dispatch.dispatcher(),
            &[dir.path().join("data.parquet")],
            &[],
        )
        .expect("ParquetTable::from_files failed"),
    );

    let mut c = Criterion::default().configure_from_args();
    bench_decode(&mut c, &dispatch, &table, rows);
    c.final_summary();

    dispatch.exit();
}
