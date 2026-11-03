//! Blackbox integration tests for the object-store backends, local filesystem,
//! S3 (MinIO) and GCS (`fake-gcs-server`), each a short Setup / Execute / Assert
//! against the public datastore + store API.
//!
//! Every behaviour is written once (in [`bodies`]) over a `&Backend` and run on
//! each backend by the [`backend_tests!`] macro. The local case always runs; the
//! S3/GCS cases bring up containers via [`harness`] and skip when Docker is
//! absent, so `cargo test` stays green offline.
//!
//! Covered: store contract (round-trip through `source` and `sink`, one-level
//! `list`, lost-update-free `update`), and the table lifecycle end to end — `CREATE TABLE`,
//! reopen, **appending a file** (out-of-band registration), **compaction**
//! (replacing files), and **reclaiming a dropped table** — all over object
//! storage.

mod common;

use datastore_pivotlake::test_support as harness;

use std::collections::HashMap;
use std::sync::Arc;

use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;

use catalog::datastore::DatastoreTransaction;
use common::{
    DispatchGuard, collect_i64s, commit_datastore_transaction, current_parquet,
    dispatch_with_buffers, strings_and_ints,
};
use datastore_pivotlake::PivotlakeDatastore;
use dispatch::Projection;
use harness::Backend;
use object_storage::ObjectPath;
use parquet_engine::table_input;
use planner::catalog::{Column, CreateTableRequest, DropTableRequest, SchemaQualifiedTableName};
use planner::types::Type;

// --- helpers (not tests) ---------------------------------------------------

/// (name Utf8, value Int64) — matches [`strings_and_ints`].
fn columns() -> Vec<Column> {
    vec![
        Column {
            name: "name".to_string(),
            col_type: Type::Utf8,
        },
        Column {
            name: "value".to_string(),
            col_type: Type::Int64,
        },
    ]
}

/// `CREATE TABLE <name> (cols) WITH (with_pre_existing_parquets = '<path>')`, `path`
/// store-relative.
fn adopting_request(name: &str, path: &str) -> CreateTableRequest {
    CreateTableRequest {
        datastore_name: None,
        schema_name: None,
        name: name.to_string(),
        columns: columns(),
        options: HashMap::from([("with_pre_existing_parquets".to_string(), path.to_string())]),
        if_not_exists: false,
    }
}

/// Snappy Parquet bytes for rows carrying the given `value`s (names are filler).
fn pq(values: &[i64]) -> Vec<u8> {
    let names: Vec<&str> = values.iter().map(|_| "x").collect();
    let batch = strings_and_ints(&names, values);
    let props = WriterProperties::builder()
        .set_compression(Compression::SNAPPY)
        .build();
    let mut buf = Vec::new();
    let mut writer = ArrowWriter::try_new(&mut buf, batch.schema(), Some(props)).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    buf
}

/// Put `files` (name → values) under `events/` and `CREATE TABLE events` over them.
fn create_events(
    d: &DispatchGuard,
    b: &Backend,
    files: &[(&str, &[i64])],
) -> Arc<PivotlakeDatastore> {
    for (name, values) in files {
        b.store
            .put(&ObjectPath::new(format!("events/{name}")), &pq(values))
            .unwrap();
    }
    let datastore = PivotlakeDatastore::open(&b.root, d).unwrap();
    let transaction = datastore.clone().begin_transaction();
    transaction
        .bind_create_table(adopting_request("events", "events"))
        .unwrap()
        .compile(d)
        .unwrap()
        .collect()
        .unwrap();
    commit_datastore_transaction(transaction).unwrap();
    datastore
}

/// Scan the table's `value` column, sorted.
fn scan(d: &DispatchGuard, datastore: &PivotlakeDatastore, name: &str) -> Vec<i64> {
    let parquet = current_parquet(datastore, name);
    let out = table_input(d, &parquet, Projection::all(2), false)
        .collect()
        .unwrap();
    let mut values = collect_i64s(&out, 1);
    values.sort();
    values
}

fn row_groups(datastore: &PivotlakeDatastore, name: &str) -> usize {
    current_parquet(datastore, name).row_groups().len()
}

/// The table's own storage directory, relative to the store root.
fn table_location(datastore: &PivotlakeDatastore, name: &str) -> ObjectPath {
    ObjectPath::new(
        datastore
            .table_handle(&SchemaQualifiedTableName::in_default_schema(name))
            .expect("table exists")
            .location(),
    )
}

/// `DROP TABLE <name>`, the way the server runs it: execute the staging
/// dataflow, then commit the transaction that unregisters the table.
fn drop_table(d: &DispatchGuard, datastore: &Arc<PivotlakeDatastore>, name: &str) {
    let transaction = datastore.clone().begin_transaction();
    transaction
        .bind_drop_table(DropTableRequest {
            datastore_name: None,
            schema_name: None,
            name: name.to_string(),
            if_exists: false,
        })
        .unwrap()
        .compile(d)
        .unwrap()
        .collect()
        .unwrap();
    commit_datastore_transaction(transaction).unwrap();
}

// --- behaviours (run on every backend) -------------------------------------

mod bodies {
    use super::*;

    /// `CREATE TABLE` over a Parquet file in the store; the rows scan back.
    pub fn create_and_scan(b: &Backend) {
        let d = dispatch_with_buffers(2, 32);
        b.store
            .put(&ObjectPath::new("events/p1.parquet"), &pq(&[1, 2, 3]))
            .unwrap();
        let datastore = PivotlakeDatastore::open(&b.root, &d).unwrap();

        let transaction = datastore.clone().begin_transaction();
        transaction
            .bind_create_table(adopting_request("events", "events"))
            .unwrap()
            .compile(&d)
            .unwrap()
            .collect()
            .unwrap();
        commit_datastore_transaction(transaction).unwrap();

        assert_eq!(scan(&d, &datastore, "events"), vec![1, 2, 3]);
    }

    /// An adopt path may be given as an absolute key, taken from the root of the
    /// store's own medium rather than from the database: a directory on the
    /// server's filesystem for a local database, a key from the bucket root for a
    /// remote one. The rows scan back the same either way.
    pub fn adopts_from_an_absolute_key(b: &Backend) {
        let d = dispatch_with_buffers(2, 32);
        b.store
            .put(&ObjectPath::new("outside/p1.parquet"), &pq(&[7, 8]))
            .unwrap();
        let absolute = b.store.absolute_key(&ObjectPath::new("outside")).unwrap();
        assert!(absolute.is_absolute(), "the store yields an absolute key");
        let datastore = PivotlakeDatastore::open(&b.root, &d).unwrap();

        let transaction = datastore.clone().begin_transaction();
        transaction
            .bind_create_table(adopting_request("events", absolute.as_str()))
            .unwrap()
            .compile(&d)
            .unwrap()
            .collect()
            .unwrap();
        commit_datastore_transaction(transaction).unwrap();

        assert_eq!(scan(&d, &datastore, "events"), vec![7, 8]);
    }

    /// The table and its data survive reopening the datastore, a server restart.
    pub fn survives_reopen(b: &Backend) {
        let d = dispatch_with_buffers(2, 32);
        drop(create_events(&d, b, &[("p1.parquet", &[1, 2, 3])]));

        let reopened = PivotlakeDatastore::open(&b.root, &d).unwrap();

        assert_eq!(scan(&d, &reopened, "events"), vec![1, 2, 3]);
    }

    /// Registering a new file (an out-of-band append) makes its rows visible to the
    /// next bind, on top of the existing ones.
    pub fn append_registers_new_file(b: &Backend) {
        let d = dispatch_with_buffers(2, 32);
        let datastore = create_events(&d, b, &[("p1.parquet", &[1, 2, 3])]);

        datastore
            .table_handle(&SchemaQualifiedTableName::in_default_schema("events"))
            .unwrap()
            .append_data_file(ObjectPath::new("p2.parquet"), &pq(&[4, 5, 6]), None)
            .unwrap();

        assert_eq!(scan(&d, &datastore, "events"), vec![1, 2, 3, 4, 5, 6]);
    }

    /// Compaction swaps the small files for one merged file in a single version:
    /// the rows are unchanged but now live in one row group.
    pub fn compaction_replaces_files(b: &Backend) {
        let d = dispatch_with_buffers(2, 32);
        let datastore = create_events(&d, b, &[("p1.parquet", &[1, 2, 3]), ("p2.parquet", &[4])]);
        let mut table = datastore
            .table_handle(&SchemaQualifiedTableName::in_default_schema("events"))
            .unwrap();
        // The adopted files are recorded by their store-absolute key; the merged
        // output is written into the table's own location.
        let inputs: Vec<ObjectPath> = table.file_refs().into_iter().map(|f| f.path).collect();
        let merged_path = ObjectPath::new("merged.parquet");
        let merged_bytes = pq(&[1, 2, 3, 4]);
        b.store
            .put(
                &ObjectPath::new(table.location()).resolve(&merged_path),
                &merged_bytes,
            )
            .unwrap();
        let merged = datastore_pivotlake::FileRef {
            path: merged_path,
            size: merged_bytes.len() as u64,
        };

        table
            .replace_data_files(&inputs, &[datastore_pivotlake::DeltaFileEntry::new(merged)])
            .unwrap();

        assert_eq!(scan(&d, &datastore, "events"), vec![1, 2, 3, 4]);
        assert_eq!(row_groups(&datastore, "events"), 1);
    }

    /// A stored object reads back through `source` — the read source the ring is
    /// handed (a presigned S3 URL, a GCS media URL, or a local path).
    pub fn source_reads_object_back(b: &Backend) {
        b.store
            .put(&ObjectPath::new("s/o.bin"), b"payload")
            .unwrap();

        let bytes = harness::read_via_source(b.store.as_ref(), &ObjectPath::new("s/o.bin"));

        assert_eq!(bytes, b"payload");
    }

    /// An object written through `sink` — the upload destination the ring is
    /// handed — lands under the key the store was asked for, and reads back
    /// through the ordinary `get`.
    pub fn sink_writes_object_back(b: &Backend) {
        b.store.create_dir(&ObjectPath::new("w")).unwrap();

        harness::write_via_sink(b.store.as_ref(), &ObjectPath::new("w/o.bin"), b"uploaded");

        assert_eq!(
            b.store.get(&ObjectPath::new("w/o.bin")).unwrap().unwrap(),
            b"uploaded"
        );
    }

    /// Concurrent `update`s of one object never lose a write: each writer bumps
    /// a counter in the object's body, and every bump survives. Locally the
    /// writers serialize on the update's file lock; remotely each replace is a
    /// compare-and-swap on the object's version, retried until it wins.
    pub fn update_never_loses_a_write(b: &Backend) {
        let key = ObjectPath::new("control/counter.txt");
        let (writers, bumps) = (4, 8);

        std::thread::scope(|scope| {
            for _ in 0..writers {
                scope.spawn(|| {
                    for _ in 0..bumps {
                        b.store
                            .update(&key, &mut |current| {
                                let count: u64 = current
                                    .map(|bytes| String::from_utf8(bytes).unwrap().parse().unwrap())
                                    .unwrap_or(0);
                                Some((count + 1).to_string().into_bytes())
                            })
                            .unwrap();
                    }
                });
            }
        });

        let bytes = b.store.get(&key).unwrap().unwrap();
        assert_eq!(
            String::from_utf8(bytes).unwrap(),
            (writers * bumps).to_string()
        );
    }

    /// A one-level listing returns direct objects and immediate child prefixes,
    /// without returning objects nested below those prefixes.
    pub fn list_returns_objects_and_child_prefixes(b: &Backend) {
        b.store.put(&ObjectPath::new("d/x.bin"), b"abc").unwrap();
        b.store.put(&ObjectPath::new("d/sub/y.bin"), b"z").unwrap();
        b.store
            .put(&ObjectPath::new("d/sub/deep/z.bin"), b"pq")
            .unwrap();

        let listing = b.store.list(&ObjectPath::new("d")).unwrap();
        let names: Vec<String> = listing
            .objects
            .into_iter()
            .map(|o| o.file.path.as_str().to_string())
            .collect();
        let prefixes: Vec<String> = listing
            .prefixes
            .into_iter()
            .map(|prefix| prefix.to_string())
            .collect();

        assert_eq!(names, vec!["x.bin".to_string()]);
        assert_eq!(prefixes, vec!["sub".to_string()]);
    }

    /// A name prefix narrows a listing to children whose own name starts with
    /// it, for direct objects and child prefixes alike.
    pub fn list_narrows_to_a_name_prefix(b: &Backend) {
        b.store.put(&ObjectPath::new("n/part-1.bin"), b"a").unwrap();
        b.store.put(&ObjectPath::new("n/other.bin"), b"b").unwrap();
        b.store
            .put(&ObjectPath::new("n/part-sub/x.bin"), b"c")
            .unwrap();
        b.store
            .put(&ObjectPath::new("n/misc-sub/y.bin"), b"d")
            .unwrap();

        let listing = b
            .store
            .list_with_name_prefix(&ObjectPath::new("n"), "part-")
            .unwrap();

        let names: Vec<String> = listing
            .objects
            .into_iter()
            .map(|o| o.file.path.as_str().to_string())
            .collect();
        let prefixes: Vec<String> = listing
            .prefixes
            .into_iter()
            .map(|prefix| prefix.to_string())
            .collect();
        assert_eq!(names, vec!["part-1.bin".to_string()]);
        assert_eq!(prefixes, vec!["part-sub".to_string()]);
    }

    /// Reclaiming a dropped table deletes everything under its location, an
    /// object nested below the one-level listings included, and forgets the
    /// tombstone only then.
    pub fn reclaim_deletes_a_dropped_tables_nested_objects(b: &Backend) {
        let d = dispatch_with_buffers(2, 32);
        let datastore = create_events(&d, b, &[("a.parquet", &[1])]);
        let location = table_location(&datastore, "events");
        drop_table(&d, &datastore, "events");
        b.store
            .put(
                &location.join("_delta_log/stray/00000000000000000001.json"),
                b"{}",
            )
            .unwrap();

        let reclaimed = datastore.reclaim_dropped_tables(u64::MAX).unwrap();

        assert_eq!(reclaimed, 1);
        let listing = b.store.list(&location).unwrap();
        assert!(
            listing.objects.is_empty() && listing.prefixes.is_empty(),
            "nothing is left under the table's location: {listing:?}"
        );
        assert_eq!(
            datastore.reclaim_dropped_tables(u64::MAX).unwrap(),
            0,
            "the tombstone is forgotten once the storage is gone"
        );
    }
}

// --- backend matrix --------------------------------------------------------

/// Emit `local` / `s3` / `gcs` tests for a [`bodies`] behaviour. S3/GCS skip
/// when Docker is absent; each gets its own bucket prefix for isolation.
macro_rules! backend_tests {
    ($name:ident) => {
        mod $name {
            use super::*;
            #[test]
            fn local() {
                let (_dir, b) = harness::local();
                bodies::$name(&b);
            }
            #[test]
            fn s3() {
                if let Some(b) = harness::s3(concat!(stringify!($name), "-s3")) {
                    bodies::$name(&b);
                }
            }
            #[test]
            fn gcs() {
                if let Some(b) = harness::gcs(concat!(stringify!($name), "-gcs")) {
                    bodies::$name(&b);
                }
            }
        }
    };
}

backend_tests!(create_and_scan);
backend_tests!(adopts_from_an_absolute_key);
backend_tests!(survives_reopen);
backend_tests!(append_registers_new_file);
backend_tests!(compaction_replaces_files);
backend_tests!(source_reads_object_back);
backend_tests!(sink_writes_object_back);
backend_tests!(update_never_loses_a_write);
backend_tests!(list_returns_objects_and_child_prefixes);
backend_tests!(list_narrows_to_a_name_prefix);
backend_tests!(reclaim_deletes_a_dropped_tables_nested_objects);
