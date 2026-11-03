//! Schemas inside a datastore: a `CREATE SCHEMA` is durable, a table created in
//! a schema resolves only under it, and a manifest that omits its optional
//! fields still loads.

mod common;
use common::*;

use std::collections::HashMap;
use std::sync::Arc;

use tempfile::TempDir;

use catalog::datastore::DatastoreTransaction;
use datastore_pivotlake::PivotlakeDatastore;
use datastore_pivotlake::test_support as harness;
use planner::DEFAULT_DATASTORE_NAME;
use planner::catalog::{
    Column, CreateSchemaRequest, CreateTableRequest, Result as CatalogResult,
    SchemaQualifiedTableName,
};
use planner::types::Type;

fn columns() -> Vec<Column> {
    vec![Column {
        name: "value".to_string(),
        col_type: Type::Int64,
    }]
}

fn create_table_request(schema: Option<&str>, name: &str) -> CreateTableRequest {
    CreateTableRequest {
        datastore_name: None,
        schema_name: schema.map(str::to_string),
        name: name.to_string(),
        columns: columns(),
        options: HashMap::new(),
        if_not_exists: false,
    }
}

/// Mirror the server's `CREATE TABLE`: run the footer-fetch dataflow, then
/// commit the transaction that persists and publishes the staged table.
fn create_table(
    dispatch: &DispatchGuard,
    datastore: &Arc<PivotlakeDatastore>,
    request: CreateTableRequest,
) -> CatalogResult<()> {
    let transaction = datastore.clone().begin_transaction();
    transaction
        .bind_create_table(request)?
        .compile(dispatch)?
        .collect()
        .map(|_| ())
        .map_err(|e| planner::catalog::Error::Other(Box::new(e)))?;
    commit_datastore_transaction(transaction)
}

/// Mirror the server's `CREATE SCHEMA`: run the staging dataflow, then commit
/// the transaction that persists and publishes the staged schema.
fn create_schema(
    dispatch: &DispatchGuard,
    datastore: &Arc<PivotlakeDatastore>,
    name: &str,
) -> CatalogResult<()> {
    stage_and_commit_schema(dispatch, datastore, name, false)
}

/// [`create_schema`] for `CREATE SCHEMA IF NOT EXISTS`, which takes a different
/// branch both where the statement resolves and where the commit persists it.
fn create_schema_if_not_exists(
    dispatch: &DispatchGuard,
    datastore: &Arc<PivotlakeDatastore>,
    name: &str,
) -> CatalogResult<()> {
    stage_and_commit_schema(dispatch, datastore, name, true)
}

fn stage_and_commit_schema(
    dispatch: &DispatchGuard,
    datastore: &Arc<PivotlakeDatastore>,
    name: &str,
    if_not_exists: bool,
) -> CatalogResult<()> {
    let transaction = datastore.clone().begin_transaction();
    transaction
        .bind_create_schema(CreateSchemaRequest {
            datastore_name: None,
            name: name.to_string(),
            if_not_exists,
        })?
        .compile(dispatch)?
        .collect()
        .map(|_| ())
        .map_err(|e| planner::catalog::Error::Other(Box::new(e)))?;
    commit_datastore_transaction(transaction)
}

/// The schemas the manifest on disk lists under `name`, so a test can tell a
/// schema that was created once from one recorded twice.
fn count_manifest_schemas(database_root: &std::path::Path, name: &str) -> usize {
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(database_root.join("_pivot_manifest.json")).unwrap())
            .unwrap();
    manifest["schemas"]
        .as_array()
        .expect("the manifest lists its schemas")
        .iter()
        .filter(|schema| schema["name"] == name)
        .count()
}

#[test]
fn a_new_datastore_defines_only_the_default_schema() {
    let dispatch = dispatch(1);
    let db = TempDir::new().unwrap();

    let datastore = PivotlakeDatastore::open(db.path().to_str().unwrap(), &dispatch).unwrap();

    assert!(datastore.contains_schema("main"));
    assert!(!datastore.contains_schema("analytics"));
}

#[test]
fn created_schema_survives_a_reopen() {
    let dispatch = dispatch(1);
    let db = TempDir::new().unwrap();
    let datastore = PivotlakeDatastore::open(db.path().to_str().unwrap(), &dispatch).unwrap();

    create_schema(&dispatch, &datastore, "analytics").unwrap();

    drop(datastore);
    let reopened = PivotlakeDatastore::open(db.path().to_str().unwrap(), &dispatch).unwrap();
    assert!(reopened.contains_schema("analytics"));
}

#[test]
fn creating_an_existing_schema_is_rejected() {
    let dispatch = dispatch(1);
    let db = TempDir::new().unwrap();
    let datastore = PivotlakeDatastore::open(db.path().to_str().unwrap(), &dispatch).unwrap();
    create_schema(&dispatch, &datastore, "analytics").unwrap();

    let error = create_schema(&dispatch, &datastore, "analytics").unwrap_err();

    assert!(
        error
            .to_string()
            .contains("schema `analytics` already exists")
    );
}

/// `IF NOT EXISTS` turns the rejection above into a success, and it has to hold
/// at both the point the statement resolves and the point the commit writes the
/// manifest. Landing a second entry there would leave the database listing one
/// schema twice, so the result is checked on disk and not just through the
/// in-memory set.
#[test]
fn creating_an_existing_schema_if_not_exists_succeeds_once() {
    let dispatch = dispatch(1);
    let db = TempDir::new().unwrap();
    let datastore = PivotlakeDatastore::open(db.path().to_str().unwrap(), &dispatch).unwrap();
    create_schema(&dispatch, &datastore, "analytics").unwrap();

    create_schema_if_not_exists(&dispatch, &datastore, "analytics").unwrap();

    assert!(datastore.contains_schema("analytics"));
    assert_eq!(count_manifest_schemas(db.path(), "analytics"), 1);
}

/// A schema is staged on its transaction and only becomes real at commit, so a
/// transaction that rolls back must leave nothing behind: not in the live set,
/// and not in the manifest a later open reads.
#[test]
fn rolling_back_a_create_schema_discards_the_staged_schema() {
    let dispatch = dispatch(1);
    let db = TempDir::new().unwrap();
    let datastore = PivotlakeDatastore::open(db.path().to_str().unwrap(), &dispatch).unwrap();
    let transaction = datastore.clone().begin_transaction();
    transaction
        .bind_create_schema(CreateSchemaRequest {
            datastore_name: None,
            name: "analytics".to_string(),
            if_not_exists: false,
        })
        .unwrap()
        .compile(&dispatch)
        .unwrap()
        .collect()
        .unwrap();

    transaction.rollback();

    commit_datastore_transaction(transaction).unwrap();
    assert!(!datastore.contains_schema("analytics"));
    drop(datastore);
    let reopened = PivotlakeDatastore::open(db.path().to_str().unwrap(), &dispatch).unwrap();
    assert!(!reopened.contains_schema("analytics"));
}

#[test]
fn a_second_local_datastore_is_rejected() {
    let dispatch = dispatch(1);
    let db = TempDir::new().unwrap();
    let _datastore = PivotlakeDatastore::open(db.path().to_str().unwrap(), &dispatch).unwrap();

    let error = PivotlakeDatastore::open(db.path().to_str().unwrap(), &dispatch).unwrap_err();

    assert!(error.to_string().contains("already in use"));
    assert!(error.to_string().contains(&std::process::id().to_string()));
}

/// Shared remote stores still support multiple Pivot processes. A datastore
/// already open on S3 learns about a schema another process created when it
/// refreshes the shared manifest.
#[test]
fn a_refresh_picks_up_a_schema_another_process_created() {
    let Some(backend) = harness::s3("schema-refresh-between-processes") else {
        return;
    };
    let dispatch = dispatch(1);
    let datastore = PivotlakeDatastore::open(&backend.root, &dispatch).unwrap();
    let other_process = PivotlakeDatastore::open(&backend.root, &dispatch).unwrap();
    create_schema(&dispatch, &datastore, "analytics").unwrap();
    assert!(!other_process.contains_schema("analytics"));

    let changed = other_process.refresh_from_store().unwrap();

    assert!(changed, "the refresh found a schema it did not hold");
    assert!(other_process.contains_schema("analytics"));
}

#[test]
fn creating_a_table_in_an_unknown_schema_is_rejected() {
    let dispatch = dispatch(1);
    let db = TempDir::new().unwrap();
    let datastore = PivotlakeDatastore::open(db.path().to_str().unwrap(), &dispatch).unwrap();

    let error = create_table(
        &dispatch,
        &datastore,
        create_table_request(Some("analytics"), "events"),
    )
    .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("schema `analytics` does not exist")
    );
}

#[test]
fn same_table_name_in_two_schemas_resolves_separately() {
    let dispatch = dispatch(1);
    let db = TempDir::new().unwrap();
    let datastore = PivotlakeDatastore::open(db.path().to_str().unwrap(), &dispatch).unwrap();
    create_schema(&dispatch, &datastore, "analytics").unwrap();

    create_table(&dispatch, &datastore, create_table_request(None, "events")).unwrap();
    create_table(
        &dispatch,
        &datastore,
        create_table_request(Some("analytics"), "events"),
    )
    .unwrap();

    let transaction = datastore.clone().begin_transaction();
    let default = transaction
        .table_revision(&SchemaQualifiedTableName::new("main", "events"))
        .expect("table lookup succeeds")
        .expect("table in the default schema");
    let analytics = transaction
        .table_revision(&SchemaQualifiedTableName::new("analytics", "events"))
        .expect("table lookup succeeds")
        .expect("table in the analytics schema");
    assert_ne!(default.identity, analytics.identity);
    assert!(
        transaction
            .table_revision(&SchemaQualifiedTableName::new("reporting", "events"))
            .expect("table lookup succeeds")
            .is_none()
    );
}

#[test]
fn a_table_is_stored_at_its_identity() {
    let dispatch = dispatch(1);
    let db = TempDir::new().unwrap();
    let datastore = PivotlakeDatastore::open(db.path().to_str().unwrap(), &dispatch).unwrap();
    create_schema(&dispatch, &datastore, "analytics").unwrap();

    // The same table name in two schemas: neither name appears in a path, so
    // the two cannot collide however they are named or renamed.
    create_table(&dispatch, &datastore, create_table_request(None, "events")).unwrap();
    create_table(
        &dispatch,
        &datastore,
        create_table_request(Some("analytics"), "events"),
    )
    .unwrap();

    let transaction = datastore.clone().begin_transaction();
    for schema in ["main", "analytics"] {
        let id = transaction
            .table_revision(&SchemaQualifiedTableName::new(schema, "events"))
            .expect("table lookup succeeds")
            .expect("table created")
            .identity;
        assert!(
            db.path().join(format!("{id}/_delta_log")).is_dir(),
            "{schema}.events should be stored at {id}"
        );
    }
}

#[test]
fn a_manifest_omitting_its_optional_fields_loads() {
    let dispatch = dispatch(1);
    let db = TempDir::new().unwrap();
    let datastore = PivotlakeDatastore::open(db.path().to_str().unwrap(), &dispatch).unwrap();
    create_table(&dispatch, &datastore, create_table_request(None, "events")).unwrap();

    // A document with no `schemas` key at all, and a schema entry with neither
    // table map: the shapes the deserializer's defaults have to cover.
    let manifest_path = db.path().join("_pivot_manifest.json");
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    let table_ids = manifest["schemas"][0]["table_ids"].clone();
    let table_locations = manifest["table_locations"].clone();
    drop(datastore);
    std::fs::write(
        &manifest_path,
        serde_json::to_vec(&serde_json::json!({ "version": 1 })).unwrap(),
    )
    .unwrap();
    let reopened = PivotlakeDatastore::open(db.path().to_str().unwrap(), &dispatch).unwrap();
    assert!(reopened.contains_schema("main"));
    drop(reopened);

    std::fs::write(
        &manifest_path,
        serde_json::to_vec(&serde_json::json!({
            "version": 1,
            "schemas": [
                { "name": "main", "table_ids": table_ids },
                { "name": "analytics" },
            ],
            "table_locations": table_locations,
        }))
        .unwrap(),
    )
    .unwrap();

    let reopened = PivotlakeDatastore::open(db.path().to_str().unwrap(), &dispatch).unwrap();
    assert!(reopened.contains_schema("analytics"));
    assert!(
        reopened
            .clone()
            .begin_transaction()
            .table(
                DEFAULT_DATASTORE_NAME,
                &SchemaQualifiedTableName::in_default_schema("events")
            )
            .is_some()
    );
}

/// Schema creates and table creates read-modify-write the one manifest
/// document. Run them concurrently: if the two paths serialized against
/// different locks, one would load the manifest before the other's entry landed
/// and overwrite it, and the lost entry would be missing after a reopen.
#[test]
fn concurrent_schema_and_table_creates_all_reach_the_manifest() {
    const CREATES: usize = 8;

    let dispatch = dispatch(1);
    let db = TempDir::new().unwrap();
    let datastore = PivotlakeDatastore::open(db.path().to_str().unwrap(), &dispatch).unwrap();

    std::thread::scope(|scope| {
        for i in 0..CREATES {
            let datastore = datastore.clone();
            let dispatch = &dispatch;
            scope.spawn(move || {
                create_schema(dispatch, &datastore, &format!("schema_{i}")).unwrap();
                create_table(
                    dispatch,
                    &datastore,
                    create_table_request(None, &format!("t_{i}")),
                )
                .unwrap();
            });
        }
    });

    drop(datastore);
    let reopened = PivotlakeDatastore::open(db.path().to_str().unwrap(), &dispatch).unwrap();
    let transaction = reopened.clone().begin_transaction();
    for i in 0..CREATES {
        assert!(
            reopened.contains_schema(&format!("schema_{i}")),
            "schema_{i} was lost from the manifest"
        );
        assert!(
            transaction
                .table(
                    DEFAULT_DATASTORE_NAME,
                    &SchemaQualifiedTableName::in_default_schema(format!("t_{i}"))
                )
                .is_some(),
            "t_{i} was lost from the manifest"
        );
    }
}

/// Resolving and compiling a `CREATE SCHEMA` must leave the catalog alone: the
/// schema appears only once the compiled dataflow runs. Planning a statement
/// without executing it (rendering `EXPLAIN`, reporting an error from a later
/// stage) would otherwise create the schema as a side effect.
#[test]
fn compiling_a_create_schema_does_not_stage_it() {
    let dispatch = dispatch(1);
    let db = TempDir::new().unwrap();
    let datastore = PivotlakeDatastore::open(db.path().to_str().unwrap(), &dispatch).unwrap();
    let transaction = datastore.clone().begin_transaction();

    let _spec = transaction
        .bind_create_schema(CreateSchemaRequest {
            datastore_name: None,
            name: "analytics".to_string(),
            if_not_exists: false,
        })
        .unwrap()
        .compile(&dispatch)
        .unwrap();

    // Bound and compiled, but never executed.
    assert!(!transaction.does_schema_exist("analytics").unwrap());
    commit_datastore_transaction(transaction).unwrap();
    assert!(!datastore.contains_schema("analytics"));

    // Running the same dataflow is what stages it.
    let transaction = datastore.clone().begin_transaction();
    let spec = transaction
        .bind_create_schema(CreateSchemaRequest {
            datastore_name: None,
            name: "analytics".to_string(),
            if_not_exists: false,
        })
        .unwrap()
        .compile(&dispatch)
        .unwrap();
    spec.collect().unwrap();
    commit_datastore_transaction(transaction).unwrap();
    assert!(datastore.contains_schema("analytics"));
}
