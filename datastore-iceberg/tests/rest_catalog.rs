//! End-to-end tests against a real Iceberg REST catalog: the
//! `apache/iceberg-rest-fixture` image, storing its tables in the MinIO the
//! object-storage harness starts, both in Docker. Tables are written through
//! the iceberg crate's own writer and read back through a Pivot catalog and
//! planner, exactly as a query would. Every test skips when Docker is not
//! reachable, like the other object-store tests.

mod support;
use support::ReadLog;

use std::collections::HashMap;
use std::ops::Deref;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use arrow_array::cast::AsArray;
use arrow_array::types::{ArrowPrimitiveType, Float64Type, Int64Type};
use arrow_array::{Float64Array, Int64Array, RecordBatch, StringArray, TimestampMicrosecondArray};
use catalog::metastore::{DEFAULT_USER_NAME, Metastore, UserAuth};
use catalog::{Datastore, PivotCatalog};
use datastore_iceberg::{IcebergCatalogConfig, IcebergDatastore};
use dispatch::{DataFlowDispatcher, Dispatch};
use iceberg::arrow::schema_to_arrow_schema;
use iceberg::spec::{
    DataFile, DataFileFormat, Datum, FormatVersion, Literal, Manifest, ManifestContentType,
    ManifestList, ManifestListWriter, ManifestWriterBuilder, NestedField, NestedFieldRef,
    Operation, PartitionKey, PrimitiveLiteral, PrimitiveType, Schema, Snapshot, SnapshotReference,
    SnapshotRetention, Struct, Summary, Transform, Type, UnboundPartitionSpec,
};
use iceberg::table::Table;
use iceberg::transaction::{ApplyTransactionAction, Transaction};
use iceberg::transform::create_transform_function;
use iceberg::writer::base_writer::data_file_writer::DataFileWriterBuilder;
use iceberg::writer::file_writer::ParquetWriterBuilder;
use iceberg::writer::file_writer::location_generator::{
    DefaultFileNameGenerator, DefaultLocationGenerator,
};
use iceberg::writer::file_writer::rolling_writer::RollingFileWriterBuilder;
use iceberg::writer::{IcebergWriter, IcebergWriterBuilder};
use iceberg::{Catalog, CatalogBuilder, NamespaceIdent, TableCreation, TableIdent, TableUpdate};
use iceberg_catalog_rest::{RestCatalog, RestCatalogBuilder};
use iceberg_storage_opendal::OpenDalStorageFactory;
use object_storage::{
    AmbientExternalStoreFactory, ExternalStoreFactory, ObjectStore, StoreError, test_support,
};
use parquet::file::properties::WriterProperties;
use planner::Planner;
use planner::catalog::SchemaQualifiedTableName;
use testcontainers::core::{
    CmdWaitFor, ContainerPort, ExecCommand, Host, IntoContainerPort, Mount, WaitFor,
};
use testcontainers::runners::SyncRunner;
use testcontainers::{Container, GenericImage, ImageExt};

/// The name the Pivot catalog serves the REST catalog's tables under.
const DATASTORE: &str = "lake";
/// The bucket the object-storage harness creates, which the fixture stores
/// its warehouse in.
const WAREHOUSE: &str = "s3://pivot-it/iceberg-warehouse";
/// How often the datastore under test asks the catalog again: short, so a
/// test that commits after opening waits little for the commit to be seen.
const REFRESH_INTERVAL: Duration = Duration::from_millis(200);
/// How long a test waits for a refresh to show a change before giving up.
const REFRESH_TIMEOUT: Duration = Duration::from_secs(15);
/// The Spark that writes what the iceberg crate's writer cannot, and the
/// Iceberg it loads; the Spark major and Scala version are in the
/// runtime's name.
const SPARK_IMAGE_TAG: &str = "3.5.9";
const ICEBERG_VERSION: &str = "1.10.1";
/// The Docker volume Spark's Maven downloads are kept in between runs.
const SPARK_IVY_VOLUME: &str = "pivot-tests-spark-ivy";

/// The REST fixture, the writer's view of it, and the pool the reads run on.
/// One per test binary. Tests serialize their use of the catalog because a
/// refresh or system-table listing observes every test's namespaces.
struct Harness {
    _fixture: Container<GenericImage>,
    catalog_uri: String,
    writer: RestCatalog,
    runtime: tokio::runtime::Runtime,
    dispatch: Dispatch,
}

/// Stop background refreshes when a test releases its catalog.
struct TestCatalog(PivotCatalog);

impl Deref for TestCatalog {
    type Target = PivotCatalog;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Drop for TestCatalog {
    fn drop(&mut self) {
        self.0.abort();
    }
}

fn harness() -> Option<MutexGuard<'static, Harness>> {
    static HARNESS: OnceLock<Option<Mutex<Harness>>> = OnceLock::new();
    HARNESS
        .get_or_init(|| start().map(Mutex::new))
        .as_ref()
        .map(|harness| {
            harness
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        })
}

/// Bring the fixture up against the harness's MinIO, or `None` (with a note)
/// when Docker is not available, so the tests skip rather than fail.
fn start() -> Option<Harness> {
    let Some(_backend) = test_support::s3("iceberg-harness") else {
        eprintln!("[rest_catalog] skipping: MinIO unavailable");
        return None;
    };
    // The harness points the process's `AWS_*` variables at MinIO; the fixture
    // reaches the same endpoint through the Docker host gateway.
    let s3_endpoint = std::env::var("AWS_ENDPOINT_URL").expect("the S3 harness sets its endpoint");
    let access_key = std::env::var("AWS_ACCESS_KEY_ID").unwrap();
    let secret_key = std::env::var("AWS_SECRET_ACCESS_KEY").unwrap();
    let region = std::env::var("AWS_REGION").unwrap();
    let minio_port = url::Url::parse(&s3_endpoint)
        .unwrap()
        .port()
        .expect("the harness endpoint names its port");

    let fixture = GenericImage::new("apache/iceberg-rest-fixture", "1.9.1")
        .with_exposed_port(ContainerPort::Tcp(8181))
        .with_wait_for(WaitFor::message_on_stderr("Started Server@"))
        // The fixture's connection pool needs a database shared by connections.
        .with_env_var("CATALOG_URI", "jdbc:sqlite:/tmp/iceberg-catalog.db")
        .with_env_var("CATALOG_WAREHOUSE", WAREHOUSE)
        .with_env_var("CATALOG_IO__IMPL", "org.apache.iceberg.aws.s3.S3FileIO")
        .with_env_var(
            "CATALOG_S3_ENDPOINT",
            format!("http://host.docker.internal:{minio_port}"),
        )
        .with_env_var("CATALOG_S3_PATH__STYLE__ACCESS", "true")
        .with_env_var("AWS_ACCESS_KEY_ID", &access_key)
        .with_env_var("AWS_SECRET_ACCESS_KEY", &secret_key)
        .with_env_var("AWS_REGION", &region)
        .with_host("host.docker.internal", Host::HostGateway);
    let fixture = fixture.start().expect("start the Iceberg REST fixture");
    let port = fixture.get_host_port_ipv4(8181.tcp()).unwrap();
    let catalog_uri = format!("http://localhost:{port}");
    wait_for_catalog(&catalog_uri);

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let writer = runtime
        .block_on(async {
            RestCatalogBuilder::default()
                .with_storage_factory(Arc::new(OpenDalStorageFactory::S3 {
                    customized_credential_load: None,
                }))
                .load(
                    "writer",
                    HashMap::from([
                        ("uri".to_string(), catalog_uri.clone()),
                        ("s3.endpoint".to_string(), s3_endpoint),
                        ("s3.access-key-id".to_string(), access_key),
                        ("s3.secret-access-key".to_string(), secret_key),
                        ("s3.region".to_string(), region),
                        ("s3.path-style-access".to_string(), "true".to_string()),
                    ]),
                )
                .await
        })
        .expect("build the writer catalog");
    Some(Harness {
        _fixture: fixture,
        catalog_uri,
        writer,
        runtime,
        dispatch: Dispatch::spin_up(2, 64, None),
    })
}

/// Poll the catalog's config endpoint until it answers: the container's log
/// says it started a moment before it accepts requests.
fn wait_for_catalog(uri: &str) {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if ureq::get(&format!("{uri}/v1/config")).call().is_ok() {
            return;
        }
        assert!(Instant::now() < deadline, "the REST catalog never answered");
        std::thread::sleep(Duration::from_millis(250));
    }
}

impl Harness {
    fn dispatcher(&self) -> &DataFlowDispatcher {
        self.dispatch.dispatcher()
    }

    fn block_on<F: std::future::Future>(&self, future: F) -> F::Output {
        self.runtime.block_on(future)
    }

    /// A namespace of the test's own, with one table created in it.
    fn namespace(&self, name: &str, table: &str, schema: Schema) -> NamespaceIdent {
        let namespace = NamespaceIdent::new(name.to_string());
        self.block_on(self.writer.create_namespace(&namespace, HashMap::new()))
            .expect("create namespace");
        self.create_table(&namespace, table, schema);
        namespace
    }

    fn create_table(&self, namespace: &NamespaceIdent, table: &str, schema: Schema) {
        let creation = TableCreation::builder()
            .name(table.to_string())
            .schema(schema)
            .build();
        self.block_on(self.writer.create_table(namespace, creation))
            .expect("create table");
    }

    /// A namespace of the test's own with one table of Iceberg format
    /// version 3.
    fn v3_namespace(&self, name: &str, table: &str, schema: Schema) -> NamespaceIdent {
        let namespace = NamespaceIdent::new(name.to_string());
        self.block_on(self.writer.create_namespace(&namespace, HashMap::new()))
            .expect("create namespace");
        let creation = TableCreation::builder()
            .name(table.to_string())
            .schema(schema)
            .properties(HashMap::from([(
                "format-version".to_string(),
                "3".to_string(),
            )]))
            .build();
        self.block_on(self.writer.create_table(&namespace, creation))
            .expect("create table");
        namespace
    }

    /// A namespace of the test's own with one table, partitioned by
    /// `transform` of the field `partition_field_id`, the partition field
    /// named `partition_name`.
    fn partitioned_namespace(
        &self,
        name: &str,
        table: &str,
        schema: Schema,
        partition_field_id: i32,
        partition_name: &str,
        transform: Transform,
    ) -> NamespaceIdent {
        let namespace = NamespaceIdent::new(name.to_string());
        self.block_on(self.writer.create_namespace(&namespace, HashMap::new()))
            .expect("create namespace");
        let spec = UnboundPartitionSpec::builder()
            .with_spec_id(0)
            .add_partition_field(partition_field_id, partition_name, transform)
            .unwrap()
            .build();
        let creation = TableCreation::builder()
            .name(table.to_string())
            .schema(schema)
            .partition_spec(spec)
            .build();
        self.block_on(self.writer.create_table(&namespace, creation))
            .expect("create table");
        namespace
    }

    fn drop_table(&self, namespace: &NamespaceIdent, table: &str) {
        let ident = TableIdent::new(namespace.clone(), table.to_string());
        self.block_on(self.writer.drop_table(&ident))
            .expect("drop table");
    }

    fn drop_namespace(&self, namespace: &NamespaceIdent) {
        self.block_on(self.writer.drop_namespace(namespace))
            .expect("drop namespace");
    }

    fn load(&self, namespace: &NamespaceIdent, table: &str) -> Table {
        let ident = TableIdent::new(namespace.clone(), table.to_string());
        self.block_on(self.writer.load_table(&ident))
            .expect("load table")
    }

    /// Write `batch` as one data file of `table` and commit it as a new
    /// snapshot, the way an engine appends. The batch's schema is the arrow
    /// form of the table's current schema.
    fn append(&self, namespace: &NamespaceIdent, table: &str, batch: RecordBatch) {
        self.append_to_partition(namespace, table, batch, None, WriterProperties::default());
    }

    /// [`append`](Self::append), writing the file in row groups of
    /// `rows_per_group` rows, so one file holds several row groups with
    /// footer statistics of their own.
    fn append_in_row_groups(
        &self,
        namespace: &NamespaceIdent,
        table: &str,
        batch: RecordBatch,
        rows_per_group: usize,
    ) {
        let properties = WriterProperties::builder()
            .set_max_row_group_row_count(Some(rows_per_group))
            .build();
        self.append_to_partition(namespace, table, batch, None, properties);
    }

    /// [`append`](Self::append) into the partition whose values are
    /// `partition`, for a partitioned table; every row of `batch` must belong
    /// to it. The files are written with `properties`.
    fn append_to_partition(
        &self,
        namespace: &NamespaceIdent,
        table: &str,
        batch: RecordBatch,
        partition: Option<Vec<Option<Literal>>>,
        properties: WriterProperties,
    ) {
        let table = self.load(namespace, table);
        let partition_key = partition.map(|values| {
            PartitionKey::new(
                table.metadata().default_partition_spec().as_ref().clone(),
                table.metadata().current_schema().clone(),
                Struct::from_iter(values),
            )
        });
        self.block_on(async {
            let files = self
                .write_data_files(&table, batch, partition_key, properties)
                .await;
            let transaction = Transaction::new(&table);
            let transaction = transaction
                .fast_append()
                .add_data_files(files)
                .apply(transaction)
                .unwrap();
            transaction.commit(&self.writer).await.unwrap();
        });
    }

    /// Write `batch` into `table`'s location as data files, uncommitted, the
    /// way the iceberg writer does: what an append and a compaction commit.
    async fn write_data_files(
        &self,
        table: &Table,
        batch: RecordBatch,
        partition_key: Option<PartitionKey>,
        properties: WriterProperties,
    ) -> Vec<DataFile> {
        let location_generator = DefaultLocationGenerator::new(table.metadata()).unwrap();
        static NEXT_FILE: AtomicU64 = AtomicU64::new(0);
        let file_name_generator = DefaultFileNameGenerator::new(
            format!("part-{}", NEXT_FILE.fetch_add(1, Ordering::Relaxed)),
            None,
            DataFileFormat::Parquet,
        );
        let parquet_writer =
            ParquetWriterBuilder::new(properties, table.metadata().current_schema().clone());
        let rolling_writer = RollingFileWriterBuilder::new_with_default_file_size(
            parquet_writer,
            table.file_io().clone(),
            location_generator,
            file_name_generator,
        );
        let mut writer = DataFileWriterBuilder::new(rolling_writer)
            .build(partition_key)
            .await
            .unwrap();
        writer.write(batch).await.unwrap();
        writer.close().await.unwrap()
    }

    /// Rewrite every live data file of `table` into one file holding `merged`
    /// and commit that as a `replace` snapshot, the way a compaction does: the
    /// new snapshot's manifest retires the old files and adds the new one.
    /// `merged` must hold exactly the rows of the files it replaces.
    fn compact(&self, namespace: &NamespaceIdent, table: &str, merged: RecordBatch) {
        let loaded = self.load(namespace, table);
        let metadata = loaded.metadata();
        let file_io = loaded.file_io();
        let current = metadata
            .current_snapshot()
            .expect("a table with data has a snapshot");
        let snapshot_id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as i64;
        let sequence_number = metadata.last_sequence_number() + 1;
        let new_snapshot = self.block_on(async {
            let list_bytes = file_io
                .new_input(current.manifest_list())
                .unwrap()
                .read()
                .await
                .unwrap();
            let list =
                ManifestList::parse_with_version(&list_bytes, metadata.format_version()).unwrap();
            let manifest_path = format!("{}/metadata/{snapshot_id}-m0.avro", metadata.location());
            let mut writer = ManifestWriterBuilder::new(
                file_io.new_output(&manifest_path).unwrap(),
                Some(snapshot_id),
                metadata.current_schema().clone(),
                metadata.default_partition_spec().as_ref().clone(),
            )
            .build_v2_data();
            for manifest_file in list.entries() {
                let manifest_bytes = file_io
                    .new_input(&manifest_file.manifest_path)
                    .unwrap()
                    .read()
                    .await
                    .unwrap();
                let manifest = Manifest::parse_avro(&manifest_bytes).unwrap();
                for entry in manifest.entries().iter().filter(|entry| entry.is_alive()) {
                    // An added entry inherits the sequence numbers of its manifest.
                    writer
                        .add_delete_file(
                            entry.data_file().clone(),
                            entry
                                .sequence_number()
                                .unwrap_or(manifest_file.sequence_number),
                            entry
                                .file_sequence_number
                                .or(Some(manifest_file.sequence_number)),
                        )
                        .unwrap();
                }
            }
            for data_file in self
                .write_data_files(&loaded, merged, None, WriterProperties::default())
                .await
            {
                writer.add_file(data_file, sequence_number).unwrap();
            }
            let manifest = writer.write_manifest_file().await.unwrap();
            let list_path = format!("{}/metadata/snap-{snapshot_id}-1.avro", metadata.location());
            let mut list_writer = ManifestListWriter::v2(
                file_io
                    .new_output(&list_path)
                    .unwrap()
                    .writer()
                    .await
                    .unwrap(),
                snapshot_id,
                Some(current.snapshot_id()),
                sequence_number,
            );
            list_writer
                .add_manifests(vec![manifest].into_iter())
                .unwrap();
            list_writer.close().await.unwrap();
            Snapshot::builder()
                .with_snapshot_id(snapshot_id)
                .with_parent_snapshot_id(Some(current.snapshot_id()))
                .with_sequence_number(sequence_number)
                .with_timestamp_ms(
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as i64,
                )
                .with_manifest_list(list_path)
                .with_summary(Summary {
                    operation: Operation::Replace,
                    additional_properties: HashMap::new(),
                })
                .with_schema_id(metadata.current_schema_id())
                .build()
        });
        let updates = [
            TableUpdate::AddSnapshot {
                snapshot: new_snapshot,
            },
            TableUpdate::SetSnapshotRef {
                ref_name: "main".to_string(),
                reference: SnapshotReference::new(
                    snapshot_id,
                    SnapshotRetention::branch(None, None, None),
                ),
            },
        ];
        let body = serde_json::json!({
            "requirements": [
                {"type": "assert-ref-snapshot-id", "ref": "main", "snapshot-id": current.snapshot_id()}
            ],
            "updates": updates.iter().map(|update| serde_json::to_value(update).unwrap()).collect::<Vec<_>>(),
        });
        ureq::post(&format!(
            "{}/v1/namespaces/{}/tables/{table}",
            self.catalog_uri,
            namespace.to_url_string()
        ))
        .send_json(body)
        .expect("commit the compaction");
    }

    /// The current snapshot's manifest list.
    fn manifest_list(&self, namespace: &NamespaceIdent, table: &str) -> ManifestList {
        let loaded = self.load(namespace, table);
        let metadata = loaded.metadata();
        let current = metadata
            .current_snapshot()
            .expect("a table with data has a snapshot");
        self.block_on(async {
            let bytes = loaded
                .file_io()
                .new_input(current.manifest_list())
                .unwrap()
                .read()
                .await
                .unwrap();
            ManifestList::parse_with_version(&bytes, metadata.format_version()).unwrap()
        })
    }

    /// Run `sql` against the catalog with Spark, the reference Iceberg
    /// writer, in a container of its own: what an engine writing the table
    /// does, for the operations the iceberg crate's writer lacks. The catalog
    /// is registered as `c` and is the default, so `sql` names tables as
    /// `namespace.table`. Spark fetches the Iceberg runtime from Maven into
    /// a Docker volume that outlives the container, so only the first run on
    /// a machine downloads it.
    fn spark_sql(&self, sql: &str) {
        let host_port = |uri: &str| {
            url::Url::parse(uri)
                .unwrap()
                .port()
                .expect("a harness endpoint names its port")
        };
        let catalog_port = host_port(&self.catalog_uri);
        let minio_port = host_port(&std::env::var("AWS_ENDPOINT_URL").unwrap());
        let spark = GenericImage::new("apache/spark", SPARK_IMAGE_TAG)
            .with_entrypoint("sleep")
            .with_cmd(["infinity"])
            .with_host("host.docker.internal", Host::HostGateway)
            .with_mount(Mount::volume_mount(SPARK_IVY_VOLUME, "/ivy"))
            // A new volume is root's; the image's own user could not fill it.
            .with_user("root")
            .with_working_dir("/tmp")
            .start()
            .expect("start Spark");
        let catalog = |setting: &str, value: String| {
            [
                "--conf".to_string(),
                format!("spark.sql.catalog.c{setting}={value}"),
            ]
        };
        let mut command = vec![
            "/opt/spark/bin/spark-sql".to_string(),
            "--master".to_string(),
            "local[2]".to_string(),
            "--packages".to_string(),
            format!(
                "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:{ICEBERG_VERSION},org.apache.iceberg:iceberg-aws-bundle:{ICEBERG_VERSION}"
            ),
            "--conf".to_string(),
            "spark.jars.ivy=/ivy".to_string(),
            "--conf".to_string(),
            "spark.sql.extensions=org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions"
                .to_string(),
            "--conf".to_string(),
            "spark.sql.defaultCatalog=c".to_string(),
            "--conf".to_string(),
            "spark.eventLog.enabled=false".to_string(),
        ];
        for (setting, value) in [
            ("", "org.apache.iceberg.spark.SparkCatalog".to_string()),
            (".type", "rest".to_string()),
            (
                ".uri",
                format!("http://host.docker.internal:{catalog_port}"),
            ),
            (".io-impl", "org.apache.iceberg.aws.s3.S3FileIO".to_string()),
            (
                ".s3.endpoint",
                format!("http://host.docker.internal:{minio_port}"),
            ),
            (".s3.path-style-access", "true".to_string()),
            (
                ".s3.access-key-id",
                std::env::var("AWS_ACCESS_KEY_ID").unwrap(),
            ),
            (
                ".s3.secret-access-key",
                std::env::var("AWS_SECRET_ACCESS_KEY").unwrap(),
            ),
            (".client.region", std::env::var("AWS_REGION").unwrap()),
        ] {
            command.extend(catalog(setting, value));
        }
        command.extend(["-e".to_string(), sql.to_string()]);
        let mut run = spark
            .exec(ExecCommand::new(command).with_cmd_ready_condition(CmdWaitFor::exit()))
            .expect("run spark-sql");
        let exit_code = run.exit_code().unwrap();
        assert_eq!(
            exit_code,
            Some(0),
            "spark-sql failed:\n{}",
            String::from_utf8_lossy(&run.stderr_to_vec().unwrap())
        );
    }

    /// Replace `table`'s current schema through the REST protocol, as an
    /// engine's `ALTER TABLE` does: the fixture assigns the new schema its id
    /// and makes it current in one commit.
    fn set_schema(&self, namespace: &NamespaceIdent, table: &str, schema: &Schema) {
        let loaded = self.load(namespace, table);
        let body = serde_json::json!({
            "requirements": [
                {"type": "assert-table-uuid", "uuid": loaded.metadata().uuid().to_string()}
            ],
            "updates": [
                {
                    "action": "add-schema",
                    "schema": serde_json::to_value(schema).unwrap(),
                    "last-column-id": schema.highest_field_id(),
                },
                {"action": "set-current-schema", "schema-id": -1}
            ]
        });
        ureq::post(&format!(
            "{}/v1/namespaces/{}/tables/{table}",
            self.catalog_uri,
            namespace.to_url_string()
        ))
        .send_json(body)
        .expect("update the table's schema");
    }

    fn set_partition_spec(
        &self,
        namespace: &NamespaceIdent,
        table: &str,
        spec: UnboundPartitionSpec,
    ) {
        ureq::post(&format!(
            "{}/v1/namespaces/{}/tables/{table}",
            self.catalog_uri,
            namespace.to_url_string()
        ))
        .send_json(serde_json::json!({
            "requirements": [],
            "updates": [
                {"action": "add-spec", "spec": spec},
                {"action": "set-default-spec", "spec-id": -1}
            ]
        }))
        .expect("update the table's partition spec");
    }

    /// The Pivot side: a fresh datastore over the fixture, served as the
    /// default datastore of a one-datastore catalog, with a planner over it.
    fn pivot(&self) -> (TestCatalog, Planner) {
        self.pivot_with(HashMap::new(), Arc::new(AmbientExternalStoreFactory))
    }

    /// A Pivot catalog over the fixture, with `properties` passed to the REST
    /// client and `store_factory` opening the roots no credentials were
    /// vended for.
    fn pivot_with(
        &self,
        properties: HashMap<String, String>,
        store_factory: Arc<dyn ExternalStoreFactory>,
    ) -> (TestCatalog, Planner) {
        let config = IcebergCatalogConfig {
            uri: self.catalog_uri.clone(),
            properties,
            ..IcebergCatalogConfig::default()
        };
        let datastore =
            IcebergDatastore::open(&config, store_factory, self.dispatcher(), REFRESH_INTERVAL)
                .expect("open the iceberg datastore");
        let catalog = PivotCatalog::new(
            HashMap::from([(DATASTORE.to_string(), datastore as Arc<dyn Datastore>)]),
            DATASTORE.to_string(),
            Arc::new(TrustMetastore),
        )
        .unwrap();
        // The refresh task spawns onto the ambient runtime, as on the server.
        let _runtime = self.runtime.enter();
        catalog.start();
        let planner =
            Planner::from_datastore_names(vec![DATASTORE.to_string()], DATASTORE.to_string())
                .unwrap();
        (TestCatalog(catalog), planner)
    }

    fn query(&self, catalog: &PivotCatalog, planner: &mut Planner, sql: &str) -> Vec<RecordBatch> {
        self.try_query(catalog, planner, sql).unwrap()
    }

    fn try_query(
        &self,
        catalog: &PivotCatalog,
        planner: &mut Planner,
        sql: &str,
    ) -> Result<Vec<RecordBatch>, String> {
        let transaction = catalog.begin_transaction();
        planner
            .plan(sql, transaction.clone())
            .map_err(|error| error.to_string())?
            .compile(self.dispatcher(), transaction.as_ref())
            .map_err(|error| error.to_string())?
            .collect()
            .map_err(|error| error.to_string())
    }

    /// Run `sql` until it fails to plan, and return the error; panic once
    /// [`REFRESH_TIMEOUT`] passes: how a test waits for a background refresh
    /// to notice a table or a schema is gone.
    fn plan_error_until(&self, catalog: &PivotCatalog, planner: &mut Planner, sql: &str) -> String {
        let deadline = Instant::now() + REFRESH_TIMEOUT;
        loop {
            if let Err(error) = self.try_query(catalog, planner, sql) {
                return error;
            }
            assert!(
                Instant::now() < deadline,
                "`{sql}` still planned after {REFRESH_TIMEOUT:?}"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    /// Run `sql` until its result satisfies `is_seen`, or panic once
    /// [`REFRESH_TIMEOUT`] passes: how a test waits for a background refresh
    /// to pick up a change made after the datastore opened.
    fn query_until(
        &self,
        catalog: &PivotCatalog,
        planner: &mut Planner,
        sql: &str,
        is_seen: impl Fn(&[RecordBatch]) -> bool,
    ) -> Vec<RecordBatch> {
        let deadline = Instant::now() + REFRESH_TIMEOUT;
        loop {
            let result = self.try_query(catalog, planner, sql);
            if let Ok(batches) = &result
                && is_seen(batches)
            {
                return result.unwrap();
            }
            assert!(
                Instant::now() < deadline,
                "`{sql}` did not see the change within {REFRESH_TIMEOUT:?}: {result:?}"
            );
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    fn plan_error(&self, catalog: &PivotCatalog, planner: &mut Planner, sql: &str) -> String {
        let transaction = catalog.begin_transaction();
        match planner.plan(sql, transaction) {
            Ok(_) => panic!("`{sql}` planned, but should have failed"),
            Err(error) => error.to_string(),
        }
    }
}

/// A store factory that opens nothing: what a load falls back to when the
/// catalog vended no credentials, so a test passes only when the credentials
/// in the table's storage properties were the ones used.
#[derive(Debug)]
struct NoStoreFactory;

impl ExternalStoreFactory for NoStoreFactory {
    fn open(&self, root_uri: &str) -> object_storage::Result<Arc<dyn ObjectStore>> {
        Err(StoreError::Config(format!(
            "no credentials were vended for `{root_uri}`"
        )))
    }
}

/// A metastore serving no datastores and only the built-in trusted user: the
/// catalog under test is built directly from an already-open datastore.
#[derive(Debug)]
struct TrustMetastore;

impl Metastore for TrustMetastore {
    fn open_datastores(
        &self,
        _dispatcher: &DataFlowDispatcher,
    ) -> catalog::metastore::Result<HashMap<String, Arc<dyn Datastore>>> {
        Ok(HashMap::new())
    }

    fn default_datastore_name(&self) -> &str {
        DATASTORE
    }

    fn user_auth(&self, username: &str) -> Option<UserAuth> {
        (username == DEFAULT_USER_NAME).then_some(UserAuth::Trust)
    }
}

/// An optional top-level field of a primitive type.
fn field(id: i32, name: &str, primitive: PrimitiveType) -> NestedFieldRef {
    Arc::new(NestedField::optional(id, name, Type::Primitive(primitive)))
}

/// A schema of `fields` under `schema_id`, its first field required (the
/// table's key).
fn schema(schema_id: i32, fields: Vec<NestedFieldRef>) -> Schema {
    let mut fields = fields;
    let key = fields.remove(0);
    fields.insert(
        0,
        Arc::new(NestedField::required(
            key.id,
            &key.name,
            *key.field_type.clone(),
        )),
    );
    Schema::builder()
        .with_schema_id(schema_id)
        .with_fields(fields)
        .build()
        .unwrap()
}

/// `id BIGINT NOT NULL, customer VARCHAR, amount DOUBLE`, with field ids 1..3.
fn orders_schema() -> Schema {
    schema(
        0,
        vec![
            field(1, "id", PrimitiveType::Long),
            field(2, "customer", PrimitiveType::String),
            field(3, "amount", PrimitiveType::Double),
        ],
    )
}

/// Rows of [`orders_schema`], in its arrow form (with field ids), ready for
/// the iceberg writer.
fn orders(ids: &[i64], customers: &[&str], amounts: &[f64]) -> RecordBatch {
    let schema = Arc::new(schema_to_arrow_schema(&orders_schema()).unwrap());
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ids.to_vec())),
            Arc::new(StringArray::from(customers.to_vec())),
            Arc::new(Float64Array::from(amounts.to_vec())),
        ],
    )
    .unwrap()
}

/// `id BIGINT NOT NULL, at TIMESTAMP`, with field ids 1..2.
fn events_schema() -> Schema {
    schema(
        0,
        vec![
            field(1, "id", PrimitiveType::Long),
            field(2, "at", PrimitiveType::Timestamp),
        ],
    )
}

/// Rows of [`events_schema`], `at` in microseconds since the epoch, in its
/// arrow form ready for the iceberg writer.
fn events(ids: &[i64], at_micros: &[i64]) -> RecordBatch {
    let schema = Arc::new(schema_to_arrow_schema(&events_schema()).unwrap());
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ids.to_vec())),
            Arc::new(TimestampMicrosecondArray::from(at_micros.to_vec())),
        ],
    )
    .unwrap()
}

/// The bucket `transform`, a bucket transform, puts `id` in.
fn bucket_of(transform: &Transform, id: i64) -> i32 {
    let bucket = create_transform_function(transform)
        .unwrap()
        .transform_literal(&Datum::long(id))
        .unwrap()
        .expect("a long has a bucket");
    let PrimitiveLiteral::Int(bucket) = *bucket.literal() else {
        panic!("a bucket is an int");
    };
    bucket
}

fn total_rows(batches: &[RecordBatch]) -> usize {
    batches.iter().map(RecordBatch::num_rows).sum()
}

/// Column `column` of `batches`, as one value per row.
fn column<T: ArrowPrimitiveType>(batches: &[RecordBatch], column: usize) -> Vec<Option<T::Native>> {
    batches
        .iter()
        .flat_map(|batch| batch.column(column).as_primitive::<T>().iter())
        .collect()
}

fn int64_column(batches: &[RecordBatch], column_index: usize) -> Vec<Option<i64>> {
    column::<Int64Type>(batches, column_index)
}

fn float64_column(batches: &[RecordBatch], column_index: usize) -> Vec<Option<f64>> {
    column::<Float64Type>(batches, column_index)
}

fn string_column(batches: &[RecordBatch], column: usize) -> Vec<Option<String>> {
    batches
        .iter()
        .flat_map(|batch| {
            batch
                .column(column)
                .as_string_view()
                .iter()
                .map(|value| value.map(str::to_string))
        })
        .collect()
}

/// Spark deletes rows of a format version 3 table by deletion vector, a
/// second delete merging into the first file's vector. The datastore leaves
/// the deleted rows out of a scan, of the count and bounds it answers from
/// metadata, and of a late-materialized read; the file holds several row
/// groups, so the deletes land on more than one.
#[test]
fn rows_spark_deleted_by_deletion_vector_are_not_read() {
    let Some(harness) = harness() else { return };
    let namespace = harness.v3_namespace("dv", "orders", orders_schema());
    let ids: Vec<i64> = (1..=10).collect();
    let customers: Vec<String> = ids.iter().map(|id| format!("customer-{id}")).collect();
    let customer_names: Vec<&str> = customers.iter().map(String::as_str).collect();
    let amounts: Vec<f64> = ids.iter().map(|id| *id as f64).collect();
    harness.append_in_row_groups(
        &namespace,
        "orders",
        orders(&ids, &customer_names, &amounts),
        4,
    );
    harness.spark_sql(
        "ALTER TABLE dv.orders SET TBLPROPERTIES ('write.delete.mode' = 'merge-on-read');
         DELETE FROM dv.orders WHERE id IN (2, 5, 10);
         DELETE FROM dv.orders WHERE id = 6;",
    );
    let (catalog, mut planner) = harness.pivot();

    let scanned = harness.query(
        &catalog,
        &mut planner,
        "SELECT id FROM dv.orders ORDER BY id",
    );
    let counted = harness.query(&catalog, &mut planner, "SELECT COUNT(*) FROM dv.orders");
    let bounded = harness.query(
        &catalog,
        &mut planner,
        "SELECT MIN(id), MAX(id) FROM dv.orders",
    );
    let limited = harness.query(
        &catalog,
        &mut planner,
        "SELECT customer FROM dv.orders LIMIT 3",
    );

    assert_eq!(
        int64_column(&scanned, 0),
        [Some(1), Some(3), Some(4), Some(7), Some(8), Some(9)]
    );
    assert_eq!(int64_column(&counted, 0), [Some(6)]);
    assert_eq!(int64_column(&bounded, 0), [Some(1)]);
    assert_eq!(int64_column(&bounded, 1), [Some(9)]);
    let limited = string_column(&limited, 0);
    assert_eq!(limited.len(), 3);
    for customer in limited.into_iter().flatten() {
        assert!(
            !["customer-2", "customer-5", "customer-6", "customer-10"].contains(&customer.as_str()),
            "{customer} was deleted"
        );
    }
}

/// Spark compacts a table deleted from by deletion vector: the rewritten
/// file holds the live rows only and the vector is removed, which leaves a
/// delete manifest with nothing live in the manifest list. The datastore
/// reads the rewritten file with no vector applied.
#[test]
fn a_table_spark_compacted_after_its_deletion_vectors_reads_the_rewritten_file() {
    let Some(harness) = harness() else { return };
    let namespace = harness.v3_namespace("rewritten", "orders", orders_schema());
    let ids: Vec<i64> = (1..=10).collect();
    let customers: Vec<String> = ids.iter().map(|id| format!("customer-{id}")).collect();
    let customer_names: Vec<&str> = customers.iter().map(String::as_str).collect();
    let amounts: Vec<f64> = ids.iter().map(|id| *id as f64).collect();
    harness.append_in_row_groups(
        &namespace,
        "orders",
        orders(&ids, &customer_names, &amounts),
        4,
    );
    harness.spark_sql(
        "ALTER TABLE rewritten.orders SET TBLPROPERTIES ('write.delete.mode' = 'merge-on-read');
         DELETE FROM rewritten.orders WHERE id IN (2, 5, 10);
         CALL c.system.rewrite_data_files(table => 'rewritten.orders', options => map('rewrite-all', 'true'));",
    );
    let manifest_list = harness.manifest_list(&namespace, "orders");
    let delete_manifests: Vec<_> = manifest_list
        .entries()
        .iter()
        .filter(|manifest| manifest.content == ManifestContentType::Deletes)
        .collect();
    assert!(
        !delete_manifests.is_empty()
            && delete_manifests
                .iter()
                .all(|manifest| !manifest.has_added_files() && !manifest.has_existing_files()),
        "the compaction leaves a delete manifest with nothing live: {delete_manifests:?}"
    );
    let (catalog, mut planner) = harness.pivot();

    let scanned = harness.query(
        &catalog,
        &mut planner,
        "SELECT id FROM rewritten.orders ORDER BY id",
    );
    let counted = harness.query(
        &catalog,
        &mut planner,
        "SELECT COUNT(*) FROM rewritten.orders",
    );

    assert_eq!(
        int64_column(&scanned, 0),
        [
            Some(1),
            Some(3),
            Some(4),
            Some(6),
            Some(7),
            Some(8),
            Some(9)
        ]
    );
    assert_eq!(int64_column(&counted, 0), [Some(7)]);
}

/// Spark deletes from a table of format version 2 by position-delete file,
/// which is not a deletion vector: the datastore refuses the scan and the
/// count alike, naming the delete file, rather than reading the deleted row.
#[test]
fn a_table_with_a_position_delete_file_is_refused() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("legacy", "orders", orders_schema());
    harness.append(
        &namespace,
        "orders",
        orders(&[1, 2, 3], &["ann", "bob", "cy"], &[1.0, 2.0, 3.0]),
    );
    harness.spark_sql(
        "ALTER TABLE legacy.orders SET TBLPROPERTIES ('write.delete.mode' = 'merge-on-read');
         DELETE FROM legacy.orders WHERE id = 2;",
    );
    let (catalog, mut planner) = harness.pivot();

    let scanned = harness.try_query(
        &catalog,
        &mut planner,
        "SELECT id FROM legacy.orders ORDER BY id",
    );
    let counted = harness.try_query(&catalog, &mut planner, "SELECT COUNT(*) FROM legacy.orders");

    assert_eq!(
        harness
            .load(&namespace, "orders")
            .metadata()
            .format_version(),
        FormatVersion::V2
    );
    for error in [scanned.unwrap_err(), counted.unwrap_err()] {
        assert!(
            error.contains("is not a deletion vector") && error.contains(".parquet"),
            "{error}"
        );
    }
}

#[test]
fn a_catalog_table_reads_through_a_pivot_query() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("sales", "orders", orders_schema());
    harness.append(
        &namespace,
        "orders",
        orders(&[1, 2, 3], &["ann", "bob", "cy"], &[10.0, 20.0, 30.0]),
    );
    harness.append(
        &namespace,
        "orders",
        orders(&[4, 5], &["dee", "eve"], &[40.0, 50.0]),
    );
    let (catalog, mut planner) = harness.pivot();

    let batches = harness.query(
        &catalog,
        &mut planner,
        "SELECT count(*), sum(amount) FROM sales.orders WHERE customer <> 'bob'",
    );

    assert_eq!(int64_column(&batches, 0), [Some(4)]);
    assert_eq!(float64_column(&batches, 1), [Some(130.0)]);
}

#[test]
fn a_commit_after_the_first_query_is_read_once_the_catalog_refreshes() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("growing", "orders", orders_schema());
    harness.append(&namespace, "orders", orders(&[1], &["ann"], &[1.0]));
    let (catalog, mut planner) = harness.pivot();
    let before = harness.query(&catalog, &mut planner, "SELECT id FROM growing.orders");

    harness.append(
        &namespace,
        "orders",
        orders(&[2, 3], &["bob", "cy"], &[2.0, 3.0]),
    );
    let after = harness.query_until(
        &catalog,
        &mut planner,
        "SELECT id FROM growing.orders ORDER BY id",
        |batches| total_rows(batches) == 3,
    );

    assert_eq!(total_rows(&before), 1);
    assert_eq!(int64_column(&after, 0), [Some(1), Some(2), Some(3)]);
}

#[test]
fn a_table_created_after_the_datastore_opened_is_served_once_the_catalog_refreshes() {
    let Some(harness) = harness() else { return };
    let (catalog, mut planner) = harness.pivot();
    let namespace = harness.namespace("late", "orders", orders_schema());
    harness.append(&namespace, "orders", orders(&[1], &["ann"], &[1.0]));

    let batches = harness.query_until(
        &catalog,
        &mut planner,
        "SELECT id FROM late.orders",
        |batches| total_rows(batches) == 1,
    );

    assert_eq!(int64_column(&batches, 0), [Some(1)]);
}

#[test]
fn renamed_and_added_columns_read_by_field_id() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("evolving", "orders", orders_schema());
    harness.append(
        &namespace,
        "orders",
        orders(&[1, 2], &["ann", "bob"], &[1.0, 2.0]),
    );
    // `customer` becomes `buyer`, `amount` is dropped, and `score` is added,
    // all under the field ids the first file was written with.
    let evolved = schema(
        1,
        vec![
            field(1, "id", PrimitiveType::Long),
            field(2, "buyer", PrimitiveType::String),
            field(4, "score", PrimitiveType::Long),
        ],
    );
    harness.set_schema(&namespace, "orders", &evolved);
    let evolved_arrow = Arc::new(schema_to_arrow_schema(&evolved).unwrap());
    harness.append(
        &namespace,
        "orders",
        RecordBatch::try_new(
            evolved_arrow,
            vec![
                Arc::new(Int64Array::from(vec![3])),
                Arc::new(StringArray::from(vec!["cy"])),
                Arc::new(Int64Array::from(vec![7])),
            ],
        )
        .unwrap(),
    );
    let (catalog, mut planner) = harness.pivot();

    let batches = harness.query(
        &catalog,
        &mut planner,
        "SELECT id, buyer, score FROM evolving.orders ORDER BY id",
    );

    assert_eq!(int64_column(&batches, 0), [Some(1), Some(2), Some(3)]);
    assert_eq!(
        string_column(&batches, 1),
        [
            Some("ann".to_string()),
            Some("bob".to_string()),
            Some("cy".to_string())
        ]
    );
    assert_eq!(int64_column(&batches, 2), [None, None, Some(7)]);
}

#[test]
fn a_v3_table_reads_with_null_for_a_column_its_older_files_predate() {
    let Some(harness) = harness() else { return };
    let namespace = harness.v3_namespace("v3", "orders", orders_schema());
    harness.append(
        &namespace,
        "orders",
        orders(&[1, 2], &["ann", "bob"], &[1.0, 2.0]),
    );
    let evolved = schema(
        1,
        vec![
            field(1, "id", PrimitiveType::Long),
            field(2, "customer", PrimitiveType::String),
            field(3, "amount", PrimitiveType::Double),
            field(4, "score", PrimitiveType::Long),
        ],
    );
    harness.set_schema(&namespace, "orders", &evolved);
    let evolved_arrow = Arc::new(schema_to_arrow_schema(&evolved).unwrap());
    harness.append(
        &namespace,
        "orders",
        RecordBatch::try_new(
            evolved_arrow,
            vec![
                Arc::new(Int64Array::from(vec![3])),
                Arc::new(StringArray::from(vec!["cy"])),
                Arc::new(Float64Array::from(vec![3.0])),
                Arc::new(Int64Array::from(vec![9])),
            ],
        )
        .unwrap(),
    );
    let (catalog, mut planner) = harness.pivot();

    let batches = harness.query(
        &catalog,
        &mut planner,
        "SELECT id, score FROM v3.orders ORDER BY id",
    );

    assert_eq!(
        harness
            .load(&namespace, "orders")
            .metadata()
            .format_version(),
        FormatVersion::V3
    );
    assert_eq!(int64_column(&batches, 0), [Some(1), Some(2), Some(3)]);
    assert_eq!(int64_column(&batches, 1), [None, None, Some(9)]);
}

#[test]
fn a_column_with_an_initial_default_refuses_the_table_by_name() {
    let Some(harness) = harness() else { return };
    let scored = schema(
        0,
        vec![
            field(1, "id", PrimitiveType::Long),
            Arc::new(
                NestedField::optional(2, "score", Type::Primitive(PrimitiveType::Long))
                    .with_initial_default(Literal::long(7)),
            ),
        ],
    );
    let namespace = harness.v3_namespace("defaults", "orders", scored);
    let (catalog, mut planner) = harness.pivot();

    let error = harness.plan_error(&catalog, &mut planner, "SELECT id FROM defaults.orders");
    harness.drop_table(&namespace, "orders");

    assert!(
        error.contains("score") && error.contains("initial default"),
        "{error}"
    );
}

#[test]
fn a_query_over_only_a_newer_column_returns_null_for_older_files() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("widened", "orders", orders_schema());
    harness.append(
        &namespace,
        "orders",
        orders(&[1, 2], &["ann", "bob"], &[1.0, 2.0]),
    );
    let mut fields = orders_schema().as_struct().fields().to_vec();
    fields.push(field(4, "score", PrimitiveType::Long));
    let widened = schema(1, fields);
    harness.set_schema(&namespace, "orders", &widened);
    let (catalog, mut planner) = harness.pivot();

    let batches = harness.query(
        &catalog,
        &mut planner,
        "SELECT count(*), count(score) FROM widened.orders",
    );

    assert_eq!(int64_column(&batches, 0), [Some(2)]);
    assert_eq!(int64_column(&batches, 1), [Some(0)]);
}

#[test]
fn a_predicate_on_an_added_column_keeps_the_files_that_predate_it() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("evolved", "orders", orders_schema());
    harness.append(
        &namespace,
        "orders",
        orders(&[1, 2], &["ann", "bob"], &[1.0, 2.0]),
    );
    let mut fields = orders_schema().as_struct().fields().to_vec();
    fields.push(field(4, "score", PrimitiveType::Long));
    let widened = schema(1, fields);
    harness.set_schema(&namespace, "orders", &widened);
    let widened_arrow = Arc::new(schema_to_arrow_schema(&widened).unwrap());
    harness.append(
        &namespace,
        "orders",
        RecordBatch::try_new(
            widened_arrow,
            vec![
                Arc::new(Int64Array::from(vec![3])),
                Arc::new(StringArray::from(vec!["cy"])),
                Arc::new(Float64Array::from(vec![3.0])),
                Arc::new(Int64Array::from(vec![9])),
            ],
        )
        .unwrap(),
    );
    let (catalog, mut planner) = harness.pivot();

    // The first file's manifest knows nothing of `score`: no bound and no
    // null count. It reads as NULL there, so a comparison finds only the
    // newer file's row, IS NULL finds the older rows, and the extremes come
    // from the one file that holds the column.
    let above = harness.query(
        &catalog,
        &mut planner,
        "SELECT id FROM evolved.orders WHERE score > 5",
    );
    let missing = harness.query(
        &catalog,
        &mut planner,
        "SELECT id FROM evolved.orders WHERE score IS NULL ORDER BY id",
    );
    let extremes = harness.query(
        &catalog,
        &mut planner,
        "SELECT min(score), max(score), count(score) FROM evolved.orders",
    );

    assert_eq!(int64_column(&above, 0), [Some(3)]);
    assert_eq!(int64_column(&missing, 0), [Some(1), Some(2)]);
    assert_eq!(int64_column(&extremes, 0), [Some(9)]);
    assert_eq!(int64_column(&extremes, 1), [Some(9)]);
    assert_eq!(int64_column(&extremes, 2), [Some(1)]);
}

#[test]
fn missing_tables_and_nested_namespaces_do_not_bind() {
    let Some(harness) = harness() else { return };
    let _namespace = harness.namespace("present", "orders", orders_schema());
    let nested = NamespaceIdent::from_strs(["present", "inner"]).unwrap();
    harness
        .block_on(harness.writer.create_namespace(&nested, HashMap::new()))
        .unwrap();
    harness
        .block_on(
            harness.writer.create_table(
                &nested,
                TableCreation::builder()
                    .name("orders".to_string())
                    .schema(orders_schema())
                    .build(),
            ),
        )
        .unwrap();
    let (catalog, mut planner) = harness.pivot();

    let missing_table = harness.plan_error(&catalog, &mut planner, "SELECT * FROM present.nope");
    let nested_namespace = harness.plan_error(
        &catalog,
        &mut planner,
        r#"SELECT * FROM "present.inner".orders"#,
    );
    let present = harness.query(
        &catalog,
        &mut planner,
        "SELECT count(*) FROM present.orders",
    );

    assert!(missing_table.contains("nope"), "{missing_table}");
    assert!(
        nested_namespace.contains("present.inner"),
        "{nested_namespace}"
    );
    assert_eq!(int64_column(&present, 0), [Some(0)]);
}

#[test]
fn a_column_pivot_cannot_represent_refuses_the_table_by_name() {
    let Some(harness) = harness() else { return };
    let tokens = schema(
        0,
        vec![
            field(1, "id", PrimitiveType::Long),
            field(2, "token", PrimitiveType::Uuid),
        ],
    );
    let namespace = harness.namespace("odd", "tokens", tokens);
    let (catalog, mut planner) = harness.pivot();

    let error = harness.plan_error(&catalog, &mut planner, "SELECT id FROM odd.tokens");

    harness.drop_table(&namespace, "tokens");

    assert!(error.contains("token") && error.contains("uuid"), "{error}");
}

#[test]
fn s3_keys_in_the_tables_storage_properties_open_its_stores() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("vended", "orders", orders_schema());
    harness.append(
        &namespace,
        "orders",
        orders(&[1, 2], &["ann", "bob"], &[1.0, 2.0]),
    );
    // The fixture vends nothing, so the keys enter the same merged storage
    // properties through the client's own configuration instead.
    let keys = HashMap::from([
        (
            "s3.access-key-id".to_string(),
            std::env::var("AWS_ACCESS_KEY_ID").unwrap(),
        ),
        (
            "s3.secret-access-key".to_string(),
            std::env::var("AWS_SECRET_ACCESS_KEY").unwrap(),
        ),
        (
            "s3.region".to_string(),
            std::env::var("AWS_REGION").unwrap(),
        ),
        (
            "s3.endpoint".to_string(),
            std::env::var("AWS_ENDPOINT_URL").unwrap(),
        ),
    ]);
    let (catalog, mut planner) = harness.pivot_with(keys, Arc::new(NoStoreFactory));

    let batches = harness.query(
        &catalog,
        &mut planner,
        "SELECT id FROM vended.orders ORDER BY id",
    );

    assert_eq!(int64_column(&batches, 0), [Some(1), Some(2)]);
}

#[test]
fn without_vended_credentials_the_store_factory_opens_the_roots() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("unvended", "orders", orders_schema());
    harness.append(&namespace, "orders", orders(&[1], &["ann"], &[1.0]));
    let (catalog, mut planner) = harness.pivot_with(HashMap::new(), Arc::new(NoStoreFactory));

    let error = harness.plan_error(&catalog, &mut planner, "SELECT id FROM unvended.orders");

    assert!(error.contains("no credentials were vended"), "{error}");
}

#[test]
fn the_datastore_describes_its_tables_to_the_system_catalog() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("described", "orders", orders_schema());
    harness.append(
        &namespace,
        "orders",
        orders(&[1, 2, 3], &["ann", "bob", "cy"], &[1.0, 2.0, 3.0]),
    );
    let (catalog, _) = harness.pivot();
    let datastore = catalog.get_datastore(DATASTORE).unwrap().clone();

    let tables = datastore.begin_transaction().tables().unwrap();

    let orders = tables
        .iter()
        .find(|table| table.name == SchemaQualifiedTableName::new("described", "orders"))
        .expect("the described namespace's table is listed");
    assert_eq!(orders.total_rows, 3);
    assert_eq!(orders.files.len(), 1);
    assert!(
        orders.files[0].path.starts_with(WAREHOUSE),
        "{}",
        orders.files[0].path
    );
    assert_eq!(
        orders
            .columns
            .iter()
            .map(|column| column.name.as_str())
            .collect::<Vec<_>>(),
        ["id", "customer", "amount"]
    );
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&orders.files[0].min_max_stats).unwrap(),
        serde_json::json!({
            "id": { "min": 1, "max": 3 },
            "customer": { "min": "ann", "max": "cy" },
            "amount": { "min": 1.0, "max": 3.0 },
        })
    );
}

#[test]
fn a_dropped_table_is_gone_once_the_catalog_refreshes() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("dropping", "orders", orders_schema());
    harness.append(&namespace, "orders", orders(&[1], &["ann"], &[1.0]));
    let (catalog, mut planner) = harness.pivot();
    let before = harness.query(&catalog, &mut planner, "SELECT id FROM dropping.orders");

    harness.drop_table(&namespace, "orders");
    let error = harness.plan_error_until(&catalog, &mut planner, "SELECT id FROM dropping.orders");

    assert_eq!(total_rows(&before), 1);
    assert!(error.contains("orders"), "{error}");
}

#[test]
fn a_dropped_schema_is_gone_once_the_catalog_refreshes() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("vanishing", "orders", orders_schema());
    let (catalog, mut planner) = harness.pivot();
    let before = harness.query(
        &catalog,
        &mut planner,
        "SELECT count(*) FROM vanishing.orders",
    );

    harness.drop_table(&namespace, "orders");
    harness.drop_namespace(&namespace);
    let error = harness.plan_error_until(
        &catalog,
        &mut planner,
        "SELECT count(*) FROM vanishing.orders",
    );

    assert_eq!(int64_column(&before, 0), [Some(0)]);
    assert!(error.contains("vanishing"), "{error}");
}

#[test]
fn a_table_that_could_not_be_indexed_is_left_out_of_the_listing() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("unindexed", "orders", orders_schema());
    harness.append(&namespace, "orders", orders(&[1], &["ann"], &[1.0]));
    // Every table's store fails to open, so none is indexed.
    let (catalog, mut planner) = harness.pivot_with(HashMap::new(), Arc::new(NoStoreFactory));
    let datastore = catalog.get_datastore(DATASTORE).unwrap().clone();

    let listed = datastore.begin_transaction().tables().unwrap();
    let error = harness.plan_error(&catalog, &mut planner, "SELECT id FROM unindexed.orders");

    assert!(
        !listed
            .iter()
            .any(|table| table.name == SchemaQualifiedTableName::new("unindexed", "orders")),
        "an unindexed table was listed"
    );
    assert!(error.contains("no credentials were vended"), "{error}");
}

#[test]
fn a_filter_prunes_across_several_files() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("multi", "orders", orders_schema());
    harness.append(
        &namespace,
        "orders",
        orders(&[1, 2], &["ann", "bob"], &[1.0, 2.0]),
    );
    harness.append(
        &namespace,
        "orders",
        orders(&[3, 4], &["cy", "di"], &[3.0, 4.0]),
    );
    harness.append(
        &namespace,
        "orders",
        orders(&[5, 6], &["ed", "flo"], &[5.0, 6.0]),
    );
    let (catalog, mut planner) = harness.pivot();

    let above = harness.query(
        &catalog,
        &mut planner,
        "SELECT count(*), CAST(sum(id) AS BIGINT) FROM multi.orders WHERE id > 2",
    );
    let middle = harness.query(
        &catalog,
        &mut planner,
        "SELECT id FROM multi.orders WHERE id BETWEEN 3 AND 4 ORDER BY id",
    );

    assert_eq!(int64_column(&above, 0), [Some(4)]);
    assert_eq!(int64_column(&above, 1), [Some(18)]);
    assert_eq!(int64_column(&middle, 0), [Some(3), Some(4)]);
}

#[test]
fn a_predicate_no_file_can_match_returns_an_empty_result_with_its_columns() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("nomatch", "orders", orders_schema());
    harness.append(
        &namespace,
        "orders",
        orders(&[1, 2], &["ann", "bob"], &[1.0, 2.0]),
    );
    harness.append(
        &namespace,
        "orders",
        orders(&[3, 4], &["cy", "di"], &[3.0, 4.0]),
    );
    let reads = ReadLog::default();
    let (catalog, mut planner) = harness.pivot_with(HashMap::new(), Arc::new(reads.clone()));

    // Every file's `id` bounds sit below 100, so the manifests prune them all
    // and no footer is read; the scan still shapes the projected columns.
    let count = harness.query(
        &catalog,
        &mut planner,
        "SELECT count(*) FROM nomatch.orders WHERE id > 100",
    );
    let transaction = catalog.begin_transaction();
    let plan = planner
        .plan(
            "SELECT id, customer FROM nomatch.orders WHERE id > 100",
            transaction.clone(),
        )
        .unwrap();
    let rows = plan
        .compile(harness.dispatcher(), transaction.as_ref())
        .unwrap()
        .collect()
        .unwrap();

    assert!(reads.take().iter().all(|path| !path.ends_with(".parquet")));
    assert_eq!(int64_column(&count, 0), [Some(0)]);
    assert_eq!(total_rows(&rows), 0);
    assert_eq!(plan.output_names, ["id", "customer"]);
}

#[test]
fn a_predicate_keeping_one_file_reads_it_and_late_materializes_from_it() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("onefile", "orders", orders_schema());
    harness.append(
        &namespace,
        "orders",
        orders(&[1, 2], &["ann", "bob"], &[1.0, 2.0]),
    );
    harness.append(
        &namespace,
        "orders",
        orders(&[3, 4], &["cy", "di"], &[3.0, 4.0]),
    );
    harness.append(
        &namespace,
        "orders",
        orders(&[5, 6], &["ed", "flo"], &[5.0, 6.0]),
    );
    let reads = ReadLog::default();
    let (catalog, mut planner) = harness.pivot_with(HashMap::new(), Arc::new(reads.clone()));

    // Equality admits only the third file. The ordered limit admits two files
    // and materializes its row using the scan's row-group indexes.
    let one = harness.query(
        &catalog,
        &mut planner,
        "SELECT customer FROM onefile.orders WHERE id = 5",
    );
    let first_reads = reads.take();
    let transaction = catalog.begin_transaction();
    let plan = planner
        .plan(
            "SELECT customer FROM onefile.orders WHERE id > 2 ORDER BY id LIMIT 1",
            transaction.clone(),
        )
        .unwrap();
    let plan_text = plan.to_string();
    let ranked = plan
        .compile(harness.dispatcher(), transaction.as_ref())
        .unwrap()
        .collect()
        .unwrap();
    let ranked_reads = reads.take();

    assert!(plan_text.contains("Materialize"), "{plan_text}");
    assert_eq!(
        first_reads
            .iter()
            .filter(|path| path.ends_with(".parquet"))
            .count(),
        1
    );
    let footers = ranked_reads
        .iter()
        .filter(|path| path.ends_with(".parquet"))
        .collect::<Vec<_>>();
    assert_eq!(footers.len(), 2, "{ranked_reads:?}");
    let mut unique_footers = footers.clone();
    unique_footers.sort_unstable();
    unique_footers.dedup();
    assert_eq!(unique_footers.len(), 2, "{ranked_reads:?}");
    for path in unique_footers {
        assert_eq!(footers.iter().filter(|&footer| *footer == path).count(), 1);
    }

    assert_eq!(string_column(&one, 0), [Some("ed".to_string())]);
    assert_eq!(string_column(&ranked, 0), [Some("cy".to_string())]);
}

#[test]
fn late_materialization_preserves_row_indexes_across_files_and_row_groups() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("ordered_groups", "orders", orders_schema());
    for (ids, customers) in [
        ([1, 4], ["ann", "di"]),
        ([2, 5], ["bob", "ed"]),
        ([3, 6], ["cy", "flo"]),
    ] {
        harness.append_in_row_groups(
            &namespace,
            "orders",
            orders(&ids, &customers, &[1.0, 2.0]),
            1,
        );
    }
    let reads = ReadLog::default();
    let (catalog, mut planner) = harness.pivot_with(HashMap::new(), Arc::new(reads.clone()));
    let transaction = catalog.begin_transaction();

    // Prepare a narrower binding before independently preparing the wider scan.
    let first = planner
        .plan(
            "SELECT customer FROM ordered_groups.orders WHERE id = 5",
            transaction.clone(),
        )
        .unwrap()
        .compile(harness.dispatcher(), transaction.as_ref())
        .unwrap()
        .collect()
        .unwrap();
    let plan = planner
        .plan(
            "SELECT customer FROM ordered_groups.orders WHERE id >= 2 ORDER BY id DESC LIMIT 5",
            transaction.clone(),
        )
        .unwrap();
    let plan_text = plan.to_string();
    let rows = plan
        .compile(harness.dispatcher(), transaction.as_ref())
        .unwrap()
        .collect()
        .unwrap();
    let reads = reads.take();

    assert!(plan_text.contains("Materialize"), "{plan_text}");
    assert_eq!(string_column(&first, 0), [Some("ed".to_string())]);
    assert_eq!(
        string_column(&rows, 0),
        ["flo", "ed", "di", "cy", "bob"].map(|name| Some(name.to_string()))
    );
    assert_eq!(
        reads
            .iter()
            .filter(|path| path.ends_with(".parquet"))
            .count(),
        5,
        "{reads:?}"
    );
}

#[test]
fn min_and_max_return_exact_values_across_files() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("bounds", "orders", orders_schema());
    harness.append(
        &namespace,
        "orders",
        orders(&[3, 4], &["cy", "di"], &[3.0, 4.0]),
    );
    harness.append(
        &namespace,
        "orders",
        orders(&[1, 2], &["ann", "bob"], &[1.0, 2.0]),
    );
    harness.append(
        &namespace,
        "orders",
        orders(&[5, 6], &["ed", "flo"], &[5.0, 6.0]),
    );
    let reads = ReadLog::default();
    let (catalog, mut planner) = harness.pivot_with(HashMap::new(), Arc::new(reads.clone()));

    let extremes = harness.query(
        &catalog,
        &mut planner,
        "SELECT min(id), max(id), min(customer), max(customer), min(amount), max(amount) \
         FROM bounds.orders",
    );

    assert_eq!(int64_column(&extremes, 0), [Some(1)]);
    assert_eq!(int64_column(&extremes, 1), [Some(6)]);
    assert_eq!(string_column(&extremes, 2), [Some("ann".to_string())]);
    assert_eq!(string_column(&extremes, 3), [Some("flo".to_string())]);
    assert_eq!(float64_column(&extremes, 4), [Some(1.0)]);
    assert_eq!(float64_column(&extremes, 5), [Some(6.0)]);
    assert_eq!(
        reads
            .take()
            .iter()
            .filter(|path| path.ends_with(".parquet"))
            .count(),
        3,
        "the scan fallback must reuse metadata loaded for the integer aggregates"
    );
}

#[test]
fn multiple_column_aggregates_load_metadata_once() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("cached_bounds", "events", events_schema());
    harness.append(&namespace, "events", events(&[1, 2], &[30, 40]));
    harness.append(&namespace, "events", events(&[3, 4], &[10, 20]));
    let reads = ReadLog::default();
    let (catalog, mut planner) = harness.pivot_with(HashMap::new(), Arc::new(reads.clone()));

    let rows = harness.query(
        &catalog,
        &mut planner,
        r#"SELECT min(id), max(id), min("at"), max("at") FROM cached_bounds.events"#,
    );
    let paths = reads.take();

    assert_eq!(int64_column(&rows, 0), [Some(1)]);
    assert_eq!(int64_column(&rows, 1), [Some(4)]);
    assert_eq!(
        column::<arrow_array::types::TimestampMicrosecondType>(&rows, 2),
        [Some(10)]
    );
    assert_eq!(
        column::<arrow_array::types::TimestampMicrosecondType>(&rows, 3),
        [Some(40)]
    );
    assert_eq!(
        paths
            .iter()
            .filter(|path| path.ends_with(".parquet"))
            .count(),
        2,
        "{paths:?}"
    );
    assert_eq!(
        paths.iter().filter(|path| path.ends_with(".avro")).count(),
        3,
        "one list and two manifests: {paths:?}"
    );
}

#[test]
fn missing_statistics_reuse_loaded_metadata_for_the_scan() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("missing_bounds", "orders", orders_schema());
    harness.append_to_partition(
        &namespace,
        "orders",
        orders(&[1, 60, 100], &["ann", "bob", "cy"], &[1.0, 2.0, 3.0]),
        None,
        WriterProperties::builder()
            .set_statistics_enabled(parquet::file::properties::EnabledStatistics::None)
            .build(),
    );
    let reads = ReadLog::default();
    let (catalog, mut planner) = harness.pivot_with(HashMap::new(), Arc::new(reads.clone()));

    let rows = harness.query(
        &catalog,
        &mut planner,
        "SELECT min(id), max(id) FROM missing_bounds.orders",
    );
    let paths = reads.take();

    assert_eq!(int64_column(&rows, 0), [Some(1)]);
    assert_eq!(int64_column(&rows, 1), [Some(100)]);
    assert_eq!(
        paths
            .iter()
            .filter(|path| path.ends_with(".parquet"))
            .count(),
        1,
        "{paths:?}"
    );
}

#[test]
fn concurrent_clones_share_one_metadata_load() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("concurrent_bounds", "orders", orders_schema());
    harness.append(
        &namespace,
        "orders",
        orders(&[1, 2], &["ann", "bob"], &[1.0, 2.0]),
    );
    let reads = ReadLog::default();
    let (catalog, _) = harness.pivot_with(HashMap::new(), Arc::new(reads.clone()));
    let transaction = catalog.begin_transaction();
    let reference = planner::catalog::TableReference {
        datastore: DATASTORE.to_string(),
        schema: "concurrent_bounds".to_string(),
        table: "orders".to_string(),
    };
    let binding = transaction.bind_table(&reference).unwrap().unwrap();
    let bindings: Vec<_> = (0..8).map(|_| binding.clone_box()).collect();
    let barrier = std::sync::Barrier::new(bindings.len());
    reads.take();

    let bounds = std::thread::scope(|scope| {
        let tasks: Vec<_> = bindings
            .iter()
            .map(|binding| {
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    let (lower, upper) = binding.column_min_max(0).unwrap();
                    (
                        lower.into_inner().as_primitive::<Int64Type>().value(0),
                        upper.into_inner().as_primitive::<Int64Type>().value(0),
                    )
                })
            })
            .collect();
        tasks
            .into_iter()
            .map(|task| task.join().unwrap())
            .collect::<Vec<_>>()
    });
    let paths = reads.take();

    assert_eq!(bounds, vec![(1, 2); 8]);
    assert_eq!(
        paths
            .iter()
            .filter(|path| path.ends_with(".parquet"))
            .count(),
        1,
        "{paths:?}"
    );
    assert_eq!(
        paths.iter().filter(|path| path.ends_with(".avro")).count(),
        1,
        "{paths:?}"
    );
}

#[test]
fn failed_metadata_loads_can_be_retried() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("retry_metadata", "orders", orders_schema());
    harness.append(&namespace, "orders", orders(&[1], &["ann"], &[1.0]));
    let reads = ReadLog::default();
    let (catalog, mut planner) = harness.pivot_with(HashMap::new(), Arc::new(reads.clone()));
    let transaction = catalog.begin_transaction();
    let plan = planner
        .plan("SELECT id FROM retry_metadata.orders", transaction.clone())
        .unwrap();
    reads.fail_next_footer();

    let error = plan
        .compile(harness.dispatcher(), transaction.as_ref())
        .err()
        .expect("the first load must fail");
    let rows = plan
        .compile(harness.dispatcher(), transaction.as_ref())
        .unwrap()
        .collect()
        .unwrap();
    let loaded_paths = reads.take();
    let repeated = plan
        .compile(harness.dispatcher(), transaction.as_ref())
        .unwrap()
        .collect()
        .unwrap();

    assert!(
        error.to_string().contains("injected footer read failure"),
        "{error}"
    );
    assert_eq!(int64_column(&rows, 0), [Some(1)]);
    assert_eq!(int64_column(&repeated, 0), [Some(1)]);
    assert_eq!(
        loaded_paths
            .iter()
            .filter(|path| path.ends_with(".parquet"))
            .count(),
        2,
        "one failure and one successful load: {loaded_paths:?}"
    );
    assert!(
        reads.take().is_empty(),
        "the successful retry must be cached"
    );
}

#[test]
fn min_and_max_under_a_filter_come_from_the_pruned_scan() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("filtered", "orders", orders_schema());
    harness.append(
        &namespace,
        "orders",
        orders(&[1, 2], &["ann", "bob"], &[1.0, 2.0]),
    );
    harness.append(
        &namespace,
        "orders",
        orders(&[3, 4], &["cy", "di"], &[3.0, 4.0]),
    );
    harness.append(
        &namespace,
        "orders",
        orders(&[5, 6], &["ed", "flo"], &[5.0, 6.0]),
    );
    let (catalog, mut planner) = harness.pivot();

    // The surviving [3, 4] row group has minimum 3, but the filter admits only 4.
    // Its metadata cannot answer the filtered aggregate directly.
    let above = harness.query(
        &catalog,
        &mut planner,
        "SELECT min(id), max(id), count(*) FROM filtered.orders WHERE id > 3",
    );
    let one_file = harness.query(
        &catalog,
        &mut planner,
        "SELECT min(id), max(id), min(customer) FROM filtered.orders WHERE id BETWEEN 3 AND 4",
    );
    let none = harness.query(
        &catalog,
        &mut planner,
        "SELECT min(id), max(id), count(*) FROM filtered.orders WHERE id > 100",
    );

    assert_eq!(int64_column(&above, 0), [Some(4)]);
    assert_eq!(int64_column(&above, 1), [Some(6)]);
    assert_eq!(int64_column(&above, 2), [Some(3)]);
    assert_eq!(int64_column(&one_file, 0), [Some(3)]);
    assert_eq!(int64_column(&one_file, 1), [Some(4)]);
    assert_eq!(string_column(&one_file, 2), [Some("cy".to_string())]);
    assert_eq!(int64_column(&none, 0), [None]);
    assert_eq!(int64_column(&none, 1), [None]);
    assert_eq!(int64_column(&none, 2), [Some(0)]);
}

#[test]
fn row_groups_of_a_surviving_file_are_pruned_by_their_own_stats() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("grouped", "orders", orders_schema());
    // One file of three row groups, two rows each: its manifest bounds span
    // ids 1 to 6, so no filter below prunes the file itself; only its row
    // groups' footer statistics can narrow what is read.
    harness.append_in_row_groups(
        &namespace,
        "orders",
        orders(
            &[1, 2, 3, 4, 5, 6],
            &["ann", "bob", "cy", "di", "ed", "flo"],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ),
        2,
    );
    let (catalog, mut planner) = harness.pivot();

    let all = harness.query(
        &catalog,
        &mut planner,
        "SELECT count(*), min(id), max(id) FROM grouped.orders",
    );
    let middle = harness.query(
        &catalog,
        &mut planner,
        "SELECT id, customer FROM grouped.orders WHERE id BETWEEN 3 AND 4 ORDER BY id",
    );
    let top = harness.query(
        &catalog,
        &mut planner,
        "SELECT count(*), min(id), min(customer) FROM grouped.orders WHERE id > 4",
    );

    assert_eq!(int64_column(&all, 0), [Some(6)]);
    assert_eq!(int64_column(&all, 1), [Some(1)]);
    assert_eq!(int64_column(&all, 2), [Some(6)]);
    assert_eq!(int64_column(&middle, 0), [Some(3), Some(4)]);
    assert_eq!(
        string_column(&middle, 1),
        [Some("cy".to_string()), Some("di".to_string())]
    );
    assert_eq!(int64_column(&top, 0), [Some(2)]);
    assert_eq!(int64_column(&top, 1), [Some(5)]);
    assert_eq!(string_column(&top, 2), [Some("ed".to_string())]);
}

#[test]
fn an_ordered_limit_reads_the_row_it_selects_across_files() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("ranked", "orders", orders_schema());
    harness.append(
        &namespace,
        "orders",
        orders(&[1, 2], &["ann", "bob"], &[1.0, 9.0]),
    );
    harness.append(
        &namespace,
        "orders",
        orders(&[3, 4], &["cy", "di"], &[3.0, 4.0]),
    );
    let (catalog, mut planner) = harness.pivot();

    let batches = harness.query(
        &catalog,
        &mut planner,
        "SELECT id, customer FROM ranked.orders ORDER BY amount DESC LIMIT 1",
    );

    assert_eq!(int64_column(&batches, 0), [Some(2)]);
    assert_eq!(string_column(&batches, 1), [Some("bob".to_string())]);
}

#[test]
fn a_partitioned_table_reports_its_partition_and_reads_by_it() {
    let Some(harness) = harness() else { return };
    let namespace = harness.partitioned_namespace(
        "parted",
        "orders",
        orders_schema(),
        2,
        "customer",
        Transform::Identity,
    );
    harness.append_to_partition(
        &namespace,
        "orders",
        orders(&[1, 2], &["ann", "ann"], &[1.0, 2.0]),
        Some(vec![Some(Literal::string("ann"))]),
        WriterProperties::default(),
    );
    harness.append_to_partition(
        &namespace,
        "orders",
        orders(&[3], &["bob"], &[3.0]),
        Some(vec![Some(Literal::string("bob"))]),
        WriterProperties::default(),
    );
    let (catalog, mut planner) = harness.pivot();
    let datastore = catalog.get_datastore(DATASTORE).unwrap().clone();

    let batches = harness.query(
        &catalog,
        &mut planner,
        "SELECT count(*) FROM parted.orders WHERE customer = 'ann'",
    );
    let tables = datastore.begin_transaction().tables().unwrap();

    assert_eq!(int64_column(&batches, 0), [Some(2)]);
    let orders = tables
        .iter()
        .find(|table| table.name == SchemaQualifiedTableName::new("parted", "orders"))
        .expect("the partitioned table is listed");
    assert_eq!(orders.partition_by, ["customer"]);
    let mut partitions: Vec<&str> = orders
        .files
        .iter()
        .map(|file| file.partition.as_str())
        .collect();
    partitions.sort_unstable();
    assert_eq!(partitions, ["customer=ann", "customer=bob"]);
    assert!(
        orders
            .columns
            .iter()
            .any(|column| column.name == "customer" && column.is_partition_key),
        "customer is the partition key"
    );
}

#[test]
fn a_compacted_table_reads_only_the_rewritten_file() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("compacted", "orders", orders_schema());
    harness.append(
        &namespace,
        "orders",
        orders(&[1, 2], &["ann", "bob"], &[1.0, 2.0]),
    );
    harness.append(
        &namespace,
        "orders",
        orders(&[3, 4], &["cy", "di"], &[3.0, 4.0]),
    );
    let (catalog, mut planner) = harness.pivot();
    let datastore = catalog.get_datastore(DATASTORE).unwrap().clone();
    let name = SchemaQualifiedTableName::new("compacted", "orders");
    let files_of = |datastore: &Arc<dyn Datastore>| {
        let tables = datastore.clone().begin_transaction().tables().unwrap();
        let table = tables.iter().find(|table| table.name == name).unwrap();
        table.files.len()
    };
    let before = files_of(&datastore);

    harness.compact(
        &namespace,
        "orders",
        orders(
            &[1, 2, 3, 4],
            &["ann", "bob", "cy", "di"],
            &[1.0, 2.0, 3.0, 4.0],
        ),
    );
    let deadline = Instant::now() + REFRESH_TIMEOUT;
    while files_of(&datastore) != 1 {
        assert!(Instant::now() < deadline, "the compaction was not seen");
        std::thread::sleep(Duration::from_millis(50));
    }
    let after = harness.query(
        &catalog,
        &mut planner,
        "SELECT count(*), CAST(sum(id) AS BIGINT) FROM compacted.orders",
    );

    assert_eq!(before, 2);
    assert_eq!(int64_column(&after, 0), [Some(4)]);
    assert_eq!(int64_column(&after, 1), [Some(10)]);
}

#[test]
fn a_bucket_partition_prunes_files_by_the_constants_bucket() {
    let Some(harness) = harness() else { return };
    let buckets = Transform::Bucket(8);
    let namespace = harness.partitioned_namespace(
        "bucketed",
        "orders",
        orders_schema(),
        1,
        "id_bucket",
        buckets,
    );
    let bucket_of_5 = bucket_of(&buckets, 5);
    harness.append_to_partition(
        &namespace,
        "orders",
        orders(&[5], &["ann"], &[5.0]),
        Some(vec![Some(Literal::int(bucket_of_5))]),
        WriterProperties::default(),
    );
    let other_bucket = (1..=4)
        .map(|id| bucket_of(&buckets, id))
        .find(|bucket| *bucket != bucket_of_5)
        .unwrap();
    let below = (1..=4)
        .find(|id| bucket_of(&buckets, *id) == other_bucket)
        .unwrap();
    let above = (6..100)
        .find(|id| bucket_of(&buckets, *id) == other_bucket)
        .unwrap();
    harness.append_to_partition(
        &namespace,
        "orders",
        orders(&[below, above], &["bob", "cy"], &[1.0, 6.0]),
        Some(vec![Some(Literal::int(other_bucket))]),
        WriterProperties::default(),
    );
    let reads = ReadLog::default();
    let (catalog, mut planner) = harness.pivot_with(HashMap::new(), Arc::new(reads.clone()));

    let equal = harness.query(
        &catalog,
        &mut planner,
        "SELECT count(*) FROM bucketed.orders WHERE id = 5",
    );
    let equal_reads = reads.take();
    let range = harness.query(
        &catalog,
        &mut planner,
        "SELECT count(*) FROM bucketed.orders WHERE id > 0",
    );

    assert_eq!(
        equal_reads
            .iter()
            .filter(|path| path.ends_with(".avro"))
            .count(),
        2,
        "{equal_reads:?}"
    );
    assert_eq!(
        equal_reads
            .iter()
            .filter(|path| path.ends_with(".parquet"))
            .count(),
        1
    );
    assert_eq!(int64_column(&equal, 0), [Some(1)]);
    // A bucket carries no range, so the range reads both files.
    assert_eq!(int64_column(&range, 0), [Some(3)]);
}

#[test]
fn a_day_partition_keeps_the_file_holding_the_boundary_row() {
    let Some(harness) = harness() else { return };
    let namespace = harness.partitioned_namespace(
        "daily",
        "events",
        events_schema(),
        2,
        "at_day",
        Transform::Day,
    );
    const MICROS_PER_DAY: i64 = 86_400_000_000;
    const MICROS_PER_HOUR: i64 = 3_600_000_000;
    let (jan_1, jan_2) = (18_262, 18_263);
    let day = |days: i32| Some(vec![Some(Literal::Primitive(PrimitiveLiteral::Int(days)))]);
    let midnight = |days: i32| i64::from(days) * MICROS_PER_DAY;
    harness.append_to_partition(
        &namespace,
        "events",
        events(
            &[1, 2],
            &[midnight(jan_1), midnight(jan_1) + MICROS_PER_HOUR],
        ),
        day(jan_1),
        WriterProperties::default(),
    );
    harness.append_to_partition(
        &namespace,
        "events",
        events(
            &[3, 4],
            &[midnight(jan_2), midnight(jan_2) + MICROS_PER_HOUR],
        ),
        day(jan_2),
        WriterProperties::default(),
    );
    let (catalog, mut planner) = harness.pivot();
    let count_where = |planner: &mut Planner, condition: &str| {
        let batches = harness.query(
            &catalog,
            planner,
            &format!(r#"SELECT count(*) FROM daily.events WHERE "at" {condition}"#),
        );
        int64_column(&batches, 0)[0]
    };

    let before_jan_2 = count_where(&mut planner, "< TIMESTAMP '2020-01-02 00:00:00'");
    let up_to_jan_2 = count_where(&mut planner, "<= TIMESTAMP '2020-01-02 00:00:00'");
    let from_jan_2 = count_where(&mut planner, ">= TIMESTAMP '2020-01-02 00:00:00'");
    let after_jan_1_one_am = count_where(&mut planner, "> TIMESTAMP '2020-01-01 01:00:00'");

    assert_eq!(before_jan_2, Some(2));
    assert_eq!(up_to_jan_2, Some(3));
    assert_eq!(from_jan_2, Some(2));
    assert_eq!(after_jan_1_one_am, Some(2));
}

#[test]
fn partition_values_prune_files_with_overlapping_bounds_in_one_manifest() {
    let Some(harness) = harness() else { return };
    let namespace = harness.partitioned_namespace(
        "partition_files",
        "orders",
        orders_schema(),
        2,
        "customer",
        Transform::Identity,
    );
    let table = harness.load(&namespace, "orders");
    harness.block_on(async {
        let mut files = Vec::new();
        for customer in ["ann", "bob"] {
            let key = PartitionKey::new(
                table.metadata().default_partition_spec().as_ref().clone(),
                table.metadata().current_schema().clone(),
                Struct::from_iter([Some(Literal::string(customer))]),
            );
            // Disabling column statistics makes the partition tuple the only
            // way to distinguish these files before reading their footers.
            let properties = WriterProperties::builder()
                .set_statistics_enabled(parquet::file::properties::EnabledStatistics::None)
                .build();
            files.extend(
                harness
                    .write_data_files(
                        &table,
                        orders(&[1], &[customer], &[1.0]),
                        Some(key),
                        properties,
                    )
                    .await,
            );
        }
        let transaction = Transaction::new(&table);
        transaction
            .fast_append()
            .add_data_files(files)
            .apply(transaction)
            .unwrap()
            .commit(&harness.writer)
            .await
            .unwrap();
    });
    let reads = ReadLog::default();
    let (catalog, mut planner) = harness.pivot_with(HashMap::new(), Arc::new(reads.clone()));

    let rows = harness.query(
        &catalog,
        &mut planner,
        "SELECT id FROM partition_files.orders WHERE customer = 'ann'",
    );
    let reads = reads.take();

    assert_eq!(int64_column(&rows, 0), [Some(1)]);
    assert_eq!(
        reads.iter().filter(|path| path.ends_with(".avro")).count(),
        2,
        "{reads:?}"
    );
    assert_eq!(
        reads
            .iter()
            .filter(|path| path.ends_with(".parquet"))
            .count(),
        1,
        "{reads:?}"
    );
}

#[test]
fn separate_bindings_keep_their_own_filters() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("independent", "orders", orders_schema());
    harness.append(&namespace, "orders", orders(&[1], &["ann"], &[1.0]));
    harness.append(&namespace, "orders", orders(&[2], &["ann"], &[2.0]));
    let reads = ReadLog::default();
    let (catalog, mut planner) = harness.pivot_with(HashMap::new(), Arc::new(reads.clone()));

    let rows = harness.query(&catalog, &mut planner,
        "SELECT a.id, b.id FROM independent.orders a JOIN independent.orders b ON a.customer = b.customer WHERE a.id = 1 AND b.id = 2");
    let reads = reads.take();

    assert_eq!(int64_column(&rows, 0), [Some(1)]);
    assert_eq!(int64_column(&rows, 1), [Some(2)]);
    assert_eq!(
        reads
            .iter()
            .filter(|path| path.ends_with(".parquet"))
            .count(),
        2,
        "{reads:?}"
    );
}

#[test]
fn overlapping_bindings_reload_metadata_and_keep_independent_row_group_filters() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("overlapping", "orders", orders_schema());
    for ids in [[1, 2], [3, 4]] {
        harness.append_in_row_groups(
            &namespace,
            "orders",
            orders(&ids, &["ann", "ann"], &[1.0, 2.0]),
            1,
        );
    }
    let reads = ReadLog::default();
    let (catalog, mut planner) = harness.pivot_with(HashMap::new(), Arc::new(reads.clone()));

    let rows = harness.query(
        &catalog,
        &mut planner,
        "SELECT a.id, b.id FROM overlapping.orders a
         JOIN overlapping.orders b ON a.customer = b.customer
         WHERE a.id <= 3 AND b.id >= 2 ORDER BY a.id, b.id",
    );
    let reads = reads.take();

    assert_eq!(
        int64_column(&rows, 0),
        [1, 1, 1, 2, 2, 2, 3, 3, 3].map(Some)
    );
    assert_eq!(
        int64_column(&rows, 1),
        [2, 3, 4, 2, 3, 4, 2, 3, 4].map(Some)
    );
    assert_eq!(
        reads
            .iter()
            .filter(|path| path.ends_with(".parquet"))
            .count(),
        4,
        "{reads:?}"
    );
    assert_eq!(
        reads.iter().filter(|path| path.ends_with(".avro")).count(),
        5,
        "{reads:?}"
    );
}

#[test]
fn floating_point_predicates_prune_files_without_nans() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("float_files", "orders", orders_schema());
    harness.append(&namespace, "orders", orders(&[1], &["ann"], &[1.0]));
    harness.append(&namespace, "orders", orders(&[3], &["bob"], &[3.0]));
    let reads = ReadLog::default();
    let (catalog, mut planner) = harness.pivot_with(HashMap::new(), Arc::new(reads.clone()));

    for (predicate, expected) in [
        ("= 1", 1),
        ("<> 1", 3),
        ("< 2", 1),
        ("<= 1", 1),
        ("> 2", 3),
        (">= 3", 3),
    ] {
        reads.take();
        let rows = harness.query(
            &catalog,
            &mut planner,
            &format!("SELECT id FROM float_files.orders WHERE amount {predicate}"),
        );
        let paths = reads.take();

        assert_eq!(int64_column(&rows, 0), [Some(expected)], "{predicate}");
        assert_eq!(
            paths
                .iter()
                .filter(|path| path.ends_with(".parquet"))
                .count(),
            1,
            "{predicate}: {paths:?}"
        );
    }
}

#[test]
fn floating_partition_summaries_prune_manifests_without_nans() {
    let Some(harness) = harness() else { return };
    let namespace = harness.partitioned_namespace(
        "float_partitions",
        "orders",
        orders_schema(),
        3,
        "amount_partition",
        Transform::Identity,
    );
    for value in [1, 3] {
        harness.append_to_partition(
            &namespace,
            "orders",
            orders(&[value], &["ann"], &[value as f64]),
            Some(vec![Some(Literal::Primitive(PrimitiveLiteral::Double(
                (value as f64).into(),
            )))]),
            WriterProperties::default(),
        );
    }
    let reads = ReadLog::default();
    let (catalog, mut planner) = harness.pivot_with(HashMap::new(), Arc::new(reads.clone()));

    let rows = harness.query(
        &catalog,
        &mut planner,
        "SELECT id FROM float_partitions.orders WHERE amount > 2",
    );
    let paths = reads.take();

    assert_eq!(int64_column(&rows, 0), [Some(3)]);
    assert_eq!(
        paths.iter().filter(|path| path.ends_with(".avro")).count(),
        2,
        "{paths:?}"
    );
    assert_eq!(
        paths
            .iter()
            .filter(|path| path.ends_with(".parquet"))
            .count(),
        1,
        "{paths:?}"
    );
}

#[test]
fn loose_manifest_bounds_do_not_answer_exact_min_and_max() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("loose_bounds", "orders", orders_schema());
    let table = harness.load(&namespace, "orders");
    harness.block_on(async {
        let files = harness
            .write_data_files(
                &table,
                orders(&[5], &["ann"], &[1.0]),
                None,
                WriterProperties::default(),
            )
            .await;
        let original = &files[0];
        let loose = iceberg::spec::DataFileBuilder::default()
            .content(iceberg::spec::DataContentType::Data)
            .file_path(original.file_path().to_string())
            .file_format(DataFileFormat::Parquet)
            .file_size_in_bytes(original.file_size_in_bytes())
            .record_count(original.record_count())
            .lower_bounds(HashMap::from([(1, iceberg::spec::Datum::long(0))]))
            .upper_bounds(HashMap::from([(1, iceberg::spec::Datum::long(10))]))
            .build()
            .unwrap();
        let transaction = Transaction::new(&table);
        transaction
            .fast_append()
            .add_data_files([loose])
            .apply(transaction)
            .unwrap()
            .commit(&harness.writer)
            .await
            .unwrap();
    });
    let (catalog, mut planner) = harness.pivot();

    let rows = harness.query(
        &catalog,
        &mut planner,
        "SELECT min(id), max(id) FROM loose_bounds.orders",
    );

    assert_eq!(int64_column(&rows, 0), [Some(5)]);
    assert_eq!(int64_column(&rows, 1), [Some(5)]);
}

#[test]
fn a_table_without_a_snapshot_preserves_its_empty_scan_schema() {
    let Some(harness) = harness() else { return };
    let _namespace = harness.namespace("no_snapshot", "orders", orders_schema());
    let (catalog, mut planner) = harness.pivot();

    let transaction = catalog.begin_transaction();
    let plan = planner
        .plan(
            "SELECT id, customer FROM no_snapshot.orders",
            transaction.clone(),
        )
        .unwrap();
    let rows = plan
        .compile(harness.dispatcher(), transaction.as_ref())
        .unwrap()
        .collect()
        .unwrap();
    let count = harness.query(
        &catalog,
        &mut planner,
        "SELECT count(*) FROM no_snapshot.orders",
    );

    assert_eq!(total_rows(&rows), 0);
    assert_eq!(plan.output_names, ["id", "customer"]);
    assert_eq!(int64_column(&count, 0), [Some(0)]);
}

#[test]
fn an_unfiltered_count_reads_only_the_manifest_list() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("count_metadata", "orders", orders_schema());
    harness.append(
        &namespace,
        "orders",
        orders(&[1, 2], &["ann", "bob"], &[1.0, 2.0]),
    );
    let reads = ReadLog::default();
    let (catalog, mut planner) = harness.pivot_with(HashMap::new(), Arc::new(reads.clone()));

    let count = harness.query(
        &catalog,
        &mut planner,
        "SELECT count(*) FROM count_metadata.orders",
    );
    let reads = reads.take();

    assert_eq!(int64_column(&count, 0), [Some(2)]);
    assert_eq!(reads.len(), 1, "{reads:?}");
    assert!(reads[0].ends_with(".avro"));
}

#[test]
fn a_dropped_partition_source_keeps_its_historical_spec_readable() {
    let Some(harness) = harness() else { return };
    let namespace = harness.partitioned_namespace(
        "evolved_partition",
        "orders",
        orders_schema(),
        2,
        "customer",
        Transform::Identity,
    );
    harness.append_to_partition(
        &namespace,
        "orders",
        orders(&[1], &["ann"], &[1.0]),
        Some(vec![Some(Literal::string("ann"))]),
        WriterProperties::default(),
    );
    harness.set_partition_spec(
        &namespace,
        "orders",
        UnboundPartitionSpec::builder().with_spec_id(1).build(),
    );
    harness.set_schema(
        &namespace,
        "orders",
        &schema(
            1,
            vec![
                field(1, "id", PrimitiveType::Long),
                field(3, "amount", PrimitiveType::Double),
            ],
        ),
    );
    let reads = ReadLog::default();
    let (catalog, mut planner) = harness.pivot_with(HashMap::new(), Arc::new(reads.clone()));

    let batches = harness.query(
        &catalog,
        &mut planner,
        "SELECT id FROM evolved_partition.orders WHERE id = 1",
    );

    assert_eq!(int64_column(&batches, 0), [Some(1)]);
    assert_eq!(
        reads
            .take()
            .iter()
            .filter(|path| path.ends_with(".parquet"))
            .count(),
        1
    );
}

#[test]
fn decimal_constants_prune_using_their_unscaled_value() {
    let Some(harness) = harness() else { return };
    let prices = schema(
        0,
        vec![field(
            1,
            "price",
            PrimitiveType::Decimal {
                precision: 10,
                scale: 2,
            },
        )],
    );
    let arrow_schema = Arc::new(schema_to_arrow_schema(&prices).unwrap());
    let namespace = harness.namespace("decimal_bounds", "prices", prices);
    for values in [vec![100, 200], vec![500, 600]] {
        let prices = arrow_array::Decimal128Array::from(values)
            .with_precision_and_scale(10, 2)
            .unwrap();
        harness.append(
            &namespace,
            "prices",
            RecordBatch::try_new(arrow_schema.clone(), vec![Arc::new(prices)]).unwrap(),
        );
    }
    let reads = ReadLog::default();
    let (catalog, mut planner) = harness.pivot_with(HashMap::new(), Arc::new(reads.clone()));

    let batches = harness.query(
        &catalog,
        &mut planner,
        "SELECT CAST(price AS BIGINT) FROM decimal_bounds.prices WHERE price = 5.00",
    );

    assert_eq!(int64_column(&batches, 0), [Some(5)]);
    assert_eq!(
        reads
            .take()
            .iter()
            .filter(|path| path.ends_with(".parquet"))
            .count(),
        1
    );
}

#[test]
fn promoted_partition_bounds_skip_older_files_before_reading_their_footers() {
    let Some(harness) = harness() else { return };
    let original = schema(0, vec![field(1, "id", PrimitiveType::Int)]);
    let namespace = harness.partitioned_namespace(
        "promoted_partition",
        "numbers",
        original.clone(),
        1,
        "id_partition",
        Transform::Identity,
    );
    let batch = RecordBatch::try_new(
        Arc::new(schema_to_arrow_schema(&original).unwrap()),
        vec![Arc::new(arrow_array::Int32Array::from(vec![5]))],
    )
    .unwrap();
    harness.append_to_partition(
        &namespace,
        "numbers",
        batch,
        Some(vec![Some(Literal::int(5))]),
        WriterProperties::default(),
    );
    let promoted = schema(1, vec![field(1, "id", PrimitiveType::Long)]);
    harness.set_schema(&namespace, "numbers", &promoted);
    let large = i64::from(i32::MAX) + 1;
    let batch = RecordBatch::try_new(
        Arc::new(schema_to_arrow_schema(&promoted).unwrap()),
        vec![Arc::new(Int64Array::from(vec![large]))],
    )
    .unwrap();
    harness.append_to_partition(
        &namespace,
        "numbers",
        batch,
        Some(vec![Some(Literal::long(large))]),
        WriterProperties::default(),
    );
    let reads = ReadLog::default();
    let (catalog, mut planner) = harness.pivot_with(HashMap::new(), Arc::new(reads.clone()));

    let rows = harness.query(
        &catalog,
        &mut planner,
        "SELECT id FROM promoted_partition.numbers WHERE id > 2147483647",
    );
    let objects = reads.take();
    // Catalog-wide descriptions would try to read the old file, whose physical
    // type promotion is not supported by the Parquet reader.
    harness.drop_table(&namespace, "numbers");

    assert_eq!(int64_column(&rows, 0), [Some(large)]);
    assert_eq!(
        objects
            .iter()
            .filter(|path| path.ends_with(".avro"))
            .count(),
        2,
        "{objects:?}"
    );
    assert_eq!(
        objects
            .iter()
            .filter(|path| path.ends_with(".parquet"))
            .count(),
        1,
        "{objects:?}"
    );
}

#[test]
fn one_metadata_batch_reads_month_day_and_unpartitioned_files_together() {
    let Some(harness) = harness() else { return };
    let namespace = harness.partitioned_namespace(
        "mixed_layouts",
        "events",
        events_schema(),
        2,
        "at_month",
        Transform::Month,
    );
    const DAY: i64 = 86_400_000_000;
    for (id, day, month) in [(1, 18_262, 600), (2, 18_293, 601)] {
        harness.append_to_partition(
            &namespace,
            "events",
            events(&[id], &[day * DAY]),
            Some(vec![Some(Literal::int(month))]),
            WriterProperties::default(),
        );
    }
    harness.set_partition_spec(
        &namespace,
        "events",
        UnboundPartitionSpec::builder()
            .with_spec_id(1)
            .add_partition_field(2, "at_day", Transform::Day)
            .unwrap()
            .build(),
    );
    for (id, day) in [(3, 18_263), (4, 18_262)] {
        harness.append_to_partition(
            &namespace,
            "events",
            events(&[id], &[i64::from(day) * DAY + DAY / 2]),
            Some(vec![Some(Literal::int(day))]),
            WriterProperties::default(),
        );
    }
    harness.set_partition_spec(
        &namespace,
        "events",
        UnboundPartitionSpec::builder().with_spec_id(2).build(),
    );
    // An empty partition key still carries the evolved spec ID. Omitting the
    // key makes the SDK writer default to spec 0 even after evolution.
    harness.append_to_partition(
        &namespace,
        "events",
        events(&[5], &[18_262 * DAY]),
        Some(vec![]),
        WriterProperties::default(),
    );
    let reads = ReadLog::default();
    let (catalog, mut planner) = harness.pivot_with(HashMap::new(), Arc::new(reads.clone()));
    let before = harness.query(&catalog, &mut planner,
        r#"SELECT id FROM mixed_layouts.events WHERE "at" < TIMESTAMP '2020-01-02 00:00:00' ORDER BY id"#);
    assert_eq!(int64_column(&before, 0), [Some(1), Some(4), Some(5)]);
    let before_reads = reads.take();
    // One manifest list plus the matching month, day and unpartitioned manifests.
    assert_eq!(
        before_reads
            .iter()
            .filter(|path| path.ends_with(".avro"))
            .count(),
        4,
        "{before_reads:?}"
    );
    assert_eq!(
        before_reads
            .iter()
            .filter(|path| path.ends_with(".parquet"))
            .count(),
        3,
        "{before_reads:?}"
    );
    let after = harness.query(&catalog, &mut planner,
        r#"SELECT id FROM mixed_layouts.events WHERE "at" >= TIMESTAMP '2020-01-02 00:00:00' ORDER BY id"#);
    assert_eq!(int64_column(&after, 0), [Some(2), Some(3)]);
}

#[test]
fn an_in_list_and_an_or_read_only_the_files_they_can_match() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("listed", "orders", orders_schema());
    for id in [1, 3, 5] {
        harness.append(&namespace, "orders", orders(&[id], &["ann"], &[1.0]));
    }
    let reads = ReadLog::default();
    let (catalog, mut planner) = harness.pivot_with(HashMap::new(), Arc::new(reads.clone()));

    for predicate in ["id IN (1, 5)", "id = 1 OR id = 5"] {
        reads.take();
        let rows = harness.query(
            &catalog,
            &mut planner,
            &format!("SELECT id FROM listed.orders WHERE {predicate} ORDER BY id"),
        );
        let paths = reads.take();

        assert_eq!(int64_column(&rows, 0), [Some(1), Some(5)], "{predicate}");
        assert_eq!(
            paths
                .iter()
                .filter(|path| path.ends_with(".parquet"))
                .count(),
            2,
            "{predicate}: {paths:?}"
        );
    }
}

#[test]
fn a_timestamp_column_compared_with_a_timestamptz_constant_prunes_files() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("zoned", "events", events_schema());
    const MICROS_PER_DAY: i64 = 86_400_000_000;
    let (jan_1, jan_2) = (18_262, 18_263);
    harness.append(
        &namespace,
        "events",
        events(&[1], &[jan_1 * MICROS_PER_DAY]),
    );
    harness.append(
        &namespace,
        "events",
        events(&[2], &[jan_2 * MICROS_PER_DAY]),
    );
    let reads = ReadLog::default();
    let (catalog, mut planner) = harness.pivot_with(HashMap::new(), Arc::new(reads.clone()));

    let rows = harness.query(
        &catalog,
        &mut planner,
        r#"SELECT id FROM zoned.events WHERE "at" >= TIMESTAMPTZ '2020-01-02 00:00:00+00'"#,
    );
    let paths = reads.take();

    assert_eq!(int64_column(&rows, 0), [Some(2)]);
    assert_eq!(
        paths
            .iter()
            .filter(|path| path.ends_with(".parquet"))
            .count(),
        1,
        "{paths:?}"
    );
}

#[test]
fn a_string_range_reads_only_the_files_it_can_match() {
    let Some(harness) = harness() else { return };
    let namespace = harness.namespace("named", "orders", orders_schema());
    harness.append(&namespace, "orders", orders(&[1], &["ann"], &[1.0]));
    harness.append(&namespace, "orders", orders(&[2], &["zed"], &[2.0]));
    let reads = ReadLog::default();
    let (catalog, mut planner) = harness.pivot_with(HashMap::new(), Arc::new(reads.clone()));

    let rows = harness.query(
        &catalog,
        &mut planner,
        "SELECT id FROM named.orders WHERE customer > 'm'",
    );
    let paths = reads.take();

    assert_eq!(int64_column(&rows, 0), [Some(2)]);
    assert_eq!(
        paths
            .iter()
            .filter(|path| path.ends_with(".parquet"))
            .count(),
        1,
        "{paths:?}"
    );
}
