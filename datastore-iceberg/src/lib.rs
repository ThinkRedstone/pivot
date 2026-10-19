//! A read-only [`catalog::datastore::Datastore`] over the tables of an Iceberg
//! REST catalog.
//!
//! The catalog is the only source of table identity: namespaces are the
//! datastore's schemas (single-level namespaces only, as a Pivot schema is one
//! identifier), and a table is what the catalog last reported it to be.
//!
//! # Freshness
//!
//! The datastore holds the catalog's answer for every table: the load-table
//! response, which is the table's metadata and where it lives. That is fetched
//! whole when the datastore opens and again every refresh interval in the
//! background, the way the pivotlake datastore refreshes its own tables, so a
//! query never waits on the catalog and reads a snapshot at most one interval
//! old. A table created or committed to between refreshes is seen at the next
//! one. What is indexed is small (kilobytes per table). Binding reads the
//! manifest list; preparing a scan lazily reads selected manifests and footers
//! through the ring's caches. Those objects are not indexed.
//!
//! Every bind of one query resolves against the same index, so a refresh landing
//! mid-query does not move a table under it.
//!
//! # Reading through the ring
//!
//! Manifest lists and manifests are read as whole objects over the io_uring
//! ring ([`object_storage::load_objects`]), and deletion vectors as their
//! blob's range of its Puffin file, so they pass through the compressed cache
//! and the disk cache exactly as data does: a restarted server finds them on
//! local disk. A filtered query reads only the manifests whose
//! partition summaries admit its predicates, and only the footers of the files
//! whose partition values and column bounds do. Data files are Parquet, read
//! by the same engine that reads every other datastore, with row-group
//! pruning, late materialization and dynamic filters. Columns are matched to
//! each file by Parquet field id, so renamed columns read correctly and a column added after
//! a file was written reads as NULL for that file.
//!
//! Bounds are taken as recorded: Iceberg and Parquet leave NaNs out of theirs,
//! so a range filter on a floating-point column can skip a file or row group
//! whose only matching rows are NaNs.
//!
//! # Credentials
//!
//! Every table load asks the catalog to vend storage credentials
//! (`X-Iceberg-Access-Delegation: vended-credentials`). A catalog that does
//! answers with S3 keys in the table's storage properties, and the table's
//! store is opened with them; those keys are usually an STS session's,
//! short-lived, and scoped to the table, so the store is opened again at every
//! refresh from the keys the catalog vended then. A catalog that vends nothing
//! leaves the store to the factory the datastore was opened with, the
//! process's own credential policy. Only the flat `s3.*` form is read; the
//! per-prefix `storage-credentials` list is not.
//!
//! A table's files are read through one store, the bucket its metadata file
//! is in. The managed catalogs keep a table in one bucket; a table split
//! across buckets is refused rather than read through the wrong store.
//!
//! # What is refused
//!
//! A table is refused, with an error naming the reason, rather than served
//! partially or wrongly: encryption, a snapshot that carries delete files
//! other than deletion vectors (position-delete and equality-delete files are
//! not applied), data files that are not Parquet, and a column whose Iceberg
//! type has no Pivot type or whose initial default would require synthesizing
//! values for older files.

mod binding;
mod columns;
mod deletes;
mod describe;
pub mod env;
mod manifests;
mod pruning;
mod rest_catalog;
mod store;
mod table;

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use async_trait::async_trait;
use catalog::datastore::{Datastore, DatastoreTableMetadata, DatastoreTransaction};
use dispatch::DataFlowDispatcher;
use iceberg::table::Table;
use iceberg::{Catalog, ErrorKind};
use iceberg_catalog_rest::RestCatalog;
use object_storage::ExternalStoreFactory;
use planner::catalog::{
    BoundTable, Error as CatalogError, Result as CatalogResult, SchemaQualifiedTableName,
    TableReference, TableRevision,
};

use crate::binding::IcebergTableBinding;
use crate::rest_catalog::{block_on, build_rest_catalog};
use crate::store::TableStore;
use crate::table::{LoadedTable, fetch_manifest_list_size, open_table_store};

/// How to reach the REST catalog. The fields mirror the properties the
/// Iceberg REST client is configured with; `properties` passes any further
/// ones (a `prefix`, a `header.*`) through verbatim.
#[derive(Clone, Debug, Default)]
pub struct IcebergCatalogConfig {
    /// The catalog's base URI, such as `https://catalog.example.com/api`.
    pub uri: String,
    /// The warehouse the catalog serves, when it serves several.
    pub warehouse: Option<String>,
    /// What the catalog is authenticated to with; `None` for a catalog that
    /// requires nothing.
    pub auth: Option<IcebergCatalogAuth>,
    /// Further client properties, passed through as written.
    pub properties: HashMap<String, String>,
}

/// What a REST catalog is authenticated to with.
#[derive(Clone)]
pub enum IcebergCatalogAuth {
    /// A bearer token sent on every request.
    Token(String),
    /// An OAuth2 client credential (`client_id:client_secret`, or a bare
    /// secret) exchanged for a token at `server_uri`, or at the catalog's own
    /// token endpoint when there is none, for `scope` when one is given.
    OAuth2 {
        credential: String,
        server_uri: Option<String>,
        scope: Option<String>,
    },
}

/// By hand with the secret redacted: a token or credential must not leak
/// into a log through a `{:?}` of the config that holds it.
impl std::fmt::Debug for IcebergCatalogAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Token(_) => f.write_str("Token(redacted)"),
            Self::OAuth2 {
                server_uri, scope, ..
            } => f
                .debug_struct("OAuth2")
                .field("credential", &"redacted")
                .field("server_uri", server_uri)
                .field("scope", scope)
                .finish(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Iceberg REST catalog: {0}")]
    Catalog(#[from] Box<iceberg::Error>),
    #[error("table `{table}` is staged in the catalog and has no metadata location yet")]
    StagedTable { table: String },
    #[error("table `{table}` could not be indexed at the last refresh: {message}")]
    TableUnavailable { table: String, message: String },
    #[error("table `{table}` is encrypted; encrypted tables are not read")]
    EncryptedTable { table: String },
    #[error(
        "table `{table}` column `{column}` has Iceberg type `{iceberg_type}`, which Pivot cannot represent"
    )]
    UnsupportedColumnType {
        table: String,
        column: String,
        iceberg_type: String,
    },
    #[error(
        "table `{table}` column `{column}` has initial default {default}; a column absent from a file is only read as NULL, so the table is not served"
    )]
    UnsupportedColumnDefault {
        table: String,
        column: String,
        default: String,
    },
    #[error(
        "delete file `{file}` is not a deletion vector; only deletion vectors are applied, so the table is not served"
    )]
    DeleteFile { file: String },
    #[error("the deletion vector for `{file}` is invalid: {message}")]
    DeletionVector { file: String, message: String },
    #[error("table `{table}` data file `{file}` is {format}; only Parquet data files are read")]
    NonParquetFile {
        table: String,
        file: String,
        format: String,
    },
    #[error("table `{table}` metadata object `{path}` is not in the store")]
    MissingMetadataObject { table: String, path: String },
    #[error("table `{table}` partition metadata is invalid: {source}")]
    InvalidPartitionMetadata {
        table: String,
        #[source]
        source: Box<iceberg::Error>,
    },
    #[error("metadata object `{path}` does not parse: {source}")]
    MalformedMetadataObject {
        path: String,
        #[source]
        source: Box<iceberg::Error>,
    },
    #[error("file location `{path}` names no supported store: {message}")]
    UnsupportedLocation { path: String, message: String },
    #[error(
        "table `{table}` file `{file}` is outside `{root}`, where the table's metadata lives; a table's files must all live in one bucket"
    )]
    FileOutsideTableStore {
        table: String,
        file: String,
        root: String,
    },
    #[error("table `{table}` was vended an S3 access key without its secret key, or the reverse")]
    IncompleteVendedCredentials { table: String },
    #[error(transparent)]
    Store(#[from] object_storage::StoreError),
    #[error("reading table metadata over the pool: {0}")]
    Load(#[from] dispatch::DataFlowError),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl From<Error> for CatalogError {
    fn from(error: Error) -> Self {
        CatalogError::Other(Box::new(error))
    }
}

/// The tables of one REST catalog, as the catalog last reported them.
pub struct IcebergDatastore {
    /// What the REST catalog client is built from: the catalog is the source
    /// of truth for what exists and where each table's current metadata
    /// lives, asked at open and at every refresh, each through a client of
    /// its own. A client exchanges an OAuth2 credential for a token on its
    /// first request and never renews it, so none is kept past its refresh,
    /// and the catalog's tokens never expire on one.
    config: IcebergCatalogConfig,
    /// The catalog's URI, which stands in for a data path: every file location
    /// a table reports is absolute.
    catalog_uri: String,
    /// Opens the object store a table's files live in (a bucket, the local
    /// filesystem), applying the process's credential policy. Each refresh
    /// opens every table's store through it, so credentials are resolved every
    /// interval rather than pinned for the process, and the catalog's vended
    /// credentials take its place for a table it vended for.
    store_factory: Arc<dyn ExternalStoreFactory>,
    /// The worker pool a table's metadata (manifest list, manifests, footers)
    /// is read through when it loads. Held by the datastore because a resolve
    /// drives its own dataflows, with no dispatcher passed in.
    dispatcher: DataFlowDispatcher,
    /// The catalog's schemas and tables as of the last refresh: what every
    /// bind resolves against. Replaced whole by each refresh; a query holds
    /// the `Arc` it started with, so a refresh never moves a table under a
    /// query in flight.
    index: RwLock<Arc<DatastoreIndex>>,
    /// How often the background refresh asks the catalog again.
    refresh_interval: Duration,
    /// The refresh task [`start`](Datastore::start) spawned, so
    /// [`abort`](Datastore::abort) can stop it on shutdown.
    refresh_task: Mutex<Option<tokio::task::AbortHandle>>,
}

/// The datastore's index, built from one answer of the catalog and shaped as
/// the catalog is: its schemas (the catalog's top-level namespaces), each
/// holding its tables by name.
#[derive(Default)]
struct DatastoreIndex {
    schemas: HashMap<String, SchemaEntry>,
}

impl DatastoreIndex {
    /// The entry of the table `name`, or `None` when the catalog listed no
    /// such schema or table.
    fn table_entry(&self, name: &SchemaQualifiedTableName) -> Option<&Result<TableEntry, String>> {
        self.schemas.get(&name.schema)?.tables.get(&name.table)
    }
}

/// One schema's tables. A table the catalog listed but that could not be
/// indexed (it would not load, or its manifest list could not be sized) is
/// entered with the reason, so a query over it reports that reason rather
/// than "no such table".
#[derive(Default)]
struct SchemaEntry {
    tables: HashMap<String, Result<TableEntry, String>>,
}

/// One table's entry in the index: the table as the catalog returned it, with
/// what a load needs that the response does not say: the store its files are
/// read through, opened with the credentials the catalog vended, and the size
/// of the current snapshot's manifest list.
struct TableEntry {
    table: Table,
    store: TableStore,
    /// Iceberg names the manifest list without a length, and a ring read needs
    /// one, so the store is asked once, at refresh, and the answer kept here
    /// rather than asked again by every query. Ugly, but off the query path.
    /// `None` for a table with no current snapshot.
    manifest_list_size: Option<u64>,
}

impl TableEntry {
    fn manifest_list_path(&self) -> Option<&str> {
        self.table
            .metadata()
            .current_snapshot()
            .map(|snapshot| snapshot.manifest_list())
    }
}

impl std::fmt::Debug for IcebergDatastore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IcebergDatastore")
            .field("catalog_uri", &self.catalog_uri)
            .finish_non_exhaustive()
    }
}

impl IcebergDatastore {
    /// Open the datastore over the catalog `config` describes, fetching every
    /// table it holds; a catalog that cannot be reached fails the open, as a
    /// store that cannot be read fails a pivotlake datastore's.
    /// `store_factory` opens the object stores the tables' files live in, so
    /// their credentials follow the same policy as every other datastore's;
    /// `refresh_interval` is how often the catalog is asked again once
    /// [`start`](Datastore::start) is called.
    pub fn open(
        config: &IcebergCatalogConfig,
        store_factory: Arc<dyn ExternalStoreFactory>,
        dispatcher: &DataFlowDispatcher,
        refresh_interval: Duration,
    ) -> Result<Arc<Self>> {
        let datastore = Self {
            config: config.clone(),
            catalog_uri: config.uri.clone(),
            store_factory,
            dispatcher: dispatcher.clone(),
            index: RwLock::new(Arc::new(DatastoreIndex::default())),
            refresh_interval,
            refresh_task: Mutex::new(None),
        };
        datastore.refresh()?;
        Ok(Arc::new(datastore))
    }

    /// Ask the catalog, through a client built for the occasion, for
    /// everything it holds and replace the index with the answer. A failure
    /// leaves the index as it was.
    pub fn refresh(&self) -> Result<()> {
        let catalog = build_rest_catalog(&self.config)?;
        let fetched = self.fetch_index(&catalog)?;
        *self.index.write().unwrap() = Arc::new(fetched);
        Ok(())
    }

    /// The index as `catalog` would have it now: a schema per top-level
    /// namespace, and every table of theirs. Namespaces nested below the top
    /// level are not schemas of this datastore and are not listed.
    fn fetch_index(&self, catalog: &RestCatalog) -> Result<DatastoreIndex> {
        let previous = self.index();
        let mut index = DatastoreIndex::default();
        let namespaces = block_on(catalog.list_namespaces(None)).map_err(Box::new)?;
        for namespace in namespaces {
            let listed = block_on(catalog.list_tables(&namespace)).map_err(Box::new)?;
            // Listed under no parent, the namespace is a top-level one: a
            // single identifier, which is its schema name.
            let schema = namespace
                .inner()
                .pop()
                .expect("a namespace has at least one level");
            let mut tables = HashMap::new();
            for ident in listed {
                let name = SchemaQualifiedTableName::new(schema.clone(), ident.name.clone());
                let entry = match block_on(catalog.load_table(&ident)) {
                    Ok(table) => self
                        .build_table_entry(&name, table, previous.table_entry(&name))
                        .map_err(|error| error.to_string()),
                    // A table dropped between the listing and its load is
                    // simply not there any more.
                    Err(error)
                        if matches!(
                            error.kind(),
                            ErrorKind::TableNotFound | ErrorKind::NamespaceNotFound
                        ) =>
                    {
                        continue;
                    }
                    // A table the catalog lists but will not hand over (a
                    // permission it lacks; a view it lists among tables) is
                    // entered with the catalog's reason, so a query naming it
                    // reports that rather than "no such table", and so it
                    // does not keep every other table from refreshing.
                    Err(error) => {
                        tracing::warn!(
                            table = %name,
                            error = %error,
                            "a table the catalog lists could not be loaded from it"
                        );
                        Err(error.to_string())
                    }
                };
                tables.insert(ident.name, entry);
            }
            index.schemas.insert(schema, SchemaEntry { tables });
        }
        Ok(index)
    }

    /// `table`'s entry: with its store, opened now from the credentials the
    /// catalog just vended, and its manifest list's size, the one `previous`
    /// learned when the snapshot is the one it entered, else asked of the
    /// store now.
    fn build_table_entry(
        &self,
        name: &SchemaQualifiedTableName,
        table: Table,
        previous: Option<&Result<TableEntry, String>>,
    ) -> Result<TableEntry> {
        let store = open_table_store(name, &table, &*self.store_factory)?;
        let path = table
            .metadata()
            .current_snapshot()
            .map(|snapshot| snapshot.manifest_list());
        let unchanged = previous
            .and_then(|previous| previous.as_ref().ok())
            .filter(|previous| path.is_some() && previous.manifest_list_path() == path);
        let manifest_list_size = match unchanged {
            Some(previous) => previous.manifest_list_size,
            None => fetch_manifest_list_size(name, &table, &store)?,
        };
        Ok(TableEntry {
            table,
            store,
            manifest_list_size,
        })
    }

    /// The index as of now: the `Arc` a query pins for its lifetime, so the
    /// refresh's swap moves nothing under it.
    fn index(&self) -> Arc<DatastoreIndex> {
        self.index.read().unwrap().clone()
    }

    /// The table `name` as `index` has it, its schema being the namespace,
    /// loaded; or `None` when the catalog listed no such table.
    fn load_table(
        &self,
        index: &DatastoreIndex,
        name: &SchemaQualifiedTableName,
    ) -> Result<Option<Arc<LoadedTable>>> {
        let Some(entry) = index.table_entry(name) else {
            return Ok(None);
        };
        let entry = entry.as_ref().map_err(|message| Error::TableUnavailable {
            table: name.to_string(),
            message: message.clone(),
        })?;
        let loaded = LoadedTable::load(
            name,
            &entry.table,
            entry.store.clone(),
            entry.manifest_list_size,
            &self.dispatcher,
        )?;
        Ok(Some(Arc::new(loaded)))
    }

    /// Every table `index` has that the catalog handed over, loaded. A table
    /// the catalog lists but would not hand over (Unity Catalog lists its
    /// `information_schema` views, which are not Iceberg tables) is not
    /// served, so it is not listed; a query naming it reports why.
    fn load_all_tables(&self, index: &DatastoreIndex) -> Result<Vec<Arc<LoadedTable>>> {
        let mut tables = Vec::new();
        for (schema, schema_entry) in &index.schemas {
            for (table, entry) in &schema_entry.tables {
                if entry.is_err() {
                    continue;
                }
                let name = SchemaQualifiedTableName::new(schema.clone(), table.clone());
                if let Some(loaded) = self.load_table(index, &name)? {
                    tables.push(loaded);
                }
            }
        }
        Ok(tables)
    }
}

impl Datastore for IcebergDatastore {
    fn begin_transaction(self: Arc<Self>) -> Arc<dyn DatastoreTransaction> {
        let index = self.index();
        Arc::new(IcebergTransaction {
            datastore: self,
            index,
            loaded_tables: Mutex::new(HashMap::new()),
        })
    }

    fn kind(&self) -> &'static str {
        "iceberg"
    }

    /// The catalog's URI: every file path a table reports is a full location
    /// of its own, since Iceberg names files absolutely.
    fn data_path(&self) -> String {
        self.catalog_uri.clone()
    }

    /// Refresh the index every `refresh_interval`, on the ambient runtime. A
    /// refresh that fails is logged and the index kept, so an
    /// unreachable catalog degrades to stale tables rather than none.
    fn start(self: Arc<Self>) {
        let datastore = Arc::clone(&self);
        let interval = self.refresh_interval;
        let task = tokio::spawn(async move {
            let mut tick = tokio::time::interval(interval);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            // The catalog was just fetched at open, so skip the interval's
            // immediate first tick and refresh one full interval from now.
            tick.tick().await;
            loop {
                tick.tick().await;
                // The refresh blocks on catalog calls, so it runs off the
                // reactor.
                let datastore = Arc::clone(&datastore);
                match tokio::task::spawn_blocking(move || datastore.refresh()).await {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => {
                        tracing::warn!(error = %error, "Iceberg catalog refresh failed")
                    }
                    Err(error) => {
                        tracing::warn!(error = %error, "Iceberg catalog refresh panicked")
                    }
                }
            }
        });
        *self.refresh_task.lock().unwrap() = Some(task.abort_handle());
    }

    fn abort(&self) {
        if let Some(task) = self.refresh_task.lock().unwrap().take() {
            task.abort();
        }
    }
}

/// One query's view of the catalog: the index as it was when the query began,
/// which every schema check and bind of the query resolves against, so a
/// refresh landing mid-query moves nothing under it. A table is loaded the
/// first time the query touches it and kept for the query's lifetime, so a
/// plan's scan and its late materialize see the same load, and a table the
/// query names twice is loaded once.
pub struct IcebergTransaction {
    datastore: Arc<IcebergDatastore>,
    index: Arc<DatastoreIndex>,
    loaded_tables: Mutex<HashMap<SchemaQualifiedTableName, Option<Arc<LoadedTable>>>>,
}

impl std::fmt::Debug for IcebergTransaction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IcebergTransaction")
            .field("datastore", &self.datastore)
            .finish_non_exhaustive()
    }
}

impl IcebergTransaction {
    fn resolve_table(&self, name: &SchemaQualifiedTableName) -> Result<Option<Arc<LoadedTable>>> {
        if let Some(table) = self.loaded_tables.lock().unwrap().get(name) {
            return Ok(table.clone());
        }
        let table = self.datastore.load_table(&self.index, name)?;
        self.loaded_tables
            .lock()
            .unwrap()
            .insert(name.clone(), table.clone());
        Ok(table)
    }
}

#[async_trait]
impl DatastoreTransaction for IcebergTransaction {
    /// Whether the catalog had a top-level namespace named `schema` when the
    /// query began.
    fn does_schema_exist(&self, schema: &str) -> CatalogResult<bool> {
        Ok(self.index.schemas.contains_key(schema))
    }

    fn bind_table(
        &self,
        datastore: &str,
        name: &SchemaQualifiedTableName,
    ) -> CatalogResult<Option<Box<dyn BoundTable>>> {
        let Some(table) = self.resolve_table(name)? else {
            return Ok(None);
        };
        let reference = TableReference {
            datastore: datastore.to_string(),
            schema: name.schema.clone(),
            table: name.table.clone(),
        };
        Ok(Some(Box::new(IcebergTableBinding::new(reference, table))))
    }

    fn table_revision(
        &self,
        name: &SchemaQualifiedTableName,
    ) -> CatalogResult<Option<TableRevision>> {
        Ok(self.resolve_table(name)?.map(|table| table.revision()))
    }

    fn tables(&self) -> CatalogResult<Vec<DatastoreTableMetadata>> {
        let tables = self.datastore.load_all_tables(&self.index)?;
        let mut memo = self.loaded_tables.lock().unwrap();
        let described = tables
            .into_iter()
            .map(|table| {
                memo.insert(table.name.clone(), Some(table.clone()));
                crate::describe::describe_table(&table)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(described)
    }
}
