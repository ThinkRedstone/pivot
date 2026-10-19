//! Lifecycle for one embedded, in-process Pivot instance.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use catalog::metastore::{Metastore, UserAuth};
use catalog::{DEFAULT_DATASTORE_NAME, Datastore, PivotCatalog};
use datastore_iceberg::IcebergDatastore;
use datastore_pivotlake::{DEFAULT_REFRESH_INTERVAL, MaintenanceConfig, PivotlakeDatastore};
use dispatch::{BUFFER_SIZE, DataFlowDispatcher, Dispatch};
use object_storage::{AmbientExternalStoreFactory, ObjectStore, open_store};
use sysinfo::{MemoryRefreshKind, RefreshKind, System};

use crate::memory::MemoryBudget;
use crate::shell::ShellTarget;

const MIB: u64 = 1024 * 1024;

/// Optional resource limits for an embedded shell instance.
///
/// Omitted fields keep the production CLI defaults: all available dispatch
/// workers and the same share of physical memory `pivot server` takes, see
/// [`MemoryBudget::default`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShellLimits {
    /// Buffer-pool budget. `None` uses the default pool budget.
    pub memory: Option<MemoryBudget>,
    /// Dispatch worker count. `None` uses every available core.
    pub workers: Option<usize>,
}

#[derive(Debug, thiserror::Error)]
enum ResourceLimitError {
    #[error("the dispatch worker count must be at least 1")]
    NoWorkers,
    #[error(
        "the buffer pool memory budget must be at least {} MiB (one pool slot), but {requested_bytes} bytes was requested",
        dispatch::BUFFER_SIZE as u64 / MIB,
    )]
    MemoryTooSmall { requested_bytes: u64 },
    #[error(
        "the buffer pool needs {} MiB but the machine only has {} MiB available right now; every pool slot is faulted in at startup, so opening would be killed by the OOM killer part way through. Free memory on the machine, or lower the budget with `--memory`",
        .requested_bytes / MIB,
        .available_bytes / MIB,
    )]
    InsufficientMemory {
        requested_bytes: u64,
        available_bytes: u64,
    },
    #[error("the buffer pool memory budget is too large for this platform")]
    MemoryTooLarge,
}

#[derive(Debug)]
struct EphemeralMetastore;

impl Metastore for EphemeralMetastore {
    fn open_datastores(
        &self,
        _dispatcher: &DataFlowDispatcher,
    ) -> catalog::metastore::Result<HashMap<String, Arc<dyn Datastore>>> {
        Ok(HashMap::new())
    }

    fn default_datastore_name(&self) -> &str {
        DEFAULT_DATASTORE_NAME
    }

    fn user_auth(&self, _username: &str) -> Option<UserAuth> {
        None
    }
}

struct DispatchOwner(Option<Dispatch>);

impl DispatchOwner {
    fn new(dispatch: Dispatch) -> Self {
        Self(Some(dispatch))
    }

    fn dispatcher(&self) -> &DataFlowDispatcher {
        self.0.as_ref().unwrap().dispatcher()
    }

    fn shutdown(mut self) {
        self.stop();
    }

    fn stop(&mut self) {
        if let Some(dispatch) = self.0.take() {
            dispatch.exit();
        }
    }
}

impl Drop for DispatchOwner {
    fn drop(&mut self) {
        self.stop();
    }
}

struct ShellState {
    executor: crate::execution::Executor,
    catalog: Arc<PivotCatalog>,
    dispatch: DispatchOwner,
    /// Dropped after the workers have joined: a pivotlake datastore's lock stays
    /// held until then.
    datastore: Arc<dyn Datastore>,
}

impl ShellState {
    fn shutdown(self) {
        let Self {
            executor,
            catalog,
            dispatch,
            datastore,
        } = self;

        catalog.abort();
        drop(executor);
        drop(catalog);
        dispatch.shutdown();
        drop(datastore);
    }
}

/// An embedded Pivot executor over one datastore.
pub struct ShellInstance {
    state: Option<ShellState>,
    location: String,
}

impl ShellInstance {
    /// Open the production CLI instance over a pivotlake datastore on all
    /// available workers with the default pool budget assigned to dispatch.
    /// `location` is a local directory, or an object-store URI
    /// (`s3://bucket/prefix`, `gs://bucket/prefix`) whose credentials come
    /// from the environment.
    pub fn open(location: &str) -> Result<Self, Box<dyn std::error::Error>> {
        Self::open_with_limits(&ShellTarget::pivot(location), ShellLimits::default())
    }

    /// Open `target` with optional production resource limits. Values omitted
    /// from `limits` use the same defaults as [`open`](Self::open).
    pub fn open_with_limits(
        target: &ShellTarget,
        limits: ShellLimits,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let workers = limits
            .workers
            .unwrap_or_else(dispatch::default_worker_count);
        if workers == 0 {
            return Err(ResourceLimitError::NoWorkers.into());
        }

        let memory_bytes = limits
            .memory
            .unwrap_or_default()
            .resolve(memory().total_memory())?;
        if memory_bytes < BUFFER_SIZE as u64 {
            return Err(ResourceLimitError::MemoryTooSmall {
                requested_bytes: memory_bytes,
            }
            .into());
        }

        let available_bytes = memory().available_memory();
        if memory_bytes > available_bytes {
            return Err(ResourceLimitError::InsufficientMemory {
                requested_bytes: memory_bytes,
                available_bytes,
            }
            .into());
        }

        let buffers = usize::try_from(memory_bytes / BUFFER_SIZE as u64)
            .map_err(|_| ResourceLimitError::MemoryTooLarge)?;
        Self::open_with_resources(target, workers, buffers, DEFAULT_REFRESH_INTERVAL)
    }

    /// Open `target` with an explicit dispatch shape and refresh cadence. This
    /// is useful for embedding and for small black-box test instances.
    ///
    /// The instance reloads its tables every `refresh_interval`, the same
    /// background sweep the server runs, so data another process commits to a
    /// shared store, or a table committed to the catalog, becomes visible to
    /// later queries. Background compaction and vacuum stay off: those belong
    /// to one owning process per datastore, and a shell over a shared store is
    /// not it. The sweep spawns onto the ambient tokio runtime, so call this
    /// from within one.
    pub fn open_with_resources(
        target: &ShellTarget,
        workers: usize,
        buffers: usize,
        refresh_interval: Duration,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let dispatch = DispatchOwner::new(Dispatch::spin_up(workers, buffers, None));
        let datastore = open_datastore(target, dispatch.dispatcher(), refresh_interval)?;
        let metastore: Arc<dyn Metastore> = Arc::new(EphemeralMetastore);
        let catalog = Arc::new(
            PivotCatalog::new(
                HashMap::from([(DEFAULT_DATASTORE_NAME.to_string(), datastore.clone())]),
                DEFAULT_DATASTORE_NAME.to_string(),
                metastore,
            )?
            .with_external_parquet_read_context(
                dispatch.dispatcher(),
                Arc::new(AmbientExternalStoreFactory),
            ),
        );
        catalog.start();
        let executor =
            crate::execution::Executor::new(catalog.clone(), dispatch.dispatcher().clone())?;
        Ok(Self {
            state: Some(ShellState {
                executor,
                catalog,
                dispatch,
                datastore,
            }),
            location: target.location().to_string(),
        })
    }

    pub fn executor(&self) -> &crate::execution::Executor {
        &self
            .state
            .as_ref()
            .expect("shell instance has already shut down")
            .executor
    }

    /// The location this instance was opened on, as it was spelled: a local
    /// directory, an object-store URI, or an Iceberg catalog's URI.
    pub fn location(&self) -> &str {
        &self.location
    }

    fn shutdown(&mut self) {
        if let Some(state) = self.state.take() {
            state.shutdown();
        }
    }
}

impl Drop for ShellInstance {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Open the datastore `target` names, refreshing it every `refresh_interval`
/// once the catalog starts it. A pivotlake datastore is opened without
/// maintenance (see [`ShellInstance::open_with_resources`]); an Iceberg
/// datastore reads its tables' files through the process's own credential
/// policy, the same one the `AWS_*` and `GOOGLE_*` variables set for a pivotlake
/// datastore in a bucket.
fn open_datastore(
    target: &ShellTarget,
    dispatcher: &DataFlowDispatcher,
    refresh_interval: Duration,
) -> Result<Arc<dyn Datastore>, Box<dyn std::error::Error>> {
    match target {
        ShellTarget::Pivotlake { location } => {
            let store: Arc<dyn ObjectStore> = open_store(location)?.into();
            let maintenance = MaintenanceConfig {
                refresh_interval,
                compaction: None,
                vacuum: None,
            };
            Ok(PivotlakeDatastore::from_store(
                store,
                dispatcher,
                Some(maintenance),
            )?)
        }
        ShellTarget::Iceberg(config) => Ok(IcebergDatastore::open(
            config,
            Arc::new(AmbientExternalStoreFactory),
            dispatcher,
            refresh_interval,
        )?),
    }
}

fn memory() -> System {
    System::new_with_specifics(RefreshKind::nothing().with_memory(MemoryRefreshKind::everything()))
}
