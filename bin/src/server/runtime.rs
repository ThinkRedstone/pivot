//! Construction and foreground execution of a configured Pivot server.

use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use catalog::PivotCatalog;
use catalog::metastore::Metastore;
use clap::Args;
use dispatch::{BUFFER_SIZE, DataFlowDispatcher, Dispatch, SpillDirectory};
use metastore_disk::{DiskMetastore, MetastoreConfig};
use tracing::{error, info};

use crate::server::config::MetastoreStore;
use crate::server::mapped_files::populate_mapped_files;
use crate::server::{Config, Error, Server, raise_open_file_limit};

/// Options accepted by `pivot server`.
#[derive(Args, Debug)]
pub struct ServerOptions {
    /// Config file (YAML) describing this instance: its memory and worker
    /// budgets, spill directory, disk cache and log; a `server` section for
    /// the endpoint;
    /// `datastores`, `secrets` and `users` for what it serves; and
    /// `metastore` naming the file the server writes users it is told to
    /// create into.
    #[arg(long, value_name = "FILE")]
    config: PathBuf,
}

/// Wait for an interactive interrupt or the termination signal sent by a
/// service manager.
fn wait_for_shutdown_signal() -> impl std::future::Future<Output = ()> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let terminate = signal(SignalKind::terminate());
        let interrupt = signal(SignalKind::interrupt());
        async move {
            let (mut terminate, mut interrupt) = match (terminate, interrupt) {
                (Ok(terminate), Ok(interrupt)) => (terminate, interrupt),
                (Err(error), _) | (_, Err(error)) => {
                    error!(%error, "failed to listen for shutdown signals");
                    return;
                }
            };
            tokio::select! {
                _ = interrupt.recv() => {}
                _ = terminate.recv() => {}
            }
        }
    }

    #[cfg(not(unix))]
    async {
        if let Err(error) = tokio::signal::ctrl_c().await {
            error!(%error, "failed to listen for Ctrl-C");
        }
    }
}

/// Return the machine's total physical memory in bytes.
fn get_total_memory() -> usize {
    read_memory().total_memory() as usize
}

/// Return how many bytes a new allocation on this machine can actually get hold
/// of: free pages plus what the kernel can reclaim without swapping.
fn get_available_memory() -> usize {
    read_memory().available_memory() as usize
}

/// One reading of the machine's memory, as the OS reports it now.
fn read_memory() -> sysinfo::System {
    sysinfo::System::new_with_specifics(
        sysinfo::RefreshKind::nothing().with_memory(sysinfo::MemoryRefreshKind::everything()),
    )
}

fn build_metastore(
    entries: MetastoreConfig,
    path: &Path,
    store: Option<MetastoreStore>,
    refresh_interval: Duration,
) -> Result<Arc<DiskMetastore>, Error> {
    let metastore_file = store.map(|MetastoreStore::File { path }| path);
    let metastore = DiskMetastore::open(entries, metastore_file.as_deref(), refresh_interval)
        .map_err(|source| Error::Metastore {
            path: path.to_path_buf(),
            source: Box::new(source),
        })?;
    Ok(Arc::new(metastore))
}

fn build_catalog(
    path: &Path,
    metastore: &Arc<DiskMetastore>,
    dispatcher: &DataFlowDispatcher,
) -> Result<Arc<PivotCatalog>, Error> {
    let default_name = metastore.default_datastore_name().to_string();
    let datastores =
        metastore
            .open_datastores(dispatcher)
            .map_err(|source| Error::OpenDatastores {
                path: path.to_path_buf(),
                source,
            })?;
    let catalog = PivotCatalog::new(datastores, default_name, metastore.clone())?
        .with_external_parquet_read_context(dispatcher, metastore.external_store_factory());
    Ok(Arc::new(catalog))
}

/// Run a configured server in the foreground until SIGINT or SIGTERM.
pub fn run(options: ServerOptions) -> Result<(), Error> {
    let config = Config::open(&options.config).map_err(Box::new)?;
    // The server's log is its stdout: journald and `docker logs` read it there.
    crate::logging::install(
        &config.log,
        std::io::stdout,
        std::io::stdout().is_terminal(),
    )?;
    let server_config = config.server;

    let tls = server_config
        .tls
        .as_ref()
        .map(crate::server::tls::build_acceptor)
        .transpose()?;

    raise_open_file_limit();

    let workers = config
        .workers
        .unwrap_or_else(dispatch::default_worker_count);
    info!(workers, "initialising dispatch");
    crate::network::limit_remote_reads_to_network_bandwidth(config.network_bandwidth);
    let disk_cache = config
        .disk_cache
        .as_ref()
        .map(|cache| {
            info!(
                dir = %cache.dir.display(),
                bytes = cache.size.as_bytes(),
                max_objects = cache.max_objects,
                "disk cache enabled"
            );
            cache.open()
        })
        .transpose()?;
    // Opening the directory checks that a spill file works in it, here, once:
    // a directory it does not work in stops the server from starting, rather
    // than failing the first query that spills.
    let spill_directory = match config.spill_directory {
        Some(directory) => {
            let spill_directory = SpillDirectory::open(directory)?;
            info!(dir = %spill_directory.path().display(), "spilling enabled");
            Some(spill_directory)
        }
        None => {
            info!(
                "no `spill_directory` configured: a query that needs more memory than the \
                 buffer pool has fails instead of spilling"
            );
            None
        }
    };
    let pool_bytes = config.memory.resolve(get_total_memory() as u64)? as usize;
    // The pool is a share of *total* memory, but every one of its slots is
    // faulted in while the workers start, so what it has to fit into is what the
    // machine has free. Asking for more than that is not a slower server: it is
    // an OOM kill part way through boot, which leaves nobody around to say why.
    let available_bytes = get_available_memory();
    info!(pool_bytes, available_bytes, "buffer pool memory budget");
    if pool_bytes > available_bytes {
        return Err(Error::InsufficientMemory {
            requested_bytes: pool_bytes,
            available_bytes,
        });
    }
    // Reading the executable in is disk-bound and the pool prefault is
    // memory-bound, so the two overlap instead of adding up.
    let mapped_files = std::thread::spawn(populate_mapped_files);
    let dispatch = Dispatch::spin_up_with_spill_directory(
        workers,
        pool_bytes / BUFFER_SIZE,
        disk_cache,
        spill_directory,
    );
    mapped_files
        .join()
        .expect("populating mapped files panicked");

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()?;

    runtime.block_on(async move {
        let metastore = build_metastore(
            config.entries,
            &options.config,
            config.metastore,
            config.datastore_refresh_interval.as_duration(),
        )?;
        let catalog = build_catalog(&options.config, &metastore, dispatch.dispatcher())?;
        let mut server = Server::new(server_config.bind, dispatch, catalog, metastore)?;
        if let Some(acceptor) = tls {
            server = server.with_tls(acceptor);
        }
        server.serve(Box::pin(wait_for_shutdown_signal())).await
    })
}
