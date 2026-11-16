//! `server` exposes pivotdb over the PostgreSQL v3 wire protocol so any
//! Postgres client (`psql`, `tokio-postgres`, JDBC, etc.) can connect and
//! issue queries.
//!
//! The crate is a thin glue layer: [`pgwire`] drives the wire protocol,
//! [`planner`] turns each SQL string into an executable plan against a
//! caller-supplied [`PivotCatalog`](catalog::PivotCatalog), and [`dispatch`]
//! runs the resulting dataflow on its thread-per-core worker pool. Each query
//! hops to `tokio::task::spawn_blocking` to drive the (non-`Send`) DuckDB
//! planner; the planner is cached in a thread-local on each blocking-pool
//! thread and reused across queries. [`run`] provides the configured foreground
//! process used by `pivot server`.
//!
//! The public interface: hand a bind address to [`Server::new`] together with a
//! [`Dispatch`](dispatch::Dispatch) (from
//! [`Dispatch::spin_up`](dispatch::Dispatch::spin_up)), your catalog, and the
//! metastore that produced it; then call [`Server::serve`] with a shutdown
//! future. Add [`Server::with_tls`] to let clients encrypt their connections
//! (see the [`tls`] module). The returned future runs the accept loop until
//! shutdown is signalled or a worker dies. Each datastore self-manages its own refresh and (optional)
//! compaction, re-encoding Parquet on the same dispatch workers; those tasks
//! watch the shared exit flag the server flips on shutdown and stop before the
//! workers do.
//!
//! # Example
//!
//! ```no_run
//! use std::sync::Arc;
//!
//! use catalog::PivotCatalog;
//! use dispatch::Dispatch;
//! use catalog::metastore::Metastore;
//! use metastore_disk::DiskMetastore;
//! use bin::server::{Config, Server};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! let config = Config::open("pivot.yaml")?;
//! let workers = config.workers.unwrap_or_else(dispatch::default_worker_count);
//! let dispatch = Dispatch::spin_up(workers, 32, None);
//! let metastore = Arc::new(DiskMetastore::open(
//!     config.entries,
//!     None,
//!     config.datastore_refresh_interval.as_duration(),
//! )?);
//! let catalog = Arc::new(PivotCatalog::new(
//!     metastore.open_datastores(dispatch.dispatcher())?,
//!     metastore.default_datastore_name().to_string(),
//!     metastore.clone(),
//! )?.with_external_parquet_read_context(
//!     dispatch.dispatcher(),
//!     metastore.external_store_factory(),
//! ));
//!
//! let server = Server::new(config.server.bind, dispatch, catalog, metastore)?;
//! // Returns when ctrl_c fires, or earlier if a dispatch worker dies.
//! server.serve(Box::pin(async {
//!     let _ = tokio::signal::ctrl_c().await;
//! })).await?;
//! # Ok(())
//! # }
//! ```

mod arrow_to_pgwire;
mod auth;
pub mod config;
mod copy_session;
mod limits;
mod listener;
mod mapped_files;
mod peer_hang_up;
#[cfg(feature = "perf")]
mod perf;
mod query_handler;
mod runtime;
pub mod tls;

pub use config::Config;
pub use limits::raise_open_file_limit;
pub use listener::{Error, Server};
pub use runtime::{ServerOptions, run};
