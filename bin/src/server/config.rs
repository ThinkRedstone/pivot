//! The config file: one YAML file describing a whole instance.
//!
//! The top level is the instance: its memory and worker budgets, spill
//! directory, refresh cadence, disk cache and log. `server` is
//! [`ServerConfig`] below: the endpoint.
//! `datastores`, `secrets` and `users` are the operator's entries, a
//! [`MetastoreConfig`] written at the top level of the file: read at startup
//! and never rewritten by the server. `metastore` is [`MetastoreStore`]: the
//! store the server writes to, a file of the same three maps that this process
//! owns, where `CREATE USER` lands. All of it is read in a single pass, so a
//! mistake anywhere is reported with its place in the file and stops startup.
//!
//! ```yaml
//! memory: 32g
//! workers: 16
//! spill_directory: /var/lib/pivot/spill
//! network_bandwidth: 50gbit
//! datastore_refresh_interval: 30s
//! log: info
//! disk_cache:
//!   dir: /var/cache/pivot
//!   size: 64g
//!
//! server:
//!   bind: 0.0.0.0:5432
//!   tls:
//!     cert: /etc/pivot/server.crt
//!     key: /etc/pivot/server.key
//!
//! datastores:
//!   hot:
//!     kind: pivotlake
//!     location: /var/lib/pivot/datastores/hot
//!     default: true
//!
//! users:
//!   pivot:
//!     auth:
//!       method: trust
//!
//! metastore:
//!   kind: file
//!   path: /var/lib/pivot/metastore.yaml
//! ```
//!
//! Every instance setting and every field of `server` has a default, so any of
//! them, or the whole `server` section, may be left out. So may `metastore`:
//! the server then serves the file's own entries and has nowhere to write,
//! which `CREATE USER` reports. Unknown keys are rejected rather than ignored:
//! a misspelled setting would otherwise leave the instance running on a
//! default nobody asked for.

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use datastore_pivotlake::DEFAULT_REFRESH_INTERVAL;
use dispatch::io::DiskCache;
use metastore_disk::{
    ByteSize, DatastoreConfig, Interval, MetastoreConfig, SecretConfig, UserConfig,
};
use serde::Deserialize;

use crate::memory::MemoryBudget;
use crate::network::Bandwidth;

/// The address the PostgreSQL endpoint binds to when `bind` is not set. Loopback
/// so an unconfigured server is not exposed to the network.
const DEFAULT_BIND: SocketAddr = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 5432);

/// The disk cache's byte budget when only `dir` is given.
const DEFAULT_DISK_CACHE_SIZE: ByteSize = ByteSize::from_bytes(64 * 1024 * 1024 * 1024);

/// The disk cache's object-count limit when only `dir` is given.
const DEFAULT_DISK_CACHE_MAX_OBJECTS: usize = 65536;

/// A parsed config file.
///
/// Unknown top-level keys are rejected: a section under a misspelled name would
/// otherwise leave the whole instance on defaults without a word about it.
///
/// The fields that are `Option` here are the ones whose default is a property
/// of the machine (its core count) rather than a constant, so the binary
/// resolves them at startup instead of baking them in here.
#[derive(Deserialize)]
#[serde(from = "RawConfig")]
pub struct Config {
    /// Memory budget for the buffer pool: a size such as `32g`, or a share of
    /// the machine's total memory such as `80%` (the default) minus a reserve
    /// for allocations outside the pool, see [`MemoryBudget`].
    pub memory: MemoryBudget,
    /// Number of dispatch worker threads. Defaults to the machine's core count.
    pub workers: Option<usize>,
    /// Directory that a query needing more memory than the buffer pool has
    /// spills rows to. Omit it and nothing is spilled: such a query fails
    /// instead. The server checks at startup that spill files work in it.
    pub spill_directory: Option<PathBuf>,
    /// What the machine's network card may carry, such as `50gbit`: remote
    /// reads are requested no faster. Omitted, the server looks it up from the
    /// instance type it runs on.
    pub network_bandwidth: Option<Bandwidth>,
    /// How often every datastore brings itself up to date with its store:
    /// the tables it has, and each table's committed files. For shared remote
    /// stores, this bounds how stale a query's view of another process's
    /// commits can be; this process's own commits are visible immediately.
    pub datastore_refresh_interval: Interval,
    /// On-disk cache for remote (object store) reads. Omit to disable it; local
    /// files are never cached, they are read from the filesystem directly.
    pub disk_cache: Option<DiskCacheConfig>,
    /// What the process logs, as `tracing` filter directives: a level such as
    /// `info` or `debug` for everything, or per-target levels such as
    /// `info,dispatch=debug`. The server writes the log to stdout.
    pub log: String,
    /// The endpoint. Every field has a default, so the section may be left out
    /// entirely.
    pub server: ServerConfig,
    /// The operator's `datastores`, `secrets` and `users`, written at the top
    /// level of the file. Never rewritten by the server.
    pub entries: MetastoreConfig,
    /// The store the server writes to. Omitted, the server serves `entries`
    /// alone and refuses to create users.
    pub metastore: Option<MetastoreStore>,
}

/// The file as written: the instance settings and the three entry maps sit at
/// the top level beside the sections, and the maps are gathered into
/// [`Config::entries`] once parsed. Spelling them out here, rather than
/// flattening a [`MetastoreConfig`] in, is what lets a misspelled top-level
/// key be rejected.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawConfig {
    #[serde(default)]
    memory: MemoryBudget,
    workers: Option<usize>,
    spill_directory: Option<PathBuf>,
    network_bandwidth: Option<Bandwidth>,
    #[serde(default = "default_datastore_refresh_interval")]
    datastore_refresh_interval: Interval,
    disk_cache: Option<DiskCacheConfig>,
    #[serde(default = "default_log_filter")]
    log: String,
    #[serde(default)]
    server: ServerConfig,
    #[serde(default)]
    datastores: HashMap<String, DatastoreConfig>,
    #[serde(default)]
    secrets: HashMap<String, SecretConfig>,
    #[serde(default)]
    users: HashMap<String, UserConfig>,
    metastore: Option<MetastoreStore>,
}

fn default_datastore_refresh_interval() -> Interval {
    Interval::from_duration(DEFAULT_REFRESH_INTERVAL)
}

fn default_log_filter() -> String {
    crate::logging::DEFAULT_FILTER.to_string()
}

impl From<RawConfig> for Config {
    fn from(raw: RawConfig) -> Self {
        Self {
            memory: raw.memory,
            workers: raw.workers,
            spill_directory: raw.spill_directory,
            network_bandwidth: raw.network_bandwidth,
            datastore_refresh_interval: raw.datastore_refresh_interval,
            disk_cache: raw.disk_cache,
            log: raw.log,
            server: raw.server,
            entries: MetastoreConfig::new(raw.datastores, raw.secrets, raw.users),
            metastore: raw.metastore,
        }
    }
}

/// The `metastore` section: where the server keeps what it is told at runtime.
/// `kind` selects the store, as it selects a datastore's implementation.
#[derive(Debug, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
pub enum MetastoreStore {
    /// A YAML file of the same `datastores`, `secrets` and `users` maps as the
    /// config, owned and rewritten by this server process. It must exist: a
    /// path that is not there means users are missing, not that none were
    /// created yet.
    File { path: PathBuf },
}

impl Config {
    /// Read and parse the config file at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        // The lossy form names the file in errors; the read itself keeps the
        // original bytes, which a path need not spell in UTF-8.
        let displayed = path.display().to_string();
        let text = std::fs::read_to_string(path).map_err(|source| Error::Read {
            path: displayed.clone(),
            source,
        })?;
        Self::from_yaml(&text, &displayed)
    }

    /// Parse config from YAML text, using `path` in errors.
    fn from_yaml(text: &str, path: &str) -> Result<Self> {
        serde_yaml_ng::from_str(text).map_err(|source| Error::Parse {
            path: path.to_string(),
            source,
        })
    }
}

/// The `server` section: what only serving the instance has, the PostgreSQL
/// endpoint.
#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct ServerConfig {
    /// TCP socket the PostgreSQL endpoint binds to.
    pub bind: SocketAddr,
    /// Certificate the PostgreSQL endpoint presents to a client that asks to
    /// encrypt its connection. Omit to answer every such request with a refusal,
    /// leaving the endpoint plaintext-only.
    pub tls: Option<TlsConfig>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: DEFAULT_BIND,
            tls: None,
        }
    }
}

/// The `server.tls` section. Both fields are required: an endpoint that offers
/// encryption has to present a certificate, and a certificate is worth nothing
/// without the key that proves the server holds it.
///
/// Writing the section turns SSL on; it does not make it compulsory. A client
/// that asks to upgrade gets an encrypted session, and one that does not still
/// gets a plaintext one, which is how PostgreSQL's own `ssl = on` behaves.
#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TlsConfig {
    /// PEM file holding the certificate to present: the server's own first,
    /// followed by whatever intermediates a client needs to reach a root it
    /// trusts.
    pub cert: PathBuf,
    /// PEM file holding that certificate's private key, in any of PKCS#8,
    /// PKCS#1 or SEC1. Readable by the server's user and by nobody else.
    pub key: PathBuf,
}

/// The `disk_cache` section. `dir` is required: the section exists to enable
/// the cache, and a cache with no directory to live in cannot be one.
#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DiskCacheConfig {
    /// Directory the cached byte ranges are stored in. The contents persist
    /// across restarts.
    pub dir: PathBuf,
    /// Size budget for the cached bytes.
    #[serde(default = "default_disk_cache_size")]
    pub size: ByteSize,
    /// Maximum number of cached objects. This bounds the open file descriptors
    /// the cache holds, one per object, so keep it below the process's limit.
    #[serde(default = "default_disk_cache_max_objects")]
    pub max_objects: usize,
}

fn default_disk_cache_size() -> ByteSize {
    DEFAULT_DISK_CACHE_SIZE
}

fn default_disk_cache_max_objects() -> usize {
    DEFAULT_DISK_CACHE_MAX_OBJECTS
}

impl DiskCacheConfig {
    /// Open the cache the section describes, creating its directory.
    pub fn open(&self) -> Result<Arc<DiskCache>, DiskCacheError> {
        DiskCache::open(self.dir.clone(), self.size.as_bytes(), self.max_objects)
            .map(Arc::new)
            .map_err(|source| DiskCacheError {
                dir: self.dir.clone(),
                source,
            })
    }
}

/// The disk cache's directory could not be created or read.
#[derive(Debug, thiserror::Error)]
#[error("opening the disk cache at `{}`: {source}", .dir.display())]
pub struct DiskCacheError {
    dir: PathBuf,
    #[source]
    source: std::io::Error,
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("reading config file `{path}`: {source}")]
    Read {
        path: String,
        source: std::io::Error,
    },
    #[error(
        "parsing config file `{path}`: {source}\nthe file holds the instance's settings, a `server` section, the `datastores`, `secrets` and `users` maps, and a `metastore` section naming the file the server writes to"
    )]
    Parse {
        path: String,
        source: serde_yaml_ng::Error,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    use catalog::metastore::Metastore;
    use metastore_disk::DiskMetastore;

    const DATASTORES_SECTION: &str =
        "datastores:\n  hot:\n    kind: pivotlake\n    location: /tmp/hot\n    default: true\n";

    /// A file whose top level holds `settings`, on top of a minimal
    /// `datastores` map.
    fn from_yaml_with(settings: &str) -> Result<Config> {
        Config::from_yaml(&format!("{settings}{DATASTORES_SECTION}"), "test")
    }

    /// A file whose `server` section holds `settings`, on top of a minimal
    /// `datastores` map.
    fn from_yaml_with_server(settings: &str) -> Result<Config> {
        from_yaml_with(&format!("server:\n{settings}"))
    }

    #[test]
    fn example_config_parses() {
        let yaml = include_str!("../../config.example.yaml");

        let config = Config::from_yaml(yaml, "config.example.yaml").unwrap();

        assert_eq!(config.server.bind, "127.0.0.1:5432".parse().unwrap());
    }

    #[test]
    fn a_network_bandwidth_is_read_in_decimal_bits_per_second() {
        let config = from_yaml_with("network_bandwidth: 12.5gbit\n").unwrap();

        assert_eq!(
            config.network_bandwidth,
            Some(Bandwidth {
                bits_per_second: 12_500_000_000
            })
        );
    }

    #[test]
    fn a_network_bandwidth_without_a_unit_is_rejected() {
        let result = from_yaml_with("network_bandwidth: \"50\"\n");

        assert!(result.is_err());
    }

    #[test]
    fn one_file_configures_both_the_instance_and_its_data() {
        let config = from_yaml_with(
            "memory: 32g\nworkers: 8\nspill_directory: /var/lib/pivot/spill\n\
             datastore_refresh_interval: 5s\nserver:\n  bind: 0.0.0.0:5433\n",
        )
        .unwrap();
        let metastore = DiskMetastore::open(
            config.entries,
            None,
            config.datastore_refresh_interval.as_duration(),
        )
        .unwrap();

        assert_eq!(config.server.bind, "0.0.0.0:5433".parse().unwrap());
        assert_eq!(
            config.memory,
            MemoryBudget::Bytes(ByteSize::from_bytes(32 * 1024 * 1024 * 1024))
        );
        assert_eq!(config.workers, Some(8));
        assert_eq!(
            config.spill_directory,
            Some(PathBuf::from("/var/lib/pivot/spill"))
        );
        assert_eq!(
            config.datastore_refresh_interval.as_duration(),
            Duration::from_secs(5)
        );
        assert_eq!(metastore.default_datastore_name(), "hot");
    }

    #[test]
    fn an_omitted_setting_leaves_the_instance_on_its_default() {
        let config = Config::from_yaml(DATASTORES_SECTION, "test").unwrap();

        assert_eq!(config.server, ServerConfig::default());
        assert_eq!(config.server.bind, "127.0.0.1:5432".parse().unwrap());
        assert_eq!(config.memory, MemoryBudget::Percent(80));
        assert_eq!(config.workers, None);
        assert_eq!(config.spill_directory, None);
        assert_eq!(
            config.datastore_refresh_interval.as_duration(),
            DEFAULT_REFRESH_INTERVAL
        );
        assert_eq!(config.disk_cache, None);
    }

    #[test]
    fn memory_is_a_share_of_the_machine_by_percentage() {
        let config = from_yaml_with("memory: 50%\n").unwrap();

        assert_eq!(config.memory, MemoryBudget::Percent(50));
    }

    #[test]
    fn the_log_filter_defaults_to_info() {
        let by_default = Config::from_yaml(DATASTORES_SECTION, "test").unwrap();
        let level = from_yaml_with("log: debug\n").unwrap();
        let per_target = from_yaml_with("log: info,dispatch=debug\n").unwrap();

        assert_eq!(by_default.log, "info");
        assert_eq!(level.log, "debug");
        assert_eq!(per_target.log, "info,dispatch=debug");
    }

    #[test]
    fn a_misspelled_setting_is_rejected_rather_than_ignored() {
        let error = from_yaml_with("wrokers: 8\n").err().unwrap();

        assert!(
            error.to_string().contains("unknown field `wrokers`"),
            "{error}"
        );
    }

    #[test]
    fn an_instance_setting_under_the_server_section_is_rejected() {
        let error = from_yaml_with_server("  memory: 32g\n").err().unwrap();

        assert!(
            error.to_string().contains("unknown field `memory`"),
            "{error}"
        );
    }

    #[test]
    fn a_metastore_section_names_the_file_the_server_writes() {
        let yaml = format!(
            "{DATASTORES_SECTION}metastore:\n  kind: file\n  path: /var/lib/pivot/metastore.yaml\n"
        );

        let config = Config::from_yaml(&yaml, "test").unwrap();

        assert_eq!(
            config.metastore,
            Some(MetastoreStore::File {
                path: PathBuf::from("/var/lib/pivot/metastore.yaml")
            })
        );
    }

    #[test]
    fn an_omitted_metastore_section_leaves_the_server_nothing_to_write() {
        let config = Config::from_yaml(DATASTORES_SECTION, "test").unwrap();

        assert_eq!(config.metastore, None);
    }

    #[test]
    fn a_metastore_section_of_an_unknown_kind_is_rejected() {
        let yaml =
            format!("{DATASTORES_SECTION}metastore:\n  kind: postgres\n  url: postgres://x\n");

        let error = Config::from_yaml(&yaml, "test").err().unwrap();

        assert!(
            error.to_string().contains("unknown variant `postgres`"),
            "{error}"
        );
    }

    #[test]
    fn entries_written_under_the_metastore_section_are_rejected() {
        let yaml = "metastore:\n  datastores:\n    hot:\n      kind: pivotlake\n      \
                    location: /tmp/hot\n      default: true\n";

        let error = Config::from_yaml(yaml, "test").err().unwrap();

        assert!(
            error.to_string().contains("missing field `kind`"),
            "{error}"
        );
    }

    #[test]
    fn a_misspelled_section_is_rejected_rather_than_dropped() {
        let yaml = format!("sever:\n  bind: 0.0.0.0:5433\n{DATASTORES_SECTION}");

        let error = Config::from_yaml(&yaml, "test").err().unwrap();

        assert!(
            error.to_string().contains("unknown field `sever`"),
            "{error}"
        );
    }

    #[test]
    fn a_misspelled_server_setting_is_rejected_rather_than_ignored() {
        let error = from_yaml_with_server("  bnid: 0.0.0.0:5433\n")
            .err()
            .unwrap();

        assert!(
            error.to_string().contains("unknown field `bnid`"),
            "{error}"
        );
    }

    #[test]
    fn a_disk_cache_takes_its_budgets_from_defaults() {
        let config = from_yaml_with("disk_cache:\n  dir: /var/cache/pivot\n").unwrap();

        let cache = config.disk_cache.unwrap();
        assert_eq!(cache.dir, PathBuf::from("/var/cache/pivot"));
        assert_eq!(cache.size, DEFAULT_DISK_CACHE_SIZE);
        assert_eq!(cache.max_objects, DEFAULT_DISK_CACHE_MAX_OBJECTS);
    }

    #[test]
    fn a_tls_section_names_the_certificate_to_present() {
        let config = from_yaml_with_server(
            "  tls:\n    cert: /etc/pivot/server.crt\n    key: /etc/pivot/server.key\n",
        )
        .unwrap();

        let tls = config.server.tls.unwrap();
        assert_eq!(tls.cert, PathBuf::from("/etc/pivot/server.crt"));
        assert_eq!(tls.key, PathBuf::from("/etc/pivot/server.key"));
    }

    #[test]
    fn a_certificate_without_its_key_is_rejected() {
        let error = from_yaml_with_server("  tls:\n    cert: /etc/pivot/server.crt\n")
            .err()
            .expect("a certificate the server cannot prove it holds should be rejected");

        assert!(error.to_string().contains("missing field `key`"), "{error}");
    }

    #[test]
    fn a_disk_cache_without_a_directory_is_rejected() {
        let error = from_yaml_with("disk_cache:\n  size: 8g\n").err().unwrap();

        assert!(error.to_string().contains("missing field `dir`"), "{error}");
    }

    #[test]
    fn a_missing_file_names_itself() {
        let error = Config::open("/nonexistent/pivot.yaml").err().unwrap();

        assert!(
            matches!(&error, Error::Read { path, .. } if path == "/nonexistent/pivot.yaml"),
            "{error}"
        );
    }
}
