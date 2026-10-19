//! A generic key→bytes object store — local filesystem, S3, or GCS — and nothing
//! catalog-specific. It knows how to `get`/`put`/`list`/`delete` objects and
//! turn a key into a ring-readable [`DataFile`]; the table manifest,
//! table log, and catalog build on it one layer up.
//!
//! Everything here is **synchronous** and pulls in no async runtime: local
//! access is plain `std::fs`; S3 and GCS go over [`ureq`] (blocking HTTP +
//! rustls). S3 requests are signed with `aws_sigv4::http_request::sign` — a pure
//! function we call inline (the tokio it transitively links is never driven);
//! GCS requests carry an OAuth2 bearer token. Credentials come from whoever
//! opened the store: a metastore hands them in per datastore, and [`open_store`]
//! falls back to the environment. [`StoreConnection`] exposes those resolved
//! settings to format-specific adapters without making this crate depend on
//! them. S3 falls back to unsigned anonymous requests when no credentials are
//! available. This runs off the io_uring ring on purpose: a
//! LIST isn't a range-GET the ring can serve, and it's rare and tiny (a few KB
//! per query) next to the hot column-chunk reads, which stay on the ring.

use dispatch::io::{AuthHeader, OpenFile, RemoteFile, open_direct_read};
use std::fmt::Debug;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

pub mod file_injector;
mod file_slice_injector;
mod gcs;
mod local;
mod object_fetcher;
mod object_path;
mod s3;
#[cfg(feature = "test-support")]
pub mod test_support;
pub use gcs::GcsStore;
pub use local::LocalStore;
pub use object_fetcher::{FileSlice, LoadedObject, load_file_slices, load_objects};
pub use object_path::ObjectPath;
pub use s3::{S3Credentials, S3Keys, S3Store};

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("io error on `{key}`: {source}")]
    Io {
        key: String,
        #[source]
        source: std::io::Error,
    },
    #[error("http error talking to object store: {0}")]
    Http(String),
    #[error("unsupported catalog uri `{0}` (expected a local path, file://, s3://, or gs://)")]
    UnsupportedUri(String),
    #[error("missing credential/config: {0}")]
    Config(String),
    /// The bucket is served from another region than the one the store was
    /// configured with, so S3 redirected the request instead of serving it.
    #[error("S3 bucket `{bucket}` is in region `{actual}`, not the configured `{configured}`")]
    WrongRegion {
        bucket: String,
        configured: String,
        actual: String,
    },
    /// A conditional replace lost to a concurrent writer: the object changed
    /// between the read and the swap. The conditional-update path retries on
    /// it, so it surfaces only once those retries are exhausted.
    #[error("object `{key}` was changed by another writer")]
    VersionConflict { key: String },
}

pub type Result<T> = std::result::Result<T, StoreError>;

/// A table's data file: its [`ObjectPath`] in the store and its size in bytes.
/// A durable file identity. [`ObjectStore::list`] returns these, while the size
/// lets a reader locate a footer without a separate HEAD/`stat`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FileRef {
    pub path: ObjectPath,
    pub size: u64,
}

/// A listed object paired with its storage modification time (Unix
/// milliseconds): the backend's own last-modified stamp — a filesystem mtime, or
/// S3's `LastModified`. Kept out of [`FileRef`] because that timestamp is a
/// backend clock reading, not part of a file's durable identity; only vacuum's
/// orphan detection needs it, where an unreferenced file's mtime is the sole
/// "how new is this" signal for deciding it is safe to delete.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ListedObject {
    pub file: FileRef,
    pub modified_unix_ms: u64,
}

/// One level of an object-store namespace. `objects` are the files directly
/// under the requested prefix; `prefixes` are its immediate child prefixes,
/// each named relative to that same requested prefix.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DirectoryListing {
    pub objects: Vec<ListedObject>,
    pub prefixes: Vec<ObjectPath>,
}

impl FileRef {
    /// Turn this ref into a [`DataFile`] to read it for a table at `location`:
    /// its path is relative to the table's location (an absolute one escapes to
    /// the store root), so resolve it against `location` for the read source —
    /// keeping the ref itself as the identity that flows onto the `TableFile` and
    /// matches the manifest.
    pub fn into_data_file(
        self,
        store: &dyn ObjectStore,
        location: &ObjectPath,
    ) -> Result<DataFile> {
        let source = store.source(&location.resolve(&self.path))?;
        Ok(DataFile { file: self, source })
    }
}

/// A [`FileRef`] located for reading: its identity (`file`, whose size locates
/// the footer without a `stat`/HEAD) plus where its bytes live (`source`).
/// Produced transiently by [`FileRef::into_data_file`] and consumed straight by
/// the metadata fetcher, which stamps the `file` onto the `TableFile` it emits —
/// so the file's identity travels with its bytes through the load.
#[derive(Clone, Debug)]
pub struct DataFile {
    pub file: FileRef,
    pub source: DataFileLocation,
}

/// Where a data file's bytes live, for reading or writing: a local filesystem
/// path (via the io_uring file path) or a remote URL (HTTP on the same ring). A
/// remote URL either carries its own auth (an S3 presigned URL), needs no auth
/// (anonymous S3), or pairs a stable URL with an [`AuthHeader`] that mints a
/// fresh bearer token per request. Which variant a store yields is its business,
/// not the caller's.
///
/// The same key maps to a different URL per direction — [`source`](ObjectStore::source)
/// builds the read URL, [`sink`](ObjectStore::sink) the write URL (e.g. S3 presigns
/// a GET vs a PUT) — so both return this one type. Uploads are always a `PUT`.
#[derive(Clone)]
pub enum DataFileLocation {
    Local(PathBuf),
    Remote {
        url: url::Url,
        auth: Option<AuthHeader>,
    },
}

impl DataFileLocation {
    /// Open this read location as a ring-readable [`OpenFile`]: a local path
    /// becomes an `O_DIRECT` fd, a remote URL a [`RemoteFile`] (`size` locates
    /// its footer tail with no HEAD probe). Only valid once the file exists, so
    /// callers open it after the listing that found it or the upload that wrote
    /// it. This is the read (`GET`) counterpart; a `sink()` location is a `PUT`.
    pub fn open_read(self, size: u64) -> std::io::Result<OpenFile> {
        Ok(match self {
            DataFileLocation::Local(path) => OpenFile::Local(open_direct_read(&path)?),
            DataFileLocation::Remote { url, auth } => {
                OpenFile::Remote(Arc::new(RemoteFile::open(url, auth, size)?))
            }
        })
    }
}

impl Debug for DataFileLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Local(path) => f.debug_tuple("Local").field(path).finish(),
            Self::Remote { url, auth } => f
                .debug_struct("Remote")
                .field("url", url)
                .field("auth", &auth.is_some())
                .finish(),
        }
    }
}

impl DataFile {
    /// A data file on the local filesystem; its [`FileRef`] path is the
    /// filesystem path (these constructors feed the whole-directory readers,
    /// where the path is the file's identity directly).
    pub fn local(path: PathBuf, size: u64) -> Self {
        Self {
            file: FileRef {
                path: ObjectPath::new(path.to_string_lossy().into_owned()),
                size,
            },
            source: DataFileLocation::Local(path),
        }
    }

    /// A data file at a self-authenticating remote URL (no per-request auth); its
    /// [`FileRef`] path is the whole URL, so files on different hosts stay
    /// distinct.
    pub fn remote(url: url::Url, size: u64) -> Self {
        Self {
            file: FileRef {
                path: ObjectPath::new(url.as_str()),
                size,
            },
            source: DataFileLocation::Remote { url, auth: None },
        }
    }
}

/// A flat key→bytes object store rooted at one database. [`ObjectPath`] keys are
/// relative to that root, e.g. `_pivot_manifest.json` or `events/a.parquet`.
pub trait ObjectStore: Debug + Send + Sync {
    /// The filesystem directory backing this store, or `None` for a remote
    /// store. The datastore layer uses this to enforce exclusive ownership of
    /// local roots without putting that policy in the storage adapter itself.
    fn local_root(&self) -> Option<&Path> {
        None
    }

    /// Fetch an object in full, or `None` if it does not exist.
    fn get(&self, key: &ObjectPath) -> Result<Option<Vec<u8>>>;

    /// Atomically replace `key` with `data` (overwriting any existing object).
    /// Reads see either the old or the new object whole, never a torn write.
    /// Concurrent writers are last-writer-wins, so this backs objects with a
    /// single writer (a table's data files); a shared mutable document goes
    /// through [`update`](Self::update) instead.
    fn put(&self, key: &ObjectPath, data: &[u8]) -> Result<()>;

    /// Atomically read-modify-write the object at `key`, so no concurrent
    /// update through this method — from this or any other process — is ever
    /// lost: `apply` receives the current object (`None` if it does not exist)
    /// and returns the bytes to replace it with, or `None` to leave the store
    /// untouched.
    ///
    /// A local backend holds an exclusive advisory file lock across the whole
    /// cycle, so two writers never interleave. A remote backend replaces the
    /// object with a version-conditional PUT (compare-and-swap on its ETag /
    /// generation) and re-runs `apply` against the fresh object when a
    /// concurrent writer wins the swap — so `apply` must tolerate running more
    /// than once. Backs a small, rarely-written control document like the
    /// database manifest.
    fn update(
        &self,
        key: &ObjectPath,
        apply: &mut dyn FnMut(Option<Vec<u8>>) -> Option<Vec<u8>>,
    ) -> Result<()>;

    /// Delete `key`. Deleting an object that does not exist is not an error —
    /// the caller's goal (key absent) is already met.
    fn delete(&self, key: &ObjectPath) -> Result<()>;

    /// List one level under `prefix`: direct objects and immediate child
    /// prefixes, all named relative to `prefix`.
    fn list(&self, prefix: &ObjectPath) -> Result<DirectoryListing> {
        self.list_with_name_prefix(prefix, "")
    }

    /// [`list`](Self::list), narrowed to children whose own name starts with
    /// `name_prefix`. A remote backend pushes the narrowing into the request,
    /// so a narrow listing does not page through the whole directory.
    fn list_with_name_prefix(
        &self,
        prefix: &ObjectPath,
        name_prefix: &str,
    ) -> Result<DirectoryListing>;

    /// How the io_uring reader should fetch object `key`: a local backend yields
    /// a filesystem path, a remote one a GET URL — either presigned or paired
    /// with a per-request bearer token. (Identity — the [`FileRef`] — is the
    /// caller's; this is only how to read the bytes.)
    fn source(&self, key: &ObjectPath) -> Result<DataFileLocation>;

    /// How a worker should asynchronously create a data object at `key`.
    fn sink(&self, key: &ObjectPath) -> Result<DataFileLocation>;

    /// Prime any credentials workers will need for writes. Called on the
    /// coordinator before launching an INSERT dataflow.
    fn prepare_write(&self) -> Result<()> {
        Ok(())
    }

    /// Ensure the directory named by `prefix` exists so a writer can create
    /// objects under it. A local filesystem backend must create it, because a
    /// writer that addresses the prefix as a real path fails if its directory is
    /// missing.
    fn create_dir(&self, prefix: &ObjectPath) -> Result<()>;

    /// The same object addressed from the store's own root (the filesystem
    /// root, or the bucket root) instead of from the database root, as an
    /// [absolute](ObjectPath::is_absolute) key. A reference recorded this way
    /// resolves the same from anywhere in the store, which is how a table names
    /// a data file that sits outside its own location.
    fn absolute_key(&self, key: &ObjectPath) -> Result<ObjectPath>;

    /// A human-readable description of where this store is rooted - e.g.
    /// `file:///var/lib/pivot`, `s3://bucket/prefix`, or `gs://bucket/prefix`.
    /// Purely for diagnostics (saying whether a table lives on local disk or
    /// object storage); never an addressable key.
    /// The default falls back to the backend's `Debug` form.
    fn describe(&self) -> String {
        format!("{self:?}")
    }

    /// The machine-addressable URI this store is rooted at — `file:///path` for
    /// a local store, the `s3://bucket/prefix` or `gs://bucket/prefix` it was
    /// opened with for a remote one. This
    /// is what a durable reference to the store's contents is derived from, so
    /// it must be a parseable URL whose
    /// path includes the store's key prefix; contrast [`describe`](Self::describe),
    /// which is free-form text for humans. Required (no default): a backend
    /// cannot silently fall back to something unaddressable.
    fn location_uri(&self) -> String;

    /// The resolved connection settings behind this store. A consumer that
    /// needs a separate client implementation can build it from the same root
    /// and credentials without adding that client's dependencies here.
    fn connection(&self) -> StoreConnection;
}

/// Resolved backend settings, shared with adapters that need to address the
/// same store through another client implementation.
pub enum StoreConnection {
    Local,
    S3 {
        uri: String,
        region: String,
        credentials: Option<S3Credentials>,
        endpoint: Option<String>,
    },
    Gcs {
        uri: String,
        credentials_file: Option<String>,
        emulator_endpoint: Option<String>,
    },
}

/// Opens the store root needed by an external table-function invocation. A
/// provider resolves credentials against `root_uri`, the longest non-wildcard
/// parent that the returned store can address.
pub trait ExternalStoreFactory: Debug + Send + Sync {
    fn open(&self, root_uri: &str) -> Result<Arc<dyn ObjectStore>>;
}

/// External store opener for embedded datastores. Credentials come from the
/// same ambient environment chain as [`open_store`].
#[derive(Debug, Default)]
pub struct AmbientExternalStoreFactory;

impl ExternalStoreFactory for AmbientExternalStoreFactory {
    fn open(&self, root_uri: &str) -> Result<Arc<dyn ObjectStore>> {
        Ok(Arc::from(open_store(root_uri)?))
    }
}

/// The backend a location URI addresses. The backend is inferred from the
/// scheme rather than configured, so this is the one place that reads one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StoreScheme {
    S3,
    Gcs,
    Local,
}

impl StoreScheme {
    /// The backend `uri` addresses. `s3://` and its `s3a://` spelling are the
    /// same backend over the same buckets; a URI carrying no scheme at all is a
    /// filesystem path, which `file://` is the explicit spelling of.
    ///
    /// A URI whose scheme names no backend is an error rather than a local
    /// path: a mistyped or unsupported scheme would otherwise open a directory
    /// named after the URI, and the datastore would come up empty where its
    /// data was expected.
    pub fn of(uri: &str) -> Result<Self> {
        match uri.split_once("://") {
            Some(("s3" | "s3a", _)) => Ok(Self::S3),
            Some(("gs", _)) => Ok(Self::Gcs),
            Some(("file", _)) => Ok(Self::Local),
            Some(_) => Err(StoreError::UnsupportedUri(uri.to_string())),
            None => Ok(Self::Local),
        }
    }

    /// The URI prefix a location of this backend is written with, and the form
    /// one is printed back in.
    pub fn uri_prefix(self) -> &'static str {
        match self {
            Self::S3 => "s3://",
            Self::Gcs => "gs://",
            Self::Local => "file://",
        }
    }
}

/// The filesystem path a [local](StoreScheme::Local) location names: `file://`
/// is an optional spelling of a plain path, not a backend of its own.
pub fn local_path(uri: &str) -> &str {
    // Java writes a local URI without the authority, as `file:/path`.
    uri.strip_prefix("file://")
        .or_else(|| uri.strip_prefix("file:"))
        .unwrap_or(uri)
}

/// A remote backend's own token for one stored version of an object — an S3
/// ETag, a GCS generation — presented back on a conditional replace so the swap
/// fails (instead of clobbering) when another writer got there first.
#[derive(Debug, Clone)]
pub(crate) struct ObjectVersion(pub(crate) String);

/// How many times a remote backend re-runs its read-apply-swap cycle when a
/// concurrent writer swaps the object first. A retry normally follows someone
/// else's *completed* write, so the update converges under any realistic
/// contention; the cap turns a backend that reports conflicts forever into an
/// error instead of a spin.
const VERSION_SWAP_ATTEMPTS: u32 = 10;
/// Full-jitter exponential backoff slept between swap attempts, doubling from
/// the base up to the cap: racing writers decorrelate instead of re-colliding,
/// and a conflict reported while a concurrent conditional write is still
/// settling (S3's 409) gets time to clear before the next attempt.
const VERSION_SWAP_BACKOFF_BASE: Duration = Duration::from_millis(10);
const VERSION_SWAP_BACKOFF_CAP: Duration = Duration::from_secs(1);

/// A sleep drawn from `[0, ceiling)` to decorrelate retries. Seeded from the
/// clock's subsecond nanoseconds: not statistically random, but plenty to keep
/// two writers that just conflicted from retrying in lockstep.
fn jittered(ceiling: Duration) -> Duration {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .subsec_nanos() as u64;
    Duration::from_nanos(nanos % (ceiling.as_nanos() as u64).max(1))
}

/// A remote backend's optimistic [`ObjectStore::update`]: `read` the object
/// with its version token, `apply` the modification, and `swap` the result in
/// conditionally on that token (`None`: the object must not exist). A swap that
/// loses to a concurrent writer re-runs the cycle against the fresh object,
/// after a jittered backoff. Blocking sleeps, like every store call here: this
/// runs on the blocking pool or control thread, never on the reactor.
pub(crate) fn update_by_version_swap(
    read: impl Fn() -> Result<Option<(Vec<u8>, ObjectVersion)>>,
    swap: impl Fn(&[u8], Option<&ObjectVersion>) -> Result<()>,
    apply: &mut dyn FnMut(Option<Vec<u8>>) -> Option<Vec<u8>>,
) -> Result<()> {
    let mut conflict = None;
    for attempt in 0..VERSION_SWAP_ATTEMPTS {
        let (bytes, version) = match read()? {
            Some((bytes, version)) => (Some(bytes), Some(version)),
            None => (None, None),
        };
        let Some(replacement) = apply(bytes) else {
            return Ok(());
        };
        match swap(&replacement, version.as_ref()) {
            Ok(()) => return Ok(()),
            Err(error @ StoreError::VersionConflict { .. }) => conflict = Some(error),
            Err(error) => return Err(error),
        }
        if attempt + 1 < VERSION_SWAP_ATTEMPTS {
            let ceiling = VERSION_SWAP_BACKOFF_BASE
                .saturating_mul(1 << attempt)
                .min(VERSION_SWAP_BACKOFF_CAP);
            std::thread::sleep(jittered(ceiling));
        }
    }
    Err(conflict.expect("a retry only follows a recorded conflict"))
}

/// Open the object store for a catalog root URI: `s3://bucket/prefix`,
/// `gs://bucket/prefix`, or a local path (optionally `file://`). Credentials are
/// whatever the process itself can resolve. S3 uses anonymous access when both
/// key variables are absent; a caller holding its own opens the backend directly
/// (`S3Store::with_credentials`, `GcsStore::with_credentials_file`).
pub fn open_store(uri: &str) -> Result<Box<dyn ObjectStore>> {
    match StoreScheme::of(uri)? {
        StoreScheme::S3 => Ok(Box::new(S3Store::with_env_credentials(uri)?)),
        StoreScheme::Gcs => Ok(Box::new(GcsStore::with_default_credentials(uri)?)),
        StoreScheme::Local => Ok(Box::new(LocalStore::new(local_path(uri))?)),
    }
}

/// The final path segment of a raw object-store key string (a backend's listing
/// response), as the manifest records the object's name.
pub(crate) fn key_name(key: &str) -> String {
    key.rsplit('/').next().unwrap_or(key).to_string()
}

/// Strip the store/list prefix from a recursively listed backend key, yielding
/// the path callers use relative to the requested listing root.
pub(crate) fn relative_key(prefix: &str, key: &str) -> String {
    let prefix = prefix.trim_matches('/');
    if prefix.is_empty() {
        return key.trim_start_matches('/').to_string();
    }
    key.strip_prefix(prefix)
        .unwrap_or(key)
        .trim_start_matches('/')
        .to_string()
}

/// The in-bucket object key a remote backend should address for a store key.
/// An [absolute](ObjectPath::is_absolute) key is taken from the bucket root,
/// ignoring `prefix` (the database's own prefix within the bucket); any other
/// key lives under `prefix`.
pub(crate) fn object_key(prefix: &str, key: &ObjectPath) -> String {
    if key.is_absolute() {
        key.as_str().trim_start_matches('/').to_string()
    } else {
        let prefix = prefix.trim_matches('/');
        if prefix.is_empty() {
            key.as_str().to_string()
        } else if key.is_empty() {
            prefix.to_string()
        } else {
            format!("{prefix}/{}", key.as_str())
        }
    }
}

/// Percent-encode one object-key component for a URL path segment or query
/// value per RFC 3986: unreserved characters pass through, everything else
/// (including `/`) is escaped.
pub(crate) fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for &byte in s.as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char)
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

/// Parse an object store's listing timestamp (RFC 3339, always UTC — S3's
/// `LastModified`, GCS's `updated`) into Unix milliseconds. Returns `None` on
/// any malformed field, which the caller turns into a listing error rather than
/// a silently-wrong (too-old) timestamp.
pub(crate) fn parse_iso8601_millis(s: &str) -> Option<u64> {
    let nanos = arrow_cast::parse::string_to_timestamp_nanos(s).ok()?;
    u64::try_from(nanos / 1_000_000).ok()
}

/// The bucket-root-absolute form of a store key, for a backend whose keys live
/// under an in-bucket `prefix`: the object's own name in the bucket, marked
/// absolute so reading it back does not put the database's prefix in front of it
/// a second time. Shared by every bucket backend so they all spell the key the
/// same way.
pub(crate) fn absolute_object_key(prefix: &str, key: &ObjectPath) -> ObjectPath {
    ObjectPath::new(format!("/{}", object_key(prefix, key)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absolute_object_key_is_the_in_bucket_name_from_the_bucket_root() {
        // A key under the database's prefix keeps that prefix, addressed from
        // the bucket root rather than from the database.
        assert_eq!(
            absolute_object_key("mydb", &ObjectPath::new("events/a.parquet")).as_str(),
            "/mydb/events/a.parquet"
        );
        // One that already escapes the prefix is unchanged by the round trip.
        assert_eq!(
            absolute_object_key("mydb", &ObjectPath::new("/shared/a.parquet")).as_str(),
            "/shared/a.parquet"
        );
    }

    #[test]
    fn object_key_relative_lives_under_prefix_absolute_at_bucket_root() {
        // Relative keys hang under the database's own prefix in the bucket.
        assert_eq!(
            object_key("mydb", &ObjectPath::new("events/a.parquet")),
            "mydb/events/a.parquet"
        );
        // An absolute key escapes the database prefix to the bucket root.
        assert_eq!(
            object_key("mydb", &ObjectPath::new("/shared/a.parquet")),
            "shared/a.parquet"
        );
        // Bucket root with no database prefix configured behaves the same.
        assert_eq!(
            object_key("", &ObjectPath::new("events/a.parquet")),
            "events/a.parquet"
        );
        assert_eq!(
            object_key("", &ObjectPath::new("/shared/a.parquet")),
            "shared/a.parquet"
        );
        // The empty relative key names the configured store root itself.
        assert_eq!(object_key("mydb", &ObjectPath::default()), "mydb");
    }

    #[test]
    fn version_swap_update_reapplies_over_a_concurrent_winner() {
        use std::cell::{Cell, RefCell};
        // A fake remote object: (bytes, version). The first swap loses to a
        // concurrent writer that lands "7" and bumps the version.
        let stored = RefCell::new((b"1".to_vec(), 1u64));
        let conflicts = Cell::new(1);
        let read = || {
            let stored = stored.borrow();
            Ok(Some((
                stored.0.clone(),
                ObjectVersion(stored.1.to_string()),
            )))
        };
        let swap = |data: &[u8], expected: Option<&ObjectVersion>| {
            let mut stored = stored.borrow_mut();
            if conflicts.get() > 0 {
                conflicts.set(conflicts.get() - 1);
                *stored = (b"7".to_vec(), stored.1 + 1);
                return Err(StoreError::VersionConflict { key: "k".into() });
            }
            assert_eq!(expected.unwrap().0, stored.1.to_string());
            *stored = (data.to_vec(), stored.1 + 1);
            Ok(())
        };

        update_by_version_swap(read, swap, &mut |current| {
            let value: u64 = String::from_utf8(current.unwrap())
                .unwrap()
                .parse()
                .unwrap();
            Some((value + 1).to_string().into_bytes())
        })
        .unwrap();

        // The increment re-applied over the winner's "7", not over the stale "1".
        assert_eq!(stored.borrow().0, b"8");
    }

    #[test]
    fn open_store_routes_local_and_file_uri() {
        let dir = tempfile::tempdir().unwrap();
        let store = open_store(dir.path().to_str().unwrap()).unwrap();
        let key = ObjectPath::new("k");
        store.put(&key, b"v").unwrap();
        // Reopening through a `file://` URI lands on the same root.
        let uri = format!("file://{}", dir.path().to_str().unwrap());
        let reopened = open_store(&uri).unwrap();
        assert_eq!(reopened.get(&key).unwrap().unwrap(), b"v");
    }

    #[test]
    fn open_store_rejects_a_uri_of_an_unknown_scheme() {
        let error = open_store("blob://bucket/prefix").unwrap_err();

        assert!(matches!(&error, StoreError::UnsupportedUri(uri) if uri == "blob://bucket/prefix"));
    }

    #[test]
    fn a_relative_path_is_a_local_store() {
        assert_eq!(StoreScheme::of("data/warm").unwrap(), StoreScheme::Local);
    }
}
