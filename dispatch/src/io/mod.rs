//! I/O subsystem.
//!
//! We use direct I/O (bypassing the kernel page cache) for a few reasons:
//!
//! 1. We need to distinguish real disk reads from memory-satisfied reads.
//!    Our execution engine pipelines IO and CPU work: while the disk is
//!    serving the next batch of pages, workers process already-resident data.
//!    If a "read" silently hits the page cache and returns instantly, the
//!    scheduler sees it as completed IO and eagerly submits more, flooding
//!    the pipeline with new record batches instead of letting workers drain
//!    the ones already in flight (hot in cache).
//!
//! 2. We want explicit control over memory residency and eviction policy,
//!    rather than competing with the OS for the same physical pages.
//!
//! Provides direct I/O reads with platform-specific backends (io_uring on Linux,
//! a blocking pread/pwrite thread pool on other Unix), reading straight into the
//! file cache's 4 KB-aligned slot regions.
//!
//! # Architecture
//!
//! - [`IORequester`] — submits and completes read requests on behalf of dataflows.
//! - [`backend::IOBackend`] — platform-specific submission/completion engine.

mod backend;
pub(crate) mod hardware_queues;

use crate::Identifier;
use crate::memory::FileBytes;
use crate::memory::compressed_cache::MissingExtent;
use std::fmt::{Debug, Formatter};
use std::fs::{File, OpenOptions};
use std::hash::{Hash, Hasher};
use std::net::{SocketAddr, ToSocketAddrs};
use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use url::Url;

mod requester;
pub use backend::RingWakeHandle;
pub use requester::{Error as IORequesterError, IORequester, RING_SIZE};

mod operator;
#[cfg(any(test, feature = "test-util"))]
pub use operator::TestOperatorIO;
pub use operator::{
    FileRange, OperatorIO, PendingReadRequest, PendingWriteRequest, ReadData, ReadRequestId,
    ReadResponse,
};

mod cached_http;

pub mod disk_cache;
pub use disk_cache::{DiskCache, clear_disk_cache};

pub mod http;

/// A local file descriptor together with the block-device index resolved when
/// it was opened. Cloning this wrapper keeps the descriptor alive without
/// repeating the device lookup on the I/O submission path.
#[derive(Clone)]
pub struct LocalFile {
    file: Arc<File>,
    device_idx: hardware_queues::DeviceIndex,
}

impl LocalFile {
    /// Wrap a newly opened file and register its block device once.
    pub fn new(file: File) -> std::io::Result<Self> {
        let file = Arc::new(file);
        let device_idx = hardware_queues::register_file_device(&file)?;
        Ok(Self { file, device_idx })
    }

    pub(crate) fn device_idx(&self) -> hardware_queues::DeviceIndex {
        self.device_idx
    }

    /// Attributes of the underlying file - notably its current size, which the
    /// requester consults to tell a read that stopped at end-of-file from one
    /// that genuinely came up short.
    pub(crate) fn metadata(&self) -> std::io::Result<std::fs::Metadata> {
        self.file.metadata()
    }

    #[cfg(test)]
    pub(crate) fn strong_count(&self) -> usize {
        Arc::strong_count(&self.file)
    }
}

impl AsRawFd for LocalFile {
    fn as_raw_fd(&self) -> std::os::fd::RawFd {
        self.file.as_raw_fd()
    }
}

/// An open file the engine can read: a local file (the [`LocalFile`] keeps the
/// descriptor alive for as long as anything — a table, an in-flight request, a
/// cached region — still references it), or a remote HTTP(S) object fetched
/// via range requests.
///
/// This is also the key the [`CompressedCache`](crate::memory::compressed_cache::CompressedCache)
/// uses to bucket 2 MB regions, so it must be cheap to `Hash`/`Eq` — the
/// cache's pin re-check runs on every hit. `Local` compares the raw fd (an
/// `i32`; stable while the `LocalFile` is held, and holding it in the key means
/// a cached file's fd can never be closed and reused under the cache); `Remote`
/// carries an [`Arc<RemoteFile>`] that compares equal to every other open of
/// the same object.
#[derive(Clone)]
pub enum OpenFile {
    Local(LocalFile),
    Remote(Arc<RemoteFile>),
}

impl PartialEq for OpenFile {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (OpenFile::Local(a), OpenFile::Local(b)) => a.as_raw_fd() == b.as_raw_fd(),
            (OpenFile::Remote(a), OpenFile::Remote(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for OpenFile {}

impl Hash for OpenFile {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            OpenFile::Local(file) => {
                0u8.hash(state);
                file.as_raw_fd().hash(state);
            }
            OpenFile::Remote(remote) => {
                1u8.hash(state);
                remote.hash(state);
            }
        }
    }
}

impl Debug for OpenFile {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            OpenFile::Local(file) => write!(f, "Local({})", file.as_raw_fd()),
            OpenFile::Remote(r) => write!(f, "Remote({})", r.display_url()),
        }
    }
}

/// Produces the `Authorization` header value for a remote read — typically a
/// GCS bearer token. Invoked once per request on a worker thread, so it must be
/// cheap and non-blocking: it reads an already-minted, cached token, it never
/// mints one. `None` means the URL is self-authenticating (an S3 presigned URL)
/// or needs no auth at all. `Arc` so a [`crate::io::OpenFile`]/source can be
/// cloned (the work-stealing fetch queue clones each file) while sharing one
/// token cell.
pub type AuthHeader = Arc<dyn Fn() -> Option<Arc<str>> + Send + Sync>;

/// A remote HTTP(S) object, resolved once and reused for every range request
/// against it.
///
/// Two opens of one object are equal: `Hash`/`Eq` look only at its
/// [`cache_identity`](Self::cache_identity), as the disk cache does, so what the
/// caches hold for an object keeps serving a reopen. A remote object is never
/// rewritten in place: new data is written under a new name.
///
/// The fields are pre-parsed from the URL at [`open`](Self::open) so the request
/// hot path never re-parses it; the `Url` itself isn't kept.
pub struct RemoteFile {
    /// `IP:port`, resolved once at construction (the port lives here too).
    addr: SocketAddr,
    /// Bare hostname for TLS SNI — `addr` only carries the IP, and SNI must not
    /// include a port.
    host: String,
    /// Value for the `Host:` request header: the bare host, plus the port when
    /// the URL carries a non-default one. A presigned URL's SigV4 signature
    /// covers this exact `host` header, so an S3-compatible endpoint on a custom
    /// port (e.g. MinIO at `localhost:9000`) rejects the request as a signature
    /// mismatch if the port is dropped.
    host_header: String,
    /// Origin-form request target (path + query) for the HTTP request line.
    request_target: String,
    is_https: bool,
    /// Mints the `Authorization` header for each request (a bearer token whose
    /// freshness is the store's concern), or `None` for a self-authenticating
    /// URL. Read per request, never on the URL's `Hash`/`Eq` path.
    auth: Option<AuthHeader>,
    /// Stable object identity for the on-disk cache: the authority (`host` plus
    /// any non-default port) + path, *without* the query string. S3 presigns its
    /// read URLs, so the query holds a SigV4 signature that changes every time the
    /// catalog re-signs; excluding it keeps `https://host/bucket/key?sig=A` and
    /// `...?sig=B` mapped to one cache file. (GCS reads use a stable URL with a
    /// `Bearer` header, so their query is empty anyway.) Including the port keeps
    /// two endpoints that share a host but differ by port (e.g. two MinIO
    /// instances) on separate cache files.
    cache_identity: String,
    /// Total object size in bytes (from the store listing). Lets the disk cache
    /// size its resident-block bitmap exactly, up front.
    size: u64,
}

impl RemoteFile {
    /// Parse `url`, resolve its host to a [`SocketAddr`], and intern it. The DNS
    /// lookup happens here (once) so the per-request hot path never blocks on
    /// resolution. `auth` supplies a fresh `Authorization` header per request, or
    /// `None` when the URL carries its own auth; `size` is the object's byte
    /// length (from the store listing), carried so the disk cache can size its
    /// bitmap exactly.
    pub fn open(url: Url, auth: Option<AuthHeader>, size: u64) -> std::io::Result<Self> {
        let host = url
            .host_str()
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidInput, "url has no host")
            })?
            .to_string();
        let is_https = match url.scheme() {
            "https" => true,
            "http" => false,
            other => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("unsupported url scheme: {other}"),
                ));
            }
        };
        let port = url.port_or_known_default().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "url has no port")
        })?;
        let addr = (host.as_str(), port)
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "host resolved to no addresses",
                )
            })?;

        let path = url.path();

        let mut request_target = path.to_string();
        if let Some(query) = url.query() {
            request_target.push('?');
            request_target.push_str(query);
        }

        // `Url::port()` is `Some` only for an explicit, non-default port — the
        // same condition under which the catalog's presign signs `host:port`.
        let host_header = match url.port() {
            Some(port) => format!("{host}:{port}"),
            None => host.clone(),
        };

        // Key the disk cache by the authority (host plus any non-default port) and
        // path, so two endpoints sharing a host but on different ports never
        // collide onto one cache file.
        let cache_identity = format!("{host_header}\0{path}");

        Ok(Self {
            addr,
            host,
            host_header,
            request_target,
            is_https,
            auth,
            cache_identity,
            size,
        })
    }

    /// The `Authorization` header value for the next request, or `None` when the
    /// URL is self-authenticating. Cheap and non-blocking — reads a cached token.
    pub fn auth_header(&self) -> Option<Arc<str>> {
        self.auth.as_ref().and_then(|f| f())
    }

    /// Total object size in bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Stable identity (authority + path, no query) for keying the on-disk cache.
    /// See [`cache_identity`](Self::cache_identity) field docs.
    pub fn cache_identity(&self) -> &str {
        &self.cache_identity
    }

    pub fn host(&self) -> &str {
        &self.host
    }

    /// The `Host:` request-header value (host plus a non-default port). Distinct
    /// from [`host`](Self::host), which is the bare hostname for TLS SNI.
    pub fn host_header(&self) -> &str {
        &self.host_header
    }
    pub fn port(&self) -> u16 {
        self.addr.port()
    }
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }
    pub fn request_target(&self) -> &str {
        &self.request_target
    }
    pub fn is_https(&self) -> bool {
        self.is_https
    }

    /// The origin URL, reconstructed for display (the parsed `Url` isn't kept).
    /// The port is shown only when non-default.
    fn display_url(&self) -> String {
        let scheme = if self.is_https { "https" } else { "http" };
        let default_port = if self.is_https { 443 } else { 80 };
        let port = self.addr.port();
        if port == default_port {
            format!("{scheme}://{}{}", self.host, self.request_target)
        } else {
            format!("{scheme}://{}:{port}{}", self.host, self.request_target)
        }
    }
}

impl PartialEq for RemoteFile {
    fn eq(&self, other: &Self) -> bool {
        self.cache_identity == other.cache_identity
    }
}

impl Eq for RemoteFile {}

impl Hash for RemoteFile {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.cache_identity.hash(state);
    }
}

/// A filesystem read: read `block` from `file` into its pinned cache slot.
/// Owning the [`LocalFile`] keeps the descriptor alive for the read's whole
/// flight and carries its pre-resolved device index.
pub struct FsReadRequest {
    pub file: LocalFile,
    pub block: MissingExtent,
}

/// An asynchronous filesystem write owned by a dataflow operator. `data` stays
/// alive until the shared io_uring reports the write complete, and goes out one
/// run at a time: holding it as runs is what keeps a file's bytes on the ring
/// instead of in one heap allocation.
pub struct FsWriteRequest {
    pub file: LocalFile,
    pub data: Arc<FileBytes>,
}

/// A filesystem operation issued by a dataflow operator. Reads and writes use
/// the same worker-local io_uring and operator request/completion path.
pub enum FsRequest {
    Read(FsReadRequest),
    Write(FsWriteRequest),
}

impl FsRequest {
    /// Borrow the cache-backed read payload, or return `None` for a write.
    pub fn as_read(&self) -> Option<&FsReadRequest> {
        match self {
            Self::Read(request) => Some(request),
            Self::Write(_) => None,
        }
    }
}

/// A cache-backed HTTP(S) range GET: read `block` from `remote` into its pinned
/// cache slot.
pub struct HttpGetRequest {
    pub remote: Arc<RemoteFile>,
    pub block: MissingExtent,
}

/// An object upload driven by the same per-worker HTTP connection pool and
/// io_uring as range GETs. Always a `PUT`. The body goes out run by run, like a
/// filesystem write's.
pub struct HttpUploadRequest {
    pub remote: Arc<RemoteFile>,
    pub data: Arc<FileBytes>,
}

/// An HTTP operation issued by a dataflow operator. GETs pass through the
/// compressed/disk cache; uploads bypass it but share the same worker-local
/// transport and connection pool.
pub enum HttpRequest {
    Get(HttpGetRequest),
    Upload(HttpUploadRequest),
}

/// An I/O request that knows how many bytes it transfers, so stats can total
/// bytes alongside request counts.
pub trait ReadyBytesLen {
    fn ready_bytes_len(&self) -> u64;
}

impl ReadyBytesLen for FsReadRequest {
    fn ready_bytes_len(&self) -> u64 {
        self.block.len() as u64
    }
}

impl ReadyBytesLen for HttpGetRequest {
    fn ready_bytes_len(&self) -> u64 {
        self.block.len() as u64
    }
}

impl ReadyBytesLen for FsWriteRequest {
    fn ready_bytes_len(&self) -> u64 {
        self.data.len() as u64
    }
}

impl ReadyBytesLen for FsRequest {
    fn ready_bytes_len(&self) -> u64 {
        match self {
            Self::Read(request) => request.ready_bytes_len(),
            Self::Write(request) => request.ready_bytes_len(),
        }
    }
}

impl ReadyBytesLen for HttpUploadRequest {
    fn ready_bytes_len(&self) -> u64 {
        self.data.len() as u64
    }
}

impl ReadyBytesLen for HttpRequest {
    fn ready_bytes_len(&self) -> u64 {
        match self {
            Self::Get(request) => request.ready_bytes_len(),
            Self::Upload(request) => request.ready_bytes_len(),
        }
    }
}

/// Associates an I/O request with the dataflow and operator that issued it, so
/// its completion can be routed back to the correct operator.
pub struct DataFlowRequest<R> {
    pub data_flow_id: Identifier,
    pub operator_idx: Identifier,
    pub request: R,
    /// Identity the requester assigns to each physical read it tracks. Writes,
    /// uploads, and callers using the transport API directly leave this unset.
    pub(crate) pending_read_id: Option<Identifier>,
    /// When the issuing dataflow handed this operation off, stamped only when
    /// the query opted into stats. `None` (no clock read) otherwise.
    pub submitted_at: Option<std::time::Instant>,
}

impl<R> DataFlowRequest<R> {
    pub fn new(data_flow_id: Identifier, operator_idx: Identifier, request: R) -> Self {
        Self {
            data_flow_id,
            operator_idx,
            request,
            pending_read_id: None,
            submitted_at: None,
        }
    }

    pub(crate) fn with_pending_read_id(mut self, id: Identifier) -> Self {
        self.pending_read_id = Some(id);
        self
    }

    /// Transform the request payload while preserving its routing and timing.
    pub fn map<T>(self, f: impl FnOnce(R) -> T) -> DataFlowRequest<T> {
        DataFlowRequest {
            data_flow_id: self.data_flow_id,
            operator_idx: self.operator_idx,
            request: f(self.request),
            pending_read_id: self.pending_read_id,
            submitted_at: self.submitted_at,
        }
    }
}

/// How one remote operation is charged across its transports. A GET counts each
/// resident piece as a disk-cache read and each hole as an HTTP fetch; an upload
/// is one HTTP operation with no disk-cache component.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct RemoteSplit {
    pub disk_cache_requests: u64,
    pub disk_cache_bytes: u64,
    pub http_requests: u64,
    pub http_bytes: u64,
}

/// The in-flight time of one remote read, split across the tiers that served it:
/// each piece bills the time from the read's submission to that piece landing, to
/// the cache or the network. Accumulated as the pieces complete and billed when
/// the read finishes, so a read split across both tiers charges each its own wait.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct RemoteReadTime {
    pub disk_cache: Duration,
    pub http: Duration,
}

/// One completed I/O operation drained from [`IORequester::completions`], with
/// its originating dataflow/operator attached. Each variant carries the concrete
/// request it finished; a completed cache-backed GET also carries the
/// [`RemoteReadTime`] its pieces accrued.
pub enum Completion {
    FsRead(DataFlowRequest<FsReadRequest>),
    FsWrite(DataFlowRequest<FsWriteRequest>),
    HttpGet(DataFlowRequest<HttpGetRequest>, RemoteReadTime),
    HttpUpload(DataFlowRequest<HttpUploadRequest>),
}

/// An operation that failed transport-side — an HTTP operation that exhausted
/// its retries (or hit a non-retryable error), or a disk operation whose CQE
/// came back negative.
/// Carries the dataflow/operator that issued it so the worker can cancel just
/// that dataflow, plus the error to report. Surfaced as the `Err` arm of a
/// per-operation result from [`IORequester::completions`], so one failed operation never
/// aborts the whole completion drain or tears down the worker. The block is
/// left uncommitted.
pub struct FailedIO {
    pub data_flow_id: Identifier,
    pub operator_idx: Identifier,
    pub(crate) pending_read_id: Option<Identifier>,
    pub error: IORequesterError,
}

/// Opens a file for direct (uncached) reads.
///
/// - **Linux**: uses `O_DIRECT`.
/// - **macOS**: uses `F_NOCACHE` (best-effort, not true direct I/O).
/// - **Other**: falls back to normal cached reads.
pub fn open_direct_read(path: &Path) -> std::io::Result<LocalFile> {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::OpenOptionsExt;
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(path)?;
        LocalFile::new(file)
    }

    #[cfg(target_os = "macos")]
    {
        use std::os::fd::AsRawFd;
        let file = OpenOptions::new().read(true).open(path)?;
        let rc = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) };
        if rc == -1 {
            return Err(std::io::Error::last_os_error());
        }
        LocalFile::new(file)
    }

    #[cfg(all(unix, not(any(target_os = "linux", target_os = "macos"))))]
    {
        let file = OpenOptions::new().read(true).open(path)?;
        return LocalFile::new(file);
    }

    #[cfg(not(unix))]
    {
        let file = OpenOptions::new().read(true).open(path)?;
        return LocalFile::new(file);
    }
}
