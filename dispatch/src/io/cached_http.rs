//! HTTP(S) range reads backed by an optional on-disk cache.
//!
//! Every remote read becomes one [`RequestedRead`], satisfied by one or more
//! pieces and completed once every piece has landed:
//!
//! * With a [`DiskCache`], the requested run is split against what's on disk -
//!   resident pieces are read from the local cache file, holes are fetched over
//!   HTTP and written back.
//! * With no cache, it's a single piece fetched from the network.
//!
//! The original request is held by its [`RequestedRead`] and handed back
//! unchanged once the last piece lands, so the operator sees exactly the request
//! it submitted regardless of how it was satisfied.
//!
//! Like [`HttpEngine`], this is **ring-less**: the owning [`IORequester`] lends
//! it the worker's shared [`IOBackend`] (and the disk-op id counter) at submit
//! and drain time, so cache-file reads and write-backs ride the same per-core
//! io_uring as fs reads and HTTP sockets. The requester drains that ring and
//! routes each completion back here.
//!
//! [`IORequester`]: super::IORequester

use crate::Identifier;
use crate::io::IORequesterError as Error;
use crate::io::backend::IOBackend;
use crate::io::disk_cache::{DiskCache, Object, Segment};
use crate::io::http::{BodyPart, HttpEngine, RemoteRead, RemoteUpload, default_client_config};
use crate::io::{
    Completion, DataFlowRequest, FailedIO, HttpGetRequest, HttpUploadRequest, OpenFile,
    RemoteReadTime, RemoteSplit,
};
use crate::memory::cache::MissingExtent;
use std::collections::HashMap;
use std::sync::Arc;

#[cfg(target_os = "linux")]
use crate::io::http::HTTP_TAG;
#[cfg(target_os = "linux")]
use crate::io::receive_rate::{ReceiveRateLimiter, process_receive_rate_limiter};
#[cfg(target_os = "linux")]
use io_uring::{opcode, types};
#[cfg(target_os = "linux")]
use std::collections::VecDeque;
#[cfg(target_os = "linux")]
use std::time::{Duration, Instant};

/// The `user_data` of the timeout that wakes a worker's ring once the receive
/// allowance covers its oldest held fetch. It carries [`HTTP_TAG`] so the
/// requester routes it here, and a bit no exchange slot or step deadline uses.
#[cfg(target_os = "linux")]
const RECEIVE_ALLOWANCE_TIMER: u64 = HTTP_TAG | (1 << 61);

/// The most bytes between two ranges of one object that are fetched and
/// dropped so that one request serves both ranges. None by default, so only
/// ranges that touch are joined: bytes fetched to be dropped take the network
/// card's rate from bytes that are wanted, which costs a read bound by the
/// network more than the request it saves, and decrypting them costs about
/// what a request does. Read from `PIVOT_REMOTE_READ_MAX_GAP_BYTES`.
const DEFAULT_MAX_DROPPED_GAP_BYTES: usize = 0;

/// The longest range one request asks for when ranges are joined. A store
/// serves one connection at a limited rate, so a range much longer than this
/// arrives sooner as several requests side by side. Read from
/// `PIVOT_REMOTE_READ_MAX_BYTES`.
const DEFAULT_MAX_JOINED_READ_BYTES: usize = 8 * 1024 * 1024;

type Result<T> = std::result::Result<T, Error>;

/// One drained read result: a completion or a terminal failure, matching
/// [`IORequester::completions`](super::IORequester::completions)'s element type.
type ReadResult = std::result::Result<Completion, FailedIO>;

/// HTTP range reads with an optional disk cache in front of the network.
pub(crate) struct CachedHttpEngine {
    /// Declared before `http_reads`/`cache_writes` on purpose: on non-Linux the
    /// engine's `Drop` blocks until every in-flight fetch reports back, and Rust
    /// drops fields in declaration order, so `http` must drain while those maps
    /// still hold the slot pins for reads a pool thread may still be writing into.
    /// The pin is a reader refcount on a ring slot, not an allocation: dropping it
    /// frees no memory (the ring mmap is process-lifetime), it only makes the slot
    /// eligible for cache eviction/reuse. So reordering `http` below the maps would
    /// release those pins first and let the cache recycle a slot out from under an
    /// in-flight write, corrupting whichever read the slot is handed to next.
    http: HttpEngine,
    /// The shared disk cache, or `None` to read straight from the network.
    disk_cache: Option<Arc<DiskCache>>,

    /// One per in-flight read; holds the original request until every piece lands.
    requested_reads: HashMap<Identifier, RequestedRead>,
    next_read_id: Identifier,
    /// Pieces being read from a local cache file (backend disk-id space).
    cache_reads: HashMap<Identifier, CacheRead>,
    /// Pieces being fetched over HTTP (engine request-id space).
    http_reads: HashMap<Identifier, HttpRead>,
    /// Whole-object uploads, keyed by the same engine request id space and
    /// driven by the same connection pool as reads.
    http_uploads: HashMap<Identifier, DataFlowRequest<HttpUploadRequest>>,
    next_http_id: Identifier,
    /// Write-backs populating the cache file after an HTTP piece landed (backend
    /// disk-id space). Each holds the slot pin until the write has read it.
    cache_writes: HashMap<Identifier, CacheWrite>,

    /// Network pieces registered and not yet issued, in the order they were
    /// registered. They wait for the rest of their logical read, so that
    /// pieces lying close together in an object go out as one request (see
    /// [`issue_registered_fetches`](Self::issue_registered_fetches)).
    unissued_fetches: Vec<(Identifier, RemoteRead)>,
    /// The pieces a joined request serves, by the request's id, which is its
    /// first piece's. A request for a single piece has no entry.
    joined_fetches: HashMap<Identifier, Vec<Identifier>>,
    /// Requests that could not be started, failed on the next drain.
    failed_starts: Vec<(Identifier, Error)>,
    max_dropped_gap_bytes: usize,
    max_joined_read_bytes: usize,

    /// The process's receive allowance: every network fetch takes its length
    /// from it before it is issued.
    #[cfg(target_os = "linux")]
    receive_limiter: Arc<ReceiveRateLimiter>,
    /// Network fetches the allowance has not covered yet, oldest first. Each
    /// is already tracked in `http_reads`; only its request is still to go out.
    /// A cancelled dataflow's held fetches are issued in their turn all the
    /// same, like its fetches already in flight.
    #[cfg(target_os = "linux")]
    held_fetches: VecDeque<(Identifier, RemoteRead)>,
    /// How long the armed allowance timer waits. Boxed so the address its SQE
    /// carries stays put.
    #[cfg(target_os = "linux")]
    allowance_wait: Box<types::Timespec>,
    /// Whether a [`RECEIVE_ALLOWANCE_TIMER`] is in flight. Fetches are held
    /// only while one is: the allowance grows with time, not with anything
    /// completing, so a worker with nothing else in flight has only this to
    /// bring it back to its held fetches.
    #[cfg(target_os = "linux")]
    allowance_timer_armed: bool,
}

/// The overarching record for one remote read an operator requested, in flight as
/// one or more pieces (cache-file reads and/or HTTP fetches). It holds the
/// original request and yields it unchanged once the last piece has landed.
struct RequestedRead {
    remaining: usize,
    request: DataFlowRequest<HttpGetRequest>,
    /// In-flight time accrued so far, per tier: each piece adds its own wait as it
    /// lands, and the total bills the dataflow's stats once the read completes.
    time: RemoteReadTime,
}

/// One piece read from the local cache file.
struct CacheRead {
    read: Identifier,
    block: MissingExtent,
    /// Keeps the cache file open for the read's lifetime.
    _object: Arc<Object>,
}

/// One piece fetched over HTTP. `object` is `Some` when it should be written back
/// to that cache file once it lands, `None` for an uncached read.
struct HttpRead {
    read: Identifier,
    block: MissingExtent,
    object: Option<Arc<Object>>,
    /// The exact body size the transport requires before reporting completion -
    /// the block's length, clamped to the object's end when the block-aligned run
    /// overhangs it.
    expected_body_len: usize,
}

/// A write-back populating the cache file after an HTTP piece landed.
struct CacheWrite {
    object: Arc<Object>,
    file_offset: usize,
    len: usize,
    /// Keeps the source slot pinned until the write has read it.
    _block: MissingExtent,
}

impl CachedHttpEngine {
    pub fn new(
        http_config: Arc<rustls::ClientConfig>,
        disk_cache: Option<Arc<DiskCache>>,
    ) -> Result<Self> {
        Ok(Self {
            http: HttpEngine::new(http_config)?,
            disk_cache,
            requested_reads: HashMap::new(),
            next_read_id: 0,
            cache_reads: HashMap::new(),
            http_reads: HashMap::new(),
            http_uploads: HashMap::new(),
            next_http_id: 0,
            cache_writes: HashMap::new(),
            unissued_fetches: Vec::new(),
            joined_fetches: HashMap::new(),
            failed_starts: Vec::new(),
            max_dropped_gap_bytes: crate::env::get_env_var_with_default(
                "PIVOT_REMOTE_READ_MAX_GAP_BYTES",
                DEFAULT_MAX_DROPPED_GAP_BYTES,
            ),
            max_joined_read_bytes: crate::env::get_env_var_with_default(
                "PIVOT_REMOTE_READ_MAX_BYTES",
                DEFAULT_MAX_JOINED_READ_BYTES,
            ),
            #[cfg(target_os = "linux")]
            receive_limiter: process_receive_rate_limiter(),
            #[cfg(target_os = "linux")]
            held_fetches: VecDeque::new(),
            #[cfg(target_os = "linux")]
            allowance_wait: Box::new(types::Timespec::new()),
            #[cfg(target_os = "linux")]
            allowance_timer_armed: false,
        })
    }

    /// Take network fetches from `limiter` instead of the process's allowance,
    /// so a test controls the rate.
    #[cfg(all(test, target_os = "linux"))]
    pub fn force_receive_rate_limiter(&mut self, limiter: Arc<ReceiveRateLimiter>) {
        self.receive_limiter = limiter;
    }

    /// Join ranges with up to `bytes` between them into one request, so a
    /// test controls what is dropped.
    #[cfg(test)]
    pub fn force_max_dropped_gap_bytes(&mut self, bytes: usize) {
        self.max_dropped_gap_bytes = bytes;
    }

    pub fn with_default_config(disk_cache: Option<Arc<DiskCache>>) -> Result<Self> {
        Self::new(default_client_config(), disk_cache)
    }

    /// Submit a read for a remote region as one [`RequestedRead`]: resident pieces
    /// are served from the cache file and holes fetched over HTTP, so only the
    /// missing ranges hit the network. With no cache the whole block is a single
    /// fetched piece. `disk_id` allocates ids for cache-file ops on the backend.
    ///
    /// The fetched pieces are registered here and go out with
    /// [`issue_registered_fetches`](Self::issue_registered_fetches), once the
    /// caller has submitted every region of its logical read.
    pub fn get(
        &mut self,
        backend: &mut IOBackend,
        disk_id: &mut Identifier,
        request: DataFlowRequest<HttpGetRequest>,
    ) -> Result<RemoteSplit> {
        // The cache file for this object, or `None` to read straight from the
        // network. A resident segment is only ever produced for a `Some` object,
        // so its fd is read inside the resident branch below.
        let object = self
            .disk_cache
            .as_deref()
            .and_then(|dc| dc.open_object(&request.request.remote));
        let segments = match &object {
            Some(obj) => obj.split_into_segments(
                request.request.block.file_offset(),
                request.request.block.len(),
            ),
            // No cache: the whole block is one hole to fetch.
            None => vec![Segment {
                rel_offset: 0,
                len: request.request.block.len(),
                present: false,
            }],
        };

        let read = self.next_read_id;
        self.next_read_id += 1;

        // Count each piece against the tier that serves it: a resident segment is
        // one cache-file read, a hole one HTTP fetch. A read split across both
        // reports under each tier so neither's work is hidden.
        let mut split = RemoteSplit::default();

        for seg in &segments {
            let block = request.request.block.carve(seg.rel_offset, seg.len);
            if seg.present {
                split.disk_cache_requests += 1;
                split.disk_cache_bytes += seg.len as u64;
                // Resident on disk: read it from the cache file into the slot. A
                // resident segment is only ever produced for a `Some` object.
                let object = object
                    .clone()
                    .expect("a resident segment implies a cache object");
                let id = *disk_id;
                *disk_id += 1;
                backend.submit_read(
                    object.fd(),
                    block.file_offset() as u64,
                    block.dest(),
                    block.len(),
                    id,
                )?;
                self.cache_reads.insert(
                    id,
                    CacheRead {
                        read,
                        block,
                        _object: object,
                    },
                );
            } else {
                // A hole (or an uncached read): fetch over HTTP, written back once
                // it lands iff there's a cache object. The requests are counted
                // as they go out, since several pieces may share one.
                split.http_bytes += seg.len as u64;
                let id = self.next_http_id;
                self.next_http_id += 1;
                let remote_read = RemoteRead {
                    remote: request.request.remote.clone(),
                    offset: block.file_offset() as u64,
                    parts: vec![BodyPart::Kept {
                        dest: block.dest(),
                        len: block.len(),
                    }],
                };
                let expected_body_len = remote_read.expected_body_len();
                self.unissued_fetches.push((id, remote_read));
                self.http_reads.insert(
                    id,
                    HttpRead {
                        read,
                        block,
                        object: object.clone(),
                        expected_body_len,
                    },
                );
            }
            // Flush each segment as we go: one block can split into more pieces
            // than the ring's submission queue holds, so batching all the pushes
            // before a single submit could overflow it.
            backend.submit()?;
        }

        // Register the read once every piece is in flight. If a submit above
        // failed we never reach here: `request` drops and the pieces already
        // issued complete harmlessly (their read is simply absent).
        self.requested_reads.insert(
            read,
            RequestedRead {
                remaining: segments.len(),
                request,
                time: RemoteReadTime::default(),
            },
        );
        Ok(split)
    }

    /// Issue the network pieces registered since the last call, and return how
    /// many requests they made. Pieces of one object that follow each other
    /// with at most `max_dropped_gap_bytes` between them go out as one request
    /// of at most `max_joined_read_bytes`: a logical read's ranges often lie
    /// next to each other in the object, and a range the cache placed in two
    /// slots is two pieces with nothing between them.
    ///
    /// A request that cannot be started fails its pieces on the next drain,
    /// like one that failed on the network.
    pub fn issue_registered_fetches(&mut self, backend: &mut IOBackend) -> Result<u64> {
        let mut requests = 0;
        let mut pieces = std::mem::take(&mut self.unissued_fetches)
            .into_iter()
            .peekable();
        while let Some((id, mut fetch)) = pieces.next() {
            let mut joined = Vec::new();
            while let Some((_, next)) = pieces.peek() {
                let end = fetch.offset + fetch.len() as u64;
                let follows_closely = Arc::ptr_eq(&fetch.remote, &next.remote)
                    && next.offset >= end
                    && (next.offset - end) as usize <= self.max_dropped_gap_bytes
                    && (next.offset - fetch.offset) as usize + next.len()
                        <= self.max_joined_read_bytes;
                if !follows_closely {
                    break;
                }
                let (next_id, next) = pieces.next().expect("the piece just peeked at");
                let gap = (next.offset - end) as usize;
                if gap > 0 {
                    fetch.parts.push(BodyPart::Dropped { len: gap });
                }
                fetch.parts.extend(next.parts);
                if joined.is_empty() {
                    joined.push(id);
                }
                joined.push(next_id);
            }
            if !joined.is_empty() {
                self.joined_fetches.insert(id, joined);
            }
            requests += 1;
            if let Err(error) = self.issue_or_hold_fetch(backend, id, fetch) {
                self.failed_starts.push((id, error));
            }
            // Flushed as we go: one logical read can make more requests than
            // the ring's submission queue holds.
            backend.submit()?;
        }
        Ok(requests)
    }

    /// Issue a network fetch if the receive allowance covers it, and hold it
    /// back otherwise. A fetch never overtakes one already held: the oldest
    /// would wait on every later, smaller one that still fits.
    #[cfg(target_os = "linux")]
    fn issue_or_hold_fetch(
        &mut self,
        backend: &mut IOBackend,
        id: Identifier,
        read: RemoteRead,
    ) -> Result<()> {
        if !self.held_fetches.is_empty() {
            self.held_fetches.push_back((id, read));
            return Ok(());
        }
        match self.take_receive_allowance(&read) {
            Ok(()) => self.start_http(backend, id, read),
            Err(wait) => {
                // Armed before the fetch is queued: a fetch left in the queue
                // after its caller was told it failed would be issued later
                // into a slot its caller has given up.
                if !self.allowance_timer_armed {
                    self.arm_allowance_timer(backend, wait)?;
                }
                self.held_fetches.push_back((id, read));
                Ok(())
            }
        }
    }

    /// Off Linux nothing limits the receive rate, so every fetch is issued.
    #[cfg(not(target_os = "linux"))]
    fn issue_or_hold_fetch(
        &mut self,
        backend: &mut IOBackend,
        id: Identifier,
        read: RemoteRead,
    ) -> Result<()> {
        self.start_http(backend, id, read)
    }

    /// Take the bytes `read` will receive from the receive allowance, or
    /// report how long until it covers them.
    #[cfg(target_os = "linux")]
    fn take_receive_allowance(&self, read: &RemoteRead) -> std::result::Result<(), Duration> {
        self.receive_limiter
            .try_take(read.expected_body_len(), Instant::now())
    }

    /// Issue held fetches, oldest first, for as long as the receive allowance
    /// covers them, and wake the ring again when it will cover the next. A
    /// fetch that cannot be started fails its read.
    #[cfg(target_os = "linux")]
    fn issue_held_fetches(
        &mut self,
        backend: &mut IOBackend,
        out: &mut Vec<ReadResult>,
    ) -> Result<()> {
        while let Some((_, read)) = self.held_fetches.front() {
            if let Err(wait) = self.take_receive_allowance(read) {
                if !self.allowance_timer_armed {
                    self.arm_allowance_timer(backend, wait)?;
                }
                break;
            }
            let (id, read) = self
                .held_fetches
                .pop_front()
                .expect("the fetch just inspected at the front");
            if let Err(error) = self.start_http(backend, id, read) {
                self.fail_fetch(id, error, out);
            }
        }
        Ok(())
    }

    /// Have the ring complete an entry after `wait`, so a worker blocked on it
    /// with nothing else in flight comes back to its held fetches.
    #[cfg(target_os = "linux")]
    fn arm_allowance_timer(&mut self, backend: &mut IOBackend, wait: Duration) -> Result<()> {
        *self.allowance_wait = types::Timespec::new()
            .sec(wait.as_secs())
            .nsec(wait.subsec_nanos());
        let timer = opcode::Timeout::new(&*self.allowance_wait as *const types::Timespec)
            .build()
            .user_data(RECEIVE_ALLOWANCE_TIMER);
        unsafe {
            backend
                .ring
                .submission()
                .push(&timer)
                .map_err(|_| crate::io::backend::Error::SubmissionQueueFull)?;
        }
        backend.submit()?;
        self.allowance_timer_armed = true;
        Ok(())
    }

    /// Hand a remote read to the engine (ring-driven on Linux, synchronous else).
    fn start_http(
        &mut self,
        backend: &mut IOBackend,
        id: Identifier,
        read: RemoteRead,
    ) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            self.http.start_get(&mut backend.ring, id, read)?;
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = backend;
            self.http.start_get(id, read)?;
        }
        Ok(())
    }

    /// Submit an upload directly to the underlying transport. Uploads bypass the
    /// read cache but otherwise share its worker-local HTTP engine. Unlike a read,
    /// an upload touches no disk, so it needs no backend disk-op id.
    pub fn upload(
        &mut self,
        backend: &mut IOBackend,
        request: DataFlowRequest<HttpUploadRequest>,
    ) -> Result<()> {
        let id = self.next_http_id;
        self.next_http_id += 1;
        let upload = RemoteUpload {
            remote: request.request.remote.clone(),
            data: request.request.data.clone(),
        };
        self.start_upload(backend, id, upload)?;
        self.http_uploads.insert(id, request);
        Ok(())
    }

    fn start_upload(
        &mut self,
        backend: &mut IOBackend,
        id: Identifier,
        upload: RemoteUpload,
    ) -> Result<()> {
        #[cfg(target_os = "linux")]
        {
            self.http.start_upload(&mut backend.ring, id, upload)?;
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = backend;
            self.http.start_upload(id, upload)?;
        }
        Ok(())
    }

    /// Feed an HTTP socket completion to the engine (it may submit follow-up SQEs,
    /// including transparent reconnect-and-retry). Linux only - elsewhere the
    /// engine runs synchronously and never produces ring completions.
    #[cfg(target_os = "linux")]
    pub fn on_socket_completion(
        &mut self,
        backend: &mut IOBackend,
        user_data: u64,
        result: i32,
    ) -> Result<()> {
        if user_data == RECEIVE_ALLOWANCE_TIMER {
            // The drain of this pass issues what the allowance now covers.
            self.allowance_timer_armed = false;
            return Ok(());
        }
        self.http.on_cqe(&mut backend.ring, user_data, result)?;
        Ok(())
    }

    /// Handle a finished backend disk op that belongs to this engine - a
    /// cache-file read (advances its read) or a write-back (records the bytes
    /// resident). The caller has already ruled out its own fs reads.
    pub fn complete_disk(&mut self, id: Identifier, result: i32, out: &mut Vec<ReadResult>) {
        if let Some(cache_read) = self.cache_reads.remove(&id) {
            self.complete_cache_read(cache_read, result, out);
        } else if let Some(write) = self.cache_writes.remove(&id) {
            // Only record the bytes resident if the *whole* range landed - a
            // short/failed write must not mark blocks present that a later read
            // would then serve as garbage. Folding the new bytes into the budget
            // (and any eviction) is the cache's job.
            if result >= 0
                && result as usize == write.len
                && let Some(dc) = self.disk_cache.as_deref()
            {
                dc.mark_resident(&write.object, write.file_offset, write.len);
            }
        } else {
            // The requester only delegates ids that aren't its fs reads, so an id
            // unknown to both maps means a completion was tracked nowhere - a bug.
            debug_assert!(false, "disk completion id {id} belongs to no in-flight op");
        }
    }

    /// Drain HTTP reads the engine finished this pass: each piece commits, gets
    /// written back if cached, and advances its read; the original request is
    /// yielded once its last piece lands. Terminal failures cancel the dataflow.
    pub fn drain(
        &mut self,
        backend: &mut IOBackend,
        disk_id: &mut Identifier,
        out: &mut Vec<ReadResult>,
    ) -> Result<()> {
        // A caller that registered pieces without issuing them has them issued
        // here, so no piece waits on a call that never comes.
        self.issue_registered_fetches(backend)?;
        for (id, error) in std::mem::take(&mut self.failed_starts) {
            self.fail_fetch(id, error, out);
        }
        #[cfg(target_os = "linux")]
        self.issue_held_fetches(backend, out)?;
        // Before the completions: a piece issued again and finished within one
        // pass is still tracked here, so its read is billed the re-issue. A read
        // a sibling piece already failed is gone, with nothing left to bill.
        #[cfg(target_os = "linux")]
        for id in self.http.take_reissued() {
            if let Some(http_read) = self.http_reads.get(&id)
                && let Some(read) = self.requested_reads.get_mut(&http_read.read)
            {
                read.time.http_reissues += 1;
            }
        }
        for id in self.http.take_completed() {
            if let Some(upload) = self.http_uploads.remove(&id) {
                out.push(Ok(Completion::HttpUpload(upload)));
                continue;
            }
            // Every piece the request served has landed with it.
            let joined = self.joined_fetches.remove(&id);
            for piece in joined.as_deref().unwrap_or(std::slice::from_ref(&id)) {
                if let Some(http_read) = self.http_reads.remove(piece) {
                    self.complete_http_read(backend, disk_id, http_read, out)?;
                }
            }
        }
        for (id, error) in self.http.take_failed() {
            if self.http_reads.contains_key(&id) {
                self.fail_fetch(id, error.into(), out);
            } else if let Some(upload) = self.http_uploads.remove(&id) {
                out.push(Err(FailedIO {
                    data_flow_id: upload.data_flow_id,
                    operator_idx: upload.operator_idx,
                    pending_read_id: upload.pending_read_id,
                    error: error.into(),
                }));
            }
        }
        Ok(())
    }

    /// Fail every piece the request `id` was to serve (see
    /// [`fail_http_read`](Self::fail_http_read)).
    fn fail_fetch(&mut self, id: Identifier, error: Error, out: &mut Vec<ReadResult>) {
        match self.joined_fetches.remove(&id) {
            None => self.fail_http_read(id, error, out),
            Some(pieces) => {
                let message = error.to_string();
                for piece in pieces {
                    let error = std::io::Error::other(message.clone()).into();
                    self.fail_http_read(piece, error, out);
                }
            }
        }
    }

    /// Fail the read the network piece `id` belongs to: its extent leaves the
    /// cache so the next reader starts a fill of its own, and the owning
    /// dataflow is told. A read a sibling piece already failed is gone, and
    /// has been reported.
    fn fail_http_read(&mut self, id: Identifier, error: Error, out: &mut Vec<ReadResult>) {
        if let Some(http_read) = self.http_reads.remove(&id)
            && let Some(request) = self.fail_read(http_read.read)
        {
            request
                .request
                .block
                .remove_from_cache(&crate::io::OpenFile::Remote(request.request.remote.clone()));
            out.push(Err(FailedIO {
                data_flow_id: request.data_flow_id,
                operator_idx: request.operator_idx,
                pending_read_id: request.pending_read_id,
                error,
            }));
        }
    }

    /// `true` while any HTTP GET or upload is in flight or held back.
    pub fn has_network_pending(&self) -> bool {
        self.http.has_active() || !self.http_reads.is_empty() || !self.http_uploads.is_empty()
    }

    /// `true` if an HTTP completion is already in hand (non-Linux only, where the
    /// engine delivers over its own channel rather than the shared ring).
    #[cfg(not(target_os = "linux"))]
    pub fn has_ready_completion(&self) -> bool {
        self.http.has_ready_completion()
    }

    /// The HTTP completion channel, so the requester can park on it alongside the
    /// disk channel (non-Linux: the two pools deliver on independent channels with
    /// no shared ring).
    #[cfg(not(target_os = "linux"))]
    pub fn completion_receiver(
        &self,
    ) -> &crossbeam_channel::Receiver<crate::io::http::HttpCompletion> {
        self.http.completion_receiver()
    }

    /// `true` while any cache-file read or write-back is in flight on the backend.
    pub fn has_disk_pending(&self) -> bool {
        !self.cache_reads.is_empty() || !self.cache_writes.is_empty()
    }

    /// Number of network operations currently outstanding. With no cache this
    /// is one per GET; with a cache it counts GET holes, plus every upload.
    pub fn network_in_flight(&self) -> usize {
        self.http_reads.len() + self.http_uploads.len()
    }

    /// A cache-file piece landed: commit its sub-blocks and, if it was the last
    /// piece of its read, yield the original request. A failed read fails the
    /// whole requested read.
    fn complete_cache_read(
        &mut self,
        cache_read: CacheRead,
        result: i32,
        out: &mut Vec<ReadResult>,
    ) {
        // A negative result is an error; a non-negative result shorter than the
        // block left the slot only partially filled. Either way the bytes the
        // bitmap promised aren't all there, so fail the read rather than commit and
        // serve stale slot bytes - symmetric with the write-back's `result == len`
        // guard.
        if result < 0 || result as usize != cache_read.block.len() {
            if let Some(request) = self.fail_read(cache_read.read) {
                request
                    .request
                    .block
                    .remove_from_cache(&crate::io::OpenFile::Remote(
                        request.request.remote.clone(),
                    ));
                let error = if result < 0 {
                    std::io::Error::from_raw_os_error(-result)
                } else {
                    std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "short read from disk cache",
                    )
                };
                out.push(Err(FailedIO {
                    data_flow_id: request.data_flow_id,
                    operator_idx: request.operator_idx,
                    pending_read_id: request.pending_read_id,
                    error: error.into(),
                }));
            }
            return;
        }
        cache_read.block.commit();
        if let Some((request, time)) = self.finish_piece(cache_read.read, true) {
            out.push(Ok(Completion::HttpGet(request, time)));
        }
    }

    /// An HTTP piece landed: commit it, queue (and flush) a write-back into the
    /// cache file if it came from a cached read, and advance its read.
    fn complete_http_read(
        &mut self,
        backend: &mut IOBackend,
        disk_id: &mut Identifier,
        http_read: HttpRead,
        out: &mut Vec<ReadResult>,
    ) -> Result<()> {
        // The transport reports completion only after exactly `expected_body_len`
        // bytes arrive; a shorter response fails in the protocol/body path. That
        // expected length can still be shorter than the block-aligned extent when
        // the known object EOF clamps it. Require the clamped prefix to reach the
        // extent's final block. If it does not, the logical range extends past the
        // current object size far enough to leave a whole cache block unwritten.
        if !http_read
            .block
            .prefix_reaches_last_block(http_read.expected_body_len)
        {
            if let Some(request) = self.fail_read(http_read.read) {
                request
                    .request
                    .block
                    .remove_from_cache(&OpenFile::Remote(request.request.remote.clone()));
                out.push(Err(FailedIO {
                    data_flow_id: request.data_flow_id,
                    operator_idx: request.operator_idx,
                    pending_read_id: request.pending_read_id,
                    error: std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "read reaches past the end of the remote object",
                    )
                    .into(),
                }));
            }
            return Ok(());
        }
        // The validated body reached the final block, so committing its touched
        // prefix commits every block in this extent.
        http_read.block.commit_prefix(http_read.expected_body_len);

        if let Some(object) = http_read.object {
            let id = *disk_id;
            *disk_id += 1;
            backend.submit_write(
                object.fd(),
                http_read.block.file_offset() as u64,
                http_read.block.dest(),
                http_read.block.len(),
                id,
            )?;
            // Flush each write-back as it's queued: a burst of completed reads can
            // queue more write-backs than the ring's submission queue holds, the
            // same reason `request` flushes each segment.
            backend.submit()?;
            self.cache_writes.insert(
                id,
                CacheWrite {
                    object,
                    file_offset: http_read.block.file_offset(),
                    len: http_read.block.len(),
                    _block: http_read.block,
                },
            );
        }

        if let Some((request, time)) = self.finish_piece(http_read.read, false) {
            out.push(Ok(Completion::HttpGet(request, time)));
        }
        Ok(())
    }

    /// Record one piece of requested read `read` as landed, adding its in-flight
    /// time (submission to now) to the tier that served it - the disk cache when
    /// `from_disk_cache`, else the network. Returns the original request and the
    /// read's accumulated per-tier time once the last piece lands (the pieces have
    /// already committed every sub-block). Yields `None` if the read already failed.
    fn finish_piece(
        &mut self,
        read: Identifier,
        from_disk_cache: bool,
    ) -> Option<(DataFlowRequest<HttpGetRequest>, RemoteReadTime)> {
        let r = self.requested_reads.get_mut(&read)?;
        debug_assert!(r.remaining > 0, "finish_piece: no pieces left");
        if let Some(submitted_at) = r.request.submitted_at {
            let elapsed = submitted_at.elapsed();
            if from_disk_cache {
                r.time.disk_cache += elapsed;
            } else {
                r.time.http += elapsed;
            }
        }
        r.remaining -= 1;
        if r.remaining == 0 {
            let finished = self.requested_reads.remove(&read).unwrap();
            Some((finished.request, finished.time))
        } else {
            None
        }
    }

    /// Tear down a requested read whose piece failed, returning the dataflow to
    /// cancel (or `None` if an earlier failed piece already tore it down).
    fn fail_read(&mut self, read: Identifier) -> Option<DataFlowRequest<HttpGetRequest>> {
        self.requested_reads.remove(&read).map(|r| r.request)
    }
}
