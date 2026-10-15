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
use crate::io::http::{HttpEngine, RemoteRead, RemoteUpload, default_client_config};
use crate::io::{
    Completion, DataFlowRequest, FailedIO, HttpGetRequest, HttpUploadRequest, OpenFile,
    RemoteReadTime, RemoteSplit,
};
use crate::memory::compressed_cache::MissingExtent;
use std::collections::HashMap;
use std::sync::Arc;

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
        })
    }

    pub fn with_default_config(disk_cache: Option<Arc<DiskCache>>) -> Result<Self> {
        Self::new(default_client_config(), disk_cache)
    }

    /// Submit a read for a remote region as one [`RequestedRead`]: resident pieces
    /// are served from the cache file and holes fetched over HTTP, so only the
    /// missing ranges hit the network. With no cache the whole block is a single
    /// fetched piece. `disk_id` allocates ids for cache-file ops on the backend.
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
                // it lands iff there's a cache object.
                split.http_requests += 1;
                split.http_bytes += seg.len as u64;
                let id = self.next_http_id;
                self.next_http_id += 1;
                let remote_read = RemoteRead {
                    remote: request.request.remote.clone(),
                    offset: block.file_offset() as u64,
                    len: block.len(),
                    dest: block.dest(),
                };
                let expected_body_len = remote_read.expected_body_len();
                self.start_http(backend, id, remote_read)?;
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
        for id in self.http.take_completed() {
            if let Some(http_read) = self.http_reads.remove(&id) {
                self.complete_http_read(backend, disk_id, http_read, out)?;
            } else if let Some(upload) = self.http_uploads.remove(&id) {
                out.push(Ok(Completion::HttpUpload(upload)));
            }
        }
        for (id, error) in self.http.take_failed() {
            if let Some(http_read) = self.http_reads.remove(&id)
                && let Some(request) = self.fail_read(http_read.read)
            {
                request
                    .request
                    .block
                    .remove_from_cache(&crate::io::OpenFile::Remote(
                        request.request.remote.clone(),
                    ));
                out.push(Err(FailedIO {
                    data_flow_id: request.data_flow_id,
                    operator_idx: request.operator_idx,
                    pending_read_id: request.pending_read_id,
                    error: error.into(),
                }));
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

    /// `true` while any HTTP GET or upload is in flight.
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
