//! Per-dataflow execution counters and the collector that accumulates them.
//!
//! A [`DataFlow`](crate::data_flow::DataFlow) owns a [`StatsCollector`] and
//! exposes it via `stats()`; the worker drives it (`flow.stats().record_*`) as it
//! issues reads, sees them complete, and runs operator work, then ships the tally
//! with [`report`](StatsCollector::report) when the dataflow finishes. The
//! [`DataFlowHandle`](crate::DataFlowHandle) folds every worker's. Collection is
//! gated so it costs nothing — no clock reads, no counting — when a query didn't
//! opt in.

use crate::io::{DataFlowRequest, FsRequest, ReadyBytesLen, RemoteReadTime, RemoteSplit};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// Execution counters for one dataflow, summed across the workers that ran it.
///
/// The `cpu` and `*_time` figures are sums (over workers, and for the IO times
/// over in-flight operations), so they can each exceed the query's wall time. The
/// shape they reveal is the point: a high `*_time` relative to `cpu` means the
/// dataflow spent its time waiting on I/O. A remote read split across tiers counts
/// each piece in its own tier, so the HTTP and disk-cache counters never hide each
/// other; GET/upload and read/write latency also stay in separate time buckets.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct DataFlowStats {
    /// HTTP operations issued: uploads, plus one fetch per hole piece of a
    /// remote read. A read split across the cache and the network counts a fetch
    /// here and a cache read under `disk_cache_requests`.
    pub http_requests: u64,
    /// Bytes transferred over HTTP, summed over fetches and uploads.
    pub http_bytes: u64,
    /// Wall time HTTP GETs were in flight.
    pub http_get_time: Duration,
    /// HTTP fetches issued again because their connection stalled. A count of
    /// these well above zero means packets are being lost on the way in.
    pub http_reissues: u64,
    /// Wall time HTTP uploads were in flight.
    pub http_upload_time: Duration,
    /// Cache-file reads issued, one per resident piece of a remote read (a hit
    /// serving a read or part of one). Pairs with `disk_cache_time`.
    pub disk_cache_requests: u64,
    /// Bytes served from the on-disk cache file (the resident pieces), summed over
    /// every cache read.
    pub disk_cache_bytes: u64,
    /// Wall time the cache-file reads were in flight, summed over every cache read.
    pub disk_cache_time: Duration,
    /// Filesystem operations issued (local object reads and writes).
    pub disk_requests: u64,
    /// Bytes transferred by those filesystem operations.
    pub disk_bytes: u64,
    /// Wall time local-filesystem reads were in flight.
    pub disk_read_time: Duration,
    /// Wall time local-filesystem writes were in flight.
    pub disk_write_time: Duration,
    /// CPU time spent in this dataflow's operators (decode, group-by, …), summed
    /// across workers. Excludes time parked waiting on IO.
    pub cpu: Duration,
}

impl DataFlowStats {
    /// Fold one worker's tally into the running total.
    pub fn merge(&mut self, other: &Self) {
        self.http_requests += other.http_requests;
        self.http_bytes += other.http_bytes;
        self.http_get_time += other.http_get_time;
        self.http_reissues += other.http_reissues;
        self.http_upload_time += other.http_upload_time;
        self.disk_cache_requests += other.disk_cache_requests;
        self.disk_cache_bytes += other.disk_cache_bytes;
        self.disk_cache_time += other.disk_cache_time;
        self.disk_requests += other.disk_requests;
        self.disk_bytes += other.disk_bytes;
        self.disk_read_time += other.disk_read_time;
        self.disk_write_time += other.disk_write_time;
        self.cpu += other.cpu;
    }

    /// Log the IO tally at WARN when any operation happened, for a path that has no
    /// other reporting: a failed query never reaches the server's stats NOTICE,
    /// so this keeps a slow or failed scan diagnosable. A no-op when the query did
    /// no IO (or did not opt into stats, leaving every counter zero).
    pub fn log_failed_query(&self) {
        if self.http_requests + self.disk_requests + self.disk_cache_requests == 0 {
            return;
        }
        tracing::warn!(
            http_requests = self.http_requests,
            http_mib = self.http_bytes >> 20,
            http_get_ms = self.http_get_time.as_millis() as u64,
            http_reissues = self.http_reissues,
            http_upload_ms = self.http_upload_time.as_millis() as u64,
            disk_cache_requests = self.disk_cache_requests,
            disk_cache_mib = self.disk_cache_bytes >> 20,
            disk_requests = self.disk_requests,
            disk_mib = self.disk_bytes >> 20,
            disk_read_ms = self.disk_read_time.as_millis() as u64,
            disk_write_ms = self.disk_write_time.as_millis() as u64,
            cpu_ms = self.cpu.as_millis() as u64,
            "query failed; IO completed before the failure"
        );
    }
}

/// Accumulates one worker's [`DataFlowStats`] for a dataflow, or nothing when the
/// query didn't ask for stats — in which case every `record_*` is a no-op and no
/// clock is read. Owned by the [`DataFlow`](crate::data_flow::DataFlow) and driven
/// by the worker through `flow.stats()`.
pub struct StatsCollector {
    /// `Some` only when stats are on.
    stats: Option<DataFlowStats>,
    /// Where [`report`](Self::report) ships the tally for the handle to fold.
    tx: mpsc::Sender<DataFlowStats>,
}

impl StatsCollector {
    pub(crate) fn new(tx: mpsc::Sender<DataFlowStats>, enabled: bool) -> Self {
        Self {
            stats: enabled.then(DataFlowStats::default),
            tx,
        }
    }

    /// Build a no-op collector for standalone test and benchmark drivers.
    #[cfg(any(test, feature = "test-util"))]
    pub fn disabled() -> Self {
        let (tx, _rx) = mpsc::channel();
        Self { stats: None, tx }
    }

    /// Whether collection is on, so a caller can skip a clock read it would only
    /// hand to a no-op.
    pub fn enabled(&self) -> bool {
        self.stats.is_some()
    }

    /// Stamp each read with its issue time so its completion can bill the read's
    /// in-flight latency. Shared by the local-filesystem and remote submit paths.
    pub fn stamp_issued<R>(&mut self, requests: &mut [DataFlowRequest<R>]) {
        if self.stats.is_some() {
            let now = Instant::now();
            for request in requests.iter_mut() {
                request.submitted_at = Some(now);
            }
        }
    }

    /// Count local-filesystem operations, total their bytes, and stamp them so
    /// their completion can bill the matching read or write time.
    pub fn record_issued_disk(&mut self, requests: &mut [DataFlowRequest<FsRequest>]) {
        if let Some(stats) = &mut self.stats {
            stats.disk_requests += requests.len() as u64;
            stats.disk_bytes += requests
                .iter()
                .map(|request| request.request.ready_bytes_len())
                .sum::<u64>();
        }
        self.stamp_issued(requests);
    }

    /// Count a remote operation against the transports that serve it. Unlike
    /// the disk path this is known only after the requester resolves a GET
    /// against the cache or accepts an upload.
    pub fn record_issued_remote(&mut self, split: RemoteSplit) {
        if let Some(stats) = &mut self.stats {
            stats.disk_cache_requests += split.disk_cache_requests;
            stats.disk_cache_bytes += split.disk_cache_bytes;
            stats.http_requests += split.http_requests;
            stats.http_bytes += split.http_bytes;
        }
    }

    /// Bill a completed remote read's in-flight time, each piece's wait charged to
    /// the tier that served it (see [`RemoteReadTime`]).
    pub fn record_http_get_time(&mut self, time: RemoteReadTime) {
        if let Some(stats) = &mut self.stats {
            stats.http_get_time += time.http;
            stats.http_reissues += time.http_reissues;
            stats.disk_cache_time += time.disk_cache;
        }
    }

    /// Bill one completed HTTP upload from submission through response drain.
    pub fn record_http_upload_time(&mut self, submitted_at: Option<Instant>) {
        self.record_elapsed(submitted_at, |stats| &mut stats.http_upload_time);
    }

    /// Bill a completed local-filesystem read's in-flight time.
    pub fn record_disk_read_time(&mut self, submitted_at: Option<Instant>) {
        self.record_elapsed(submitted_at, |stats| &mut stats.disk_read_time);
    }

    /// Bill a completed local-filesystem write's in-flight time.
    pub fn record_disk_write_time(&mut self, submitted_at: Option<Instant>) {
        self.record_elapsed(submitted_at, |stats| &mut stats.disk_write_time);
    }

    fn record_elapsed(
        &mut self,
        submitted_at: Option<Instant>,
        bucket: fn(&mut DataFlowStats) -> &mut Duration,
    ) {
        if let (Some(stats), Some(submitted_at)) = (&mut self.stats, submitted_at) {
            *bucket(stats) += submitted_at.elapsed();
        }
    }

    /// Add `elapsed` to the dataflow's operator CPU time.
    pub fn record_cpu(&mut self, elapsed: Duration) {
        if let Some(stats) = &mut self.stats {
            stats.cpu += elapsed;
        }
    }

    /// Ship this worker's tally for the handle to fold (a no-op when off).
    pub fn report(&self) {
        if let Some(stats) = self.stats {
            let _ = self.tx.send(stats);
        }
    }
}
