//! The [`RowGroupFetcher`]: drives the column-chunk IO for in-flight row groups.
//!
//! It accepts claimed row groups ([`QueryRowGroupMetadata`]), takes the pages
//! the decompressed cache already holds for each one, and reads the rest of
//! its projected column chunks — local files via io_uring disk reads, remote
//! objects via HTTP range reads on the same ring — emitting a
//! [`RowGroupBuffer`] once a row group's last block lands. The
//! slot/routing/in-flight bookkeeping lives in the dispatch requester; this
//! type retains only the [`RowGroupRequest`] keyed by the logical request id.
//!
//! Disk and HTTP have very different latencies, so the fetcher bounds logical
//! row-group reads for each medium independently. Dispatch separately bounds the
//! resulting physical submission batches. A homogeneous scan only fills its
//! own medium's pool, so the other never gates it.

use crate::types::metadata::QueryRowGroupMetadata;
use crate::types::projection::Projection;
use crate::types::requests::{RowGroupBuffer, RowGroupRequest};
use dispatch::Sender;
use dispatch::Unary;
use dispatch::io::{OperatorIO, ReadRequestId, ReadResponse};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Disk-backed read blocks outstanding per worker before admitting another row
/// group. One keeps disk reads serial — io_uring gives a single read ample
/// depth.
const MAX_DISK_IN_FLIGHT: usize = 1;

/// Row groups a worker may hold "claimed but not yet cut into decode ranges"
/// before it stops claiming more, when the table's files are local: the one
/// whose pages are arriving plus one prefetched next, the minimum that keeps
/// its pipeline overlapped. A row group's pages are collected only on the
/// worker that claimed it (the range cutter holds per-row-group state), so
/// every claim beyond this bound queues more pages behind one worker while
/// others may sit idle. That matters most when a scan has few row groups per
/// worker, where a single worker hoarding a third row group while the queue
/// runs dry sets the whole query's critical path.
const MAX_PENDING_LOCAL_ROW_GROUPS: usize = 2;

/// The remote-file counterpart of [`MAX_PENDING_LOCAL_ROW_GROUPS`]. A remote
/// claim spends most of its life waiting on the network, not decoding, so the
/// hoarding concern above barely applies while read-ahead depth is the whole
/// throughput game: a claim's reads must be in flight long before its decode
/// is due, or every row group pays a full network round trip serially.
const MAX_PENDING_REMOTE_ROW_GROUPS: usize = 16;

/// How many undecoded row groups one worker may hold claims on while
/// scanning `table`. Local tables keep claims minimal for decode fairness;
/// tables with any remote file get read-ahead depth.
pub fn pending_claim_bound(table: &crate::ParquetTable) -> usize {
    let any_remote = table
        .row_groups()
        .iter()
        .any(|rg| matches!(rg.open_file, dispatch::io::OpenFile::Remote(_)));
    if any_remote {
        MAX_PENDING_REMOTE_ROW_GROUPS
    } else {
        MAX_PENDING_LOCAL_ROW_GROUPS
    }
}

pub struct RowGroupFetcher {
    /// The columns the scan reads from every row group.
    projection: Arc<Projection>,
    in_flight: HashMap<ReadRequestId, RowGroupRequest>,
    /// How many row groups this worker has claimed but not yet cut into
    /// decode ranges. Incremented here per claim, decremented by the worker's
    /// range cutter when a row group is cut (or pruned), and consulted as
    /// claim backpressure.
    pending_row_groups: Arc<AtomicUsize>,
    /// The claim bound for this scan (see [`pending_claim_bound`]).
    max_pending_row_groups: usize,
}

impl RowGroupFetcher {
    pub fn new(
        projection: Arc<Projection>,
        pending_row_groups: Arc<AtomicUsize>,
        max_pending_row_groups: usize,
    ) -> Self {
        Self {
            projection,
            in_flight: HashMap::new(),
            pending_row_groups,
            max_pending_row_groups,
        }
    }
}

impl Unary<QueryRowGroupMetadata, RowGroupBuffer> for RowGroupFetcher {
    fn consume(
        &mut self,
        row_group: QueryRowGroupMetadata,
        sender: &mut dyn Sender<RowGroupBuffer>,
        io: &mut OperatorIO,
    ) -> dispatch::UnaryResult<()> {
        self.pending_row_groups.fetch_add(1, Ordering::Relaxed);
        let request = RowGroupRequest::from(row_group, &self.projection);
        if request.reads_nothing() {
            // Every projected leaf is absent from this file or already
            // decompressed in memory, so the buffer is built straight away.
            sender.send(request.into_row_group_buffer(None))?;
            return Ok(());
        }
        let id = io.read(request.open_file().clone(), request.file_ranges())?;
        self.in_flight.insert(id, request);
        Ok(())
    }

    fn ready_for_more_work(&mut self) -> bool {
        let disk_in_flight = self
            .in_flight
            .values()
            .filter(|request| matches!(request.open_file(), dispatch::io::OpenFile::Local(_)))
            .count();
        let http_in_flight = self.in_flight.len() - disk_in_flight;
        disk_in_flight < MAX_DISK_IN_FLIGHT
            && http_in_flight < crate::http_readahead()
            && self.pending_row_groups.load(Ordering::Relaxed) < self.max_pending_row_groups
    }

    fn process_read_response(
        &mut self,
        sender: &mut dyn Sender<RowGroupBuffer>,
        _io: &mut OperatorIO,
        response: ReadResponse,
    ) -> dispatch::UnaryResult<()> {
        let request = self
            .in_flight
            .remove(&response.id())
            .expect("response for an unknown row-group request");
        sender.send(request.into_row_group_buffer(Some(response)))?;
        Ok(())
    }

    fn finish(&mut self, _sender: &mut dyn Sender<RowGroupBuffer>) -> dispatch::UnaryResult<bool> {
        Ok(self.in_flight.is_empty())
    }
}
