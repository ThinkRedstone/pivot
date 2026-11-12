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
//! Disk and HTTP have very different latencies, so what bounds a worker's
//! claims depends on where the table's files are (see [`ClaimBound`]): a
//! fixed, small number of row groups for local files, and the worker's
//! read-ahead budget of bytes for remote ones.

use crate::types::metadata::QueryRowGroupMetadata;
use crate::types::projection::Projection;
use crate::types::requests::{RowGroupBuffer, RowGroupRequest};
use dispatch::Sender;
use dispatch::Unary;
use dispatch::io::{OperatorIO, ReadAheadCharge, ReadRequestId, ReadResponse};
use dispatch::memory::memory_ctx;
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

/// What bounds the row groups one worker may hold claimed while scanning a
/// table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClaimBound {
    /// A fixed number of row groups claimed and not yet cut into decode
    /// ranges.
    RowGroups(usize),
    /// The worker's read-ahead budget (see [`dispatch::io::ReadAheadBudget`]):
    /// the worker asks the fetcher for another row group while it has asked
    /// for less than the budget, and each claim counts its row group's bytes
    /// against the budget until a worker starts on the row group.
    ReadAheadBudget,
}

/// What bounds one worker's claims while scanning `table`. Local tables keep
/// claims minimal for decode fairness. A table with any remote file reads
/// ahead against the worker's read-ahead budget: a remote claim spends most of
/// its life waiting on the network, so its reads must be on their way long
/// before its decode is due, or every row group pays a full round trip
/// serially.
pub fn pending_claim_bound(table: &crate::ParquetTable) -> ClaimBound {
    let any_remote = table
        .row_groups()
        .iter()
        .any(|rg| matches!(rg.open_file, dispatch::io::OpenFile::Remote(_)));
    if any_remote {
        ClaimBound::ReadAheadBudget
    } else {
        ClaimBound::RowGroups(MAX_PENDING_LOCAL_ROW_GROUPS)
    }
}

pub struct RowGroupFetcher {
    /// The columns the scan reads from every row group.
    projection: Arc<Projection>,
    /// The requests whose reads are on their way, each with what it counts
    /// against the worker's read-ahead budget.
    in_flight: HashMap<ReadRequestId, (RowGroupRequest, Option<ReadAheadCharge>)>,
    /// How many row groups this worker has claimed but not yet cut into
    /// decode ranges. Incremented here per claim, decremented by the worker's
    /// range cutter when a row group is cut (or pruned), and consulted as
    /// claim backpressure under [`ClaimBound::RowGroups`].
    pending_row_groups: Arc<AtomicUsize>,
    claim_bound: ClaimBound,
}

impl RowGroupFetcher {
    pub fn new(
        projection: Arc<Projection>,
        pending_row_groups: Arc<AtomicUsize>,
        claim_bound: ClaimBound,
    ) -> Self {
        Self {
            projection,
            in_flight: HashMap::new(),
            pending_row_groups,
            claim_bound,
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
        let read_ahead_charge = match self.claim_bound {
            ClaimBound::RowGroups(_) => None,
            // The row group's whole projected size, wherever its bytes come
            // from: one served from memory is as much work waiting for the
            // worker as one read from the store.
            ClaimBound::ReadAheadBudget => {
                let projected_bytes: i64 = self
                    .projection
                    .indices()
                    .iter()
                    .map(|&column| {
                        row_group.row_group_metadata.columns[column].total_compressed_size
                    })
                    .sum();
                Some(io.charge_read_ahead(projected_bytes as usize))
            }
        };
        let request = RowGroupRequest::from(row_group, &self.projection);
        if request.reads_nothing() {
            // Every projected leaf is absent from this file or already
            // decompressed in memory, so the buffer is built straight away.
            sender.send(request.into_row_group_buffer(None, read_ahead_charge))?;
            return Ok(());
        }
        let id = io.read(
            &memory_ctx().page_caches().compressed,
            request.open_file().clone(),
            request.file_ranges(),
        )?;
        self.in_flight.insert(id, (request, read_ahead_charge));
        Ok(())
    }

    fn ready_for_more_work(&mut self) -> bool {
        let disk_in_flight = self
            .in_flight
            .values()
            .filter(|(request, _)| matches!(request.open_file(), dispatch::io::OpenFile::Local(_)))
            .count();
        if disk_in_flight >= MAX_DISK_IN_FLIGHT {
            return false;
        }
        match self.claim_bound {
            ClaimBound::RowGroups(max_pending_row_groups) => {
                self.pending_row_groups.load(Ordering::Relaxed) < max_pending_row_groups
            }
            // The worker asks for another row group only while it has room.
            ClaimBound::ReadAheadBudget => true,
        }
    }

    fn takes_input_to_saturate_io(&self) -> bool {
        self.claim_bound == ClaimBound::ReadAheadBudget
    }

    fn process_read_response(
        &mut self,
        sender: &mut dyn Sender<RowGroupBuffer>,
        _io: &mut OperatorIO,
        response: ReadResponse,
    ) -> dispatch::UnaryResult<()> {
        let (request, read_ahead_charge) = self
            .in_flight
            .remove(&response.id())
            .expect("response for an unknown row-group request");
        sender.send(request.into_row_group_buffer(Some(response), read_ahead_charge))?;
        Ok(())
    }

    fn finish(&mut self, _sender: &mut dyn Sender<RowGroupBuffer>) -> dispatch::UnaryResult<bool> {
        Ok(self.in_flight.is_empty())
    }
}
