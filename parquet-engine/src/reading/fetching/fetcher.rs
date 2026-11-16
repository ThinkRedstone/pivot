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
//! Row groups reach the fetcher from three places:
//!
//! - Those expected to be in memory arrive on its input channel and are taken
//!   on the worker's CPU turn, which reaches the fetcher only when nothing
//!   further down the pipeline has work. One that is in memory is sent on
//!   straight away. One that needs a read after all is not read there: it is
//!   pushed to the queue of uncached row groups, below.
//! - Those expected to need a read wait in a receiver the fetcher holds
//!   itself.
//! - The uncached row groups, shared by every worker's fetcher, are kept in
//!   the order they were found.
//!
//! The next read asked for is always an uncached row group when there is one,
//! and only then one expected to need a read: the cache holds part of an
//! uncached row group, so its read is the smaller one and those pages are
//! there to be used before they are evicted. A plain scan asks for the next
//! read whenever the worker saturates the available IO, ahead of its CPU work,
//! so the read is on its way before the worker needs it. A Top-N scan asks
//! only on the CPU turn, once no in-memory row group is left: a row group it
//! asks for early is judged against a boundary that has not tightened yet,
//! and is read for nothing.
//!
//! Every read counts its row group's projected bytes against the worker's
//! read-ahead budget for the medium the table is read from (see
//! [`dispatch::io::ReadAheadBudget`] and
//! [`ParquetTable::get_read_medium`](crate::ParquetTable::get_read_medium)),
//! and the fetcher asks for one only while that budget has room.
//!
//! A row group whose read has come back is not sent on at once. It waits in
//! the fetcher, still counted against the budget, and is sent on once no
//! in-memory row group is left, so what is in memory is worked on first.

use crate::types::metadata::QueryRowGroupMetadata;
use crate::types::projection::Projection;
use crate::types::requests::{RowGroupBuffer, RowGroupRequest};
use dispatch::Receiver;
use dispatch::Sender;
use dispatch::Unary;
use dispatch::WorkStatus;
use dispatch::io::{OperatorIO, ReadAheadCharge, ReadMedium, ReadRequestId, ReadResponse};
use dispatch::memory::memory_ctx;
use dispatch::{SharedWorkQueueReceiver, SharedWorkQueueSender};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

/// `IoRowGroupsReceiver` is the source of the row groups expected to need a
/// read: a table scan's injector, or the queue a late materialization sends
/// them to.
pub struct RowGroupFetcher<IoRowGroupsReceiver: Receiver<QueryRowGroupMetadata>> {
    /// The columns the scan reads from every row group.
    projection: Arc<Projection>,
    /// The medium the table is read from, whose read-ahead budget the
    /// fetcher's reads count against.
    read_medium: ReadMedium,
    /// Whether the row groups expected to need a read are asked for ahead of
    /// the worker's CPU work. A Top-N scan asks for them on the CPU turn.
    reads_ahead_of_cpu_work: bool,
    /// The row groups expected to need a read.
    io_row_groups_receiver: IoRowGroupsReceiver,
    /// Where a row group that was expected to be in memory and needs a read
    /// is pushed, for any worker's fetcher to ask for.
    uncached_row_groups_sender: SharedWorkQueueSender<QueryRowGroupMetadata>,
    uncached_row_groups_receiver: SharedWorkQueueReceiver<QueryRowGroupMetadata>,
    /// The requests whose reads are on their way, each with what it counts
    /// against the worker's read-ahead budget.
    in_flight: HashMap<ReadRequestId, (RowGroupRequest, ReadAheadCharge)>,
    /// The row groups whose reads are back, oldest first, waiting to be sent
    /// on until no in-memory row group is left.
    fetched_row_groups: VecDeque<RowGroupBuffer>,
}

impl<I: Receiver<QueryRowGroupMetadata>> RowGroupFetcher<I> {
    pub fn new(
        projection: Arc<Projection>,
        read_medium: ReadMedium,
        reads_ahead_of_cpu_work: bool,
        io_row_groups_receiver: I,
        uncached_row_groups_sender: SharedWorkQueueSender<QueryRowGroupMetadata>,
        uncached_row_groups_receiver: SharedWorkQueueReceiver<QueryRowGroupMetadata>,
    ) -> Self {
        Self {
            projection,
            read_medium,
            reads_ahead_of_cpu_work,
            io_row_groups_receiver,
            uncached_row_groups_sender,
            uncached_row_groups_receiver,
            in_flight: HashMap::new(),
            fetched_row_groups: VecDeque::new(),
        }
    }

    /// What `row_group` counts against a read-ahead budget: its whole
    /// projected size.
    fn compute_projected_bytes(&self, row_group: &QueryRowGroupMetadata) -> usize {
        let projected_bytes: i64 = self
            .projection
            .indices()
            .iter()
            .map(|&column| row_group.row_group_metadata.columns[column].total_compressed_size)
            .sum();
        projected_bytes as usize
    }

    /// Ask for the next read while the read-ahead budget has room: an
    /// uncached row group when there is one, otherwise one expected to need a
    /// read, this worker's own queue first, then a peer's.
    fn issue_next_read(&mut self, io: &mut OperatorIO) -> dispatch::UnaryResult<WorkStatus> {
        if !io.has_read_ahead_room(self.read_medium) {
            return Ok(WorkStatus::Pending);
        }
        let next = self
            .uncached_row_groups_receiver
            .try_recv()
            .or_else(|| self.io_row_groups_receiver.try_recv())
            .or_else(|| self.io_row_groups_receiver.steal());
        let Some(row_group) = next else {
            return Ok(WorkStatus::Pending);
        };

        let read_ahead_charge =
            io.charge_read_ahead(self.read_medium, self.compute_projected_bytes(&row_group));
        let request = RowGroupRequest::from(row_group, &self.projection);
        if request.reads_nothing() {
            // In memory after all. It still waits its turn behind the row
            // groups that were expected to be.
            self.fetched_row_groups
                .push_back(request.into_row_group_buffer(None, Some(read_ahead_charge)));
        } else {
            let id = io.read(
            &memory_ctx().page_caches().compressed,
            request.open_file().clone(),
            request.file_ranges(),
        )?;
        self.in_flight.insert(id, (request, read_ahead_charge));
        }
        Ok(WorkStatus::Ran)
    }
}

impl<I: Receiver<QueryRowGroupMetadata>>
    Unary<QueryRowGroupMetadata, RowGroupBuffer> for RowGroupFetcher<I>
{
    /// Take a row group expected to be in memory. One that is builds its
    /// buffer straight away and counts against no budget. One that needs a
    /// read is pushed to the uncached row groups, to be asked for once no
    /// in-memory row group is left.
    fn consume(
        &mut self,
        row_group: QueryRowGroupMetadata,
        sender: &mut dyn Sender<RowGroupBuffer>,
        _io: &mut OperatorIO,
    ) -> dispatch::UnaryResult<()> {
        let request = RowGroupRequest::from(row_group, &self.projection);
        if request.reads_nothing() {
            sender.send(request.into_row_group_buffer(None, None))?;
            return Ok(());
        }
        self.uncached_row_groups_sender
            .send(request.into_metadata())?;
        Ok(())
    }

    fn saturates_available_io(&self) -> bool {
        true
    }

    /// Ask for the next read ahead of the worker's CPU work, unless this is a
    /// scan that asks only on the CPU turn.
    fn saturate_available_io(
        &mut self,
        _sender: &mut dyn Sender<RowGroupBuffer>,
        io: &mut OperatorIO,
    ) -> dispatch::UnaryResult<WorkStatus> {
        if !self.reads_ahead_of_cpu_work {
            return Ok(WorkStatus::Pending);
        }
        self.issue_next_read(io)
    }

    fn process_read_response(
        &mut self,
        _sender: &mut dyn Sender<RowGroupBuffer>,
        _io: &mut OperatorIO,
        response: ReadResponse,
    ) -> dispatch::UnaryResult<()> {
        let (request, read_ahead_charge) = self
            .in_flight
            .remove(&response.id())
            .expect("response for an unknown row-group request");
        self.fetched_row_groups
            .push_back(request.into_row_group_buffer(Some(response), Some(read_ahead_charge)));
        Ok(())
    }

    /// The worker's CPU turn with no in-memory row group to take. Ask for the
    /// next read while the budget has room. Otherwise send on a row group
    /// whose read is back.
    fn run(
        &mut self,
        sender: &mut dyn Sender<RowGroupBuffer>,
        io: &mut OperatorIO,
    ) -> dispatch::UnaryResult<WorkStatus> {
        if let WorkStatus::Ran = self.issue_next_read(io)? {
            return Ok(WorkStatus::Ran);
        }
        match self.fetched_row_groups.pop_front() {
            Some(row_group_buffer) => {
                sender.send(row_group_buffer)?;
                Ok(WorkStatus::Ran)
            }
            None => Ok(WorkStatus::Pending),
        }
    }

    fn finish(&mut self, _sender: &mut dyn Sender<RowGroupBuffer>) -> dispatch::UnaryResult<bool> {
        Ok(self.in_flight.is_empty() && self.fetched_row_groups.is_empty())
    }

    /// The row groups still to be asked for or sent on keep the fetcher from
    /// finishing. Those in the shared queues do so on every worker: any of
    /// them may be the one to take the next.
    fn has_pending_work(&self) -> bool {
        !self.fetched_row_groups.is_empty()
            || !self.io_row_groups_receiver.is_empty()
            || !self.uncached_row_groups_receiver.is_empty()
    }
}
