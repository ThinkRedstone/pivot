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
//! What bounds a worker's claims is a read-ahead budget of bytes (see
//! [`dispatch::io::ReadAheadBudget`]): the worker's budget for the network
//! when the table's files are remote, and its budget for disks otherwise (see
//! [`ParquetTable::get_read_medium`](crate::ParquetTable::get_read_medium)).
//! The worker asks the fetcher for another row group
//! while it has asked for less than that budget, and each claim counts its
//! row group's projected bytes against it until a worker starts on the row
//! group.

use crate::types::metadata::QueryRowGroupMetadata;
use crate::types::projection::Projection;
use crate::types::requests::{RowGroupBuffer, RowGroupRequest};
use dispatch::Sender;
use dispatch::Unary;
use dispatch::io::{OperatorIO, ReadAheadCharge, ReadMedium, ReadRequestId, ReadResponse};
use dispatch::memory::memory_ctx;
use std::collections::HashMap;
use std::sync::Arc;

pub struct RowGroupFetcher {
    /// The columns the scan reads from every row group.
    projection: Arc<Projection>,
    /// The medium whose read-ahead budget the claims count against.
    read_medium: ReadMedium,
    /// The requests whose reads are on their way, each with what it counts
    /// against the worker's read-ahead budget.
    in_flight: HashMap<ReadRequestId, (RowGroupRequest, ReadAheadCharge)>,
}

impl RowGroupFetcher {
    pub fn new(projection: Arc<Projection>, read_medium: ReadMedium) -> Self {
        Self {
            projection,
            read_medium,
            in_flight: HashMap::new(),
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
        // The row group's whole projected size, wherever its bytes come
        // from: one served from memory is as much work waiting for the worker
        // as one read from storage.
        let projected_bytes: i64 = self
            .projection
            .indices()
            .iter()
            .map(|&column| row_group.row_group_metadata.columns[column].total_compressed_size)
            .sum();
        let read_ahead_charge = io.charge_read_ahead(self.read_medium, projected_bytes as usize);
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

    fn takes_input_to_saturate_io(&self) -> bool {
        true
    }

    fn has_read_ahead_room(&self, io: &OperatorIO) -> bool {
        io.has_read_ahead_room(self.read_medium)
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
