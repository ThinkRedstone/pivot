//! The logical I/O interface exposed to operators.
//!
//! Operators describe *what* file bytes they need. They do not look up cache
//! entries, allocate cache extents, or retain destination pointers. The
//! worker's requester performs those physical tasks the moment the operator
//! asks, through the [`OperatorIO`] handle dispatch wraps around each call.

use super::{LocalFile, OpenFile};
use crate::io::requester::resolve_read_through_cache;
use crate::memory::cache::{CacheId, CompressedCache, RingCache};
use crate::memory::{FileBytes, Slab};
use bytes::Bytes;
use std::cell::RefCell;
use std::sync::Arc;

/// Identifies one logical read within an operator node.
///
/// Identifiers are local to an [`OperatorIO`]. Dispatch routes them together
/// with the dataflow and operator identifiers, so operators can use this small
/// value directly as a key in their own in-flight maps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReadRequestId(pub usize);

/// Identifies one write issued through an [`OperatorIO`]. Handed back with the
/// completed request, so an operator with several writes in flight can tell
/// which one finished. Drawn from the same per-dataflow counter as read ids.
///
/// The top bit marks a spill write. Those are issued by the dataflow on an
/// operator's behalf and completed by the dataflow, so an operator never sees
/// one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WriteRequestId(pub usize);

impl WriteRequestId {
    const SPILL_BIT: usize = 1 << (usize::BITS - 1);

    fn next(sequence: &mut usize, spill: bool) -> Self {
        let id = *sequence;
        *sequence += 1;
        assert!(id & Self::SPILL_BIT == 0, "write request ids exhausted");
        Self(if spill { id | Self::SPILL_BIT } else { id })
    }

    /// Whether this write is a spill, which the dataflow completes itself.
    pub fn is_spill(self) -> bool {
        self.0 & Self::SPILL_BIT != 0
    }
}

/// One byte range requested from a file.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FileRange {
    pub offset: usize,
    pub len: usize,
}

impl FileRange {
    pub fn new(offset: usize, len: usize) -> Self {
        Self { offset, len }
    }
}

/// One requested file range, resolved: the bytes stored in the file at
/// `offset`, served by the compressed cache. Several buffers are possible
/// because one logical range may cross cache extents; they are in file order.
pub struct ReadData {
    pub offset: usize,
    pub bytes: Vec<Bytes>,
}

/// The completed result of one logical read.
pub struct ReadResponse {
    id: ReadRequestId,
    open_file: OpenFile,
    /// One entry per requested [`FileRange`], in the same order.
    locations: Vec<ReadData>,
}

impl ReadResponse {
    pub(crate) fn new(id: ReadRequestId, open_file: OpenFile, locations: Vec<ReadData>) -> Self {
        Self {
            id,
            open_file,
            locations,
        }
    }

    pub fn id(&self) -> ReadRequestId {
        self.id
    }

    pub fn open_file(&self) -> &OpenFile {
        &self.open_file
    }

    pub fn into_locations(self) -> Vec<ReadData> {
        self.locations
    }

    /// Consume a one-location response and concatenate its cache fragments.
    pub fn into_bytes(self) -> Vec<u8> {
        assert_eq!(
            self.locations.len(),
            1,
            "into_bytes requires exactly one requested location"
        );
        let mut bytes = Vec::new();
        for run in self.locations.into_iter().next().unwrap().bytes {
            bytes.extend_from_slice(&run);
        }
        bytes
    }
}

/// A logical read waiting for the worker to register it with the cache.
pub struct PendingReadRequest {
    pub id: ReadRequestId,
    pub open_file: OpenFile,
    pub locations: Vec<FileRange>,
    /// The compressed cache the bytes are read through: the one of the family
    /// the data belongs to, so they are balanced against its derived forms.
    pub cache: CacheId,
}

/// One range of a direct read: the `slab.size()` bytes at `offset` of the
/// file, read into `slab`.
pub struct SlabRead {
    pub offset: usize,
    pub slab: Slab,
}

/// A logical read of a local file around the compressed cache, into slabs
/// the reader supplies.
pub struct PendingDirectReadRequest {
    pub id: ReadRequestId,
    pub file: LocalFile,
    pub reads: Vec<SlabRead>,
}

/// A logical write waiting for the worker to select its transport.
pub struct PendingWriteRequest {
    pub id: WriteRequestId,
    pub open_file: OpenFile,
    /// Where in a local file `data` goes; a remote object is always written
    /// whole.
    pub offset: u64,
    pub data: Arc<FileBytes>,
}

/// The logical I/O handle passed to an operator whenever it runs.
///
/// A thin borrow of the worker's requester, bound to the operator's routing
/// identity by dispatch. A read or write submits immediately; the operator
/// retains only its logical request identifiers and domain state, and never
/// sees the requester's worker-lifecycle surface.
pub struct OperatorIO<'a> {
    /// The worker's requester, borrowed for the length of each call rather
    /// than held: an operator allocating ring memory mid-run may have to
    /// evict, and eviction borrows the requester to wait on spill writes.
    requester: &'a RefCell<crate::io::IORequester>,
    data_flow_id: crate::Identifier,
    operator_idx: crate::Identifier,
    stats: &'a mut crate::stats::StatsCollector,
    /// The dataflow's request-id counter for reads and writes, owned by the
    /// flow so identifiers stay unique across this handle's short lifetimes.
    next_request_id: &'a mut usize,
}

impl<'a> OperatorIO<'a> {
    pub(crate) fn new(
        requester: &'a RefCell<crate::io::IORequester>,
        data_flow_id: crate::Identifier,
        operator_idx: crate::Identifier,
        stats: &'a mut crate::stats::StatsCollector,
        next_request_id: &'a mut usize,
    ) -> Self {
        Self {
            requester,
            data_flow_id,
            operator_idx,
            stats,
            next_request_id,
        }
    }

    /// Read file bytes through the compressed cache `cache`, so what the read
    /// brings in is cached beside that cache's family.
    pub fn read(
        &mut self,
        cache: &CompressedCache,
        open_file: OpenFile,
        locations: impl IntoIterator<Item = FileRange>,
    ) -> Result<ReadRequestId, crate::io::IORequesterError> {
        let id = ReadRequestId(*self.next_request_id);
        *self.next_request_id += 1;
        let request = PendingReadRequest {
            id,
            open_file,
            locations: locations.into_iter().collect(),
            cache: cache.id(),
        };
        // Resolving may allocate cache slots, and so evict, before the borrow.
        let cache_lookups = resolve_read_through_cache(&request);
        self.requester.borrow_mut().request_read(
            self.data_flow_id,
            self.operator_idx,
            request,
            cache_lookups,
            self.stats,
        )?;
        Ok(id)
    }

    /// Read ranges of `file` around the compressed cache, each into the slab
    /// given for it. For bytes only their writer reads back, such as a spill
    /// file's: a cache fill reads whole blocks, and so bytes that writes still
    /// in flight are changing, and keeps them. The response comes back the
    /// same way as [`read`](Self::read)'s, one location per range in the
    /// order given, each a view of its slab.
    pub fn read_direct(
        &mut self,
        file: LocalFile,
        reads: Vec<SlabRead>,
    ) -> Result<ReadRequestId, crate::io::IORequesterError> {
        let id = ReadRequestId(*self.next_request_id);
        *self.next_request_id += 1;
        self.requester.borrow_mut().request_read_direct(
            self.data_flow_id,
            self.operator_idx,
            PendingDirectReadRequest { id, file, reads },
            self.stats,
        )?;
        Ok(id)
    }

    /// Write `data` to a local file or remote object. Returns the id the
    /// completed write will carry back to the operator.
    pub fn write(
        &mut self,
        open_file: OpenFile,
        data: Arc<FileBytes>,
    ) -> Result<WriteRequestId, crate::io::IORequesterError> {
        let id = WriteRequestId::next(self.next_request_id, false);
        let request = PendingWriteRequest {
            id,
            open_file,
            offset: 0,
            data,
        };
        self.requester.borrow_mut().request_write(
            self.data_flow_id,
            self.operator_idx,
            request,
            self.stats,
        )?;
        Ok(id)
    }

    /// Write spilled `data` to `open_file` at `offset`. Like
    /// [`write`](Self::write), except the id is marked as a spill so the
    /// worker completes it instead of the operator.
    pub(crate) fn write_spill(
        &mut self,
        open_file: OpenFile,
        offset: u64,
        data: Arc<FileBytes>,
    ) -> Result<WriteRequestId, crate::io::IORequesterError> {
        let id = WriteRequestId::next(self.next_request_id, true);
        let request = PendingWriteRequest {
            id,
            open_file,
            offset,
            data,
        };
        self.requester.borrow_mut().request_write(
            self.data_flow_id,
            self.operator_idx,
            request,
            self.stats,
        )?;
        Ok(id)
    }

    pub(crate) fn data_flow_id(&self) -> crate::Identifier {
        self.data_flow_id
    }

    pub(crate) fn operator_idx(&self) -> crate::Identifier {
        self.operator_idx
    }

    /// Point this handle at another operator of the same dataflow. A traversal
    /// builds one handle and retargets it per node, which keeps the per-node
    /// cost of an IO-free pass at a single store.
    #[inline]
    pub(crate) fn set_operator_idx(&mut self, operator_idx: crate::Identifier) {
        self.operator_idx = operator_idx;
    }
}

/// Owns everything an [`OperatorIO`] borrows, so a standalone operator test
/// can hand its operator a real handle without running a worker.
#[cfg(any(test, feature = "test-util"))]
pub struct TestOperatorIO {
    requester: RefCell<crate::io::IORequester>,
    stats: crate::stats::StatsCollector,
    next_request_id: usize,
    data_flow_id: crate::Identifier,
    operator_idx: crate::Identifier,
}

#[cfg(any(test, feature = "test-util"))]
impl Default for TestOperatorIO {
    fn default() -> Self {
        Self::for_operator(0, 0)
    }
}

#[cfg(any(test, feature = "test-util"))]
impl TestOperatorIO {
    /// An IO owner whose requests belong to operator `operator_idx` of
    /// dataflow `data_flow_id`, so that tests which touch per-operator state,
    /// such as spill files, stay apart.
    pub fn for_operator(data_flow_id: crate::Identifier, operator_idx: crate::Identifier) -> Self {
        Self {
            requester: RefCell::new(crate::io::IORequester::default()),
            stats: crate::stats::StatsCollector::disabled(),
            next_request_id: 0,
            data_flow_id,
            operator_idx,
        }
    }

    pub fn io(&mut self) -> OperatorIO<'_> {
        OperatorIO::new(
            &self.requester,
            self.data_flow_id,
            self.operator_idx,
            &mut self.stats,
            &mut self.next_request_id,
        )
    }

    /// A spiller over this IO that writes under `spill_directory`. Its writes
    /// are recorded with the memory context, as the worker's are.
    pub fn spiller<'a>(
        &'a mut self,
        spill_directory: &'a crate::memory::SpillDirectory,
    ) -> crate::memory::Spiller<'a> {
        crate::memory::Spiller::new(self.io(), spill_directory)
    }

    /// Wait for every spill write in flight to land, counting each off its
    /// buffer as the worker does.
    pub fn finish_spill_writes(&mut self) {
        let mut requester = self.requester.borrow_mut();
        while crate::memory::memory_ctx().has_spill_writes_in_flight() {
            requester.wait().unwrap();
            requester.collect_completions().unwrap();
            crate::memory::memory_ctx().complete_landed_spill_writes(&mut requester);
        }
    }

    /// Wait for IO to complete and hand every finished logical read to
    /// `deliver`, as the worker does for its operators.
    pub fn deliver_completed_reads(&mut self, mut deliver: impl FnMut(ReadResponse)) {
        let mut requester = self.requester.borrow_mut();
        requester.wait().unwrap();
        for completion in requester.completions().unwrap() {
            assert!(completion.is_ok(), "a test's IO does not fail");
        }
        for ready in requester.take_ready_reads() {
            deliver(ready.response);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spill_ids_are_told_apart_from_plain_ones() {
        let mut sequence = 7;

        let plain = WriteRequestId::next(&mut sequence, false);
        let spill = WriteRequestId::next(&mut sequence, true);

        assert!(!plain.is_spill());
        assert!(spill.is_spill());
        assert_eq!(plain, WriteRequestId(7));
        assert_ne!(spill, WriteRequestId(8));
        assert_eq!(sequence, 9);
    }
}
