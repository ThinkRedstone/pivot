//! Worker-local registration and tracking of logical operator reads.
//!
//! Operators enqueue file ranges in [`OperatorIO`](crate::io::OperatorIO). This
//! tracker is the boundary where those logical ranges become physical cache
//! extents and transport operations. It is owned privately by the worker's
//! [`IORequester`](crate::io::IORequester). Registration returns each physical
//! operation immediately to that requester; the tracker has no submission queue.

use crate::Identifier;
use crate::io::{
    DataFlowRequest, FsReadRequest, FsRequest, HttpGetRequest, HttpRequest, OpenFile,
    PendingReadRequest, ReadData, ReadResponse,
};
use crate::memory::compressed_cache::MissingExtent;
use crate::memory::{CacheLookup, memory_ctx};
use std::collections::HashMap;

/// Where a logical operator request must be delivered when it becomes ready.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct RequestRoute {
    pub data_flow_id: Identifier,
    pub operator_idx: Identifier,
}

/// A ready logical response paired with its destination operator.
pub(crate) struct RoutedReadResponse {
    pub route: RequestRoute,
    pub response: ReadResponse,
}

/// One requested range as the compressed cache resolved it: its hits, plus
/// the extents allocated for its misses, whose fills the request waits on.
struct PendingReadData {
    offset: usize,
    lookups: Vec<CacheLookup>,
}

impl PendingReadData {
    fn into_ready(self) -> ReadData {
        ReadData {
            offset: self.offset,
            bytes: self
                .lookups
                .into_iter()
                .map(CacheLookup::into_data)
                .collect(),
        }
    }
}

struct TrackedRead {
    route: RequestRoute,
    request: PendingReadRequest,
    locations: Vec<PendingReadData>,
    /// Number of physical cache fills this logical request still awaits.
    remaining: usize,
}

impl TrackedRead {
    fn into_response(self) -> RoutedReadResponse {
        RoutedReadResponse {
            route: self.route,
            response: ReadResponse::new(
                self.request.id,
                self.request.open_file,
                self.locations
                    .into_iter()
                    .map(PendingReadData::into_ready)
                    .collect(),
            ),
        }
    }
}

/// All logical and physical I/O state owned by one dispatch worker.
#[derive(Default)]
pub(crate) struct RequestTracker {
    requests: Vec<Option<TrackedRead>>,
    free_slots: Vec<usize>,
    /// Each submitted physical read has a distinct id, even when two reads
    /// cover the same file range. Coalescing is deliberately not performed.
    routing: HashMap<Identifier, usize>,
    next_physical_read_id: Identifier,
    ready: Vec<RoutedReadResponse>,
}

impl RequestTracker {
    pub(crate) fn register_read(
        &mut self,
        route: RequestRoute,
        request: PendingReadRequest,
    ) -> Vec<RegisteredRead> {
        let mut locations = Vec::with_capacity(request.locations.len());
        let mut physical_reads = Vec::new();

        for location in &request.locations {
            let lookups = memory_ctx().compressed_cache().get(
                &request.open_file,
                location.offset,
                location.len,
            );
            for lookup in &lookups {
                if let Some(block) = lookup.missing() {
                    physical_reads.push(match &request.open_file {
                        OpenFile::Local(file) => FsOrHttpRead::Fs(FsReadRequest {
                            file: file.clone(),
                            block: block.clone(),
                        }),
                        OpenFile::Remote(remote) => FsOrHttpRead::Http(HttpGetRequest {
                            remote: remote.clone(),
                            block: block.clone(),
                        }),
                    });
                }
            }
            locations.push(PendingReadData {
                offset: location.offset,
                lookups,
            });
        }

        let remaining = physical_reads.len();
        let slot = self.insert_request(TrackedRead {
            route,
            request,
            locations,
            remaining,
        });

        let physical_reads = physical_reads
            .into_iter()
            .map(|read| self.track_physical_read(route, slot, read))
            .collect();

        if remaining == 0 {
            self.finish_request(slot);
        }

        physical_reads
    }

    fn insert_request(&mut self, request: TrackedRead) -> usize {
        if let Some(slot) = self.free_slots.pop() {
            self.requests[slot] = Some(request);
            slot
        } else {
            self.requests.push(Some(request));
            self.requests.len() - 1
        }
    }

    fn track_physical_read(
        &mut self,
        route: RequestRoute,
        slot: usize,
        read: FsOrHttpRead,
    ) -> RegisteredRead {
        let id = self.next_physical_read_id;
        self.next_physical_read_id += 1;
        self.routing.insert(id, slot);
        match read {
            FsOrHttpRead::Fs(request) => RegisteredRead::Fs(
                DataFlowRequest::new(
                    route.data_flow_id,
                    route.operator_idx,
                    FsRequest::Read(request),
                )
                .with_tracked_read_id(id),
            ),
            FsOrHttpRead::Http(request) => RegisteredRead::Http(
                DataFlowRequest::new(
                    route.data_flow_id,
                    route.operator_idx,
                    HttpRequest::Get(request),
                )
                .with_tracked_read_id(id),
            ),
        }
    }

    pub fn complete(&mut self, physical_read_id: Identifier) {
        let Some(slot) = self.routing.remove(&physical_read_id) else {
            // The logical request may have been cancelled while this physical
            // read was still in flight.
            return;
        };
        let Some(request) = self.requests[slot].as_mut() else {
            return;
        };
        request.remaining -= 1;
        if request.remaining == 0 {
            self.finish_request(slot);
        }
    }

    /// Fail the logical request owning one physical read. Its other reads may
    /// still complete and warm the cache, but no longer have a callback target.
    pub fn fail(&mut self, physical_read_id: Identifier) -> Option<RequestRoute> {
        let slot = self.routing.remove(&physical_read_id)?;
        let request = self.requests[slot].take()?;
        self.routing.retain(|_, routed_slot| *routed_slot != slot);
        self.free_slots.push(slot);
        Some(request.route)
    }

    fn finish_request(&mut self, slot: usize) {
        let request = self.requests[slot]
            .take()
            .expect("finishing an empty request slot");
        debug_assert_eq!(request.remaining, 0);
        self.free_slots.push(slot);
        self.ready.push(request.into_response());
    }

    pub fn take_ready(&mut self) -> Vec<RoutedReadResponse> {
        std::mem::take(&mut self.ready)
    }

    /// Forget logical state for a dataflow that has gone away. Physical reads
    /// already submitted are allowed to finish and warm the cache; their later
    /// completions simply find no waiter.
    pub fn cancel_dataflow(&mut self, data_flow_id: Identifier) {
        let cancelled_slots: Vec<_> = self
            .requests
            .iter_mut()
            .enumerate()
            .filter_map(|(slot, request)| {
                (request.as_ref()?.route.data_flow_id == data_flow_id).then_some(slot)
            })
            .collect();
        for slot in &cancelled_slots {
            if self.requests[*slot].take().is_some() {
                self.free_slots.push(*slot);
            }
        }
        self.routing
            .retain(|_, slot| !cancelled_slots.contains(slot));
        self.ready
            .retain(|ready| ready.route.data_flow_id != data_flow_id);
    }
}

enum FsOrHttpRead {
    Fs(FsReadRequest),
    Http(HttpGetRequest),
}

pub(crate) enum RegisteredRead {
    Fs(DataFlowRequest<FsRequest>),
    Http(DataFlowRequest<HttpRequest>),
}

impl RegisteredRead {
    /// The cache extent this tracked read is waiting for.
    pub(crate) fn missing_extent(&self) -> &MissingExtent {
        match self {
            Self::Fs(request) => match &request.request {
                FsRequest::Read(read) => &read.block,
                FsRequest::Write(_) => unreachable!("registered reads never contain writes"),
            },
            Self::Http(request) => match &request.request {
                HttpRequest::Get(read) => &read.block,
                HttpRequest::Upload(_) => {
                    unreachable!("registered reads never contain uploads")
                }
            },
        }
    }

    pub(crate) fn data_flow_id(&self) -> Identifier {
        match self {
            Self::Fs(request) => request.data_flow_id,
            Self::Http(request) => request.data_flow_id,
        }
    }

    pub(crate) fn tracked_read_id(&self) -> Identifier {
        match self {
            Self::Fs(request) => request.tracked_read_id,
            Self::Http(request) => request.tracked_read_id,
        }
        .expect("the tracker assigns every registered read an id")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::{FileRange, LocalFile, ReadRequestId};
    use crate::memory::{BlockKey, init_test_free_pool};
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use tempfile::TempDir;

    const ROUTE: RequestRoute = RequestRoute {
        data_flow_id: 7,
        operator_idx: 3,
    };

    fn registered_file() -> (OpenFile, TempDir) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("data");
        std::fs::write(&path, vec![0u8; 1 << 20]).unwrap();
        let open_file =
            OpenFile::Local(LocalFile::new(std::fs::File::open(path).unwrap()).unwrap());
        memory_ctx()
            .compressed_cache()
            .open_entry(open_file.clone());
        (open_file, dir)
    }

    fn pending_read(
        file: OpenFile,
        locations: impl IntoIterator<Item = FileRange>,
    ) -> (ReadRequestId, PendingReadRequest) {
        let id = ReadRequestId(0);
        let request = PendingReadRequest {
            id,
            open_file: file,
            locations: locations.into_iter().collect(),
        };
        (id, request)
    }

    fn register_local(
        tracker: &mut RequestTracker,
        route: RequestRoute,
        request: PendingReadRequest,
    ) -> Vec<DataFlowRequest<FsRequest>> {
        tracker
            .register_read(route, request)
            .into_iter()
            .map(|request| match request {
                RegisteredRead::Fs(request) => request,
                RegisteredRead::Http(_) => panic!("expected a filesystem read"),
            })
            .collect()
    }

    fn physical_id(request: &DataFlowRequest<FsRequest>) -> Identifier {
        request.tracked_read_id.unwrap()
    }

    fn complete_reads(
        tracker: &mut RequestTracker,
        requests: impl IntoIterator<Item = DataFlowRequest<FsRequest>>,
    ) {
        for request in requests {
            let id = physical_id(&request);
            let FsRequest::Read(read) = request.request else {
                panic!("expected a read")
            };
            read.block.commit();
            tracker.complete(id);
        }
    }

    fn warm_compressed_cache(tracker: &mut RequestTracker, file: OpenFile, location: FileRange) {
        let (_, pending) = pending_read(file, [location]);
        let physical = register_local(tracker, ROUTE, pending);

        complete_reads(tracker, physical);
        tracker.take_ready();
    }

    fn cache_decompressed(file: OpenFile, location: FileRange) {
        let key = BlockKey {
            open_file: file,
            offset: location.offset,
            len: location.len,
        };
        let cache = memory_ctx().decompressed_cache();
        let mut reservation = cache.reserve(&key, 4);
        reservation.as_mut_slices()[0].copy_from_slice(b"data");
        cache.insert(
            key,
            vec![bytes::Bytes::from_static(b"header")],
            reservation,
            Arc::new(AtomicUsize::new(0)),
        );
    }

    #[test]
    fn uncached_request_becomes_ready_after_its_physical_read() {
        init_test_free_pool(4);
        let (file, _dir) = registered_file();
        let (id, pending) = pending_read(file, [FileRange::new(17, 100)]);
        let mut tracker = RequestTracker::default();

        let physical = register_local(&mut tracker, ROUTE, pending);
        let physical_count = physical.len();
        complete_reads(&mut tracker, physical);
        let ready = tracker.take_ready().pop().unwrap();

        assert_eq!(physical_count, 1);
        assert_eq!(ready.route, ROUTE);
        assert_eq!(ready.response.id(), id);
        assert_eq!(ready.response.into_bytes().len(), 100);
    }

    #[test]
    fn compressed_cache_hit_needs_no_transport_io() {
        init_test_free_pool(4);
        let (file, _dir) = registered_file();
        let location = FileRange::new(0, 4096);
        let mut tracker = RequestTracker::default();
        warm_compressed_cache(&mut tracker, file.clone(), location);
        let (id, pending) = pending_read(file, [location]);

        let physical = register_local(&mut tracker, ROUTE, pending);
        let ready = tracker.take_ready().pop().unwrap();

        assert!(physical.is_empty());
        assert_eq!(ready.response.id(), id);
    }

    /// A read is answered from the file's stored bytes alone: a decompressed
    /// page cached over its range neither spares the read nor replaces its
    /// bytes (a footer probe spanning a small file's data pages gets the file).
    #[test]
    fn a_read_ignores_decompressed_pages_cached_over_its_range() {
        init_test_free_pool(4);
        let (file, _dir) = registered_file();
        cache_decompressed(file.clone(), FileRange::new(0, 4096));
        let (_, pending) = pending_read(file, [FileRange::new(0, 8192)]);
        let mut tracker = RequestTracker::default();

        let physical = register_local(&mut tracker, ROUTE, pending);
        let needed_io = !physical.is_empty();
        complete_reads(&mut tracker, physical);
        let response = tracker.take_ready().pop().unwrap().response;

        assert!(needed_io);
        assert_eq!(response.into_bytes(), vec![0u8; 8192]);
    }

    #[test]
    fn identical_requests_use_separate_physical_reads() {
        init_test_free_pool(4);
        let (file, _dir) = registered_file();
        let first_route = RequestRoute {
            data_flow_id: 1,
            operator_idx: 2,
        };
        let second_route = RequestRoute {
            data_flow_id: 3,
            operator_idx: 4,
        };
        let (_, first) = pending_read(file.clone(), [FileRange::new(0, 4096)]);
        let (_, second) = pending_read(file, [FileRange::new(0, 4096)]);
        let mut tracker = RequestTracker::default();

        let mut physical = register_local(&mut tracker, first_route, first);
        physical.extend(register_local(&mut tracker, second_route, second));
        let physical_count = physical.len();
        complete_reads(&mut tracker, physical);
        let mut routes: Vec<_> = tracker
            .take_ready()
            .into_iter()
            .map(|ready| ready.route)
            .collect();
        routes.sort_by_key(|route| route.data_flow_id);

        assert_eq!(physical_count, 2);
        assert_eq!(routes, vec![first_route, second_route]);
    }

    #[test]
    fn failed_physical_read_discards_its_logical_request() {
        init_test_free_pool(4);
        let (file, _dir) = registered_file();
        let (_, pending) =
            pending_read(file, [FileRange::new(0, 4096), FileRange::new(32768, 4096)]);
        let mut tracker = RequestTracker::default();
        let physical = register_local(&mut tracker, ROUTE, pending);
        let ids: Vec<_> = physical.iter().map(physical_id).collect();

        let failed_route = tracker.fail(ids[0]);
        tracker.complete(ids[1]);
        let ready = tracker.take_ready();

        assert_eq!(failed_route, Some(ROUTE));
        assert!(ready.is_empty());
    }

    #[test]
    fn one_logical_request_reads_multiple_locations() {
        init_test_free_pool(4);
        let (file, _dir) = registered_file();
        let (_, pending) =
            pending_read(file, [FileRange::new(0, 4096), FileRange::new(32768, 4096)]);
        let mut tracker = RequestTracker::default();

        let physical = register_local(&mut tracker, ROUTE, pending);
        let physical_count = physical.len();
        complete_reads(&mut tracker, physical);
        let locations = tracker
            .take_ready()
            .pop()
            .unwrap()
            .response
            .into_locations();

        assert_eq!(physical_count, 2);
        assert_eq!(locations.len(), 2);
    }
}
