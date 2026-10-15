//! Whole-object reads through the io_uring ring.
//!
//! [`load_objects`] fetches every byte of each given object over the worker
//! pool, so the read goes through the same compressed cache and disk cache as
//! a table's column chunks: an immutable metadata file (a table log, a manifest)
//! read this way is served from cache the next time round.

use crate::file_injector::FileInjectorFactory;
use crate::{DataFile, FileRef, ObjectPath};
use dispatch::io::{FileRange, OperatorIO, ReadRequestId, ReadResponse};
use dispatch::memory::memory_ctx;
use dispatch::{
    DataFlowDispatcher, OperatorSpec, RootUnaryOperatorFactory, Sender, Unary, UnaryFactory,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

/// One object read in full: its store identity and every byte of it.
pub struct LoadedObject {
    pub file: FileRef,
    pub bytes: Vec<u8>,
}

/// Whole-object reads one worker keeps in flight before admitting the next
/// object, so per-object latency overlaps.
const MAX_OBJECTS_IN_FLIGHT: usize = 32;

/// Read every object in `files` in full, in parallel over the pool, and
/// collect them on the coordinator: one object per file, in the order of
/// `files`, whose paths must be distinct. Drives the dataflow, so it must run
/// on the **coordinator** (a
/// thread outside the dispatch workers, such as the one planning a query):
/// it blocks until every worker has finished its share, so a worker calling
/// it (from an operator, or a `run_on_worker` closure) would wait on itself.
pub fn load_objects(
    dispatcher: &DataFlowDispatcher,
    files: &[DataFile],
) -> Result<Vec<LoadedObject>, dispatch::DataFlowError> {
    if files.is_empty() {
        return Ok(Vec::new());
    }
    let injector = FileInjectorFactory::new(files);
    let workers = dispatcher.worker_count();
    let siblings = Arc::new(AtomicUsize::new(workers));
    let factories: Vec<_> = (0..workers)
        .map(|_| {
            RootUnaryOperatorFactory::new(ObjectFetcherFactory, injector.clone(), siblings.clone())
        })
        .collect();
    let loaded: Vec<LoadedObject> = OperatorSpec::new(dispatcher.clone(), factories).collect()?;
    // Workers finish in any order: hand the objects back in the order of `files`.
    let mut objects: HashMap<ObjectPath, LoadedObject> = loaded
        .into_iter()
        .map(|object| (object.file.path.clone(), object))
        .collect();
    Ok(files
        .iter()
        .map(|file| {
            objects
                .remove(&file.file.path)
                .expect("every distinct file is read once")
        })
        .collect())
}

/// Builds each worker's [`ObjectFetcher`].
struct ObjectFetcherFactory;

impl UnaryFactory<DataFile, LoadedObject> for ObjectFetcherFactory {
    type Unary = ObjectFetcher;

    fn build_unary(self) -> Self::Unary {
        ObjectFetcher {
            in_flight: HashMap::new(),
        }
    }
}

/// Fetches each object it is handed as one logical read of the whole object,
/// and emits the bytes once the ring has them.
struct ObjectFetcher {
    in_flight: HashMap<ReadRequestId, FileRef>,
}

impl Unary<DataFile, LoadedObject> for ObjectFetcher {
    fn consume(
        &mut self,
        file: DataFile,
        sender: &mut dyn Sender<LoadedObject>,
        io: &mut OperatorIO,
    ) -> dispatch::UnaryResult<()> {
        let DataFile { file, source } = file;
        let size = file.size as usize;
        if size == 0 {
            // Nothing to read, and the ring is never asked for an empty range.
            sender.send(LoadedObject {
                file,
                bytes: Vec::new(),
            })?;
            return Ok(());
        }
        let open_file = source
            .open_read(file.size)
            .map_err(|error| dispatch::UnaryError::Operator(Box::new(error)))?;
        // Register the freshly opened descriptor, so a reused fd cannot serve
        // stale extents of whatever it named before.
        memory_ctx()
            .compressed_cache()
            .open_entry(open_file.clone());
        // One range covering the object: the ring tiles it into cache slots
        // itself, as it does a column chunk of any size.
        let id = io.read(open_file, [FileRange::new(0, size)])?;
        self.in_flight.insert(id, file);
        Ok(())
    }

    fn ready_for_more_work(&mut self) -> bool {
        self.in_flight.len() < MAX_OBJECTS_IN_FLIGHT
    }

    fn process_read_response(
        &mut self,
        sender: &mut dyn Sender<LoadedObject>,
        _io: &mut OperatorIO,
        response: ReadResponse,
    ) -> dispatch::UnaryResult<()> {
        let file = self
            .in_flight
            .remove(&response.id())
            .expect("response for an unknown object read");
        sender.send(LoadedObject {
            file,
            bytes: response.into_bytes(),
        })?;
        Ok(())
    }

    fn finish(&mut self, _sender: &mut dyn Sender<LoadedObject>) -> dispatch::UnaryResult<bool> {
        Ok(self.in_flight.is_empty())
    }
}
