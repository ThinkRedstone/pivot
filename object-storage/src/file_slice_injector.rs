//! Source stage for slice reads: a work-stealing queue that hands out
//! every requested slice exactly once, to whatever worker steals it next. The
//! twin of [`crate::file_injector`], whose items are whole files.

use crate::FileSlice;
use crossbeam_deque::{Injector, Steal};
use dispatch::{Receiver, RootChannelFactory};
use std::sync::Arc;

/// Builds a [`FileSliceInjector`] per worker, all sharing one queue of the slices.
#[derive(Clone)]
pub struct FileSliceInjectorFactory {
    slices: Arc<Injector<FileSlice>>,
}

impl FileSliceInjectorFactory {
    pub fn new(slices: &[FileSlice]) -> Self {
        let injector = Injector::new();
        for slice in slices {
            injector.push(slice.clone());
        }
        Self {
            slices: Arc::new(injector),
        }
    }
}

impl RootChannelFactory<FileSlice> for FileSliceInjectorFactory {
    type Receiver = FileSliceInjector;

    fn build(self) -> FileSliceInjector {
        FileSliceInjector {
            slices: self.slices,
        }
    }
}

pub struct FileSliceInjector {
    slices: Arc<Injector<FileSlice>>,
}

impl FileSliceInjector {
    /// Wake the pool if this claim was the one that emptied the queue, for the
    /// reason [`crate::file_injector::FileInjector`] does: a worker parked
    /// between seeing the queue non-empty and re-checking it owes its finish,
    /// and nothing else wakes it.
    fn wake_if_drained(&self) {
        if self.slices.is_empty() {
            dispatch::waker::waker_set().notify_all();
        }
    }
}

impl Receiver<FileSlice> for FileSliceInjector {
    fn is_empty(&self) -> bool {
        self.slices.is_empty()
    }

    fn try_recv(&self) -> Option<FileSlice> {
        // A single non-spinning attempt, as the file injector's: the hot loop
        // must not spin on `Steal::Retry`, and the next iteration retries.
        match self.slices.steal() {
            Steal::Success(slice) => {
                self.wake_if_drained();
                Some(slice)
            }
            Steal::Empty | Steal::Retry => None,
        }
    }

    fn steal(&self) -> Option<FileSlice> {
        loop {
            match self.slices.steal() {
                Steal::Empty => return None,
                Steal::Retry => continue,
                Steal::Success(slice) => {
                    self.wake_if_drained();
                    return Some(slice);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DataFile, DataFileLocation, FileRef, ObjectPath};
    use dispatch::io::FileRange;
    use dispatch::waker::{WakerSet, WorkerWaker, init_waker_set, init_worker_waker};
    use std::path::PathBuf;

    /// The injector only ever hands these out, so they need no file behind them.
    fn file_slice(name: &str) -> FileSlice {
        FileSlice {
            file: DataFile {
                file: FileRef {
                    path: ObjectPath::new(name),
                    size: 0,
                },
                source: DataFileLocation::Local(PathBuf::from(name)),
            },
            range: FileRange::new(0, 0),
        }
    }

    /// Claiming the last slice has to wake the pool, on both claim paths, or a
    /// worker parked owing its finish never runs it and the read hangs.
    #[test]
    fn claiming_the_last_range_wakes_the_pool() {
        let waker = Arc::new(WorkerWaker::new(1));
        init_worker_waker(&waker);
        init_waker_set(WakerSet::new(vec![waker.clone()], 1));
        let injector = FileSliceInjectorFactory::new(&[file_slice("a"), file_slice("b")]).build();

        let before = waker.wake_count();
        injector.steal().expect("first slice");
        let after_first = waker.wake_count();
        // `try_recv` is deliberately single-shot, so retry past a lost steal race.
        (0..1_000)
            .find_map(|_| injector.try_recv())
            .expect("last slice");
        let after_last = waker.wake_count();

        assert_eq!(after_first, before);
        assert!(after_last > after_first);
    }
}
