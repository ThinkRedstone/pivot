//! What a compressed-cache lookup hands back: a [`CacheLookup`] per fragment
//! of the requested range, and for a fragment not yet resident the
//! [`MissingExtent`] whose fill the IO layer drives.
//!
//! ```text
//!   requested range ──▶ [ hit ][ missing, owned ][ hit ][ missing, following ]
//!                         │        │                        │
//!                         │        └ this lookup reads it   └ another lookup's
//!                         └ bytes already there              fill is in flight;
//!                                                             wait for it
//! ```

use crate::io::OpenFile;
use crate::memory::cache::compressed::extent::{BLOCK_SIZE, Extent};
use crate::memory::cache::id::CacheId;
use crate::memory::context::memory_ctx;
use crate::memory::read_buffer::ReadBuffer;
use bytes::Bytes;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// One fragment of a [`CompressedCache::get`](super::CompressedCache::get)
/// result: looked-up bytes (a zero-copy view into the slot that holds them)
/// plus the fragment's missing part, if any. A fragment is resolved whole, so
/// it is either fully resident (a hit) or has one [`MissingExtent`] to resolve.
/// `data` only becomes valid once the extent has committed.
pub struct CacheLookup {
    data: Bytes,
    missing: Option<MissingExtent>,
}

impl CacheLookup {
    pub(crate) fn hit(data: Bytes) -> Self {
        Self {
            data,
            missing: None,
        }
    }

    pub(crate) fn missing(data: Bytes, missing: MissingExtent) -> Self {
        Self {
            data,
            missing: Some(missing),
        }
    }

    /// The fragment's not-yet-resident part. Its fill owner must read and
    /// commit it; a follower waits for that owner. `None` on a hit.
    pub fn missing_extent(&self) -> Option<&MissingExtent> {
        self.missing.as_ref()
    }

    /// Take the looked-up bytes - valid to read once every missing part has
    /// been filled. A zero-copy view into the cache slot, kept alive by the
    /// returned [`Bytes`].
    pub fn into_data(self) -> Bytes {
        self.data
    }
}

/// An extent still missing from the cache: its file→slot mapping, whether
/// this lookup owns the fill, and a pin keeping its slot alive. The owning
/// read targets [`dest`](Self::dest) directly, with no intermediate buffer,
/// and once it lands [`commit`](Self::commit) marks the whole extent valid. A
/// follower retains the same mapping and waits for either those valid bits
/// or the shared failure flag.
///
/// The pin is shared with the lookup's `data`, so the slot can't be evicted
/// while a read into it is outstanding, even if the owning query is cancelled
/// first.
#[derive(Clone)]
pub struct MissingExtent {
    /// The compressed cache the extent belongs to.
    cache: CacheId,
    /// The extent to read: which file blocks map to which slot blocks.
    extent: Extent,
    /// The fill this lookup is responsible for, when it inserted the extent.
    /// `None` for a follower: an invalid extent found in the table is already
    /// being filled elsewhere.
    ownership: Option<Arc<FillOwnership>>,
    /// Keeps the destination slot pinned for the read's whole lifetime.
    pin: Arc<ReadBuffer>,
}

/// The fill an owning lookup answers for. Registering the extent published it
/// to every other reader of the range as "a fill is in flight", and they
/// subscribe and wait for it to commit or fail. A fill can be abandoned with
/// no code of the owner's running: the read that would have issued it unwinds
/// after a panic on the way to the ring, or is dropped from a backlog with the
/// dataflow that queued it. So the ownership itself settles the extent when
/// dropped: unless a commit or an explicit removal settled it first, it
/// publishes failure, removes the extent so a later reader starts a fill of
/// its own, and wakes the followers.
struct FillOwnership {
    cache: CacheId,
    open_file: OpenFile,
    extent: Extent,
    settled: AtomicBool,
}

impl FillOwnership {
    /// Give the fill up: publish failure, take the extent out of the table so
    /// a later reader starts a fill of its own, and wake the followers.
    fn remove_from_cache(&self) {
        self.settled.store(true, Ordering::Release);
        memory_ctx()
            .caches()
            .compressed(self.cache)
            .remove_extent(&self.open_file, &self.extent);
        self.extent.wake_subscribers();
    }
}

impl Drop for FillOwnership {
    fn drop(&mut self) {
        if self.settled.load(Ordering::Acquire) || !crate::memory::has_memory_context() {
            return;
        }
        self.remove_from_cache();
    }
}

impl MissingExtent {
    /// A follower of a fill already in flight.
    pub(crate) fn following(cache: CacheId, extent: Extent, pin: Arc<ReadBuffer>) -> Self {
        Self {
            cache,
            extent,
            ownership: None,
            pin,
        }
    }

    /// The extent `open_file`'s lookup just inserted, owned by that lookup.
    pub(crate) fn owned(
        cache: CacheId,
        extent: Extent,
        pin: Arc<ReadBuffer>,
        open_file: &OpenFile,
    ) -> Self {
        let ownership = FillOwnership {
            cache,
            open_file: open_file.clone(),
            extent: extent.clone(),
            settled: AtomicBool::new(false),
        };
        Self {
            cache,
            extent,
            ownership: Some(Arc::new(ownership)),
            pin,
        }
    }

    #[cfg(test)]
    pub(crate) fn extent(&self) -> &Extent {
        &self.extent
    }

    /// Whether this lookup is responsible for filling the extent. A non-owner
    /// follows the request that inserted the extent instead of issuing another.
    pub(crate) fn is_fill_owner(&self) -> bool {
        self.ownership.is_some()
    }

    /// Note that the fill reached an outcome the followers can see, so
    /// dropping the ownership has nothing left to publish.
    fn settle(&self) {
        if let Some(ownership) = &self.ownership {
            ownership.settled.store(true, Ordering::Release);
        }
    }

    /// Register a worker to be woken when the owning fill finishes.
    pub(crate) fn subscribe(&self, worker_id: usize) {
        self.extent.subscribing_workers.push(worker_id);
        // If the outcome was published and its subscriber queue drained just
        // before this push, no owner remains to wake us. Notify ourselves so
        // the worker reaches its next normal completions pass without sleeping.
        if self.has_failed() || self.is_committed() {
            crate::waker::waker_set().notify_worker(worker_id);
        }
    }

    /// Whether every block covered by this view has been committed.
    pub(crate) fn is_committed(&self) -> bool {
        memory_ctx()
            .caches()
            .compressed(self.cache)
            .valid_blocks(self.extent.slot_idx as usize)
            .all_set(
                self.extent.first_slot_block as usize,
                self.extent.block_count as usize,
            )
    }

    /// Whether the worker owning the shared fill reported failure.
    pub(crate) fn has_failed(&self) -> bool {
        self.extent.failed.load(Ordering::Acquire)
    }

    /// File offset the extent's bytes are read from - always a multiple of
    /// [`BLOCK_SIZE`], the direct-I/O alignment.
    pub fn file_offset(&self) -> usize {
        self.extent.file_offset()
    }

    /// Number of bytes to read - a multiple of `BLOCK_SIZE`.
    pub fn len(&self) -> usize {
        self.extent.byte_len()
    }

    /// Whether the extent covers zero bytes. Pairs with [`len`](Self::len).
    pub fn is_empty(&self) -> bool {
        self.extent.block_count == 0
    }

    /// Split off the sub-extent covering `[rel_offset, rel_offset + len)`
    /// within this extent (both relative to its start and both
    /// `BLOCK_SIZE`-aligned). The sub-extent shares the slot pin, reads into
    /// the matching slice of the same slot, and [`commit`](Self::commit)s only
    /// its own blocks. Lets one missing extent be filled from several
    /// transports, each part writing its own slice.
    pub fn carve(&self, rel_offset: usize, len: usize) -> MissingExtent {
        debug_assert_eq!(rel_offset % BLOCK_SIZE, 0);
        debug_assert_eq!(len % BLOCK_SIZE, 0);
        debug_assert!(rel_offset + len <= self.len());
        let blocks_in = rel_offset / BLOCK_SIZE;
        MissingExtent {
            cache: self.cache,
            extent: self
                .extent
                .sub_extent(self.extent.first_file_block + blocks_in, len / BLOCK_SIZE),
            ownership: self.ownership.clone(),
            pin: self.pin.clone(),
        }
    }

    /// The destination to read the extent's [`len`](Self::len) bytes into: a
    /// 4 KB-aligned region `[dest, dest+len)` inside the pinned slot - a valid
    /// O_DIRECT target. The slot stays alive because this holds a pin, and only
    /// this extent's (currently invalid) blocks live here, so the read never
    /// races a reader of valid bytes.
    pub fn dest(&self) -> *mut u8 {
        (self.pin.ptr as usize + self.extent.first_slot_block as usize * BLOCK_SIZE) as *mut u8
    }

    /// Mark the extent's blocks valid - call once its bytes have been read into
    /// the slot.
    pub fn commit(&self) {
        self.commit_prefix(self.len());
    }

    /// Mark valid only the leading blocks the first `bytes` bytes landed in -
    /// every block that received at least one byte. For a run whose
    /// block-aligned tail overhangs the end of its file, the fully-unwritten
    /// blocks stay invalid, so a later lookup re-reads them instead of
    /// consuming whatever the slot already held.
    pub fn commit_prefix(&self, bytes: usize) {
        let blocks = bytes
            .div_ceil(BLOCK_SIZE)
            .min(self.extent.block_count as usize);
        memory_ctx()
            .caches()
            .compressed(self.cache)
            .valid_blocks(self.extent.slot_idx as usize)
            .set(self.extent.first_slot_block as usize, blocks);
        self.settle();
    }

    /// Whether a contiguous `bytes`-byte prefix reaches the extent's final
    /// 4 KB block. Because the transfer starts at the extent's beginning,
    /// reaching the final block means every extent block received at least one
    /// byte. It does not mean every byte in the extent, or every logical byte
    /// requested, was read. Since an extent contains exactly the blocks
    /// intersecting its logical range, a prefix ending at EOF before the final
    /// block means that logical range is out of bounds against the current file
    /// size (or became stale because the file shrank after the range was
    /// formed).
    pub fn prefix_reaches_last_block(&self, bytes: usize) -> bool {
        bytes.div_ceil(BLOCK_SIZE) >= self.extent.block_count as usize
    }

    /// Mark this fill failed, remove its extent, and wake its followers. A
    /// later lookup can then allocate a fresh extent and issue a new request.
    pub(crate) fn remove_from_cache(&self, open_file: &OpenFile) {
        match &self.ownership {
            Some(ownership) => {
                debug_assert_eq!(&ownership.open_file, open_file);
                ownership.remove_from_cache();
            }
            None => {
                memory_ctx()
                    .caches()
                    .compressed(self.cache)
                    .remove_extent(open_file, &self.extent);
                self.extent.wake_subscribers();
            }
        }
    }

    pub(crate) fn wake_subscribers(&self) {
        self.extent.wake_subscribers();
    }
}
