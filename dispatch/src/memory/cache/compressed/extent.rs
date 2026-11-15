//! An [`Extent`]: a run of a file's 4 KB blocks and where it sits in a ring
//! slot, and the [`ExtentTable`] one file's extents are kept in.
//!
//! The same extent appears on two offset 4 KB-block rulers, file blocks and
//! slot blocks:
//!
//! ```text
//!   file block:   … 48   49   50   51 …    first_file_block = 48, block_count = 4
//!                    │    │    │    │
//!   slot block:     130  131  132  133 …   first_slot_block = 130   (inside slot_idx)
//! ```
//!
//! So the extent's Nth block is file block `first_file_block + N` and slot
//! block `first_slot_block + N`. An extent never spans more than one slot, and
//! a file's extents never overlap: allocation only ever covers blocks no live
//! extent already claims.

use crate::memory::read_buffer::ReadBuffer;
use crate::memory::ring::BUFFER_SIZE;
use bytes::Bytes;
use crossbeam_deque::{Injector, Steal};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

/// Block granularity for validity tracking and disk reads (the direct-I/O
/// alignment). A read never pulls less than this, and every cached byte range
/// is rounded out to whole blocks.
pub const BLOCK_SIZE: usize = 4096;
/// Number of 4 KB blocks per 2 MB slot (512).
pub const BLOCKS_PER_SLOT: usize = BUFFER_SIZE / BLOCK_SIZE;

/// One cached extent: a contiguous range of file blocks mapped to a contiguous
/// range of blocks within one ring slot.
#[derive(Clone)]
pub(crate) struct Extent {
    /// The extent's first file block. Also its key in the file's table.
    pub first_file_block: usize,
    /// The ring slot holding the extent's bytes.
    pub slot_idx: u32,
    /// The extent's first block within that slot (0..512).
    pub first_slot_block: u16,
    /// Length of the extent in 4 KB blocks - the same count on both rulers.
    pub block_count: u16,
    /// Set when the worker filling this extent fails. Followers retain this
    /// shared flag after the failed extent is removed from the table.
    pub failed: Arc<AtomicBool>,
    /// Workers waiting for this extent's fill to finish. The owner drains the
    /// queue after either committing bytes or reporting failure.
    pub subscribing_workers: Arc<Injector<usize>>,
}

impl Extent {
    /// A fresh extent nobody has filled yet.
    pub fn new(
        first_file_block: usize,
        slot_idx: usize,
        first_slot_block: usize,
        block_count: usize,
    ) -> Self {
        Self {
            first_file_block,
            slot_idx: slot_idx as u32,
            first_slot_block: first_slot_block as u16,
            block_count: block_count as u16,
            failed: Arc::new(AtomicBool::new(false)),
            subscribing_workers: Arc::new(Injector::new()),
        }
    }

    /// The extent's last file block (inclusive).
    pub fn last_file_block(&self) -> usize {
        self.first_file_block + self.block_count as usize - 1
    }

    pub fn covers(&self, file_block: usize) -> bool {
        self.first_file_block <= file_block && file_block <= self.last_file_block()
    }

    /// The slot block holding file block `file_block`: the extent's slot-block
    /// start plus how far `file_block` sits into the extent. Meaningful only
    /// for a `file_block` the extent covers.
    pub fn slot_block_of(&self, file_block: usize) -> usize {
        self.first_slot_block as usize + (file_block - self.first_file_block)
    }

    /// The file byte offset the extent starts at.
    pub fn file_offset(&self) -> usize {
        self.first_file_block * BLOCK_SIZE
    }

    /// The extent's length in bytes.
    pub fn byte_len(&self) -> usize {
        self.block_count as usize * BLOCK_SIZE
    }

    /// The sub-extent covering `[file_block, file_block + block_count)`, which
    /// this extent must cover, sharing the fill's failure flag and subscribers.
    pub fn sub_extent(&self, file_block: usize, block_count: usize) -> Extent {
        Extent {
            first_file_block: file_block,
            slot_idx: self.slot_idx,
            first_slot_block: self.slot_block_of(file_block) as u16,
            block_count: block_count as u16,
            failed: self.failed.clone(),
            subscribing_workers: self.subscribing_workers.clone(),
        }
    }

    /// Whether `other` names the same placement: same slot, and the same slot
    /// block for its first file block.
    pub fn same_placement_as(&self, other: &Extent) -> bool {
        self.slot_idx == other.slot_idx
            && self.covers(other.first_file_block)
            && self.slot_block_of(other.first_file_block) == other.first_slot_block as usize
    }

    /// A zero-copy view of the extent's bytes that fall within the requested
    /// byte range `[start_offset, end_offset)`, read straight from its slot.
    /// `pin` must hold that slot and keeps it alive for as long as the
    /// returned [`Bytes`] lives.
    pub fn clipped_bytes(
        &self,
        pin: &Arc<ReadBuffer>,
        start_offset: usize,
        end_offset: usize,
    ) -> Bytes {
        let extent_start = self.file_offset();
        let clip_start = extent_start.max(start_offset);
        let clip_end = (extent_start + self.byte_len()).min(end_offset);
        let slot_start = self.first_slot_block as usize * BLOCK_SIZE + (clip_start - extent_start);
        Bytes::from_owner(SlotPin(pin.clone()))
            .slice(slot_start..slot_start + (clip_end - clip_start))
    }

    /// Wake every worker that subscribed to this extent's outcome.
    pub fn wake_subscribers(&self) {
        loop {
            match self.subscribing_workers.steal() {
                Steal::Success(worker_id) => crate::waker::waker_set().notify_worker(worker_id),
                Steal::Retry => continue,
                Steal::Empty => return,
            }
        }
    }
}

/// Owns a slot pin and exposes its 2 MB contents, so a [`Bytes`] can borrow a
/// slice of the slot zero-copy while keeping the slot pinned.
struct SlotPin(Arc<ReadBuffer>);

impl AsRef<[u8]> for SlotPin {
    fn as_ref(&self) -> &[u8] {
        self.0.as_slice()
    }
}

/// One file's extents, by first file block, never overlapping.
#[derive(Default)]
pub(crate) struct ExtentTable(BTreeMap<usize, Extent>);

impl ExtentTable {
    /// The extent covering `file_block`, if any: the greatest-keyed extent at
    /// or before it that reaches it. Extents never overlap, so at most one
    /// qualifies.
    pub fn find_covering(&self, file_block: usize) -> Option<&Extent> {
        let (_, extent) = self.0.range(..=file_block).next_back()?;
        extent.covers(file_block).then_some(extent)
    }

    /// How many blocks from `file_block` are free of any extent, up to `want`:
    /// the available prefix of the gap. Zero if `file_block` is covered.
    pub fn available_blocks_from(&self, file_block: usize, want: usize) -> usize {
        if self.find_covering(file_block).is_some() {
            return 0;
        }
        let next_extent = self
            .0
            .range(file_block + 1..)
            .next()
            .map(|(&first, _)| first)
            .unwrap_or(usize::MAX);
        (next_extent - file_block).min(want)
    }

    pub fn insert(&mut self, extent: Extent) {
        self.0.insert(extent.first_file_block, extent);
    }

    pub fn get(&self, first_file_block: usize) -> Option<&Extent> {
        self.0.get(&first_file_block)
    }

    pub fn remove(&mut self, first_file_block: usize) -> Option<Extent> {
        self.0.remove(&first_file_block)
    }

    /// Remove the extent `placement` is a part of, if it is still there.
    pub fn remove_placement(&mut self, placement: &Extent) {
        let key = self
            .find_covering(placement.first_file_block)
            .filter(|extent| extent.same_placement_as(placement))
            .map(|extent| extent.first_file_block);
        if let Some(key) = key {
            self.0.remove(&key);
        }
    }

    /// Every extent overlapping file blocks `[first, last]`.
    pub fn iter_overlapping(&self, first: usize, last: usize) -> impl Iterator<Item = &Extent> {
        // The extent covering `first` may start before it, so begin the walk at
        // the greatest key at or before it and skip a leading extent that does
        // not actually reach the range.
        let walk_start = self
            .0
            .range(..=first)
            .next_back()
            .map(|(&key, _)| key)
            .unwrap_or(first);
        self.0
            .range(walk_start..=last)
            .map(|(_, extent)| extent)
            .filter(move |extent| extent.last_file_block() >= first)
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn drain(&mut self) -> impl Iterator<Item = Extent> {
        std::mem::take(&mut self.0).into_values()
    }
}
