//! [`CompressedCache`]: file bytes as they were read, packed into shared ring
//! slots at 4 KB block granularity.
//!
//! ```text
//!   files: FileMap<ExtentTable>                       per slot
//!   foo.parquet ─┬─ block 0  → Extent{ slot 7, slot block 130, 3 blocks }    valid[7]:   bitmap of landed blocks
//!                └─ block 40 → Extent{ slot 3, slot block 88,  2 blocks }    tenants[7]: (foo,0) (bar,12) …
//!   bar.parquet ─── block 12 → Extent{ slot 7, slot block 200, 1 block  }
//! ```
//!
//! Reads are not positional: a missed read is bump-allocated as a contiguous
//! extent of blocks at this worker's cursor inside whatever slot it is
//! currently filling, so hundreds of unrelated reads from many files share one
//! slot and the working set is measured in bytes, not slots.
//!
//! ## In-flight fills
//!
//! Unlike a derived cache, an extent is published *before* its bytes arrive,
//! so a second reader of the same range follows the first's fill instead of
//! issuing another. The per-slot [`ValidBitmap`] records which blocks have
//! actually landed; an extent can be in the table yet still filling. See
//! [`lookup`] for the owner/follower protocol.
//!
//! ## Eviction
//!
//! The clock reclaims a whole slot, so every extent living in it goes; the
//! per-slot tenant list says which. A reader holds the file's table read lock
//! across its `try_read` pin, and the evictor removes an extent under that
//! table's write lock before recycling its slot, so a reader never pins a slot
//! recycled out from under the extent it just read.
//!
//! ## Many instances
//!
//! Every family that keeps raw bytes has its own `CompressedCache`, so those
//! bytes are balanced against the family's derived forms by its shares. A
//! read names the compressed cache it fills by [`CacheId`].

mod extent;
mod lookup;
mod valid_bitmap;

pub use lookup::{CacheLookup, MissingExtent};

use crate::io::OpenFile;
use crate::memory::cache::compressed::extent::{BLOCK_SIZE, BLOCKS_PER_SLOT, Extent, ExtentTable};
use crate::memory::cache::compressed::valid_bitmap::ValidBitmap;
use crate::memory::cache::file_map::FileMap;
use crate::memory::cache::fill_cursor::FillCursor;
use crate::memory::cache::id::CacheId;
use crate::memory::cache::ring_cache::{Reclaimed, RingCache, SourceRange};
use crate::memory::cache::slot_tenants::SlotTenants;
use crate::memory::context::memory_ctx;
use crate::memory::write_buffer::ProbedSlot;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::Ordering;

/// One `(file, first_file_block)` packed into a slot, recorded so the evictor
/// can drop the extent when it recycles the slot.
struct Tenant {
    open_file: OpenFile,
    first_file_block: usize,
}

/// One step while resolving a lookup: the fragment found or allocated at the
/// current file block, and the next file block to visit.
struct LookupStep {
    lookup: CacheLookup,
    next_file_block: usize,
}

pub struct CompressedCache {
    id: CacheId,
    files: FileMap<ExtentTable>,
    /// Which of each slot's blocks have been read in, by ring slot.
    valid: Box<[ValidBitmap]>,
    tenants: SlotTenants<Tenant>,
}

impl CompressedCache {
    /// `slot_count` is the ring's size.
    pub fn new(id: CacheId, slot_count: usize) -> Self {
        Self {
            id,
            files: FileMap::new(),
            valid: (0..slot_count).map(|_| ValidBitmap::default()).collect(),
            tenants: SlotTenants::new(slot_count),
        }
    }

    pub(crate) fn valid_blocks(&self, slot: usize) -> &ValidBitmap {
        &self.valid[slot]
    }

    /// Look up the file byte range `[offset, offset + len)` of `open_file`.
    /// The range is resolved into contiguous fragments in file order, one
    /// [`CacheLookup`] each: read and fill every lookup's missing extent, then
    /// concatenate the fragments' bytes.
    pub fn get(&self, open_file: &OpenFile, offset: usize, len: usize) -> Vec<CacheLookup> {
        if len == 0 {
            return Vec::new();
        }
        let end_offset = offset + len;
        let last_file_block = (end_offset - 1) / BLOCK_SIZE;

        let mut lookups = Vec::new();
        let mut file_block = offset / BLOCK_SIZE;
        while file_block <= last_file_block {
            // Resolve one fragment from an existing extent, or allocate an
            // extent for an uncovered fragment. Allocation declines only when
            // another worker just covered `file_block`, so retrying resolves
            // it as a hit.
            let step = loop {
                if let Some(step) = self.resolve_existing(
                    open_file,
                    file_block,
                    last_file_block,
                    offset,
                    end_offset,
                ) {
                    break step;
                }
                if let Some(step) = self.allocate_missing(
                    open_file,
                    file_block,
                    last_file_block,
                    offset,
                    end_offset,
                ) {
                    break step;
                }
            };
            file_block = step.next_file_block;
            lookups.push(step.lookup);
        }
        lookups
    }

    /// Resolve the fragment at `file_block` from an existing extent. Pins the
    /// extent's slot and takes the maximal span of equal validity, bounded by
    /// the extent and `last_file_block`. A valid span is a hit; an invalid span
    /// is a fill in flight to follow.
    ///
    /// `None` when no extent covers `file_block` (the file's table may have
    /// been pruned) or the covering extent's slot is being reclaimed. The
    /// caller then allocates.
    fn resolve_existing(
        &self,
        open_file: &OpenFile,
        file_block: usize,
        last_file_block: usize,
        start_offset: usize,
        end_offset: usize,
    ) -> Option<LookupStep> {
        self.files
            .with_table(open_file, |table| {
                let covering_extent = table.find_covering(file_block)?;
                let pin = Arc::new(
                    memory_ctx()
                        .ring()
                        .try_read(covering_extent.slot_idx as usize)?,
                );
                memory_ctx()
                    .clock()
                    .touch(covering_extent.slot_idx as usize);
                let valid = self.valid_blocks(covering_extent.slot_idx as usize);

                // Grow the fragment over blocks of the same validity.
                let last_fragment_block = covering_extent.last_file_block().min(last_file_block);
                let is_valid = valid.is_set(covering_extent.slot_block_of(file_block));
                let mut last_matching_block = file_block;
                while last_matching_block < last_fragment_block
                    && valid.is_set(covering_extent.slot_block_of(last_matching_block + 1))
                        == is_valid
                {
                    last_matching_block += 1;
                }

                let extent =
                    covering_extent.sub_extent(file_block, last_matching_block - file_block + 1);
                let data = extent.clipped_bytes(&pin, start_offset, end_offset);
                let next_file_block = file_block + extent.block_count as usize;
                let lookup = if is_valid {
                    CacheLookup::hit(data)
                } else {
                    CacheLookup::missing(data, MissingExtent::following(self.id, extent, pin))
                };
                Some(LookupStep {
                    lookup,
                    next_file_block,
                })
            })
            .flatten()
    }

    /// Handle a miss at `file_block`: bump-allocate an extent in this worker's
    /// fill slot and register it, returning a lookup over the reserved region
    /// plus the [`MissingExtent`] whose bytes must be read in. The extent is
    /// capped by whichever runs out first, the request or the fill slot's
    /// remaining room, so a read spilling past the slot boundary continues in
    /// the next lookup. `None` if another worker covered `file_block` first.
    fn allocate_missing(
        &self,
        open_file: &OpenFile,
        file_block: usize,
        last_file_block: usize,
        start_offset: usize,
        end_offset: usize,
    ) -> Option<LookupStep> {
        let cursor = memory_ctx().fill_cursor(self.id);
        if cursor.is_exhausted() {
            // Rotation may evict, so no table lock is held across it.
            self.rotate_fill_slot(cursor);
        }
        let first_slot_block = cursor.next_byte() / BLOCK_SIZE;
        let requested_blocks =
            (last_file_block - file_block + 1).min(BLOCKS_PER_SLOT - first_slot_block);

        let extent = self.register_extent(
            open_file,
            cursor.slot(),
            first_slot_block,
            file_block,
            requested_blocks,
        )?;
        cursor.take(extent.byte_len());
        let pin = cursor.pin();
        let data = extent.clipped_bytes(&pin, start_offset, end_offset);
        let next_file_block = file_block + extent.block_count as usize;
        Some(LookupStep {
            lookup: CacheLookup::missing(
                data,
                MissingExtent::owned(self.id, extent, pin, open_file),
            ),
            next_file_block,
        })
    }

    /// Add an extent of up to `requested_blocks` available blocks for
    /// `open_file`, recreating its table first if eviction pruned it. The
    /// table's write lock serializes overlapping misses, so only the first
    /// worker to cover `file_block` succeeds; `None` for the others.
    fn register_extent(
        &self,
        open_file: &OpenFile,
        slot_idx: usize,
        first_slot_block: usize,
        file_block: usize,
        requested_blocks: usize,
    ) -> Option<Extent> {
        self.files.with_table_or_create(open_file, |table| {
            let block_count = table.available_blocks_from(file_block, requested_blocks);
            if block_count == 0 {
                return None;
            }
            let extent = Extent::new(file_block, slot_idx, first_slot_block, block_count);
            // Record the tenant before the extent: an extent with no tenant
            // would leak (eviction only drops extents listed as tenants),
            // while a tenant with no extent is harmless (eviction skips it).
            self.tenants.push(
                slot_idx,
                Tenant {
                    open_file: open_file.clone(),
                    first_file_block: file_block,
                },
            );
            table.insert(extent.clone());
            // Mark it referenced so the fresh extent survives one sweep.
            memory_ctx().clock().touch(slot_idx);
            Some(extent)
        })
    }

    /// Point this worker's fill cursor at a fresh, empty slot: reset its bitmap
    /// and tenant list while it is held exclusively, bind it, then publish it
    /// as readable.
    fn rotate_fill_slot(&self, cursor: &mut FillCursor) {
        let write_buffer = memory_ctx().get_write_buffer(false);
        let slot_idx = write_buffer.slot_idx;
        self.valid_blocks(slot_idx).clear();
        self.tenants.clear(slot_idx);
        memory_ctx().clock().bind(slot_idx, self.id);
        cursor.rotate_to(write_buffer);
    }

    /// Mark `placement` failed and take its extent out of `open_file`'s table.
    /// Runs from an unwinding owner's drop as well, so a lock another thread
    /// poisoned is used anyway: leaving the extent published would be the
    /// worse outcome.
    pub(crate) fn remove_extent(&self, open_file: &OpenFile, placement: &Extent) {
        let removed = self
            .files
            .with_table_mut_tolerating_poison(open_file, |table| {
                // Publish failure while holding the table write lock that
                // excludes a new lookup from discovering this extent.
                placement.failed.store(true, Ordering::Release);
                table.remove_placement(placement);
            });
        if removed.is_none() {
            placement.failed.store(true, Ordering::Release);
        }
    }

    /// Reset a slot's validity and release it from the clock. Sound only while
    /// the slot is held exclusively and its tenant list has been taken.
    fn recycle_slot(&self, slot_idx: usize) {
        self.valid_blocks(slot_idx).clear();
        memory_ctx().clock().release(slot_idx);
    }

    /// Take `slot_idx` exclusively if it is still ours: drop every extent packed
    /// in it from its file's table, recycle it, and return it writable with
    /// the dropped extents' source ranges.
    fn take_slot(&self, slot_idx: usize) -> Option<Reclaimed> {
        // The probe releases the hold in place if anything below unwinds (a
        // poisoned table lock): a slot the cache still references must never
        // fall into the free pool.
        let probe = ProbedSlot::new(memory_ctx().ring().try_write(slot_idx)?);
        if memory_ctx().clock().owner(slot_idx) != Some(self.id) {
            return None;
        }
        let mut dropped = Vec::new();
        let mut emptied_files = Vec::new();
        for tenant in self.tenants.take(slot_idx) {
            let removed = self.files.with_table_mut(&tenant.open_file, |table| {
                // Skip an extent re-placed into another slot after this tenant
                // was recorded.
                let still_here = table
                    .get(tenant.first_file_block)
                    .is_some_and(|extent| extent.slot_idx as usize == slot_idx);
                if !still_here {
                    return None;
                }
                let extent = table
                    .remove(tenant.first_file_block)
                    .expect("checked present");
                Some((extent, table.is_empty()))
            });
            if let Some(Some((extent, emptied))) = removed {
                dropped.push(SourceRange::new(
                    tenant.open_file.clone(),
                    extent.file_offset(),
                    extent.byte_len(),
                ));
                if emptied {
                    emptied_files.push(tenant.open_file);
                }
            }
        }
        for file in emptied_files {
            self.files.prune_if_empty(&file, ExtentTable::is_empty);
        }
        self.recycle_slot(slot_idx);
        Some(Reclaimed {
            buffer: probe.take_for_reuse(),
            dropped,
        })
    }

    #[cfg(test)]
    pub(crate) fn holds_file(&self, open_file: &OpenFile) -> bool {
        self.files.contains_file(open_file)
    }

    #[cfg(test)]
    pub(crate) fn poison_file_map(&self) {
        self.files.poison();
    }
}

impl RingCache for CompressedCache {
    fn id(&self) -> CacheId {
        self.id
    }

    fn reclaim(&self, slot: usize) -> Option<Reclaimed> {
        self.take_slot(slot)
    }

    /// Slot granularity, so a multi-tenant slot's other extents shelter under
    /// the same reinforcement.
    fn reinforce_range(&self, range: &SourceRange) {
        if range.len == 0 {
            return;
        }
        let first_file_block = range.offset / BLOCK_SIZE;
        let last_file_block = (range.end() - 1) / BLOCK_SIZE;
        self.files.with_table(&range.file, |table| {
            for extent in table.iter_overlapping(first_file_block, last_file_block) {
                memory_ctx().clock().reinforce(extent.slot_idx as usize);
            }
        });
    }

    /// Drop `file`'s table, so the next lookup misses and the cache keeps no
    /// handle to a file it holds nothing of. The orphaned tenants in their
    /// slots self-clean on eviction.
    fn invalidate(&self, file: &OpenFile) {
        self.files.remove(file);
    }

    /// Sound only while no query is in flight: it recycles bound slots, which
    /// must not race a reader holding a pin (a slot another worker is still
    /// filling is left intact; its extents are already dropped, so it serves
    /// no stale hits).
    fn clear(&self) -> usize {
        // Release this worker's fill pin first, so its own fill slot is
        // recyclable below rather than skipped as pinned.
        memory_ctx().fill_cursor(self.id).release();

        let mut slots = HashSet::new();
        let mut dropped = 0;
        self.files.for_each_table_mut(|_, table| {
            for extent in table.drain() {
                slots.insert(extent.slot_idx as usize);
                dropped += 1;
            }
        });
        for slot_idx in slots {
            if let Some(reclaimed) = self.take_slot(slot_idx) {
                drop(reclaimed.buffer); // back to the free pool
            }
        }
        dropped
    }
}

#[cfg(test)]
mod tests;
