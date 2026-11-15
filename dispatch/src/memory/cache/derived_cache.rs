//! [`Cache<V>`]: entries of one derived form, packed into shared ring slots.
//!
//! ```text
//!   files: FileMap                       tenants: SlotTenants        ring
//!   file A ─┬─ 4096  → Entry{ extents:[(17,0,700k)],  meta }    slot 17: [A@4096, A@90000]
//!           └─ 90000 → Entry{ extents:[(17,700k,1.3M)], meta }
//!   file B ─── 0     → Entry{ extents:[(17,2M-97k,97k),(42,0,50k)], meta }   slot 42: [B@0]
//! ```
//!
//! An entry at rest holds no pin, only placements, so the clock can take any
//! slot. The map insert is what publishes an entry: the builder writes the
//! bytes, then `insert` takes the file's write lock and adds the entry, so the
//! lock is the happens-before between the writes and any reader; no validity
//! bitmap is needed.
//!
//! Eviction takes a whole slot and drops every entry with bytes in it. An
//! entry straddling into another slot dies whole; its bytes there become dead
//! space until that slot is itself reclaimed. Each view pins exactly the slots
//! it reads, so a view stays valid however long it is held.

use crate::env::get_env_var_with_default;
use crate::io::OpenFile;
use crate::memory::cache::cached::Cached;
use crate::memory::cache::file_map::FileMap;
use crate::memory::cache::fill::{CacheFill, CarvedSlab, SlotExtent};
use crate::memory::cache::id::CacheId;
use crate::memory::cache::ring_cache::{Reclaimed, RingCache, SourceRange};
use crate::memory::cache::slot_tenants::SlotTenants;
use crate::memory::context::memory_ctx;
use crate::memory::slab::{FrozenSlab, Slab, SlabBacking};
use crate::memory::write_buffer::ProbedSlot;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Identity of a cache entry: the file bytes it was derived from. The map key
/// is `(file, offset)`; `span` is checked on lookup so an entry derived from a
/// differently sized region at the same offset (an earlier version of the
/// file) is never served.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct CacheKey {
    pub file: OpenFile,
    pub offset: usize,
    pub span: usize,
}

impl CacheKey {
    pub fn new(file: OpenFile, offset: usize, span: usize) -> Self {
        Self { file, offset, span }
    }
}

/// One segment of a range lookup, in file order.
pub enum Segment<V> {
    /// An entry the cache holds over `[offset, offset + span)`, rebuilt and
    /// pinned for as long as it is held.
    Cached {
        offset: usize,
        span: usize,
        value: V,
    },
    /// A stretch of the range no entry covers.
    Gap { offset: usize, len: usize },
}

/// A cached value at rest: where its bytes are and what rebuilding it needs.
struct Entry<V: Cached> {
    /// The source span, from the key, for the lookup check and for walking a
    /// range in file order.
    span: usize,
    extents: Vec<SlotExtent>,
    meta: V::Meta,
}

impl<V: Cached> Entry<V> {
    /// Pin every extent's slot, touch the clock, and rebuild the value. `None`
    /// (no pins kept) if any slot is mid-reclaim.
    fn pin_and_rebuild(&self) -> Option<V> {
        let mut slabs = Vec::with_capacity(self.extents.len());
        for extent in &self.extents {
            let pin = std::sync::Arc::new(memory_ctx().ring().try_read(extent.slot as usize)?);
            memory_ctx().clock().touch(extent.slot as usize);
            let ptr = unsafe { (pin.ptr as *mut u8).add(extent.offset as usize) };
            slabs.push(Slab::new(ptr, extent.len as usize, SlabBacking::Cached(pin)).freeze());
        }
        Some(V::from_slabs(&self.meta, slabs))
    }

    fn lives_in(&self, slot: usize) -> bool {
        self.extents
            .iter()
            .any(|extent| extent.slot as usize == slot)
    }
}

/// One file's entries, by source offset.
type EntryMap<V> = BTreeMap<usize, Entry<V>>;

pub struct Cache<V: Cached> {
    id: CacheId,
    files: FileMap<EntryMap<V>>,
    tenants: SlotTenants<CacheKey>,
    /// Entries across every file; the cold fast path of `get_range`.
    entries: AtomicUsize,
    /// When off, `insert` builds into working memory and `get` always misses.
    enabled: bool,
}

impl<V: Cached> Cache<V> {
    /// `slot_count` is the ring's size. `kill_switch` names the environment
    /// variable that disables the cache (default on).
    pub fn new(id: CacheId, slot_count: usize, kill_switch: &str) -> Self {
        Self::with_enabled(id, slot_count, get_env_var_with_default(kill_switch, true))
    }

    pub fn with_enabled(id: CacheId, slot_count: usize, enabled: bool) -> Self {
        Self {
            id,
            files: FileMap::new(),
            tenants: SlotTenants::new(slot_count),
            entries: AtomicUsize::new(0),
            enabled,
        }
    }

    /// Build the entry for `key` straight into the cache and return it. The
    /// builder carves slabs from the fill and writes them; the slabs it
    /// carved, in order, are what the value is rebuilt over. A builder error
    /// caches nothing: the carved bytes stay dead until their slots recycle.
    /// A key already cached by then (a duplicate-insert race) keeps the
    /// incumbent; the freshly built value is still returned, readable.
    pub fn insert<E>(
        &self,
        key: CacheKey,
        build: impl FnOnce(&mut CacheFill) -> Result<V::Meta, E>,
    ) -> Result<V, E> {
        if !self.enabled {
            let mut fill = CacheFill::transient();
            let meta = build(&mut fill)?;
            return Ok(V::from_slabs(
                &meta,
                Self::collect_slabs(fill.into_carved_slabs()),
            ));
        }
        let cursor = memory_ctx().fill_cursor(self.id);
        let mut fill = CacheFill::packed(self.id, cursor, &self.tenants, &key);
        let meta = build(&mut fill)?;
        let carved = fill.into_carved_slabs();
        let extents = carved
            .iter()
            .map(|carved| carved.extent.expect("a packed fill places every slab"))
            .collect();
        let value = V::from_slabs(&meta, Self::collect_slabs(carved));
        self.claim(key, extents, meta);
        Ok(value)
    }

    fn collect_slabs(carved: Vec<CarvedSlab>) -> Vec<FrozenSlab> {
        carved.into_iter().map(|carved| carved.slab).collect()
    }

    /// Publish the entry under the file's write lock. A key whose offset is
    /// already present loses: the incumbent stays.
    fn claim(&self, key: CacheKey, extents: Vec<SlotExtent>, meta: V::Meta) {
        let inserted = self.files.with_table_or_create(&key.file, |table| {
            if table.contains_key(&key.offset) {
                return false;
            }
            table.insert(
                key.offset,
                Entry {
                    span: key.span,
                    extents,
                    meta,
                },
            );
            true
        });
        if inserted {
            self.entries.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The entry for `key`, pinned and rebuilt. Misses if absent, if the
    /// cached span differs from `key.span`, or if a slot is mid-reclaim.
    pub fn get(&self, key: &CacheKey) -> Option<V> {
        if !self.enabled {
            return None;
        }
        self.files
            .with_table(&key.file, |table| {
                table
                    .get(&key.offset)
                    .filter(|entry| entry.span == key.span)
                    .and_then(Entry::pin_and_rebuild)
            })
            .flatten()
    }

    /// Resolve `[offset, offset + len)` of `file` into segments in file order:
    /// each entry found (pinned) and each stretch between them. Entries are
    /// keyed by the spans the caller derived them from, so the gaps are whole
    /// units of whatever the caller caches (whole pages, for the page cache).
    pub fn get_range(&self, file: &OpenFile, offset: usize, len: usize) -> Vec<Segment<V>> {
        let mut segments = SegmentList::new(offset, len);
        if len == 0 {
            return segments.finish();
        }
        // Cold fast path: nothing is cached anywhere, so skip the locks.
        if !self.enabled || self.is_empty() {
            return segments.finish();
        }
        self.files.with_table(file, |table| {
            for (&entry_offset, entry) in table.range(offset..offset + len) {
                match entry.pin_and_rebuild() {
                    Some(value) => segments.push_cached(entry_offset, entry.span, value),
                    // Mid-reclaim: as good as absent.
                    None => segments.skip(entry_offset, entry.span),
                }
            }
        });
        segments.finish()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.load(Ordering::Relaxed) == 0
    }

    /// Drop every entry of `slot` still placed there, returning their source
    /// ranges. Stale tenants (dropped another way, lost races, failed builds,
    /// re-inserted elsewhere) are filtered by the placement check. Sound only
    /// while `slot` is held exclusively.
    fn purge_slot(&self, slot: usize) -> Vec<SourceRange> {
        let mut dropped = Vec::new();
        let mut emptied_files = Vec::new();
        for key in self.tenants.take(slot) {
            let removed = self.files.with_table_mut(&key.file, |table| {
                let still_here = table
                    .get(&key.offset)
                    .is_some_and(|entry| entry.lives_in(slot));
                if !still_here {
                    return None;
                }
                let entry = table.remove(&key.offset).expect("checked present");
                Some((entry.span, table.is_empty()))
            });
            if let Some(Some((span, emptied))) = removed {
                self.entries.fetch_sub(1, Ordering::Relaxed);
                dropped.push(SourceRange::new(key.file.clone(), key.offset, span));
                if emptied {
                    emptied_files.push(key.file);
                }
            }
        }
        for file in emptied_files {
            self.files.prune_if_empty(&file, BTreeMap::is_empty);
        }
        dropped
    }

    /// Take `slot` exclusively if it is still ours, purge it, release it.
    fn take_slot(&self, slot: usize) -> Option<Reclaimed> {
        // The probe releases the hold in place if anything below unwinds: a
        // slot the cache still references must never reach the free pool.
        let probe = ProbedSlot::new(memory_ctx().ring().try_write(slot)?);
        if memory_ctx().clock().owner(slot) != Some(self.id) {
            return None;
        }
        let dropped = self.purge_slot(slot);
        memory_ctx().clock().release(slot);
        Some(Reclaimed {
            buffer: probe.take_for_reuse(),
            dropped,
        })
    }
}

impl<V: Cached> RingCache for Cache<V> {
    fn id(&self) -> CacheId {
        self.id
    }

    fn reclaim(&self, slot: usize) -> Option<Reclaimed> {
        self.take_slot(slot)
    }

    fn reinforce_range(&self, range: &SourceRange) {
        if !self.enabled || range.len == 0 {
            return;
        }
        self.files.with_table(&range.file, |table| {
            // An entry starting before the range may still reach into it.
            if let Some((&entry_offset, entry)) = table.range(..range.offset).next_back()
                && entry_offset + entry.span > range.offset
            {
                for extent in &entry.extents {
                    memory_ctx().clock().reinforce(extent.slot as usize);
                }
            }
            for (_, entry) in table.range(range.offset..range.end()) {
                for extent in &entry.extents {
                    memory_ctx().clock().reinforce(extent.slot as usize);
                }
            }
        });
    }

    fn invalidate(&self, file: &OpenFile) {
        if let Some(table) = self.files.remove(file) {
            self.entries.fetch_sub(table.len(), Ordering::Relaxed);
        }
    }

    fn clear(&self) -> usize {
        // Release this worker's fill pin first so its own fill slot can be
        // recycled below rather than skipped as pinned.
        memory_ctx().fill_cursor(self.id).release();
        let mut dropped = 0;
        let mut slots = std::collections::HashSet::new();
        for (_, table) in self.files.drain() {
            for (_, entry) in table {
                dropped += 1;
                slots.extend(entry.extents.iter().map(|extent| extent.slot as usize));
            }
        }
        self.entries.fetch_sub(dropped, Ordering::Relaxed);
        // A slot a peer still pins (its fill cursor, a reader) stays bound and
        // recycles lazily; one re-packed by a peer since the drain keeps
        // whatever its current entries say, which the purge's placement check
        // honours.
        for slot in slots {
            if let Some(reclaimed) = self.take_slot(slot) {
                drop(reclaimed.buffer); // back to the free pool
            }
        }
        dropped
    }
}

/// Accumulates a range lookup's segments in file order, filling the stretches
/// between entries with gaps and merging adjacent gaps.
struct SegmentList<V> {
    segments: Vec<Segment<V>>,
    cursor: usize,
    end: usize,
}

impl<V> SegmentList<V> {
    fn new(offset: usize, len: usize) -> Self {
        Self {
            segments: Vec::new(),
            cursor: offset,
            end: offset + len,
        }
    }

    fn push_cached(&mut self, offset: usize, span: usize, value: V) {
        self.push_gap_up_to(offset);
        self.segments.push(Segment::Cached {
            offset,
            span,
            value,
        });
        self.cursor = offset + span;
    }

    /// An entry that could not be used: its span is a gap.
    fn skip(&mut self, offset: usize, span: usize) {
        self.push_gap_up_to(offset + span);
    }

    fn push_gap_up_to(&mut self, offset: usize) {
        let len = offset.saturating_sub(self.cursor);
        if len == 0 {
            return;
        }
        match self.segments.last_mut() {
            Some(Segment::Gap { len: previous, .. }) => *previous += len,
            _ => self.segments.push(Segment::Gap {
                offset: self.cursor,
                len,
            }),
        }
        self.cursor = offset;
    }

    fn finish(mut self) -> Vec<Segment<V>> {
        self.push_gap_up_to(self.end);
        self.segments
    }
}
