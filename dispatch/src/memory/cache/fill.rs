//! [`CacheFill`]: the region a cache entry is built into.
//!
//! An entry's builder gets one of these and carves slabs from it; the fill
//! records where every slab landed:
//!
//! ```text
//!   builder                      CacheFill                         ring
//!   ───────                      ─────────                         ────
//!   get_bytes(..)        ──────▶ fills the slot's tail, rotates,   slot 17 [........###]
//!                                continues in the next one          slot 42 [##.........]
//!                                carved.push(slot, off, len) each
//!                        ◀────── Vec<Slab> (writable, pinned)
//!   returns Meta
//!                                into_carved_slabs() ─▶ the cache keeps the extents,
//!                                                       the caller gets the frozen slabs back
//! ```
//!
//! [`get_bytes`](CacheFill::get_bytes) may split across slots, so raw page
//! bytes pack tightly and a page larger than a slot still fits.
//!
//! Writing into a reader-pinned slot is sound because a carved region is not
//! yet in any map (no reader can reach it), the bump pointer keeps regions
//! disjoint, and the worker is the slot's only filler.
//!
//! With the cache disabled the fill is *transient*: slabs come from working
//! memory, nothing is recorded, and the built value is handed back without
//! being cached.

use crate::memory::cache::derived_cache::CacheKey;
use crate::memory::cache::fill_cursor::FillCursor;
use crate::memory::cache::id::CacheId;
use crate::memory::cache::slot_tenants::SlotTenants;
use crate::memory::context::memory_ctx;
use crate::memory::ring::BUFFER_SIZE;
use crate::memory::slab::{FrozenSlab, Slab, SlabAllocator, SlabBacking};

/// One contiguous run of an entry's bytes inside a ring slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotExtent {
    pub slot: u32,
    pub offset: u32,
    pub len: u32,
}

/// One slab the builder carved, with where it landed. `extent` is `None` for
/// a transient fill, whose slabs are nowhere a cache would record.
pub(crate) struct CarvedSlab {
    pub slab: FrozenSlab,
    pub extent: Option<SlotExtent>,
}

enum FillRegion<'a> {
    /// Packing into the cache's slots through this worker's cursor.
    Packed {
        cache: CacheId,
        cursor: &'a mut FillCursor,
        tenants: &'a SlotTenants<CacheKey>,
        key: &'a CacheKey,
    },
    /// Working memory; nothing is cached.
    Transient { allocator: Option<SlabAllocator> },
}

pub struct CacheFill<'a> {
    region: FillRegion<'a>,
    carved: Vec<CarvedSlab>,
}

impl<'a> CacheFill<'a> {
    pub(crate) fn packed(
        cache: CacheId,
        cursor: &'a mut FillCursor,
        tenants: &'a SlotTenants<CacheKey>,
        key: &'a CacheKey,
    ) -> Self {
        Self {
            region: FillRegion::Packed {
                cache,
                cursor,
                tenants,
                key,
            },
            carved: Vec::with_capacity(1),
        }
    }

    pub(crate) fn transient() -> Self {
        Self {
            region: FillRegion::Transient { allocator: None },
            carved: Vec::with_capacity(1),
        }
    }

    /// `len` bytes that may be split across slots, in order. Packs the current
    /// slot's tail first, so consecutive entries sit back to back.
    pub fn get_bytes(&mut self, len: usize) -> Vec<Slab> {
        let mut slabs = Vec::with_capacity(1);
        let mut remaining = len;
        while remaining > 0 {
            let slab = match &mut self.region {
                FillRegion::Packed { .. } => {
                    if self.cursor().is_exhausted() {
                        self.rotate();
                    }
                    let take = remaining.min(self.cursor().remaining());
                    self.carve(take)
                }
                FillRegion::Transient { .. } => {
                    self.carve_transient_slab(remaining.min(BUFFER_SIZE))
                }
            };
            remaining -= slab.size();
            slabs.push(slab);
        }
        slabs
    }

    /// The slabs carved so far, in order, with their extents. Consumes the
    /// fill: the builder is done writing.
    pub(crate) fn into_carved_slabs(self) -> Vec<CarvedSlab> {
        self.carved
    }

    fn cursor(&mut self) -> &mut FillCursor {
        match &mut self.region {
            FillRegion::Packed { cursor, .. } => cursor,
            FillRegion::Transient { .. } => unreachable!("a transient fill has no cursor"),
        }
    }

    /// Point the cursor at a fresh slot: reset the slot's tenant list while it
    /// is held exclusively, bind it to the cache, then publish it as readable.
    /// Taking the slot may evict, so the caller must hold no map lock, which
    /// `Cache::insert` guarantees by running the builder before claiming.
    fn rotate(&mut self) {
        let FillRegion::Packed {
            cache,
            cursor,
            tenants,
            ..
        } = &mut self.region
        else {
            unreachable!("only a packed fill rotates")
        };
        let write_buffer = memory_ctx().get_write_buffer(false);
        tenants.clear(write_buffer.slot_idx);
        memory_ctx().clock().bind(write_buffer.slot_idx, *cache);
        cursor.rotate_to(write_buffer);
    }

    /// Carve `len` bytes at the cursor's current position, record the key as a
    /// tenant of the slot and remember the slab and its extent.
    fn carve(&mut self, len: usize) -> Slab {
        let FillRegion::Packed {
            cursor,
            tenants,
            key,
            ..
        } = &mut self.region
        else {
            unreachable!("only a packed fill carves from a cursor")
        };
        let slot = cursor.slot();
        let offset = cursor.take(len);
        let pin = cursor.pin();
        // The tenant is recorded before anything is published, so even if the
        // build fails or loses a duplicate race the slot's list still names
        // every entry that could reference it.
        tenants.push(slot, (*key).clone());
        let ptr = unsafe { (pin.ptr as *mut u8).add(offset) };
        let slab = Slab::new(ptr, len, SlabBacking::Cached(pin));
        self.carved.push(CarvedSlab {
            slab: slab.shared_view(),
            extent: Some(SlotExtent {
                slot: slot as u32,
                offset: offset as u32,
                len: len as u32,
            }),
        });
        slab
    }

    /// Carve `size` bytes of working memory and remember the slab.
    fn carve_transient_slab(&mut self, size: usize) -> Slab {
        let FillRegion::Transient { allocator } = &mut self.region else {
            unreachable!("only a transient fill allocates working memory")
        };
        let allocator = allocator.get_or_insert_with(|| SlabAllocator::new(false));
        let slab = allocator.get_slab_of_size(size, false);
        self.carved.push(CarvedSlab {
            slab: slab.shared_view(),
            extent: None,
        });
        slab
    }
}
