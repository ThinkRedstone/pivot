//! Slab-based sub-allocation on top of 2MB [`WriteBuffer`]s.
//!
//! A [`Slab`] is a contiguous byte region within one ring slot, held alive by a
//! [`SlabBacking`]: working memory (an `Arc<WriteBuffer>`, returned to the free pool once
//! *every* slab carved from it has been dropped) or a cache slot (a reader pin, which keeps
//! the slot from being reclaimed while the slab lives). A single long-lived slab pins its
//! whole slot either way - avoid mixing short-lived and long-lived slabs on one buffer.
//!
//! Two typed wrappers provide indexable access over slabs:
//!
//! - [`SlabBuffer<T>`] — backed by a single slab (must fit within one 2MB buffer). Cheaper to
//!   index since there is no buffer lookup. Use this when the data is guaranteed to be < 2MB.
//!
//! - [`MultiSlabBuffer<T>`] — backed by one or more slabs, supporting allocations that span
//!   multiple `WriteBuffer`s. Indexing requires a buffer lookup via bit-shift and mask. Use this
//!   for allocations whose size may exceed 2MB (e.g., hash tables).
//!
//! Both support `Index<usize>` / `IndexMut<usize>` for typed element access.
//!
//! [`SlabAllocator`] manages the allocation by bumping through the current `WriteBuffer` and
//! requesting new ones from the free pool as needed. It provides [`SlabAllocator::create_slab_buffer`]
//! and [`SlabAllocator::create_multi_slab_buffer`] as the primary entry points.

use crate::memory::WriteBuffer;
use crate::memory::read_buffer::ReadBuffer;
use std::sync::{Arc, Weak};

mod allocator;
pub use allocator::SlabAllocator;

mod slab_buffer;
pub use slab_buffer::SlabBuffer;

mod multi_slab_buffer;
pub use multi_slab_buffer::MultiSlabBuffer;

mod slab_vec;
pub use slab_vec::SlabVec;

mod slab_top_k;
pub(crate) use slab_top_k::{HeapBuffer, MultiTopK, Ranked, SingleTopK, SlabTopK, slots_per_slab};

/// What keeps a slab's ring slot alive while the slab exists.
///
/// Working memory holds the slot exclusively: the last slab dropped returns
/// it to the free pool. A cache-packed slab instead holds a reader pin on a
/// slot a cache owns: the last pin dropped only makes the slot *reclaimable*,
/// and the cache's CLOCK decides when it actually goes.
#[derive(Clone)]
pub enum SlabBacking {
    Working(Arc<WriteBuffer>),
    Cached(Arc<ReadBuffer>),
}

impl SlabBacking {
    fn downgrade(&self) -> WeakSlabBacking {
        match self {
            SlabBacking::Working(buffer) => WeakSlabBacking::Working(Arc::downgrade(buffer)),
            SlabBacking::Cached(buffer) => WeakSlabBacking::Cached(Arc::downgrade(buffer)),
        }
    }
}

/// A [`SlabBacking`] that does not keep its slot alive.
#[derive(Clone)]
enum WeakSlabBacking {
    Working(Weak<WriteBuffer>),
    Cached(Weak<ReadBuffer>),
}

impl WeakSlabBacking {
    /// The backing, holding the slot again, if the slot is still resident.
    fn upgrade(&self) -> Option<SlabBacking> {
        match self {
            WeakSlabBacking::Working(buffer) => buffer.upgrade().map(SlabBacking::Working),
            WeakSlabBacking::Cached(buffer) => buffer.upgrade().map(SlabBacking::Cached),
        }
    }
}

/// A contiguous byte region within one ring slot.
///
/// Keeps the slot alive via its [`SlabBacking`]. Multiple slabs can share the same
/// slot; what happens when the last one drops depends on the backing.
pub struct Slab {
    /// Raw pointer to the start of this slab's region within the parent slot.
    pub(crate) ptr: *mut u8,
    /// Size of this slab in bytes.
    pub(crate) size: usize,
    /// Shared ownership of the parent slot.
    backing: SlabBacking,
}

unsafe impl Send for Slab {}

unsafe impl Sync for Slab {}

impl Slab {
    /// A slab over `size` bytes starting at `ptr`, which must lie inside the
    /// slot `backing` holds. The slab keeps that slot alive for as long as it
    /// exists.
    pub(crate) fn new(ptr: *mut u8, size: usize, backing: SlabBacking) -> Self {
        Self { ptr, size, backing }
    }

    /// Zeroes the slab's entire region.
    pub fn zero_out(&mut self) {
        unsafe {
            std::ptr::write_bytes(self.ptr, 0, self.size);
        }
    }

    /// Bytes this slab holds.
    pub fn size(&self) -> usize {
        self.size
    }

    /// The slab's region as bytes.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: the region is `size` bytes of live buffer memory, kept alive
        // by the backing this slab holds, and no other slab overlaps it, since
        // whoever carved it bumps past each one it carves.
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }

    /// The slab's region as mutable bytes. Same reasoning as
    /// [`as_slice`](Self::as_slice), with `&mut self` for the exclusivity.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }

    /// A read-only handle over the same bytes, sharing the backing, while this
    /// writable one lives on. The reader must not look until the writer is
    /// done; the cache fill hands the writable slab to a builder and reads
    /// back through the view only after the builder returned.
    pub(crate) fn shared_view(&self) -> FrozenSlab {
        FrozenSlab {
            ptr: self.ptr,
            size: self.size,
            backing: self.backing.clone(),
        }
    }

    /// Give up writing to this slab for good, so that handles to it can be
    /// shared.
    pub fn freeze(self) -> FrozenSlab {
        self.shared_view()
    }
}

/// A slab that can no longer be written, so any number of handles to it may
/// exist at once: cloning one is cheap and shares the same bytes. Every handle
/// keeps the parent slot alive, and the last one dropped releases it.
#[derive(Clone)]
pub struct FrozenSlab {
    ptr: *const u8,
    size: usize,
    /// Keeps the slot alive.
    backing: SlabBacking,
}

unsafe impl Send for FrozenSlab {}

unsafe impl Sync for FrozenSlab {}

impl FrozenSlab {
    /// Bytes this slab holds.
    pub fn size(&self) -> usize {
        self.size
    }

    /// The slab's region as bytes.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: the region is `size` bytes of live buffer memory, kept alive
        // by the backing every handle holds, and nothing writes it any more:
        // freezing consumed the only writable handle.
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }

    /// A handle to this slab that does not keep its slot alive.
    pub fn downgrade(&self) -> WeakSlab {
        WeakSlab {
            ptr: self.ptr,
            size: self.size,
            backing: self.backing.downgrade(),
        }
    }
}

impl AsRef<[u8]> for FrozenSlab {
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

/// A handle to a frozen slab that does not keep its slot alive. Whoever owns
/// the slot may give it up while this exists; upgrading tells whether it has,
/// and yields the slab, holding the slot again, when it has not.
#[derive(Clone)]
pub struct WeakSlab {
    ptr: *const u8,
    size: usize,
    backing: WeakSlabBacking,
}

unsafe impl Send for WeakSlab {}

unsafe impl Sync for WeakSlab {}

impl WeakSlab {
    /// Bytes the slab holds, whether or not it is still in memory.
    pub fn size(&self) -> usize {
        self.size
    }

    /// The slab, holding its slot, if the slot is still resident.
    pub fn upgrade(&self) -> Option<FrozenSlab> {
        self.backing.upgrade().map(|backing| FrozenSlab {
            ptr: self.ptr,
            size: self.size,
            backing,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::memory::{SlabAllocator, init_test_free_pool};

    #[test]
    fn slab_zero_out() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(false);
        let mut slab = alloc.get_slab_of_size(256, false);
        unsafe {
            std::ptr::write_bytes(slab.ptr, 0xFF, 256);
        }

        slab.zero_out();

        let slice = unsafe { std::slice::from_raw_parts(slab.ptr, 256) };
        assert!(slice.iter().all(|&b| b == 0));
    }
}
