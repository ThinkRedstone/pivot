//! The memory ring consists of fixed-size blocks.
//! Each block is held by one of the caches, pinned by an operator, or free.
//! Because every block is the same size, counting rows measures memory usage.

use crate::memory::BUFFER_SIZE;
use crate::memory::cache::Caches;
use crate::memory::clock::Clock;
use crate::memory::layout::RingLayout;
use crate::memory::ring::{Ring, SlotUsage};

/// What one block of the ring is being used for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MemoryBlockState {
    /// Held by no one and owned by no cache: sitting in a free pool, ready to
    /// be handed out without evicting anything.
    Free,
    /// Held by an operator as working memory - a
    /// [`WriteBuffer`](crate::memory::WriteBuffer) backing a hash table, a sort
    /// run, a read in flight. Not evictable: it is released when the work that
    /// took it finishes.
    Pinned,
    /// Lent to the cache named, holding whatever form of data that cache keeps.
    Cached { cache: &'static str },
}

/// What one block of the ring currently holds.
#[derive(Clone, Copy, Debug)]
pub struct MemoryBlockStatus {
    /// The block's position in the ring.
    pub slot: usize,
    /// The NUMA node whose region owns the block. Its workers are the only ones
    /// that may allocate or evict it, so pressure is a per-node property.
    pub node: usize,
    pub state: MemoryBlockState,
    /// Live holds on the block's bytes, counting a cache's own hold on what it
    /// has filled, so a cached block at rest commonly reports one. Zero is what
    /// eviction needs: a block anyone is holding cannot be taken back yet. A
    /// block a writer has exclusively reports none, the two being mutually
    /// exclusive.
    pub readers: u32,
}

/// Bytes in one block, the unit everything in the ring is allocated in.
pub fn block_size_bytes() -> usize {
    BUFFER_SIZE
}

/// Read what one block currently holds.
///
/// Its two words are read one after the other while the workers around them
/// keep allocating, caching, and evicting, so this describes the block at the
/// instant it was read. Holding it still to answer would cost far more than
/// the reading is worth, and measuring memory pressure does not need it.
pub(crate) fn read_block_status(
    ring: &Ring,
    clock: &Clock,
    caches: &Caches,
    layout: &RingLayout,
    slot: usize,
) -> MemoryBlockStatus {
    let usage = ring.slot_usage(slot);
    // Ownership decides the state first: a cache that owns a block accounts for
    // it whether or not a reader happens to be holding it at this instant,
    // since the block goes on holding cached bytes either way. Only a block no
    // cache owns is the free pool's or an operator's, and there the holder
    // decides.
    let state = match (clock.owner(slot), usage) {
        (Some(cache), _) => MemoryBlockState::Cached {
            cache: caches.name(cache),
        },
        (None, SlotUsage::Idle) => MemoryBlockState::Free,
        (None, SlotUsage::Writing | SlotUsage::Reading(_)) => MemoryBlockState::Pinned,
    };
    MemoryBlockStatus {
        slot,
        node: layout.node_of_slot(slot),
        state,
        readers: match usage {
            SlotUsage::Reading(readers) => readers,
            SlotUsage::Idle | SlotUsage::Writing => 0,
        },
    }
}

#[cfg(test)]
mod tests {
    //! The classification, exercised through the worker-side read that serves
    //! it. Each test installs its own memory context (and its own ring) on its
    //! own thread, so nothing is shared and no lock is needed.

    use super::*;
    use crate::memory::cache::RingCache;
    use crate::memory::{WriteBuffer, init_test_free_pool, memory_ctx};

    const COMPRESSED: MemoryBlockState = MemoryBlockState::Cached {
        cache: "compressed_cache",
    };
    const DECOMPRESSED: MemoryBlockState = MemoryBlockState::Cached {
        cache: "decompressed_cache",
    };

    fn get_compressed_cache_id() -> crate::memory::cache::CacheId {
        memory_ctx().page_caches().compressed.id()
    }

    fn get_decompressed_cache_id() -> crate::memory::cache::CacheId {
        memory_ctx().page_caches().decompressed.id()
    }

    /// Every block of the ring, in ring order.
    fn blocks() -> Vec<MemoryBlockStatus> {
        memory_ctx().read_all_blocks()
    }

    fn states() -> Vec<MemoryBlockState> {
        blocks().into_iter().map(|block| block.state).collect()
    }

    /// Take block `slot` for writing, the way an operator asking for working
    /// memory ends up holding one.
    fn hold_for_writing(slot: usize) -> WriteBuffer {
        memory_ctx().ring().try_write(slot).unwrap()
    }

    #[test]
    fn every_block_of_a_fresh_ring_is_free() {
        init_test_free_pool(0);

        let blocks = blocks();

        assert_eq!(blocks.len(), 128);
        assert!(
            blocks
                .iter()
                .all(|block| block.state == MemoryBlockState::Free)
        );
    }

    #[test]
    fn the_whole_ring_is_read_in_order_with_each_block_named_by_its_position() {
        init_test_free_pool(0);

        let slots: Vec<_> = blocks().into_iter().map(|block| block.slot).collect();

        assert_eq!(slots, (0..128).collect::<Vec<_>>());
    }

    #[test]
    fn a_block_held_for_writing_is_pinned() {
        init_test_free_pool(0);

        let _held = hold_for_writing(1);

        assert_eq!(
            states()[0..2],
            [MemoryBlockState::Free, MemoryBlockState::Pinned]
        );
    }

    #[test]
    fn releasing_a_block_frees_it_again() {
        init_test_free_pool(0);
        let held = hold_for_writing(0);

        drop(held);

        assert_eq!(states()[0], MemoryBlockState::Free);
    }

    #[test]
    fn a_block_reports_the_cache_that_owns_it() {
        init_test_free_pool(0);

        memory_ctx().clock().bind(0, get_compressed_cache_id());
        memory_ctx().clock().bind(1, get_decompressed_cache_id());

        assert_eq!(states()[0..2], [COMPRESSED, DECOMPRESSED]);
    }

    #[test]
    fn a_cached_block_being_read_still_counts_as_cached() {
        init_test_free_pool(0);
        memory_ctx().clock().bind(0, get_decompressed_cache_id());

        let _reader = memory_ctx().ring().try_read(0).unwrap();

        let block = blocks()[0];
        assert_eq!(block.state, DECOMPRESSED);
        assert_eq!(block.readers, 1);
    }

    #[test]
    fn readers_are_counted_per_block() {
        init_test_free_pool(0);

        let _first = memory_ctx().ring().try_read(0).unwrap();
        let _second = memory_ctx().ring().try_read(0).unwrap();

        assert_eq!(blocks()[0].readers, 2);
    }

    #[test]
    fn a_block_a_cache_gives_back_is_free_again() {
        init_test_free_pool(0);
        memory_ctx().clock().bind(0, get_compressed_cache_id());

        memory_ctx().clock().release(0);

        assert_eq!(states()[0], MemoryBlockState::Free);
    }

    #[test]
    fn every_block_is_named_by_the_node_that_owns_it() {
        init_test_free_pool(0);

        let nodes: Vec<_> = blocks().into_iter().map(|block| block.node).collect();

        assert_eq!(nodes, vec![0; 128]);
    }
}
