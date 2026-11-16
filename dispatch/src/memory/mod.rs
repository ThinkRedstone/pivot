//! The `memory` module contains all dispatch's memory management/utilities.
//!
//! There are a few targets in our memory management:
//!     1. Hit as few page faults as possible
//!     2. Know when we're hitting disk and when we have something cached
//!     3. Allow spilling to disk, into a [`SpillDirectory`] the user names
//!
//! Memory usage is divided into two- [`WriteBuffer`]s (each 2 mb) for large allocations,
//! regular `malloc` for either small allocations or allocations that necessitate continuous memory.
//!
//! Our assumption is that we're the only important process on the machine; we therefore immediately
//! allocate on initialization half the system memory for [`WriteBuffer`]s, as this will be our main
//! usage of memory. Every worker upon initialization faults in all [`WriteBuffer`]s and pushes equal
//! amounts to local pools for use afterward. These [`WriteBuffer`]s will be used for any disk access
//! (and subsequently saved in the [`CompressedCache`]) as well as large allocations.
//!
//! [`WriteBuffer`]s can also be used for many miscellaneous things, such as Vectors and HashTables. It
//! is generally preferred to use [`WriteBuffer`]s as the memory is easily accounted for. See [`SlabAllocator`].
//! Bytes on their way out of the engine are held the same way: an encoded Parquet file is a
//! [`FileBytes`] of the slabs its pages were written into, so a large write is accounted against the
//! ring rather than growing the heap beside it.
//!
//! On a multi-NUMA-node machine the one ring is split into a contiguous region per node
//! (see [`RingLayout`]): a worker faults, acquires, and evicts only its own node's slots,
//! so every allocation is node-local memory, while data cached anywhere in the ring stays
//! readable by every worker (a remote read beats re-reading from disk).

mod ring;
pub use ring::{BUFFER_SIZE, Ring, SlotUsage};

mod status;
pub use status::{MemoryBlockState, MemoryBlockStatus, block_size_bytes};

mod layout;
pub use layout::RingLayout;

pub mod cache;
pub use cache::{
    Cache, CacheFill, CacheId, CacheKey, CacheLookup, Cached, Caches, CompressedCache,
    DecompressedPage, DecompressedPageMeta, MissingExtent, PageCaches, RingCache, Segment,
    SourceRange,
};

pub mod clock;
pub use clock::Clock;

mod file_bytes;
pub use file_bytes::FileBytes;

mod free_pool;

mod write_buffer;
pub use write_buffer::WriteBuffer;

mod read_buffer;
pub use read_buffer::ReadBuffer;

mod slab;
pub use slab::{FrozenSlab, MultiSlabBuffer, Slab, SlabAllocator, SlabBuffer, SlabVec, WeakSlab};
pub(crate) use slab::{HeapBuffer, MultiTopK, Ranked, SingleTopK, SlabTopK, slots_per_slab};

mod reader;
pub use reader::{MultiBufferReader, ReaderPosition};

mod context;
pub use context::{MemoryContextFactory, has_memory_context, init_memory_context, memory_ctx};

pub mod evict;
pub use evict::{DataflowEvict, Evict};

mod spill;
#[cfg(test)]
pub(crate) use spill::TestSpillDirectory;
pub use spill::{
    Dynamic, Fixed, Layout, NO_SLAB, PARTITIONS, PartitionAllocator, PartitionContents,
    PartitionEvictor, PartitionSlab, SPILL_BLOCK_SIZE, SharedPartitionEvictor, SlabEntry,
    SpillDirectory, SpillDirectoryError, SpilledBuffer, SpilledSlab, Spiller,
    decompress_spilled_rows, padded_to_spill_block,
};

#[cfg(any(test, feature = "test-util"))]
pub use context::init_test_free_pool;
