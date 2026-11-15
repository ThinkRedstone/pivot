//! The one interface every cache presents to the evictor and the registry.
//!
//! Two types implement it: [`CompressedCache`](super::CompressedCache), which
//! holds raw file bytes, and [`Cache<V>`](super::Cache), which holds a form
//! derived from them. The evictor only ever sees `dyn RingCache`, so it can
//! reclaim a slot from whichever cache the clock says owns it without knowing
//! what the slot holds.

use crate::io::OpenFile;
use crate::memory::WriteBuffer;
use crate::memory::cache::id::CacheId;

/// The file bytes a cache entry was derived from. Every entry in every cache
/// has one, and it is how entries of different caches are matched up: a
/// decompressed page and the compressed extent it came from overlap in the
/// file even though neither knows the other exists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceRange {
    pub file: OpenFile,
    pub offset: usize,
    pub len: usize,
}

impl SourceRange {
    pub fn new(file: OpenFile, offset: usize, len: usize) -> Self {
        Self { file, offset, len }
    }

    pub fn end(&self) -> usize {
        self.offset + self.len
    }
}

/// What a successful [`RingCache::reclaim`] hands back: the slot, now the
/// caller's working memory, and the source ranges of every entry that died
/// with it, for the registry to reinforce the siblings over.
pub struct Reclaimed {
    pub buffer: WriteBuffer,
    pub dropped: Vec<SourceRange>,
}

pub trait RingCache: Send + Sync {
    fn id(&self) -> CacheId;

    /// The clock picked `slot`, owned by this cache, as a victim. Take it
    /// exclusively, drop every entry with bytes in it, release it from the
    /// clock and return it. `None` when a reader still pins the slot or it is
    /// no longer this cache's. Holds none of its own locks on return, so the
    /// caller may reach into any cache next.
    fn reclaim(&self, slot: usize) -> Option<Reclaimed>;

    /// A sibling dropped its copy of `range`: give every entry of this cache
    /// overlapping it the reinforced bump, so the surviving copy climbs faster
    /// than ordinary entries under the same pressure.
    fn reinforce_range(&self, range: &SourceRange);

    /// Drop every entry of `file`, because a reopened descriptor may now name
    /// different bytes. Slots stay bound and recycle lazily through the clock.
    fn invalidate(&self, file: &OpenFile);

    /// Drop every entry and recycle every slot that can be taken exclusively,
    /// returning how many entries were dropped. Backs `drop_cache()`.
    fn clear(&self) -> usize;
}
