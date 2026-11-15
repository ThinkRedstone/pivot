//! The pages family: Parquet pages as read ([`CompressedCache`]) and as
//! decompressed ([`DecompressedPage`]).
//!
//! A decompressed page is just bytes, so its cached form is the raw slabs it
//! was decompressed into, which may be split across slots (the decompressors
//! write scattered). Beside the bytes the cache keeps the page's serialized
//! header, so a range lookup can hand back a complete page without touching
//! disk, and a counter of the row group's resident pages, which the reader
//! uses to scan cache-first.

use crate::memory::cache::cached::Cached;
use crate::memory::cache::compressed::CompressedCache;
use crate::memory::cache::derived_cache::Cache;
use crate::memory::slab::FrozenSlab;
use bytes::Bytes;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// A page's decompressed bytes, pinned for as long as this is held, with the
/// header bytes the reader stored beside them.
pub struct DecompressedPage {
    pub header: Vec<Bytes>,
    pub data: Vec<Bytes>,
}

/// What the cache keeps beside a page's bytes.
pub struct DecompressedPageMeta {
    header: Vec<Bytes>,
    _live: LivePage,
}

impl DecompressedPageMeta {
    /// `header` is opaque to the cache and handed back verbatim on a hit.
    /// `live_pages` counts the resident pages of whatever grouping the caller
    /// tracks; this page counts toward it from now until its entry is gone.
    pub fn new(header: Vec<Bytes>, live_pages: Arc<AtomicUsize>) -> Self {
        Self {
            header,
            _live: LivePage::count(live_pages),
        }
    }
}

/// One page's membership in a caller-supplied residency count: counted on
/// creation, uncounted on drop, so the count tracks residency exactly through
/// every removal path (eviction, invalidation, clear, a lost insert race).
struct LivePage(Arc<AtomicUsize>);

impl LivePage {
    fn count(pages: Arc<AtomicUsize>) -> Self {
        pages.fetch_add(1, Ordering::Relaxed);
        Self(pages)
    }
}

impl Drop for LivePage {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

impl Cached for DecompressedPage {
    type Meta = DecompressedPageMeta;

    fn from_slabs(meta: &Self::Meta, slabs: Vec<FrozenSlab>) -> Self {
        Self {
            header: meta.header.clone(),
            data: slabs.into_iter().map(Bytes::from_owner).collect(),
        }
    }
}

/// The pages family's caches, for the Parquet reader.
#[derive(Clone)]
pub struct PageCaches {
    pub compressed: Arc<CompressedCache>,
    pub decompressed: Arc<Cache<DecompressedPage>>,
}
