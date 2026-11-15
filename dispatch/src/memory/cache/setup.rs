//! The caches this engine runs with, declared in one place.
//!
//! ```text
//!   family   member               lives (insert / touch / reinforce)   share
//!   ──────   ──────               ──────────────────────────────────   ─────
//!   pages    compressed_cache     1 / 1 / 6                            30 %
//!            decompressed_cache   1 / 1 / 6                            70 %
//! ```
//!
//! The compressed share and the reinforce value are the defaults of the
//! environment variables below.

use crate::env::get_env_var_with_default;
use crate::memory::cache::decompressed_page::PageCaches;
use crate::memory::cache::id::CachePolicy;
use crate::memory::cache::registry::{Caches, CachesBuilder};
use std::sync::Arc;

/// Reads the target share of the pages family's slots the compressed cache
/// should hold, in percent (env `PIVOT_COMPRESSED_CACHE_PCT`, default 30). The
/// decompressed cache's target is the rest.
fn get_compressed_share_pct() -> u8 {
    let compressed_share_pct = get_env_var_with_default("PIVOT_COMPRESSED_CACHE_PCT", 30u8);
    assert!(
        compressed_share_pct <= 100,
        "PIVOT_COMPRESSED_CACHE_PCT is a percentage, got {compressed_share_pct}"
    );
    compressed_share_pct
}

/// Reads the lives a page slot gains per read once its sibling dropped its
/// copy of the same bytes (env `PIVOT_REINFORCE_BUMP`, default 6).
fn get_lives_at_reinforce() -> u8 {
    get_env_var_with_default("PIVOT_REINFORCE_BUMP", 6)
}

/// Every cache, as the registry the evictor uses and as the typed handles the
/// readers use. The handles hold the same `Arc`s the registry does.
#[derive(Clone)]
pub struct CacheSetup {
    pub caches: Arc<Caches>,
    pub pages: PageCaches,
}

impl CacheSetup {
    /// The standard set over a ring of `slot_count` slots.
    pub fn standard(slot_count: usize) -> Self {
        let compressed_share_pct = get_compressed_share_pct();
        let lives_at_reinforce = get_lives_at_reinforce();
        let mut builder = CachesBuilder::new(slot_count);

        let pages = builder.add_family(|family| PageCaches {
            compressed: family.add_compressed(
                CachePolicy {
                    name: "compressed_cache",
                    lives_at_insert: 1,
                    lives_per_touch: 1,
                    lives_at_reinforce,
                },
                compressed_share_pct,
            ),
            decompressed: family.add_derived(
                CachePolicy {
                    name: "decompressed_cache",
                    lives_at_insert: 1,
                    lives_per_touch: 1,
                    lives_at_reinforce,
                },
                100 - compressed_share_pct,
                "PIVOT_DECOMPRESSED_CACHE",
            ),
        });

        Self {
            caches: Arc::new(builder.build()),
            pages,
        }
    }
}
