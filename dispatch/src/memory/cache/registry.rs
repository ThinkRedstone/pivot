//! [`Caches`]: every cache, indexed by [`CacheId`], and the families that
//! group them.
//!
//! ```text
//!   Caches
//!   ├── caches[id]    Arc<dyn RingCache>      what the evictor reclaims from
//!   ├── policies[id]  CachePolicy             name, lives at insert, lives per touch
//!   ├── compressed[id] Option<Arc<CompressedCache>>   typed handle, for the IO path
//!   ├── families[f]   FamilySpec              members and their target shares
//!   └── family_of[id] f                       which family a cache belongs to
//! ```
//!
//! Built once at startup through [`CachesBuilder`] and shared by every worker's
//! memory context. The hot path through here is
//! [`reclaim`](Caches::reclaim): reclaim from the owning cache, then reinforce
//! its siblings over whatever it dropped.

use crate::io::OpenFile;
use crate::memory::WriteBuffer;
use crate::memory::cache::cached::Cached;
use crate::memory::cache::compressed::CompressedCache;
use crate::memory::cache::derived_cache::Cache;
use crate::memory::cache::family::FamilySpec;
use crate::memory::cache::id::{CacheId, CachePolicy};
use crate::memory::cache::ring_cache::RingCache;
use std::sync::Arc;

pub struct Caches {
    caches: Vec<Arc<dyn RingCache>>,
    policies: Vec<CachePolicy>,
    compressed: Vec<Option<Arc<CompressedCache>>>,
    families: Vec<FamilySpec>,
    family_of: Vec<usize>,
}

impl Caches {
    pub fn len(&self) -> usize {
        self.caches.len()
    }

    pub fn is_empty(&self) -> bool {
        self.caches.is_empty()
    }

    pub fn ids(&self) -> impl Iterator<Item = CacheId> {
        (0..self.caches.len()).map(|index| CacheId::new(index as u8))
    }

    pub fn policies(&self) -> &[CachePolicy] {
        &self.policies
    }

    pub fn policy(&self, id: CacheId) -> &CachePolicy {
        &self.policies[id.index()]
    }

    pub fn name(&self, id: CacheId) -> &'static str {
        self.policy(id).name
    }

    pub fn cache(&self, id: CacheId) -> &dyn RingCache {
        self.caches[id.index()].as_ref()
    }

    /// The compressed cache registered as `id`. Panics for an id that names a
    /// derived cache: only reads of raw file bytes come through here, and
    /// they always name a compressed cache.
    pub fn compressed(&self, id: CacheId) -> &CompressedCache {
        self.compressed[id.index()]
            .as_deref()
            .unwrap_or_else(|| panic!("cache {} is not a compressed cache", self.name(id)))
    }

    pub fn family_of(&self, id: CacheId) -> &FamilySpec {
        &self.families[self.family_of[id.index()]]
    }

    /// Every family, in the order they were added.
    pub fn families(&self) -> &[FamilySpec] {
        &self.families
    }

    /// How many members the largest family has: the most hands the evictor
    /// may have to try in turn before a family has nothing more to give.
    pub fn count_members_of_largest_family(&self) -> usize {
        self.families
            .iter()
            .map(FamilySpec::count_members)
            .max()
            .expect("the engine registers at least one family of caches")
    }

    /// Reclaim `slot` from cache `id`, then reinforce its siblings over every
    /// range that died with it. The reinforcement runs after the cache has
    /// released its locks (its `reclaim` contract), so no cache ever holds a
    /// lock while another is entered: there is no lock order between caches
    /// to get wrong.
    pub fn reclaim(&self, id: CacheId, slot: usize) -> Option<WriteBuffer> {
        let reclaimed = self.cache(id).reclaim(slot)?;
        let family = self.family_of(id);
        for range in &reclaimed.dropped {
            for sibling in family.siblings_of(id) {
                self.cache(sibling).reinforce_range(range);
            }
        }
        Some(reclaimed.buffer)
    }

    /// Drop everything every cache holds of a local file that was just
    /// (re)opened: its descriptor may now name a different file than an
    /// equal, earlier descriptor did, so nothing cached under it may serve a
    /// later read. A remote object is keyed by its own identity and keeps
    /// what it cached.
    pub fn invalidate_reopened_file(&self, file: &OpenFile) {
        if !matches!(file, OpenFile::Local(_)) {
            return;
        }
        for cache in &self.caches {
            cache.invalidate(file);
        }
    }

    /// Drop everything from every cache; the total entries dropped.
    pub fn clear_all(&self) -> usize {
        self.caches.iter().map(|cache| cache.clear()).sum()
    }
}

/// Assembles a [`Caches`], one family at a time. The builder constructs each
/// cache itself, so a cache gets its id, its place in the registry and its
/// family membership in a single call:
///
/// ```text
///   let mut builder = CachesBuilder::new(slot_count);
///   let pages = builder.add_family(|family| PageCaches {
///       compressed: family.add_compressed(compressed_policy, 30),
///       decompressed: family.add_derived(decompressed_policy, 70, "KILL_SWITCH"),
///   });
///   let caches = builder.build();
/// ```
pub struct CachesBuilder {
    slot_count: usize,
    caches: Vec<Arc<dyn RingCache>>,
    policies: Vec<CachePolicy>,
    compressed: Vec<Option<Arc<CompressedCache>>>,
    families: Vec<FamilySpec>,
    family_of: Vec<usize>,
}

impl CachesBuilder {
    /// `slot_count` is the size of the ring every cache will pack into.
    pub fn new(slot_count: usize) -> Self {
        Self {
            slot_count,
            caches: Vec::new(),
            policies: Vec::new(),
            compressed: Vec::new(),
            families: Vec::new(),
            family_of: Vec::new(),
        }
    }

    /// Add one family. `declare` adds the family's members through the
    /// [`FamilyBuilder`] it is handed and returns whatever it assembled from
    /// their handles, which is passed back to the caller. The members' target
    /// shares must sum to 100.
    pub fn add_family<Handles>(
        &mut self,
        declare: impl FnOnce(&mut FamilyBuilder) -> Handles,
    ) -> Handles {
        let mut family = FamilyBuilder {
            builder: self,
            members: Vec::new(),
        };
        let handles = declare(&mut family);
        let members = family.members;
        self.families.push(FamilySpec::new(members));
        handles
    }

    pub fn build(self) -> Caches {
        Caches {
            caches: self.caches,
            policies: self.policies,
            compressed: self.compressed,
            families: self.families,
            family_of: self.family_of,
        }
    }

    /// The id the next cache added will be built with.
    fn next_id(&self) -> CacheId {
        assert!(
            self.caches.len() < CacheId::MAX_CACHES,
            "too many caches added"
        );
        CacheId::new(self.caches.len() as u8)
    }
}

/// The members of one family, while it is being added to a [`CachesBuilder`].
pub struct FamilyBuilder<'a> {
    builder: &'a mut CachesBuilder,
    members: Vec<(CacheId, u8)>,
}

impl FamilyBuilder<'_> {
    /// Add a cache of raw file bytes that should hold `target_share_pct` of
    /// the family's slots.
    pub fn add_compressed(
        &mut self,
        policy: CachePolicy,
        target_share_pct: u8,
    ) -> Arc<CompressedCache> {
        let id = self.builder.next_id();
        let cache = Arc::new(CompressedCache::new(id, self.builder.slot_count));
        self.register_member(policy, target_share_pct, cache.clone(), Some(cache.clone()));
        cache
    }

    /// Add a cache of the derived form `V` that should hold
    /// `target_share_pct` of the family's slots. `kill_switch` names the
    /// environment variable that disables it.
    pub fn add_derived<V: Cached>(
        &mut self,
        policy: CachePolicy,
        target_share_pct: u8,
        kill_switch: &str,
    ) -> Arc<Cache<V>> {
        let id = self.builder.next_id();
        let cache = Arc::new(Cache::new(id, self.builder.slot_count, kill_switch));
        self.register_member(policy, target_share_pct, cache.clone(), None);
        cache
    }

    /// Record `cache` in the registry and as a member of this family.
    /// `compressed` is its typed handle when it holds raw file bytes.
    fn register_member(
        &mut self,
        policy: CachePolicy,
        target_share_pct: u8,
        cache: Arc<dyn RingCache>,
        compressed: Option<Arc<CompressedCache>>,
    ) {
        self.members.push((cache.id(), target_share_pct));
        self.builder.family_of.push(self.builder.families.len());
        self.builder.policies.push(policy);
        self.builder.compressed.push(compressed);
        self.builder.caches.push(cache);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::cache::decompressed_page::DecompressedPage;

    fn policy(name: &'static str) -> CachePolicy {
        CachePolicy {
            name,
            lives_at_insert: 1,
            lives_per_touch: 1,
            lives_at_reinforce: 1,
        }
    }

    #[test]
    fn caches_added_to_one_family_are_siblings() {
        let mut builder = CachesBuilder::new(4);

        let (compressed, decoded) = builder.add_family(|family| {
            (
                family.add_compressed(policy("compressed"), 30),
                family.add_derived::<DecompressedPage>(policy("decoded"), 70, "TEST_KILL_SWITCH"),
            )
        });
        let caches = builder.build();

        let siblings: Vec<CacheId> = caches
            .family_of(compressed.id())
            .siblings_of(compressed.id())
            .collect();
        assert_eq!(siblings, vec![decoded.id()]);
    }

    #[test]
    fn caches_added_to_different_families_are_not_siblings() {
        let mut builder = CachesBuilder::new(4);

        let first = builder.add_family(|family| family.add_compressed(policy("first"), 100));
        let second = builder.add_family(|family| family.add_compressed(policy("second"), 100));
        let caches = builder.build();

        assert_ne!(first.id(), second.id());
        assert!(!caches.family_of(first.id()).contains(second.id()));
        assert_eq!(caches.name(second.id()), "second");
    }

    #[test]
    #[should_panic(expected = "sum to 100")]
    fn a_family_whose_shares_do_not_sum_to_one_hundred_is_rejected() {
        let mut builder = CachesBuilder::new(4);

        builder.add_family(|family| family.add_compressed(policy("lonely"), 30));
    }
}
