//! The CLOCK eviction policy over every ring slot, shared by every cache.
//!
//! ```text
//!   region 0 (NUMA node 0)                         region 1
//!   ┌────┬────┬────┬────┬────┬────┬────┬────┐      ┌────┬────┬─ …
//!   │ c0 │ c1 │ -- │ c1 │ c2 │ c0 │ -- │ c1 │      │ c1 │ c0 │
//!   │ l3 │ l0 │    │ l1 │ l7 │ l0 │    │ l2 │      │ l1 │ l5 │
//!   └────┴──▲─┴────┴────┴────┴──▲─┴────┴────┘      └──▲─┴──▲─┴─ …
//!        c1's hand           c0's hand           c1's hand  c0's hand
//!   cN = owned by cache N, -- = no cache, lN = lives left
//! ```
//!
//! ## One hand per cache per region
//!
//! Each cache has a hand of its own in each NUMA region. A hand acts only on
//! slots of its cache: any other slot, free or another cache's, is stepped
//! over without its counter being touched. A slot of the cache spends one
//! life; one with none left is the victim.
//!
//! So a cache's slots age only while that cache is being evicted from, and
//! its hand passes each of them exactly once per revolution, however often
//! the evictor turned to another cache in between. A slot's lives therefore
//! count passes of its own cache's hand since it was last read, the same for
//! every slot of the cache.
//!
//! Which hand moves is the evictor's choice, made from the target shares of
//! the cache's family (see [`FamilySpec`](crate::memory::cache::FamilySpec)),
//! which makes the balance between the forms of the same data an explicit
//! target instead of whatever their lifetimes add up to.
//!
//! The sweep is per region so eviction only ever frees slots the evicting
//! worker may acquire (node-local memory, see
//! [`RingLayout`](super::RingLayout)); touches are global, since any worker's
//! hit may land on any slot.
//!
//! ## Weights
//!
//! Each cache's [`CachePolicy`] says how many lives a slot of its starts with
//! and how many a touch adds. A hand takes one life per pass, so a slot of a
//! cache with a heavier policy survives more passes untouched, and a cache
//! nobody reads still drains and goes. Lives are capped at [`MAX_LIVES`] so
//! no slot survives more than that many passes untouched.
//!
//! ## Reinforcement
//!
//! When a cache drops its copy of some bytes, the surviving copies in its
//! family are reinforced: their per-touch bump is raised to their cache's
//! `lives_at_reinforce`. A copy still being read then climbs faster than
//! eviction pressure drains it, so the last copy of hot bytes effectively
//! stops being a victim; once cold, it drains like anything else.
//!
//! Everything is lock-free; a little racing between the atomics is fine
//! because eviction only needs to be approximate.

use crate::env::get_env_var_with_default;
use crate::memory::cache::{CacheId, CachePolicy};
use std::sync::LazyLock;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Relaxed;

/// Hard ceiling on any slot's accumulated lives (env `PIVOT_MAX_LIVES`,
/// default 16).
static MAX_LIVES: LazyLock<u8> = LazyLock::new(|| get_env_var_with_default("PIVOT_MAX_LIVES", 16));

const NO_OWNER: u8 = 0;

/// What one step of a cache's hand did.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Sweep {
    /// The slot is free or another cache's: stepped over.
    Skipped,
    /// The slot is the cache's and spent one of its lives.
    Aged,
    /// The slot is the cache's and had no life left: the victim.
    Candidate(usize),
}

/// The clock's view of a cache's policy: what a bound slot starts with and
/// what a reinforced one moves to.
#[derive(Clone, Copy)]
struct Lives {
    at_insert: u8,
    per_touch: u8,
    at_reinforce: u8,
}

/// How many of one cache's slots in one region readers hold. Kept on cache
/// lines of its own: every worker of the region writes it as its readers take
/// and let go of slots, far more often than anything else in the clock is
/// written, and a neighbour on the same line would be slowed by each of those
/// writes.
#[repr(align(128))]
#[derive(Default)]
struct PinnedCount(AtomicUsize);

struct Slot {
    owner: AtomicU8,
    lives: AtomicU8,
    /// Lives gained per touch: the owner's `lives_per_touch`, or its
    /// `lives_at_reinforce` once a sibling copy has died.
    bump: AtomicU8,
}

pub struct Clock {
    slots: Box<[Slot]>,
    /// One sweep hand per cache per region, `[cache][region]`; region `r`
    /// covers `r * slots_per_region..(r + 1) * slots_per_region`.
    hands: Box<[Box<[AtomicUsize]>]>,
    slots_per_region: usize,
    /// Slots each cache owns in each region, `[cache][region]`, maintained by
    /// [`bind`](Self::bind) and [`release`](Self::release), the two functions
    /// every ownership change goes through. Per region because the hands are:
    /// a region's evictor can only take from the caches as they stand in its
    /// own region, so that is the balance it must judge.
    owned: Box<[Box<[AtomicUsize]>]>,
    /// Of the slots counted above, how many at least one reader holds,
    /// `[cache][region]`. Eviction cannot take such a slot until its last
    /// reader lets go, so a cache whose every slot in a region is pinned has
    /// nothing to give there however long its hand sweeps. Maintained by
    /// [`count_pinned`](Self::count_pinned) and
    /// [`count_unpinned`](Self::count_unpinned), which the ring calls as a
    /// slot gains its first reader and loses its last.
    pinned: Box<[Box<[PinnedCount]>]>,
    lives: Box<[Lives]>,
}

impl Clock {
    /// A single-region clock over `len` slots (single-node machines, tests).
    pub fn new(len: usize, policies: &[CachePolicy]) -> Self {
        Self::with_regions(1, len, policies)
    }

    /// A clock over `region_count * slots_per_region` slots with one hand per
    /// region for each of the caches `policies` describes (indexed by
    /// [`CacheId`]).
    pub fn with_regions(
        region_count: usize,
        slots_per_region: usize,
        policies: &[CachePolicy],
    ) -> Self {
        Self {
            slots: (0..region_count * slots_per_region)
                .map(|_| Slot {
                    owner: AtomicU8::new(NO_OWNER),
                    lives: AtomicU8::new(0),
                    bump: AtomicU8::new(1),
                })
                .collect(),
            hands: policies
                .iter()
                .map(|_| (0..region_count).map(|_| AtomicUsize::new(0)).collect())
                .collect(),
            slots_per_region,
            owned: policies
                .iter()
                .map(|_| (0..region_count).map(|_| AtomicUsize::new(0)).collect())
                .collect(),
            pinned: policies
                .iter()
                .map(|_| (0..region_count).map(|_| PinnedCount::default()).collect())
                .collect(),
            lives: policies
                .iter()
                .map(|policy| Lives {
                    at_insert: policy.lives_at_insert,
                    per_touch: policy.lives_per_touch,
                    at_reinforce: policy.lives_at_reinforce,
                })
                .collect(),
        }
    }

    fn region_of(&self, slot: usize) -> usize {
        slot / self.slots_per_region
    }

    /// Give a free slot to `cache`, with its policy's starting lives and
    /// ordinary bump.
    pub fn bind(&self, slot: usize, cache: CacheId) {
        let lives = self.lives[cache.index()];
        self.slots[slot].lives.store(lives.at_insert, Relaxed);
        self.slots[slot].bump.store(lives.per_touch, Relaxed);
        self.slots[slot].owner.store(cache.to_owner_byte(), Relaxed);
        self.owned[cache.index()][self.region_of(slot)].fetch_add(1, Relaxed);
    }

    /// Mark a slot used again: gain its bump's worth of lives, up to
    /// [`MAX_LIVES`]. Every use restores what hand passes took, so a slot that
    /// keeps being read keeps climbing; how fast is its bump.
    pub fn touch(&self, slot: usize) {
        let bump = self.slots[slot].bump.load(Relaxed);
        let _ = self.slots[slot]
            .lives
            .fetch_update(Relaxed, Relaxed, |lives| {
                (lives < *MAX_LIVES).then(|| lives.saturating_add(bump).min(*MAX_LIVES))
            });
    }

    /// A sibling copy of the slot's bytes died: raise its per-touch bump to
    /// its cache's `lives_at_reinforce` and grant one bump now. Fire-and-forget:
    /// if the slot was concurrently released, there is no owner to read the
    /// policy from and nothing happens; if it was rebound, [`bind`](Self::bind)
    /// resets the bump and the stray lives merely delay an unrelated block, an
    /// imprecision this CLOCK tolerates everywhere.
    pub fn reinforce(&self, slot: usize) {
        let Some(cache) = self.owner(slot) else {
            return;
        };
        let at_reinforce = self.lives[cache.index()].at_reinforce;
        self.slots[slot].bump.store(at_reinforce, Relaxed);
        self.touch(slot);
    }

    /// Return a slot to no owner: never a victim.
    pub fn release(&self, slot: usize) {
        if let Some(cache) = self.owner(slot) {
            self.owned[cache.index()][self.region_of(slot)].fetch_sub(1, Relaxed);
        }
        self.slots[slot].owner.store(NO_OWNER, Relaxed);
    }

    pub fn owner(&self, slot: usize) -> Option<CacheId> {
        CacheId::from_owner_byte(self.slots[slot].owner.load(Relaxed))
    }

    /// Slots `cache` owns over the whole ring.
    pub fn count_owned(&self, cache: CacheId) -> usize {
        self.owned[cache.index()]
            .iter()
            .map(|count| count.load(Relaxed))
            .sum()
    }

    /// Slots `cache` owns in `region`.
    pub fn count_owned_in_region(&self, cache: CacheId, region: usize) -> usize {
        self.owned[cache.index()][region].load(Relaxed)
    }

    /// Slots any cache owns in `region`.
    pub fn count_cached_in_region(&self, region: usize) -> usize {
        self.owned
            .iter()
            .map(|per_region| per_region[region].load(Relaxed))
            .sum()
    }

    /// Count `slot` as pinned, if a cache owns it. For the ring to call when
    /// the slot gains its first reader.
    ///
    /// A slot is bound and released only while it is held for writing, which
    /// excludes readers. So the owner read here is the slot's owner for as
    /// long as any reader holds it, and the same one
    /// [`count_unpinned`](Self::count_unpinned) is given.
    pub fn count_pinned(&self, slot: usize) {
        if let Some(cache) = self.owner(slot) {
            self.pinned[cache.index()][self.region_of(slot)]
                .0
                .fetch_add(1, Relaxed);
        }
    }

    /// Stop counting `slot` as pinned. For the ring to call when the slot
    /// loses its last reader. `owner` is the slot's owner as read while that
    /// reader still held it: once the reader has let go, eviction may release
    /// the slot before its owner could be read here.
    pub fn count_unpinned(&self, slot: usize, owner: Option<CacheId>) {
        if let Some(cache) = owner {
            let pinned_before = self.pinned[cache.index()][self.region_of(slot)]
                .0
                .fetch_sub(1, Relaxed);
            debug_assert!(
                pinned_before > 0,
                "slot {slot} lost a last reader it was never counted for"
            );
        }
    }

    /// Slots `cache` owns in `region` that no reader holds, which are the
    /// ones its hand could take. The two counts behind it are read one after
    /// the other while peers bind, release, pin and unpin, so the answer may
    /// be off by the few slots that changed in between.
    pub fn count_unpinned_in_region(&self, cache: CacheId, region: usize) -> usize {
        self.count_owned_in_region(cache, region)
            .saturating_sub(self.pinned[cache.index()][region].0.load(Relaxed))
    }

    /// The most passes of its hand any slot can survive untouched (the lives
    /// ceiling). Sizes the evictor's patience thresholds.
    pub fn max_lives(&self) -> u8 {
        *MAX_LIVES
    }

    /// Advance `cache`'s hand in `region` one slot. A slot that is free or
    /// another cache's is stepped over untouched; a slot of `cache` spends a
    /// life, or is the victim if it has none. Only sweeps `region`'s slots,
    /// so the caller is never handed another node's memory.
    pub fn sweep(&self, cache: CacheId, region: usize) -> Sweep {
        let offset =
            self.hands[cache.index()][region].fetch_add(1, Relaxed) % self.slots_per_region;
        let slot = region * self.slots_per_region + offset;
        if self.owner(slot) != Some(cache) {
            return Sweep::Skipped;
        }
        // Decrement while positive; an already-zero counter is the victim.
        let had_a_life = self.slots[slot]
            .lives
            .fetch_update(Relaxed, Relaxed, |lives| (lives > 0).then(|| lives - 1))
            .is_ok();
        if had_a_life {
            Sweep::Aged
        } else {
            Sweep::Candidate(slot)
        }
    }

    /// How many steps `cache`'s hand in `region` has taken in total -
    /// test-only visibility for bounding an eviction's sweep.
    #[cfg(test)]
    pub fn hand_position(&self, cache: CacheId, region: usize) -> usize {
        self.hands[cache.index()][region].load(Relaxed)
    }

    /// The slot's remaining lives - test-only visibility for asserting
    /// reinforcement and aging behaviour.
    #[cfg(test)]
    pub fn lives(&self, slot: usize) -> u8 {
        self.slots[slot].lives.load(Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIGHT: CacheId = CacheId::new(0);
    const HEAVY: CacheId = CacheId::new(1);

    const LIGHT_REINFORCE: u8 = 6;

    fn policies() -> Vec<CachePolicy> {
        vec![
            CachePolicy {
                name: "light",
                lives_at_insert: 1,
                lives_per_touch: 1,
                lives_at_reinforce: LIGHT_REINFORCE,
            },
            CachePolicy {
                name: "heavy",
                lives_at_insert: 3,
                lives_per_touch: 4,
                lives_at_reinforce: 10,
            },
        ]
    }

    fn clock(len: usize) -> Clock {
        Clock::new(len, &policies())
    }

    /// Sweep `cache`'s region-0 hand until a victim turns up.
    fn evict(clock: &Clock, cache: CacheId) -> usize {
        loop {
            if let Sweep::Candidate(slot) = clock.sweep(cache, 0) {
                return slot;
            }
        }
    }

    #[test]
    fn a_free_slot_is_never_a_victim() {
        let clock = clock(1);

        assert_eq!(clock.sweep(LIGHT, 0), Sweep::Skipped);
        assert_eq!(clock.sweep(HEAVY, 0), Sweep::Skipped);
    }

    #[test]
    fn a_bound_slot_starts_with_its_policys_lives() {
        let clock = clock(2);

        clock.bind(0, LIGHT);
        clock.bind(1, HEAVY);

        assert_eq!(clock.lives(0), 1);
        assert_eq!(clock.lives(1), 3);
    }

    #[test]
    fn a_hand_only_takes_slots_of_its_own_cache() {
        let clock = clock(2);
        clock.bind(0, HEAVY);
        clock.bind(1, LIGHT);

        let light_victim = evict(&clock, LIGHT);
        let heavy_victim = evict(&clock, HEAVY);

        assert_eq!(light_victim, 1);
        assert_eq!(heavy_victim, 0);
    }

    #[test]
    fn a_hand_does_not_age_another_caches_slots() {
        let clock = clock(2);
        clock.bind(0, LIGHT);
        clock.bind(1, HEAVY);

        for _ in 0..10 {
            clock.sweep(HEAVY, 0);
        }

        assert_eq!(clock.lives(0), 1, "only its own cache's hand ages a slot");
        assert_eq!(clock.lives(1), 0);
    }

    #[test]
    fn a_revolution_ages_each_slot_of_the_cache_once_whatever_was_swept_between() {
        let clock = clock(4);
        clock.bind(0, HEAVY);
        clock.bind(1, LIGHT);
        clock.bind(2, HEAVY);
        clock.bind(3, LIGHT);

        clock.sweep(HEAVY, 0); // slot 0
        for _ in 0..3 {
            clock.sweep(LIGHT, 0); // the evictor turns to the other cache
        }
        for _ in 0..3 {
            clock.sweep(HEAVY, 0); // slots 1 to 3: the rest of the revolution
        }

        assert_eq!(clock.lives(0), 2);
        assert_eq!(clock.lives(2), 2);
    }

    #[test]
    fn a_touch_adds_the_policys_bump() {
        let clock = clock(2);
        clock.bind(0, LIGHT);
        clock.bind(1, HEAVY);

        clock.touch(0);
        clock.touch(1);

        assert_eq!(clock.lives(0), 2);
        assert_eq!(clock.lives(1), 7);
    }

    #[test]
    fn touches_accumulate_up_to_the_max() {
        let clock = clock(1);
        clock.bind(0, LIGHT);

        for _ in 0..20 {
            clock.touch(0);
        }

        assert_eq!(clock.lives(0), 16);
    }

    #[test]
    fn binding_and_releasing_track_the_per_cache_counts() {
        let clock = clock(2);
        clock.bind(0, LIGHT);
        clock.bind(1, HEAVY);

        clock.release(0);

        assert_eq!(clock.count_owned(LIGHT), 0);
        assert_eq!(clock.count_owned(HEAVY), 1);
        assert_eq!(clock.count_cached_in_region(0), 1);
    }

    #[test]
    fn a_pinned_slot_is_not_one_its_cache_can_give() {
        let clock = clock(2);
        clock.bind(0, LIGHT);
        clock.bind(1, LIGHT);

        clock.count_pinned(0);
        let while_pinned = clock.count_unpinned_in_region(LIGHT, 0);
        clock.count_unpinned(0, clock.owner(0));
        let after_unpinning = clock.count_unpinned_in_region(LIGHT, 0);

        assert_eq!(while_pinned, 1);
        assert_eq!(after_unpinning, 2);
    }

    #[test]
    fn a_pinned_slot_counts_against_its_own_cache_only() {
        let clock = clock(2);
        clock.bind(0, LIGHT);
        clock.bind(1, HEAVY);

        clock.count_pinned(0);

        assert_eq!(clock.count_unpinned_in_region(LIGHT, 0), 0);
        assert_eq!(clock.count_unpinned_in_region(HEAVY, 0), 1);
    }

    #[test]
    fn pinning_a_slot_no_cache_owns_is_not_counted() {
        let clock = clock(2);
        clock.bind(0, LIGHT);

        clock.count_pinned(1);

        assert_eq!(clock.count_unpinned_in_region(LIGHT, 0), 1);
        assert_eq!(clock.count_unpinned_in_region(HEAVY, 0), 0);
    }

    #[test]
    fn reinforce_grants_extra_sweeps_immediately() {
        let clock = clock(1);
        clock.bind(0, LIGHT);
        clock.sweep(LIGHT, 0); // its one insert-time life, spent

        clock.reinforce(0);

        for _ in 0..LIGHT_REINFORCE {
            assert_eq!(clock.sweep(LIGHT, 0), Sweep::Aged);
        }
        assert_eq!(clock.sweep(LIGHT, 0), Sweep::Candidate(0));
    }

    #[test]
    fn reinforce_takes_the_owners_own_value() {
        let clock = clock(2);
        clock.bind(0, LIGHT);
        clock.bind(1, HEAVY);

        clock.reinforce(0);
        clock.reinforce(1);

        assert_eq!(clock.lives(0), 1 + LIGHT_REINFORCE);
        assert_eq!(clock.lives(1), 3 + 10);
    }

    #[test]
    fn reinforcing_a_released_slot_does_nothing() {
        let clock = clock(1);
        clock.bind(0, LIGHT);
        clock.release(0);

        clock.reinforce(0);

        assert_eq!(clock.lives(0), 1, "no owner, no policy to apply");
    }

    #[test]
    fn a_reinforced_slot_climbs_faster_than_an_ordinary_one() {
        let clock = clock(2);
        clock.bind(0, LIGHT);
        clock.bind(1, LIGHT);
        clock.reinforce(0);

        // One read per pass against heavier per-pass hand pressure.
        for _ in 0..4 {
            clock.touch(0);
            clock.touch(1);
            clock.sweep(LIGHT, 0);
            clock.sweep(LIGHT, 0);
        }

        assert!(clock.lives(0) > 8, "last copy should climb");
        assert!(clock.lives(1) <= 2, "ordinary block must not climb");
    }

    #[test]
    fn rebinding_resets_the_reinforced_bump() {
        let clock = clock(1);
        clock.bind(0, LIGHT);
        clock.reinforce(0);

        clock.release(0);
        clock.bind(0, LIGHT);
        clock.touch(0);

        assert_eq!(clock.lives(0), 2, "fresh block must be back to bump 1");
    }

    #[test]
    fn a_released_slot_stops_being_a_victim() {
        let clock = clock(1);
        clock.bind(0, LIGHT);
        clock.release(0);

        assert_eq!(clock.sweep(LIGHT, 0), Sweep::Skipped);
    }

    #[test]
    fn a_regions_sweep_never_touches_another_regions_slots() {
        let clock = Clock::with_regions(2, 2, &policies());
        clock.bind(0, LIGHT);
        clock.bind(3, LIGHT);

        for _ in 0..8 {
            if let Sweep::Candidate(slot) = clock.sweep(LIGHT, 1) {
                assert!(slot >= 2);
            }
        }

        assert_eq!(evict(&clock, LIGHT), 0);
    }
}
