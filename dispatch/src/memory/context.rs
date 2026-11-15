use crate::Identifier;
use crate::WriteRequestId;
use crate::io::{Completion, IORequester};
use crate::memory::cache::{CacheId, CacheSetup, Caches, FillCursor, PageCaches};
use crate::memory::clock::{Clock, Sweep};
use crate::memory::evict::DataflowEvict;
use crate::memory::free_pool::{FreePool, PoolFactory};
use crate::memory::status::{MemoryBlockStatus, read_block_status};
use crate::memory::{BUFFER_SIZE, Ring, RingLayout, WriteBuffer};
use crate::worker::WORKER_IDX;
use std::cell::{Cell, OnceCell, RefCell, UnsafeCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Share of a node's ring slots that spilling brings the reserve back up to.
/// The reserve is what eviction can take back without waiting on a write:
/// slots that are free, or cached and so reclaimable by the CLOCK sweep.
const RECLAIMABLE_SLOTS_TARGET_PERCENT: usize = 20;

/// Share of a node's ring slots below which spilling starts. Lower than the
/// target, so the reserve is left to run down to it and is then refilled
/// towards the target in one go, which gives each worker a batch of buffers
/// to spill together.
const RECLAIMABLE_SLOTS_START_PERCENT: usize = 10;

/// [`RECLAIMABLE_SLOTS_TARGET_PERCENT`], or what `PIVOT_SPILL_RESERVE_PERCENT`
/// sets in its place.
fn get_reclaimable_slots_target_percent() -> usize {
    static TARGET_PERCENT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    crate::env::get_tunable(
        &TARGET_PERCENT,
        "PIVOT_SPILL_RESERVE_PERCENT",
        RECLAIMABLE_SLOTS_TARGET_PERCENT,
        |percent| percent <= 100,
    )
}

/// [`RECLAIMABLE_SLOTS_START_PERCENT`], or what
/// `PIVOT_SPILL_START_RESERVE_PERCENT` sets in its place, which may be no
/// higher than the target. Set to the target, a worker spills as soon as the
/// reserve is at all short. A target set below the usual start is the start
/// too.
fn get_reclaimable_slots_start_percent() -> usize {
    static START_PERCENT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let target = get_reclaimable_slots_target_percent();
    crate::env::get_tunable(
        &START_PERCENT,
        "PIVOT_SPILL_START_RESERVE_PERCENT",
        RECLAIMABLE_SLOTS_START_PERCENT.min(target),
        |percent| percent <= target,
    )
}

thread_local! {
    static MEMORY_CONTEXT_OWNER: RefCell<Option<Box<MemoryContext>>> = const { RefCell::new(None) };
    static MEMORY_CTX_PTR: Cell<*const MemoryContext> = const { Cell::new(std::ptr::null()) };
}

pub fn init_memory_context(ctx: MemoryContext) {
    MEMORY_CONTEXT_OWNER.with_borrow_mut(|slot| {
        let b = Box::new(ctx);
        MEMORY_CTX_PTR.set(&*b as *const MemoryContext);
        *slot = Some(b);
    });
}

pub fn memory_ctx() -> &'static MemoryContext {
    unsafe { &*MEMORY_CTX_PTR.get() }
}

/// True when a [`MemoryContext`] is installed on the current thread - i.e. the
/// caller is running on a dispatch worker (or a test that called
/// `init_test_free_pool`). Worker-only APIs that reach into the per-thread
/// memory context use this to fail with a clear message instead of letting
/// [`memory_ctx`] dereference the null context pointer.
pub fn has_memory_context() -> bool {
    MEMORY_CTX_PTR.with(|p| !p.get().is_null())
}

/// Whether a node is short of reclaimable slots, as the last worker to count
/// them found it. In a pool with a spill directory every worker of the node
/// reads this on every loop iteration, so it has a cache line to itself and
/// is written only when the answer changes: unlike the counts it is derived
/// from, which every buffer pop, push and cache bind writes, it stays in
/// every reader's cache. In a pool without one nothing counts the slots, and
/// this is neither written nor read.
#[repr(align(128))]
struct NodeSpillPressure {
    is_short_of_reclaimable_slots: AtomicBool,
}

/// The in-flight write count of each spilled buffer that one spill write
/// carries rows of.
type BuffersWritesInFlight = Vec<Arc<AtomicUsize>>;

pub struct MemoryContextFactory {
    ring: Arc<Ring>,
    caches: CacheSetup,
    clock: Arc<Clock>,
    dirty_pool_factory: PoolFactory,
    zeroed_pool_factory: PoolFactory,

    buffers_spilled: Arc<[AtomicUsize]>,
    spill_pressure: Arc<[NodeSpillPressure]>,
    layout: RingLayout,
    node: usize,
}

impl MemoryContextFactory {
    /// Build one factory per worker (global worker order) over a single shared
    /// ring, the caches, and the clock, partitioned per `layout`: each worker
    /// prefaults, acquires, and evicts only its own node's slot region, so its
    /// memory placement stays node-local, while all cached data remains
    /// readable (and releasable) from every worker.
    pub fn create_for_layout(layout: RingLayout) -> Vec<Self> {
        assert!(
            MEMORY_CTX_PTR.get().is_null(),
            "another memory context is already active!"
        );

        let buffers = layout.total_slots();
        let topology = layout.topology();
        let ring = Arc::new(Ring::new(buffers).unwrap());
        let caches = CacheSetup::standard(buffers);
        let clock = Arc::new(Clock::with_regions(
            topology.node_count,
            buffers / topology.node_count,
            caches.caches.policies(),
        ));
        let zeroed_pool_factories = PoolFactory::create_for_layout(layout);
        let dirty_pool_factories = PoolFactory::create_for_layout(layout);
        let buffers_spilled: Arc<[AtomicUsize]> = (0..topology.node_count)
            .map(|_| AtomicUsize::new(0))
            .collect();
        let spill_pressure: Arc<[NodeSpillPressure]> = (0..topology.node_count)
            .map(|_| NodeSpillPressure {
                is_short_of_reclaimable_slots: AtomicBool::new(false),
            })
            .collect();

        zeroed_pool_factories
            .into_iter()
            .zip(dirty_pool_factories)
            .enumerate()
            .map(|(worker, (zeroed_pool_factory, dirty_pool_factory))| Self {
                ring: ring.clone(),
                caches: caches.clone(),
                clock: clock.clone(),
                dirty_pool_factory,
                zeroed_pool_factory,
                buffers_spilled: buffers_spilled.clone(),
                spill_pressure: spill_pressure.clone(),
                layout,
                node: topology.node_of_worker(worker),
            })
            .collect()
    }

    /// Single-node convenience over [`create_for_layout`](Self::create_for_layout):
    /// `count` workers sharing a `buffers`-slot ring with no NUMA split.
    pub fn create_many(count: usize, buffers: usize) -> Vec<Self> {
        Self::create_for_layout(RingLayout::single_node(count, buffers))
    }

    pub fn create_memory_ctx(self) -> MemoryContext {
        let fill_cursors = self
            .caches
            .caches
            .ids()
            .map(|_| UnsafeCell::new(FillCursor::empty()))
            .collect();
        MemoryContext {
            ring: self.ring,
            caches: self.caches.caches,
            page_caches: self.caches.pages,
            clock: self.clock,
            dirty_pool: self.dirty_pool_factory.create_pool(),
            zeroed_pool: self.zeroed_pool_factory.create_pool(),
            fill_cursors,
            layout: self.layout,
            evictors: RefCell::new(HashMap::new()),
            buffers_spilled: self.buffers_spilled,
            own_buffers_spilled: Cell::new(0),
            spill_pressure: self.spill_pressure,
            io_requester: OnceCell::new(),
            spill_writes_in_flight: RefCell::new(HashMap::new()),
            node: self.node,
            took_buffer_since_pressure_check: Cell::new(false),
        }
    }
}

pub struct MemoryContext {
    ring: Arc<Ring>,
    /// Every cache, as the evictor sees them.
    caches: Arc<Caches>,
    /// The same caches, typed, for the readers that fill and consult them.
    page_caches: PageCaches,
    /// The CLOCK eviction policy over all ring slots, shared by every cache.
    clock: Arc<Clock>,
    dirty_pool: FreePool,
    zeroed_pool: FreePool,
    /// This worker's bump cursors for packing entries into shared fill slots,
    /// one per cache, indexed by [`CacheId`]. Per-thread, so interior-mutable
    /// without a lock - the `&'static MemoryContext` is really thread-local, so
    /// there is never a second accessor. Mirrors the `UnsafeCell` discipline
    /// the caches use for their per-slot metadata.
    fill_cursors: Box<[UnsafeCell<FillCursor>]>,
    /// How ring slots map to nodes and home workers (shared by all contexts).
    layout: RingLayout,
    /// Each dataflow's operators that keep ring slots outside the caches and can
    /// give them back, keyed by dataflow id. Consulted only after the caches
    /// have swept their whole region without finding a slot nobody holds. A
    /// `RefCell` because the context is reached through a shared reference; it
    /// is per worker thread, so the borrow never contends.
    evictors: RefCell<HashMap<Identifier, DataflowEvict>>,
    /// Per node, the slots its operators have handed over for spilling and
    /// still hold, whether the write is in flight or has landed. Shared by the
    /// node's workers, so a slot one worker is already spilling is not asked
    /// for again by another.
    buffers_spilled: Arc<[AtomicUsize]>,
    /// The slots this worker itself has handed over for spilling and still
    /// holds. Only this worker's eviction can take those back, so they are
    /// what tells one worker's need to spill apart from its siblings'.
    own_buffers_spilled: Cell<usize>,
    /// Per node, whether the last worker to count the node's reclaimable
    /// slots found it short of them. A worker that took no buffer reads only
    /// this to learn that its siblings have run the node short.
    spill_pressure: Arc<[NodeSpillPressure]>,
    /// The worker's IO requester, attached once the worker has built it, so
    /// that eviction can wait for spill writes to land when nothing else can
    /// be reclaimed. Shared with the worker, which drives it.
    io_requester: OnceCell<Rc<RefCell<IORequester>>>,
    /// Every spill write in flight on this worker, pointing at the in-flight
    /// count of each spilled buffer it carries rows of. A count that reaches
    /// zero tells the buffer's evictor the whole buffer is on disk. Keyed by
    /// the issuing dataflow as well as the write id, because write ids are
    /// drawn per dataflow and two dataflows spilling on one worker repeat them.
    spill_writes_in_flight: RefCell<HashMap<(Identifier, WriteRequestId), BuffersWritesInFlight>>,
    /// This worker's NUMA node, i.e. the ring region it allocates and evicts from.
    node: usize,
    /// Whether this worker took a buffer since it last counted the node's
    /// reclaimable slots. The node runs short only by a buffer being taken,
    /// so the workers that take them are the ones that count, and they tell
    /// the rest through `spill_pressure`.
    took_buffer_since_pressure_check: Cell<bool>,
}

impl MemoryContext {
    pub fn prefault_buffers(&self) {
        // Fault in the slots this worker is home to, so under first-touch their
        // pages land on this worker's (already pinned) core's NUMA node. Homes
        // partition each node's region across that node's workers, so every
        // slot is faulted exactly once, by a worker on its own node.
        // We forget the WriteBuffer to avoid the Drop impl pushing to the dirty pool,
        // then manually release the slot and push to the zeroed pool.
        let worker = WORKER_IDX.get();
        for i in self.layout.node_slots(self.node) {
            if self.layout.home_worker(i) != worker {
                continue;
            }
            let mut write = memory_ctx().ring().try_write(i).unwrap();
            for j in (0..BUFFER_SIZE).step_by(4096) {
                write.as_mut()[j] = 1u8;
            }
            write.zero_out();
        }
    }

    /// Read what every block of the ring currently holds, in ring order.
    ///
    /// The whole ring, from whichever worker asks: a worker may only *acquire*
    /// and evict blocks in its own node's region, but reading crosses regions
    /// freely, and reading is all this does.
    pub fn read_all_blocks(&self) -> Vec<MemoryBlockStatus> {
        (0..self.ring.len())
            .map(|slot| {
                read_block_status(&self.ring, &self.clock, &self.caches, &self.layout, slot)
            })
            .collect()
    }

    pub fn caches(&self) -> &Caches {
        self.caches.as_ref()
    }

    pub fn page_caches(&self) -> &PageCaches {
        &self.page_caches
    }

    pub fn clock(&self) -> &Clock {
        self.clock.as_ref()
    }

    pub fn ring(&self) -> &Ring {
        self.ring.as_ref()
    }

    /// This worker's fill cursor for `cache`: the bump pointer its entries are
    /// packed through. Sound because the context is per-thread, so the
    /// returned `&mut` never aliases another accessor on the same thread; a
    /// caller must still not hold two of them for the same cache at once.
    #[allow(clippy::mut_from_ref)] // interior mutability; per-thread, single accessor
    pub(crate) fn fill_cursor(&self, cache: CacheId) -> &mut FillCursor {
        unsafe { &mut *self.fill_cursors[cache.index()].get() }
    }

    /// Return a buffer index to the pool.
    ///
    /// Routes the index to the home worker's zeroed or dirty pool. If the current thread
    /// *is* the home worker, the push goes directly into the local deque. Otherwise it is
    /// placed into the home worker's injector so the owner retrieves it on its next pop.
    pub fn push_free_idx(&self, idx: usize, zeroed: bool) {
        if zeroed {
            self.zeroed_pool.push(idx)
        } else {
            self.dirty_pool.push(idx)
        }
    }

    /// Pop a free buffer index for the current worker thread.
    ///
    /// When `prefer_zeroed` is true, tries the zeroed pool first then dirty.
    /// When false, tries dirty first then zeroed.
    pub fn pop_free_idx(&self, prefer_zeroed: bool) -> Option<usize> {
        let (first, second) = if prefer_zeroed {
            (&self.zeroed_pool, &self.dirty_pool)
        } else {
            (&self.dirty_pool, &self.zeroed_pool)
        };
        first.pop(true).or_else(|| second.pop(true))
    }

    /// Pop a dirty buffer from this worker's local deque only (no stealing).
    ///
    /// Used by background buffer-clean passes that should not pull buffers off
    /// peer workers - see [`crate::worker::Worker`]'s `clear_dirty_buffer_or_park`.
    ///
    /// A popped slot can be transiently locked (e.g. by an evictor probing it
    /// before backing off). The contender never re-pools an index it did not
    /// pop, so the index is pushed back - dropping it would leak the slot from
    /// every pool - and `None` is returned; the next pass retries.
    pub fn pop_dirty_buffer(&self) -> Option<WriteBuffer> {
        let idx = self.dirty_pool.pop(false)?;
        let buffer = memory_ctx().ring().try_write(idx);
        if buffer.is_none() {
            self.push_free_idx(idx, false);
        }
        buffer
    }

    /// Drain this worker's dirty buffers, zeroing each and returning it to the
    /// zeroed pool. Equivalent to the engine's idle-time dirty-buffer cleanup,
    /// but run eagerly to completion rather than opportunistically.
    ///
    /// Must run on the worker thread that owns this context (its `pop`/`push`
    /// touch the per-worker free pool). Returns the number of buffers zeroed.
    /// Intended for benchmarks: re-zeroing dirtied buffers between iterations is
    /// allocation/setup work, so doing it eagerly (outside the timed region)
    /// keeps the next query's hash-table allocation from re-zeroing inline.
    pub fn zero_dirty_buffers(&self) -> usize {
        let mut zeroed = 0;
        while let Some(buf) = self.pop_dirty_buffer() {
            buf.zero_out();
            zeroed += 1;
        }
        zeroed
    }

    /// Acquire a [`WriteBuffer`] from the free pool, falling back to eviction.
    ///
    /// When `prefer_zeroed` is true, tries the zeroed pool first - use this when the caller
    /// needs zeroed memory so we can skip a memset. When false, tries the
    /// dirty pool first - use this when the caller will overwrite the buffer entirely
    /// (e.g. decompression, I/O reads) to preserve zeroed buffers for those who need them.
    /// Either way, the other pool is used as a fallback if the preferred one is empty.
    ///
    /// If the popped index's ring slot is contended, retries with a fresh index rather
    /// than evicting.
    pub fn get_write_buffer(&self, prefer_zeroed: bool) -> WriteBuffer {
        // Held out of the pools during the loop so the retry pops a *different*
        // index instead of spinning on the locked one at the LIFO top.
        let mut contended = ContendedIndices::new(self);
        loop {
            if let Some(idx) = self.pop_free_idx(prefer_zeroed) {
                // A slot listed in the free pool must genuinely be free, i.e. owned
                // by no cache. If a cache still owns it, it leaked into the pool
                // while live and is about to be handed out under two owners.
                debug_assert_eq!(
                    self.clock.owner(idx),
                    None,
                    "pooled slot {idx} still owned by a cache",
                );
                match memory_ctx().ring().try_write(idx) {
                    Some(buffer) => {
                        self.note_buffer_taken();
                        return buffer;
                    }
                    None => {
                        // Try again with a fresh idx, we don't want to start
                        // evicting yet.
                        contended.set_aside(idx);
                        continue;
                    }
                }
            }

            // Pool empty: one last attempt at the set-aside indices (their
            // contenders hold slots only transiently), then hand the rest back
            // so peers can pop them while this worker is busy evicting.
            if let Some(buffer) = contended.pop_writable() {
                self.note_buffer_taken();
                return buffer;
            }
            contended.release();

            // Evict via the shared clock.
            let evicted = self.evict();
            self.note_buffer_taken();
            return evicted;
        }
    }

    fn note_buffer_taken(&self) {
        self.took_buffer_since_pressure_check.set(true);
    }

    /// Whether this worker should count its node's reclaimable slots now
    /// (see [`slots_to_spill`](Self::slots_to_spill)): it took a buffer since
    /// the last call, so the node may have just run short, or a worker of the
    /// node already found it short. A worker that took nothing still has to
    /// count while the node is short, because the rows it holds may be the
    /// only ones on the node that can be spilled. Otherwise the count is
    /// skipped, and with it the reads of lines its siblings keep writing.
    pub fn is_pressure_check_due(&self) -> bool {
        let took_buffer = self.took_buffer_since_pressure_check.replace(false);
        took_buffer
            || self.spill_pressure[self.node]
                .is_short_of_reclaimable_slots
                .load(Ordering::Relaxed)
    }

    /// Tell the node's workers whether the node is short of reclaimable
    /// slots, writing only when that changed.
    fn publish_spill_pressure(&self, is_short: bool) {
        let published = &self.spill_pressure[self.node].is_short_of_reclaimable_slots;
        if published.load(Ordering::Relaxed) != is_short {
            published.store(is_short, Ordering::Relaxed);
        }
    }

    /// Evict a ring slot via the shared CLOCK and return it writable.
    ///
    /// Every cache has a hand of its own over this worker's region, and each
    /// family with a slot to give moves one of them a step at a time: that of
    /// the member holding more than its target share of the family's slots
    /// (see [`FamilySpec::choose_member_to_evict`]). A slot that hand passes
    /// spends a life or, at zero, is reclaimed from its cache, which then
    /// reinforces its siblings over what it dropped. The families' hands move
    /// in step, so between families a slot goes sooner the fewer lives it
    /// holds.
    ///
    /// A member with no slot to give in this region, because it owns none or
    /// because readers hold every slot it owns, is not swept at all, and when
    /// no cache has one the registered [`Evict`](crate::memory::evict::Evict)
    /// operators are asked at once. A member swept without giving a slot
    /// hands over to the next member of its family; once every member has
    /// been swept that way, the operators are asked before the caches are
    /// swept again.
    ///
    /// [`FamilySpec::choose_member_to_evict`]: crate::memory::cache::FamilySpec::choose_member_to_evict
    pub(crate) fn evict(&self) -> WriteBuffer {
        let ring_len = self.layout.node_slots(self.node).len() as u64;
        let max_lives = self.clock.max_lives() as u64;
        // A full drain is the most revolutions a hand needs to reach any
        // victim of its cache: one per life a slot can hold, plus the
        // revolution that finds it at zero. A hand that swept that much
        // without a slot being reclaimed has nothing to give, so the next
        // member of its family gets a full drain of its own, regardless of
        // share, and the members keep taking turns in whole drains. Anything
        // shorter for the next hand cannot reach a slot that still holds
        // lives.
        //
        // A revolution that aged nothing cannot produce a victim by going on:
        // the caches being swept have no lives left to drain in this region,
        // so their zero-life slots (all held by readers or writers, since the
        // revolution ended without a victim) will look the same next time
        // round. Such a revolution hands over at once instead of repeating
        // itself for the rest of the drain.
        let drain = (max_lives + 1) * ring_len;
        let members_to_try = self.caches.count_members_of_largest_family();
        // Livelock guard. When a query's working set exceeds the ring, every
        // worker spins here finding nothing evictable (near-100% CPU, ~0 forward
        // progress, and the loop has no cancellation point). A healthy hand finds
        // a victim within the lives ceiling's worth of revolutions (a counter
        // reaches zero in at most that many), so more than a full drain of
        // every member of a family in turn is abnormal: warn and sleep to let
        // peer workers / ingest release slots; if it still finds nothing, the
        // cache is genuinely exhausted, so panic to abort the offending query.
        // The sweep covers only this worker's node region, so every bound is
        // region-sized.
        let warn_at = members_to_try as u64 * drain;
        let mut iterations: u64 = 0;
        let mut panic_at: Option<u64> = None;
        let mut rounds_without_a_slot: u64 = 0;
        let mut aged_this_revolution: u64 = 0;
        // How many times a sweep ended with nothing to give. Once that has
        // happened for every member of a family, the caches hold nothing
        // unpinned right now, and the operators keeping slots outside the
        // caches are the only remaining source.
        let mut sweeps_given_up: usize = 0;
        loop {
            iterations += 1;
            if panic_at.is_none() && iterations == warn_at {
                tracing::warn!(iterations, "evict: no memory left to evict, sleeping");
                std::thread::sleep(std::time::Duration::from_secs(1));
                panic_at = Some(iterations + ring_len);
            } else if panic_at.is_some_and(|limit| iterations >= limit) {
                panic!(
                    "evict: still no evictable memory after sleeping ({iterations} \
                     iterations) — aborting query (cache exhausted by an oversized \
                     working set)"
                );
            }

            // One round: a step of one hand for each family with a slot to
            // give. A hand can only take a slot its cache owns and no reader
            // holds. The counts of those are shared with peer workers, so a
            // slot may be bound or let go of between reading them and
            // sweeping. Passing over a member that has none is still safe: at
            // worst such a slot waits until the next time round to be found.
            let mut moved_a_hand = false;
            for family in self.caches.families() {
                let Some(cache) = family.choose_member_to_evict(
                    |member| self.clock.count_owned_in_region(member, self.node),
                    |member| self.clock.count_unpinned_in_region(member, self.node) > 0,
                    sweeps_given_up,
                ) else {
                    continue;
                };
                moved_a_hand = true;
                let step = self.clock.sweep(cache, self.node);
                if let Sweep::Candidate(slot) = step
                    && let Some(write_buffer) = self.caches.reclaim(cache, slot)
                {
                    return write_buffer;
                }
                if step == Sweep::Aged {
                    aged_this_revolution += 1;
                }
            }
            if !moved_a_hand {
                // Whatever is cached in this region is held by readers, so the
                // caches have nothing to give back and only the operators can.
                if let Some(buffer) = self.evict_from_dataflow_operators() {
                    return buffer;
                }
                if self.has_spill_writes_in_flight() {
                    self.wait_for_a_spill_write_to_land();
                    iterations = 0;
                }
                continue;
            }

            rounds_without_a_slot += 1;
            if rounds_without_a_slot.is_multiple_of(ring_len) {
                let drained = rounds_without_a_slot.is_multiple_of(drain);
                let nothing_left_to_drain = aged_this_revolution == 0;
                if drained || nothing_left_to_drain {
                    sweeps_given_up += 1;
                    if sweeps_given_up >= members_to_try {
                        if let Some(buffer) = self.evict_from_dataflow_operators() {
                            return buffer;
                        }
                        if self.has_spill_writes_in_flight() {
                            self.wait_for_a_spill_write_to_land();
                            iterations = 0;
                        }
                    }
                }
                aged_this_revolution = 0;
            }
        }
    }

    /// How many slots this worker's operators should spill now: this worker's
    /// share of the node's shortfall of reclaimable slots below the target (see
    /// [`RECLAIMABLE_SLOTS_TARGET_PERCENT`]). The shortfall is split evenly
    /// across the node's workers, rounded up, because every worker on the node
    /// sees the same shortfall and would otherwise each ask for all of it.
    ///
    /// Reclaimable means free, owned by a cache, or spilled. Cached slots count
    /// because the sweep can take them back at will, and spilled slots count
    /// because eviction can take them once their write lands, so the slots
    /// that are none of these are the ones operators still hold.
    ///
    /// A spilled slot can be taken back only by the worker that spilled it, so
    /// this worker's own spilled slots are added on top of the node's. A worker
    /// that holds many sees no shortfall and one that holds few sees it first,
    /// which sends each new spill to the worker with the fewest.
    ///
    /// Nothing is asked for until the reclaimable slots are below where
    /// spilling starts (see [`get_reclaimable_slots_start_percent`]), which may
    /// be lower than the target the shortfall is then measured against. The counts
    /// are read at slightly different instants while peers allocate and
    /// release, so the result is approximate by a few slots, which a watermark
    /// tolerates.
    ///
    /// Whether the node as a whole is short is published to its other workers
    /// (see [`is_pressure_check_due`](Self::is_pressure_check_due)). That
    /// leaves out this worker's own spilled slots, which only it can count.
    pub fn slots_to_spill(&self) -> usize {
        let free = self.dirty_pool.free_slots() + self.zeroed_pool.free_slots();
        let cached = self.clock.count_cached_in_region(self.node);
        let spilled = self.buffers_spilled[self.node].load(Ordering::Relaxed);
        let slots = self.layout.node_slots(self.node).len();
        let start = slots * get_reclaimable_slots_start_percent() / 100;
        let target = slots * get_reclaimable_slots_target_percent() / 100;
        let node_reclaimable = free + cached + spilled;
        self.publish_spill_pressure(node_reclaimable < start);
        let reclaimable = node_reclaimable + self.own_buffers_spilled.get();
        if reclaimable >= start {
            return 0;
        }
        (target - reclaimable).div_ceil(self.layout.topology().workers_per_node)
    }

    /// Count `count` more of this node's slots as spilled, so that
    /// [`slots_to_spill`](Self::slots_to_spill) stops asking for them.
    pub fn record_buffers_spilled(&self, count: usize) {
        self.buffers_spilled[self.node].fetch_add(count, Ordering::Relaxed);
        self.own_buffers_spilled
            .set(self.own_buffers_spilled.get() + count);
    }

    /// Note that `count` spilled slots have been released, so they are no
    /// longer counted as spilled: once free, the pools count them instead.
    pub fn record_spilled_buffers_released(&self, count: usize) {
        let before = self.buffers_spilled[self.node].fetch_sub(count, Ordering::Relaxed);
        assert!(
            before >= count,
            "released more spilled buffers than were spilled"
        );
        self.own_buffers_spilled
            .set(self.own_buffers_spilled.get() - count);
    }

    /// Make `evictors` reachable from this worker's eviction for as long as the
    /// dataflow runs; its `DataFlow` unregisters them when it drops.
    pub fn register_dataflow_evictors(&self, dataflow_id: Identifier, evictors: DataflowEvict) {
        let replaced = self.evictors.borrow_mut().insert(dataflow_id, evictors);
        assert!(
            replaced.is_none(),
            "dataflow {dataflow_id} registered its evictors twice"
        );
    }

    /// Forget the evictor of one operator of `dataflow_id`, which has finished
    /// and released whatever it kept.
    pub fn remove_operator_evictor(&self, dataflow_id: Identifier, operator_idx: usize) {
        self.evictors
            .borrow_mut()
            .get_mut(&dataflow_id)
            .expect("an operator's evictor is removed from a dataflow that registered evictors")
            .remove_operator(operator_idx);
    }

    /// Forget the evictors registered under `dataflow_id`.
    pub fn unregister_dataflow_evictors(&self, dataflow_id: Identifier) {
        let removed = self.evictors.borrow_mut().remove(&dataflow_id);
        assert!(
            removed.is_some(),
            "dataflow {dataflow_id} never registered evictors"
        );
        // A dataflow that failed takes its queued IO with it, so a spill write
        // of its that was never submitted never lands. Left here, it would
        // keep eviction waiting for a write that is not coming.
        self.spill_writes_in_flight
            .borrow_mut()
            .retain(|(dataflow, _), _| *dataflow != dataflow_id);
    }

    /// Share the worker's IO requester with this context, once, so that
    /// eviction can wait on it (see [`evict`](Self::evict)).
    pub fn attach_io_requester(&self, requester: Rc<RefCell<IORequester>>) {
        self.io_requester
            .set(requester)
            .unwrap_or_else(|_| panic!("the worker attached its IO requester twice"));
    }

    /// Note a spill write issued by `dataflow` that carries rows of the
    /// buffers whose in-flight counts are given, so that when it lands each
    /// of them is counted down.
    pub fn record_spill_write_issued(
        &self,
        dataflow: Identifier,
        id: WriteRequestId,
        buffers_writes_in_flight: Vec<Arc<AtomicUsize>>,
    ) {
        let replaced = self
            .spill_writes_in_flight
            .borrow_mut()
            .insert((dataflow, id), buffers_writes_in_flight);
        assert!(
            replaced.is_none(),
            "spill write {id:?} of dataflow {dataflow} issued twice"
        );
    }

    pub fn has_spill_writes_in_flight(&self) -> bool {
        !self.spill_writes_in_flight.borrow().is_empty()
    }

    /// Take the spill writes that `requester` has collected as complete and
    /// count each off its buffer. The worker runs this on every pass over its
    /// completions, and eviction runs it after waiting for one.
    pub fn complete_landed_spill_writes(&self, requester: &mut IORequester) {
        let mut spill_writes_in_flight = self.spill_writes_in_flight.borrow_mut();
        let landed = requester.extract_completions_if(|completion| {
            matches!(completion, Ok(Completion::FsWrite(write)) if write.request.id.is_spill())
        });
        for completion in landed {
            let Ok(Completion::FsWrite(write)) = completion else {
                unreachable!("extracted only spill writes");
            };
            // A write of a dataflow that has since failed is no longer
            // recorded (see `unregister_dataflow_evictors`); its buffer went
            // with the dataflow, so there is nothing left to count it off.
            if let Some(buffers_writes_in_flight) =
                spill_writes_in_flight.remove(&(write.data_flow_id, write.request.id))
            {
                for writes_in_flight in buffers_writes_in_flight {
                    writes_in_flight.fetch_sub(1, Ordering::Release);
                }
            }
        }
    }

    /// Block until one of this worker's spill writes may have landed, and
    /// count the ones that did off their buffers. Every slot this worker could
    /// reclaim is held by a spilled buffer whose write has not landed yet, so
    /// the only way forward is to let the writes finish. Runs inside an
    /// operator's run, so it leaves every other completion for the worker's
    /// next pass.
    ///
    /// A write held back because its device's hardware queue is full is not on
    /// the ring, and the worker is told through its waker when a slot in that
    /// queue frees. The waker reaches only a wait that was published to it, so
    /// the wait is published the way the worker loop publishes its own: a
    /// worker whose writes are all held back would otherwise never wake.
    fn wait_for_a_spill_write_to_land(&self) {
        let requester = self
            .io_requester
            .get()
            .expect("spill writes are in flight only on a worker with an attached IO requester");
        let mut requester = requester
            .try_borrow_mut()
            .expect("eviction does not run from inside the IO requester's own calls");
        let waker = crate::waker::worker_waker();
        let node_local_idx = self
            .layout
            .topology()
            .local_index_of_worker(WORKER_IDX.get());
        // Read before the hardware queues are tried, so a slot that frees
        // after that makes the wait return at once instead of being missed.
        let last_seen = waker.ring_wake_count();
        // Take what the ring already holds, which also puts held-back writes
        // on the device as far as its queue allows.
        if self.count_off_landed_spill_writes(&mut requester) {
            return;
        }
        if waker.begin_ring_wait(node_local_idx, last_seen) {
            let waited = requester.wait();
            waker.end_ring_wait(node_local_idx);
            waited.expect("waiting on the IO ring while evicting");
        }
        self.count_off_landed_spill_writes(&mut requester);
    }

    /// Collect the ring's completions and count the spill writes among them
    /// off their buffers. Returns whether any spill write had landed.
    fn count_off_landed_spill_writes(&self, requester: &mut IORequester) -> bool {
        requester
            .collect_completions()
            .expect("collecting IO completions while evicting");
        let in_flight_before = self.spill_writes_in_flight.borrow().len();
        self.complete_landed_spill_writes(requester);
        self.spill_writes_in_flight.borrow().len() < in_flight_before
    }

    /// Ask each registered dataflow's operators for a slot they can give up and
    /// return the first one offered, or `None` when none has a slot to spare
    /// right now. The borrow is held across the calls, which is fine because an
    /// operator's `evict` only releases memory it already holds and never
    /// reaches back into this registry.
    fn evict_from_dataflow_operators(&self) -> Option<WriteBuffer> {
        self.evictors
            .borrow_mut()
            .values_mut()
            .find_map(DataflowEvict::evict)
    }
}

/// Free-pool indices set aside because their ring slot was transiently locked
/// (e.g. an evictor probing the slot before backing off). The contender never
/// re-pools an index it did not pop, so the popper must return every set-aside
/// index or the slot leaks from every pool. Dropping the holder returns them,
/// keeping the invariant across early returns and unwinds (eviction panics).
struct ContendedIndices<'a> {
    ctx: &'a MemoryContext,
    indices: Vec<usize>,
}

impl<'a> ContendedIndices<'a> {
    fn new(ctx: &'a MemoryContext) -> Self {
        Self {
            ctx,
            indices: Vec::new(),
        }
    }

    fn set_aside(&mut self, idx: usize) {
        self.indices.push(idx);
    }

    /// One more `try_write` attempt at each set-aside index, in case its
    /// contender already backed off. A single pass per index, so a long-held
    /// slot cannot spin the caller.
    fn pop_writable(&mut self) -> Option<WriteBuffer> {
        for i in 0..self.indices.len() {
            if let Some(buffer) = self.ctx.ring().try_write(self.indices[i]) {
                self.indices.swap_remove(i);
                return Some(buffer);
            }
        }
        None
    }

    /// Return every remaining index to its pool now rather than at drop, so
    /// peers can pop the slots while the holder is still alive.
    fn release(&mut self) {
        for idx in self.indices.drain(..) {
            self.ctx
                .push_free_idx(idx, self.ctx.ring().slot_zeroed(idx));
        }
    }
}

impl Drop for ContendedIndices<'_> {
    fn drop(&mut self) {
        self.release();
    }
}

/// Install a fresh single-worker [`MemoryContext`] on the calling test
/// thread, with the first `dirty_count` ring slots pre-loaded into the dirty
/// pool so `memory_ctx().get_write_buffer(false)` can hand them out.
///
/// Each test thread gets its own context (and its own [`Ring`]), so tests
/// don't share state and don't need a serializing lock.
#[cfg(any(test, feature = "test-util"))]
pub fn init_test_free_pool(dirty_count: usize) {
    WORKER_IDX.set(0);
    crate::worker::NUM_WORKERS.set(1);
    crate::waker::install_test_worker_waker();
    let factory = MemoryContextFactory::create_many(1, 128).pop().unwrap();
    init_memory_context(factory.create_memory_ctx());
    for i in 0..dirty_count {
        memory_ctx().push_free_idx(i, false);
    }
}

#[cfg(test)]
mod tests {
    //! Tests for [`MemoryContext`] - the zeroed/dirty pair, fallback rules,
    //! and the buffer accessors. Single-pool routing lives in
    //! [`crate::memory::free_pool`]'s tests; here we only exercise behaviour
    //! that comes from owning *both* pools (plus the ring and the caches).
    //!
    //! Each test installs a fresh [`MemoryContext`] on its own thread via
    //! [`init_test_free_pool`], so there's no shared state and no lock.

    use super::*;
    use crate::memory::evict::{Evict, HandOverEvictor};
    use crate::memory::{ReadBuffer, RingCache};
    use std::rc::Rc;

    #[test]
    fn evict_turns_to_dataflow_operators_once_the_caches_have_nothing() {
        init_test_free_pool(1);
        let held = memory_ctx().get_write_buffer(false);
        let held_slot = held.slot_idx;
        let log = Rc::new(RefCell::new(Vec::new()));
        let operator: Box<dyn Evict> = Box::new(HandOverEvictor {
            label: "group",
            log: log.clone(),
            buffer: RefCell::new(Some(held)),
        });
        memory_ctx().register_dataflow_evictors(7, DataflowEvict::new(vec![Some(operator)], &[]));

        let evicted = memory_ctx().evict();

        assert_eq!(evicted.slot_idx, held_slot);
        assert_eq!(*log.borrow(), ["group"]);
        memory_ctx().unregister_dataflow_evictors(7);
    }

    #[test]
    fn evict_asks_the_operators_without_sweeping_when_readers_hold_every_cached_slot() {
        init_test_free_pool(2);
        let cached = memory_ctx().get_write_buffer(false);
        let decompressed = memory_ctx().page_caches().decompressed.id();
        memory_ctx().clock().bind(cached.slot_idx, decompressed);
        let _reader = ReadBuffer::from(cached);
        let held = memory_ctx().get_write_buffer(false);
        let held_slot = held.slot_idx;
        let operator: Box<dyn Evict> = Box::new(HandOverEvictor {
            label: "group",
            log: Rc::new(RefCell::new(Vec::new())),
            buffer: RefCell::new(Some(held)),
        });
        memory_ctx().register_dataflow_evictors(7, DataflowEvict::new(vec![Some(operator)], &[]));

        let evicted = memory_ctx().evict();

        assert_eq!(evicted.slot_idx, held_slot);
        assert_eq!(memory_ctx().clock().hand_position(decompressed, 0), 0);
        memory_ctx().unregister_dataflow_evictors(7);
    }

    #[test]
    fn a_cached_slot_can_be_taken_again_once_its_last_reader_lets_go() {
        init_test_free_pool(1);
        let cached = memory_ctx().get_write_buffer(false);
        let cached_slot = cached.slot_idx;
        let decompressed = memory_ctx().page_caches().decompressed.id();
        memory_ctx().clock().bind(cached_slot, decompressed);
        let first_reader = ReadBuffer::from(cached);
        let second_reader = memory_ctx().ring().try_read(cached_slot).unwrap();

        let with_two_readers = memory_ctx()
            .clock()
            .count_unpinned_in_region(decompressed, 0);
        drop(first_reader);
        let with_one_reader = memory_ctx()
            .clock()
            .count_unpinned_in_region(decompressed, 0);
        drop(second_reader);
        let with_no_reader = memory_ctx()
            .clock()
            .count_unpinned_in_region(decompressed, 0);

        assert_eq!(
            (with_two_readers, with_one_reader, with_no_reader),
            (0, 0, 1)
        );
    }

    #[test]
    fn cached_slots_count_as_reclaimable_for_spilling() {
        // 128 slots, so spilling starts under 12 reclaimable; 5 free plus 10
        // cached.
        init_test_free_pool(5);
        let compressed = memory_ctx().page_caches().compressed.id();
        for idx in 5..15 {
            memory_ctx().clock().bind(idx, compressed);
        }

        let shortfall = memory_ctx().slots_to_spill();

        assert_eq!(shortfall, 0);
    }

    #[test]
    fn a_pressure_check_is_due_once_after_a_buffer_is_taken() {
        init_test_free_pool(64);

        let before_taking = memory_ctx().is_pressure_check_due();
        let buffer = memory_ctx().get_write_buffer(false);
        let after_taking = memory_ctx().is_pressure_check_due();
        let without_taking_again = memory_ctx().is_pressure_check_due();
        drop(buffer);

        assert_eq!(
            [before_taking, after_taking, without_taking_again],
            [false, true, false]
        );
    }

    #[test]
    fn a_pressure_check_stays_due_while_the_node_is_short_of_reclaimable_slots() {
        // 128 slots, so spilling starts under 12 reclaimable; 30 are free
        // until 20 are taken.
        init_test_free_pool(30);
        let held: Vec<_> = (0..20)
            .map(|_| memory_ctx().get_write_buffer(false))
            .collect();
        memory_ctx().is_pressure_check_due();

        memory_ctx().slots_to_spill();
        let while_short = memory_ctx().is_pressure_check_due();
        drop(held);
        memory_ctx().slots_to_spill();
        let after_release = memory_ctx().is_pressure_check_due();

        assert_eq!([while_short, after_release], [true, false]);
    }

    #[test]
    fn a_worker_that_took_no_buffer_checks_once_a_sibling_finds_the_node_short() {
        // Two workers on one node of 128 slots, none of them free.
        WORKER_IDX.set(0);
        crate::worker::NUM_WORKERS.set(2);
        crate::waker::install_test_worker_waker();
        let mut factories = MemoryContextFactory::create_many(2, 128);
        let (ask_sibling_to_count, sibling_asked) = std::sync::mpsc::channel::<()>();
        let (report_built, sibling_built) = std::sync::mpsc::channel::<()>();
        let sibling_factory = factories.pop().unwrap();
        let sibling = std::thread::spawn(move || {
            WORKER_IDX.set(1);
            let sibling_ctx = sibling_factory.create_memory_ctx();
            report_built.send(()).unwrap();
            sibling_asked.recv().unwrap();
            sibling_ctx.slots_to_spill();
        });
        init_memory_context(factories.pop().unwrap().create_memory_ctx());
        sibling_built.recv().unwrap();

        let before_sibling_counted = memory_ctx().is_pressure_check_due();
        ask_sibling_to_count.send(()).unwrap();
        sibling.join().unwrap();
        let after_sibling_counted = memory_ctx().is_pressure_check_due();

        assert_eq!(
            [before_sibling_counted, after_sibling_counted],
            [false, true]
        );
    }

    #[test]
    fn slots_to_spill_is_the_shortfall_below_the_reclaimable_target_less_what_is_spilled() {
        // 128 slots, so spilling starts under 12 reclaimable and refills to
        // 25; seed 10 free, nothing cached.
        init_test_free_pool(10);

        let shortfall = memory_ctx().slots_to_spill();
        memory_ctx().record_buffers_spilled(shortfall);
        let while_spilled = memory_ctx().slots_to_spill();
        memory_ctx().record_spilled_buffers_released(shortfall);
        let after_release = memory_ctx().slots_to_spill();

        assert_eq!(shortfall, 15);
        assert_eq!(while_spilled, 0);
        assert_eq!(after_release, 15);
    }

    #[test]
    fn a_workers_own_spilled_slots_count_again_toward_its_target() {
        // 128 slots, so spilling starts under 12 reclaimable. 4 free and 5
        // spilled make 9; the worker's own 5 counted again make 14.
        init_test_free_pool(4);
        memory_ctx().record_buffers_spilled(5);

        let shortfall = memory_ctx().slots_to_spill();

        assert_eq!(shortfall, 0);
        memory_ctx().record_spilled_buffers_released(5);
    }

    #[test]
    fn slots_to_spill_is_this_workers_share_of_the_nodes_shortfall_rounded_up() {
        // Four workers on one node of 128 slots; 10 free leaves a shortfall of 15.
        WORKER_IDX.set(0);
        crate::worker::NUM_WORKERS.set(4);
        crate::waker::install_test_worker_waker();
        let mut factories = MemoryContextFactory::create_many(4, 128);
        // Building a context waits until every worker on the node has built one.
        let siblings: Vec<_> = factories
            .drain(1..)
            .enumerate()
            .map(|(i, factory)| {
                std::thread::spawn(move || {
                    WORKER_IDX.set(i + 1);
                    drop(factory.create_memory_ctx());
                })
            })
            .collect();
        init_memory_context(factories.pop().unwrap().create_memory_ctx());
        for sibling in siblings {
            sibling.join().unwrap();
        }
        for i in 0..10 {
            memory_ctx().push_free_idx(i, false);
        }

        let share = memory_ctx().slots_to_spill();

        assert_eq!(share, 4);
    }

    /// Install a context with no buffers seeded.
    fn fresh_ctx() {
        init_test_free_pool(0);
    }

    #[test]
    fn pop_returns_none_when_both_pools_empty() {
        // Setup
        fresh_ctx();

        // Execute
        let zeroed = memory_ctx().pop_free_idx(true);
        let dirty = memory_ctx().pop_free_idx(false);

        // Assert
        assert_eq!(zeroed, None);
        assert_eq!(dirty, None);
    }

    #[test]
    fn prefer_zeroed_picks_zeroed_over_dirty() {
        // Setup: a dirty idx and a zeroed idx, both routable to worker 0.
        fresh_ctx();
        memory_ctx().push_free_idx(0, false);
        memory_ctx().push_free_idx(1, true);

        // Execute
        let popped = memory_ctx().pop_free_idx(true);

        // Assert
        assert_eq!(popped, Some(1));
    }

    #[test]
    fn prefer_dirty_picks_dirty_over_zeroed() {
        // Setup
        fresh_ctx();
        memory_ctx().push_free_idx(0, true);
        memory_ctx().push_free_idx(1, false);

        // Execute
        let popped = memory_ctx().pop_free_idx(false);

        // Assert
        assert_eq!(popped, Some(1));
    }

    #[test]
    fn prefer_zeroed_falls_back_to_dirty_when_zeroed_is_empty() {
        // Setup
        fresh_ctx();
        memory_ctx().push_free_idx(0, false);

        // Execute
        let popped = memory_ctx().pop_free_idx(true);

        // Assert
        assert_eq!(popped, Some(0));
    }

    #[test]
    fn prefer_dirty_falls_back_to_zeroed_when_dirty_is_empty() {
        // Setup
        fresh_ctx();
        memory_ctx().push_free_idx(0, true);

        // Execute
        let popped = memory_ctx().pop_free_idx(false);

        // Assert
        assert_eq!(popped, Some(0));
    }

    #[test]
    fn pop_drains_both_pools_before_returning_none() {
        // Setup: one in each pool.
        fresh_ctx();
        memory_ctx().push_free_idx(0, false);
        memory_ctx().push_free_idx(1, true);

        // Execute
        let mut seen = vec![
            memory_ctx().pop_free_idx(false).unwrap(),
            memory_ctx().pop_free_idx(false).unwrap(),
        ];
        seen.sort();
        let after = memory_ctx().pop_free_idx(false);

        // Assert
        assert_eq!(seen, vec![0, 1]);
        assert_eq!(after, None);
    }

    #[test]
    fn get_write_buffer_hands_out_a_seeded_slot() {
        // Setup
        init_test_free_pool(4);

        // Execute
        let wb = memory_ctx().get_write_buffer(false);

        // Assert
        assert!(wb.slot_idx < 4);
    }

    #[test]
    fn pop_dirty_buffer_returns_a_buffer_when_dirty_pool_has_one() {
        // Setup
        init_test_free_pool(1);

        // Execute
        let popped = memory_ctx().pop_dirty_buffer();

        // Assert
        assert!(popped.is_some());
    }

    #[test]
    fn pop_dirty_buffer_returns_none_when_dirty_pool_is_empty() {
        // Setup: seed only the zeroed pool.
        fresh_ctx();
        memory_ctx().push_free_idx(0, true);

        // Execute
        let popped = memory_ctx().pop_dirty_buffer();

        // Assert
        assert!(popped.is_none());
    }

    #[test]
    fn pop_dirty_buffer_keeps_a_contended_slot_pooled() {
        // Setup: slot 0's index sits in the dirty pool while an evictor probe
        // transiently holds the slot itself.
        init_test_free_pool(1);
        let probe = memory_ctx().ring().try_write(0).unwrap();

        let while_contended = memory_ctx().pop_dirty_buffer();
        probe.release_in_place();
        let after_release = memory_ctx().pop_dirty_buffer();

        // Assert: the contended pop yields nothing, but the slot is still
        // poolable once the probe backs off.
        assert!(while_contended.is_none());
        assert_eq!(after_release.map(|b| b.slot_idx), Some(0));
    }

    #[test]
    fn get_write_buffer_repools_a_contended_slot() {
        // Setup: two dirty slots; an evictor probe transiently holds slot 1.
        init_test_free_pool(2);
        let probe = memory_ctx().ring().try_write(1).unwrap();

        let handed_out = memory_ctx().get_write_buffer(false);
        probe.release_in_place();
        // Pop the zeroed pool directly: the contended slot's ring flag is still
        // `zeroed`, so the push-back routes it there, while a slot the loop
        // never popped would have stayed in the dirty pool. Popping either pool
        // here would pass even if the contended branch never ran.
        let repooled = memory_ctx().zeroed_pool.pop(false);

        // Assert: the free slot is handed out and the contended one is back in
        // a pool rather than leaked.
        assert_eq!(handed_out.slot_idx, 0);
        assert_eq!(repooled, Some(1));
    }

    #[test]
    fn get_write_buffer_repools_contended_slots_when_it_panics() {
        // Setup: the only pooled slot is transiently locked by an evictor
        // probe, so get_write_buffer runs out of pool and panics on evict.
        init_test_free_pool(1);
        let probe = memory_ctx().ring().try_write(0).unwrap();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            memory_ctx().get_write_buffer(false)
        }));
        probe.release_in_place();

        // Assert: the panic unwound without leaking the set-aside slot.
        assert!(result.is_err());
        assert_eq!(memory_ctx().pop_free_idx(false), Some(0));
    }

    #[test]
    fn dropping_a_write_buffer_returns_the_slot_to_the_dirty_pool() {
        // Setup
        init_test_free_pool(1);
        let wb = memory_ctx().get_write_buffer(false);
        let slot_idx = wb.slot_idx;

        // Execute
        drop(wb);
        let recovered = memory_ctx().pop_dirty_buffer();

        // Assert
        assert_eq!(recovered.map(|b| b.slot_idx), Some(slot_idx));
    }

    #[test]
    fn zero_out_returns_the_slot_to_the_zeroed_pool() {
        // Setup
        init_test_free_pool(1);
        let wb = memory_ctx().get_write_buffer(false);
        let slot_idx = wb.slot_idx;

        // Execute
        wb.zero_out();
        let recovered_via_zeroed = memory_ctx().pop_free_idx(true);

        // Assert
        assert_eq!(recovered_via_zeroed, Some(slot_idx));
    }
}
