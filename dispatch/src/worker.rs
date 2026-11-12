//! Per-core worker threads that execute dataflows.
//!
//! Each worker is pinned to a CPU core and runs an event loop that:
//! - Receives [`DataFlowBuilder`]s from the [`Dispatch`](crate::Dispatch) via a channel
//! - Builds them into [`DataFlow`]s (creating operators, channels, etc. on the worker thread)
//! - Drives execution by interleaving CPU work, IO, and work stealing
//!
//! Workers are the execution backbone of the system. A query built with
//! [`RecordBatchOperatorSpec`](crate::api::RecordBatchOperatorSpec)
//! doesn't run until [`.collect()`](crate::api::RecordBatchOperatorSpec::collect)
//! sends one `DataFlowBuilder` to each worker. The worker then:
//!
//! 1. Calls [`DataFlowBuilder::build`] to construct the operator chain — this happens
//!    here because some resources (`Rc<Worker<T>>`, thread-local buffers) are not `Send`
//!    and must be created on the thread that uses them.
//! 2. Runs the resulting `DataFlow` through its event loop, which prioritizes keeping IO
//!    saturated (to hide disk latency), then runs CPU work, then attempts to steal work
//!    from sibling workers when idle.
//!
//! Workers coordinate with each other only at the operator level — e.g. a `Count` operator
//! uses shared atomics so the last worker to finish can emit the total. The worker itself
//! has no direct knowledge of other workers; synchronization is encapsulated in the
//! operators.

use crate::Identifier;
use crate::api::DataFlowBuilder;
use crate::data_flow::{DataFlow, WorkStatus};
use crate::io::{Completion, DiskCache, IORequester};
use crate::memory::{MemoryContextFactory, SpillDirectory, init_memory_context, memory_ctx};
use crate::operations::FinishStatus;
use crate::waker::{WakerSet, WorkerWaker, init_waker_set, init_worker_waker};
use ahash::HashMap;
use core_affinity::CoreId;
use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Live per-worker dataflows across the pool.
///
/// Dirty-buffer cleanup waits for zero to avoid competing with a running query.
static LIVE_DATAFLOWS: AtomicUsize = AtomicUsize::new(0);

/// Spin budget while a **dataflow is in flight** but this worker momentarily has
/// nothing to do (waiting on a pipeline-stage barrier / for a sibling to produce
/// stealable work). A parked worker pays a condvar/futex wakeup (tens of µs plus
/// OS scheduling) every stage transition; with many workers and little data per
/// stage that latency dominates small-query time. The core is idle at a barrier
/// regardless, so spinning here costs nothing useful and lets the worker resume
/// in nanoseconds when the next `notify` lands. Generous enough to bridge the
/// µs-scale gaps between a small query's stages; a longer real stall still falls
/// through to a park. Large queries keep workers busy and rarely reach this path.
const IN_FLIGHT_SPIN_LIMIT: u32 = 120_000;

/// The spin budget, overridable through `PIVOT_SPIN_LIMIT` (`0` parks
/// immediately). Instrumented (PGO) profiling runs set `0`. Profiling runs
/// on a small dataset where waits are short, so the spin usually catches the
/// next wake and the park path and per-pass event-loop machinery barely
/// execute; on the full dataset a cold query's IO waits exhaust any spin
/// budget and every worker parks constantly. A profile taken while spinning
/// therefore underweights exactly the control-flow mix cold execution runs,
/// and its branch weights shift with profiling-run timing, which makes every
/// build a different draw. Parking during profiling records the wait-heavy
/// mix deterministically. The optimized build runs with the variable unset
/// and keeps the full budget.
fn in_flight_spin_limit() -> u32 {
    static LIMIT: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        crate::env::get_env_var_with_default("PIVOT_SPIN_LIMIT", IN_FLIGHT_SPIN_LIMIT)
    })
}
use std::sync::mpsc::Receiver;
use std::sync::{Arc, Barrier};
use std::thread::JoinHandle;
use std::{result, thread};
use thiserror::Error;
use tracing::{debug, instrument, warn};

thread_local! {
    /// This worker's global index, dense across all NUMA node groups
    /// (`0..NUM_WORKERS`, node 0's workers first).
    pub static WORKER_IDX: Cell<usize> = const { Cell::new(usize::MAX) };
    /// Total workers across all node groups.
    pub static NUM_WORKERS: Cell<usize> = const { Cell::new(usize::MAX) };
    /// The NUMA node group this worker belongs to.
    static NODE_IDX: Cell<usize> = const { Cell::new(usize::MAX) };
}

/// The node group of the current worker thread.
pub fn current_node() -> usize {
    NODE_IDX.get()
}

#[cfg(any(test, feature = "test-util"))]
pub(crate) fn set_current_node(node: usize) {
    NODE_IDX.set(node);
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Pipeline(#[from] crate::data_flow::Error),
    #[error("{0}")]
    IORequester(#[from] crate::io::IORequesterError),
}

pub type Result<T, E = Error> = result::Result<T, E>;

/// A Worker is spun up per CPU core. The worker's main entry-point, `create`, spins up a
/// thread with affinity to a CPU which continuously requests work from the dispatcher and does it.
///
/// The main idea of a Worker is to keep everything possible "local" to it, to prevent
/// context-switching/CPU cache-invalidation. Workers are grouped by NUMA node (see
/// [`Dispatch::spin_up`](crate::Dispatch::spin_up)); every dataflow runs on all workers,
/// but each worker allocates only node-local ring memory and steals work only from
/// same-node siblings. We try to make our physical CPU cores first class citizens.
///
/// The Worker receives dataflows from the Dispatcher and runs them- the logic is outlined is as
/// follows:
///
/// 1. Process any outstanding IO completions — we want to always have IO running in background, so
///    we clear any pending if possible
/// 2. Register operators' pending IO with the requester, which submits it immediately
/// 3. Run any CPU work available - this will run a single operation if available
/// 4. Try running trough all dataflows and finishing then
///
/// If no work was done (no CPU work or finish work), then
/// - Wait on existing IO - if none exists, then:
/// - Try stealing work - if none exists, then:
/// - Sleep!
///
/// And forever back to 1!
pub struct Worker {
    /// The identifier of the worker (every worker has a unique ID)
    id: Identifier,
    should_exit: Arc<AtomicBool>,
    /// A requester (through a single per-core uring) for all IO — both disk reads
    /// and HTTP(S) reads of remote regions share the one ring. Shared with the
    /// memory context, so that eviction can wait on spill writes.
    io: Rc<RefCell<IORequester>>,
    /// A queue of dataflows for the worker to work on. The worker can work on many dataflows
    /// simultaneously
    data_flow_queue: Receiver<DataFlowBuilder>,
    /// A mapping of dataflows currently running
    data_flows: HashMap<Identifier, DataFlow>,
    /// Which dataflow is offered read-ahead room first on the next pass.
    next_saturating_flow: usize,
    /// Did the worker do work on last iteration? This is used to decide whether the worker should
    /// do background tasks, just as cleaning
    did_work_last_iteration: bool,
    /// Shared park/notify object. Keeping an owned `Arc` here means the
    /// [`init_worker_waker`] raw pointer stays valid for the lifetime of the
    /// worker thread.
    waker: Arc<WorkerWaker>,
    /// Wake-count snapshot used to decide whether the next park can sleep.
    /// Initialised once at worker startup. Iterations that do work do not
    /// refresh it — any `notify` they missed will simply make the next park
    /// observe a mismatch and return immediately, so no wake is lost.
    last_seen_wake_count: u64,
    /// Ring-wake-count snapshot used the same way for the blocking IO wait.
    /// Only ring-interrupting notifications bump that count, so data sends do
    /// not stop a worker with IO in flight from sleeping until a completion.
    last_seen_ring_wake_count: u64,
    /// This worker's index within its node group, which is also its park
    /// slot in the waker.
    node_local_idx: usize,
    /// Last delegated-broadcast epoch this worker has fanned out (see
    /// [`WorkerWaker::finish_delegated_wake`]).
    last_seen_broadcast: u64,
    /// Where this worker's dataflows spill to when the ring runs short. A
    /// pool that was given no directory spills nothing: its workers never ask
    /// an operator to, and a dataflow that needs more memory than the ring
    /// has fails when eviction finds none to give it.
    spill_directory: Option<SpillDirectory>,
}

impl Worker {
    /// Spawn a worker thread pinned to `core`. Blocks on `ready_barrier` before
    /// entering the event loop, so all workers start roughly together.
    ///
    /// The worker checks the shutdown flag set by [`Dispatch::exit`](crate::Dispatch::exit) on every
    /// iteration and returns from `run` once it flips to `true`; panics
    /// inside the event loop propagate normally and surface through the
    /// returned [`JoinHandle`].
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        idx: usize,
        num_workers: usize,
        node: usize,
        core: CoreId,
        should_exit: Arc<AtomicBool>,
        memory_context_factory: MemoryContextFactory,
        disk_cache: Option<Arc<DiskCache>>,
        spill_directory: Option<SpillDirectory>,
        receiver: Receiver<DataFlowBuilder>,
        ready_barrier: Arc<Barrier>,
        waker: Arc<WorkerWaker>,
        waker_set: WakerSet,
    ) -> JoinHandle<()> {
        thread::spawn(move || {
            // Worker startup (io_uring + memory-context setup) can fail. Every worker
            // must reach `ready_barrier` or `Dispatch::spin_up` deadlocks on it forever,
            // so run startup under `catch_unwind`: on failure we still trip the barrier
            // below (letting spin_up return its handles) and then re-raise the panic.
            let started = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                WORKER_IDX.set(idx);
                NUM_WORKERS.set(num_workers);
                NODE_IDX.set(node);
                // Pin to this worker's core BEFORE touching any memory, so everything
                // this worker first-faults - its node-local ring (via `prefault_buffers`),
                // free pools, and io_uring buffers - lands on this core's NUMA node. Pages
                // are placed where first written under the default policy, so prefaulting
                // before pinning would scatter the ring across nodes and defeat the
                // per-node split.
                core_affinity::set_for_current(core);
                // Register this worker's OS tid so the server can scope a
                // `perf record -t` to the worker pool.
                #[cfg(feature = "perf")]
                crate::profiler::register_worker_tid();
                let node_local_idx = idx % waker_set.workers_per_node();
                waker.register(node_local_idx);
                let last_seen_wake_count = waker.wake_count();
                let last_seen_ring_wake_count = waker.ring_wake_count();
                let last_seen_broadcast = waker.broadcast_epoch();
                debug!("Initializing worker waker {:?}", idx);
                init_worker_waker(&waker);
                init_waker_set(waker_set);
                // Give this worker thread a handle to the shared disk cache so
                // `drop_cache()` can clear it; the requester takes ownership.
                crate::io::disk_cache::install_worker_disk_cache(disk_cache.clone());
                let worker = Self {
                    io: Rc::new(RefCell::new(IORequester::new(disk_cache))),
                    id: core.id,
                    data_flows: Default::default(),
                    next_saturating_flow: 0,
                    data_flow_queue: receiver,
                    did_work_last_iteration: false,
                    should_exit,
                    waker,
                    last_seen_wake_count,
                    last_seen_ring_wake_count,
                    node_local_idx,
                    last_seen_broadcast,
                    spill_directory,
                };
                // Notifiers must be able to interrupt this worker's blocking
                // IO wait, not just its thread park; register the ring's wake
                // line alongside the thread handle.
                worker
                    .waker
                    .register_ring_waker(node_local_idx, worker.io.borrow().wake_handle());
                debug!("Initializing memory context for worker {:?}", idx);
                init_memory_context(memory_context_factory.create_memory_ctx());
                memory_ctx().attach_io_requester(worker.io.clone());
                debug!("Pre-faulting for worker {:?}", idx);
                memory_ctx().prefault_buffers();
                worker
            }));
            debug!("Waiting for barrier for worker {:?}", idx);
            ready_barrier.wait();
            let mut worker = match started {
                Ok(worker) => worker,
                // Re-raise now that spin_up has been released by the barrier
                Err(payload) => std::panic::resume_unwind(payload),
            };
            debug!("Starting worker {:?}", idx);
            worker.run().expect("Worker failed!");
        })
    }

    /// Whether the worker should skip `flow` this pass: a `perf` capture is in
    /// progress and `flow` is not the dataflow being profiled, so it pauses while
    /// the profiled dataflow runs alone (see [`crate::profiler`]).
    #[cfg(feature = "perf")]
    fn paused_for_profiling(flow: &DataFlow) -> bool {
        crate::profiler::profiling_active() && !flow.is_profiled()
    }

    /// Deliver resolved logical reads to their operators until none remain.
    /// Delivery can produce more ready reads synchronously: an operator's
    /// response callback may submit a follow-up (the Parquet footer overflow
    /// path) that resolves straight from the caches, so loop until quiescent.
    fn deliver_ready_reads(&mut self) {
        loop {
            let ready = self.io.borrow_mut().take_ready_reads();
            if ready.is_empty() {
                return;
            }
            for ready in ready {
                if let Some(flow) = self.data_flows.get_mut(&ready.route.data_flow_id) {
                    flow.process_read_response(ready.route.operator_idx, ready.response, &self.io);
                }
            }
        }
    }

    /// When this worker's node is short of reclaimable ring slots, ask the
    /// dataflows' operators to spill this worker's share of the shortfall. The
    /// memory comes back later, once the writes land, so this runs ahead of
    /// exhaustion rather than at it. Runs before CPU work so the slots that
    /// work will allocate are already on their way back.
    ///
    /// Without a spill directory there is nowhere to spill to, so nothing is
    /// asked of the operators and the node's slots are not even counted.
    fn maybe_spill(&mut self) {
        let Some(spill_directory) = &self.spill_directory else {
            return;
        };
        if !memory_ctx().is_pressure_check_due() {
            return;
        }
        let mut wanted = memory_ctx().slots_to_spill();
        if wanted == 0 {
            return;
        }
        for flow in self.data_flows.values_mut() {
            let spilled = flow.spill(&self.io, spill_directory, wanted);
            if spilled > 0 {
                self.did_work_last_iteration = true;
                wanted = wanted.saturating_sub(spilled);
            }
            if wanted == 0 {
                return;
            }
        }
    }

    /// While this worker has asked for less than its read-ahead budget, have
    /// each dataflow issue one more piece of remote IO. It runs ahead of the
    /// CPU work and whether or not there is any: what the worker asks for now
    /// is what it has to work on once a read's time has passed, so a worker
    /// that asked only when it ran out of work would then sit waiting.
    ///
    /// The dataflows share the worker's budget, so each pass starts from the
    /// next one in turn: when room opens for one read at a time, the same
    /// dataflow would otherwise take it every time.
    fn saturate_available_io(&mut self) {
        if self.data_flows.is_empty() || !self.io.borrow().has_read_ahead_room() {
            return;
        }
        let first = self.next_saturating_flow % self.data_flows.len();
        self.next_saturating_flow = self.next_saturating_flow.wrapping_add(1);
        let io = &self.io;
        let mut issued = false;
        let mut offer_room = |flow: &mut DataFlow| {
            #[cfg(feature = "perf")]
            if Self::paused_for_profiling(flow) {
                return;
            }
            // Asked on its own line: the requester must not stay borrowed
            // while the dataflow's operators borrow it to issue their reads.
            let has_room = io.borrow().has_read_ahead_room();
            if has_room && let WorkStatus::Ran = flow.saturate_available_io(io) {
                issued = true;
            }
        };
        self.data_flows
            .values_mut()
            .skip(first)
            .for_each(&mut offer_room);
        self.data_flows
            .values_mut()
            .take(first)
            .for_each(&mut offer_room);
        if issued {
            self.did_work_last_iteration = true;
        }
    }

    /// Run one unit of CPU work from the first dataflow that has work ready.
    fn step_run_ready_cpu_work(&mut self) {
        // While a `perf` capture is in progress, run only the profiled dataflow
        // so the profile isn't polluted by other queries / ingest on the pool.
        for flow in self.data_flows.values_mut() {
            #[cfg(feature = "perf")]
            if Self::paused_for_profiling(flow) {
                continue;
            }
            if let WorkStatus::Ran = flow.run_ready_cpu_work(&self.io) {
                self.did_work_last_iteration = true;
            }
        }
    }

    /// Deliver completed reads back to the operators that requested them, each
    /// transport to its own handler; by the time we're here each block's bytes
    /// are already committed to its cache slot.
    fn process_io_completions(&mut self) -> Result<()> {
        let completions = {
            let mut io = self.io.borrow_mut();
            io.collect_completions()?;
            // Spill writes were issued by the memory context's spiller, not by
            // an operator, so they are counted off there. A worker with no
            // spill directory issues none.
            if self.spill_directory.is_some() {
                memory_ctx().complete_landed_spill_writes(&mut io);
            }
            io.take_completions()
        };
        for completion in completions {
            match completion {
                // The owning dataflow may already be gone: a finished or
                // cancelled query/compaction leaves its read-ahead reads in
                // flight, and their completions land here afterwards. The block
                // was already committed in the requester, so a missing dataflow
                // just means drop the completion (the slot pin releases with it).
                Ok(Completion::FsRead(r)) => {
                    if let Some(data_flow) = self.data_flows.get_mut(&r.data_flow_id) {
                        data_flow.stats().record_disk_read_time(r.submitted_at);
                    }
                }
                Ok(Completion::FsReadDirect(r)) => {
                    if let Some(data_flow) = self.data_flows.get_mut(&r.data_flow_id) {
                        data_flow.stats().record_disk_read_time(r.submitted_at);
                    }
                }
                Ok(Completion::FsWrite(r)) => {
                    if let Some(data_flow) = self.data_flows.get_mut(&r.data_flow_id) {
                        data_flow.stats().record_disk_write_time(r.submitted_at);
                        data_flow.process_fs_write(r.operator_idx, r.request);
                    }
                }
                Ok(Completion::HttpGet(r, time)) => {
                    if let Some(data_flow) = self.data_flows.get_mut(&r.data_flow_id) {
                        data_flow.stats().record_http_get_time(time);
                    }
                }
                Ok(Completion::HttpUpload(r)) => {
                    if let Some(data_flow) = self.data_flows.get_mut(&r.data_flow_id) {
                        data_flow.stats().record_http_upload_time(r.submitted_at);
                        data_flow.process_http_upload_response(r.operator_idx, r.request);
                    }
                }
                Err(failed) => {
                    // A read failed terminally (an HTTP read past its retries, or
                    // a disk read whose CQE came back negative). Cancel just the
                    // owning dataflow — its query errors out to the client while
                    // the worker and every other query keep running. The dataflow
                    // may already be gone if the query was cancelled meanwhile.
                    self.cancel_failed_dataflow(failed.data_flow_id, failed.error.to_string());
                }
            }
        }
        self.deliver_ready_reads();
        Ok(())
    }

    /// Report one physical I/O failure to the dataflow owning its logical request.
    fn cancel_failed_dataflow(&mut self, data_flow_id: Identifier, message: String) {
        if let Some(data_flow) = self.data_flows.get_mut(&data_flow_id) {
            let error = std::io::Error::other(message);
            data_flow.bail_and_cancel(crate::io::IORequesterError::from(error).into());
        }
    }

    /// Check each dataflow for completion and remove finished ones.
    fn try_finishing_dataflows(&mut self) {
        let mut to_remove = Vec::new();
        for (id, flow) in &mut self.data_flows {
            match flow.maybe_finish() {
                FinishStatus::Done => to_remove.push(*id),
                // An operator is still producing output: count it as work so we
                // re-drive it via `run_cpu_work` next iteration instead of
                // parking one stage short.
                FinishStatus::Working => self.did_work_last_iteration = true,
                FinishStatus::Pending => {}
            }
        }

        for id in to_remove {
            debug!("Finished data flow {:?}", id);
            if let Some(mut flow) = self.data_flows.remove(&id) {
                flow.stats().report();
                // The threads that carry the result to the client wake about
                // now, while every worker is busy freeing the dataflow's
                // state. A thread that wakes onto a busy worker's core can
                // wait until the next scheduler tick, so offer the core
                // before and after the drop. With nothing waiting the call
                // returns at once.
                thread::yield_now();
                drop(flow);
                thread::yield_now();
                self.release_live_dataflow();
            }
        }
    }

    /// Fail every dataflow this worker still runs because the pool is
    /// shutting down. A dataflow dropped without this closes its output channel
    /// like one that finished, and a collector then takes whatever it received
    /// so far for the whole result: a compaction merge cut short this way
    /// committed the removal of its inputs with no outputs. Specs already
    /// queued for this worker are built first so they fail the same way.
    fn fail_live_dataflows_for_shutdown(&mut self) {
        self.try_receiving_new_dataflow();
        let live: Vec<DataFlow> = self.data_flows.drain().map(|(_, flow)| flow).collect();
        for mut flow in live {
            flow.fail_for_shutdown();
            flow.stats().report();
            drop(flow);
            self.release_live_dataflow();
        }
    }

    /// Decrements the pool-wide count after a dataflow instance is dropped.
    ///
    /// When the count reaches zero, wake parked workers so they can zero the
    /// query's returned buffers before the next query. Returning a buffer does
    /// not wake its home worker, which may already be parked.
    ///
    /// The dataflow must be dropped before this call so all of its buffers are
    /// in the free pools when workers wake. See
    /// [`Self::clear_dirty_buffer_or_park`].
    fn release_live_dataflow(&self) {
        if LIVE_DATAFLOWS.fetch_sub(1, Ordering::Relaxed) == 1 {
            self.waker.notify();
        }
    }

    /// Attempt to steal work from sibling workers' channels when this worker is idle.
    fn try_steal_work(&mut self) {
        for flow in self.data_flows.values_mut() {
            #[cfg(feature = "perf")]
            if Self::paused_for_profiling(flow) {
                continue;
            }
            if let WorkStatus::Ran = flow.try_stealing_work(&self.io) {
                self.did_work_last_iteration = true;
            }
        }
    }

    /// Called when no other work is available; we either park or clean a dirty buffer.
    /// We clean dirty buffers when we have nothing else to do to help future execution.
    ///
    /// Parking compares against `last_seen_wake_count` (captured under the waker's
    /// mutex at the previous park): if it has not advanced, we wait on the condvar;
    /// if it has, we return immediately to retry. The new count is captured
    /// atomically with the wait and stored back into `last_seen_wake_count` for the
    /// next park.
    fn clear_dirty_buffer_or_park(&mut self) {
        // Only clean dirty buffers while the whole pool is between queries —
        // while any dataflow is running, anywhere, we don't want to spend
        // memory bandwidth and filler-time on buffer cleanup that could
        // otherwise serve the query's own tail (a worker whose share finished
        // early would steal bandwidth from the stragglers).
        //
        // Ideally, we would do cleanup also when you have dataflows above a certain
        // threshold of dirty buffers - but we don't have that implemented yet.
        if LIVE_DATAFLOWS.load(Ordering::Relaxed) == 0
            && let Some(b) = memory_ctx().pop_dirty_buffer()
        {
            b.zero_out();
            // Zeroing is not urgent: let a thread waiting for this core run.
            thread::yield_now();

            // Bound work per pass: zero one buffer, then hand
            // control back to the main loop so a newly-arrived
            // dataflow / new stealable work isn't starved behind
            // a long cleanup run.
            return;
        }

        // Spin-before-park, but only while a dataflow is in flight. Parking on
        // the condvar costs a wakeup (tens of µs + OS scheduling) on the next
        // `notify`; for a small multi-stage query that per-stage wakeup latency
        // dominates, so a worker idle at a stage barrier polls the lock-free
        // wake count and resumes in nanoseconds when the next notify lands (the
        // core is idle at the barrier anyway). With no dataflow running we're
        // genuinely idle between queries — park immediately rather than burn CPU.
        if !self.data_flows.is_empty() {
            for _ in 0..in_flight_spin_limit() {
                let now = self.waker.wake_count();
                if now != self.last_seen_wake_count {
                    self.last_seen_wake_count = now;
                    return;
                }
                std::hint::spin_loop();
            }
        }

        self.last_seen_wake_count = self
            .waker
            .wait_if_unchanged(self.last_seen_wake_count, self.node_local_idx);
        self.waker
            .finish_delegated_wake(&mut self.last_seen_broadcast);
    }

    fn clear_cancelled_dataflows(&mut self) {
        let mut cancelled = Vec::new();
        self.data_flows.retain(|id, d| {
            if d.cancelled() {
                // Ship this worker's partial tally before dropping the flow, so a
                // failed or cancelled query's IO stats still reach the handle (a
                // finished flow reports in `try_finishing_dataflows`).
                d.stats().report();
                cancelled.push(*id);
                false
            } else {
                true
            }
        });
        for id in cancelled {
            self.io.borrow_mut().cancel_dataflow(id);
            self.release_live_dataflow();
        }
    }

    /// Let any operator that's done with its upstream (e.g. a satisfied `LIMIT`)
    /// abandon the scan feeding it, freeing its in-flight buffers and stopping
    /// further reads. Distinct from [`clear_cancelled_dataflows`](Self::clear_cancelled_dataflows):
    /// that tears down a whole query, this prunes only one query's upstream while
    /// the rest keeps running.
    fn cancel_upstream_in_dataflows(&mut self) {
        for flow in self.data_flows.values_mut() {
            flow.cancel_upstream_if_requested();
        }
    }

    fn try_receiving_new_dataflow(&mut self) {
        // Drain everything queued. Taking one builder per pass loses dataflows:
        // two dispatches can land while this worker is parked, their notifies
        // coalescing into one wake-count advance. The woken pass would take only
        // the first builder, finish it without setting `did_work`, and re-park
        // with the count already current, stranding the second dataflow's
        // builder in the queue forever (its query hangs; sibling workers spin
        // or park waiting for this worker's operator to ever exist).
        while let Ok(builder) = self.data_flow_queue.try_recv() {
            debug!("Received data flow...");
            match builder.build() {
                Ok(data_flow) => {
                    LIVE_DATAFLOWS.fetch_add(1, Ordering::Relaxed);
                    self.data_flows.insert(data_flow.id(), data_flow);
                }
                Err(e) => {
                    warn!("Failed to build dataflow {:?}", e)
                }
            }
        }
    }

    #[instrument(skip(self), fields(worker_id = %self.id))]
    pub fn run(&mut self) -> Result<()> {
        loop {
            // Cooperative shutdown: [`crate::Dispatch::exit`] flips this flag
            // and we exit cleanly so the thread can be reaped.
            if self.should_exit.load(Ordering::Relaxed) {
                debug!("Worker {} exiting via shutdown flag", self.id);
                self.fail_live_dataflows_for_shutdown();
                return Ok(());
            }
            self.did_work_last_iteration = false;

            self.try_receiving_new_dataflow();

            self.clear_cancelled_dataflows();

            // Prune any abandoned upstream before submitting IO, so a satisfied
            // LIMIT's scan issues no further reads.
            self.cancel_upstream_in_dataflows();

            self.process_io_completions()?;

            self.maybe_spill();

            self.saturate_available_io();

            self.step_run_ready_cpu_work();

            self.try_finishing_dataflows();

            if !self.did_work_last_iteration {
                // Don't clear buffers or sleep on IO while there is work to steal.
                self.try_steal_work();
                if self.did_work_last_iteration {
                    continue;
                }

                // One ring serves both disk and HTTP, so a single wait wakes on
                // either kind of completion — no dual-ring coordination needed.
                // The wait doubles as a select over worker notifications: the
                // slot published around it lets a notifier interrupt the ring
                // (see `begin_ring_wait`), so a message sent mid-wait resumes
                // the loop instead of stalling behind the slowest read.
                if self.io.borrow().has_pending() {
                    debug!("Waiting for IO...");
                    if self
                        .waker
                        .begin_ring_wait(self.node_local_idx, self.last_seen_ring_wake_count)
                    {
                        // Withdraw the slot before propagating a wait error.
                        // If the error tears this worker down while the slot
                        // still reads as parked, notifiers keep claiming it:
                        // each claimed wake is swallowed by a dead worker
                        // instead of reaching a live parked one, and the wake
                        // itself writes to an eventfd this thread closed on
                        // its way out.
                        let wait_result = self.io.borrow_mut().wait();
                        self.waker.end_ring_wait(self.node_local_idx);
                        wait_result?;
                    }
                    self.last_seen_ring_wake_count = self.waker.ring_wake_count();
                    self.last_seen_wake_count = self.waker.wake_count();
                    continue;
                }

                self.clear_dirty_buffer_or_park();
            }
        }
    }
}
