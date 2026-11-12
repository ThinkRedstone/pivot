//! A DataFlow is a graph of operators that a single worker executes.
//!
//! Built from an [`OperatorGraphBuilder`](crate::api::OperatorGraphBuilder) during
//! [`DataFlowBuilder::build`](crate::api::DataFlowBuilder::build)
//! on the worker thread. Operators are connected by channels created during the
//! build step; a graph may contain multiple roots, such as a join's build and probe paths.
//!
//! The worker drives execution by calling methods on the `DataFlow`:
//! - [`run_ready_cpu_work`](DataFlow::run_ready_cpu_work) — traverse leaf-to-root,
//!   running the first operator that has work ready. Leaf-to-root (downstream first)
//!   keeps data hot in cache — we process what was just produced before moving upstream.
//! - [`try_stealing_work`](DataFlow::try_stealing_work) — traverse root-to-leaf,
//!   attempting to steal from peer workers' channels. Root-to-leaf (upstream first)
//!   means the stealing worker picks up data early in the dataflow, to not interrupt current
//!   hot-in-cache processing
//! - [`process_read_response`](DataFlow::process_read_response),
//!   [`process_fs_write`](DataFlow::process_fs_write), and
//!   [`process_http_upload_response`](DataFlow::process_http_upload_response) —
//!   deliver completed IO to the operator that requested it.
//! - [`maybe_finish`](DataFlow::maybe_finish) — check if all operators have completed.

use crate::Identifier;
use crate::graph::Graph;
use crate::io::{FsWriteRequest, HttpUploadRequest, IORequester, OperatorIO, ReadResponse};
use crate::memory::{DataflowEvict, Evict, SpillDirectory, Spiller, memory_ctx};
use crate::operations::{FinishStatus, Operator};
use crate::stats::{DataFlowStats, StatsCollector};
use crate::waker::waker_set;
use crate::worker::WORKER_IDX;
use ahash::HashMap;
use std::cell::RefCell;
use std::fmt::{Debug, Formatter};
use std::ops::ControlFlow;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::time::Instant;
use std::{panic, result};
use thiserror::Error;
use tracing::{debug, error, info, warn};

/// One node of the operator graph. Owns its operator; edges are tracked
/// in the parent `DataFlow` as index lists.
struct OperatorNode {
    id: Identifier,
    /// The node's operator, or `None` once it has finished (or been abandoned).
    /// Dropping it at that moment frees what it held right away: the batches
    /// buffered in its channels, its hash tables, and the downstream sender it
    /// owns. Releasing that sender is also how a stage's completion becomes
    /// observable to whoever holds the other end.
    ///
    /// Retiring the node in place, rather than restructuring the graph, keeps
    /// every edge index stable: traversals skip a retired node and move on, and
    /// a late IO completion routed to one lands nowhere.
    operator: Option<Box<dyn Operator>>,

    /// One-shot publication gates. Open gates are removed after an Acquire
    /// observation, leaving no atomic loads on this node's steady-state path.
    gates: Vec<Arc<AtomicBool>>,
}

impl OperatorNode {
    /// Whether this node has retired, which is exactly "it no longer holds an
    /// operator".
    #[inline]
    fn is_finished(&self) -> bool {
        self.operator.is_none()
    }

    /// Run `f` against this node's operator, or return `None` when the node has
    /// nothing to run because it retired or is still gated.
    #[inline]
    fn run<T>(
        &mut self,
        f: &mut impl FnMut(Identifier, &mut dyn Operator) -> Result<ControlFlow<T>>,
    ) -> Result<Option<ControlFlow<T>>> {
        if self.is_gated() {
            return Ok(None);
        }
        let id = self.id;
        match self.operator.as_deref_mut() {
            Some(operator) => f(id, operator).map(Some),
            None => Ok(None),
        }
    }

    #[inline]
    fn is_gated(&mut self) -> bool {
        self.gates.retain(|gate| !gate.load(Ordering::Acquire));
        !self.gates.is_empty()
    }
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("Cannot find operation {0}")]
    CannotFindOperation(Identifier),
    #[error(transparent)]
    Operation(#[from] crate::operations::Error),
    #[error(transparent)]
    IORequester(#[from] crate::io::IORequesterError),
    #[error("{0}")]
    Panic(String),
    #[error("the worker pool shut down before the dataflow finished")]
    Shutdown,
}

pub type Result<T, E = Error> = result::Result<T, E>;

/// Whether an operator did useful work in this step.
#[derive(PartialEq, Eq, Copy, Clone)]
pub enum WorkStatus {
    /// No work was available or ready.
    Pending,
    /// The operator consumed or produced data.
    Ran,
}

/// The dataflow's operators arranged as a [`Graph`]. Backwards (leaf-to-root)
/// traversal runs CPU work and collects IO requests; forwards (root-to-leaf)
/// traversal steals and finishes.
struct OperatorGraph {
    graph: Graph<OperatorNode>,
    /// The dataflow this graph belongs to, which names its part of the spill
    /// directory.
    dataflow_id: Identifier,
    /// Whether the dataflow registered its operators' evictors with the memory
    /// context, so that a dropped operator's evictor has to be removed there.
    evictors_registered: bool,
    /// The directory this dataflow's operators have been asked to spill into
    /// on this worker, so that there may be files of theirs under it to
    /// remove. `None` until they are asked, and until then nothing is removed:
    /// each removal is a path walk through the kernel, once per retired
    /// operator and once per dropped dataflow on every worker, and a query
    /// that never came under memory pressure would pay for all of them. Set
    /// when the operators are asked, not when a buffer is counted as spilled,
    /// because a spill opens its files before its writes are issued and
    /// counted.
    spilled_into: Option<SpillDirectory>,
}

impl OperatorGraph {
    fn from_edges(
        dataflow_id: Identifier,
        evictors_registered: bool,
        operators: Vec<Box<dyn Operator>>,
        publisher_to_subscriber: Vec<(Identifier, Identifier)>,
        mut gates: HashMap<Identifier, Vec<Arc<AtomicBool>>>,
    ) -> Self {
        debug!(
            "Building from publisher to subscriber {:?}",
            publisher_to_subscriber
        );

        let nodes: Vec<OperatorNode> = operators
            .into_iter()
            .enumerate()
            .map(|(id, operator)| OperatorNode {
                id,
                operator: Some(operator),
                gates: gates.remove(&id).unwrap_or_default(),
            })
            .collect();

        Self {
            graph: Graph::from_edges(nodes, &publisher_to_subscriber),
            dataflow_id,
            evictors_registered,
            spilled_into: None,
        }
    }

    /// Drop the operator at `idx`, and with it everything it spilled to disk
    /// and the evictor through which the memory context could reach it.
    fn drop_operator(&mut self, idx: Identifier) {
        self.graph.node_mut(idx).operator = None;
        if self.evictors_registered {
            memory_ctx().remove_operator_evictor(self.dataflow_id, idx);
        }
        let Some(spill_directory) = &self.spilled_into else {
            return;
        };
        if let Err(error) =
            spill_directory.remove_operator_directory(self.dataflow_id, WORKER_IDX.get(), idx)
        {
            warn!(
                "could not remove operator {idx}'s spill files of dataflow {}: {error}",
                self.dataflow_id
            );
        }
    }

    fn nodes(&self) -> &[OperatorNode] {
        self.graph.nodes()
    }

    fn node_mut(&mut self, idx: usize) -> &mut OperatorNode {
        self.graph.node_mut(idx)
    }

    /// Walk leaf-to-root via back-edges, running the closure against every node
    /// that still has work to give: a retired or gated node is stepped over, but
    /// the walk carries on through it to its publishers.
    pub fn traverse_backwards<
        T,
        F: FnMut(Identifier, &mut dyn Operator) -> Result<ControlFlow<T>>,
    >(
        &mut self,
        mut f: F,
    ) -> Result<ControlFlow<T>> {
        self.graph.traverse_backwards(|_, node| {
            Ok(node.run(&mut f)?.unwrap_or(ControlFlow::Continue(())))
        })
    }

    /// The root-to-leaf twin of [`traverse_backwards`](Self::traverse_backwards).
    pub fn traverse_forwards<
        T,
        F: FnMut(Identifier, &mut dyn Operator) -> Result<ControlFlow<T>>,
    >(
        &mut self,
        mut f: F,
    ) -> Result<ControlFlow<T>> {
        self.graph
            .traverse_forwards(|_, node| Ok(node.run(&mut f)?.unwrap_or(ControlFlow::Continue(()))))
    }

    /// Advance finishing from every root. A pending node blocks only what is
    /// downstream of it, not sibling roots.
    fn try_finish(&mut self) -> Result<FinishStatus> {
        let mut working = false;

        let mut stack: Vec<usize> = self.graph.roots().to_vec();
        while let Some(idx) = stack.pop() {
            // A converging node cannot finish before all of its publishers.
            if self
                .graph
                .back_edges(idx)
                .iter()
                .any(|&publisher| !self.graph.node(publisher).is_finished())
            {
                continue;
            }

            let node = self.graph.node_mut(idx);
            if node.is_gated() {
                continue;
            }
            if let Some(operator) = node.operator.as_deref_mut() {
                match operator.try_finish()? {
                    // Retiring the node drops the operator, and with it the
                    // sender it holds, so a stage that watches the other end
                    // of that channel learns the stage is over.
                    FinishStatus::Done => self.drop_operator(idx),
                    FinishStatus::Working => working = true,
                    FinishStatus::Pending => {}
                }
            }

            if !self.graph.node(idx).is_finished() {
                continue;
            }
            stack.extend(self.graph.edges(idx).iter().copied());
        }

        Ok(if self.nodes().iter().all(OperatorNode::is_finished) {
            FinishStatus::Done
        } else if working {
            FinishStatus::Working
        } else {
            FinishStatus::Pending
        })
    }

    /// Abandon every operator transitively *upstream* of `node` (its publishers,
    /// their publishers, and so on) by retiring each of them. The node itself and
    /// everything downstream are left running. Used by a satisfied `LIMIT` to
    /// stop and free the scan feeding it.
    ///
    /// A publisher is only abandoned once nothing still reads it. On a chain
    /// that is every ancestor, since each feeds downstream only through `node`.
    /// Where a stage's output is read in more than one place, such as a CTE
    /// feeding several scans, a `LIMIT` over one reader must leave the stage
    /// alone: the other readers are still waiting on its rows.
    fn abandon_ancestors(&mut self, node: usize) {
        // The requester keeps running, but as far as its publishers are
        // concerned it is done reading.
        let mut done: Vec<bool> = (0..self.nodes().len())
            .map(|idx| idx == node || self.graph.node(idx).is_finished())
            .collect();

        let mut stack: Vec<usize> = self.graph.back_edges(node).to_vec();
        while let Some(idx) = stack.pop() {
            if done[idx] {
                // Guards against re-visiting on a DAG and against re-abandoning
                // across repeated calls; on a chain it simply never triggers.
                continue;
            }
            if !self
                .graph
                .edges(idx)
                .iter()
                .all(|&subscriber| done[subscriber])
            {
                // Still read by someone else, so it stays. A reader can't retire
                // before the stage feeding it, so in practice a stage read in
                // more than one place runs to completion rather than being freed
                // early. That costs the work a `LIMIT` would have saved, and is
                // the direction to err in: the alternative starves the readers
                // that had not finished.
                continue;
            }
            self.drop_operator(idx);
            done[idx] = true;
            stack.extend(self.graph.back_edges(idx).iter().copied());
        }
    }
}

/// A graph of operators executed by a single worker.
///
/// Operators are connected by channels and arranged in an [`OperatorGraph`].
/// The worker drives execution by repeatedly calling [`run_ready_cpu_work`](Self::run_ready_cpu_work),
/// [`try_stealing_work`](Self::try_stealing_work), and IO methods.
pub struct DataFlow {
    id: Identifier,
    cancelled: Arc<AtomicBool>,
    err_tx: mpsc::Sender<Error>,
    graph: OperatorGraph,
    /// Operators that can ask to abandon their upstream, paired with the flag
    /// they raise to do so (see [`Operator::upstream_cancel_flag`]). Collected
    /// once at construction so the hot loop only polls the (usually empty) list,
    /// never every operator. An entry is dropped once it has fired.
    upstream_cancellers: Vec<(usize, Arc<AtomicBool>)>,
    /// Operators that issue reads when asked to saturate the available IO (see
    /// [`Operator::saturates_available_io`]), in the order CPU work visits
    /// them. Collected once at construction, like `upstream_cancellers`.
    io_saturators: Vec<usize>,
    /// Collects this worker's execution stats for the dataflow. Inert unless the
    /// query opted in; driven by the worker through [`stats`](Self::stats).
    stats: StatsCollector,
    /// Request-id counter for reads and writes, shared by every node and
    /// borrowed into the [`OperatorIO`] handed to operators. Responses route by
    /// `(dataflow, operator)` before the id, so one counter keeps identifiers
    /// unique everywhere an operator can observe them.
    next_request_id: usize,
    /// Whether this dataflow is being profiled. While any profiled dataflow is
    /// live the worker runs *only* profiled dataflows (see
    /// [`crate::profiler`]), so a `perf` capture isn't polluted by other work
    /// sharing the pool. Set when the launching dispatcher was marked (see
    /// [`DataFlowDispatcher::with_profiling`](crate::DataFlowDispatcher::with_profiling)).
    #[cfg(feature = "perf")]
    profiled: bool,
    /// Whether this dataflow's operators were registered with the memory
    /// context as evictors (see [`DataflowEvict`]), and so must be unregistered
    /// when it drops. False when no operator declared one.
    registered_evictors: bool,
}

impl Debug for DataFlow {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut debug_struct = f.debug_struct("Dataflow");
        debug_struct.field("id", &self.id).finish()
    }
}

impl Drop for DataFlow {
    fn drop(&mut self) {
        if self.registered_evictors {
            memory_ctx().unregister_dataflow_evictors(self.id);
        }
        if let Some(spill_directory) = &self.graph.spilled_into
            && let Err(error) = spill_directory.remove_worker_directory(self.id, WORKER_IDX.get())
        {
            warn!(
                "could not remove dataflow {}'s spill files: {error}",
                self.id
            );
        }
        // Release this dataflow's hold on exclusive execution once it finishes
        // (or is cancelled/dropped), paired with the bump in `with_profiling`.
        #[cfg(feature = "perf")]
        if self.profiled && crate::profiler::PROFILED_FLOWS.fetch_sub(1, Ordering::Relaxed) == 1 {
            // Exclusive mode just lifted. A worker that parked while holding only
            // non-profiled (paused) dataflows got no channel send to wake it, so
            // notify every node unconditionally or it sleeps until the next
            // unrelated wake.
            waker_set().notify_all();
        }
    }
}

impl DataFlow {
    #[allow(clippy::too_many_arguments)] // wired straight from the builder; each is needed
    pub fn new(
        id: Identifier,
        canceled: Arc<AtomicBool>,
        err_tx: mpsc::Sender<Error>,
        operators: Vec<Box<dyn Operator>>,
        evictors: Vec<Option<Box<dyn Evict>>>,
        publisher_to_subscriber: Vec<(Identifier, Identifier)>,
        gates: HashMap<Identifier, Vec<Arc<AtomicBool>>>,
        stats_tx: mpsc::Sender<DataFlowStats>,
        collect_stats: bool,
    ) -> Self {
        assert_eq!(
            evictors.len(),
            operators.len(),
            "one evictor entry per operator"
        );
        // Operators that can give ring slots back are laid out for the memory
        // context in the operator graph's shape, so it asks them in graph order.
        let evictors = DataflowEvict::new(evictors, &publisher_to_subscriber);
        let registered_evictors = !evictors.is_empty();
        if registered_evictors {
            memory_ctx().register_dataflow_evictors(id, evictors);
        }

        let mut graph = OperatorGraph::from_edges(
            id,
            registered_evictors,
            operators,
            publisher_to_subscriber,
            gates,
        );
        // Every node is looked at, gated ones too: a gate only says when an
        // operator starts, not what it will do once it has.
        let mut io_saturators = Vec::new();
        let Ok(_) = graph.graph.traverse_backwards(|idx, node| {
            let saturates = node
                .operator
                .as_ref()
                .is_some_and(|operator| operator.saturates_available_io());
            if saturates && !io_saturators.contains(&idx) {
                io_saturators.push(idx);
            }
            Ok::<ControlFlow<()>, std::convert::Infallible>(ControlFlow::Continue(()))
        });
        let upstream_cancellers = graph
            .nodes()
            .iter()
            .enumerate()
            .filter_map(|(idx, node)| {
                node.operator
                    .as_ref()?
                    .upstream_cancel_flag()
                    .map(|flag| (idx, flag))
            })
            .collect();
        Self {
            id,
            graph,
            upstream_cancellers,
            io_saturators,
            cancelled: canceled,
            err_tx,
            stats: StatsCollector::new(stats_tx, collect_stats),
            next_request_id: 0,
            #[cfg(feature = "perf")]
            profiled: false,
            registered_evictors,
        }
    }
    pub fn id(&self) -> Identifier {
        self.id
    }

    /// Mark this dataflow profiled. Bumps the global live-profiled count (paired
    /// with the decrement in [`Drop`]) so the worker switches to running only
    /// profiled dataflows while this one exists. Call exactly once per dataflow
    /// (it is not idempotent): a second `true` call leaks a count that never
    /// drops, wedging exclusive mode on. `pub(crate)`: only the builder uses it.
    #[cfg(feature = "perf")]
    pub(crate) fn with_profiling(mut self, profiled: bool) -> Self {
        if profiled {
            crate::profiler::PROFILED_FLOWS.fetch_add(1, Ordering::Relaxed);
        }
        self.profiled = profiled;
        self
    }

    /// Whether this dataflow is profiled (and so runs exclusively).
    #[cfg(feature = "perf")]
    pub fn is_profiled(&self) -> bool {
        self.profiled
    }

    /// This dataflow's stats collector. The worker records IO/CPU work against it
    /// (`flow.stats().record_*`) and `report`s it when the dataflow finishes.
    pub fn stats(&mut self) -> &mut StatsCollector {
        &mut self.stats
    }

    /// Is this dataflow cancelled? This is an AtomicBool that can be set from other workers or from
    /// outside dispatch
    pub fn cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    /// Bail on running the current dataflow because of an error. This will set `canceled` to true
    /// (cancelling it for all workers) and send the error out. Called both from inside
    /// [`try_run`](Self::try_run) and directly by the worker when work it submitted on the
    /// dataflow's behalf (e.g. a remote read) fails terminally.
    pub fn bail_and_cancel(&self, err: Error) {
        error!("DataFlow {:?} failed: {err}", self.id());
        self.send_error_and_cancel(err);
    }

    /// Fail the dataflow because the worker pool is shutting down under it.
    /// Whoever is collecting its output then gets an error instead of a clean
    /// end of stream, so a caller cannot mistake the rows it received before
    /// the shutdown for the dataflow's whole result. Expected during shutdown,
    /// so it is not reported as a failure.
    pub fn fail_for_shutdown(&self) {
        info!("DataFlow {:?} cut short by shutdown", self.id());
        self.send_error_and_cancel(Error::Shutdown);
    }

    fn send_error_and_cancel(&self, err: Error) {
        if self.err_tx.send(err).is_err() {
            warn!("Unable to send error...");
        }
        // Cancellation must reach workers parked on any node.
        self.cancelled.store(true, Ordering::Relaxed);
        waker_set().notify_all();
    }

    /// Try running a function within the dataflow- this will gracefully catch any errors/panics and
    /// send them via `err_tx`, along with setting `cancelled` to true across all workers (thus
    /// causing the dataflow to stop processing).
    fn try_run(&mut self, f: impl FnOnce(&mut Self) -> Result<()>) {
        self.try_run_or((), f);
    }

    /// Similar to the above function, `try_run_or` will run catch any errors and handle them
    /// appropriately, while also returning a default value if there is indeed an error
    fn try_run_or<R>(&mut self, default: R, f: impl FnOnce(&mut Self) -> Result<R>) -> R {
        // Every operator step funnels through here, so it's the one place that
        // can see the dataflow's CPU work (the worker only sees opaque method
        // calls). `then` skips the clock reads when stats are off.
        let started = self.stats.enabled().then(Instant::now);
        let outcome = panic::catch_unwind(AssertUnwindSafe(|| f(self)));
        if let Some(started) = started {
            self.stats.record_cpu(started.elapsed());
        }
        match outcome {
            Ok(Ok(v)) => v,
            Ok(Err(e)) => {
                self.bail_and_cancel(e);
                default
            }
            Err(e) => {
                let msg = e
                    .downcast_ref::<String>()
                    .map(|s| s.as_str())
                    .or_else(|| e.downcast_ref::<&str>().copied())
                    .unwrap_or("unknown panic");
                self.bail_and_cancel(Error::Panic(msg.to_string()));
                default
            }
        }
    }

    /// Drive every operator's finish (root-to-leaf) and report the dataflow's
    /// aggregate [`FinishStatus`]: [`Done`](FinishStatus::Done) when all
    /// operators have finished (the worker drops the dataflow),
    /// [`Working`](FinishStatus::Working) when one is still producing output
    /// (the worker must keep driving rather than park), or
    /// [`Pending`](FinishStatus::Pending) when stalled on input/siblings (the
    /// worker may park).
    ///
    /// An operator that reports [`FinishStatus::Done`] is dropped there and then,
    /// so subsequent passes step over its node.
    pub fn maybe_finish(&mut self) -> FinishStatus {
        self.try_run_or(FinishStatus::Pending, |d| d.graph.try_finish())
    }

    /// Honour any pending upstream-cancellation request: for each operator that
    /// has raised its [`upstream_cancel_flag`](Operator::upstream_cancel_flag)
    /// (e.g. a satisfied `LIMIT`), abandon everything upstream of it. Operators
    /// downstream are left running.
    ///
    /// Called once per worker-loop iteration. When no operator can cancel
    /// upstream (the common case) `upstream_cancellers` is empty and this is a
    /// single `is_empty` check.
    pub fn cancel_upstream_if_requested(&mut self) {
        if self.upstream_cancellers.is_empty() {
            return;
        }
        // Retain only the cancellers that haven't fired yet; abandon the
        // ancestors of the ones that have, so each request is acted on once.
        let graph = &mut self.graph;
        self.upstream_cancellers.retain(|(node, flag)| {
            if flag.load(Ordering::Relaxed) {
                graph.abandon_ancestors(*node);
                false
            } else {
                true
            }
        });
    }

    /// Deliver a fully resolved logical read to its operator.
    pub fn process_read_response(
        &mut self,
        node_id: Identifier,
        response: ReadResponse,
        requester: &RefCell<IORequester>,
    ) {
        self.try_run(|d| {
            let data_flow_id = d.id;
            let stats = &mut d.stats;
            let next_request_id = &mut d.next_request_id;
            let node = d.graph.node_mut(node_id);
            if let Some(operator) = node.operator.as_deref_mut() {
                let mut io =
                    OperatorIO::new(requester, data_flow_id, node.id, stats, next_request_id);
                operator.process_read_response(&mut io, response)?;
            }
            Ok(())
        });
    }

    /// Notify the operator that requested it that one of its filesystem writes
    /// has completed. A spill write was issued by this dataflow rather than the
    /// operator, so it is completed here: the buffer is now on disk.
    pub fn process_fs_write(&mut self, node_id: Identifier, request: FsWriteRequest) {
        self.try_run(|d| {
            let node = d.graph.node_mut(node_id);
            if let Some(operator) = node.operator.as_deref_mut() {
                operator.process_fs_write_response(request)?;
            }
            Ok(())
        });
    }

    /// Notify the operator that requested it that one of its HTTP uploads has
    /// completed.
    pub fn process_http_upload_response(
        &mut self,
        node_id: Identifier,
        request: HttpUploadRequest,
    ) {
        self.try_run(|d| {
            let node = d.graph.node_mut(node_id);
            if let Some(operator) = node.operator.as_deref_mut() {
                operator.process_http_upload_response(request)?;
            }
            Ok(())
        });
    }

    /// Run one unit of CPU work, traversing leaf-to-root (downstream first for cache locality).
    /// Returns [`WorkStatus::Ran`] if any operator did work.
    pub fn run_ready_cpu_work(&mut self, requester: &RefCell<IORequester>) -> WorkStatus {
        self.try_run_or(WorkStatus::Ran, |d| {
            let mut io = OperatorIO::new(requester, d.id, 0, &mut d.stats, &mut d.next_request_id);
            d.graph
                .traverse_backwards(|node_id, operator| {
                    io.set_operator_idx(node_id);
                    let status = operator.run_cpu_work(&mut io)?;
                    match status {
                        WorkStatus::Pending => Ok(ControlFlow::Continue(())),
                        WorkStatus::Ran => Ok(ControlFlow::Break(())),
                    }
                })
                .map(|c| match c {
                    ControlFlow::Continue(_) => WorkStatus::Pending,
                    ControlFlow::Break(_) => WorkStatus::Ran,
                })
        })
    }

    /// Ask this dataflow's operators to spill up to `count` ring slots' worth of
    /// buffers into `spill_directory`. Returns how many slots they started
    /// spilling, which is less than `count` when the operators had less to
    /// give.
    pub fn spill(
        &mut self,
        requester: &RefCell<IORequester>,
        spill_directory: &SpillDirectory,
        count: usize,
    ) -> usize {
        self.try_run_or(0, |d| {
            if d.graph.spilled_into.is_none() {
                d.graph.spilled_into = Some(spill_directory.clone());
            }
            let io = OperatorIO::new(requester, d.id, 0, &mut d.stats, &mut d.next_request_id);
            let mut spiller = Spiller::new(io, spill_directory);
            // The walk stops once the count is met; only the count matters here.
            let _: ControlFlow<()> = d.graph.traverse_backwards(|node_id, operator| {
                spiller.set_operator_idx(node_id);
                let remaining = count - spiller.buffers_spilled();
                operator.spill(&mut spiller, remaining)?;
                Ok(if spiller.buffers_spilled() >= count {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                })
            })?;
            Ok(spiller.buffers_spilled())
        })
    }

    /// Have one operator issue one more piece of remote IO, asking those that
    /// can in the order CPU work visits them. Returns [`WorkStatus::Ran`] if
    /// one did. A dataflow with no such operator (the common case) costs a
    /// single `is_empty` check.
    pub fn saturate_available_io(&mut self, requester: &RefCell<IORequester>) -> WorkStatus {
        if self.io_saturators.is_empty() {
            return WorkStatus::Pending;
        }
        self.try_run_or(WorkStatus::Pending, |d| {
            let mut io = OperatorIO::new(requester, d.id, 0, &mut d.stats, &mut d.next_request_id);
            for &node_id in &d.io_saturators {
                io.set_operator_idx(node_id);
                let issued = d.graph.node_mut(node_id).run(&mut |_, operator| {
                    Ok(match operator.saturate_available_io(&mut io)? {
                        WorkStatus::Ran => ControlFlow::Break(()),
                        WorkStatus::Pending => ControlFlow::Continue(()),
                    })
                })?;
                if matches!(issued, Some(ControlFlow::Break(()))) {
                    return Ok(WorkStatus::Ran);
                }
            }
            Ok(WorkStatus::Pending)
        })
    }

    /// Attempt to steal work from peer workers, traversing root-to-leaf (upstream first
    /// so the original worker's downstream data stays hot).
    pub fn try_stealing_work(&mut self, requester: &RefCell<IORequester>) -> WorkStatus {
        self.try_run_or(WorkStatus::Ran, |d| {
            let mut io = OperatorIO::new(requester, d.id, 0, &mut d.stats, &mut d.next_request_id);
            d.graph
                .traverse_forwards(|node_id, operator| {
                    io.set_operator_idx(node_id);
                    let status = operator.try_steal_work(&mut io)?;
                    match status {
                        WorkStatus::Pending => Ok(ControlFlow::Continue(())),
                        WorkStatus::Ran => Ok(ControlFlow::Break(())),
                    }
                })
                .map(|c| match c {
                    ControlFlow::Continue(_) => WorkStatus::Pending,
                    ControlFlow::Break(_) => WorkStatus::Ran,
                })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::TestSpillDirectory;
    use std::sync::Mutex;
    use std::sync::atomic::AtomicUsize;

    /// A test operator that always reports it `Ran` and never finishes on its
    /// own, so a node still holding one is visibly "live".
    struct LiveOperator;
    impl Operator for LiveOperator {
        fn run_cpu_work(&mut self, _io: &mut OperatorIO) -> crate::operations::Result<WorkStatus> {
            Ok(WorkStatus::Ran)
        }
        fn process_fs_write_response(
            &mut self,
            _request: FsWriteRequest,
        ) -> crate::operations::Result<()> {
            Ok(())
        }
        fn process_http_upload_response(
            &mut self,
            _request: HttpUploadRequest,
        ) -> crate::operations::Result<()> {
            Ok(())
        }
        fn try_finish(&mut self) -> crate::operations::Result<FinishStatus> {
            Ok(FinishStatus::Pending)
        }
    }

    /// Reports a fixed [`FinishStatus`], and optionally raises a flag when it is
    /// dropped so a test can watch for the node letting go of it.
    struct FinishOperator {
        status: FinishStatus,
        dropped: Option<Arc<AtomicBool>>,
    }

    impl FinishOperator {
        fn new(status: FinishStatus) -> Self {
            Self {
                status,
                dropped: None,
            }
        }

        fn signalling_drop(status: FinishStatus, dropped: Arc<AtomicBool>) -> Self {
            Self {
                status,
                dropped: Some(dropped),
            }
        }
    }

    impl Drop for FinishOperator {
        fn drop(&mut self) {
            if let Some(dropped) = &self.dropped {
                dropped.store(true, Ordering::Release);
            }
        }
    }

    impl Operator for FinishOperator {
        fn run_cpu_work(&mut self, _io: &mut OperatorIO) -> crate::operations::Result<WorkStatus> {
            Ok(WorkStatus::Pending)
        }
        fn process_fs_write_response(
            &mut self,
            _request: FsWriteRequest,
        ) -> crate::operations::Result<()> {
            Ok(())
        }
        fn process_http_upload_response(
            &mut self,
            _request: HttpUploadRequest,
        ) -> crate::operations::Result<()> {
            Ok(())
        }
        fn try_finish(&mut self) -> crate::operations::Result<FinishStatus> {
            Ok(self.status)
        }
    }

    /// Spills up to `capacity` slots per request and records each count asked.
    struct SpillingOperator {
        capacity: usize,
        asked: Arc<Mutex<Vec<usize>>>,
    }
    impl Operator for SpillingOperator {
        fn run_cpu_work(&mut self, _io: &mut OperatorIO) -> crate::operations::Result<WorkStatus> {
            Ok(WorkStatus::Pending)
        }
        fn process_fs_write_response(
            &mut self,
            _request: FsWriteRequest,
        ) -> crate::operations::Result<()> {
            Ok(())
        }
        fn process_http_upload_response(
            &mut self,
            _request: HttpUploadRequest,
        ) -> crate::operations::Result<()> {
            Ok(())
        }
        fn try_finish(&mut self) -> crate::operations::Result<FinishStatus> {
            Ok(FinishStatus::Pending)
        }
        fn spill(&mut self, spiller: &mut Spiller, count: usize) -> crate::operations::Result<()> {
            self.asked.lock().unwrap().push(count);
            spiller.count_buffers_spilled_without_writing(count.min(self.capacity));
            Ok(())
        }
    }

    #[test]
    fn spill_asks_the_output_end_first_and_moves_upstream_until_met() {
        let spill_directory = TestSpillDirectory::new();
        let asked: Vec<Arc<Mutex<Vec<usize>>>> = (0..3).map(|_| Arc::default()).collect();
        let operators: Vec<Box<dyn Operator>> = [10, 10, 3]
            .into_iter()
            .zip(&asked)
            .map(|(capacity, asked)| {
                Box::new(SpillingOperator {
                    capacity,
                    asked: asked.clone(),
                }) as Box<dyn Operator>
            })
            .collect();
        let (err_tx, _err_rx) = mpsc::channel();
        let (stats_tx, _stats_rx) = mpsc::channel();
        let mut flow = DataFlow::new(
            0,
            Arc::new(AtomicBool::new(false)),
            err_tx,
            operators,
            vec![None, None, None],
            vec![(0, 1), (1, 2)],
            HashMap::default(),
            stats_tx,
            false,
        );

        let spilled = flow.spill(&RefCell::new(IORequester::default()), &spill_directory, 5);

        assert_eq!(spilled, 5);
        assert_eq!(*asked[2].lock().unwrap(), [5]);
        assert_eq!(*asked[1].lock().unwrap(), [2]);
        assert!(asked[0].lock().unwrap().is_empty());
    }

    /// A dataflow of one operator that can spill `capacity` buffers.
    fn dataflow_with_one_spilling_operator(id: Identifier, capacity: usize) -> DataFlow {
        let (err_tx, _err_rx) = mpsc::channel();
        let (stats_tx, _stats_rx) = mpsc::channel();
        DataFlow::new(
            id,
            Arc::new(AtomicBool::new(false)),
            err_tx,
            vec![Box::new(SpillingOperator {
                capacity,
                asked: Arc::default(),
            })],
            vec![None],
            Vec::new(),
            HashMap::default(),
            stats_tx,
            false,
        )
    }

    #[test]
    fn a_dataflow_that_never_spilled_leaves_the_spill_directory_alone_when_dropped() {
        let spill_directory = TestSpillDirectory::new();
        let dataflow_id = 10;
        let directory = spill_directory.operator_directory(dataflow_id, WORKER_IDX.get(), 0);
        std::fs::create_dir_all(&directory).unwrap();
        let flow = dataflow_with_one_spilling_operator(dataflow_id, 1);

        drop(flow);

        assert!(directory.exists());
    }

    #[test]
    fn a_dataflow_that_spilled_removes_its_spill_directory_when_dropped() {
        let spill_directory = TestSpillDirectory::new();
        let dataflow_id = 11;
        let directory = spill_directory.operator_directory(dataflow_id, WORKER_IDX.get(), 0);
        std::fs::create_dir_all(&directory).unwrap();
        let mut flow = dataflow_with_one_spilling_operator(dataflow_id, 1);
        flow.spill(&RefCell::new(IORequester::default()), &spill_directory, 1);

        drop(flow);

        assert!(!directory.exists());
    }

    struct CountingOperator(Arc<AtomicUsize>);
    impl Operator for CountingOperator {
        fn run_cpu_work(&mut self, _io: &mut OperatorIO) -> crate::operations::Result<WorkStatus> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(WorkStatus::Ran)
        }
        fn process_fs_write_response(
            &mut self,
            _request: FsWriteRequest,
        ) -> crate::operations::Result<()> {
            Ok(())
        }
        fn process_http_upload_response(
            &mut self,
            _request: HttpUploadRequest,
        ) -> crate::operations::Result<()> {
            Ok(())
        }
        fn try_finish(&mut self) -> crate::operations::Result<FinishStatus> {
            Ok(FinishStatus::Pending)
        }
    }

    /// Build a linear chain `0 -> 1 -> ... -> n-1` of [`LiveOperator`]s, where
    /// node 0 is the source and node n-1 is the leaf.
    fn live_chain(n: usize) -> OperatorGraph {
        let operators: Vec<Box<dyn Operator>> = (0..n)
            .map(|_| Box::new(LiveOperator) as Box<dyn Operator>)
            .collect();
        let publisher_to_subscriber: Vec<(Identifier, Identifier)> =
            (0..n - 1).map(|i| (i, i + 1)).collect();
        OperatorGraph::from_edges(
            0,
            false,
            operators,
            publisher_to_subscriber,
            HashMap::default(),
        )
    }

    /// An abandoned node has given up its operator; a node still holding a
    /// [`LiveOperator`] has not.
    fn is_abandoned(graph: &OperatorGraph, idx: usize) -> bool {
        graph.graph.node(idx).is_finished()
    }

    #[test]
    fn a_node_lets_go_of_its_operator_once_it_finishes() {
        let dropped = Arc::new(AtomicBool::new(false));
        let operators: Vec<Box<dyn Operator>> = vec![Box::new(FinishOperator::signalling_drop(
            FinishStatus::Done,
            dropped.clone(),
        ))];
        let mut graph =
            OperatorGraph::from_edges(0, false, operators, Vec::new(), HashMap::default());

        assert_eq!(graph.try_finish().unwrap(), FinishStatus::Done);

        assert!(dropped.load(Ordering::Acquire));
    }

    #[test]
    fn finishing_pending_root_does_not_block_other_roots() {
        let operators: Vec<Box<dyn Operator>> = vec![
            Box::new(FinishOperator::new(FinishStatus::Pending)),
            Box::new(FinishOperator::new(FinishStatus::Done)),
        ];
        let mut graph =
            OperatorGraph::from_edges(0, false, operators, Vec::new(), HashMap::default());

        assert_eq!(graph.try_finish().unwrap(), FinishStatus::Pending);
        assert!(!graph.graph.node(0).is_finished());
        assert!(graph.graph.node(1).is_finished());
    }

    #[test]
    fn closed_gate_blocks_cpu_work_until_published() {
        let runs = Arc::new(AtomicUsize::new(0));
        let gate = Arc::new(AtomicBool::new(false));
        let mut gates = HashMap::default();
        gates.insert(0, vec![gate.clone()]);
        let (err_tx, _err_rx) = mpsc::channel();
        let (stats_tx, _stats_rx) = mpsc::channel();
        let mut flow = DataFlow::new(
            0,
            Arc::new(AtomicBool::new(false)),
            err_tx,
            vec![Box::new(CountingOperator(runs.clone()))],
            vec![None],
            Vec::new(),
            gates,
            stats_tx,
            false,
        );

        let requester = RefCell::new(IORequester::default());
        assert!(matches!(
            flow.run_ready_cpu_work(&requester),
            WorkStatus::Pending
        ));
        assert_eq!(runs.load(Ordering::Relaxed), 0);

        gate.store(true, Ordering::Release);
        assert!(matches!(
            flow.run_ready_cpu_work(&requester),
            WorkStatus::Ran
        ));
        assert_eq!(runs.load(Ordering::Relaxed), 1);
        assert!(flow.graph.graph.node(0).gates.is_empty());
    }

    #[test]
    fn root_waits_for_all_gates_before_finishing() {
        let first = Arc::new(AtomicBool::new(false));
        let second = Arc::new(AtomicBool::new(false));
        let mut gates = HashMap::default();
        gates.insert(0, vec![first.clone(), second.clone()]);
        let operators: Vec<Box<dyn Operator>> =
            vec![Box::new(FinishOperator::new(FinishStatus::Done))];
        let mut graph = OperatorGraph::from_edges(0, false, operators, Vec::new(), gates);

        assert_eq!(graph.try_finish().unwrap(), FinishStatus::Pending);
        first.store(true, Ordering::Release);
        assert_eq!(graph.try_finish().unwrap(), FinishStatus::Pending);
        second.store(true, Ordering::Release);
        assert_eq!(graph.try_finish().unwrap(), FinishStatus::Done);
    }

    #[test]
    fn abandon_ancestors_tears_down_upstream_only() {
        // Chain: 0 (source) -> 1 -> 2 (limit) -> 3 (downstream, e.g. GROUP BY).
        let mut graph = live_chain(4);

        graph.abandon_ancestors(2);

        // Everything upstream of the limit is abandoned and freed.
        assert!(is_abandoned(&graph, 0));
        assert!(is_abandoned(&graph, 1));
        // The limit and everything downstream keep running.
        assert!(!is_abandoned(&graph, 2));
        assert!(!is_abandoned(&graph, 3));
    }

    #[test]
    fn abandoning_one_reader_spares_a_publisher_the_other_still_needs() {
        // Node 0's output is read by both 1 and 2, as a CTE's definition is read
        // by each of its scans.
        let operators: Vec<Box<dyn Operator>> = (0..3)
            .map(|_| Box::new(LiveOperator) as Box<dyn Operator>)
            .collect();
        let mut graph = OperatorGraph::from_edges(
            0,
            false,
            operators,
            vec![(0, 1), (0, 2)],
            HashMap::default(),
        );

        graph.abandon_ancestors(1);

        assert!(!is_abandoned(&graph, 0), "node 2 is still reading it");
        assert!(!is_abandoned(&graph, 2));
    }

    #[test]
    fn abandon_ancestors_of_leaf_spares_nothing_downstream() {
        // The whole chain is upstream of the leaf, so all but the leaf go.
        let mut graph = live_chain(3);

        graph.abandon_ancestors(2);

        assert!(is_abandoned(&graph, 0));
        assert!(is_abandoned(&graph, 1));
        assert!(!is_abandoned(&graph, 2));
    }
}
