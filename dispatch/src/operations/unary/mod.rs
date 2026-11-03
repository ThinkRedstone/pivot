//! Unary operators: one input channel, one output channel.
//!
//! A unary operator reads items of type `I` from its input channel, applies a
//! transform, and writes items of type `O` to its output channel. Most query stages
//! are unary: filter, project, count, order-by-limit, and group-by. The Parquet
//! decode pipeline in `catalog` (indexer, decompressor, decoder) is built from
//! these same primitives.
//!
//! # Key types
//!
//! - [`Unary<I, O>`] — The transform trait. Implementors define [`consume`](Unary::consume)
//!   (process one input item) and optionally [`finish`](Unary::finish) (emit final results
//!   after all input is drained, e.g. aggregation output).
//!
//! - [`UnaryOperator<I, O, U, R>`] — The [`Operator`] implementation that wires a
//!   [`Unary`] to a receiver and sender. Created during the factory build step on the
//!   worker thread.
//!
//! - [`UnaryFactory<I, O>`] — Trait for creating a [`Unary`] instance. Each factory is
//!   consumed once per worker to produce the unary transform for that worker.
//!
//! - [`UnaryOperatorFactory<I, O, UF, C, OP>`] — An [`OperatorFactory`](crate::api::OperatorFactory)
//!   that chains a head factory (`OP`) with a [`UnaryFactory`] (`UF`) and a
//!   [`ChannelFactory`](super::channels::ChannelFactory) (`C`). When built, it creates the
//!   channel, builds the head (passing it the channel's sender), and creates a
//!   `UnaryOperator` reading from the channel's receiver.
//!
//! # Finishing protocol
//!
//! Unary operators across workers coordinate finishing via a shared `siblings_left`
//! atomic counter. When a worker's input channel is empty, it decrements the counter.
//! Once all siblings have decremented (counter reaches zero), [`Unary::finish`] is called
//! — this is where aggregation operators (count, group-by, order-by-limit) emit their
//! final results.
//!
//! # Concrete unary transforms
//!
//! - [`FilterFactory`] — Keeps rows matching a boolean mask.
//! - [`MapFactory`] — Transforms each batch.
//! - [`OrderByLimitFactory`] — Top-N sort across workers.
//! - [`GroupFactory`] — Hash-based group-by with aggregation.

mod group;
pub use group::{
    AggregationKind, AggregationSlot, AggregationValue, Cell, Compiled, Count, CountSlot,
    CountValidSlot, Distinct, Dynamic, F64Cell, Fold, GroupFactory, GroupLimit,
    HashOnlyIntKeyExtractor, IntCell, IntKeyExtractor, IntPairKeyExtractor, IntRead,
    IntStrKeyExtractor, KeyExtractor, Max, MaxSlot, Min, MinSlot, NoRead, OpTuple, Read,
    RowKeyExtractor, RowKeySchema, StrMax, StrMin, StrRead, StringKeyExtractor, Sum, SumSlot,
    WideCell, WideSum,
};

#[cfg(any(test, feature = "test-util"))]
pub mod test_utils;

mod factory;
pub use factory::*;

use crate::data_flow::WorkStatus;
use crate::io::{FsWriteRequest, HttpUploadRequest, OperatorIO, ReadResponse};
use arrow_schema::ArrowError;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use thiserror::Error;

use super::channels::{Receiver, Sender};
use super::{FinishStatus, Operator};
use crate::waker::{waker_set, worker_waker};

mod pipeline_breaker;
pub use pipeline_breaker::{Consumer, Outputter, PipelineBreaker};

mod aggregate;
pub use aggregate::AggregateFactory;

pub mod filter;
pub use filter::FilterFactory;

mod map;
pub use map::MapFactory;

mod default_unary_factory;
pub use default_unary_factory::DefaultUnaryFactory;

mod copy_out;
mod join;
mod limit;
mod normalizer;
mod order_by;
mod order_by_limit;

pub use copy_out::copy_out;
pub(crate) use join::create_join_factories;
pub(crate) use join::create_normalizing_for_workers as create_normalizing_join_factories;
pub(crate) use join::create_range_join_factories;
pub use join::{
    DynamicRowKey, JoinBuildFilter, JoinKey, JoinKind, JoinRecordBatchOperatorFactory,
    JoinResidualFn, JoinResidualSpec, JoinSpec, KeyBitset, KeyBitsetSlot, PackedKey, RangeCompare,
    RangeJoinSpec, SingleColumnKey,
};
pub use limit::LimitFactory;
pub(crate) use normalizer::{
    BatchesOutputter, CollectorFactory, InitializableOutputter, NormalizationBatches, Normalizer,
};
pub(crate) use order_by::create_normalizing_for_workers as create_normalizing_order_by_factories;
pub use order_by::{
    KWayMergePlan, KWayMergeTask, LocatedBatch, MergeRun, MergedOutput, OrderByFactory,
    batch_sort_indices,
};
pub use order_by_limit::{OrderBy, OrderByLimitFactory};

#[derive(Debug, Error)]
pub enum Error {
    #[error("{0}")]
    Arrow(#[from] ArrowError),
    #[error("{0}")]
    Channel(#[from] super::channels::Error),
    #[error("{0}")]
    OrderByLimit(#[from] order_by_limit::Error),
    #[error("{0}")]
    Group(#[from] group::Error),
    #[error("{0}")]
    Io(#[from] crate::io::IORequesterError),
    /// An error from an operator defined outside this crate (e.g. the Parquet
    /// reader, now in `catalog`). Such operators map their own error into this.
    #[error("{0}")]
    Operator(Box<dyn std::error::Error + Send + Sync>),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// The core transform trait for unary operators.
///
/// `I` is the input type (read from the upstream channel), `O` is the output type
/// (written to the downstream channel). Most implementations are `RecordBatch -> RecordBatch`,
/// but the parquet pipeline uses other types (e.g. `CompressedPage -> DecompressedPage`).
pub trait Unary<I, O> {
    /// Process one input item, sending zero or more output items to `sender`.
    fn consume(&mut self, object: I, sender: &mut dyn Sender<O>, io: &mut OperatorIO)
    -> Result<()>;

    /// Whether this unary is ready to accept another input. Returns `false` when
    /// backpressured (e.g. waiting for IO to complete before consuming more) or
    /// while it holds a job it is still stepping through. A unary that is not
    /// ready is neither fed nor steals, but still gets [`run`](Self::run) each
    /// turn.
    fn ready_for_more_work(&mut self) -> bool {
        true
    }

    /// Handle a completed logical read. Dispatch has already resolved cache
    /// hits and committed any physical reads.
    fn process_read_response(
        &mut self,
        _sender: &mut dyn Sender<O>,
        _io: &mut OperatorIO,
        _response: ReadResponse,
    ) -> Result<()> {
        unreachable!()
    }

    fn process_fs_write_response(
        &mut self,
        _sender: &mut dyn Sender<O>,
        _request: FsWriteRequest,
    ) -> Result<()> {
        unreachable!()
    }

    /// Handle a completed HTTP upload.
    fn process_http_upload_response(
        &mut self,
        _sender: &mut dyn Sender<O>,
        _request: HttpUploadRequest,
    ) -> Result<()> {
        unreachable!()
    }

    /// Called on every turn that consumes no input: the input channel is empty,
    /// or the unary is not [ready](Self::ready_for_more_work) for more. Useful
    /// for operators that generate work independently of input (e.g. emitting
    /// buffered results, or advancing a held job by one step).
    fn run(&mut self, _sender: &mut dyn Sender<O>) -> Result<WorkStatus> {
        Ok(WorkStatus::Pending)
    }

    /// Called after all siblings have drained their input. Emit any final results
    /// (e.g. aggregation totals). Return `true` when fully done, `false` to be
    /// called again (e.g. if final output requires multiple steps).
    ///
    /// An important note here is that the siblings may not have finished processing there data, ie
    /// there may be live sibling workers doing consume while finish is being called. It is the
    /// responsibility of the specific operator to do any synchronizing necessary.
    ///
    /// Once finish is called, `consume` is guaranteed never to be called again.
    fn finish(&mut self, _sender: &mut dyn Sender<O>) -> Result<bool> {
        Ok(true)
    }

    /// Whether this transform is done consuming *before* its input has run dry,
    /// e.g. a `LIMIT` that has already buffered enough rows. The default is
    /// `false`: an operator finishes only when its input channel drains.
    ///
    /// When `true`, [`UnaryOperator::try_finish`] proceeds to finalization even
    /// though items may remain in the input channel; those leftovers are simply
    /// never consumed (the matching scan is abandoned, see
    /// [`Operator::upstream_cancel_flag`]).
    fn finished_consuming(&self) -> bool {
        false
    }

    /// Whether this operator has async work still outstanding that its
    /// [`finish`](Self::finish) depends on — e.g. writes/uploads submitted to the
    /// ring whose completions haven't landed yet. The default is `false`.
    ///
    /// While `true`, the operator is not counted as done at the cross-worker
    /// finish barrier, so `finish` runs only once *every* worker's async work has
    /// settled — letting `finish` read state that those completions produce (e.g.
    /// a shared row-count total). The worker keeps servicing the ring in the
    /// meantime, so the outstanding work still makes progress.
    fn has_pending_work(&self) -> bool {
        false
    }

    /// See [`Operator::upstream_cancel_flag`].
    /// Wired through the wrapping [`UnaryOperator`] so a transform can ask the
    /// dataflow to abandon its upstream.
    fn upstream_cancel_flag(&self) -> Option<Arc<AtomicBool>> {
        None
    }
}

/// Wires a [`Unary`] transform to a receiver (input channel) and sender (output channel),
/// implementing the [`Operator`] trait so it can be driven by the worker's event loop.
///
/// Created during the factory build step — not constructed directly. See
/// [`UnaryOperatorFactory`] in the [`factory`] module.
pub struct UnaryOperator<I, O, U: Unary<I, O>, R: Receiver<I>> {
    unary: U,
    receiver: R,
    sender: Box<dyn Sender<O>>,
    notified_finished: bool,
    siblings_left: Arc<AtomicUsize>,
    _phantom: PhantomData<(I, O)>,
}

impl<I, O, U: Unary<I, O>, R: Receiver<I>> UnaryOperator<I, O, U, R> {
    pub fn new(
        unary: U,
        receiver: R,
        sender: Box<dyn Sender<O>>,
        siblings_left: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            unary,
            receiver,
            sender,
            notified_finished: false,
            siblings_left,
            _phantom: Default::default(),
        }
    }
}

impl<I, O, U: Unary<I, O>, IN: Receiver<I>> Operator for UnaryOperator<I, O, U, IN> {
    fn run_cpu_work(&mut self, io: &mut OperatorIO) -> super::Result<WorkStatus> {
        // A transform that is not taking input may still have work of its own
        // to advance, e.g. a job it steps through one batch per turn.
        if !self.unary.ready_for_more_work() {
            return Ok(self.unary.run(&mut *self.sender)?);
        }

        let item = match self.receiver.try_recv() {
            None => return Ok(self.unary.run(&mut *self.sender)?),
            Some(t) => t,
        };

        self.unary.consume(item, &mut *self.sender, io)?;

        Ok(WorkStatus::Ran)
    }

    fn process_read_response(
        &mut self,
        io: &mut OperatorIO,
        response: ReadResponse,
    ) -> super::Result<()> {
        Ok(self
            .unary
            .process_read_response(&mut *self.sender, io, response)?)
    }

    fn process_fs_write_response(&mut self, request: FsWriteRequest) -> super::Result<()> {
        Ok(self
            .unary
            .process_fs_write_response(&mut *self.sender, request)?)
    }

    fn process_http_upload_response(&mut self, request: HttpUploadRequest) -> super::Result<()> {
        Ok(self
            .unary
            .process_http_upload_response(&mut *self.sender, request)?)
    }

    fn try_finish(&mut self) -> super::Result<FinishStatus> {
        // Normally an operator finishes only once its input has drained. A
        // transform that has decided to stop early (e.g. a satisfied `LIMIT`)
        // signals `finished_consuming` so we proceed regardless: the unconsumed
        // tail belongs to a scan that's being abandoned anyway. An operator with
        // outstanding async work (e.g. in-flight uploads) is also not yet done,
        // so its completions land before the barrier lets `finish` run.
        let done_consuming = || {
            (self.receiver.is_empty() || self.unary.finished_consuming())
                && !self.unary.has_pending_work()
        };
        if !done_consuming() {
            return Ok(FinishStatus::Pending);
        }

        // Acquire/Release on the counter so the last-out worker's `finish` sees
        // every peer's pre-decrement writes (e.g. a shared row-count total each
        // worker's async completions add to), not just channel-delivered data.
        let ready = if self.notified_finished {
            self.siblings_left.load(Ordering::Acquire) == 0
        } else {
            self.notified_finished = true;
            let was_last = self.siblings_left.fetch_sub(1, Ordering::AcqRel) == 1;
            if was_last {
                // Sibling counter just hit 0: every worker's `try_finish` for this
                // operator can now run. Wake peers parked on any node's waker so
                // they advance to their `finish` instead of sleeping out the park.
                waker_set().notify_all();
            }
            was_last
        };

        if ready {
            // Race guard: a peer may have stolen work into our channel between
            // the emptiness check above and the decrement. Re-check and back
            // off if new work appeared (unless we're finishing early, in which
            // case we don't care about leftover input). Relaxed ordering
            // suffices because the channel ops themselves provide Acquire/Release.
            if !done_consuming() {
                self.notified_finished = false;
                self.siblings_left.fetch_add(1, Ordering::Relaxed);
                return Ok(FinishStatus::Pending);
            }

            if self.unary.finish(&mut *self.sender)? {
                // `finish` may have emitted final batches downstream, so wake
                // any parked peers to pick that work up.
                worker_waker().notify();
                return Ok(FinishStatus::Done);
            }
            // A pipeline breaker still draining its outputter. The worker re-
            // drives it through `run` next iteration; reporting `Working` keeps
            // the worker from parking in between (its finish pass doesn't set
            // `did_work`). No notify — the output's own `send`s wake peers.
            return Ok(FinishStatus::Working);
        }
        Ok(FinishStatus::Pending)
    }

    fn try_steal_work(&mut self, io: &mut OperatorIO) -> super::Result<WorkStatus> {
        if !self.unary.ready_for_more_work() {
            return Ok(WorkStatus::Pending);
        }
        match self.receiver.steal() {
            Some(s) => {
                self.unary.consume(s, &mut *self.sender, io)?;
                Ok(WorkStatus::Ran)
            }
            None => Ok(WorkStatus::Pending),
        }
    }

    fn upstream_cancel_flag(&self) -> Option<Arc<AtomicBool>> {
        self.unary.upstream_cancel_flag()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::TestOperatorIO;
    use crate::operations::channels::mpsc_channel;
    use test_utils::CollectSender;

    /// A transform holding a job it steps through: it takes no input, and each
    /// `run` is one step.
    struct HeldJob {
        steps: usize,
    }

    impl Unary<(), ()> for HeldJob {
        fn consume(&mut self, _: (), _: &mut dyn Sender<()>, _: &mut OperatorIO) -> Result<()> {
            unreachable!("a transform that is not ready is never fed")
        }

        fn ready_for_more_work(&mut self) -> bool {
            false
        }

        fn run(&mut self, _: &mut dyn Sender<()>) -> Result<WorkStatus> {
            self.steps += 1;
            Ok(WorkStatus::Ran)
        }
    }

    /// A transform that is not ready for input still gets its turn through
    /// `run`, and the input it declined stays queued.
    #[test]
    fn a_transform_not_ready_for_input_is_still_run() {
        let (mut input, receiver) = mpsc_channel::<()>();
        input.send(()).unwrap();
        let mut operator = UnaryOperator::new(
            HeldJob { steps: 0 },
            receiver,
            Box::new(CollectSender::<()>::new()),
            Arc::new(AtomicUsize::new(1)),
        );
        let mut test_io = TestOperatorIO::default();

        let status = operator.run_cpu_work(&mut test_io.io()).unwrap();

        assert!(matches!(status, WorkStatus::Ran));
        assert_eq!(operator.unary.steps, 1);
        assert!(!operator.receiver.is_empty());
    }
}
