use crate::api::BuildContext;
use crate::operations::channels::{
    ChannelFactory, RootChannelFactory, Sender, StealableChannelFactory, mpsc_channel, stealable,
};
use crate::operations::{
    ChannelInputSender, ChannelSourceFactory, DefaultUnaryFactory, Forward, InjectorSourceFactory,
    MapFactory, RootUnaryOperatorFactory, UnaryFactory, UnaryOperatorFactory,
};
use crate::{DataFlowBuilder, DataFlowDispatcher, DataFlowHandle, OperatorGraphBuilder};
use arrow_array::RecordBatch;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize};

/// Generic, strongly-typed spec for building multi-stage pipelines.
///
/// Holds one factory per worker, where the factory type `OF` carries the full nested
/// generic chain (e.g. `UnaryOperatorFactory<..., UnaryOperatorFactory<..., OF>>`).
/// This is needed when a pipeline flows through non-RecordBatch types — e.g.
/// `catalog`'s Parquet reader, whose decode stages pass format-specific buffers
/// between them before producing `RecordBatch`es, where each
/// stage uses a different channel/sender type — including `WorkerAwareSender` which
/// requires `O: WorkerIdOutput`.
///
/// The spec stays strongly typed, but [`OperatorFactory::build`] takes its sender as
/// `Box<dyn Sender<O>>`, so the trait is object-safe and `Box<dyn OperatorFactory<O>>`
/// works directly. Every sender kind goes through the one method, including
/// `WorkerAwareSender<O>`: its `Sender` impl requires `O: WorkerIdOutput`, and the
/// caller that constructs it already satisfies that, so the bound never has to appear
/// on the trait.
pub struct OperatorSpec<O, OF: OperatorFactory<O>> {
    dispatcher: DataFlowDispatcher,
    factories: VecDeque<OF>,
    _phantom: std::marker::PhantomData<O>,
}

impl<O, OF: OperatorFactory<O>> OperatorSpec<O, OF> {
    pub fn new(dispatcher: DataFlowDispatcher, factories: impl IntoIterator<Item = OF>) -> Self {
        Self {
            dispatcher,
            factories: factories.into_iter().collect(),
            _phantom: Default::default(),
        }
    }

    pub fn dispatcher(&self) -> &DataFlowDispatcher {
        &self.dispatcher
    }

    pub fn into_parts(self) -> (DataFlowDispatcher, VecDeque<OF>) {
        (self.dispatcher, self.factories)
    }

    pub fn factories(self) -> VecDeque<OF> {
        self.factories
    }
}

impl<O: Send + 'static, OF: OperatorFactory<O> + Send + 'static> OperatorSpec<O, OF> {
    pub fn execute(self) -> DataFlowHandle<O> {
        self.execute_inner(false)
    }

    /// Like [`execute`](Self::execute) but with per-dataflow stats collection
    /// enabled; read the aggregated tally back via
    /// [`DataFlowHandle::collect_with_stats`](crate::DataFlowHandle::collect_with_stats).
    pub fn execute_with_stats(self) -> DataFlowHandle<O> {
        self.execute_inner(true)
    }

    fn execute_inner(self, collect_stats: bool) -> DataFlowHandle<O> {
        let (tx, rx) = mpsc_channel();
        let (err_tx, err_rx) = std::sync::mpsc::channel();
        let (stats_tx, stats_rx) = std::sync::mpsc::channel();
        let cancelled = Arc::new(AtomicBool::new(false));
        #[cfg(feature = "perf")]
        let profiled = self.dispatcher.profiled();
        let wakers = self
            .dispatcher
            .push_data_flow(self.factories.into_iter().map(|f| {
                let tx = tx.clone();
                let build = Box::new(move |context: &mut BuildContext| {
                    Box::new(f).build(Box::new(tx), context)
                });
                let builder = DataFlowBuilder::new(
                    build,
                    cancelled.clone(),
                    err_tx.clone(),
                    stats_tx.clone(),
                    collect_stats,
                );
                #[cfg(feature = "perf")]
                let builder = builder.with_profiling(profiled);
                builder
            }));
        // Close our local copies of the senders so the channels close once
        // every worker drops theirs.
        drop(err_tx);
        drop(stats_tx);

        let (rx, _) = rx.into_parts();
        DataFlowHandle::new(rx, err_rx, stats_rx, cancelled, wakers)
    }

    /// Run the dataflow and drain every produced item into a `Vec`. Shortcut
    /// for `self.execute().collect()`.
    pub fn collect(self) -> crate::data_flow::Result<Vec<O>> {
        self.execute().collect()
    }

    /// Append one unary stage to the pipeline: one operator per worker, the
    /// `i`th reading through `channels[i]` and transforming with `unaries[i]`.
    /// Both vecs must have one entry per worker.
    ///
    /// This is the general stage-builder that out-of-crate pipelines (e.g.
    /// `catalog`'s Parquet reader) compose, and that [`map_each`](Self::map_each)
    /// is a special case of. Choosing
    /// the channel kind lets a stage fan out work ([`stealable`]) or pin items
    /// to a worker (`return_to_worker_mpsc`); per-worker `unaries` let a
    /// pipeline breaker give one worker a distinct role (e.g. the receiver end
    /// of a single-worker merge). The `siblings_left` finishing counter is set up for
    /// you.
    pub fn chain<O2, UF, C>(
        self,
        channels: Vec<C>,
        unaries: Vec<UF>,
    ) -> OperatorSpec<O2, UnaryOperatorFactory<O, O2, UF, C, OF>>
    where
        O2: Send + 'static,
        UF: UnaryFactory<O, O2>,
        C: ChannelFactory<O>,
    {
        let siblings_left = Arc::new(AtomicUsize::new(self.factories.len()));
        let factories: Vec<_> = channels
            .into_iter()
            .zip(unaries)
            .zip(self.factories)
            .map(|((channel, unary), head)| {
                UnaryOperatorFactory::new(head, unary, channel, siblings_left.clone())
            })
            .collect();
        OperatorSpec::new(self.dispatcher, factories)
    }

    /// Append a parallel 1→1 map stage: every item is transformed by `f` on
    /// whatever worker handles it, and an item `f` fails on fails the
    /// dataflow. The stage reads through a work-stealing channel, so items
    /// rebalance across idle workers — the basis of a perfectly parallel job
    /// pipeline (e.g. [`values_input`] → `map_each` to encode one Parquet page
    /// per item).
    ///
    /// A thin wrapper over [`chain`](Self::chain): a [`stealable`] channel and
    /// `f` cloned once per worker (so `f` must be `Clone` — capture only
    /// cheap/shared state).
    #[allow(clippy::type_complexity)] // the fully-spelled builder factory type is the point
    pub fn map_each<O2, F>(
        self,
        f: F,
    ) -> OperatorSpec<O2, UnaryOperatorFactory<O, O2, MapFactory<F>, StealableChannelFactory<O>, OF>>
    where
        O2: Send + 'static,
        F: FnMut(O) -> crate::UnaryResult<O2> + Clone + Send + 'static,
    {
        let worker_count = self.factories.len();
        let channels: Vec<_> = stealable::<O>(self.dispatcher.topology())
            .into_iter()
            .collect();
        let unaries: Vec<_> = (0..worker_count).map(|_| MapFactory(f.clone())).collect();
        self.chain(channels, unaries)
    }
}

/// Source: stream a fixed, in-memory set of values across the worker pool.
///
/// The items are loaded into a shared work-stealing queue and fan out across
/// all workers exactly like a table scan distributes row groups — the
/// in-memory, any-type counterpart of `table_input`.
/// Chain [`map_each`](OperatorSpec::map_each) to process each item in parallel.
#[allow(clippy::type_complexity)]
pub fn values_input<T: Send + 'static>(
    dispatcher: &DataFlowDispatcher,
    items: impl IntoIterator<Item = T>,
) -> OperatorSpec<
    T,
    RootUnaryOperatorFactory<T, T, DefaultUnaryFactory<Forward<T>>, InjectorSourceFactory<T>>,
> {
    source_input(dispatcher, InjectorSourceFactory::new(items))
}

/// One [`Forward`] root per worker over a cloned `source`, sharing the sibling
/// counter the finish protocol needs: the assembly every stolen-item source
/// (in-memory or producer-fed) fans out with.
#[allow(clippy::type_complexity)]
fn source_input<T: Send + 'static, S: RootChannelFactory<T> + Clone>(
    dispatcher: &DataFlowDispatcher,
    source: S,
) -> OperatorSpec<T, RootUnaryOperatorFactory<T, T, DefaultUnaryFactory<Forward<T>>, S>> {
    let worker_count = dispatcher.worker_count();
    let siblings_left = Arc::new(AtomicUsize::new(worker_count));
    let factories: Vec<_> = (0..worker_count)
        .map(|_| {
            RootUnaryOperatorFactory::new(
                DefaultUnaryFactory::<Forward<T>>::new(),
                source.clone(),
                siblings_left.clone(),
            )
        })
        .collect();
    OperatorSpec::new(dispatcher.clone(), factories)
}

/// Source: stream producer-fed values across the worker pool.
///
/// The open-ended counterpart of [`values_input`]: the item set is not known
/// up front, so the returned [`ChannelInputSender`] feeds the queue while the
/// dataflow runs and [`close`](ChannelInputSender::close)s it when the stream
/// ends. Capacity bounds how many unclaimed items may queue; a refused send
/// hands the item back, and `on_claim` fires on every claim so the producer
/// can wake and retry.
#[allow(clippy::type_complexity)]
pub fn channel_input<T: Send + 'static>(
    dispatcher: &DataFlowDispatcher,
    capacity: usize,
    on_claim: Box<dyn Fn() + Send + Sync>,
) -> (
    ChannelInputSender<T>,
    OperatorSpec<
        T,
        RootUnaryOperatorFactory<T, T, DefaultUnaryFactory<Forward<T>>, ChannelSourceFactory<T>>,
    >,
) {
    let (sender, source) =
        ChannelSourceFactory::new(capacity, on_claim, dispatcher.waker_set.clone());
    (sender, source_input(dispatcher, source))
}

impl<OF: OperatorFactory<RecordBatch> + 'static> OperatorSpec<RecordBatch, OF> {
    /// Re-enter [`RecordBatchOperatorSpec`](super::record_batch_operator::RecordBatchOperatorSpec) so the RB-only fluent methods
    /// (`.aggregate()`, `.order_by_limit()`, `.group_by_aggregate()`, …) can be
    /// chained after a `.map()` whose output type is `RecordBatch`.
    pub fn record_batches(self) -> super::record_batch_operator::RecordBatchOperatorSpec {
        super::record_batch_operator::RecordBatchOperatorSpec::from_spec(self)
    }
}

/// A factory that builds one worker's operator chain, given an output sender.
///
/// Called on the worker thread during [`DataFlowBuilder::build`](crate::api::DataFlowBuilder::build).
/// The `build` method consumes the factory, creates channels between stages, and returns
/// an [`OperatorGraphBuilder`] ready to finalize and execute.
///
/// Stages connect via different sender types (work-stealing, mpsc, worker-aware), so
/// `build` takes the sender as a trait object rather than a type parameter. That keeps
/// the trait object-safe, and it means an operator is compiled once rather than once per
/// sender type it happens to be built with. A sender is used per batch, so the indirect
/// call costs nothing next to producing the batch.
pub trait OperatorFactory<O>: Send {
    /// Build the operator chain, outputting to `sender`.
    fn build(
        self: Box<Self>,
        sender: Box<dyn Sender<O>>,
        context: &mut BuildContext,
    ) -> OperatorGraphBuilder;
}

/// A boxed factory is itself a factory, so an already-erased head can be handed
/// straight to a stage that wants one by value.
impl<O> OperatorFactory<O> for Box<dyn OperatorFactory<O>> {
    fn build(
        self: Box<Self>,
        sender: Box<dyn Sender<O>>,
        context: &mut BuildContext,
    ) -> OperatorGraphBuilder {
        (*self).build(sender, context)
    }
}
