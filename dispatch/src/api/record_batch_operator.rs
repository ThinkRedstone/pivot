//! A [`RecordBatchOperatorSpec`] represents a parallel operator running over Arrow
//! [`RecordBatch`]es. A dataflow is built with it in a "fluent API" style — start
//! with `table_input`, chain operations, and call [`.collect()`](RecordBatchOperatorSpec::collect)
//! to execute:
//!
//! ```ignore
//! # use std::sync::Arc;
//! # use arrow_array::{RecordBatch, StringViewArray};
//! # use dispatch::*;
//! # use dispatch::table_input;
//! # let table = Arc::new(ParquetTable::from_files(dispatcher, &["/tmp/data.parquet"], &[]).unwrap());
//! # let dispatch = Dispatch::spin_up(1, 32, None);
//! # let dispatcher = dispatch.dispatcher();
//! // SELECT COUNT(*) FROM events WHERE url LIKE '%google%'
//! let results = table_input(&dispatcher, &table, Projection::columns([0]), false)
//!     .filter(|| {
//!         let mut contains = Contains::new("google");
//!         move |batch: RecordBatch| {
//!             let col = batch.column(0).as_any()
//!                 .downcast_ref::<StringViewArray>().unwrap();
//!             let mask = contains.run(col);
//!             filter_record_batch(&batch, &mask).unwrap()
//!         }
//!     })
//!     .aggregate::<i64>(vec![AggregationSlot::new(AggregationKind::CountStar, 0, DataType::Int64)])
//!     .collect();
//! ```
//!
//! Operations like [`.filter()`](RecordBatchOperatorSpec::filter) and
//! [`.project()`](RecordBatchOperatorSpec::project) take a **builder closure**
//! (`Fn() -> F`) that is called once per worker thread, so each worker gets its own
//! independent operator instance with private mutable state — no synchronization needed.
//!
//! Nothing executes until [`.collect()`](RecordBatchOperatorSpec::collect) is called.
//! At that point the spec is shipped to worker threads, built into operator chains,
//! and run in parallel. Results are collected into `Vec<RecordBatch>`.
//!
//! # Internals
//!
//! Factories are stored as `Box<dyn OperatorFactory<RecordBatch>>` — an object-safe trait
//! that wraps the generic [`OperatorFactory<O>`]. This keeps the return type of every
//! chained method as plain `RecordBatchOperatorSpec`, rather than deeply nested generics.
//! See [`OperatorFactory`] and [`operator_spec`](super::operator_spec) for
//! details on why this split exists.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use arrow_array::ArrowNativeTypeOp;
use arrow_array::RecordBatch;
use arrow_schema::DataType;

use crate::api::operator_spec::{OperatorFactory, OperatorSpec};
use crate::boundary_slot::BoundarySlot;
use crate::operations::channels::{
    SingleWorkerMpscFactory, StealableChannelFactory, stealable, to_single_worker_mpsc,
};
use crate::operations::{
    AggregateFactory, AggregationSlot, AggregationValue, CteFactory, CteScanFactory, Distinct,
    DynamicRowKey, F64Cell, FilterFactory, GroupFactory, GroupLimit, IntCell, JoinKey, JoinKind,
    JoinRecordBatchOperatorFactory, JoinSpec, KeyExtractor, LimitFactory, MapFactory,
    NoOpNullaryFactory, NullaryFactory, NullaryOperatorFactory, OrderBy, OrderByFactory,
    OrderByLimitFactory, PackedKey, RangeJoinSpec, SingleColumnKey, UnaryFactory,
    UnaryOperatorFactory, WideCell, copy_out, create_join_factories,
    create_normalizing_join_factories, create_normalizing_order_by_factories,
    create_range_join_factories,
};
use crate::{DataFlowDispatcher, DataFlowHandle, DataFlowStats};
pub const RECORD_BATCH_SIZE: usize = 8192;

/// RecordBatchOperatorSpec represents a parallel operator running over Arrow [`RecordBatch`]es.
///
/// Holds one factory per worker, stored as `Box<dyn OperatorFactory<RecordBatch>>` to
/// erase the nested generic types that would otherwise grow with each chained operation. Factories
/// that implement `Send` are sent to workers (instead of the actual operations/channels) as the
/// final channel/operation may not be `Send`.
///
/// # Building a query
///
/// Start from a source — `values_input`, `from_nullary`, or a source built on
/// `OperatorSpec` such as `catalog`'s `table_input` — then chain operations. Each
/// method consumes `self` and returns a new `RecordBatchOperatorSpec`:
///
/// ```ignore
/// # use std::sync::Arc;
/// # use arrow_array::{RecordBatch, StringViewArray};
/// # use dispatch::*;
/// # use dispatch::table_input;
/// # let table = Arc::new(ParquetTable::from_files(dispatcher, &["/tmp/data.parquet"], &[]).unwrap());
/// # let dispatch = Dispatch::spin_up(1, 32, None);
/// # let dispatcher = dispatch.dispatcher();
/// let results = table_input(&dispatcher, &table, Projection::columns([0]), false)
///     .filter(|| {
///         let mut contains = Contains::new("google");
///         move |batch: &RecordBatch| {
///             let col = batch.column(0).as_any()
///                 .downcast_ref::<StringViewArray>().unwrap();
///             contains.run(col)
///         }
///     })
///     .aggregate::<i64>(vec![AggregationSlot::new(AggregationKind::CountStar, 0, DataType::Int64)])
///     .collect();
/// ```
///
/// # Closures are per-worker
///
/// Methods like [`filter`](Self::filter) and [`project`](Self::project) take a
/// **builder closure** (`Fn() -> F`) that is called once per worker to create
/// independent operator instances. This allows each worker's operator to hold mutable
/// state without synchronization:
///
/// ```ignore
/// # use std::sync::Arc;
/// # use arrow_array::{RecordBatch, StringViewArray};
/// # use dispatch::*;
/// # use dispatch::table_input;
/// # let table = Arc::new(ParquetTable::from_files(dispatcher, &["/tmp/data.parquet"], &[]).unwrap());
/// # let dispatch = Dispatch::spin_up(1, 32, None);
/// # let dispatcher = dispatch.dispatcher();
/// # let spec = table_input(&dispatcher, &table, Projection::columns([0]), false);
/// spec.filter(|| {
///     // This closure is called N times (once per worker).
///     // Each worker gets its own `Contains` instance.
///     let mut contains = Contains::new("google");
///
///     // Called once per RecordBatch on this worker.
///     move |batch: &RecordBatch| {
///         let col = batch.column(0).as_any()
///             .downcast_ref::<StringViewArray>().unwrap();
///         contains.run(col)
///     }
/// })
/// # ;
/// ```
pub struct RecordBatchOperatorSpec {
    dispatcher: DataFlowDispatcher,
    factories: VecDeque<Box<dyn OperatorFactory<RecordBatch>>>,
    /// Whether batches form one ordered stream. Downstream stages must then
    /// consume through FIFO rather than work-stealing queues, which may reorder
    /// batches even when every batch is internally sorted.
    stream_ordered: bool,
}

/// An owned batch representation that can leave a dispatch worker safely.
///
/// The conversion runs on the worker that produced the input. Implementations
/// must not retain ring-backed Arrow buffers; copy or encode everything kept in
/// the returned value.
pub trait OutputBatch: Sized + Send + 'static {
    fn from_record_batch(batch: RecordBatch) -> Self;
}

impl OutputBatch for RecordBatch {
    fn from_record_batch(batch: RecordBatch) -> Self {
        copy_out(batch).expect("copying a valid RecordBatch should succeed")
    }
}

/// Return type of [`RecordBatchOperatorSpec::map`]: a generic
/// [`OperatorSpec<T, _>`] wrapping the RB→T map stage.
type MapOperatorSpec<T, F> = OperatorSpec<
    T,
    UnaryOperatorFactory<
        RecordBatch,
        T,
        MapFactory<F>,
        StealableChannelFactory<RecordBatch>,
        Box<dyn OperatorFactory<RecordBatch>>,
    >,
>;

/// FIFO counterpart to [`MapOperatorSpec`], used when mapping a stream whose
/// batch order is already significant.
type OrderedMapOperatorSpec<T, F> = OperatorSpec<
    T,
    UnaryOperatorFactory<
        RecordBatch,
        T,
        MapFactory<F>,
        SingleWorkerMpscFactory<RecordBatch>,
        Box<dyn OperatorFactory<RecordBatch>>,
    >,
>;

impl RecordBatchOperatorSpec {
    /// Convert a generic [`OperatorSpec`] into a type-erased `RecordBatchOperatorSpec`.
    ///
    /// Used by builders such as `catalog`'s `table_input` / `materialize` to erase
    /// the concrete nested factory types a multi-stage pipeline produces.
    pub fn from_spec<OF: OperatorFactory<RecordBatch> + 'static>(
        spec: OperatorSpec<RecordBatch, OF>,
    ) -> Self {
        let (dispatcher, factories) = spec.into_parts();
        Self {
            dispatcher,
            factories: factories
                .into_iter()
                .map(|f| Box::new(f) as Box<dyn OperatorFactory<RecordBatch>>)
                .collect(),
            stream_ordered: false,
        }
    }

    /// Decompose into the dispatcher and per-worker factories, so out-of-crate
    /// code (e.g. late materialization in `catalog`) can chain further stages.
    pub fn into_parts(
        self,
    ) -> (
        DataFlowDispatcher,
        VecDeque<Box<dyn OperatorFactory<RecordBatch>>>,
    ) {
        (self.dispatcher, self.factories)
    }

    /// Build a `RecordBatchOperatorSpec` from per-worker nullary factories.
    ///
    /// Each factory is wrapped in a [`NullaryOperatorFactory`] and the actual nullary is
    /// built on the worker thread. This is useful for source-like or side-effect-only
    /// operators such as DDL.
    pub fn from_nullary<NF: NullaryFactory<RecordBatch>>(
        dispatcher: &DataFlowDispatcher,
        nullary_factories: impl IntoIterator<Item = NF>,
    ) -> Self {
        Self::from_spec(OperatorSpec::new(
            dispatcher.clone(),
            nullary_factories
                .into_iter()
                .map(NullaryOperatorFactory::new),
        ))
    }

    /// Build a spec that emits no rows and completes immediately: each worker
    /// gets a [`NoOpNullary`](crate::operations::NoOpNullary), which has no
    /// work and finishes on its first finish pass. This is the plan for a
    /// statement that turns out to have nothing to do.
    pub fn no_op(dispatcher: &DataFlowDispatcher) -> Self {
        Self::from_nullary(
            dispatcher,
            (0..dispatcher.worker_count()).map(|_| NoOpNullaryFactory),
        )
    }

    /// One scan site of a CTE: a source over every row the CTE's definition
    /// produces, in whatever order they arrive.
    ///
    /// A site reads its own channel, so several sites each see the full result
    /// rather than dividing it between them. Pair with
    /// [`with_cte`](Self::with_cte) on a spec containing the sites, which is
    /// what supplies the rows: on its own a site produces nothing and never
    /// finishes.
    pub fn cte_scan(dispatcher: &DataFlowDispatcher, cte_index: usize) -> Self {
        let siblings_left = Arc::new(AtomicUsize::new(dispatcher.worker_count()));
        let factories = stealable::<RecordBatch>(dispatcher.topology())
            .into_iter()
            .map(|channel_factory| {
                Box::new(CteScanFactory {
                    channel_factory,
                    cte_index,
                    siblings_left: siblings_left.clone(),
                }) as Box<dyn OperatorFactory<RecordBatch>>
            })
            .collect();
        Self {
            dispatcher: dispatcher.clone(),
            factories,
            stream_ordered: false,
        }
    }

    /// Run `definition` alongside this spec, feeding its rows to each of the
    /// `sites` [`cte_scan`](Self::cte_scan) sites for `cte_index` that this spec
    /// contains.
    ///
    /// The definition runs once however many sites read it, and runs
    /// concurrently with them: a site consumes batches as they are produced
    /// rather than waiting for the whole result.
    pub fn with_cte(
        self,
        definition: RecordBatchOperatorSpec,
        cte_index: usize,
        sites: usize,
    ) -> Self {
        let (_, definition_heads) = definition.into_parts();
        assert_eq!(
            self.factories.len(),
            definition_heads.len(),
            "a CTE and its definition must use the same worker count"
        );
        let factories = self
            .factories
            .into_iter()
            .zip(definition_heads)
            .map(|(body_head, definition_head)| {
                Box::new(CteFactory {
                    definition_head,
                    body_head,
                    cte_index,
                    sites,
                }) as Box<dyn OperatorFactory<RecordBatch>>
            })
            .collect();
        Self {
            dispatcher: self.dispatcher,
            factories,
            stream_ordered: self.stream_ordered,
        }
    }

    /// Borrow the dispatcher this spec was built against.
    pub fn dispatcher(&self) -> &DataFlowDispatcher {
        &self.dispatcher
    }

    /// Append a unary (one-in, one-out) stage to the dataflow.
    ///
    /// Takes an iterator of [`UnaryFactory`] instances (one per worker) and wraps
    /// each existing factory with a [`UnaryOperatorFactory`].
    fn unary<UF: UnaryFactory<RecordBatch, RecordBatch>>(
        self,
        unary_factories: impl IntoIterator<Item = UF>,
    ) -> Self {
        let siblings_left = Arc::new(AtomicUsize::new(self.worker_count()));
        let factories = if self.stream_ordered {
            let target = self.dispatcher.next_worker();
            to_single_worker_mpsc::<RecordBatch>(self.worker_count(), target)
                .into_iter()
                .zip(unary_factories)
                .zip(self.factories)
                .map(|((channel_factory, unary_factory), head)| {
                    Box::new(UnaryOperatorFactory::new(
                        head,
                        unary_factory,
                        channel_factory,
                        siblings_left.clone(),
                    )) as Box<dyn OperatorFactory<RecordBatch>>
                })
                .collect()
        } else {
            stealable::<RecordBatch>(self.dispatcher.topology())
                .into_iter()
                .zip(unary_factories)
                .zip(self.factories)
                .map(|((channel_factory, unary_factory), head)| {
                    Box::new(UnaryOperatorFactory::new(
                        head,
                        unary_factory,
                        channel_factory,
                        siblings_left.clone(),
                    )) as Box<dyn OperatorFactory<RecordBatch>>
                })
                .collect()
        };
        Self {
            dispatcher: self.dispatcher,
            factories,
            stream_ordered: self.stream_ordered,
        }
    }

    fn worker_count(&self) -> usize {
        self.factories.len()
    }

    /// Filter rows from each batch.
    ///
    /// Takes an **outer builder closure** (`FB`) that is called once per worker thread
    /// during setup. The builder returns an **inner closure** (`F`) that is called once
    /// per `RecordBatch` during execution, returning the batch of surviving rows. The
    /// inner closure also receives the operator'"'"'s [`SlabAllocator`](crate::memory::SlabAllocator)
    /// so survivors can be compacted into slab-backed buffers via
    /// [`take`](crate::arrays::take).
    ///
    /// This two-level pattern lets each worker own private mutable state (allocated in
    /// the builder):
    ///
    /// ```ignore
    /// # use std::sync::Arc;
    /// # use arrow_array::{RecordBatch, StringViewArray};
    /// # use dispatch::*;
    /// # use dispatch::table_input;
    /// # let table = Arc::new(ParquetTable::from_files(dispatcher, &["/tmp/data.parquet"], &[]).unwrap());
    /// # let dispatch = Dispatch::spin_up(1, 32, None);
    /// # let dispatcher = dispatch.dispatcher();
    /// # let spec = table_input(&dispatcher, &table, Projection::columns([0]), false);
    /// spec.filter(|| {
    ///     // Called once per worker — allocate per-worker state here.
    ///     let mut contains = Contains::new("google");
    ///
    ///     // Called once per RecordBatch on this worker.
    ///     move |batch: RecordBatch| {
    ///         let col = batch.column(0).as_any()
    ///             .downcast_ref::<StringViewArray>().unwrap();
    ///         let mask = contains.run(col);
    ///         filter_record_batch(&batch, &mask).unwrap()
    ///     }
    /// })
    /// # ;
    /// ```
    pub fn filter<F, FB>(self, builder: FB) -> Self
    where
        F: FnMut(
                &RecordBatch,
                &mut crate::memory::SlabAllocator,
                &mut crate::SelectionScratch,
            ) -> crate::RowSelection
            + Send
            + 'static,
        FB: Fn() -> F,
    {
        self.filter_with_delivery(builder, crate::RowDelivery::Coalesced)
    }

    /// [`filter`](Self::filter), choosing when survivors reach the operator
    /// below. Use [`RowDelivery::Immediate`](crate::RowDelivery::Immediate)
    /// when that operator acts on early rows, such as a LIMIT that cancels its
    /// input or a Top-N whose boundary prunes the scan; holding rows back to
    /// fill a batch defers the decision until the filter has selected a whole
    /// batch's worth, which under a selective filter reads far more of the
    /// table than the query needs.
    pub fn filter_with_delivery<F, FB>(self, builder: FB, delivery: crate::RowDelivery) -> Self
    where
        F: FnMut(
                &RecordBatch,
                &mut crate::memory::SlabAllocator,
                &mut crate::SelectionScratch,
            ) -> crate::RowSelection
            + Send
            + 'static,
        FB: Fn() -> F,
    {
        let worker_count = self.worker_count();
        self.unary((0..worker_count).map(|_| FilterFactory(builder(), delivery)))
    }

    /// Apply a per-batch 1→1 transform that can change the output type.
    ///
    /// Replaces both the old `project` (RB → RB) and the old `for_each`
    /// (RB → T) — one method, generic over the output type. The closure
    /// takes an owned `RecordBatch` and returns a single `T`; every `send`
    /// downstream is monomorphized over the resulting `OperatorSpec<T, _>`.
    ///
    /// Lifts out of `RecordBatchOperatorSpec` into the generic
    /// [`OperatorSpec<T, _>`]. To chain back into RB-only methods like
    /// `.aggregate()` / `.order_by_limit()`, call
    /// [`record_batches`](OperatorSpec::record_batches) on the result when
    /// `T = RecordBatch`.
    ///
    /// Same two-level closure pattern as [`filter`](Self::filter): the outer
    /// builder closure is called once per worker.
    ///
    /// # Example
    ///
    /// ```ignore
    /// # use std::sync::Arc;
    /// # use arrow_array::{Int64Array, RecordBatch};
    /// # use dispatch::*;
    /// # let table = Arc::new(ParquetTable::from_files(dispatcher, &["/tmp/data.parquet"], &[]).unwrap());
    /// # let dispatch = Dispatch::spin_up(1, 32, None);
    /// # let dispatcher = dispatch.dispatcher();
    /// # let spec = table_input(&dispatcher, &table, Projection::columns([0]), false);
    /// // RB → RB (the old `project`)
    /// spec.map(|| {
    ///     let indices = vec![0];
    ///     move |batch: RecordBatch| batch.project(&indices).unwrap()
    /// })
    /// # ;
    /// ```
    pub fn map<T, F, FB>(self, builder: FB) -> MapOperatorSpec<T, F>
    where
        T: Send + 'static,
        F: FnMut(RecordBatch) -> T + Send + 'static,
        FB: Fn() -> F,
    {
        let worker_count = self.worker_count();
        let siblings_left = Arc::new(AtomicUsize::new(worker_count));
        let factories: Vec<_> = stealable::<RecordBatch>(self.dispatcher.topology())
            .into_iter()
            .zip((0..worker_count).map(|_| MapFactory(builder())))
            .zip(self.factories)
            .map(|((channel_factory, unary_factory), head)| {
                UnaryOperatorFactory::new(
                    head,
                    unary_factory,
                    channel_factory,
                    siblings_left.clone(),
                )
            })
            .collect();
        OperatorSpec::new(self.dispatcher, factories)
    }

    /// Apply a type-changing map without changing an already-established
    /// batch order. The ordered producer emits from one worker, and the FIFO
    /// channel delivers those batches to one consumer in the same sequence.
    fn map_ordered<T, F, FB>(self, builder: FB) -> OrderedMapOperatorSpec<T, F>
    where
        T: Send + 'static,
        F: FnMut(RecordBatch) -> T + Send + 'static,
        FB: Fn() -> F,
    {
        let worker_count = self.worker_count();
        let siblings_left = Arc::new(AtomicUsize::new(worker_count));
        let target = self.dispatcher.next_worker();
        let factories: Vec<_> = to_single_worker_mpsc::<RecordBatch>(worker_count, target)
            .into_iter()
            .zip((0..worker_count).map(|_| MapFactory(builder())))
            .zip(self.factories)
            .map(|((channel_factory, unary_factory), head)| {
                UnaryOperatorFactory::new(
                    head,
                    unary_factory,
                    channel_factory,
                    siblings_left.clone(),
                )
            })
            .collect();
        OperatorSpec::new(self.dispatcher, factories)
    }

    /// Sugar for [`map`](Self::map) when the output is `RecordBatch`: applies
    /// the transform and stays in `RecordBatchOperatorSpec` so you can keep
    /// chaining RB-only methods like `.aggregate()` / `.order_by_limit()`.
    ///
    /// Unordered streams use the same work-stealing delivery as [`map`](Self::map).
    /// After `ORDER BY`, the stage instead reads the batches through one FIFO
    /// consumer so their global row order is preserved.
    pub fn project<F, FB>(self, builder: FB) -> Self
    where
        F: FnMut(RecordBatch) -> RecordBatch + Send + 'static,
        FB: Fn() -> F,
    {
        let worker_count = self.worker_count();
        self.unary((0..worker_count).map(|_| MapFactory(builder())))
    }

    /// Global aggregates (no GROUP BY): one or more `SUM`/`COUNT` slots over
    /// columns, computed in a single pass. Emits one single-row output column
    /// per slot (`Decimal128(38, 0)` for SUM, `Int64` for COUNT). `AVG` arrives
    /// pre-lowered to a SUM slot + a COUNT slot with a downstream divide.
    pub fn aggregate<A: IntCell + F64Cell + WideCell>(self, slots: Vec<AggregationSlot>) -> Self {
        let worker_count = self.worker_count();
        self.unary(AggregateFactory::<A>::create_for_workers(
            slots,
            worker_count,
        ))
    }

    /// Sort by the given columns and keep only the first `limit` rows.
    ///
    /// Workers independently collect their top-`limit` rows, then coordinate to
    /// produce the global top-`limit` result.
    ///
    /// # Example
    ///
    /// ```ignore
    /// # use std::sync::Arc;
    /// # use dispatch::*;
    /// # use dispatch::table_input;
    /// # let table = Arc::new(ParquetTable::from_files(dispatcher, &["/tmp/data.parquet"], &[]).unwrap());
    /// # let dispatch = Dispatch::spin_up(1, 32, None);
    /// # let dispatcher = dispatch.dispatcher();
    /// # let spec = table_input(&dispatcher, &table, Projection::columns([0]), false);
    /// // Top 10 rows ordered by column 0 descending
    /// spec.order_by_limit(vec![OrderBy::new(0, true, false)], 10)
    /// # ;
    /// ```
    pub fn order_by_limit(self, order_by: Vec<OrderBy>, limit: usize) -> Self {
        self.order_by_limit_offset(order_by, limit, 0, None)
    }

    /// Like [`order_by_limit`](Self::order_by_limit) but skips the first
    /// `offset` rows of the globally sorted result (SQL `LIMIT … OFFSET`), and
    /// optionally publishes the running boundary into a shared
    /// [`BoundarySlot`] so sibling scans can prune row groups.
    pub fn order_by_limit_offset(
        self,
        order_by: Vec<OrderBy>,
        limit: usize,
        offset: usize,
        dynamic_filter: Option<Arc<BoundarySlot>>,
    ) -> Self {
        let worker_count = self.worker_count();
        let mut result = self.unary(OrderByLimitFactory::create_for_workers(
            order_by,
            limit,
            offset,
            worker_count,
            dynamic_filter,
        ));
        result.stream_ordered = true;
        result
    }

    /// SQL `ORDER BY` with no LIMIT: emit the whole input sorted by
    /// `order_by`.
    ///
    /// Each worker sorts batches as they arrive. Sorted runs first merge within
    /// each NUMA node, then one k-way merge combines the node results. Both
    /// levels split large outputs into parallel, node-routed slices.
    /// See the `operations::unary::order_by` module.
    pub fn order_by(self, order_by: Vec<OrderBy>) -> Self {
        let topology = self.dispatcher.topology();
        let mut result = self.unary(OrderByFactory::create_for_workers(order_by, topology));
        result.stream_ordered = true;
        result
    }

    /// Full ORDER BY with the optional variant-layout normalization breaker
    /// inserted between the worker-local sorters and the ordinary run merger.
    pub fn order_by_normalizing(self, order_by: Vec<OrderBy>) -> Self {
        let topology = self.dispatcher.topology();
        let worker_count = self.worker_count();
        let collector_worker = self.dispatcher.next_worker();
        let (sorters, collectors) =
            create_normalizing_order_by_factories(order_by, topology, collector_worker);
        let input_siblings = Arc::new(AtomicUsize::new(worker_count));
        let collector_siblings = Arc::new(AtomicUsize::new(worker_count));
        let group_channels = to_single_worker_mpsc(worker_count, collector_worker);

        let factories = if self.stream_ordered {
            let input_worker = self.dispatcher.next_worker();
            let input_channels = to_single_worker_mpsc::<RecordBatch>(worker_count, input_worker);
            self.factories
                .into_iter()
                .zip(sorters)
                .zip(collectors)
                .zip(input_channels)
                .zip(group_channels)
                .map(
                    |((((head, sorter), collector), input_channel), group_channel)| {
                        let sorter = UnaryOperatorFactory::new(
                            head,
                            sorter,
                            input_channel,
                            input_siblings.clone(),
                        );
                        Box::new(UnaryOperatorFactory::new(
                            sorter,
                            collector,
                            group_channel,
                            collector_siblings.clone(),
                        )) as Box<dyn OperatorFactory<RecordBatch>>
                    },
                )
                .collect()
        } else {
            let input_channels = stealable::<RecordBatch>(topology);
            self.factories
                .into_iter()
                .zip(sorters)
                .zip(collectors)
                .zip(input_channels)
                .zip(group_channels)
                .map(
                    |((((head, sorter), collector), input_channel), group_channel)| {
                        let sorter = UnaryOperatorFactory::new(
                            head,
                            sorter,
                            input_channel,
                            input_siblings.clone(),
                        );
                        Box::new(UnaryOperatorFactory::new(
                            sorter,
                            collector,
                            group_channel,
                            collector_siblings.clone(),
                        )) as Box<dyn OperatorFactory<RecordBatch>>
                    },
                )
                .collect()
        };

        Self {
            dispatcher: self.dispatcher,
            factories,
            stream_ordered: true,
        }
    }

    /// SQL `LIMIT … OFFSET …` with no ORDER BY: keep `limit` rows after skipping
    /// the first `offset`, in arbitrary (input) order.
    ///
    /// Unlike an `ORDER BY … LIMIT`, this terminates early: once `limit + offset`
    /// rows have been buffered across all workers it stops, emits, and abandons
    /// the scan feeding it (see [`LimitFactory`] and the `limit` operator module).
    /// So `SELECT * FROM huge LIMIT 1` reads only as far as the first row group,
    /// not the whole table.
    pub fn limit(self, limit: usize, offset: usize) -> Self {
        let worker_count = self.worker_count();
        self.unary(LimitFactory::create_for_workers(
            limit,
            offset,
            worker_count,
        ))
    }

    /// Global `COUNT(DISTINCT x)`: GROUP BY `key_cols` with no aggregate, emitting
    /// only each hash partition's distinct-key count (one `Int64` row) rather than
    /// the keys. Partitions are hash-disjoint, so a downstream global `SUM` over
    /// those rows yields the total — without materialising the (potentially huge)
    /// key column. Pair with a keys-only extractor (e.g. `HashOnlyIntKeyExtractor`)
    /// for an 8-byte entry.
    pub fn group_by_distinct_count<K: KeyExtractor<Config: Default>>(
        self,
        key_cols: Vec<usize>,
    ) -> Self {
        let topology = self.dispatcher.topology();
        let buffers = self.dispatcher.buffers;
        self.unary(GroupFactory::<K, Distinct>::create_for_workers(
            key_cols,
            Vec::new(),
            K::Config::default(),
            None,
            true,
            topology,
            buffers,
        ))
    }

    /// GROUP BY one or more key columns computing one or more aggregate value
    /// slots (`COUNT(*)`/`SUM`/`COUNT(col)`) per group. `K` selects the key shape,
    /// `V` the aggregate shape (e.g. its arity). `key_config` configures the key
    /// extractor (e.g. a [`RowKeySchema`](crate::RowKeySchema)); pass `()` for the
    /// extractors whose key shape is fully determined by their type.
    pub fn group_by_aggregate<K: KeyExtractor, V: AggregationValue + ?Sized>(
        self,
        key_cols: Vec<usize>,
        value_slots: Vec<AggregationSlot>,
        output_limit: Option<GroupLimit>,
        key_config: K::Config,
    ) -> Self {
        let topology = self.dispatcher.topology();
        let buffers = self.dispatcher.buffers;
        self.unary(GroupFactory::<K, V>::create_for_workers(
            key_cols,
            value_slots,
            key_config,
            output_limit,
            false,
            topology,
            buffers,
        ))
    }

    /// Hash equi-join: build a hash table from `build`'s rows keyed on
    /// `spec.build_key_indices`, then probe it with `self`'s rows keyed on
    /// `spec.probe_key_indices`, emitting one output row per matching pair.
    /// Each output row is the probe columns listed in
    /// `spec.probe_output_indices` followed by `spec.build_output_indices`.
    /// Rows with a null value in any key column on either side never match.
    ///
    /// `spec.kind` picks which rows reach the output: the matching pairs
    /// alone, those plus one row per unmatched build row (with null probe
    /// columns) or per unmatched probe row (with null build columns), or one
    /// row per probe row that matched anything at all.
    ///
    /// `key_types` are the Arrow types the key columns arrive as, one per
    /// condition (the planner casts mismatched sides of each condition to a
    /// common type first); together they pick the join's [`JoinKey`]
    /// instantiation. A single key joins on any hashable fixed-width type,
    /// which excludes floats by construction.
    ///
    /// Both sides are roots in one dataflow. Probe input may be produced while
    /// the build runs, but the probe operator does not consume it until the
    /// completed build table is published.
    pub fn join(
        self,
        build: RecordBatchOperatorSpec,
        key_types: &[DataType],
        spec: JoinSpec,
    ) -> Self {
        self.join_with_optional_normalizer(build, key_types, spec, false)
    }

    /// Hash join variant of [`join`](Self::join) that inserts the reusable
    /// batch-group normalizer on the build side. Callers should select this
    /// only when the build schema contains variant columns; variant-free joins
    /// retain the direct one-breaker build path.
    pub fn join_normalizing_build(
        self,
        build: RecordBatchOperatorSpec,
        key_types: &[DataType],
        spec: JoinSpec,
    ) -> Self {
        self.join_with_optional_normalizer(build, key_types, spec, true)
    }

    fn join_with_optional_normalizer(
        self,
        build: RecordBatchOperatorSpec,
        key_types: &[DataType],
        spec: JoinSpec,
        normalize_build: bool,
    ) -> Self {
        use arrow_array::types as t;
        use arrow_schema::TimeUnit;
        assert_eq!(
            spec.build_key_indices.len(),
            key_types.len(),
            "one build key column per key type"
        );
        assert_eq!(
            spec.probe_key_indices.len(),
            key_types.len(),
            "one probe key column per key type"
        );
        assert_eq!(
            spec.probe_fields.len(),
            spec.probe_output_indices.len(),
            "one probe field per listed probe output column"
        );
        assert_eq!(
            spec.build_fields.len(),
            spec.build_output_indices.len(),
            "one build field per listed build output column"
        );
        // Multi-key shapes: the packed-lane combos compiled so far. Any int
        // pair combo (or a higher arity) is one more arm naming its
        // PackedKey tuple; the impls already exist for every arity.
        match key_types {
            [DataType::Int32, DataType::Int32] => {
                return self.join_dispatch::<PackedKey<(t::Int32Type, t::Int32Type)>>(
                    build,
                    spec,
                    normalize_build,
                );
            }
            [DataType::Int32, DataType::Int64] => {
                return self.join_dispatch::<PackedKey<(t::Int32Type, t::Int64Type)>>(
                    build,
                    spec,
                    normalize_build,
                );
            }
            [DataType::Int64, DataType::Int32] => {
                return self.join_dispatch::<PackedKey<(t::Int64Type, t::Int32Type)>>(
                    build,
                    spec,
                    normalize_build,
                );
            }
            [DataType::Int64, DataType::Int64] => {
                return self.join_dispatch::<PackedKey<(t::Int64Type, t::Int64Type)>>(
                    build,
                    spec,
                    normalize_build,
                );
            }
            _ => {}
        }
        // Every shape without a compiled instantiation: 3+ keys, mixed types,
        // strings. Hash-stored with per-candidate verification.
        let [key_type] = key_types else {
            return self.join_dispatch::<DynamicRowKey>(build, spec, normalize_build);
        };
        match key_type {
            DataType::Int8 => {
                self.join_dispatch::<SingleColumnKey<t::Int8Type>>(build, spec, normalize_build)
            }
            DataType::Int16 => {
                self.join_dispatch::<SingleColumnKey<t::Int16Type>>(build, spec, normalize_build)
            }
            DataType::Int32 => {
                self.join_dispatch::<SingleColumnKey<t::Int32Type>>(build, spec, normalize_build)
            }
            DataType::Int64 => {
                self.join_dispatch::<SingleColumnKey<t::Int64Type>>(build, spec, normalize_build)
            }
            DataType::UInt8 => {
                self.join_dispatch::<SingleColumnKey<t::UInt8Type>>(build, spec, normalize_build)
            }
            DataType::UInt16 => {
                self.join_dispatch::<SingleColumnKey<t::UInt16Type>>(build, spec, normalize_build)
            }
            DataType::UInt32 => {
                self.join_dispatch::<SingleColumnKey<t::UInt32Type>>(build, spec, normalize_build)
            }
            DataType::UInt64 => {
                self.join_dispatch::<SingleColumnKey<t::UInt64Type>>(build, spec, normalize_build)
            }
            DataType::Date32 => {
                self.join_dispatch::<SingleColumnKey<t::Date32Type>>(build, spec, normalize_build)
            }
            DataType::Date64 => {
                self.join_dispatch::<SingleColumnKey<t::Date64Type>>(build, spec, normalize_build)
            }
            DataType::Timestamp(TimeUnit::Second, _) => self
                .join_dispatch::<SingleColumnKey<t::TimestampSecondType>>(
                    build,
                    spec,
                    normalize_build,
                ),
            DataType::Timestamp(TimeUnit::Millisecond, _) => self.join_dispatch::<SingleColumnKey<
                t::TimestampMillisecondType,
            >>(
                build, spec, normalize_build
            ),
            DataType::Timestamp(TimeUnit::Microsecond, _) => self.join_dispatch::<SingleColumnKey<
                t::TimestampMicrosecondType,
            >>(
                build, spec, normalize_build
            ),
            DataType::Timestamp(TimeUnit::Nanosecond, _) => self.join_dispatch::<SingleColumnKey<
                t::TimestampNanosecondType,
            >>(
                build, spec, normalize_build
            ),
            DataType::Decimal64(_, _) => self.join_dispatch::<SingleColumnKey<t::Decimal64Type>>(
                build,
                spec,
                normalize_build,
            ),
            DataType::Decimal128(_, _) => self.join_dispatch::<SingleColumnKey<t::Decimal128Type>>(
                build,
                spec,
                normalize_build,
            ),
            // A single-column key of any other type (a string) also takes the
            // dynamic shape.
            _ => self.join_dispatch::<DynamicRowKey>(build, spec, normalize_build),
        }
    }

    /// Pick the kind's instantiation, so the probe's match loop carries no
    /// runtime test for it.
    fn join_dispatch<K: JoinKey>(
        self,
        build: RecordBatchOperatorSpec,
        spec: JoinSpec,
        normalize_build: bool,
    ) -> Self {
        match spec.kind {
            JoinKind::Inner => self.join_typed::<K, false, false, false, false, false>(
                build,
                spec,
                normalize_build,
            ),
            JoinKind::BuildOuter => {
                self.join_typed::<K, true, false, false, false, false>(build, spec, normalize_build)
            }
            JoinKind::ProbeOuter => {
                self.join_typed::<K, false, false, true, false, false>(build, spec, normalize_build)
            }
            // With a residual predicate the first key match may not be a real
            // match, so the semi join runs on the pair-recording instantiation
            // and drops duplicate probe rows at drain time instead of exiting
            // the match loop early.
            JoinKind::ProbeSemi if spec.residual_filters.is_some() => self
                .join_typed::<K, false, false, false, false, false>(build, spec, normalize_build),
            JoinKind::ProbeSemi => {
                self.join_typed::<K, false, true, false, false, false>(build, spec, normalize_build)
            }
            // An anti join runs the probe-side outer join's miss tracking and
            // emits the misses alone. One instantiation serves with and
            // without a residual: the settled flags already classify the rows
            // whose every pair a residual rejects.
            JoinKind::ProbeAnti => {
                self.join_typed::<K, false, false, true, true, false>(build, spec, normalize_build)
            }
            // Likewise on the build side: the outer join's flag array and
            // unmatched scan run unchanged, and only the unmatched build rows
            // come out.
            JoinKind::BuildAnti => {
                self.join_typed::<K, true, false, false, true, false>(build, spec, normalize_build)
            }
            // A build-side semi join is that anti join with the scan's
            // polarity flipped, a runtime byte in the outputter, so it rides
            // the same instantiation: a match only flags its build
            // row, and the scan emits the flagged rows instead.
            JoinKind::BuildSemi => {
                self.join_typed::<K, true, false, false, true, false>(build, spec, normalize_build)
            }
            // A mark join classifies every probe row: the semi join's
            // first-match exit decides the hits, the probe-side miss tracking
            // the misses, and neither is ever a residual question (a mark
            // join carries none).
            JoinKind::ProbeMark => {
                self.join_typed::<K, false, true, true, false, true>(build, spec, normalize_build)
            }
        }
    }

    fn join_typed<
        K: JoinKey,
        const TRACK_MATCHED_BUILD_ROWS: bool,
        const STOP_AFTER_FIRST_MATCH: bool,
        const TRACK_UNMATCHED_PROBE_ROWS: bool,
        const DISCARD_MATCHED_ROWS: bool,
        const EMIT_MARK_COLUMN: bool,
    >(
        self,
        build: RecordBatchOperatorSpec,
        spec: JoinSpec,
        normalize_build: bool,
    ) -> Self {
        let worker_count = self.worker_count();
        assert_eq!(
            worker_count,
            build.worker_count(),
            "join inputs must use the same worker count"
        );
        assert!(
            self.dispatcher
                .waker_set
                .wakes_same_pool(&build.dispatcher.waker_set),
            "join inputs must use the same worker pool"
        );

        let topology = self.dispatcher.topology();
        let (_, build_heads) = build.into_parts();
        let probe_siblings_left = Arc::new(AtomicUsize::new(worker_count));
        let probe_channels = stealable::<RecordBatch>(topology);

        let factories = if normalize_build {
            let collector_worker = self.dispatcher.next_worker();
            let (build_factories, collector_factories, probe_factories, build_ready) =
                create_normalizing_join_factories::<
                    K,
                    TRACK_MATCHED_BUILD_ROWS,
                    STOP_AFTER_FIRST_MATCH,
                    TRACK_UNMATCHED_PROBE_ROWS,
                    DISCARD_MATCHED_ROWS,
                    EMIT_MARK_COLUMN,
                >(spec, worker_count, collector_worker);
            let build_siblings_left = Arc::new(AtomicUsize::new(worker_count));
            let collector_siblings_left = Arc::new(AtomicUsize::new(worker_count));
            let build_channels = stealable::<RecordBatch>(topology);
            let group_channels = to_single_worker_mpsc(worker_count, collector_worker);

            self.factories
                .into_iter()
                .zip(build_heads)
                .zip(build_factories)
                .zip(collector_factories)
                .zip(probe_factories)
                .zip(build_channels)
                .zip(group_channels)
                .zip(probe_channels)
                .map(
                    |(
                        (
                            (
                                (
                                    (((probe_head, build_head), build_factory), collector_factory),
                                    probe_factory,
                                ),
                                build_channel,
                            ),
                            group_channel,
                        ),
                        probe_channel_factory,
                    )| {
                        let first_breaker = UnaryOperatorFactory::new(
                            build_head,
                            build_factory,
                            build_channel,
                            build_siblings_left.clone(),
                        );
                        let build_graph = Box::new(UnaryOperatorFactory::new(
                            first_breaker,
                            collector_factory,
                            group_channel,
                            collector_siblings_left.clone(),
                        ))
                            as Box<dyn OperatorFactory<()>>;
                        Box::new(JoinRecordBatchOperatorFactory {
                            probe_head,
                            build_graph,
                            probe_factory,
                            probe_channel_factory,
                            probe_siblings_left: probe_siblings_left.clone(),
                            build_ready: build_ready.clone(),
                        }) as Box<dyn OperatorFactory<RecordBatch>>
                    },
                )
                .collect()
        } else {
            let (build_factories, probe_factories, build_ready) = create_join_factories::<
                K,
                TRACK_MATCHED_BUILD_ROWS,
                STOP_AFTER_FIRST_MATCH,
                TRACK_UNMATCHED_PROBE_ROWS,
                DISCARD_MATCHED_ROWS,
                EMIT_MARK_COLUMN,
            >(spec, worker_count);
            let build_siblings_left = Arc::new(AtomicUsize::new(worker_count));
            let build_channels = stealable::<RecordBatch>(topology);

            self.factories
                .into_iter()
                .zip(build_heads)
                .zip(build_factories)
                .zip(probe_factories)
                .zip(build_channels)
                .zip(probe_channels)
                .map(
                    |(
                        ((((probe_head, build_head), build_factory), probe_factory), build_channel),
                        probe_channel_factory,
                    )| {
                        let build_graph = Box::new(UnaryOperatorFactory::new(
                            build_head,
                            build_factory,
                            build_channel,
                            build_siblings_left.clone(),
                        ))
                            as Box<dyn OperatorFactory<()>>;
                        Box::new(JoinRecordBatchOperatorFactory {
                            probe_head,
                            build_graph,
                            probe_factory,
                            probe_channel_factory,
                            probe_siblings_left: probe_siblings_left.clone(),
                            build_ready: build_ready.clone(),
                        }) as Box<dyn OperatorFactory<RecordBatch>>
                    },
                )
                .collect()
        };

        Self {
            dispatcher: self.dispatcher,
            factories,
            stream_ordered: false,
        }
    }

    /// Join `self` (the probe side) with `build` on one `<`/`<=`/`>`/`>=`
    /// comparison — see [`RangeJoinSpec`]. `key_type` is the arrow type both
    /// key columns arrive as; it must be a fixed-width ordered type (ints,
    /// decimals, date, timestamp — the caller casts mismatched sides to a
    /// common type first). Floats are out: their SQL ordering (NULL/NaN) is
    /// not their native `Ord`.
    ///
    /// Both sides are roots in one dataflow, with the same build-before-probe
    /// gating as [`join`](Self::join).
    pub fn range_join(
        self,
        build: RecordBatchOperatorSpec,
        key_type: &DataType,
        spec: RangeJoinSpec,
    ) -> Self {
        self.range_join_with_optional_normalizer(build, key_type, spec, false)
    }

    /// Range join whose build-side ORDER BY uses the variant normalization
    /// composition before publishing its sorted chunks.
    pub fn range_join_normalizing_build(
        self,
        build: RecordBatchOperatorSpec,
        key_type: &DataType,
        spec: RangeJoinSpec,
    ) -> Self {
        self.range_join_with_optional_normalizer(build, key_type, spec, true)
    }

    fn range_join_with_optional_normalizer(
        self,
        build: RecordBatchOperatorSpec,
        key_type: &DataType,
        spec: RangeJoinSpec,
        normalize_build: bool,
    ) -> Self {
        use arrow_array::types as t;
        use arrow_schema::TimeUnit;
        match key_type {
            DataType::Int8 => self.range_join_typed::<t::Int8Type>(build, spec, normalize_build),
            DataType::Int16 => self.range_join_typed::<t::Int16Type>(build, spec, normalize_build),
            DataType::Int32 => self.range_join_typed::<t::Int32Type>(build, spec, normalize_build),
            DataType::Int64 => self.range_join_typed::<t::Int64Type>(build, spec, normalize_build),
            DataType::UInt8 => self.range_join_typed::<t::UInt8Type>(build, spec, normalize_build),
            DataType::UInt16 => {
                self.range_join_typed::<t::UInt16Type>(build, spec, normalize_build)
            }
            DataType::UInt32 => {
                self.range_join_typed::<t::UInt32Type>(build, spec, normalize_build)
            }
            DataType::UInt64 => {
                self.range_join_typed::<t::UInt64Type>(build, spec, normalize_build)
            }
            // Floats compare by IEEE totalOrder end to end: the build sort's
            // row encoding and `ArrowNativeTypeOp::compare` agree on it, and
            // NaN sorting greatest matches SQL's comparison semantics for
            // every NaN arithmetic produces.
            DataType::Float32 => {
                self.range_join_typed::<t::Float32Type>(build, spec, normalize_build)
            }
            DataType::Float64 => {
                self.range_join_typed::<t::Float64Type>(build, spec, normalize_build)
            }
            DataType::Date32 => {
                self.range_join_typed::<t::Date32Type>(build, spec, normalize_build)
            }
            DataType::Date64 => {
                self.range_join_typed::<t::Date64Type>(build, spec, normalize_build)
            }
            DataType::Timestamp(TimeUnit::Second, _) => {
                self.range_join_typed::<t::TimestampSecondType>(build, spec, normalize_build)
            }
            DataType::Timestamp(TimeUnit::Millisecond, _) => {
                self.range_join_typed::<t::TimestampMillisecondType>(build, spec, normalize_build)
            }
            DataType::Timestamp(TimeUnit::Microsecond, _) => {
                self.range_join_typed::<t::TimestampMicrosecondType>(build, spec, normalize_build)
            }
            DataType::Timestamp(TimeUnit::Nanosecond, _) => {
                self.range_join_typed::<t::TimestampNanosecondType>(build, spec, normalize_build)
            }
            DataType::Decimal64(_, _) => {
                self.range_join_typed::<t::Decimal64Type>(build, spec, normalize_build)
            }
            DataType::Decimal128(_, _) => {
                self.range_join_typed::<t::Decimal128Type>(build, spec, normalize_build)
            }
            other => panic!("range join key type {other:?} is unsupported"),
        }
    }

    fn range_join_typed<T: arrow_array::types::ArrowPrimitiveType>(
        self,
        build: RecordBatchOperatorSpec,
        spec: RangeJoinSpec,
        normalize_build: bool,
    ) -> Self
    where
        T::Native: ArrowNativeTypeOp + Send,
    {
        let worker_count = self.worker_count();
        assert_eq!(
            worker_count,
            build.worker_count(),
            "join inputs must use the same worker count"
        );
        assert!(
            self.dispatcher
                .waker_set
                .wakes_same_pool(&build.dispatcher.waker_set),
            "join inputs must use the same worker pool"
        );

        // Sort the build side by the key (NULLs last, so the join can drop
        // them as one tail), then funnel the sorted chunks to one worker,
        // whose consumer receives them in the order the sort emitted them.
        // Successive single-consumer stages take turns hosting that work.
        let target = self.dispatcher.next_worker();
        let order = vec![OrderBy::new(spec.build_key_index, false, false)];
        let build = if normalize_build {
            build.order_by_normalizing(order)
        } else {
            build.order_by(order)
        };

        let (build_factories, probe_factories, build_ready) =
            create_range_join_factories::<T>(spec, worker_count, target);

        let (_, build_heads) = build.into_parts();
        let build_siblings_left = Arc::new(AtomicUsize::new(worker_count));
        let probe_siblings_left = Arc::new(AtomicUsize::new(worker_count));
        let build_channels = to_single_worker_mpsc::<RecordBatch>(worker_count, target);
        let probe_channels = stealable::<RecordBatch>(self.dispatcher.topology());
        let factories = self
            .factories
            .into_iter()
            .zip(build_heads)
            .zip(build_factories)
            .zip(probe_factories)
            .zip(build_channels)
            .zip(probe_channels)
            .map(
                |(
                    (
                        (((probe_head, build_head), build_factory), probe_factory),
                        build_channel_factory,
                    ),
                    probe_channel_factory,
                )| {
                    let build_graph = Box::new(UnaryOperatorFactory::new(
                        build_head,
                        build_factory,
                        build_channel_factory,
                        build_siblings_left.clone(),
                    )) as Box<dyn OperatorFactory<()>>;
                    Box::new(JoinRecordBatchOperatorFactory {
                        probe_head,
                        build_graph,
                        probe_factory,
                        probe_channel_factory,
                        probe_siblings_left: probe_siblings_left.clone(),
                        build_ready: build_ready.clone(),
                    }) as Box<dyn OperatorFactory<RecordBatch>>
                },
            )
            .collect();

        Self {
            dispatcher: self.dispatcher,
            factories,
            stream_ordered: false,
        }
    }

    /// Execute the dataflow and collect all output batches.
    ///
    /// Sends one factory per worker to the dispatcher, waits for all workers to
    /// finish, and returns the collected `RecordBatch` results.
    ///
    /// This is a blocking call — it does not return until all workers have completed.
    ///
    /// # Example
    ///
    /// ```ignore
    /// # use std::sync::Arc;
    /// # use arrow_array::{RecordBatch, StringViewArray};
    /// # use dispatch::*;
    /// # let table = Arc::new(ParquetTable::from_files(dispatcher, &["/tmp/data.parquet"], &[]).unwrap());
    /// # let dispatch = Dispatch::spin_up(1, 32, None);
    /// # let dispatcher = dispatch.dispatcher();
    /// let batches: Vec<RecordBatch> = table_input(&dispatcher, &table, Projection::columns([0]), false)
    ///     .filter(|| {
    ///         let mut contains = Contains::new("google");
    ///         move |batch: &RecordBatch| {
    ///             let col = batch.column(0).as_any()
    ///                 .downcast_ref::<StringViewArray>().unwrap();
    ///             contains.run(col)
    ///         }
    ///     })
    ///     .collect()
    ///     .unwrap();
    /// ```
    /// Dispatch the pipeline and return its ring-backed Arrow output.
    pub fn execute(self) -> DataFlowHandle<RecordBatch> {
        let factories: Vec<_> = self.factories.into_iter().collect();
        OperatorSpec::new(self.dispatcher, factories).execute()
    }

    /// Convert each result batch to `T` on its dispatch worker, then launch the
    /// dataflow.
    pub fn execute_as<T: OutputBatch>(self) -> DataFlowHandle<T> {
        if self.stream_ordered {
            self.map_ordered(|| T::from_record_batch).execute()
        } else {
            self.map(|| T::from_record_batch).execute()
        }
    }

    /// Like [`execute`](Self::execute), with worker statistics enabled.
    pub fn execute_with_stats(self) -> DataFlowHandle<RecordBatch> {
        let factories: Vec<_> = self.factories.into_iter().collect();
        OperatorSpec::new(self.dispatcher, factories).execute_with_stats()
    }

    /// Like [`execute_as`](Self::execute_as), with worker statistics enabled.
    pub fn execute_with_stats_as<T: OutputBatch>(self) -> DataFlowHandle<T> {
        if self.stream_ordered {
            self.map_ordered(|| T::from_record_batch)
                .execute_with_stats()
        } else {
            self.map(|| T::from_record_batch).execute_with_stats()
        }
    }

    /// Run the dataflow and collect heap-backed Arrow batches.
    pub fn collect(self) -> crate::data_flow::Result<Vec<RecordBatch>> {
        self.collect_as::<RecordBatch>()
    }

    /// Run the dataflow and collect output batches converted on their workers.
    pub fn collect_as<T: OutputBatch>(self) -> crate::data_flow::Result<Vec<T>> {
        self.execute_as::<T>().collect()
    }

    /// Like [`collect`](Self::collect), but also returns the dataflow's IO/CPU
    /// stats folded across workers.
    pub fn collect_with_stats(self) -> crate::data_flow::Result<(Vec<RecordBatch>, DataFlowStats)> {
        self.collect_with_stats_as::<RecordBatch>()
    }

    /// Like [`collect_as`](Self::collect_as), with worker statistics enabled.
    pub fn collect_with_stats_as<T: OutputBatch>(
        self,
    ) -> crate::data_flow::Result<(Vec<T>, DataFlowStats)> {
        self.execute_with_stats_as::<T>().collect_with_stats()
    }
}
