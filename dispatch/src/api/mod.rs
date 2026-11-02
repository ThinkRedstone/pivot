//! API for building parallel RecordBatch query dataflows.
//!
//! # Usage
//!
//! Build a query by chaining operations on [`RecordBatchOperatorSpec`]:
//!
//! ```ignore
//! # use std::sync::Arc;
//! # use arrow_array::{RecordBatch, StringViewArray};
//! # use dispatch::*;
//! # use dispatch::table_input;
//! # let table = Arc::new(ParquetTable::from_files(dispatcher, &["/tmp/data.parquet"], &[]).unwrap());
//! # let dispatch = Dispatch::spin_up(1, 32, None);
//! let results = table_input(dispatch.dispatcher(), &table, Projection::columns([0]), false)
//!     .filter(|| {
//!         let mut contains = Contains::new("google");
//!         move |batch: &RecordBatch, _: &mut SlabAllocator, scratch: &mut SelectionScratch| {
//!             let col = batch.column(0).as_any()
//!                 .downcast_ref::<StringViewArray>().unwrap();
//!             collect_selected_indices(&contains.run(col), &mut scratch.positions);
//!             Ok(RowSelection::Indices)
//!         }
//!     })
//!     .aggregate::<i64>(vec![AggregationSlot::new(AggregationKind::CountStar, 0, DataType::Int64)])
//!     .collect();
//! ```
//!
//! Each method appends a stage to the dataflow and returns `Self`, so the full query
//! type is always just `RecordBatchOperatorSpec` — no nested generics leak out.
//!
//! # Internal architecture
//!
//! A query is a blueprint: one factory per worker thread. Nothing runs until
//! [`RecordBatchOperatorSpec::collect`] is called. The lifecycle is:
//!
//! 1. **Build factories** — Each chained method (`.filter(...)`, `.aggregate(...)`, etc.) wraps
//!    the previous factories in a new layer of [`UnaryOperatorFactory`](crate::UnaryOperatorFactory), producing
//!    already one factory per worker.
//!    The factories are stored as `Box<dyn OperatorFactory<RecordBatch>>`
//!    to erase the nested generic types.
//!
//! 2. **Ship to workers** — `collect()` pairs each factory with an `MpscSender<RecordBatch>`
//!    inside a [`DataFlowBuilder`] and sends it to the corresponding worker thread. Building
//!    is deferred because the build step creates things which may not be `Send` such as `Rc<Worker<T>>`
//!    which live on the worker thread.
//!
//! 3. **Build on worker** — The worker calls [`DataFlowBuilder::build`], which calls
//!    `build(sender)` on the factory. This recursively builds the full operator
//!    chain: each factory creates a channel, builds its head with the channel's sender,
//!    then appends its own operator reading from the channel's receiver.
//!
//! 4. **Execute** — The worker runs the resulting [`DataFlow`](crate::data_flow::DataFlow),
//!    driving data through the operator chain.
//!
//! # Key types
//!
//! - [`RecordBatchOperatorSpec`] — The user-facing query builder.
//!   Holds `VecDeque<Box<dyn OperatorFactory<RecordBatch>>>` (one per worker).
//!
//! - [`OperatorFactory<O>`](OperatorFactory) — Factory trait with
//!   `build(self: Box<Self>, sender: Box<dyn Sender<O>>)`. The sender is a trait
//!   object, so the trait is object-safe and a factory can be boxed at any stage.
//!   Also carries pipelines that flow through non-RecordBatch intermediate types —
//!   e.g. the multi-stage decode in `catalog`'s Parquet reader.
//!
//! - [`OperatorSpec<O, OF>`](OperatorSpec) — Generic spec holding `VecDeque<OF>`. Can be used for
//!   any operator that does not expose `RecordBatch`. Used to build multi-stage
//!   pipelines (e.g. `catalog`'s Parquet reader) before erasing into
//!   `RecordBatchOperatorSpec` via `from_spec`.
//!
//! - [`OperatorGraphBuilder`] — Accumulates operators and edges during the build step,
//!   then converts them to a `DataFlow`.
//!
//! - [`DataFlowBuilder`] — Pairs a `Box<dyn OperatorFactory<RecordBatch>>` with the output
//!   `MpscSender`. Sent to a worker thread, which calls `.build()` to produce a `DataFlow`.

mod build_context;
pub use build_context::BuildContext;

mod builder;
pub use builder::{DataFlowBuilder, OperatorGraphBuilder};

mod operator_spec;
pub use operator_spec::{OperatorFactory, OperatorSpec};

mod data_flow_handle;
mod record_batch_operator;
pub use data_flow_handle::{CancelToken, DataFlowHandle};

pub use operator_spec::{channel_input, values_input};
pub use record_batch_operator::{OutputBatch, RECORD_BATCH_SIZE, RecordBatchOperatorSpec};
