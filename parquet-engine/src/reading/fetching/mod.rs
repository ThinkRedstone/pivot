//! Row-group fetching pipeline: issuing IO requests and collecting raw column
//! buffers.
//!
//! This module turns claimed row groups
//! ([`QueryRowGroupMetadata`])
//! into fully populated [`RowGroupBuffer`](crate::RowGroupBuffer)s by reading
//! column chunks from disk via io-uring, with cache lookups to avoid redundant
//! reads.
//!
//! - [`RowGroupInjectorFactory`] / `RowGroupInjector` — the root data source
//!   that enqueues every row group in a [`ParquetTable`](crate::ParquetTable)
//!   into work-stealing [`Injector`](crossbeam_deque::Injector)s so workers
//!   can pull row groups on demand: one set for the row groups expected to be
//!   in memory, one for those expected to need a read.
//! - [`RowGroupFetcherFactory`] / [`fetcher::RowGroupFetcher`] — the unary
//!   operator that receives a claimed row group, plans its read against the
//!   decompressed cache, submits IO requests for what is missing (disk or
//!   HTTP), collects completions, and emits the finished `RowGroupBuffer`
//!   downstream. It takes the row groups expected to be in memory when its
//!   worker has no other work, and asks for the others ahead of the worker's
//!   CPU work, bounded by the worker's read-ahead budgets of bytes.

/// Builds one worker's [`RowGroupFetcher`], carrying the scan's projection,
/// the medium its table is read from (see
/// [`ParquetTable::get_read_medium`](crate::ParquetTable::get_read_medium)),
/// whether it asks for reads ahead of its worker's CPU work, the factory of
/// the receiver the row groups expected to need a read wait in (a table
/// scan's [`RowGroupInjectorFactory`] over them, or the queue a late
/// materialization's materializer sends them to), and the queue every
/// worker's fetcher pushes the uncached row groups to and takes them from.
pub struct RowGroupFetcherFactory<IoRowGroupsReceiverFactory> {
    projection: std::sync::Arc<crate::types::projection::Projection>,
    read_medium: dispatch::io::ReadMedium,
    reads_ahead_of_cpu_work: bool,
    io_row_groups_receiver_factory: IoRowGroupsReceiverFactory,
    uncached_row_groups_channel: dispatch::SharedWorkQueueChannelFactory<QueryRowGroupMetadata>,
}

impl<IoRowGroupsReceiverFactory> RowGroupFetcherFactory<IoRowGroupsReceiverFactory> {
    pub fn new(
        projection: std::sync::Arc<crate::types::projection::Projection>,
        read_medium: dispatch::io::ReadMedium,
        reads_ahead_of_cpu_work: bool,
        io_row_groups_receiver_factory: IoRowGroupsReceiverFactory,
        uncached_row_groups_channel: dispatch::SharedWorkQueueChannelFactory<QueryRowGroupMetadata>,
    ) -> Self {
        Self {
            projection,
            read_medium,
            reads_ahead_of_cpu_work,
            io_row_groups_receiver_factory,
            uncached_row_groups_channel,
        }
    }
}

impl<IoRowGroupsReceiverFactory>
    dispatch::UnaryFactory<QueryRowGroupMetadata, crate::RowGroupBuffer>
    for RowGroupFetcherFactory<IoRowGroupsReceiverFactory>
where
    IoRowGroupsReceiverFactory: dispatch::RootChannelFactory<QueryRowGroupMetadata> + 'static,
{
    type Unary = RowGroupFetcher<IoRowGroupsReceiverFactory::Receiver>;

    /// Runs on the worker thread, where the receivers are built: a table
    /// scan's snapshots which node's queue is local to the worker.
    fn build_unary(self, _context: &mut BuildContext) -> BuiltUnary<Self::Unary> {
        let (uncached_row_groups_sender, uncached_row_groups_receiver) =
            dispatch::ChannelFactory::build(self.uncached_row_groups_channel);
        Built::new(RowGroupFetcher::new(
            self.projection,
            self.read_medium,
            self.reads_ahead_of_cpu_work,
            dispatch::RootChannelFactory::build(self.io_row_groups_receiver_factory),
            uncached_row_groups_sender,
            uncached_row_groups_receiver,
        ))
    }
}
mod fetcher;

mod table_source;
use crate::reading::fetching::fetcher::RowGroupFetcher;
use crate::types::metadata::QueryRowGroupMetadata;
use dispatch::BuildContext;
use dispatch::{Built, BuiltUnary};
pub use table_source::RowGroupInjectorFactory;
