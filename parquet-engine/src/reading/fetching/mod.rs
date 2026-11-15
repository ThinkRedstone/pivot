//! Row-group fetching pipeline: issuing IO requests and collecting raw column
//! buffers.
//!
//! This module turns claimed row groups
//! ([`QueryRowGroupMetadata`](crate::types::metadata::QueryRowGroupMetadata))
//! into fully populated [`RowGroupBuffer`](crate::RowGroupBuffer)s by reading
//! column chunks from disk via io-uring, with cache lookups to avoid redundant
//! reads.
//!
//! - [`RowGroupInjectorFactory`] / `RowGroupInjector` — the root data source
//!   that enqueues every row group in a [`ParquetTable`](crate::ParquetTable)
//!   into a work-stealing [`Injector`](crossbeam_deque::Injector) so workers
//!   can pull row groups on demand.
//! - [`RowGroupFetcherFactory`] / [`fetcher::RowGroupFetcher`] — the unary
//!   operator that receives a claimed row group, plans its read against the
//!   decompressed cache, submits IO requests for what is missing (disk or
//!   HTTP), collects completions, and emits the finished `RowGroupBuffer`
//!   downstream. What it has asked for is bounded by its worker's read-ahead
//!   budget of bytes, for disk and HTTP alike.

/// Builds one worker's [`RowGroupFetcher`], carrying the scan's projection
/// and the medium whose read-ahead budget bounds its claims (see
/// [`ParquetTable::get_read_medium`](crate::ParquetTable::get_read_medium)).
pub struct RowGroupFetcherFactory {
    projection: std::sync::Arc<crate::types::projection::Projection>,
    read_medium: dispatch::io::ReadMedium,
}

impl RowGroupFetcherFactory {
    pub fn new(
        projection: std::sync::Arc<crate::types::projection::Projection>,
        read_medium: dispatch::io::ReadMedium,
    ) -> Self {
        Self {
            projection,
            read_medium,
        }
    }
}

impl dispatch::UnaryFactory<crate::types::metadata::QueryRowGroupMetadata, crate::RowGroupBuffer>
    for RowGroupFetcherFactory
{
    type Unary = RowGroupFetcher;

    fn build_unary(self, _context: &mut BuildContext) -> BuiltUnary<RowGroupFetcher> {
        Built::new(RowGroupFetcher::new(self.projection, self.read_medium))
    }
}
mod fetcher;

mod table_source;
use crate::reading::fetching::fetcher::RowGroupFetcher;
use dispatch::BuildContext;
use dispatch::{Built, BuiltUnary};
pub use table_source::RowGroupInjectorFactory;
