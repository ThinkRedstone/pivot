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
//!   downstream. It keeps disk- and HTTP-backed row groups outstanding under
//!   separate caps.

/// Builds one worker's [`RowGroupFetcher`], carrying the scan's projection and
/// that worker's claimed-but-undecoded row-group count as claim backpressure,
/// bounded by the scan's claim bound (see [`fetcher::pending_claim_bound`]).
pub struct RowGroupFetcherFactory {
    projection: std::sync::Arc<crate::types::projection::Projection>,
    pending_row_groups: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    max_pending_row_groups: usize,
}

impl RowGroupFetcherFactory {
    pub fn new(
        projection: std::sync::Arc<crate::types::projection::Projection>,
        pending_row_groups: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        max_pending_row_groups: usize,
    ) -> Self {
        Self {
            projection,
            pending_row_groups,
            max_pending_row_groups,
        }
    }
}

impl dispatch::UnaryFactory<crate::types::metadata::QueryRowGroupMetadata, crate::RowGroupBuffer>
    for RowGroupFetcherFactory
{
    type Unary = RowGroupFetcher;

    fn build_unary(self) -> RowGroupFetcher {
        RowGroupFetcher::new(
            self.projection,
            self.pending_row_groups,
            self.max_pending_row_groups,
        )
    }
}
mod fetcher;
pub use fetcher::pending_claim_bound;

mod table_source;
use crate::reading::fetching::fetcher::RowGroupFetcher;
pub use table_source::RowGroupInjectorFactory;
