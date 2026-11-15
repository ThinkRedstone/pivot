//! The **reading** pipeline — the per-query scan, as a dispatch dataflow:
//!
//! - [`fetching`] — the source: injects a table's row groups and reads each
//!   projected column chunk over the io_uring ring (disk or presigned HTTP).
//! - [`indexer`] — splits the fetched chunks into compressed pages.
//! - [`decompressor`] — decompresses each page.
//! - [`range_cutter`] — collects each row group's pages and cuts its rows
//!   into decode ranges.
//! - [`decoding`] — decodes ranges into Arrow arrays.
//! - [`materializer`] — late materialization: re-reads surviving rows by row id.
//! - [`scan`] — the builders that chain the stages into one spec.
//!
//! It scans the row groups the [`metadata`](super::metadata) (table-load)
//! pipeline produced at `CREATE`/`ATTACH` time.

mod fetching;
pub use fetching::{RowGroupFetcherFactory, RowGroupInjectorFactory};

mod indexer;
pub use indexer::IndexerFactory;

mod decompressor;
pub use decompressor::{DecompressorFactory, Error as DecompressorError};

pub(crate) mod decoding;
pub use decoding::{ColumnDecoderError, DecoderFactory, ScanEqualityPredicate, WorkerAllocator};

mod range_cutter;
pub use range_cutter::{DecodeRange, RangeCutterFactory};

mod materializer;
pub use materializer::MaterializerFactory;

mod record_batch_metadata;

mod empty_projection_scan;

mod scan;
pub use scan::{
    materialize, table_input, table_input_with_filter, table_input_with_filter_and_eq_predicates,
};
