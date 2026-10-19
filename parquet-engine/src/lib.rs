//! The Parquet engine: two separate dataflow pipelines over the same files.
//!
//! - [`reading`] — the per-query scan: fetch a table's projected column chunks
//!   over the ring, index, decompress, and decode them into record batches.
//! - [`writing`] — the encode pipeline: partition record batches into files,
//!   encode each column chunk, and assemble the Parquet bytes (shared by
//!   compaction and SQL `INSERT`).
//! - [`metadata`] — the table load, run once at `CREATE`/`ATTACH`: read every
//!   file's *footer* into the [`ParquetTable`] whose row groups all later scans
//!   reuse.
//!
//! [`types`] (the table, row-group metadata, pages, requests), [`pushdown`]
//! (static filter pushdown shared by catalog and external bindings), and
//! [`row_group_stats`] (min/max elimination shared by static and dynamic
//! filters) are common to both.

#![allow(rustdoc::private_intra_doc_links)]

/// Map a Parquet-stage error (thrift parse / decompress / decode) onto
/// dispatch's generic operator error, so an operator's `?` converts into
/// `dispatch::UnaryResult` now that the pipeline lives outside dispatch.
pub fn op_err(e: impl std::error::Error + Send + Sync + 'static) -> dispatch::UnaryError {
    dispatch::UnaryError::Operator(Box::new(e))
}

/// Remote read-ahead depth: how many HTTP block reads a fetcher keeps in flight
/// per worker before admitting the next row group. The effective scan concurrency
/// against object storage is this times the worker count, so for a table of many
/// small files (round-trip bound) raising it can hide more latency. Tunable via
/// `PIVOT_HTTP_READAHEAD`; resolved once.
pub(crate) fn http_readahead() -> usize {
    static VALUE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| dispatch::env::get_env_var_with_default("PIVOT_HTTP_READAHEAD", 64))
}

mod external;
pub use external::bind_read_parquet;

pub(crate) mod reading;
pub use reading::{
    ColumnDecoderError, DecodeRange, DecoderFactory, DecompressorError, DecompressorFactory,
    IndexerFactory, MaterializerFactory, RangeCutterFactory, RowGroupFetcherFactory,
    RowGroupInjectorFactory, ScanEqualityPredicate, WorkerAllocator, materialize,
    pending_claim_bound, table_input, table_input_with_filter,
    table_input_with_filter_and_eq_predicates,
};

pub mod writing;
pub use writing::aggregate_file_stats;

mod metadata;
pub use metadata::{FileRowGroups, create_load_and_stage_spec};
pub use metadata::{file_row_groups_from_metadata, load_file_row_groups};

mod pushdown;
pub use pushdown::{PushedPredicate, equality_predicates, prune_parquet};

mod row_group_stats;
pub use row_group_stats::{
    RowGroupFilter, ScanOrder, bounds_eliminate, row_group_eliminated, row_group_filter_from,
    scan_order_from,
};

pub mod thrift;

pub(crate) mod types;
pub use types::arrow_map::{
    DECIMAL_FIXED_LEN, DecimalWriteStorage, LeafAnnotation, arrow_to_annotation,
    arrow_to_parquet_physical, decimal_write_storage,
};
pub use types::columns::{ColumnResolution, TableColumns};
pub use types::leaves::{
    ShreddedScalarPath, first_leaf, leaf_count, leaf_fields, variant_shredded_leaves,
    variant_value_leaf_is_semantically_null,
};
pub use types::metadata::{ColumnStatistics, RowGroupMetadata};
pub use types::page::{CompressedPage, DecompressedPage};
pub use types::requests::RowGroupBuffer;
pub use types::table::{
    Error as ParquetTableError, ParquetTable, is_variant_field, variant_extension_metadata,
};

mod values;
pub use values::{
    FileStats, PartitionValues, pivot_scalar, scalar_equal, scalar_values_equal,
    scalar_values_from_row,
};

#[cfg(test)]
pub(crate) mod test_utils {
    use crate::types::metadata::{QueryRowGroupMetadata, RowGroupMetadata, RowSelection};
    use crate::types::table::ParquetTable;
    use arrow_schema::Schema;
    use dispatch::io::{LocalFile, OpenFile};
    use roaring::RoaringTreemap;
    use std::sync::Arc;

    pub fn dummy_row_group() -> Arc<RowGroupMetadata> {
        Arc::new(RowGroupMetadata {
            open_file: OpenFile::Local(
                LocalFile::new(std::fs::File::open("/dev/null").unwrap()).unwrap(),
            ),
            schema: Arc::new(Schema::empty()),
            columns: vec![],
            statistics: Arc::default(),
            num_rows: 0,
            file_row_offset: 0,
            deleted_rows: None,
            file_row_group_idx: 0,
            live_decompressed_pages: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        })
    }

    /// The `deleted` rows, by position in the row group starting at the
    /// file's `file_row_offset`, as the row group's deleted rows; `None` when
    /// there are none.
    pub fn deleted_rows(file_row_offset: u64, deleted: &[u32]) -> Option<Arc<RoaringTreemap>> {
        (!deleted.is_empty()).then(|| {
            Arc::new(
                deleted
                    .iter()
                    .map(|row| file_row_offset + *row as u64)
                    .collect(),
            )
        })
    }

    pub fn dummy_metadata(selection: RowSelection) -> QueryRowGroupMetadata {
        let table = ParquetTable::new(vec![dummy_row_group()]);
        QueryRowGroupMetadata::new(&table, 0, selection)
    }
}
