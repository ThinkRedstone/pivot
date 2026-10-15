//! Types representing Parquet pages as they move through the decompression
//! pipeline.
//!
//! A Parquet column chunk is made up of pages — an optional dictionary page
//! followed by one or more data pages. This module defines the structs that
//! carry page data (and associated metadata) from the IO reader through the
//! decompressor and into the decoder:
//!
//! 1. [`CompressedPage`] — raw bytes straight from disk, before decompression.
//! 2. [`DecompressedPage`] / [`DecompressedPageType`] — after decompression,
//!    classified as dictionary, data, or skipped.
//! 3. [`DataPage`] — the data-page–specific subset used by decoders.
//!
//! Each page optionally carries a [`FilterMask`] so decoders can skip rows
//! that were filtered out upstream.

use crate::thrift::general::CompressionCodec;
use crate::thrift::headers::{DataPageHeader, DictionaryPageHeader, PageHeader};
use crate::types::filter_mask::FilterMask;
use crate::types::metadata::QueryRowGroupMetadata;
use bytes::Bytes;
use dispatch::WorkerIdOutput;
use std::fmt::{Debug, Formatter};

/// A page read from disk whose payload is still compressed.
///
/// Produced by the IO reader and consumed by the decompressor. Carries the
/// full Thrift [`PageHeader`] (compressed and uncompressed sizes, encoding
/// info) plus the column chunk's compression codec.
pub struct CompressedPage {
    /// Worker that owns this page (for routing back through the pipeline).
    pub worker_id: usize,
    /// Compression codec of the page body, from the column chunk's footer
    /// metadata (page headers don't carry it).
    pub codec: CompressionCodec,
    /// Row-group-level query metadata (includes filtered indices).
    pub row_group: QueryRowGroupMetadata,
    /// Column index within the *projected* set (the position in
    /// [`RowGroupBuffer::columns`](crate::types::requests::RowGroupBuffer),
    /// used to route the decompressed page back to its column decoder).
    pub column_idx: usize,
    /// Absolute byte offset of this page in the file. Unique per page, so it is
    /// the decompressed cache's key.
    pub file_offset: usize,
    /// This page's total on-disk size (header bytes + compressed payload). Together
    /// with `file_offset`, this is the decompressed cache's key length - it's what
    /// lets a cache lookup tell this page apart from a differently-sized one
    /// written at the same offset by an earlier version of the file.
    pub span: usize,
    /// Sequential page index within the column chunk.
    pub page_idx: usize,
    /// The row group row of a data page's first value; zero for a
    /// dictionary page.
    pub first_row: u32,
    /// Thrift page header (type, sizes, encoding info).
    pub header: PageHeader,
    /// Raw compressed byte buffers making up the page body. Empty when
    /// `decompressed` is `Some` - the page's bytes live there instead.
    pub data: Vec<Bytes>,
    /// This page's bytes, already decompressed - set when a
    /// [`DecompressedCache`](dispatch::memory::DecompressedCache) range lookup found
    /// it before any IO was issued (see
    /// [`requests::CachedColumnChunk`](crate::types::requests::CachedColumnChunk)).
    /// `None` for a normal compressed page.
    pub decompressed: Option<Vec<Bytes>>,
    /// Optional row-level filter mask to propagate to the decoder.
    pub filter_mask: Option<FilterMask>,
}

/// A decompressed data page ready for decoding.
///
/// Contains the page header, the uncompressed byte buffers, and an optional
/// [`FilterMask`]. Decoders use [`rows`](Self::rows) to determine how many
/// output rows to produce.
pub struct DataPage {
    /// Parsed data-page header (num values, encoding, etc.).
    pub header: DataPageHeader,
    /// Uncompressed page body buffers.
    pub data: Vec<Bytes>,
    /// Row-level filter mask; when present, only `true` runs are decoded.
    pub filter_mask: Option<FilterMask>,
}

impl DataPage {
    /// Number of rows this page will produce.
    ///
    /// If a filter mask is present, returns the number of kept rows;
    /// otherwise falls back to the header's `num_values`.
    pub fn rows(&self) -> usize {
        self.filter_mask
            .as_ref()
            .map(|f| f.rows())
            .unwrap_or(self.header.num_values as usize)
    }
}

/// The payload of a decompressed page, classified by kind.
///
/// After decompression, a page falls into one of three categories:
/// - `Dict` — a dictionary page that decoders use to build a lookup table.
/// - `Data` — a regular data page with rows to decode.
/// - `SkippedData` — a data page whose rows were entirely filtered out; only
///   the header is retained (for bookkeeping) and no bytes are decoded.
pub enum DecompressedPageType {
    /// Dictionary page — defines the value dictionary for the column chunk.
    Dict {
        header: DictionaryPageHeader,
        data: Vec<Bytes>,
    },
    /// Data page with rows to decode.
    Data(DataPage),
    /// Data page that was entirely filtered out; no payload bytes are kept.
    SkippedData { header: DataPageHeader },
}

impl Debug for DecompressedPageType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            DecompressedPageType::Dict { .. } => "dict",
            DecompressedPageType::Data(_) => "data",
            DecompressedPageType::SkippedData { .. } => "skipped",
        })
    }
}

/// A page that has been decompressed and is ready for decoding.
///
/// Wraps a [`DecompressedPageType`] together with routing metadata (worker ID,
/// row group, column, and page index) so the downstream decoder can correlate
/// the page with the correct column builder.
pub struct DecompressedPage {
    /// Worker that owns this page.
    pub worker_id: usize,
    /// Row-group-level query metadata.
    pub query_row_group_metadata: QueryRowGroupMetadata,
    /// Column index within the row group.
    pub column_idx: usize,
    /// Sequential page index within the column chunk.
    pub idx: usize,
    /// The row group row of a data page's first value; zero for a
    /// dictionary page.
    pub first_row: u32,
    /// The decompressed page payload (dict, data, or skipped).
    pub data: DecompressedPageType,
}

impl WorkerIdOutput for DecompressedPage {
    fn worker_id(&self) -> usize {
        self.worker_id
    }
}
