use crate::thrift::general::CompressionCodec;
use crate::thrift::headers::PageHeader;
use crate::thrift::parquet_thrift::ThriftReadInputProtocol;
use crate::types::leaves::projected_leaves;
use crate::types::metadata::{ColumnChunkMeta, QueryRowGroupMetadata};
use crate::types::projection::Projection;
use bytes::Bytes;
use dispatch::io::{FileRange, OpenFile, ReadResponse};
use dispatch::memory::{MultiBufferReader, ReaderPosition, Segment, memory_ctx};

/// A row group's read as the fetcher issues it: for each projected leaf, the
/// pages the decompressed cache already held when the read was planned and the
/// byte ranges still to read. Built on the fetching worker right before its IO
/// (see [`from`](Self::from)), so the cached pages it pins are pinned only
/// while the read is in flight. Compressed-cache lookup, extent allocation and
/// physical reads happen in dispatch's worker-local requester.
pub struct RowGroupRequest {
    metadata: QueryRowGroupMetadata,
    open_file: OpenFile,
    /// Each projected leaf's chunk, in leaf order.
    columns: Vec<ColumnReference>,
}

enum ColumnReference {
    /// The chunk's parts in file order, tagged with the page codec its footer
    /// entry recorded so the buffer tags the column's parts with it.
    InFile {
        codec: CompressionCodec,
        parts: Vec<ColumnChunkReference>,
    },
    /// The file has no chunk for the leaf (see [`ColumnChunkMeta::absent`]):
    /// nothing is read for it, and its column of the buffer comes back empty.
    Absent,
}

enum ColumnChunkReference {
    /// A page the decompressed cache held: resolved, nothing to read.
    Cached(CachedColumnChunk),
    /// A stretch of whole pages the cache did not hold, read from the file.
    ToRead(FileRange),
}

/// One resolved piece of a column chunk, in file order, named by the form its
/// bytes are in:
///
/// - [`Compressed`](Self::Compressed): raw compressed bytes for the indexer to
///   Thrift-parse into pages. Dispatch has already resolved any compressed-cache
///   misses before constructing this value, so every byte run is pinned and
///   ready to read.
/// - [`Decompressed`](Self::Decompressed): a page served straight from the
///   decompressed cache before any IO was issued: its identity (`offset`,
///   `span`), its header (parsed back out of the hit - see
///   [`parse_page_header`]), and its decompressed bytes, pinned. The indexer
///   builds a `CompressedPage` from it directly, no parsing or decompression
///   needed.
pub enum CachedColumnChunk {
    Compressed {
        offset: usize,
        bytes: Vec<Bytes>,
    },
    Decompressed {
        offset: usize,
        span: usize,
        header: Box<PageHeader>,
        data: Vec<Bytes>,
    },
}

impl RowGroupRequest {
    /// Build a request for all projected columns in the given row group: each
    /// chunk is looked up in the decompressed cache, the pages it holds are
    /// taken, pinned, and the stretches between them become ranges to read.
    pub fn from(metadata_handle: QueryRowGroupMetadata, projection: &Projection) -> Self {
        let open_file = metadata_handle.get_metadata().open_file.clone();
        let columns = metadata_handle.columns();
        // The projection names top-level columns; fetch each one's run of leaf
        // chunks in this file's layout. The decoder resolves leaves the same
        // way, so a fetched chunk's position lines up with its decoder.
        let fields = metadata_handle.get_metadata().schema.fields();
        let leaves = projected_leaves(fields, &metadata_handle, projection);

        let planned = leaves
            .iter()
            .map(|&leaf| {
                let meta: &ColumnChunkMeta = &columns[leaf];
                if meta.absent {
                    return ColumnReference::Absent;
                }
                let offset = meta.dictionary_page_offset.unwrap_or(meta.data_page_offset) as usize;
                let len = meta.total_compressed_size as usize;
                ColumnReference::InFile {
                    codec: meta.codec,
                    parts: memory_ctx()
                        .decompressed_cache()
                        .get_range(&open_file, offset, len)
                        .into_iter()
                        .map(|segment| match segment {
                            Segment::Cached {
                                offset,
                                span,
                                header,
                                data,
                            } => ColumnChunkReference::Cached(CachedColumnChunk::Decompressed {
                                offset,
                                span,
                                header: Box::new(Self::parse_page_header(&header)),
                                data,
                            }),
                            Segment::Gap { offset, len } => {
                                ColumnChunkReference::ToRead(FileRange::new(offset, len))
                            }
                        })
                        .collect(),
                }
            })
            .collect();

        Self {
            metadata: metadata_handle,
            open_file,
            columns: planned,
        }
    }

    /// Parse a page's Thrift header back out of the bytes a decompressed-cache hit
    /// returned (see [`dispatch::memory::Segment::Cached`], which
    /// stores whatever the `Decompressor`
    /// serialized at insert time) - cheap, a few bytes, entirely in memory; no disk
    /// access and nothing to do with the (possibly much larger) page payload.
    fn parse_page_header(bytes: &[Bytes]) -> PageHeader {
        let mut position = ReaderPosition::default();
        let mut reader = MultiBufferReader::new(bytes, &mut position);
        let mut prot = ThriftReadInputProtocol::new(&mut reader);
        PageHeader::read_thrift_without_stats(&mut prot)
            .expect("a decompressed-cache hit's header round-trips: this cache wrote it")
    }

    pub fn open_file(&self) -> &OpenFile {
        &self.open_file
    }

    /// The byte ranges left to read, in file order within each leaf.
    pub fn file_ranges(&self) -> impl Iterator<Item = FileRange> + '_ {
        self.columns
            .iter()
            .flat_map(|column| match column {
                ColumnReference::InFile { parts, .. } => parts.as_slice(),
                ColumnReference::Absent => &[],
            })
            .filter_map(|part| match part {
                ColumnChunkReference::ToRead(range) => Some(*range),
                ColumnChunkReference::Cached(_) => None,
            })
    }

    /// Whether nothing is left to read: every projected leaf is absent from
    /// the file or entirely held by the decompressed cache, so the buffer is
    /// built with no response
    /// ([`into_row_group_buffer`](Self::into_row_group_buffer) with `None`).
    pub fn reads_nothing(&self) -> bool {
        self.file_ranges().next().is_none()
    }

    /// Pair this row group's domain metadata with dispatch's resolved
    /// `response`, `None` for a request that [reads nothing](Self::reads_nothing).
    /// Runs on the claiming worker (the fetcher emits from the worker that
    /// admitted the request), so the buffer records it as the decode owner.
    pub fn into_row_group_buffer(self, response: Option<ReadResponse>) -> RowGroupBuffer {
        let mut resolved = response
            .map(|response| response.into_locations())
            .unwrap_or_default()
            .into_iter();
        let columns = self
            .columns
            .into_iter()
            .map(|column| match column {
                ColumnReference::InFile { codec, parts } => ColumnBuffer::Resolved {
                    codec,
                    parts: parts
                        .into_iter()
                        .map(|part| match part {
                            ColumnChunkReference::Cached(part) => part,
                            ColumnChunkReference::ToRead(_) => {
                                let read = resolved
                                    .next()
                                    .expect("the response resolves every requested range");
                                CachedColumnChunk::Compressed {
                                    offset: read.offset,
                                    bytes: read.bytes,
                                }
                            }
                        })
                        .collect(),
                },
                ColumnReference::Absent => ColumnBuffer::Absent,
            })
            .collect();
        assert!(
            resolved.next().is_none(),
            "the response resolves only the requested ranges"
        );
        RowGroupBuffer {
            metadata: self.metadata,
            columns,
            worker_id: dispatch::worker::WORKER_IDX.get(),
        }
    }
}

pub struct RowGroupBuffer {
    pub metadata: QueryRowGroupMetadata,
    /// Each projected column's resolved bytes, in file order.
    pub columns: Vec<ColumnBuffer>,
    /// The worker every page of this row group returns to for decode. It must
    /// be the worker that claimed the row group: that worker holds the row
    /// group's decoder state and accounts for the claim (see
    /// `RowGroupFetcher`). The middle stages may run on stealing siblings, so
    /// the id is stamped at claim time, not by whoever runs a stage.
    pub worker_id: usize,
}

/// One projected column of a row group as the ring resolved it.
pub enum ColumnBuffer {
    /// The column's parts, in file order, tagged with the codec its compressed
    /// pages were written with (from the column chunk's footer metadata).
    Resolved {
        codec: CompressionCodec,
        parts: Vec<CachedColumnChunk>,
    },
    /// A leaf the file has no chunk for (see [`ColumnChunkMeta::absent`]):
    /// nothing was read, and the indexer stands in one page spanning every row
    /// so the column decodes as NULL.
    Absent,
}
