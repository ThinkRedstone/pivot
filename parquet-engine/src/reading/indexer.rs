//! Splits raw column buffers into individual compressed pages and sends them in an order
//! optimised for downstream decoding.
//!
//! The Indexer sits between IO (which delivers whole column chunks, or pages already
//! resolved straight from the decompressed cache - see
//! [`CachedColumnChunk`]) and the Decompressor
//! (which expects individual pages). For each column it walks its parts in file
//! order: a `Compressed` part's raw bytes are Thrift-parsed into pages same as
//! always; a `Decompressed` part is already one whole decompressed page, so it
//! becomes a `CompressedPage` directly, no parsing needed. When filtered indices are
//! present it attaches a [`FilterMask`] to each data page so the decoder knows which
//! rows to keep.
//!
//! ## Emission order
//!
//! Pages are expected to be emitted onto a **LIFO** channel, so the last page sent is the first one
//! received.
//! We exploit this by sending in reverse-priority order:
//!
//! 1. **Data pages** — interleaved across columns so the column with the fewest emitted rows
//!    goes next. This lets the decoder produce `RecordBatch`es as early as possible, keeping
//!    recently-decompressed pages hot in cache.
//! 2. **Dictionary pages** — sent last so they arrive first, ensuring the decoder has the
//!    dictionary before any RLE-dictionary-encoded data page.

use crate::thrift::general::{CompressionCodec, Encoding, PageType};
use crate::thrift::headers::{DataPageHeader, PageHeader};
use crate::thrift::parquet_thrift::{ParquetError, ThriftReadInputProtocol};
use crate::types::filter_mask::FilterMask;
use crate::types::metadata::{QueryRowGroupMetadata, RowSelection};
use crate::types::page::CompressedPage;
use crate::types::requests::{CachedColumnChunk, ColumnBuffer, RowGroupBuffer};
use bytes::Bytes;
use dispatch::DefaultUnaryFactory;
use dispatch::Sender;
use dispatch::Unary;
use dispatch::memory::{MultiBufferReader, ReaderPosition};

pub type IndexerFactory = DefaultUnaryFactory<Indexer>;

/// Splits a [`RowGroupBuffer`] into individual [`CompressedPage`]s.
#[derive(Default)]
pub struct Indexer {}

/// Running position while emitting one column's pages: how many data pages (and
/// their rows) have been emitted so far. Threaded across all of a column's parts
/// so `page_idx` and filter-mask row offsets stay continuous regardless of which
/// parts came pre-resolved from the decompressed cache.
#[derive(Default)]
struct PageCursor {
    data_page_idx: usize,
    row_offset: u32,
}

impl PageCursor {
    /// This page's `(page_idx, row_offset)`, then advance past it if it's a data
    /// page (dictionary pages don't count).
    fn advance(&mut self, header: &PageHeader) -> (usize, u32) {
        let at = (self.data_page_idx, self.row_offset);
        if header.r#type == PageType::DATA_PAGE {
            self.data_page_idx += 1;
            self.row_offset += header.data_page_num_values() as u32;
        }
        at
    }
}

/// A page's body, either still compressed (needs the decompressor) or already
/// decompressed (a decompressed-cache hit, resolved before this row group's IO was
/// even issued).
enum PagePayload {
    Compressed(Vec<Bytes>),
    Decompressed(Vec<Bytes>),
}

/// Carries the context for emitting one column's pages: the column's
/// identity and codec, the query's view of the row group, the worker that
/// claimed it, and the running [`PageCursor`] threaded across the column's
/// parts.
struct ColumnPageBuilder {
    col_idx: usize,
    codec: CompressionCodec,
    query_row_group_metadata: QueryRowGroupMetadata,
    worker_id: usize,
    cursor: PageCursor,
}

impl ColumnPageBuilder {
    /// Build one `CompressedPage`, attaching the current query's filter mask
    /// and advancing the cursor.
    fn build_page(
        &mut self,
        file_offset: usize,
        span: usize,
        header: PageHeader,
        payload: PagePayload,
    ) -> CompressedPage {
        let (page_idx, row_offset) = self.cursor.advance(&header);
        // A whole row group carries no mask; the materializer's row groups
        // mask down to their surviving rows.
        let filter_mask = (header.r#type == PageType::DATA_PAGE)
            .then(|| {
                let page_end = row_offset + header.data_page_num_values() as u32;
                match self.query_row_group_metadata.selection() {
                    RowSelection::All => None,
                    RowSelection::Indices(indices) => {
                        Some(FilterMask::new(row_offset, page_end, indices))
                    }
                }
            })
            .flatten();
        let (data, decompressed) = match payload {
            PagePayload::Compressed(bytes) => (bytes, None),
            PagePayload::Decompressed(bytes) => (Vec::new(), Some(bytes)),
        };

        CompressedPage {
            // This is the claimer, not necessarily this worker: the indexer
            // often runs on a stealing sibling, but the decode must return to
            // the worker that claimed the row group (it owns the decoder
            // state and the claim accounting).
            worker_id: self.worker_id,
            codec: self.codec,
            row_group: self.query_row_group_metadata.clone(),
            column_idx: self.col_idx,
            file_offset,
            span,
            page_idx,
            first_row: row_offset,
            data,
            decompressed,
            filter_mask,
            header,
        }
    }

    /// Walk a `Compressed` part's raw byte stream and Thrift-parse it into pages,
    /// appending each to `pages`. Each iteration reads a page header, then slices out
    /// the compressed payload via [`MultiBufferReader::copy_out_buffers`] (zero-copy).
    fn parse_compressed_pages(
        &mut self,
        part_offset: usize,
        buffers: &[Bytes],
        pages: &mut Vec<CompressedPage>,
    ) -> Result<(), ParquetError> {
        let mut position = ReaderPosition::default();
        let buffers_length = buffers.len();
        let mut reader = MultiBufferReader::new(buffers, &mut position);
        loop {
            // Bytes consumed so far = this page's offset within the part, which the
            // part's own file offset turns into the page's absolute file offset.
            let page_start = reader.consumed();
            let header = {
                let mut prot = ThriftReadInputProtocol::new(&mut reader);
                PageHeader::read_thrift_without_stats(&mut prot)?
            };
            let data = reader.copy_out_buffers(header.compressed_page_size as usize);
            let span = reader.consumed() - page_start;

            pages.push(self.build_page(
                part_offset + page_start,
                span,
                header,
                PagePayload::Compressed(data),
            ));

            if reader.remaining_in_cur() == 0
                && reader.position().buffer_index == buffers_length - 1
            {
                break;
            }
        }
        Ok(())
    }
}

/// Build one column's pages, in file order, from its resolved parts.
///
/// A column the file has no chunk for gets one stand-in data page spanning
/// every row of the row group, carrying no bytes: it moves through the
/// pipeline like any page, so the row group's ranges become ready and the
/// column's decoder emits a NULL per row.
fn build_column_pages(
    col_idx: usize,
    query_row_group_metadata: QueryRowGroupMetadata,
    column: ColumnBuffer,
    worker_id: usize,
) -> Result<Vec<CompressedPage>, ParquetError> {
    let (codec, parts) = match column {
        ColumnBuffer::Resolved { codec, parts } => (codec, parts),
        ColumnBuffer::Absent => {
            let mut builder = ColumnPageBuilder {
                col_idx,
                codec: CompressionCodec::UNCOMPRESSED,
                query_row_group_metadata,
                worker_id,
                cursor: PageCursor::default(),
            };
            let rows = builder.query_row_group_metadata.num_rows().max(0) as i32;
            let header = PageHeader {
                r#type: PageType::DATA_PAGE,
                uncompressed_page_size: 0,
                compressed_page_size: 0,
                crc: None,
                data_page_header: Some(DataPageHeader {
                    num_values: rows,
                    encoding: Encoding::PLAIN,
                    definition_level_encoding: Encoding::RLE,
                    repetition_level_encoding: Encoding::RLE,
                    statistics: None,
                }),
                index_page_header: None,
                dictionary_page_header: None,
                data_page_header_v2: None,
            };
            return Ok(vec![builder.build_page(
                0,
                0,
                header,
                PagePayload::Decompressed(Vec::new()),
            )]);
        }
    };
    let mut builder = ColumnPageBuilder {
        col_idx,
        codec,
        query_row_group_metadata,
        worker_id,
        cursor: PageCursor::default(),
    };
    let mut pages = Vec::with_capacity(128);
    for part in parts {
        match part {
            CachedColumnChunk::Compressed { offset, bytes } => {
                builder.parse_compressed_pages(offset, &bytes, &mut pages)?
            }
            CachedColumnChunk::Decompressed {
                offset,
                span,
                header,
                data,
            } => pages.push(builder.build_page(
                offset,
                span,
                *header,
                PagePayload::Decompressed(data),
            )),
        }
    }
    Ok(pages)
}

impl Unary<RowGroupBuffer, CompressedPage> for Indexer {
    fn consume(
        &mut self,
        buffer: RowGroupBuffer,
        sender: &mut dyn Sender<CompressedPage>,
        _io: &mut dispatch::OperatorIO,
    ) -> dispatch::UnaryResult<()> {
        let mut pages_per_column: Vec<_> = buffer
            .columns
            .into_iter()
            .enumerate()
            .map(|(col_idx, column)| {
                build_column_pages(col_idx, buffer.metadata.clone(), column, buffer.worker_id)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(crate::op_err)?;

        // We want to send the pages out in an order that will be best for decoding. If we were to,
        // for example, send all of column A pages and then only column B pages, we would be unable to
        // send out record batches until AFTER we finished ALL of column A. This is horrible from a
        // cache standpoint, as it means pages we touched for decompressing won't be in cache for
        // the decoder.
        // Therefore, we try to send out pages in "intelligently"- first we send out dict pages,
        // and then we try sending out pages in an order where we will be able to send out record
        // batches as soon as possible. We do this by always selecting pages from the column that
        // has sent out the minimum amount of records so far
        let dict_pages: Vec<_> = pages_per_column
            .iter_mut()
            .flat_map(|pages| {
                pages
                    .extract_if(..pages.len(), |p| {
                        p.header.r#type == PageType::DICTIONARY_PAGE
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        let mut rows_emitted = vec![0u64; pages_per_column.len()];

        while let Some(col) = (0..pages_per_column.len())
            .filter(|i| !pages_per_column[*i].is_empty())
            .min_by_key(|i| rows_emitted[*i])
        {
            let page = pages_per_column[col].pop().unwrap();
            rows_emitted[col] += page.header.data_page_num_values() as u64;
            sender.send(page)?;
        }

        // We send the dict pages at the end, so they'll be received first (remember, we're LIFO!)
        for page in dict_pages {
            sender.send(page)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::dummy_metadata;
    use crate::thrift::general::Encoding;
    use crate::thrift::headers::{
        DataPageHeader as ThriftDataPageHeader, DictionaryPageHeader as ThriftDictionaryPageHeader,
    };
    use crate::thrift::parquet_thrift::{ThriftCompactOutputProtocol, WriteThrift};
    use bytes::Bytes;
    use dispatch::test_utils::run_unary;

    /// Serialize a dispatch `PageHeader` to Thrift compact bytes.
    fn serialize_header(header: &PageHeader) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let mut prot = ThriftCompactOutputProtocol::new(&mut buf);
            header.write_thrift(&mut prot).unwrap();
        }
        buf
    }

    /// Build a data page header for serialization.
    fn data_page_header(num_values: i32, compressed_size: i32) -> PageHeader {
        PageHeader {
            r#type: PageType::DATA_PAGE,
            uncompressed_page_size: compressed_size,
            compressed_page_size: compressed_size,
            crc: None,
            data_page_header: Some(ThriftDataPageHeader {
                num_values,
                encoding: Encoding::PLAIN,
                definition_level_encoding: Encoding::RLE,
                repetition_level_encoding: Encoding::RLE,
                statistics: None,
            }),
            index_page_header: None,
            dictionary_page_header: None,
            data_page_header_v2: None,
        }
    }

    /// Build a dictionary page header for serialization.
    fn dict_page_header(num_values: i32, compressed_size: i32) -> PageHeader {
        PageHeader {
            r#type: PageType::DICTIONARY_PAGE,
            uncompressed_page_size: compressed_size,
            compressed_page_size: compressed_size,
            crc: None,
            data_page_header: None,
            index_page_header: None,
            dictionary_page_header: Some(ThriftDictionaryPageHeader {
                num_values,
                encoding: Encoding::PLAIN_DICTIONARY,
                is_sorted: None,
            }),
            data_page_header_v2: None,
        }
    }

    /// Concatenate serialized page headers + payloads into a single `Compressed` part.
    fn make_column_buffer(pages: &[(PageHeader, Vec<u8>)]) -> Vec<CachedColumnChunk> {
        let mut buf = Vec::new();
        for (header, payload) in pages {
            buf.extend_from_slice(&serialize_header(header));
            buf.extend_from_slice(payload);
        }
        vec![CachedColumnChunk::Compressed {
            offset: 0,
            bytes: vec![Bytes::from(buf)],
        }]
    }

    fn make_row_group_buffer(
        columns: Vec<Vec<CachedColumnChunk>>,
        selection: RowSelection,
    ) -> RowGroupBuffer {
        RowGroupBuffer {
            metadata: dummy_metadata(selection),
            columns: columns
                .into_iter()
                .map(|parts| ColumnBuffer::Resolved {
                    codec: CompressionCodec::SNAPPY,
                    parts,
                })
                .collect(),
            worker_id: 0,
        }
    }

    /// Single data page → 1 CompressedPage with correct column_idx, page_idx, and payload.
    #[test]
    fn test_single_data_page() {
        let payload = vec![1u8, 2, 3, 4, 5];
        let header = data_page_header(10, payload.len() as i32);
        let col = make_column_buffer(&[(header, payload.clone())]);
        let buffer = make_row_group_buffer(vec![col], RowSelection::All);

        let pages = run_unary(Indexer {}, vec![buffer]);

        assert_eq!(pages.len(), 1);
        let page = &pages[0];
        assert_eq!(page.column_idx, 0);
        assert_eq!(page.page_idx, 0);
        assert_eq!(page.header.r#type, PageType::DATA_PAGE);
        assert_eq!(page.header.data_page_num_values(), 10);
        let data: Vec<u8> = page.data.iter().flat_map(|b| b.to_vec()).collect();
        assert_eq!(data, payload);
        assert!(page.filter_mask.is_none());
    }

    /// Three data pages → all 3 emitted with correct page_idx and payload.
    /// Note: .pop() reverses order within a column, so pages come out last-first.
    #[test]
    fn test_multiple_data_pages() {
        let payloads: Vec<Vec<u8>> = vec![vec![10, 11], vec![20, 21, 22], vec![30]];
        let pages: Vec<_> = payloads
            .iter()
            .map(|p| (data_page_header(5, p.len() as i32), p.clone()))
            .collect();
        let col = make_column_buffer(&pages);
        let buffer = make_row_group_buffer(vec![col], RowSelection::All);

        let pages = run_unary(Indexer {}, vec![buffer]);

        assert_eq!(pages.len(), 3);
        let mut page_indices: Vec<usize> = pages.iter().map(|p| p.page_idx).collect();
        page_indices.sort();
        assert_eq!(page_indices, vec![0, 1, 2]);
        for page in &pages {
            assert_eq!(page.header.r#type, PageType::DATA_PAGE);
            let data: Vec<u8> = page.data.iter().flat_map(|b| b.to_vec()).collect();
            assert_eq!(data, payloads[page.page_idx]);
        }
    }

    /// Dict page + 2 data pages → dict page sent last (LIFO: received first).
    #[test]
    fn test_dict_pages_sent_last() {
        let dict_payload = vec![0xDD];
        let data_payload_0 = vec![0xA0];
        let data_payload_1 = vec![0xA1];
        let col = make_column_buffer(&[
            (dict_page_header(3, dict_payload.len() as i32), dict_payload),
            (
                data_page_header(10, data_payload_0.len() as i32),
                data_payload_0,
            ),
            (
                data_page_header(10, data_payload_1.len() as i32),
                data_payload_1,
            ),
        ]);
        let buffer = make_row_group_buffer(vec![col], RowSelection::All);

        let pages = run_unary(Indexer {}, vec![buffer]);

        assert_eq!(pages.len(), 3);
        assert_eq!(pages[0].header.r#type, PageType::DATA_PAGE);
        assert_eq!(pages[1].header.r#type, PageType::DATA_PAGE);
        assert_eq!(pages[2].header.r#type, PageType::DICTIONARY_PAGE);
    }

    /// Two columns, each with 1 data page → pages interleaved by column.
    #[test]
    fn test_two_columns_interleaved() {
        let col_a = make_column_buffer(&[(data_page_header(100, 2), vec![0xAA, 0xAA])]);
        let col_b = make_column_buffer(&[(data_page_header(100, 2), vec![0xBB, 0xBB])]);
        let buffer = make_row_group_buffer(vec![col_a, col_b], RowSelection::All);

        let pages = run_unary(Indexer {}, vec![buffer]);

        assert_eq!(pages.len(), 2);
        let col_ids: Vec<usize> = pages.iter().map(|p| p.column_idx).collect();
        assert!(col_ids.contains(&0));
        assert!(col_ids.contains(&1));
    }

    /// Two columns with unequal pages: col A has 2 pages (50 rows each), col B has 1 page
    /// (100 rows). Min-rows-emitted interleaving: A(50), B(100), A(50).
    #[test]
    fn test_interleave_unequal_pages() {
        let col_a = make_column_buffer(&[
            (data_page_header(50, 1), vec![0xA0]),
            (data_page_header(50, 1), vec![0xA1]),
        ]);
        let col_b = make_column_buffer(&[(data_page_header(100, 1), vec![0xB0])]);
        let buffer = make_row_group_buffer(vec![col_a, col_b], RowSelection::All);

        let pages = run_unary(Indexer {}, vec![buffer]);

        assert_eq!(pages.len(), 3);
        let col_ids: Vec<usize> = pages.iter().map(|p| p.column_idx).collect();
        assert_eq!(col_ids, vec![0, 1, 0]);
    }

    /// With an index selection, data pages get a FilterMask; dict pages don't.
    #[test]
    fn test_filter_mask_on_data_pages_only() {
        let col = make_column_buffer(&[
            (dict_page_header(3, 1), vec![0xDD]),
            (data_page_header(10, 1), vec![0xAA]),
        ]);
        let buffer = make_row_group_buffer(vec![col], RowSelection::Indices(vec![2, 5].into()));

        let pages = run_unary(Indexer {}, vec![buffer]);

        assert_eq!(pages.len(), 2);
        let data_page = pages
            .iter()
            .find(|p| p.header.r#type == PageType::DATA_PAGE)
            .unwrap();
        let dict_page = pages
            .iter()
            .find(|p| p.header.r#type == PageType::DICTIONARY_PAGE)
            .unwrap();
        assert!(data_page.filter_mask.is_some());
        assert_eq!(data_page.filter_mask.as_ref().unwrap().rows(), 2);
        assert!(dict_page.filter_mask.is_none());
    }

    #[test]
    fn test_no_filter_mask_without_indices() {
        let col = make_column_buffer(&[(data_page_header(10, 2), vec![0xAA, 0xBB])]);
        let buffer = make_row_group_buffer(vec![col], RowSelection::All);

        let pages = run_unary(Indexer {}, vec![buffer]);

        assert_eq!(pages.len(), 1);
        assert!(pages[0].filter_mask.is_none());
    }

    #[test]
    fn test_dict_does_not_increment_page_idx() {
        let col = make_column_buffer(&[
            (dict_page_header(3, 1), vec![0xDD]),
            (data_page_header(5, 1), vec![0xA0]),
            (data_page_header(5, 1), vec![0xA1]),
        ]);
        let buffer = make_row_group_buffer(vec![col], RowSelection::All);

        let pages = run_unary(Indexer {}, vec![buffer]);

        let mut data_page_indices: Vec<usize> = pages
            .iter()
            .filter(|p| p.header.r#type == PageType::DATA_PAGE)
            .map(|p| p.page_idx)
            .collect();
        data_page_indices.sort();
        assert_eq!(data_page_indices, vec![0, 1]);
    }

    /// A `Decompressed` part (a decompressed-cache hit) becomes a `CompressedPage`
    /// with no parsing: its bytes flow through as `decompressed`, not `data`.
    #[test]
    fn test_decompressed_part_skips_parsing() {
        let header = data_page_header(7, 0);
        let data = vec![Bytes::from(vec![9u8, 9, 9])];
        let col = vec![CachedColumnChunk::Decompressed {
            offset: 42,
            span: 20,
            header: Box::new(header),
            data: data.clone(),
        }];
        let buffer = make_row_group_buffer(vec![col], RowSelection::All);

        let pages = run_unary(Indexer {}, vec![buffer]);

        assert_eq!(pages.len(), 1);
        assert_eq!(pages[0].file_offset, 42);
        assert_eq!(pages[0].span, 20);
        assert!(pages[0].data.is_empty());
        assert_eq!(pages[0].decompressed, Some(data));
    }

    /// `page_idx` and filter-mask row offsets stay continuous across a
    /// `Compressed` part followed by a `Decompressed` one in the same column.
    #[test]
    fn test_cursor_continuous_across_mixed_parts() {
        let payload = vec![0xA0];
        let mut parts =
            make_column_buffer(&[(data_page_header(10, payload.len() as i32), payload)]);
        parts.push(CachedColumnChunk::Decompressed {
            offset: 1000,
            span: 5,
            header: Box::new(data_page_header(10, 0)),
            data: vec![Bytes::from(vec![1u8])],
        });
        let buffer = make_row_group_buffer(vec![parts], RowSelection::Indices(vec![15].into()));

        let pages = run_unary(Indexer {}, vec![buffer]);

        assert_eq!(pages.len(), 2);
        let mut by_idx: Vec<_> = pages.iter().collect();
        by_idx.sort_by_key(|p| p.page_idx);
        assert_eq!(by_idx[0].page_idx, 0);
        assert_eq!(by_idx[1].page_idx, 1);
        // Row 15 falls in the second page's [10, 20) range, not the first's [0, 10).
        assert!(by_idx[0].filter_mask.as_ref().unwrap().all_false());
        assert_eq!(by_idx[1].filter_mask.as_ref().unwrap().rows(), 1);
    }
}
