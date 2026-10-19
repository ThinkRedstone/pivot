//! Decodes decompressed Parquet pages into Arrow [`RecordBatch`]es.
//!
//! The Decoder is the final stage of the Parquet pipeline. Its input is a
//! stream of [`DecodeRange`] jobs: a row group's rows are cut into ranges by
//! the [`RangeCutter`](super::range_cutter::RangeCutter) on the worker that
//! claimed the row group, and each range is decoded by whichever worker
//! takes it, from the pages the range carries. A worker following its own
//! row group range after range decodes it as one stream; a worker that
//! took a range from a peer skips into the range's first pages.

use crate::reading::range_cutter::DecodeRange;
use crate::reading::record_batch_metadata::with_row_group;
use arrow_array::{ArrayRef, RecordBatch, Scalar};
use arrow_buffer::BooleanBuffer;
use dispatch::Sender;
use dispatch::WorkStatus;
use dispatch::arrays::accumulator::BatchAccumulator;
use dispatch::memory::SlabAllocator;
use dispatch::{Unary, UnaryFactory};
use row_group_decoder::DecodedBatch;
use std::sync::{Arc, Mutex};

pub(crate) mod leaf_decoders;

mod column_decoder;
pub use column_decoder::Error as ColumnDecoderError;

mod row_group_decoder;
pub(crate) use row_group_decoder::DecodePlan;
pub use row_group_decoder::RowGroupDecoder;

/// A pushed-down equality predicate (`column == value`) used for dictionary
/// pruning at scan time. When a row group's dictionary for the compared column
/// excludes `value` (and the column chunk is fully dictionary encoded), the row
/// group can emit no rows and is dropped without decoding its data pages.
#[derive(Clone, Debug)]
pub struct ScanEqualityPredicate {
    /// The top-level column the comparison reads, indexing the table's full
    /// schema. For a variant path this is the variant column.
    pub column_idx: usize,
    /// The object-field path inside the variant column
    /// (`CAST(col->'a'->'b' AS T) = value`), empty for a plain column
    /// comparison. A path predicate applies to the shredded typed leaf that
    /// path resolves to in each file.
    pub path: Vec<String>,
    pub value: Scalar<ArrayRef>,
}

/// One worker's slab allocator for one scan stage: its range cutter's
/// dictionaries, or its decoder's batches. Taken from the ring on first use,
/// on the worker; the lock is never contended, a worker runs one stage at a
/// time.
pub struct WorkerAllocator {
    allocator: Mutex<Option<SlabAllocator>>,
}

impl WorkerAllocator {
    pub fn new() -> Self {
        Self {
            allocator: Mutex::new(None),
        }
    }

    pub fn with<R>(&self, f: impl FnOnce(&mut SlabAllocator) -> R) -> R {
        let mut allocator = self.allocator.lock().unwrap();
        f(allocator.get_or_insert_with(|| SlabAllocator::new(true)))
    }
}

impl Default for WorkerAllocator {
    fn default() -> Self {
        Self::new()
    }
}

/// Factory for creating [`Decoder`] instances, one per worker thread.
pub struct DecoderFactory {
    /// Maximum number of rows per output [`RecordBatch`].
    pub batch_size: usize,
    /// Whether to append row-group-id and row-index metadata columns to each
    /// output batch (used by the materializer path).
    pub add_row_group_metadata: bool,
    /// The worker's allocator for decoded batches.
    pub allocator: Arc<WorkerAllocator>,
}

impl UnaryFactory<DecodeRange, RecordBatch> for DecoderFactory {
    type Unary = Decoder;

    fn build_unary(self) -> Self::Unary {
        Decoder::new(self.batch_size, self.add_row_group_metadata, self.allocator)
    }
}

/// Decodes [`DecodeRange`]s and emits Arrow [`RecordBatch`]es.
///
/// Works one range at a time, and takes the next only once the current one
/// is emitted, so the ranges it has not started stay in the channel for
/// idle peers to take. A row group's decoder is kept between ranges: the
/// next range of the same row group usually continues it, and it is dropped
/// once every range of the row group is decoded, on any worker.
///
/// A batch the table format deleted rows of does not leave as the runt the
/// drop would make of it: its live rows are gathered, as a filter's survivors
/// are, and leave in full batches. The rows leave in the order they were
/// decoded, so a batch with nothing deleted goes after the live rows held.
///
/// There is no table-wide output schema here: each [`RowGroupDecoder`] derives
/// its own from its file, because a variant column's physical layout (its
/// shredded leaves) can differ file to file.
pub struct Decoder {
    batch_size: usize,
    /// Slab allocator for decoded Arrow buffers, the worker's own.
    allocator: Arc<WorkerAllocator>,
    /// The decoder of the range being decoded.
    active: Option<RowGroupDecoder>,
    /// Decoders that finished a range, kept for the row group's next range.
    idle: Vec<RowGroupDecoder>,
    add_row_group_metadata: bool,
    /// Accumulates the live rows of partially deleted batches until a full
    /// batch of them is ready, all of one row group: a batch carries one.
    live_row_accumulator: Option<LiveRowAccumulator>,
}

/// The live rows held back, all of row group `group`.
struct LiveRowAccumulator {
    group: usize,
    rows: BatchAccumulator,
}

impl Decoder {
    pub fn new(
        batch_size: usize,
        add_row_group_metadata: bool,
        allocator: Arc<WorkerAllocator>,
    ) -> Self {
        Self {
            batch_size,
            allocator,
            active: None,
            idle: Vec::new(),
            add_row_group_metadata,
            live_row_accumulator: None,
        }
    }

    /// Sends `batch`, whose rows are all of row group `group`.
    fn send(
        &self,
        batch: RecordBatch,
        group: usize,
        sender: &mut dyn Sender<RecordBatch>,
    ) -> dispatch::UnaryResult<()> {
        let batch = if self.add_row_group_metadata {
            with_row_group(batch, group)
        } else {
            batch
        };
        sender.send(batch)?;
        Ok(())
    }

    /// Sends the live rows held, if any.
    fn flush_live_rows(
        &mut self,
        sender: &mut dyn Sender<RecordBatch>,
    ) -> dispatch::UnaryResult<()> {
        if let Some(accumulator) = &mut self.live_row_accumulator
            && !accumulator.rows.is_empty()
        {
            let group = accumulator.group;
            let batch = self
                .allocator
                .with(|allocator| accumulator.rows.take_batch(allocator))?;
            self.send(batch, group, sender)?;
        }
        Ok(())
    }

    /// Holds the rows of `batch` whose bit of `live_rows` is set, of row
    /// group `group`, after the live rows held, sending every full batch that
    /// makes. A deletion vector leaves most rows live, so they mostly come in
    /// runs, which the accumulator copies as such.
    fn hold_live_rows(
        &mut self,
        batch: &RecordBatch,
        live_rows: &BooleanBuffer,
        group: usize,
        sender: &mut dyn Sender<RecordBatch>,
    ) -> dispatch::UnaryResult<()> {
        let other_schema = self
            .live_row_accumulator
            .as_ref()
            .is_some_and(|accumulator| accumulator.rows.schema() != &batch.schema());
        let other_group = self
            .live_row_accumulator
            .as_ref()
            .is_some_and(|accumulator| accumulator.group != group);
        if other_schema || other_group {
            self.flush_live_rows(sender)?;
        }
        if other_schema {
            self.live_row_accumulator = None;
        }
        let mut full_batches = Vec::new();
        let Self {
            live_row_accumulator,
            allocator,
            ..
        } = self;
        allocator.with(|allocator| {
            let accumulator = live_row_accumulator.get_or_insert_with(|| LiveRowAccumulator {
                group,
                rows: BatchAccumulator::new(batch.schema(), allocator),
            });
            accumulator.group = group;
            accumulator
                .rows
                .append_batch_by_mask_flushing::<dispatch::UnaryError>(
                    batch,
                    live_rows,
                    allocator,
                    &mut |full| {
                        full_batches.push(full);
                        Ok(())
                    },
                )
        })?;
        for full in full_batches {
            self.send(full, group, sender)?;
        }
        Ok(())
    }

    /// Emits one batch of the active range, retiring the range once its
    /// rows are all emitted. Returns whether a batch was sent.
    fn produce_batch(
        &mut self,
        sender: &mut dyn Sender<RecordBatch>,
    ) -> dispatch::UnaryResult<bool> {
        let Some(decoder) = self.active.as_mut() else {
            return Ok(false);
        };
        let Some(DecodedBatch { batch, live_rows }) = self
            .allocator
            .with(|allocator| decoder.try_read(allocator))
            .map_err(crate::op_err)?
        else {
            return Ok(false);
        };
        let group = decoder.row_group_index();
        if decoder.exhausted() {
            let mut decoder = self.active.take().unwrap();
            decoder.finish_range().decoded();
            self.idle.push(decoder);
            // A row group whose ranges are all decoded gets no further range
            // on any worker; its decoders go, and with them the pages they
            // still hold.
            self.idle.retain(|idle| !idle.row_group_done());
        }
        match live_rows {
            Some(live_rows) => self.hold_live_rows(&batch, &live_rows, group, sender)?,
            None => {
                self.flush_live_rows(sender)?;
                self.send(batch, group, sender)?;
            }
        }
        Ok(true)
    }
}

impl Unary<DecodeRange, RecordBatch> for Decoder {
    fn consume(
        &mut self,
        range: DecodeRange,
        output: &mut dyn Sender<RecordBatch>,
        _io: &mut dispatch::OperatorIO,
    ) -> dispatch::UnaryResult<()> {
        assert!(self.active.is_none(), "a decoder takes one range at a time");
        let row_group_index = range.metadata().row_group_index;
        let mut decoder = match self
            .idle
            .iter()
            .position(|idle| idle.row_group_index() == row_group_index)
        {
            Some(pos) => self.idle.swap_remove(pos),
            None => RowGroupDecoder::new(&range, self.batch_size, self.add_row_group_metadata)
                .map_err(crate::op_err)?,
        };
        decoder.attach(range).map_err(crate::op_err)?;
        self.active = Some(decoder);
        self.produce_batch(output)?;
        Ok(())
    }

    fn ready_for_more_work(&mut self) -> bool {
        self.active.is_none()
    }

    fn run(&mut self, sender: &mut dyn Sender<RecordBatch>) -> dispatch::UnaryResult<WorkStatus> {
        if self.produce_batch(sender)? {
            Ok(WorkStatus::Ran)
        } else {
            Ok(WorkStatus::Pending)
        }
    }

    fn finish(&mut self, output: &mut dyn Sender<RecordBatch>) -> dispatch::UnaryResult<bool> {
        // No range arrives after this, so the decoders kept for a next range
        // are done with.
        self.idle.clear();
        if self.active.is_some() {
            return Ok(false);
        }
        self.flush_live_rows(output)?;
        Ok(true)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::reading::range_cutter::RangeCutter;
    use crate::reading::record_batch_metadata::row_index;
    use crate::test_utils::deleted_rows;
    use crate::thrift::general::Encoding;
    use crate::thrift::headers::PageHeader;
    use crate::types::filter_mask::FilterMask;
    use crate::types::metadata::{
        ColumnChunkMeta, QueryRowGroupMetadata, RowGroupMetadata, RowSelection,
    };
    use crate::types::page::{DataPage, DecompressedPage, DecompressedPageType};
    use crate::types::projection::Projection;
    use crate::types::table::ParquetTable;
    use arrow_array::{Array, Int32Array};
    use arrow_schema::{DataType, Field, Schema, SchemaRef};
    use bytes::Bytes;
    use dispatch::memory::init_test_free_pool;
    use dispatch::test_utils::run_unary_to_completion;
    use std::sync::atomic::AtomicUsize;

    pub(crate) fn make_test_table(schema: SchemaRef, num_rows: i64) -> Arc<ParquetTable> {
        let num_cols = schema.fields().len();
        let file = dispatch::io::LocalFile::new(std::fs::File::open("/dev/null").unwrap()).unwrap();
        Arc::new(ParquetTable::new(vec![Arc::new(RowGroupMetadata {
            open_file: dispatch::io::OpenFile::Local(file),
            schema,
            columns: (0..num_cols)
                .map(|_| ColumnChunkMeta {
                    codec: crate::thrift::general::CompressionCodec::SNAPPY,
                    dictionary_page_offset: None,
                    data_page_offset: 0,
                    total_compressed_size: 0,
                    total_uncompressed_size: 0,
                    max_def_level: 0,
                    physical_type: 0,
                    fixed_len_byte_width: None,
                    data_pages_all_dictionary: false,
                    absent: false,
                })
                .collect(),
            statistics: Arc::default(),
            num_rows,
            file_row_offset: 0,
            deleted_rows: None,
            file_row_group_idx: 0,
            live_decompressed_pages: Arc::new(AtomicUsize::new(0)),
        })]))
    }

    pub(crate) fn encode_i32s(values: &[i32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    pub(crate) fn make_data_page(
        metadata: QueryRowGroupMetadata,
        column_idx: usize,
        data: Vec<u8>,
        num_values: usize,
        page_idx: usize,
        first_row: u32,
    ) -> DecompressedPage {
        let header = PageHeader::for_data_page(num_values as i32, Encoding::PLAIN);
        DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: metadata,
            column_idx,
            idx: page_idx,
            first_row,
            data: DecompressedPageType::Data(DataPage {
                header: header.data_page_header.unwrap(),
                data: vec![Bytes::from(data)],
                filter_mask: None,
            }),
        }
    }

    pub(crate) fn make_skipped_page(
        metadata: QueryRowGroupMetadata,
        column_idx: usize,
        num_values: usize,
        page_idx: usize,
        first_row: u32,
    ) -> DecompressedPage {
        let header = PageHeader::for_data_page(num_values as i32, Encoding::PLAIN);
        DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: metadata,
            column_idx,
            idx: page_idx,
            first_row,
            data: DecompressedPageType::SkippedData {
                header: header.data_page_header.unwrap(),
            },
        }
    }

    pub(crate) fn extract_i32s(batch: &RecordBatch, col: usize) -> Vec<i32> {
        let arr = batch.column(col);
        let a = arr.as_any().downcast_ref::<Int32Array>().unwrap();
        (0..a.len()).map(|i| a.value(i)).collect()
    }

    pub(crate) fn i32_schema(names: &[&str]) -> SchemaRef {
        Arc::new(Schema::new(
            names
                .iter()
                .map(|n| Field::new(*n, DataType::Int32, false))
                .collect::<Vec<_>>(),
        ))
    }

    /// A table over `(s: Utf8View, v: Int32)` with `s` dictionary encoded
    /// and marked fully so (or not), the shape the view-equality tests need.
    pub(crate) fn string_and_i32_table(num_rows: i64, all_dictionary: bool) -> Arc<ParquetTable> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("s", DataType::Utf8View, false),
            Field::new("v", DataType::Int32, false),
        ]));
        let file = dispatch::io::LocalFile::new(std::fs::File::open("/dev/null").unwrap()).unwrap();
        let column = |dictionary: bool, all_dictionary: bool| ColumnChunkMeta {
            codec: crate::thrift::general::CompressionCodec::SNAPPY,
            dictionary_page_offset: dictionary.then_some(0),
            data_page_offset: 0,
            total_compressed_size: 0,
            total_uncompressed_size: 0,
            max_def_level: 0,
            physical_type: 0,
            fixed_len_byte_width: None,
            data_pages_all_dictionary: all_dictionary,
            absent: false,
        };
        Arc::new(ParquetTable::new(vec![Arc::new(RowGroupMetadata {
            open_file: dispatch::io::OpenFile::Local(file),
            schema,
            columns: vec![column(true, all_dictionary), column(false, false)],
            statistics: Arc::default(),
            num_rows,
            file_row_offset: 0,
            deleted_rows: None,
            file_row_group_idx: 0,
            live_decompressed_pages: Arc::new(AtomicUsize::new(0)),
        })]))
    }

    pub(crate) fn make_dict_page(
        metadata: QueryRowGroupMetadata,
        column_idx: usize,
        entries: &[&str],
    ) -> DecompressedPage {
        let mut data = Vec::new();
        for s in entries {
            data.extend_from_slice(&(s.len() as u32).to_le_bytes());
            data.extend_from_slice(s.as_bytes());
        }
        let header = PageHeader::for_dict_page(entries.len() as i32);
        DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: metadata,
            column_idx,
            idx: 0,
            first_row: 0,
            data: DecompressedPageType::Dict {
                header: header.dictionary_page_header.unwrap(),
                data: vec![Bytes::from(data)],
            },
        }
    }

    /// An RLE-dictionary data page holding `keys` (bit width 8: one RLE run
    /// per key keeps the encoding trivial).
    pub(crate) fn make_rle_data_page(
        metadata: QueryRowGroupMetadata,
        column_idx: usize,
        keys: &[u8],
        page_idx: usize,
    ) -> DecompressedPage {
        let mut data = vec![8u8];
        for &key in keys {
            data.push(1 << 1);
            data.push(key);
        }
        let header = PageHeader::for_data_page(keys.len() as i32, Encoding::RLE_DICTIONARY);
        DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: metadata,
            column_idx,
            idx: page_idx,
            first_row: 0,
            data: DecompressedPageType::Data(DataPage {
                header: header.data_page_header.unwrap(),
                data: vec![Bytes::from(data)],
                filter_mask: None,
            }),
        }
    }

    pub(crate) fn string_eq_predicate(column_idx: usize, value: &str) -> ScanEqualityPredicate {
        ScanEqualityPredicate {
            column_idx,
            path: Vec::new(),
            value: Scalar::new(
                Arc::new(arrow_array::StringViewArray::from(vec![value])) as ArrayRef
            ),
        }
    }

    pub(crate) fn extract_strings(batch: &RecordBatch, col: usize) -> Vec<String> {
        let a = batch
            .column(col)
            .as_any()
            .downcast_ref::<arrow_array::StringViewArray>()
            .unwrap();
        (0..a.len()).map(|i| a.value(i).to_string()).collect()
    }

    /// A cutter for `table` with no fetcher behind it: the claim counters
    /// are seeded high enough that releases never hit the underflow
    /// assertion.
    pub(crate) fn new_cutter(
        table: &Arc<ParquetTable>,
        eq_predicates: Vec<ScanEqualityPredicate>,
    ) -> RangeCutter {
        RangeCutter::new(
            Projection::all_from_schema(table.schema()),
            Arc::new(eq_predicates),
            Arc::new(AtomicUsize::new(usize::MAX / 2)),
            Arc::new(AtomicUsize::new(usize::MAX / 2)),
            Arc::new(WorkerAllocator::new()),
        )
    }

    pub(crate) fn new_decoder(batch_size: usize) -> Decoder {
        Decoder::new(batch_size, false, Arc::new(WorkerAllocator::new()))
    }

    /// Cuts `pages` into ranges.
    pub(crate) fn cut(table: &Arc<ParquetTable>, pages: Vec<DecompressedPage>) -> Vec<DecodeRange> {
        run_unary_to_completion(new_cutter(table, Vec::new()), pages)
    }

    /// Cuts `pages` into ranges and decodes them on one worker.
    fn decode(
        table: &Arc<ParquetTable>,
        pages: Vec<DecompressedPage>,
        batch_size: usize,
        eq_predicates: Vec<ScanEqualityPredicate>,
    ) -> Vec<RecordBatch> {
        let ranges = run_unary_to_completion(new_cutter(table, eq_predicates), pages);
        run_unary_to_completion(new_decoder(batch_size), ranges)
    }

    /// The rows of a row group as `i32`s equal to their row number, in
    /// pages of `page_rows`.
    pub(crate) fn row_number_pages(
        metadata: &QueryRowGroupMetadata,
        rows: u32,
        page_rows: u32,
    ) -> Vec<DecompressedPage> {
        (0..rows)
            .step_by(page_rows as usize)
            .enumerate()
            .map(|(idx, first_row)| {
                let values: Vec<i32> = (first_row..(first_row + page_rows).min(rows))
                    .map(|row| row as i32)
                    .collect();
                make_data_page(
                    metadata.clone(),
                    0,
                    encode_i32s(&values),
                    values.len(),
                    idx,
                    first_row,
                )
            })
            .collect()
    }

    /// A pushed string equality filters emitted batches down to matching rows
    /// via view equality against the dictionary's view of the constant.
    #[test]
    fn view_eq_filters_batches_to_matching_rows() {
        init_test_free_pool(4);
        let table = string_and_i32_table(5, true);
        let metadata = QueryRowGroupMetadata::new(&table, 0, RowSelection::All);
        let dict = make_dict_page(metadata.clone(), 0, &["MAIL", "DELIVER IN PERSON", "SHIP"]);
        let strings = make_rle_data_page(metadata.clone(), 0, &[0, 1, 2, 1, 0], 0);
        let values = make_data_page(metadata, 1, encode_i32s(&[1, 2, 3, 4, 5]), 5, 0, 0);

        let out = decode(
            &table,
            vec![dict, strings, values],
            1024,
            vec![string_eq_predicate(0, "DELIVER IN PERSON")],
        );

        let rows: Vec<i32> = out.iter().flat_map(|b| extract_i32s(b, 1)).collect();
        assert_eq!(rows, vec![2, 4]);
        let kept: Vec<String> = out.iter().flat_map(|b| extract_strings(b, 0)).collect();
        assert_eq!(kept, vec!["DELIVER IN PERSON", "DELIVER IN PERSON"]);
    }

    /// A duplicated dictionary entry makes the single-view test unfaithful,
    /// so batches pass through unfiltered (the upstream Filter still applies
    /// the predicate).
    #[test]
    fn duplicate_dictionary_entries_disable_view_eq() {
        init_test_free_pool(4);
        let table = string_and_i32_table(3, true);
        let metadata = QueryRowGroupMetadata::new(&table, 0, RowSelection::All);
        let dict = make_dict_page(metadata.clone(), 0, &["AIR", "AIR", "RAIL"]);
        let strings = make_rle_data_page(metadata.clone(), 0, &[0, 1, 2], 0);
        let values = make_data_page(metadata, 1, encode_i32s(&[1, 2, 3]), 3, 0, 0);

        let out = decode(
            &table,
            vec![dict, strings, values],
            1024,
            vec![string_eq_predicate(0, "AIR")],
        );

        let rows: Vec<i32> = out.iter().flat_map(|b| extract_i32s(b, 1)).collect();
        assert_eq!(rows, vec![1, 2, 3]);
    }

    /// A chunk with non-dictionary data pages must not view-filter: equal
    /// strings from a plain page carry different views.
    #[test]
    fn mixed_encoding_chunk_is_not_view_filtered() {
        init_test_free_pool(4);
        let table = string_and_i32_table(3, false);
        let metadata = QueryRowGroupMetadata::new(&table, 0, RowSelection::All);
        let dict = make_dict_page(metadata.clone(), 0, &["MAIL", "SHIP"]);
        let strings = make_rle_data_page(metadata.clone(), 0, &[0, 1, 0], 0);
        let values = make_data_page(metadata, 1, encode_i32s(&[1, 2, 3]), 3, 0, 0);

        let out = decode(
            &table,
            vec![dict, strings, values],
            1024,
            vec![string_eq_predicate(0, "MAIL")],
        );

        let rows: Vec<i32> = out.iter().flat_map(|b| extract_i32s(b, 1)).collect();
        assert_eq!(rows, vec![1, 2, 3]);
    }

    /// The decoder reads dictionary-encoded pages through the dictionary
    /// the cutter built from the dictionary page.
    #[test]
    fn the_decoder_reads_through_the_dictionary_the_cutter_built() {
        init_test_free_pool(4);
        let table = string_and_i32_table(3, true);
        let metadata = QueryRowGroupMetadata::new(&table, 0, RowSelection::All);
        let dict = make_dict_page(metadata.clone(), 0, &["MAIL", "SHIP"]);
        let strings = make_rle_data_page(metadata.clone(), 0, &[1, 0, 1], 0);
        let values = make_data_page(metadata, 1, encode_i32s(&[1, 2, 3]), 3, 0, 0);

        let out = decode(&table, vec![dict, strings, values], 1024, Vec::new());

        let kept: Vec<String> = out.iter().flat_map(|b| extract_strings(b, 0)).collect();
        assert_eq!(kept, vec!["SHIP", "MAIL", "SHIP"]);
    }

    #[test]
    fn a_single_page_decodes_to_one_batch() {
        init_test_free_pool(4);
        let table = make_test_table(i32_schema(&["a"]), 5);
        let metadata = QueryRowGroupMetadata::new(&table, 0, RowSelection::All);
        let page = make_data_page(metadata, 0, encode_i32s(&[10, 20, 30, 40, 50]), 5, 0, 0);

        let out = decode(&table, vec![page], 1024, Vec::new());

        assert_eq!(out.len(), 1);
        assert_eq!(extract_i32s(&out[0], 0), vec![10, 20, 30, 40, 50]);
    }

    #[test]
    fn a_batch_holds_every_column() {
        init_test_free_pool(4);
        let table = make_test_table(i32_schema(&["a", "b"]), 3);
        let metadata = QueryRowGroupMetadata::new(&table, 0, RowSelection::All);
        let page_a = make_data_page(metadata.clone(), 0, encode_i32s(&[10, 20, 30]), 3, 0, 0);
        let page_b = make_data_page(metadata, 1, encode_i32s(&[40, 50, 60]), 3, 0, 0);

        let out = decode(&table, vec![page_a, page_b], 1024, Vec::new());

        assert_eq!(out.len(), 1);
        assert_eq!(extract_i32s(&out[0], 0), vec![10, 20, 30]);
        assert_eq!(extract_i32s(&out[0], 1), vec![40, 50, 60]);
    }

    #[test]
    fn the_batch_size_splits_the_output() {
        init_test_free_pool(4);
        let table = make_test_table(i32_schema(&["a"]), 5);
        let metadata = QueryRowGroupMetadata::new(&table, 0, RowSelection::All);
        let page = make_data_page(metadata, 0, encode_i32s(&[10, 20, 30, 40, 50]), 5, 0, 0);

        let out = decode(&table, vec![page], 2, Vec::new());

        assert_eq!(out.len(), 3);
        assert_eq!(extract_i32s(&out[0], 0), vec![10, 20]);
        assert_eq!(extract_i32s(&out[1], 0), vec![30, 40]);
        assert_eq!(extract_i32s(&out[2], 0), vec![50]);
    }

    #[test]
    fn the_pages_of_a_column_decode_in_order() {
        init_test_free_pool(4);
        let table = make_test_table(i32_schema(&["a"]), 5);
        let metadata = QueryRowGroupMetadata::new(&table, 0, RowSelection::All);
        let page1 = make_data_page(metadata.clone(), 0, encode_i32s(&[40, 50]), 2, 1, 3);
        let page0 = make_data_page(metadata, 0, encode_i32s(&[10, 20, 30]), 3, 0, 0);

        let out = decode(&table, vec![page1, page0], 1024, Vec::new());

        let rows: Vec<i32> = out.iter().flat_map(|b| extract_i32s(b, 0)).collect();
        assert_eq!(rows, vec![10, 20, 30, 40, 50]);
    }

    /// `table` with the `deleted` rows of its one row group marked deleted.
    fn with_deleted_rows(mut table: Arc<ParquetTable>, deleted: &[u32]) -> Arc<ParquetTable> {
        let row_groups = Arc::get_mut(&mut table).unwrap().row_groups_mut();
        let row_group = Arc::get_mut(&mut row_groups[0]).unwrap();
        row_group.deleted_rows = deleted_rows(row_group.file_row_offset, deleted);
        table
    }

    #[test]
    fn deleted_rows_are_dropped_from_every_batch() {
        init_test_free_pool(4);
        let table = with_deleted_rows(make_test_table(i32_schema(&["a"]), 5), &[1, 3]);
        let metadata = QueryRowGroupMetadata::new(&table, 0, RowSelection::All);
        let page = make_data_page(metadata, 0, encode_i32s(&[10, 20, 30, 40, 50]), 5, 0, 0);

        let out = decode(&table, vec![page], 2, Vec::new());

        let rows: Vec<i32> = out.iter().flat_map(|b| extract_i32s(b, 0)).collect();
        assert_eq!(rows, vec![10, 30, 50]);
    }

    #[test]
    fn a_batch_fills_up_with_the_rows_kept_past_the_deleted_ones() {
        init_test_free_pool(4);
        let table = with_deleted_rows(make_test_table(i32_schema(&["a"]), 5), &[1, 3]);
        let metadata = QueryRowGroupMetadata::new(&table, 0, RowSelection::All);
        let page = make_data_page(metadata, 0, encode_i32s(&[10, 20, 30, 40, 50]), 5, 0, 0);

        let out = decode(&table, vec![page], 3, Vec::new());

        assert_eq!(out.len(), 1);
        assert_eq!(extract_i32s(&out[0], 0), vec![10, 30, 50]);
    }

    #[test]
    fn the_rows_kept_keep_their_row_index() {
        init_test_free_pool(4);
        let table = with_deleted_rows(make_test_table(i32_schema(&["a"]), 5), &[1, 3]);
        let metadata = QueryRowGroupMetadata::new(&table, 0, RowSelection::All);
        let page = make_data_page(metadata, 0, encode_i32s(&[10, 20, 30, 40, 50]), 5, 0, 0);

        let ranges = run_unary_to_completion(new_cutter(&table, Vec::new()), vec![page]);
        let out = run_unary_to_completion(
            Decoder::new(2, true, Arc::new(WorkerAllocator::new())),
            ranges,
        );

        let positions: Vec<u32> = out
            .iter()
            .flat_map(|b| row_index(b).values().to_vec())
            .collect();
        assert_eq!(positions, vec![0, 2, 4]);
        let rows: Vec<i32> = out.iter().flat_map(|b| extract_i32s(b, 0)).collect();
        assert_eq!(rows, vec![10, 30, 50]);
    }

    #[test]
    fn a_batch_deleted_whole_still_carries_its_row_index() {
        init_test_free_pool(4);
        let table = with_deleted_rows(make_test_table(i32_schema(&["a"]), 5), &[0, 1]);
        let metadata = QueryRowGroupMetadata::new(&table, 0, RowSelection::All);
        let page = make_data_page(metadata, 0, encode_i32s(&[10, 20, 30, 40, 50]), 5, 0, 0);

        let ranges = run_unary_to_completion(new_cutter(&table, Vec::new()), vec![page]);
        let out = run_unary_to_completion(
            Decoder::new(2, true, Arc::new(WorkerAllocator::new())),
            ranges,
        );

        let positions: Vec<u32> = out
            .iter()
            .flat_map(|b| row_index(b).values().to_vec())
            .collect();
        assert_eq!(positions, vec![2, 3, 4]);
        let rows: Vec<i32> = out.iter().flat_map(|b| extract_i32s(b, 0)).collect();
        assert_eq!(rows, vec![30, 40, 50]);
    }

    #[test]
    fn an_index_selection_emits_only_its_rows() {
        init_test_free_pool(4);
        let table = make_test_table(i32_schema(&["a"]), 5);
        let metadata =
            QueryRowGroupMetadata::new(&table, 0, RowSelection::Indices(vec![3, 4].into()));
        let skipped = make_skipped_page(metadata.clone(), 0, 3, 0, 0);
        let mut kept = make_data_page(metadata.clone(), 0, encode_i32s(&[40, 50]), 2, 1, 3);
        if let DecompressedPageType::Data(data) = &mut kept.data {
            data.filter_mask = Some(FilterMask::new(3, 5, &[3, 4]));
        }

        let out = decode(&table, vec![skipped, kept], 1024, Vec::new());

        let rows: Vec<i32> = out.iter().flat_map(|b| extract_i32s(b, 0)).collect();
        assert_eq!(rows, vec![40, 50]);
    }

    #[test]
    fn the_output_schema_follows_the_projection() {
        init_test_free_pool(4);
        let table = make_test_table(i32_schema(&["x", "y"]), 2);
        let metadata = QueryRowGroupMetadata::new(&table, 0, RowSelection::All);
        let page_x = make_data_page(metadata.clone(), 0, encode_i32s(&[10, 20]), 2, 0, 0);
        let page_y = make_data_page(metadata, 1, encode_i32s(&[30, 40]), 2, 0, 0);

        let out = decode(&table, vec![page_x, page_y], 1024, Vec::new());

        assert_eq!(out[0].schema().fields().len(), 2);
        assert_eq!(out[0].schema().field(0).name(), "x");
        assert_eq!(out[0].schema().field(1).name(), "y");
        assert_eq!(*out[0].schema().field(0).data_type(), DataType::Int32);
    }

    /// The ranges of one row group decoded on two workers: the second worker
    /// takes the middle range, which starts inside a page, and the first
    /// worker goes on past the hole it left.
    #[test]
    fn a_range_taken_by_another_worker_decodes_its_rows_from_inside_a_page() {
        init_test_free_pool(4);
        let table = make_test_table(i32_schema(&["a"]), 40_000);
        let metadata = QueryRowGroupMetadata::new(&table, 0, RowSelection::All);
        let mut ranges = cut(&table, row_number_pages(&metadata, 40_000, 20_000));
        let middle = ranges.remove(1);

        let first_worker = run_unary_to_completion(new_decoder(8192), ranges);
        let second_worker = run_unary_to_completion(new_decoder(8192), vec![middle]);

        let first: Vec<i32> = first_worker
            .iter()
            .flat_map(|b| extract_i32s(b, 0))
            .collect();
        let second: Vec<i32> = second_worker
            .iter()
            .flat_map(|b| extract_i32s(b, 0))
            .collect();
        let expected_first: Vec<i32> = (0..16_384).chain(32_768..40_000).collect();
        assert_eq!(first, expected_first);
        assert_eq!(second, (16_384..32_768).collect::<Vec<i32>>());
    }
}
