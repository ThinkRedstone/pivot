//! Cuts each row group's rows into [`DecodeRange`]s as its pages arrive.
//!
//! One cutter runs on every worker, fed the decompressed pages of the row
//! groups that worker claimed. It collects them per row group in a
//! [`RowGroupPages`], and as soon as every column has the pages covering
//! the next range, emits that range as a job carrying those pages for the
//! decoders. The jobs go out in row order, so the worker's own decoder
//! follows the row group as one stream while idle peers take the ranges it
//! has not reached.
//!
//! The cutter builds each column's dictionary as its page arrives, once for
//! the row group; every range carries it for its decoder to read through.
//! A dictionary that excludes a pushed-down equality constant prunes the
//! row group there and then: none of its data pages is decoded, and the
//! decompressor drops the ones still to come.

pub(crate) mod row_group_pages;
pub use row_group_pages::DecodeRange;

use crate::reading::decoding::leaf_decoders::{BuiltDictionary, LeafDecoder};
use crate::reading::decoding::{DecodePlan, ScanEqualityPredicate, WorkerAllocator};
use crate::types::page::{DecompressedPage, DecompressedPageType};
use crate::types::projection::Projection;
use ahash::{HashMap, HashSet};
use dispatch::{Sender, Unary, UnaryFactory};
use row_group_pages::{PageContent, RowGroupPages};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Factory for creating [`RangeCutter`] instances, one per worker thread.
pub struct RangeCutterFactory {
    /// Which columns are decoded.
    pub projection: Projection,
    /// Pushed-down equality predicates for dictionary pruning (may be empty).
    pub eq_predicates: Arc<Vec<ScanEqualityPredicate>>,
    /// This worker's claimed-but-not-yet-cut row-group count, shared with
    /// its fetcher; decremented as row groups are cut or pruned so the
    /// fetcher's claim backpressure releases (see `RowGroupFetcher`).
    pub pending_row_groups: Arc<AtomicUsize>,
    /// The scan's claimed-but-not-yet-decoded row-group count, shared with
    /// the row-group injector, which throttles the speculative claims of an
    /// unarmed Top-N scan against it. A row group counts until its last
    /// range is decoded, on whichever worker; one that is pruned or empty
    /// is released here.
    pub outstanding_row_groups: Arc<AtomicUsize>,
    /// The worker's allocator for the dictionaries it builds.
    pub allocator: Arc<WorkerAllocator>,
}

impl UnaryFactory<DecompressedPage, DecodeRange> for RangeCutterFactory {
    type Unary = RangeCutter;

    fn build_unary(self) -> Self::Unary {
        RangeCutter::new(
            self.projection,
            self.eq_predicates,
            self.pending_row_groups,
            self.outstanding_row_groups,
            self.allocator,
        )
    }
}

/// A row group whose pages are still arriving.
struct OpenRowGroup {
    pages: RowGroupPages,
    /// How many ranges, from the first, have been emitted.
    emitted: usize,
    /// The cutter's own leaf decoders, which build the dictionaries and
    /// decide the pruning.
    leaf_decoders: Vec<Box<dyn LeafDecoder>>,
}

/// Collects the pages of this worker's row groups and emits their
/// [`DecodeRange`]s.
pub struct RangeCutter {
    projection: Projection,
    eq_predicates: Arc<Vec<ScanEqualityPredicate>>,
    /// Slab allocator for the dictionaries built here, the worker's own.
    allocator: Arc<WorkerAllocator>,
    open: HashMap<usize, OpenRowGroup>,
    /// Row groups whose ranges are all emitted or that were pruned; a page
    /// arriving late for one is dropped.
    closed: HashSet<usize>,
    pending_row_groups: Arc<AtomicUsize>,
    outstanding_row_groups: Arc<AtomicUsize>,
}

impl RangeCutter {
    pub fn new(
        projection: Projection,
        eq_predicates: Arc<Vec<ScanEqualityPredicate>>,
        pending_row_groups: Arc<AtomicUsize>,
        outstanding_row_groups: Arc<AtomicUsize>,
        allocator: Arc<WorkerAllocator>,
    ) -> Self {
        Self {
            projection,
            eq_predicates,
            allocator,
            open: HashMap::default(),
            closed: HashSet::default(),
            pending_row_groups,
            outstanding_row_groups,
        }
    }

    /// Mark one claimed row group fully cut (or pruned), releasing its
    /// share of the fetcher's claim backpressure.
    fn release_claim(&self) {
        let previous = self.pending_row_groups.fetch_sub(1, Ordering::Relaxed);
        // An underflow means this cutter released a claim its own fetcher
        // never made, i.e. the row group's pages landed on a different worker
        // than its claimer. Besides breaking the accounting (the wrapped
        // counter permanently stops the worker's claims), it would mean the
        // claim/cut pairing contract broke.
        debug_assert!(previous > 0, "released a row-group claim never made");
    }

    /// Forgets `row_group_index`: its pages from here on are dropped. A row
    /// group closed with no range to decode is released from the scan's
    /// outstanding count here, as no decoder will.
    fn close(&mut self, row_group_index: usize) {
        let open = self
            .open
            .remove(&row_group_index)
            .expect("a row group is closed once");
        self.closed.insert(row_group_index);
        self.release_claim();
        if open.pages.range_count() == 0 || open.pages.metadata().is_pruned() {
            self.outstanding_row_groups.fetch_sub(1, Ordering::Relaxed);
        }
    }

    /// Emits the ranges that became ready, closing the row group after its
    /// last one.
    fn emit_ready_ranges(
        &mut self,
        row_group_index: usize,
        sender: &mut dyn Sender<DecodeRange>,
    ) -> dispatch::UnaryResult<()> {
        let open = self
            .open
            .get_mut(&row_group_index)
            .expect("ranges are emitted for an open row group");
        let ready = open.pages.ready_ranges();
        for index in open.emitted..ready {
            sender.send(open.pages.cut_range(index))?;
        }
        open.emitted = ready;
        if open.emitted == open.pages.range_count() {
            self.close(row_group_index);
        }
        Ok(())
    }
}

impl Unary<DecompressedPage, DecodeRange> for RangeCutter {
    fn consume(
        &mut self,
        page: DecompressedPage,
        output: &mut dyn Sender<DecodeRange>,
        _io: &mut dispatch::OperatorIO,
    ) -> dispatch::UnaryResult<()> {
        let metadata = page.query_row_group_metadata.clone();
        let row_group_index = metadata.row_group_index;
        if self.closed.contains(&row_group_index) {
            return Ok(());
        }
        if !self.open.contains_key(&row_group_index) {
            let plan = Arc::new(
                DecodePlan::new(&metadata, &self.projection, &self.eq_predicates)
                    .map_err(crate::op_err)?,
            );
            let leaf_decoders = plan
                .create_leaf_decoders(&metadata)
                .map_err(crate::op_err)?;
            let pages =
                RowGroupPages::new(metadata.clone(), plan, self.outstanding_row_groups.clone());
            self.open.insert(
                row_group_index,
                OpenRowGroup {
                    pages,
                    emitted: 0,
                    leaf_decoders,
                },
            );
        }
        let open = self.open.get_mut(&row_group_index).unwrap();
        let column = page.column_idx;
        match page.data {
            DecompressedPageType::Dict { header, data } => {
                // Built once here, for every decoder of the row group to
                // read through; a dictionary excluding the pushed-down
                // constant prunes the row group before any range is cut.
                let leaf = &open.leaf_decoders[column];
                let built = self
                    .allocator
                    .with(|allocator| leaf.build_dictionary(header, data, allocator));
                match built {
                    BuiltDictionary::Pruned => {
                        // Seen by every page of this row group still in
                        // flight: the decompressor drops its remaining data
                        // pages.
                        metadata.mark_pruned();
                        self.close(row_group_index);
                        return Ok(());
                    }
                    BuiltDictionary::Built(dictionary) => {
                        open.pages.insert_dictionary(column, dictionary)
                    }
                }
            }
            DecompressedPageType::Data(data) => {
                let rows = data.header.num_values as u32;
                open.pages.insert_page(
                    column,
                    page.idx,
                    page.first_row,
                    rows,
                    PageContent::Data(data),
                );
            }
            DecompressedPageType::SkippedData { header } => {
                let rows = header.num_values as u32;
                open.pages.insert_page(
                    column,
                    page.idx,
                    page.first_row,
                    rows,
                    PageContent::Skipped(header),
                );
            }
        }
        self.emit_ready_ranges(row_group_index, output)
    }

    fn finish(&mut self, _output: &mut dyn Sender<DecodeRange>) -> dispatch::UnaryResult<bool> {
        Ok(self.open.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reading::decoding::tests::{
        cut, encode_i32s, i32_schema, make_data_page, make_rle_data_page, make_test_table,
        new_cutter, row_number_pages,
    };
    use crate::thrift::headers::PageHeader;
    use crate::types::metadata::{
        ColumnChunkMeta, QueryRowGroupMetadata, RowGroupMetadata, RowSelection,
    };
    use crate::types::table::ParquetTable;
    use arrow_array::{ArrayRef, Int32Array, Scalar};
    use bytes::Bytes;
    use dispatch::memory::init_test_free_pool;
    use dispatch::test_utils::{feed_unary, run_unary_to_completion};

    /// A table over one dictionary encoded `Int32` column.
    fn i32_dictionary_table(num_rows: i64) -> Arc<ParquetTable> {
        let file = dispatch::io::LocalFile::new(std::fs::File::open("/dev/null").unwrap()).unwrap();
        Arc::new(ParquetTable::new(vec![Arc::new(RowGroupMetadata {
            open_file: dispatch::io::OpenFile::Local(file),
            schema: i32_schema(&["a"]),
            columns: vec![ColumnChunkMeta {
                codec: crate::thrift::general::CompressionCodec::SNAPPY,
                dictionary_page_offset: Some(0),
                data_page_offset: 0,
                total_compressed_size: 0,
                total_uncompressed_size: 0,
                max_def_level: 0,
                physical_type: 0,
                fixed_len_byte_width: None,
                data_pages_all_dictionary: true,
                absent: false,
            }],
            statistics: Arc::default(),
            num_rows,
            file_row_offset: 0,
            deleted_rows: None,
            file_row_group_idx: 0,
            live_decompressed_pages: Arc::new(AtomicUsize::new(0)),
        })]))
    }

    fn i32_dict_page(metadata: QueryRowGroupMetadata, entries: &[i32]) -> DecompressedPage {
        let header = PageHeader::for_dict_page(entries.len() as i32);
        DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: metadata,
            column_idx: 0,
            idx: 0,
            first_row: 0,
            data: DecompressedPageType::Dict {
                header: header.dictionary_page_header.unwrap(),
                data: vec![Bytes::from(encode_i32s(entries))],
            },
        }
    }

    fn i32_eq_predicate(value: i32) -> ScanEqualityPredicate {
        ScanEqualityPredicate {
            column_idx: 0,
            path: Vec::new(),
            value: Scalar::new(Arc::new(Int32Array::from(vec![value])) as ArrayRef),
        }
    }

    fn range_starts(ranges: &[DecodeRange]) -> Vec<u32> {
        ranges.iter().map(|range| range.rows().start).collect()
    }

    #[test]
    fn a_row_group_is_cut_into_ranges_in_row_order() {
        init_test_free_pool(4);
        let table = make_test_table(i32_schema(&["a"]), 40_000);
        let metadata = QueryRowGroupMetadata::new(&table, 0, RowSelection::All);

        let ranges = cut(&table, row_number_pages(&metadata, 40_000, 20_000));

        assert_eq!(range_starts(&ranges), vec![0, 16_384, 32_768]);
        assert_eq!(ranges[2].rows(), 32_768..40_000);
    }

    /// The second page arrives first: no range is ready until the first
    /// page fills the gap, then every range is.
    #[test]
    fn a_range_waits_for_the_pages_before_it() {
        init_test_free_pool(4);
        let table = make_test_table(i32_schema(&["a"]), 40_000);
        let metadata = QueryRowGroupMetadata::new(&table, 0, RowSelection::All);
        let mut pages = row_number_pages(&metadata, 40_000, 20_000);
        let first_page = pages.remove(0);
        let mut cutter = new_cutter(&table, Vec::new());

        let before = feed_unary(&mut cutter, pages);
        let ranges = feed_unary(&mut cutter, vec![first_page]);

        assert!(before.is_empty());
        assert_eq!(range_starts(&ranges), vec![0, 16_384, 32_768]);
    }

    #[test]
    fn a_range_waits_for_every_column() {
        init_test_free_pool(4);
        let table = make_test_table(i32_schema(&["a", "b"]), 3);
        let metadata = QueryRowGroupMetadata::new(&table, 0, RowSelection::All);
        let page_a = make_data_page(metadata.clone(), 0, encode_i32s(&[10, 20, 30]), 3, 0, 0);
        let page_b = make_data_page(metadata, 1, encode_i32s(&[40, 50, 60]), 3, 0, 0);
        let mut cutter = new_cutter(&table, Vec::new());

        let after_one = feed_unary(&mut cutter, vec![page_a]);
        let after_both = feed_unary(&mut cutter, vec![page_b]);

        assert!(after_one.is_empty());
        assert_eq!(after_both.len(), 1);
    }

    #[test]
    fn a_page_arriving_after_the_last_range_is_dropped() {
        init_test_free_pool(4);
        let table = make_test_table(i32_schema(&["a"]), 3);
        let metadata = QueryRowGroupMetadata::new(&table, 0, RowSelection::All);
        let page = make_data_page(metadata.clone(), 0, encode_i32s(&[10, 20, 30]), 3, 0, 0);
        let late_page = make_data_page(metadata, 0, encode_i32s(&[99]), 1, 1, 3);

        let ranges = cut(&table, vec![page, late_page]);

        assert_eq!(ranges.len(), 1);
    }

    /// The fetcher's claim is released once the row group is cut; the scan's
    /// outstanding count once its last range is decoded.
    #[test]
    fn the_claim_is_released_after_the_last_range() {
        init_test_free_pool(4);
        let table = make_test_table(i32_schema(&["a"]), 3);
        let metadata = QueryRowGroupMetadata::new(&table, 0, RowSelection::All);
        let page = make_data_page(metadata, 0, encode_i32s(&[10, 20, 30]), 3, 0, 0);
        let pending = Arc::new(AtomicUsize::new(1));
        let outstanding = Arc::new(AtomicUsize::new(1));
        let cutter = RangeCutter::new(
            Projection::all_from_schema(table.schema()),
            Arc::new(Vec::new()),
            pending.clone(),
            outstanding.clone(),
            Arc::new(WorkerAllocator::new()),
        );

        let mut ranges = run_unary_to_completion(cutter, vec![page]);

        assert_eq!(pending.load(Ordering::Relaxed), 0);
        let before_decode = outstanding.load(Ordering::Relaxed);
        ranges.remove(0).decoded();
        assert_eq!((before_decode, outstanding.load(Ordering::Relaxed)), (1, 0));
    }

    /// A dictionary that excludes the pushed-down constant prunes the row
    /// group as soon as it arrives: no range is emitted and the claim is
    /// released.
    #[test]
    fn a_dictionary_excluding_the_constant_prunes_the_row_group() {
        init_test_free_pool(4);
        let table = i32_dictionary_table(3);
        let metadata = QueryRowGroupMetadata::new(&table, 0, RowSelection::All);
        let dict = i32_dict_page(metadata.clone(), &[10, 20]);
        let pending = Arc::new(AtomicUsize::new(1));
        let cutter = RangeCutter::new(
            Projection::all_from_schema(table.schema()),
            Arc::new(vec![i32_eq_predicate(5)]),
            pending.clone(),
            Arc::new(AtomicUsize::new(1)),
            Arc::new(WorkerAllocator::new()),
        );

        let ranges = run_unary_to_completion(cutter, vec![dict]);

        assert!(ranges.is_empty());
        assert!(metadata.is_pruned());
        assert_eq!(pending.load(Ordering::Relaxed), 0);
    }

    /// The footer records no dictionary page, yet the chunk has one and a
    /// data page indexing it overtakes it: the range waits for the
    /// dictionary and carries it.
    #[test]
    fn a_range_waits_for_a_dictionary_the_footer_does_not_record() {
        init_test_free_pool(4);
        let table = make_test_table(i32_schema(&["a"]), 3);
        let metadata = QueryRowGroupMetadata::new(&table, 0, RowSelection::All);
        let page = make_rle_data_page(metadata.clone(), 0, &[1, 0, 1], 0);
        let dict = i32_dict_page(metadata, &[10, 20]);
        let mut cutter = new_cutter(&table, Vec::new());

        let before = feed_unary(&mut cutter, vec![page]);
        let ranges = feed_unary(&mut cutter, vec![dict]);

        assert!(before.is_empty());
        assert_eq!(ranges.len(), 1);
        assert!(ranges[0].column(0).dictionary.is_some());
    }
}
