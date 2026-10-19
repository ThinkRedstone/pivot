//! Per-row-group decoder that decodes the ranges of a row group into Arrow
//! [`RecordBatch`]es.
//!
//! The parent [`Decoder`](super::Decoder) keeps one [`RowGroupDecoder`] per
//! row group it has decoded rows of. A [`DecodeRange`] that starts where the
//! decoder stopped goes on from the page it has open; any other range
//! repositions it, which costs a skip into the range's first page of every
//! column.

use crate::reading::decoding::ScanEqualityPredicate;
use crate::reading::decoding::column_decoder::{ColumnDecoder, Error, Result, create_leaf_decoder};
use crate::reading::decoding::leaf_decoders::LeafDecoder;
use crate::reading::range_cutter::row_group_pages::{DecodeRange, PageContent, StoredPage};
use crate::reading::record_batch_metadata::with_row_index;
use crate::types::filter_mask::FilterMask;
use crate::types::leaves::{leaf_fields, plan_leaves, resolve_output_reads};
use crate::types::metadata::{ColumnChunkMeta, QueryRowGroupMetadata, RowSelection};
use crate::types::page::{DataPage, DecompressedPage, DecompressedPageType};
use crate::types::projection::Projection;
use arrow_array::{ArrayRef, RecordBatch, Scalar};
use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder};
use arrow_schema::{FieldRef, Fields, Schema, SchemaRef};
use dispatch::memory::SlabAllocator;
use std::cmp::min;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// An output column whose pushed-down equality constant can prune the row
/// group, and the decoded leaf that constant was installed on.
#[derive(Clone, Copy)]
pub(crate) struct PrunableColumn {
    /// The column's position in the emitted batch.
    pub output_idx: usize,
    /// The leaf whose dictionary decides the pruning.
    pub leaf: usize,
}

/// How a row group's projected columns decode: which leaves are read, one
/// view per output column, and the leaves whose dictionaries can prune the
/// row group. Built once per row group and shared by every decoder of its
/// rows; each decoder creates its own leaf decoders from it.
pub(crate) struct DecodePlan {
    /// The fields of the leaves read, in the order the fetcher requested
    /// them, so a page's `column_idx` indexes straight into this.
    leaf_fields: Vec<FieldRef>,
    /// The file leaves the decoders read, in the same order.
    pub file_leaves: Vec<usize>,
    /// One view per output column, in projection order. Several can fold the
    /// same decoded leaf.
    column_decoders: Vec<ColumnDecoder>,
    /// Output schema (projected). A pushed-down extract column carries the
    /// type the extract emits, not the variant it was read from.
    schema: SchemaRef,
    pub prunable: Vec<PrunableColumn>,
    /// The pushed-down equality constant each prunable leaf's decoder gets.
    eq_constants: Vec<(usize, Scalar<ArrayRef>)>,
}

impl DecodePlan {
    pub fn new(
        row_group_metadata: &QueryRowGroupMetadata,
        projection: &Projection,
        eq_predicates: &[ScanEqualityPredicate],
    ) -> Result<Self> {
        // Expand projected columns into this file's leaves. Variant layouts can
        // differ between files, so each row group resolves them independently,
        // and the fetcher resolved them the same way.
        let fields = row_group_metadata.get_metadata().schema.fields();
        let leaves = leaf_fields(fields);
        let reads = resolve_output_reads(fields, row_group_metadata, projection);
        let plan = plan_leaves(&reads);

        let mut column_decoders = Vec::with_capacity(projection.column_indices.len());
        for ((output_idx, &column), positions) in projection
            .column_indices
            .iter()
            .enumerate()
            .zip(plan.output_positions)
        {
            column_decoders.push(ColumnDecoder::new(
                column,
                projection.extract_at(output_idx),
                &reads[output_idx],
                positions,
                &leaves,
                fields,
            )?);
        }

        let (prunable, eq_constants) = prunable_columns(
            &column_decoders,
            &plan.file_leaves,
            row_group_metadata.columns(),
            projection,
            eq_predicates,
        );

        let output_fields: Fields = column_decoders
            .iter()
            .map(|decoder| decoder.output_field().clone())
            .collect();
        Ok(Self {
            leaf_fields: plan
                .file_leaves
                .iter()
                .map(|&leaf| leaves[leaf].clone())
                .collect(),
            file_leaves: plan.file_leaves,
            column_decoders,
            schema: Arc::new(Schema::new(output_fields)),
            prunable,
            eq_constants,
        })
    }

    /// How many leaves are decoded.
    pub fn leaf_count(&self) -> usize {
        self.file_leaves.len()
    }

    /// Fresh leaf decoders, one per decoded leaf, with the pushed-down
    /// constants installed on the prunable ones.
    pub fn create_leaf_decoders(
        &self,
        row_group_metadata: &QueryRowGroupMetadata,
    ) -> Result<Vec<Box<dyn LeafDecoder>>> {
        let column_chunks = row_group_metadata.columns();
        let mut leaf_decoders = self
            .leaf_fields
            .iter()
            .zip(&self.file_leaves)
            .map(|(field, &leaf)| create_leaf_decoder(field.data_type(), &column_chunks[leaf]))
            .collect::<Result<Vec<_>>>()?;
        for (leaf, constant) in &self.eq_constants {
            leaf_decoders[*leaf].set_eq_constant(constant);
        }
        Ok(leaf_decoders)
    }
}

/// A batch of a row group's rows, and, when the table format deleted any of
/// them, a bit per row of the batch set for the rows it left live.
pub struct DecodedBatch {
    pub batch: RecordBatch,
    pub live_rows: Option<BooleanBuffer>,
}

/// Decodes the ranges of a single row group into [`RecordBatch`]es.
pub struct RowGroupDecoder {
    metadata: QueryRowGroupMetadata,
    plan: Arc<DecodePlan>,
    leaf_decoders: Vec<Box<dyn LeafDecoder>>,
    /// Max rows per batch.
    batch_size: usize,
    /// The next row to emit and the row the attached range ends at, in
    /// decoded-row terms (see [`DecodeRange::rows`]).
    next_row: usize,
    end_row: usize,
    /// The range being decoded.
    range: Option<DecodeRange>,
    /// One bit per row of the attached range, set for the rows the table
    /// format left live, when it deleted any of them.
    live_in_range: Option<BooleanBuffer>,
    /// The row group's ranges still to decode, on any worker.
    ranges_left: Arc<AtomicUsize>,
    /// Per leaf, the index of the next page to hand the leaf decoder.
    next_page: Vec<usize>,
    /// Whether to append row-group-id / row-index metadata columns.
    add_row_group_metadata: bool,
    /// Whether emitted batches may drop rows that fail a pushed-down
    /// equality constant. The materializer path must not: it addresses rows
    /// by their position inside the row group, so every row has to stay in
    /// place.
    filter_batches: bool,
}

impl RowGroupDecoder {
    /// A decoder for the row group of `range`, ready to take it on, reading
    /// through the dictionaries the range carries. Every range of a row
    /// group carries the same dictionaries, built once by the cutter.
    pub fn new(
        range: &DecodeRange,
        batch_size: usize,
        add_row_group_metadata: bool,
    ) -> Result<Self> {
        let metadata = range.metadata().clone();
        let plan = range.plan().clone();
        let filter_batches =
            !add_row_group_metadata && !matches!(metadata.selection(), RowSelection::Indices(_));
        let mut leaf_decoders = plan.create_leaf_decoders(&metadata)?;
        for (leaf, decoder) in leaf_decoders.iter_mut().enumerate() {
            if let Some(dictionary) = &range.column(leaf).dictionary {
                decoder.adopt_dictionary(dictionary.clone());
            }
        }
        Ok(Self {
            next_page: vec![0; plan.leaf_count()],
            leaf_decoders,
            plan,
            batch_size,
            next_row: 0,
            end_row: 0,
            range: None,
            live_in_range: None,
            ranges_left: range.ranges_left().clone(),
            add_row_group_metadata,
            filter_batches,
            metadata,
        })
    }

    pub fn row_group_index(&self) -> usize {
        self.metadata.row_group_index
    }

    /// Whether every range of the row group is decoded, on any worker, so
    /// no further range can arrive.
    pub fn row_group_done(&self) -> bool {
        self.ranges_left.load(Ordering::Acquire) == 0
    }

    /// Takes on `range`. A range starting at the row this decoder stopped
    /// at continues from the page it has open; any other range repositions
    /// every leaf at the range's first page.
    ///
    /// A dictionary encoded page in a range that carries no dictionary
    /// could never be decoded, so it is an error rather than a wait.
    pub fn attach(&mut self, range: DecodeRange) -> Result<()> {
        assert!(self.exhausted(), "a decoder takes on one range at a time");
        for leaf in 0..self.leaf_decoders.len() {
            let column = range.column(leaf);
            if column.dictionary.is_none()
                && column
                    .pages
                    .iter()
                    .any(|page| page.content.is_dictionary_encoded())
            {
                return Err(Error::MissingDictionary { leaf });
            }
        }
        let rows = range.rows();
        let continues = rows.start as usize == self.end_row;
        self.live_in_range = self.live_rows_of(&rows);
        self.next_row = rows.start as usize;
        self.end_row = rows.end as usize;
        for leaf in 0..self.leaf_decoders.len() {
            let pages = &range.column(leaf).pages;
            if !continues {
                let first = pages
                    .first()
                    .expect("a ready range has pages in every column")
                    .idx;
                self.leaf_decoders[leaf].restart_at_page(first);
                self.next_page[leaf] = first;
            }
            for page in pages {
                if page.idx < self.next_page[leaf] {
                    continue;
                }
                self.next_page[leaf] = page.idx + 1;
                let decompressed = self.page_to_decode(leaf, page, &rows);
                self.leaf_decoders[leaf].insert_page(decompressed);
            }
        }
        self.range = Some(range);
        Ok(())
    }

    /// Gives back the range once every row of it is emitted.
    pub fn finish_range(&mut self) -> DecodeRange {
        assert!(
            self.exhausted(),
            "a range is finished once its rows are emitted"
        );
        self.range
            .take()
            .expect("a range is finished once, after being attached")
    }

    /// The stored page as the leaf decoder takes it. The first page of a
    /// range that starts inside it gets a mask skipping the rows before the
    /// range; an index selection's pages carry their own masks.
    fn page_to_decode(
        &self,
        leaf: usize,
        page: &StoredPage,
        rows: &Range<u32>,
    ) -> DecompressedPage {
        let data = match &page.content {
            PageContent::Data(data) => {
                let filter_mask = match self.metadata.selection() {
                    RowSelection::All if page.first_row < rows.start => {
                        let span = page.row_span();
                        Some(FilterMask::for_row_range(
                            span.start,
                            span.end,
                            &(rows.start..span.end),
                        ))
                    }
                    RowSelection::All => None,
                    RowSelection::Indices(_) => data.filter_mask.clone(),
                };
                DecompressedPageType::Data(DataPage {
                    header: data.header.clone(),
                    data: data.data.clone(),
                    filter_mask,
                })
            }
            PageContent::Skipped(header) => DecompressedPageType::SkippedData {
                header: header.clone(),
            },
        };
        DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: self.metadata.clone(),
            column_idx: leaf,
            idx: page.idx,
            first_row: page.first_row,
            data,
        }
    }

    /// The rows of `rows`, a range of the row group's rows, the table format
    /// left live, as a bit per row, when it deleted any of them. The
    /// materializer's index list was drawn from rows that already passed
    /// here, so an index selection has none deleted.
    fn live_rows_of(&self, rows: &Range<u32>) -> Option<BooleanBuffer> {
        let metadata = self.metadata.get_metadata();
        if metadata.deleted_rows.is_none()
            || !matches!(self.metadata.selection(), RowSelection::All)
            || metadata.count_deleted_in(rows.clone()) == 0
        {
            return None;
        }
        let mut live = BooleanBufferBuilder::new(rows.len());
        live.append_n(rows.len(), true);
        for position in metadata.deleted_rows_in(rows.clone()) {
            live.set_bit((position - rows.start) as usize, false);
        }
        Some(live.finish())
    }

    /// Whether every row of the attached range has been emitted.
    pub fn exhausted(&self) -> bool {
        self.next_row == self.end_row
    }

    /// Attempts to produce the next [`DecodedBatch`].
    ///
    /// Returns `Ok(Some(batch))` if every column decoder has enough buffered
    /// rows, `Ok(None)` if more pages are needed, or an error if decoding
    /// fails.
    pub fn try_read(&mut self, allocator: &mut SlabAllocator) -> Result<Option<DecodedBatch>> {
        let size = min(self.batch_size, self.end_row - self.next_row);
        let available = min(
            self.leaf_decoders
                .iter()
                .map(|leaf| leaf.available())
                .min()
                .unwrap(),
            size,
        );

        if size > 0 && available > 0 {
            // Every distinct leaf is decoded once; the output columns then fold
            // their own views of the result, so two extracts over one variant
            // column share the decode instead of repeating it.
            let decoded = self
                .leaf_decoders
                .iter_mut()
                .map(|leaf| leaf.read(allocator, available).map_err(Into::into))
                .collect::<Result<Vec<_>>>()?;
            let columns = self
                .plan
                .column_decoders
                .iter()
                .map(|decoder| decoder.read(&decoded))
                .collect::<Result<Vec<_>>>()?;
            let record_batch = RecordBatch::try_new(self.plan.schema.clone(), columns)?;
            let first_row = self.next_row;
            self.next_row += available;
            // The rows the table format deleted are named by the batch's
            // bits rather than dropped here, for the parent to copy the rest
            // into full batches.
            let live_rows = self.live_in_range.as_ref().and_then(|live| {
                let live = live.slice(live.len() - (self.end_row - first_row), available);
                (live.count_set_bits() < available).then_some(live)
            });
            // Give each column carrying a pushed-down equality constant a
            // chance to drop rows that provably fail it, before the batch
            // travels any further. Dictionary encoded string columns filter it
            // with a cheap view comparison; the rest leave it untouched. Never
            // done on the materializer path, which needs every row to stay in
            // place, nor to a batch whose live rows are named by its bits.
            let record_batch = if self.filter_batches && live_rows.is_none() {
                self.plan.prunable.iter().fold(record_batch, |batch, p| {
                    self.leaf_decoders[p.leaf].fast_filter_record_batch(batch, p.output_idx)
                })
            } else {
                record_batch
            };
            let batch = if self.add_row_group_metadata {
                with_row_index(record_batch, first_row)
            } else {
                record_batch
            };
            Ok(Some(DecodedBatch { batch, live_rows }))
        } else {
            Ok(None)
        }
    }
}

/// Decides which output columns the row group can be pruned by, and the
/// constant each one's leaf decoder gets.
///
/// A constant only goes in when its output column emits exactly one decoded
/// leaf unchanged, because that is the only shape both uses of the constant can
/// read: the row group is pruned from the leaf's own dictionary, and the batch
/// filter compares the emitted array against the dictionary's view of the
/// constant. A column that casts its leaf or reconstructs a variant emits
/// something else entirely.
///
/// It also goes in only when every data page of that chunk is dictionary
/// encoded. The decoder uses the constant to skip building a dictionary that
/// excludes it, which is sound only when an excluded dictionary prunes the whole
/// row group. A PLAIN fallback page could hold the constant even if the
/// dictionary does not, so such a chunk is still scanned and its dictionary must
/// be built to decode it.
///
/// A constant whose type does not match the leaf is dropped by the leaf decoder,
/// which forgoes the pushdown; the query's `Filter` still applies the
/// comparison.
fn prunable_columns(
    column_decoders: &[ColumnDecoder],
    file_leaves: &[usize],
    column_chunks: &[ColumnChunkMeta],
    projection: &Projection,
    eq_predicates: &[ScanEqualityPredicate],
) -> (Vec<PrunableColumn>, Vec<(usize, Scalar<ArrayRef>)>) {
    let mut prunable = Vec::new();
    let mut eq_constants = Vec::new();
    for (output_idx, &column) in projection.column_indices.iter().enumerate() {
        let Some(leaf) = column_decoders[output_idx].untransformed_leaf() else {
            continue;
        };
        // A whole-column read answers a comparison on the column itself; a
        // pushed extract answers one that names the very path it reads.
        let path = projection
            .extract_at(output_idx)
            .map(|extract| extract.path.as_slice())
            .unwrap_or_default();
        let Some(predicate) = eq_predicates
            .iter()
            .find(|p| p.column_idx == column && p.path == path)
        else {
            continue;
        };
        if !column_chunks[file_leaves[leaf]].data_pages_all_dictionary {
            continue;
        }
        eq_constants.push((leaf, predicate.value.clone()));
        prunable.push(PrunableColumn { output_idx, leaf });
    }
    (prunable, eq_constants)
}
