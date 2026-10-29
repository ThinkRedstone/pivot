//! [`TypedLeafDecoder`] — the generic, type-parameterised implementation of
//! [`LeafDecoder`].
//!
//! This struct is parameterised over a [`LeafType`], which names everything
//! that decoding one Parquet column type needs:
//! - its two dictionary flavours; which one is built depends on whether the
//!   dictionary page arrives in one contiguous buffer or scattered across
//!   several (see [`DictStorage`]).
//! - its [`ArrayBuilder`], which accumulates decoded values into an Arrow
//!   array.
//! - its decoder for each encoding a data page may carry.
//!
//! Concrete column decoders (e.g. `PrimitiveLeafDecoder`, `BytesViewDecoder`)
//! are type aliases over `TypedLeafDecoder` with the appropriate leaf type.

use super::{BuiltDictionary, SharedDictionary};
use crate::reading::decoding::leaf_decoders::levels::decode_def_levels;
use crate::reading::decoding::leaf_decoders::rle::RleDecoder;
use crate::reading::decoding::leaf_decoders::{
    ArrayBuilder, DecodeEncoding, DecodePlain, Dict, DictFromBytes, DictFromVecBytes, Error,
    LeafDecoder, LeafType, Result,
};
use crate::thrift::general::Encoding;
use crate::thrift::headers::DataPageHeader;
use crate::thrift::headers::DictionaryPageHeader;
use crate::types::filter_mask::RunningFilterMask;
use crate::types::page::{DataPage, DecompressedPage, DecompressedPageType};
use arrow_array::{ArrayRef, RecordBatch, Scalar};
use bytes::Bytes;
use dispatch::arrays::ValidityBuilder;
use dispatch::memory::{MultiBufferReader, ReaderPosition, SlabAllocator};
use std::sync::Arc;

/// The decoder of a data page's values, one variant per encoding a page may
/// carry: PLAIN, indices into the dictionary, or one of the encodings the
/// leaf type names a decoder for.
pub enum ValueDecoder<L: LeafType> {
    Plain(L::Plain),
    Dictionary(RleDecoder),
    DeltaBinaryPacked(L::DeltaBinaryPacked),
    DeltaLengthByteArray(L::DeltaLengthByteArray),
    Rle(L::Rle),
}

/// Decode `n` values into `builder` (a dictionary page looks each index up in
/// `dict`). Shared by the required and nullable read loops.
fn decode_run<D, L>(
    decoder: &mut ValueDecoder<L>,
    builder: &mut L::Builder,
    dict: Option<&D>,
    n: usize,
) where
    D: Dict<Builder = L::Builder, Item = <L::Builder as ArrayBuilder>::Element>,
    L: LeafType,
{
    match decoder {
        ValueDecoder::Plain(p) => p.read(builder, n),
        ValueDecoder::Dictionary(r) => r.read(builder, dict.expect("No dict available!"), n),
        ValueDecoder::DeltaBinaryPacked(d) => d.read(builder, n),
        ValueDecoder::DeltaLengthByteArray(d) => d.read(builder, n),
        ValueDecoder::Rle(d) => d.read(builder, n),
    }
}

/// Advance past `n` values without decoding them (a filtered-out run).
fn skip_run<L: LeafType>(decoder: &mut ValueDecoder<L>, n: usize) {
    match decoder {
        ValueDecoder::Plain(p) => p.skip(n),
        ValueDecoder::Dictionary(r) => r.skip(n),
        ValueDecoder::DeltaBinaryPacked(d) => d.skip(n),
        ValueDecoder::DeltaLengthByteArray(d) => d.skip(n),
        ValueDecoder::Rle(d) => d.skip(n),
    }
}

/// A nullable page's definition levels: `present[i]` is whether logical row `i`
/// holds a value (vs a null), consumed run-wise via a cursor.
struct DefLevels {
    present: Vec<bool>,
    pos: usize,
}

impl DefLevels {
    /// The next run of equal present/null rows, capped at `max`.
    fn next_run(&mut self, max: usize) -> (bool, usize) {
        let present = self.present[self.pos];
        let run = self.present[self.pos..]
            .iter()
            .take_while(|&&p| p == present)
            .count()
            .min(max);
        self.pos += run;
        (present, run)
    }
}

/// State for the page currently being read.
///
/// Tracks the value decoder, the optional filter mask cursor, how many rows
/// remain in the page, and, for a nullable column, the page's definition
/// levels (which logical rows are null).
pub struct ReadPage<L: LeafType> {
    /// The decoder for this page's values, whichever encoding it carries.
    decoder: ValueDecoder<L>,
    /// Filter mask cursor; `None` when the full page is kept.
    running_filter_mask_opt: Option<RunningFilterMask>,
    /// Rows left to decode in this page.
    pub(crate) remaining: usize,
    /// Definition levels for a page with nulls; `None` for a required column or
    /// an all-present page.
    def_levels: Option<DefLevels>,
}

impl<L: LeafType> ReadPage<L> {
    /// Decodes up to `size` rows from this page into `builder`.
    ///
    /// When a [`RunningFilterMask`] is present, false runs are skipped and
    /// only true runs are decoded, so the actual number of values pushed may
    /// be less than `size`.
    pub fn read_into<D>(&mut self, dict: Option<&D>, builder: &mut L::Builder, size: usize)
    where
        D: Dict<Builder = L::Builder, Item = <L::Builder as ArrayBuilder>::Element>,
    {
        let current_len = builder.len();
        let mut read_left = size.min(self.remaining);
        while read_left > 0 {
            let (keep, next_run) = match self.running_filter_mask_opt.as_mut() {
                Some(m) => m.next_run(read_left),
                None => (true, read_left),
            };
            if keep {
                decode_run(&mut self.decoder, builder, dict, next_run);
                read_left -= next_run;
            } else {
                skip_run(&mut self.decoder, next_run);
            }
        }
        self.remaining -= builder.len() - current_len;
    }

    /// Like [`read_into`](Self::read_into) but for a page with nulls: each
    /// logical row is decoded as a value (present) or appended as a null, per
    /// the page's definition levels, and a validity bit is pushed for each
    /// kept row. The value stream holds only the present values.
    ///
    /// A filter mask (from a pushed-down predicate) and definition levels are
    /// combined: filter runs partition the rows into kept/skipped, and within
    /// each, definition-level runs partition them into present/null. Skipped
    /// rows still advance the value stream past their present values; only
    /// kept rows reach the builder and validity. `remaining` and `size` count
    /// *kept* rows (`DataPage::rows()` is the kept count when filtered).
    pub fn read_into_nullable<D>(
        &mut self,
        dict: Option<&D>,
        builder: &mut L::Builder,
        validity: &mut ValidityBuilder,
        size: usize,
    ) where
        D: Dict<Builder = L::Builder, Item = <L::Builder as ArrayBuilder>::Element>,
    {
        let start_len = builder.len();
        let mut kept_left = size.min(self.remaining);
        while kept_left > 0 {
            // A filter run covers kept rows (capped at `kept_left`) or a full
            // skipped run; the whole page is kept when there's no mask.
            let (keep, run) = match self.running_filter_mask_opt.as_mut() {
                Some(m) => m.next_run(kept_left),
                None => (true, kept_left),
            };
            let def = self
                .def_levels
                .as_mut()
                .expect("nullable page has def levels");
            let mut run_left = run;
            while run_left > 0 {
                let (present, n) = def.next_run(run_left);
                match (keep, present) {
                    // A kept value decodes; a kept null gets a zeroed slot
                    // that `validity` masks off. The slot must hold a valid
                    // element (zero is an empty inline view / zero primitive):
                    // branchless kernels read masked slots before applying
                    // validity, and byte-view arrays require every view to be
                    // safe to interpret.
                    (true, true) => decode_run(&mut self.decoder, builder, dict, n),
                    (true, false) => {
                        builder.push(&<L::Builder as ArrayBuilder>::Element::default(), n)
                    }
                    // Skipped present rows still consume their values; skipped
                    // nulls have no value in the stream.
                    (false, true) => skip_run(&mut self.decoder, n),
                    (false, false) => {}
                }
                if keep {
                    validity.append_n(n, present);
                }
                run_left -= n;
            }
            if keep {
                kept_left -= run;
            }
        }
        self.remaining -= builder.len() - start_len;
    }
}

/// Tracks whether a page slot holds decodable data or was fully filtered out.
#[allow(clippy::large_enum_variant)]
enum PageSlot {
    Skipped,
    Data(DataPage),
}

/// The dictionary, in whichever representation its page's buffer shape
/// allowed: a page held in one contiguous buffer builds the `Contiguous`
/// flavour, a page scattered across buffers the `Scattered` one.
///
/// The variant is picked once when the dictionary page arrives; readers match
/// it once per call and run monomorphic code from there, so lookups never
/// branch per element.
enum DictStorage<DC, DS> {
    Contiguous(DC),
    Scattered(DS),
}

impl<DC: DictFromBytes, DS: DictFromVecBytes> DictStorage<DC, DS> {
    fn build(mut data: Vec<Bytes>, size: usize, allocator: &mut SlabAllocator) -> Self {
        if data.len() == 1 {
            let bytes = data.pop().expect("data holds one buffer");
            Self::Contiguous(DC::new_from_bytes(bytes, size, allocator))
        } else {
            Self::Scattered(DS::new_from_vec_bytes(data, size, allocator))
        }
    }
}

/// Generic column decoder over a [`LeafType`].
///
/// Accumulates [`DecompressedPage`]s and decodes them into Arrow arrays on
/// demand. See the [module docs](self) for what the leaf type provides.
pub struct TypedLeafDecoder<L: LeafType> {
    /// Indexed by page number. `None` means the page hasn't arrived yet.
    pages: Vec<Option<PageSlot>>,
    /// Index of the next page to decode.
    page_idx: usize,
    /// Maximum definition level for this column (0 = non-nullable).
    max_def_level: i16,
    /// The page currently being consumed, if any.
    read_page: Option<ReadPage<L>>,
    /// The dictionary, once built here or adopted from another reader of the
    /// row group on this node.
    dict: Option<Arc<DictStorage<L::ContiguousDict, L::ScatteredDict>>>,
    /// The pushed-down equality constant, in this dictionary flavour's own
    /// representation (see [`Dict::EqConstant`]). Drives row-group pruning
    /// and scan-side batch filtering once the dictionary is built.
    eq_const: Option<<L::ContiguousDict as Dict>::EqConstant>,
}

impl<L: LeafType> TypedLeafDecoder<L> {
    /// Creates a new decoder for a column with the given maximum definition
    /// level. Use `0` for non-nullable columns.
    pub fn new(max_def_level: i16) -> Self {
        Self {
            pages: Vec::new(),
            page_idx: 0,
            max_def_level,
            read_page: None,
            dict: None,
            eq_const: None,
        }
    }

    /// Queues page `idx` for reading. Pages are handed over in reading
    /// order, never one already read.
    fn queue_page(&mut self, idx: usize, slot: PageSlot) {
        assert!(idx >= self.page_idx, "a page is queued before it is read");
        if idx >= self.pages.len() {
            self.pages.resize_with(idx + 1, || None);
        }
        self.pages[idx] = Some(slot);
    }

    /// Selects the [`ValueDecoder`] for a page from its header's encoding.
    fn create_decoder(
        &self,
        header: DataPageHeader,
        data: Vec<Bytes>,
        mut position: ReaderPosition,
    ) -> Result<ValueDecoder<L>> {
        let decoder = match header.encoding {
            Encoding::PLAIN => Some(ValueDecoder::Plain(L::Plain::new(data, position))),
            Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY => {
                if self.dict.is_none() {
                    return Err(Error::NoPagesReady);
                }
                let bit_width = {
                    let mut reader = MultiBufferReader::new(&data, &mut position);
                    reader.read_u8()
                };
                Some(ValueDecoder::Dictionary(RleDecoder::new(
                    data, position, bit_width,
                )))
            }
            // The leaf type names a decoder for each of these, or
            // `Unsupported` for one it never reads.
            Encoding::DELTA_BINARY_PACKED => {
                L::DeltaBinaryPacked::new(data, position).map(ValueDecoder::DeltaBinaryPacked)
            }
            Encoding::DELTA_LENGTH_BYTE_ARRAY => {
                L::DeltaLengthByteArray::new(data, position).map(ValueDecoder::DeltaLengthByteArray)
            }
            Encoding::RLE => L::Rle::new(data, position).map(ValueDecoder::Rle),
            _ => None,
        };
        decoder.ok_or(Error::UnsupportedEncoding(header.encoding))
    }

    /// Advances to the next buffered data page and prepares it for reading.
    ///
    /// Skipped pages are silently stepped over. Returns `Ok(None)` when there
    /// are no more pages to consume.
    fn create_next_read_page(&mut self) -> Result<Option<()>> {
        if self.page_idx >= self.pages.len() {
            return Ok(None);
        }

        let page = loop {
            let page = self.pages[self.page_idx]
                .take()
                .ok_or(Error::NoPagesReady)?;
            if let PageSlot::Data(d) = page {
                break d;
            }
            self.page_idx += 1;
        };

        let mut position = ReaderPosition::default();
        // Definition levels cover every logical row in the page, but
        // `page.rows()` is the *kept* count once a filter mask is applied: so
        // decode `num_values` of them, while `remaining` tracks kept rows.
        let kept = page.rows();
        let num_values = page.header.num_values as usize;

        // A nullable column's pages prefix the values with definition levels;
        // decode them (advancing `position` past them). An all-present page
        // yields `None` and decodes by the fast, non-null path below.
        let def_levels = if self.max_def_level > 0 {
            let mut reader = MultiBufferReader::new(&page.data, &mut position);
            let def_level_byte_len = reader.read_u32_le() as usize;
            decode_def_levels(
                &mut reader,
                num_values,
                def_level_byte_len,
                self.max_def_level,
            )
            .map(|present| DefLevels { present, pos: 0 })
        } else {
            None
        };

        self.read_page = Some(ReadPage {
            decoder: self.create_decoder(page.header, page.data, position)?,
            running_filter_mask_opt: page.filter_mask.map(RunningFilterMask::new),
            remaining: kept,
            def_levels,
        });

        Ok(Some(()))
    }
}

impl<L: LeafType> LeafDecoder for TypedLeafDecoder<L> {
    fn set_eq_constant(&mut self, value: &Scalar<ArrayRef>) {
        self.eq_const = L::ContiguousDict::eq_constant_from_scalar(value);
    }

    fn fast_filter_record_batch(&self, batch: RecordBatch, column: usize) -> RecordBatch {
        match (&self.eq_const, self.dict.as_deref()) {
            (Some(needle), Some(DictStorage::Contiguous(dict))) => {
                dict.filter_record_batch_by_const(batch, column, needle)
            }
            (Some(needle), Some(DictStorage::Scattered(dict))) => {
                dict.filter_record_batch_by_const(batch, column, needle)
            }
            _ => batch,
        }
    }

    fn available(&self) -> usize {
        let (mut available, mut page_idx) = if let Some(s) = &self.read_page {
            (s.remaining, self.page_idx + 1)
        } else {
            (0, self.page_idx)
        };

        while page_idx < self.pages.len()
            && let Some(p) = &self.pages[page_idx]
        {
            available += match p {
                PageSlot::Data(d) => {
                    if self.dict.is_none()
                        && matches!(
                            d.header.encoding,
                            Encoding::RLE_DICTIONARY | Encoding::PLAIN_DICTIONARY
                        )
                    {
                        return available;
                    }
                    d.rows()
                }
                PageSlot::Skipped => 0,
            };
            page_idx += 1;
        }
        available
    }

    fn insert_page(&mut self, page: DecompressedPage) {
        match page.data {
            DecompressedPageType::Dict { .. } => {
                panic!("a dictionary page is built with build_dictionary, not inserted")
            }
            DecompressedPageType::Data(data) => self.queue_page(page.idx, PageSlot::Data(data)),
            DecompressedPageType::SkippedData { .. } => {
                self.queue_page(page.idx, PageSlot::Skipped)
            }
        }
    }

    /// Scans the raw page for the pushed-down constant before building, so
    /// a dictionary that excludes it is never materialized.
    fn build_dictionary(
        &self,
        header: DictionaryPageHeader,
        data: Vec<Bytes>,
        allocator: &mut SlabAllocator,
    ) -> BuiltDictionary {
        let size = header.num_values as usize;
        if let Some(needle) = self.eq_const.as_ref()
            && !L::ContiguousDict::maybe_contains(&data, size, needle)
        {
            return BuiltDictionary::Pruned;
        }
        let storage: DictStorage<L::ContiguousDict, L::ScatteredDict> =
            DictStorage::build(data, size, allocator);
        BuiltDictionary::Built(Arc::new(storage))
    }

    /// Every reader of a leaf decodes it with the same decoder type, so the
    /// dictionary downcasts to this one's storage.
    fn adopt_dictionary(&mut self, dictionary: SharedDictionary) {
        let dictionary = dictionary
            .downcast::<DictStorage<L::ContiguousDict, L::ScatteredDict>>()
            .unwrap_or_else(|_| panic!("a shared dictionary is adopted by its own decoder type"));
        self.dict = Some(dictionary);
    }

    fn restart_at_page(&mut self, page_idx: usize) {
        self.pages.clear();
        self.read_page = None;
        self.page_idx = page_idx;
    }

    fn read(&mut self, allocator: &mut SlabAllocator, size: usize) -> Result<ArrayRef> {
        let mut builder = L::Builder::with_capacity(allocator, size);
        match self.dict.as_deref() {
            Some(DictStorage::Contiguous(d)) => d.register_onto(&mut builder),
            Some(DictStorage::Scattered(d)) => d.register_onto(&mut builder),
            None => {}
        }
        // Validity is built lazily on slab memory only once a page actually has
        // a null. A required column, and a nullable one whose pages are all
        // present (e.g. a variant's `value` leaf), never allocates a bitmap and
        // takes the same fast path as before.
        let mut validity: Option<ValidityBuilder> = None;

        while builder.len() < size {
            // A page whose kept rows are all decoded is left in place until
            // the next read, so a reader that stops at the end of its rows
            // can continue from the same cursor.
            if self
                .read_page
                .as_ref()
                .is_some_and(|read_page| read_page.remaining == 0)
            {
                self.read_page = None;
                self.page_idx += 1;
            }
            if self.read_page.is_none() {
                match self.create_next_read_page()? {
                    Some(()) => {}
                    None => break,
                }
            }

            let remaining = size - builder.len();
            let read_page = self.read_page.as_mut().unwrap();
            if read_page.def_levels.is_some() {
                // This page has nulls: ensure a bitmap exists (backfilling the
                // rows decoded so far as present), then scatter values and nulls.
                let v = validity.get_or_insert_with(|| {
                    let mut vb = ValidityBuilder::with_capacity(allocator, size);
                    vb.append_n(builder.len(), true);
                    vb
                });
                match self.dict.as_deref() {
                    Some(DictStorage::Contiguous(d)) => {
                        read_page.read_into_nullable(Some(d), &mut builder, v, remaining)
                    }
                    Some(DictStorage::Scattered(d)) => {
                        read_page.read_into_nullable(Some(d), &mut builder, v, remaining)
                    }
                    None => read_page.read_into_nullable(
                        None::<&L::ContiguousDict>,
                        &mut builder,
                        v,
                        remaining,
                    ),
                }
            } else {
                let before = builder.len();
                match self.dict.as_deref() {
                    Some(DictStorage::Contiguous(d)) => {
                        read_page.read_into(Some(d), &mut builder, remaining)
                    }
                    Some(DictStorage::Scattered(d)) => {
                        read_page.read_into(Some(d), &mut builder, remaining)
                    }
                    None => {
                        read_page.read_into(None::<&L::ContiguousDict>, &mut builder, remaining)
                    }
                }
                if let Some(v) = &mut validity {
                    v.append_n(builder.len() - before, true);
                }
            }
        }

        let null_buffer = validity.map(|v| v.into_buffer());
        Ok(builder.into_array(null_buffer))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reading::decoding::leaf_decoders::primitive::PrimitiveLeafDecoder;
    use crate::test_utils::dummy_metadata;
    use crate::thrift::general::Encoding;
    use crate::thrift::headers::PageHeader;
    use crate::types::filter_mask::FilterMask;
    use crate::types::metadata::RowSelection;
    use crate::types::page::{DataPage, DecompressedPageType};
    use arrow_array::Int32Array;
    use arrow_array::types::Int32Type;
    use dispatch::memory::{SlabAllocator, init_test_free_pool};

    // -- Helpers --

    fn encode_i32s(values: &[i32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn int_scalar(value: i32) -> Scalar<ArrayRef> {
        Scalar::new(std::sync::Arc::new(Int32Array::from(vec![value])) as ArrayRef)
    }

    fn extract_i32s(arr: &ArrayRef) -> Vec<i32> {
        let a = arr.as_any().downcast_ref::<Int32Array>().unwrap();
        (0..a.len()).map(|i| a.value(i)).collect()
    }

    fn data_page(data: Vec<u8>, num_values: usize, idx: usize) -> DecompressedPage {
        let header = PageHeader::for_data_page(num_values as i32, Encoding::PLAIN);
        DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: dummy_metadata(RowSelection::All),
            column_idx: 0,
            idx,
            first_row: 0,
            data: DecompressedPageType::Data(DataPage {
                header: header.data_page_header.unwrap(),
                data: vec![Bytes::from(data)],
                filter_mask: None,
            }),
        }
    }

    fn filtered_data_page(
        data: Vec<u8>,
        num_values: usize,
        idx: usize,
        mask: FilterMask,
    ) -> DecompressedPage {
        let header = PageHeader::for_data_page(num_values as i32, Encoding::PLAIN);
        DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: dummy_metadata(RowSelection::All),
            column_idx: 0,
            idx,
            first_row: 0,
            data: DecompressedPageType::Data(DataPage {
                header: header.data_page_header.unwrap(),
                data: vec![Bytes::from(data)],
                filter_mask: Some(mask),
            }),
        }
    }

    fn skipped_page(num_values: usize, idx: usize) -> DecompressedPage {
        let header = PageHeader::for_data_page(num_values as i32, Encoding::PLAIN);
        DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: dummy_metadata(RowSelection::All),
            column_idx: 0,
            idx,
            first_row: 0,
            data: DecompressedPageType::SkippedData {
                header: header.data_page_header.unwrap(),
            },
        }
    }

    type Dec = PrimitiveLeafDecoder<Int32Type>;

    fn build_dictionary(dec: &Dec, entries: &[i32], alloc: &mut SlabAllocator) -> BuiltDictionary {
        let header = PageHeader::for_dict_page(entries.len() as i32)
            .dictionary_page_header
            .unwrap();
        dec.build_dictionary(header, vec![Bytes::from(encode_i32s(entries))], alloc)
    }

    fn load_dictionary(dec: &mut Dec, entries: &[i32], alloc: &mut SlabAllocator) {
        match build_dictionary(dec, entries, alloc) {
            BuiltDictionary::Built(dictionary) => dec.adopt_dictionary(dictionary),
            BuiltDictionary::Pruned => panic!("the dictionary excludes the constant"),
        }
    }

    // -- available --

    /// No pages inserted → nothing available.
    #[test]
    fn test_available_empty() {
        let dec = Dec::new(0);

        assert_eq!(dec.available(), 0);
    }

    /// Single page with 3 values → 3 available.
    #[test]
    fn test_available_single_page() {
        init_test_free_pool(4);
        let mut dec = Dec::new(0);
        dec.insert_page(data_page(encode_i32s(&[1, 2, 3]), 3, 0));

        assert_eq!(dec.available(), 3);
    }

    /// Two pages sum their rows for availability.
    #[test]
    fn test_available_two_pages() {
        init_test_free_pool(4);
        let mut dec = Dec::new(0);
        dec.insert_page(data_page(encode_i32s(&[1, 2]), 2, 0));
        dec.insert_page(data_page(encode_i32s(&[3, 4, 5]), 3, 1));

        assert_eq!(dec.available(), 5);
    }

    /// Skipped pages contribute 0 rows to availability.
    #[test]
    fn test_available_skipped_page_contributes_zero() {
        init_test_free_pool(4);
        let mut dec = Dec::new(0);
        dec.insert_page(skipped_page(10, 0));
        dec.insert_page(data_page(encode_i32s(&[1, 2]), 2, 1));

        assert_eq!(dec.available(), 2);
    }

    /// An integer leaf names no decoder for RLE, so a page in that encoding
    /// is reported as unsupported rather than decoded as something else.
    #[test]
    fn a_page_in_an_encoding_the_leaf_does_not_read_is_unsupported() {
        init_test_free_pool(4);
        let mut allocator = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        let header = PageHeader::for_data_page(3, Encoding::RLE);
        dec.insert_page(DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: dummy_metadata(RowSelection::All),
            column_idx: 0,
            idx: 0,
            first_row: 0,
            data: DecompressedPageType::Data(DataPage {
                header: header.data_page_header.unwrap(),
                data: vec![Bytes::from(encode_i32s(&[1, 2, 3]))],
                filter_mask: None,
            }),
        });

        let read = dec.read(&mut allocator, 3);

        assert!(matches!(
            read,
            Err(Error::UnsupportedEncoding(Encoding::RLE))
        ));
    }

    /// RLE-dict page without a dictionary → 0 available.
    #[test]
    fn test_available_dict_page_missing() {
        init_test_free_pool(4);
        let mut dec = Dec::new(0);

        let header = PageHeader::for_data_page(8, Encoding::RLE_DICTIONARY);
        let page = DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: dummy_metadata(RowSelection::All),
            column_idx: 0,
            idx: 0,
            first_row: 0,
            data: DecompressedPageType::Data(DataPage {
                header: header.data_page_header.unwrap(),
                data: vec![Bytes::from(vec![2u8, 3, 0x00, 0x00])],
                filter_mask: None,
            }),
        };
        dec.insert_page(page);

        assert_eq!(dec.available(), 0);
    }

    // -- Dictionary pruning (pushed-down equality constant) --

    /// No constant set: the dictionary is built.
    #[test]
    fn a_dictionary_is_built_without_a_constant() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let dec = Dec::new(0);

        let built = build_dictionary(&dec, &[10, 20, 30], &mut alloc);

        assert!(matches!(built, BuiltDictionary::Built(_)));
    }

    /// Constant present in the dictionary: the dictionary is built.
    #[test]
    fn a_dictionary_holding_the_constant_is_built() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        dec.set_eq_constant(&int_scalar(20));

        let built = build_dictionary(&dec, &[10, 20, 30], &mut alloc);

        assert!(matches!(built, BuiltDictionary::Built(_)));
    }

    /// Constant absent from the dictionary: the row group is pruned.
    #[test]
    fn a_dictionary_excluding_the_constant_prunes() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        dec.set_eq_constant(&int_scalar(99));

        let built = build_dictionary(&dec, &[10, 20, 30], &mut alloc);

        assert!(matches!(built, BuiltDictionary::Pruned));
    }

    // -- Page insertion order --

    /// Pages inserted out of order are read in index order.
    #[test]
    fn test_out_of_order_insertion() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        dec.insert_page(data_page(encode_i32s(&[30, 40]), 2, 1));
        dec.insert_page(data_page(encode_i32s(&[10, 20]), 2, 0));

        let result = dec.read(&mut alloc, 4).unwrap();

        assert_eq!(extract_i32s(&result), vec![10, 20, 30, 40]);
    }

    // -- Skipped pages --

    /// A skipped page is stepped over; read produces values from the next data page.
    #[test]
    fn test_skipped_page_stepped_over() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        dec.insert_page(skipped_page(5, 0));
        dec.insert_page(data_page(encode_i32s(&[10, 20]), 2, 1));

        let result = dec.read(&mut alloc, 2).unwrap();

        assert_eq!(extract_i32s(&result), vec![10, 20]);
    }

    /// Only skipped pages → no rows available.
    #[test]
    fn test_only_skipped_pages() {
        init_test_free_pool(4);
        let mut dec = Dec::new(0);
        dec.insert_page(skipped_page(5, 0));

        assert_eq!(dec.available(), 0);
    }

    // -- Incremental reads --

    /// Reading fewer rows than available leaves the rest for the next read.
    #[test]
    fn test_incremental_read_within_page() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        dec.insert_page(data_page(encode_i32s(&[10, 20, 30, 40, 50]), 5, 0));

        let r1 = dec.read(&mut alloc, 2).unwrap();
        let r2 = dec.read(&mut alloc, 3).unwrap();

        assert_eq!(extract_i32s(&r1), vec![10, 20]);
        assert_eq!(extract_i32s(&r2), vec![30, 40, 50]);
    }

    /// Incremental reads spanning multiple pages.
    #[test]
    fn test_incremental_read_across_pages() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        dec.insert_page(data_page(encode_i32s(&[1, 2, 3]), 3, 0));
        dec.insert_page(data_page(encode_i32s(&[4, 5]), 2, 1));

        let r1 = dec.read(&mut alloc, 4).unwrap();
        let r2 = dec.read(&mut alloc, 1).unwrap();

        assert_eq!(extract_i32s(&r1), vec![1, 2, 3, 4]);
        assert_eq!(extract_i32s(&r2), vec![5]);
    }

    // -- Filter mask --

    /// Filtered page keeps only the matching rows.
    #[test]
    fn test_filter_mask_keeps_matching_rows() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        let mask = FilterMask::new(0, 5, &[1, 3]);
        dec.insert_page(filtered_data_page(
            encode_i32s(&[10, 20, 30, 40, 50]),
            5,
            0,
            mask,
        ));

        let result = dec.read(&mut alloc, 2).unwrap();

        assert_eq!(extract_i32s(&result), vec![20, 40]);
    }

    /// All-false filter mask → 0 rows available.
    #[test]
    fn test_filter_mask_all_false() {
        init_test_free_pool(4);
        let mut dec = Dec::new(0);
        let mask = FilterMask::new(0, 5, &[]);
        dec.insert_page(filtered_data_page(
            encode_i32s(&[10, 20, 30, 40, 50]),
            5,
            0,
            mask,
        ));

        assert_eq!(dec.available(), 0);
    }

    // -- Dict-encoded --

    /// Dict page followed by RLE data page decodes correctly.
    #[test]
    fn test_dict_then_data() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);
        load_dictionary(&mut dec, &[100, 200, 300], &mut alloc);

        // bit_width=2, 1 group of 8, indices [0,1,2,0,0,1,2,0]
        let header = PageHeader::for_data_page(8, Encoding::RLE_DICTIONARY);
        let rle_page = DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: dummy_metadata(RowSelection::All),
            column_idx: 0,
            idx: 0,
            first_row: 0,
            data: DecompressedPageType::Data(DataPage {
                header: header.data_page_header.unwrap(),
                data: vec![Bytes::from(vec![2u8, 3, 0x24, 0x24])],
                filter_mask: None,
            }),
        };
        dec.insert_page(rle_page);

        let result = dec.read(&mut alloc, 8).unwrap();

        assert_eq!(
            extract_i32s(&result),
            vec![100, 200, 300, 100, 100, 200, 300, 100]
        );
    }

    /// available() returns 0 until dict page arrives, then reports the full count.
    fn rle_page_over_three_entries(idx: usize) -> DecompressedPage {
        let header = PageHeader::for_data_page(8, Encoding::RLE_DICTIONARY);
        DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: dummy_metadata(RowSelection::All),
            column_idx: 0,
            idx,
            first_row: 0,
            data: DecompressedPageType::Data(DataPage {
                header: header.data_page_header.unwrap(),
                data: vec![Bytes::from(vec![2u8, 3, 0x24, 0x24])],
                filter_mask: None,
            }),
        }
    }

    /// A decoder handed the dictionary another one built reads through it,
    /// not through a dictionary page of its own.
    #[test]
    fn a_decoder_adopts_a_dictionary_another_one_built() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let first = Dec::new(0);
        let mut second = Dec::new(0);
        let BuiltDictionary::Built(dictionary) =
            build_dictionary(&first, &[100, 200, 300], &mut alloc)
        else {
            panic!("no constant is set")
        };

        second.adopt_dictionary(dictionary);
        second.insert_page(rle_page_over_three_entries(0));
        let values = second.read(&mut alloc, 8).unwrap();

        assert_eq!(
            values
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values(),
            &[100, 200, 300, 100, 100, 200, 300, 100]
        );
    }

    #[test]
    fn test_dict_page_unblocks_availability() {
        init_test_free_pool(4);
        let mut alloc = SlabAllocator::new(true);
        let mut dec = Dec::new(0);

        let header = PageHeader::for_data_page(8, Encoding::RLE_DICTIONARY);
        let rle_page = DecompressedPage {
            worker_id: 0,
            query_row_group_metadata: dummy_metadata(RowSelection::All),
            column_idx: 0,
            idx: 0,
            first_row: 0,
            data: DecompressedPageType::Data(DataPage {
                header: header.data_page_header.unwrap(),
                data: vec![Bytes::from(vec![2u8, 3, 0x24, 0x24])],
                filter_mask: None,
            }),
        };
        dec.insert_page(rle_page);
        assert_eq!(dec.available(), 0);

        load_dictionary(&mut dec, &[100, 200, 300], &mut alloc);

        assert_eq!(dec.available(), 8);
    }
}
