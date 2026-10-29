//! Page mechanics shared by both encode strategies ([`plain`](super::plain) and
//! [`dictionary`](super::dictionary)): cut a leaf into ~1 MiB pages, frame a
//! page body behind its definition levels, and wrap it into an [`EncodedPage`].
//!
//! Pages are cut in *row* space, not value space: a leaf under a nullable path
//! stores no value for an absent row but still spends a definition level on it,
//! so a page's row range and its value range are different things. For the flat
//! required columns that make up most of a table the two coincide, and a page's
//! values are a zero-copy slice of the column.

use std::ops::Range;

use crate::thrift::general::{CompressionCodec, Encoding, PageType};
use crate::thrift::headers::{DataPageHeader, DictionaryPageHeader, PageHeader};
use crate::thrift::parquet_thrift::{ThriftCompactOutputProtocol, WriteThrift};
use arrow_array::cast::AsArray;
use arrow_array::{ArrayRef, BinaryViewArray, StringArray, StringViewArray};
use arrow_schema::{DataType, TimeUnit};

use dispatch::memory::SlabAllocator;

use super::super::compression;
use super::super::error::{WriteError, WriteResult};
use super::super::leaves::Leaf;
use super::super::types::EncodedPage;
use super::rle;

/// Target uncompressed size of one data page. Matches Parquet's usual ~1 MiB
/// data page size: large enough to amortize per-page overhead and compress
/// well, small enough to be a useful unit of parallel work.
const TARGET_PAGE_SIZE: usize = 1024 * 1024;

/// Bytes a PLAIN-encoded BYTE_ARRAY value spends before its data: a little-endian
/// `u32` length prefix (matches the BYTE_ARRAY encoding in [`plain`](super::plain)).
/// Counted per value when estimating a page's encoded size.
const BYTE_ARRAY_LEN_PREFIX: usize = size_of::<u32>();

/// One page's extent, in rows and in stored values. The two differ only for a
/// leaf with absent rows, which spend a definition level but no value.
pub(super) struct PageRange {
    pub(super) rows: Range<usize>,
    pub(super) values: Range<usize>,
}

/// Cut a leaf into ~[`TARGET_PAGE_SIZE`] pages of contiguous rows.
pub(super) fn page_ranges(leaf: &Leaf) -> WriteResult<Vec<PageRange>> {
    page_ranges_of_size(leaf, TARGET_PAGE_SIZE)
}

/// Cut a leaf into pages of about `target` uncompressed value bytes each. A leaf
/// whose every row is absent still yields one page — levels, no values — which
/// is what a shredded path's unused `value` fallback looks like.
///
/// The page is measured in bits, since a boolean's PLAIN form is one.
fn page_ranges_of_size(leaf: &Leaf, target: usize) -> WriteResult<Vec<PageRange>> {
    let rows = leaf.rows();
    if rows == 0 {
        return Ok(Vec::new());
    }
    let sizes = PlainSizes::new(&leaf.values)?;
    let target_bits = target * 8;

    let mut ranges = Vec::new();
    let (mut row_start, mut value_start, mut value, mut page_bits) = (0, 0, 0, 0usize);
    for row in 0..rows {
        if leaf.is_present(row) {
            page_bits += sizes.bits_at(value);
            value += 1;
        }
        if page_bits >= target_bits {
            ranges.push(PageRange {
                rows: row_start..row + 1,
                values: value_start..value,
            });
            (row_start, value_start, page_bits) = (row + 1, value, 0);
        }
    }
    // The rows left over once the last page was cut — and, for a leaf that never
    // reached `target`, the only page.
    if row_start < rows {
        ranges.push(PageRange {
            rows: row_start..rows,
            values: value_start..value,
        });
    }
    Ok(ranges)
}

/// A data page's body: its definition levels, then the encoded values. Parquet
/// frames a v1 page's levels as an RLE stream behind a 4-byte little-endian
/// length; a leaf nothing on whose path is nullable carries no level section at
/// all, which is what the reader assumes when `max_def_level` is 0.
pub(super) fn data_page_body(
    def_levels: Option<&[i16]>,
    max_def_level: i16,
    values: Vec<u8>,
) -> Vec<u8> {
    let Some(levels) = def_levels else {
        return values;
    };
    let levels = rle::encode_levels(levels, max_def_level);
    let mut body = Vec::with_capacity(BYTE_ARRAY_LEN_PREFIX + levels.len() + values.len());
    body.extend_from_slice(&(levels.len() as u32).to_le_bytes());
    body.extend_from_slice(&levels);
    body.extend_from_slice(&values);
    body
}

/// Measures the PLAIN size of a column's values. The value type is resolved once
/// per leaf, so the page cutter's inner loop is a size lookup rather than a
/// downcast per value.
enum PlainSizes<'a> {
    /// One bit per value.
    Boolean,
    /// Fixed-width values, all of this many bytes.
    Fixed(usize),
    Utf8(&'a StringArray),
    Utf8View(&'a StringViewArray),
    BinaryView(&'a BinaryViewArray),
}

impl<'a> PlainSizes<'a> {
    fn new(values: &'a ArrayRef) -> WriteResult<Self> {
        Ok(match values.data_type() {
            DataType::Boolean => Self::Boolean,
            // A date is a day count, stored as the INT32 it is annotated as, and
            // a value narrower than its physical type takes that type's width
            // (the narrow integers widened to INT32).
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Float32
            | DataType::Date32
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32 => Self::Fixed(4),
            DataType::Int64
            | DataType::Float64
            | DataType::UInt64
            | DataType::Timestamp(TimeUnit::Microsecond, _) => Self::Fixed(8),
            // A decimal's width follows its precision-chosen storage.
            DataType::Decimal64(precision, _) | DataType::Decimal128(precision, _) => {
                Self::Fixed(crate::decimal_write_storage(*precision).byte_width())
            }
            DataType::Utf8 => Self::Utf8(values.as_string()),
            DataType::Utf8View => Self::Utf8View(values.as_string_view()),
            DataType::BinaryView => Self::BinaryView(values.as_binary_view()),
            other => return Err(WriteError::UnsupportedType(other.clone())),
        })
    }

    /// The PLAIN size of value `i` in bits: a boolean's one bit, a fixed-width
    /// value's width, or a BYTE_ARRAY's length prefix plus its bytes.
    fn bits_at(&self, i: usize) -> usize {
        match self {
            Self::Boolean => 1,
            Self::Fixed(width) => *width * 8,
            Self::Utf8(values) => (BYTE_ARRAY_LEN_PREFIX + values.value(i).len()) * 8,
            Self::Utf8View(values) => (BYTE_ARRAY_LEN_PREFIX + values.value(i).len()) * 8,
            Self::BinaryView(values) => (BYTE_ARRAY_LEN_PREFIX + values.value(i).len()) * 8,
        }
    }
}

/// Which page to build — everything in the page header except the body sizes,
/// which [`assemble_page`] fills in after compressing.
pub(super) enum PageKind {
    Data {
        num_values: usize,
        encoding: Encoding,
    },
    Dictionary {
        num_values: usize,
    },
}

/// Compress `raw` (an encoded page body), prepend the page header, and tally
/// the sizes into an [`EncodedPage`].
pub(super) fn assemble_page(
    num_rows: i64,
    raw: Vec<u8>,
    kind: PageKind,
    codec: CompressionCodec,
    allocator: &mut SlabAllocator,
) -> WriteResult<EncodedPage> {
    let compressed = compression::compress(codec, &raw)?;
    let (page_type, data_page_header, dictionary_page_header) = match kind {
        PageKind::Data {
            num_values,
            encoding,
        } => (
            PageType::DATA_PAGE,
            Some(DataPageHeader {
                num_values: num_values as i32,
                encoding,
                // The definition levels are RLE (see `data_page_body`); a
                // required leaf writes none, but the field is required. Nothing
                // here repeats, so the repetition levels are always absent.
                definition_level_encoding: Encoding::RLE,
                repetition_level_encoding: Encoding::RLE,
                statistics: None,
            }),
            None,
        ),
        PageKind::Dictionary { num_values } => (
            PageType::DICTIONARY_PAGE,
            None,
            Some(DictionaryPageHeader {
                num_values: num_values as i32,
                encoding: Encoding::PLAIN,
                is_sorted: None,
            }),
        ),
    };
    let header = PageHeader {
        r#type: page_type,
        uncompressed_page_size: raw.len() as i32,
        compressed_page_size: compressed.len() as i32,
        crc: None,
        data_page_header,
        index_page_header: None,
        dictionary_page_header,
        data_page_header_v2: None,
    };

    // The page on the wire is the thrift header followed by the compressed body.
    // Both are written once, into a slab sized to hold exactly them: this is the
    // only copy a page's bytes make, since the assembler moves the slab rather
    // than concatenating what is in it.
    let mut head = Vec::new();
    header.write_thrift(&mut ThriftCompactOutputProtocol::new(&mut head))?;
    let header_len = head.len();
    let mut bytes = allocator.get_slab_of_size(header_len + compressed.len(), false);
    let page = bytes.as_mut_slice();
    page[..header_len].copy_from_slice(&head);
    page[header_len..].copy_from_slice(&compressed);

    Ok(EncodedPage {
        num_rows,
        uncompressed_size: raw.len(),
        header_len,
        bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writing::leaves;
    use arrow_array::{Array, Int64Array};
    use arrow_schema::Field;
    use std::sync::Arc;

    /// A leaf over `values` with no absent rows.
    fn required(values: ArrayRef) -> Leaf {
        leaf(values, false)
    }

    /// A leaf over a nullable column, so absent rows spend a level but no value.
    fn nullable(values: ArrayRef) -> Leaf {
        leaf(values, true)
    }

    fn leaf(values: ArrayRef, nullable: bool) -> Leaf {
        let field = Field::new("n", values.data_type().clone(), nullable);
        leaves::to_parquet_leaves(&field, &values)
            .unwrap()
            .pop()
            .unwrap()
    }

    fn ranges(leaf: &Leaf, target: usize) -> Vec<(Range<usize>, Range<usize>)> {
        page_ranges_of_size(leaf, target)
            .unwrap()
            .into_iter()
            .map(|r| (r.rows, r.values))
            .collect()
    }

    #[test]
    fn fixed_width_pages_split_by_row_count() {
        let leaf = required(Arc::new(Int64Array::from((0..250).collect::<Vec<i64>>())));

        // 8 bytes/value, target 800 bytes => 100 rows/page => 100, 100, 50.
        let cuts = ranges(&leaf, 800);

        assert_eq!(
            cuts,
            vec![(0..100, 0..100), (100..200, 100..200), (200..250, 200..250)]
        );
    }

    #[test]
    fn variable_width_pages_split_by_byte_size() {
        // Each value encodes as 4 + 6 = 10 bytes ("abcdef").
        let leaf = required(Arc::new(StringArray::from(vec!["abcdef"; 10])));

        // Target 25 bytes => cut after 3 values (30 >= 25): 3, 3, 3, 1.
        let cuts = ranges(&leaf, 25);

        assert_eq!(
            cuts,
            vec![(0..3, 0..3), (3..6, 3..6), (6..9, 6..9), (9..10, 9..10)]
        );
    }

    /// Absent rows advance a page's rows but not its values, so a page covers
    /// more rows than it stores values.
    #[test]
    fn absent_rows_cost_a_row_but_not_a_value() {
        // 6 rows, every other one null => 3 stored values of 8 bytes.
        let values: Vec<Option<i64>> = (0..6).map(|i| (i % 2 == 0).then_some(i)).collect();
        let leaf = nullable(Arc::new(Int64Array::from(values)));

        // Target 16 bytes => cut once two values have accumulated (rows 0..3).
        let cuts = ranges(&leaf, 16);

        assert_eq!(cuts, vec![(0..3, 0..2), (3..6, 2..3)]);
    }

    /// A leaf that is absent on every row still yields one page: the reader needs
    /// its levels to know the rows are null, and a shredded path's unused `value`
    /// fallback is exactly this.
    #[test]
    fn an_all_absent_leaf_is_one_page_of_levels() {
        let leaf = nullable(Arc::new(Int64Array::from(vec![None::<i64>; 4])));

        let cuts = ranges(&leaf, 16);

        assert_eq!(cuts, vec![(0..4, 0..0)]);
    }

    /// A required leaf's page body is the values alone; a nullable one prefixes
    /// the RLE levels behind their byte length.
    #[test]
    fn only_a_nullable_leaf_carries_a_level_section() {
        assert_eq!(data_page_body(None, 0, vec![1, 2, 3]), vec![1, 2, 3]);

        let body = data_page_body(Some(&[1, 1]), 1, vec![9]);

        let levels = rle::encode_levels(&[1, 1], 1);
        let mut expected = (levels.len() as u32).to_le_bytes().to_vec();
        expected.extend_from_slice(&levels);
        expected.push(9);
        assert_eq!(body, expected);
    }
}
