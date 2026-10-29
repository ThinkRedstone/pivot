//! Traits and types for decoding individual Parquet leaf columns into Arrow
//! arrays.
//!
//! The core abstraction is [`LeafDecoder`], an object-safe trait used by
//! [`ColumnDecoder`](super::column_decoder::ColumnDecoder) to decode each of a
//! projected column's leaf column chunks independently. The concrete, generic
//! implementation lives in [`TypedLeafDecoder`] (in the [`typed`] submodule),
//! parameterised by a [`LeafType`], which names the pieces that together
//! describe how to decode a particular Parquet type:
//!
//! - [`ArrayBuilder`] — accumulates decoded values and produces an Arrow array.
//! - [`DecodePlain`] — reads plain-encoded values from raw page bytes.
//! - [`DecodeEncoding`] — reads values stored in one of the other encodings
//!   (`DELTA_BINARY_PACKED`, `DELTA_LENGTH_BYTE_ARRAY`, `RLE`); a leaf type
//!   names one decoder per encoding it reads and [`Unsupported`] for the rest.
//! - [`Dict`] - queries a dictionary for RLE-dictionary pages, built via
//!   [`DictFromBytes`] or [`DictFromVecBytes`] depending on the dictionary
//!   page's buffer shape.
//!
//! Concrete leaf decoders are type aliases over `TypedLeafDecoder`:
//! - [`primitive::PrimitiveLeafDecoder`] for fixed-width numeric types.
//! - [`bytes_view::BytesViewDecoder`] for variable-length string / binary types.
//! - [`boolean::BooleanLeafDecoder`] for booleans.

mod absent;
pub use absent::AbsentLeafDecoder;

mod boolean;
pub use boolean::BooleanLeafDecoder;

pub(crate) mod bytes_view;
pub use bytes_view::BytesViewDecoder;

mod decimal;
pub use decimal::decimal_decoder;

pub(crate) mod delta_binary_packed;
pub use delta_binary_packed::{DeltaDecoder, DeltaPacking, FromDelta};

mod levels;

mod primitive;
pub use primitive::{PrimitiveLeafDecoder, TimestampMicrosecondLeafDecoder};

mod rle;

mod typed;
pub use typed::TypedLeafDecoder;

use crate::thrift::general::Encoding;
use crate::thrift::headers::DictionaryPageHeader;
use crate::types::page::DecompressedPage;
use arrow_array::{ArrayRef, RecordBatch, Scalar};
use bytes::Bytes;
use dispatch::memory::{ReaderPosition, SlabAllocator};
use std::convert::Infallible;
use std::marker::PhantomData;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("No pages are ready")]
    NoPagesReady,
    #[error("Dict page empty")]
    DictPageEmpty,
    #[error("Unsupported encoding: {0}")]
    UnsupportedEncoding(Encoding),
    /// A decimal column whose chunk uses a physical storage the decoders do
    /// not read. The column's arrow type itself is supported; footer parsing
    /// only admits the storages the decoders handle, so this can only be
    /// reached by metadata that is inconsistent with the schema.
    #[error(
        "decimal column {data_type} stored as parquet physical type \
         {physical_type} with type_length {type_length:?} is not supported"
    )]
    UnsupportedDecimalStorage {
        data_type: arrow_schema::DataType,
        physical_type: i32,
        type_length: Option<i32>,
    },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// A built dictionary, type-erased so one type serves every column type.
/// Every decoder of a leaf decodes it with the same decoder type, so the
/// decoder that adopts a dictionary downcasts it to its own dictionary
/// type; a mismatch is a bug, not a runtime condition.
pub type SharedDictionary = std::sync::Arc<dyn std::any::Any + Send + Sync>;

/// What building a leaf's dictionary from its page found.
pub enum BuiltDictionary {
    /// The dictionary, for every decoder of the row group to adopt.
    Built(SharedDictionary),
    /// The page excludes the pushed-down equality constant: no row of the
    /// column can match, so the row group is pruned and the dictionary is
    /// never materialized.
    Pruned,
}

/// Object-safe interface for decoding a single Parquet leaf column from
/// decompressed pages into Arrow arrays.
///
/// Used as `Box<dyn LeafDecoder>` inside
/// [`ColumnDecoder`](super::column_decoder::ColumnDecoder) so that leaves of
/// different types can be stored in a single `Vec`.
pub trait LeafDecoder {
    /// Returns `true` if at least `size` rows can be decoded from the pages
    /// buffered so far.
    fn available(&self) -> usize;

    /// Stores a decompressed data or skipped page for later decoding. A
    /// dictionary page goes through [`Self::build_dictionary`] instead.
    fn insert_page(&mut self, page: DecompressedPage);

    /// Builds the row group's dictionary from its page, or finds that the
    /// page excludes the pushed-down equality constant (see
    /// [`Self::set_eq_constant`]) and prunes the row group instead.
    fn build_dictionary(
        &self,
        header: DictionaryPageHeader,
        data: Vec<Bytes>,
        allocator: &mut SlabAllocator,
    ) -> BuiltDictionary;

    /// Reads through a dictionary built for this leaf.
    fn adopt_dictionary(&mut self, dictionary: SharedDictionary);

    /// Forgets every page, the one being read included, and expects the
    /// next pages from `page_idx` on: the decoder is about to read rows
    /// that do not follow the ones it has emitted.
    fn restart_at_page(&mut self, page_idx: usize);

    /// Decodes the next `size` rows into an Arrow array.
    fn read(&mut self, allocator: &mut SlabAllocator, size: usize) -> Result<ArrayRef>;

    /// Installs a pushed-down equality constant. A constant whose type does
    /// not match the column is ignored (the query's `Filter` still applies
    /// the condition). The constant decides row-group pruning when the
    /// dictionary is built and scan-side batch filtering once it is adopted.
    fn set_eq_constant(&mut self, value: &Scalar<ArrayRef>);

    /// Drops rows that cannot pass this column's pushed-down equality
    /// constant from a decoded batch, where `column` is this column's
    /// position in the batch. This only ever removes rows the query would
    /// discard anyway, so the default - returning the batch untouched - is
    /// always valid, and so is any partial filtering. Byte-view columns use
    /// it to pre-filter batches with a cheap view comparison instead of
    /// leaving all the string comparisons to the query's `Filter` (see
    /// [`Dict::filter_record_batch_by_const`]).
    fn fast_filter_record_batch(&self, batch: RecordBatch, _column: usize) -> RecordBatch {
        batch
    }
}

// `ArrayBuilder` (and the primitive builder) now live in `dispatch::arrays` so the
// GROUP BY output can share them; re-exported here for the parquet decoders.
pub use dispatch::arrays::ArrayBuilder;

/// How a leaf column of one type decodes: the builder its values collect in,
/// a decoder for each encoding its data pages may carry, and the dictionary
/// its dictionary page builds.
///
/// The format defines each encoding for some physical types only, so a leaf
/// type names [`Unsupported`] for the encodings it never reads, and a page in
/// one of those reports an unsupported encoding instead of decoding. PLAIN is
/// the one encoding every type reads. A dictionary page's indices decode
/// through the dictionary, in whichever of its two representations the page's
/// buffer shape allowed (see [`DictFromBytes`] and [`DictFromVecBytes`]).
pub trait LeafType {
    type Builder: ArrayBuilder;

    type Plain: DecodePlain<Builder = Self::Builder>;
    type DeltaBinaryPacked: DecodeEncoding<Builder = Self::Builder>;
    type DeltaLengthByteArray: DecodeEncoding<Builder = Self::Builder>;
    type Rle: DecodeEncoding<Builder = Self::Builder>;

    /// The dictionary built from a page held in one contiguous buffer.
    type ContiguousDict: DictFromBytes<Builder = Self::Builder, Item = <Self::Builder as ArrayBuilder>::Element>;
    /// The dictionary built from a page scattered across buffers.
    type ScatteredDict: DictFromVecBytes<
            Builder = Self::Builder,
            Item = <Self::Builder as ArrayBuilder>::Element,
            EqConstant = <Self::ContiguousDict as Dict>::EqConstant,
        >;
}

/// Reads plain-encoded values from raw page bytes into an [`ArrayBuilder`].
pub trait DecodePlain {
    type Builder: ArrayBuilder;

    /// Creates a decoder starting at `position` within `data`.
    fn new(data: Vec<Bytes>, position: ReaderPosition) -> Self;

    /// Decodes `size` values into `builder`.
    fn read(&mut self, builder: &mut Self::Builder, size: usize);

    /// Advances past `size` values without decoding them.
    fn skip(&mut self, size: usize);
}

/// Reads values stored in one of the encodings besides PLAIN and the
/// dictionary from raw page bytes into an [`ArrayBuilder`]. A [`LeafType`]
/// names one implementation for each such encoding it reads.
pub trait DecodeEncoding: Sized {
    type Builder: ArrayBuilder;

    /// Creates a decoder over a page starting at `position`, reading its
    /// header. `None` when the leaf type does not read this encoding (see
    /// [`Unsupported`]) or the header is not one the decoder accepts; either
    /// way the page reports an unsupported encoding.
    fn new(data: Vec<Bytes>, position: ReaderPosition) -> Option<Self>;

    /// Decodes `size` values into `builder`.
    fn read(&mut self, builder: &mut Self::Builder, size: usize);

    /// Advances past `size` values. A delta decoder still decodes them,
    /// because each delta is relative to the one before it.
    fn skip(&mut self, size: usize);
}

/// The decoder a [`LeafType`] names for an encoding it does not read. It has
/// no values, so it is never constructed, and a page in that encoding is
/// reported as unsupported.
pub struct Unsupported<B>(Infallible, PhantomData<fn() -> B>);

impl<B: ArrayBuilder> DecodeEncoding for Unsupported<B> {
    type Builder = B;

    fn new(_data: Vec<Bytes>, _position: ReaderPosition) -> Option<Self> {
        None
    }

    fn read(&mut self, _builder: &mut B, _size: usize) {
        match self.0 {}
    }

    fn skip(&mut self, _size: usize) {
        match self.0 {}
    }
}

/// Queries a dictionary for RLE-dictionary-encoded columns.
///
/// Construction lives in [`DictFromBytes`] / [`DictFromVecBytes`], keyed by
/// whether the dictionary page's bytes sit in one contiguous buffer or are
/// scattered across several.
pub trait Dict: Send + Sync + 'static {
    type Builder: ArrayBuilder;
    type Item;
    /// How a pushed-down equality constant is represented for this
    /// dictionary flavour. Primitive dictionaries take the native value;
    /// byte-view dictionaries take the raw bytes, because the value's Arrow
    /// view can only be resolved against a dictionary that has been built.
    type EqConstant;

    /// Converts a pushed-down constant into this dictionary flavour's own
    /// representation, or `None` when the scalar's type does not match the
    /// column (which simply forgoes the pushdown; the query's `Filter` still
    /// applies the condition).
    fn eq_constant_from_scalar(scalar: &Scalar<ArrayRef>) -> Option<Self::EqConstant>;

    /// Whether `needle` might appear among the first `size` raw entries of
    /// `data`, checked without building the dictionary. Only a definite `no`
    /// matters - it prunes the row group and the dictionary is never
    /// materialized - so answering `true` is always sound. The default cannot
    /// rule anything out; implementations override it to enable the pushdown.
    fn maybe_contains(_data: &[Bytes], _size: usize, _needle: &Self::EqConstant) -> bool {
        true
    }

    /// Number of entries in the dictionary.
    fn len(&self) -> usize;

    /// Looks up the value at `idx` in the dictionary.
    fn entry(&self, idx: usize) -> Self::Item;

    /// Looks up the value at `idx` without validating it.
    ///
    /// # Safety
    ///
    /// `idx` must be `< self.len()`. The default just calls
    /// [`entry`](Self::entry); an implementation whose storage bounds-checks
    /// overrides this so the decode hot loop, whose caller proves every key
    /// in range up front, skips the per-element check.
    unsafe fn entry_unchecked(&self, idx: usize) -> Self::Item {
        self.entry(idx)
    }

    /// Registers dictionary buffers onto the builder (e.g. for StringView
    /// block tracking). No-op by default.
    fn register_onto(&self, _builder: &mut Self::Builder) {}

    /// Drops rows of `batch` whose value in `column` cannot equal `needle`,
    /// or returns the batch untouched when the dictionary cannot decide that
    /// cheaply (the default). Only rows that provably fail the equality may
    /// be dropped; keeping extra rows is always sound because the query's
    /// `Filter` re-applies every condition.
    fn filter_record_batch_by_const(
        &self,
        batch: RecordBatch,
        _column: usize,
        _needle: &Self::EqConstant,
    ) -> RecordBatch {
        batch
    }
}

/// A [`Dict`] buildable from a dictionary page held in one contiguous buffer.
///
/// A contiguous page is at most one 2MB ring slot, so implementations can
/// pick a representation that relies on that bound, e.g. entries in a single
/// [`SlabBuffer`](dispatch::memory::SlabBuffer) indexed by plain pointer
/// arithmetic with no per-element slab lookup.
pub trait DictFromBytes: Dict {
    /// Builds the dictionary from one buffer of raw page bytes containing
    /// `size` entries.
    fn new_from_bytes(data: Bytes, size: usize, allocator: &mut SlabAllocator) -> Self;
}

/// A [`Dict`] buildable from a dictionary page scattered across buffers.
pub trait DictFromVecBytes: Dict {
    /// Builds the dictionary from scattered raw page bytes containing `size`
    /// entries in total.
    fn new_from_vec_bytes(data: Vec<Bytes>, size: usize, allocator: &mut SlabAllocator) -> Self;
}
