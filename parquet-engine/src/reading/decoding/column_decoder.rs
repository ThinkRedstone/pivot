//! Turns a row group's decoded Parquet leaves into one projected output column.
//!
//! A [`ColumnDecoder`] names the leaves this output folds, the field they fold
//! into, and the transform that finishes the folded array. A plain column folds
//! all of its leaves and needs no transform. A pushed-down variant extract
//! folds only the leaves its path needs, and its transform casts the typed leaf
//! or pulls the path out of the reconstructed variant.
//!
//! The leaves themselves are decoded once per row group and shared, because
//! several extracts on one variant column routinely want the same ones.

use crate::reading::decoding::leaf_decoders;
use crate::reading::decoding::leaf_decoders::{
    AbsentLeafDecoder, BooleanLeafDecoder, BytesViewDecoder, LeafDecoder, PrimitiveLeafDecoder,
    TimestampMicrosecondLeafDecoder, decimal_decoder,
};
use crate::types::leaves::{OutputRead, reconstruct_column_from_leaves};
use crate::types::metadata::ColumnChunkMeta;
use arrow_array::types::{
    BinaryViewType, Date32Type, Decimal64Type, Decimal128Type, Float32Type, Float64Type, Int8Type,
    Int16Type, Int32Type, Int64Type, StringViewType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow_array::{ArrayRef, StructArray, new_null_array};
use arrow_buffer::NullBuffer;
use arrow_schema::{ArrowError, DataType, Field, FieldRef, Fields, TimeUnit};
use dispatch::VariantExtract;
use parquet_variant::{VariantPath, VariantPathElement};
use parquet_variant_compute::cast_to_variant;
use planner::expression::cast_variant_array;
use std::sync::Arc;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Unsupported column type: {0:?}")]
    UnsupportedColumnType(DataType),
    #[error("{0}")]
    LeafDecoder(#[from] leaf_decoders::Error),
    #[error("{0}")]
    Arrow(#[from] ArrowError),
    #[error("leaf {leaf} has dictionary encoded pages but no dictionary")]
    MissingDictionary { leaf: usize },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Finishes a pushed-down variant extract after its leaves are decoded.
///
/// Plain output columns do not need a transform.
enum OutputTransform {
    /// Applies a SQL variant cast to a directly decoded typed leaf.
    CastTypedVariant(DataType),
    /// Extracts a path from a reconstructed variant.
    ///
    /// A requested `as_type` produces a scalar. Without one, this produces the
    /// sub-variant at the path.
    Extract {
        path: Arc<[String]>,
        as_type: Option<DataType>,
    },
    /// Replaces the folded metadata leaf with the all-NULL output of an
    /// extract whose path the row group proves absent. The leaf supplies the
    /// row count, and the metadata blob when the output is itself a variant.
    AbsentPath { as_type: Option<DataType> },
}

impl OutputTransform {
    fn apply(&self, column: &ArrayRef) -> Result<ArrayRef> {
        match self {
            OutputTransform::CastTypedVariant(as_type) => {
                let variant: ArrayRef = Arc::new(cast_to_variant(column.as_ref())?.into_inner());
                Ok(cast_variant_array(&variant, as_type)?)
            }
            OutputTransform::Extract { path, as_type } => {
                Ok(extract_variant_path(column, path, as_type)?)
            }
            OutputTransform::AbsentPath { as_type } => Ok(match as_type {
                Some(as_type) => new_null_array(as_type, column.len()),
                None => all_null_variant(column),
            }),
        }
    }
}

/// The canonical unshredded layout an all-NULL variant column uses.
fn unshredded_variant_fields() -> Fields {
    Fields::from(vec![
        Field::new("metadata", DataType::BinaryView, false),
        Field::new("value", DataType::BinaryView, true),
    ])
}

/// An all-NULL variant column over the decoded `metadata` blobs.
fn all_null_variant(metadata: &ArrayRef) -> ArrayRef {
    let value = new_null_array(&DataType::BinaryView, metadata.len());
    let nulls = NullBuffer::new_null(metadata.len());
    Arc::new(StructArray::new(
        unshredded_variant_fields(),
        vec![metadata.clone(), value],
        Some(nulls),
    ))
}

/// Extracts `path` from a reconstructed variant column.
///
/// A requested `as_type` produces a scalar. Without one, this returns the
/// sub-variant at the path.
fn extract_variant_path(
    column: &ArrayRef,
    path: &[String],
    as_type: &Option<DataType>,
) -> Result<ArrayRef, ArrowError> {
    let variant_path: VariantPath<'_> = path
        .iter()
        .map(|segment| VariantPathElement::field(segment.as_str()))
        .collect();
    planner::expression::extract_variant_path(column, variant_path, as_type.as_ref())
}

/// Decodes one output column of a row group's batches.
///
/// The leaf column chunks themselves are decoded once by the
/// [`RowGroupDecoder`](super::row_group_decoder::RowGroupDecoder) and shared:
/// this is the view that folds the subset of them this output needs and
/// finishes the result.
pub struct ColumnDecoder {
    /// The positions within the row group's decoded leaves that this output
    /// folds, in Parquet's depth-first order.
    leaf_positions: Vec<usize>,
    /// The folded leaves land under this field. It is the output field for a
    /// plain column, and for an extract it is whatever the read leaves
    /// reconstruct: the typed leaf, a pruned variant, or the whole variant.
    pre_transform_field: FieldRef,
    /// Finishes the folded array. `None` for a plain column.
    transform: Option<OutputTransform>,
    /// The field this column contributes to the batch schema.
    output_field: FieldRef,
}

impl ColumnDecoder {
    /// Builds the view for an output column that resolved to `read`.
    ///
    /// A plain column folds its leaves and is done. A scalar extract that
    /// reached a complete shredded typed leaf reads that leaf and casts it when
    /// the leaf's own type is not what the extract asks for. Every other
    /// extract reconstructs a variant, whether the pruned path subtree or the
    /// whole column, and reads the path out of it.
    pub fn new(
        column: usize,
        extract: Option<&VariantExtract>,
        read: &OutputRead,
        leaf_positions: Vec<usize>,
        leaf_fields: &[FieldRef],
        fields: &Fields,
    ) -> Result<Self> {
        let column_name = fields[column].name();
        let create_output_field =
            |data_type: DataType| Arc::new(Field::new(column_name, data_type, true));
        let (pre_transform_field, transform, output_field) = match (extract, read) {
            (None, _) => (fields[column].clone(), None, fields[column].clone()),
            (Some(extract), OutputRead::TypedLeaf(leaf)) => {
                let as_type = extract
                    .as_type
                    .as_ref()
                    .expect("only a scalar extract resolves to a typed leaf");
                let leaf_type = leaf_fields[*leaf].data_type();
                (
                    create_output_field(leaf_type.clone()),
                    (leaf_type != as_type)
                        .then(|| OutputTransform::CastTypedVariant(as_type.clone())),
                    create_output_field(as_type.clone()),
                )
            }
            (
                Some(extract),
                OutputRead::PrunedVariant {
                    reconstructed_variant_field,
                    ..
                },
            ) => (
                reconstructed_variant_field.clone(),
                Some(OutputTransform::Extract {
                    path: extract.path.clone().into(),
                    as_type: extract.as_type.clone(),
                }),
                create_output_field(match &extract.as_type {
                    Some(as_type) => as_type.clone(),
                    None => variant_path_output_type(reconstructed_variant_field, &extract.path)?,
                }),
            ),
            (Some(extract), OutputRead::AbsentPath { metadata_leaf }) => {
                let output_type = match &extract.as_type {
                    Some(as_type) => as_type.clone(),
                    None => DataType::Struct(unshredded_variant_fields()),
                };
                (
                    leaf_fields[*metadata_leaf].clone(),
                    Some(OutputTransform::AbsentPath {
                        as_type: extract.as_type.clone(),
                    }),
                    create_output_field(output_type),
                )
            }
            (Some(extract), OutputRead::WholeColumn(_)) => {
                let whole_variant_field = fields[column].clone();
                let output_type = match &extract.as_type {
                    Some(as_type) => as_type.clone(),
                    None => variant_path_output_type(&whole_variant_field, &extract.path)?,
                };
                (
                    whole_variant_field,
                    Some(OutputTransform::Extract {
                        path: extract.path.clone().into(),
                        as_type: extract.as_type.clone(),
                    }),
                    create_output_field(output_type),
                )
            }
        };
        Ok(Self {
            leaf_positions,
            pre_transform_field,
            transform,
            output_field,
        })
    }

    /// Returns the field this column contributes to the batch schema.
    pub fn output_field(&self) -> &FieldRef {
        &self.output_field
    }

    /// The single decoded leaf this column emits unchanged, if that is what it
    /// is: the only shape a pushed-down equality constant can be installed on.
    ///
    /// Both uses of such a constant read that leaf directly. The row group is
    /// pruned from the leaf's own dictionary, and the batch filter compares the
    /// emitted array against the dictionary's view of the constant. A column
    /// that casts its leaf or reconstructs a variant emits something else
    /// entirely, so it can use neither.
    pub fn untransformed_leaf(&self) -> Option<usize> {
        match (self.transform.is_some(), self.leaf_positions.as_slice()) {
            (false, [position]) => Some(*position),
            _ => None,
        }
    }

    /// Folds this column's share of the row group's `decoded` leaves into its
    /// output array.
    pub fn read(&self, decoded: &[ArrayRef]) -> Result<ArrayRef> {
        let mut leaf_arrays = self
            .leaf_positions
            .iter()
            .map(|&position| decoded[position].clone());
        let column = reconstruct_column_from_leaves(&self.pre_transform_field, &mut leaf_arrays);
        match &self.transform {
            Some(transform) => transform.apply(&column),
            None => Ok(column),
        }
    }
}

/// The type `variant_get` yields for `path` over this variant shape.
///
/// Resolving it once from an empty array fixes the row group's output schema
/// before any batch is decoded.
fn variant_path_output_type(variant_field: &FieldRef, path: &[String]) -> Result<DataType> {
    let empty_variant = arrow_array::new_empty_array(variant_field.data_type());
    Ok(extract_variant_path(&empty_variant, path, &None)?
        .data_type()
        .clone())
}

/// Creates a leaf decoder for `data_type`.
///
/// The chunk metadata disambiguates a decimal's physical storage.
pub fn create_leaf_decoder(
    data_type: &DataType,
    chunk: &ColumnChunkMeta,
) -> Result<Box<dyn LeafDecoder>> {
    if chunk.absent {
        return Ok(Box::new(AbsentLeafDecoder::new(data_type.clone())));
    }
    let max_def_level = chunk.max_def_level;
    macro_rules! primitive {
        ($t:ty) => {
            Box::new(PrimitiveLeafDecoder::<$t>::new(max_def_level)) as Box<dyn LeafDecoder>
        };
    }
    match data_type {
        DataType::Boolean => Ok(Box::new(BooleanLeafDecoder::new(max_def_level))),
        DataType::UInt8 => Ok(primitive!(UInt8Type)),
        DataType::UInt16 => Ok(primitive!(UInt16Type)),
        DataType::UInt32 => Ok(primitive!(UInt32Type)),
        DataType::UInt64 => Ok(primitive!(UInt64Type)),
        DataType::Int8 => Ok(primitive!(Int8Type)),
        DataType::Int16 => Ok(primitive!(Int16Type)),
        DataType::Int32 => Ok(primitive!(Int32Type)),
        DataType::Int64 => Ok(primitive!(Int64Type)),
        DataType::Date32 => Ok(primitive!(Date32Type)),
        DataType::Timestamp(TimeUnit::Microsecond, timezone) => Ok(Box::new(
            TimestampMicrosecondLeafDecoder::new(max_def_level, timezone.clone()),
        )),
        DataType::Float32 => Ok(primitive!(Float32Type)),
        DataType::Float64 => Ok(primitive!(Float64Type)),
        DataType::Decimal64(precision, scale) => {
            Ok(decimal_decoder::<Decimal64Type>(chunk, *precision, *scale)?)
        }
        DataType::Decimal128(precision, scale) => Ok(decimal_decoder::<Decimal128Type>(
            chunk, *precision, *scale,
        )?),
        // The byte-view decoder matches the leaf's declared string or binary
        // type, so no finishing conversion is needed.
        DataType::Utf8View | DataType::Utf8 | DataType::LargeUtf8 => Ok(Box::new(
            BytesViewDecoder::<StringViewType>::new(max_def_level),
        )),
        DataType::BinaryView | DataType::Binary | DataType::LargeBinary => Ok(Box::new(
            BytesViewDecoder::<BinaryViewType>::new(max_def_level),
        )),
        other => Err(Error::UnsupportedColumnType(other.clone())),
    }
}
