//! [`StructExtract`] — reads one field of a struct value.

use super::{Expression, Function};
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use crate::types::Type;
use arrow::compute::nullif;
use arrow_array::cast::AsArray;
use arrow_array::{Array, ArrayRef, BooleanArray, Datum, RecordBatch, Scalar};
use std::fmt::{self, Display};

/// One field of a struct: `pair.label`, or `pair[1]` for a field read by
/// position. DuckDB binds both to a field position, which is what this holds.
#[derive(Debug, Clone)]
pub struct StructExtract {
    pub input: Box<Expression>,
    /// The position of the field in the struct type's fields, from 0.
    pub field_idx: usize,
    /// The field's type.
    pub return_type: Type,
}

impl Display for StructExtract {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "struct_extract({}, {})", self.input, self.field_idx)
    }
}

impl StructExtract {
    /// Whether the field read can be NULL, given the nullability of each input
    /// column. A column's nullability says nothing about the fields inside a
    /// struct column, so the field is only known not to be NULL when the
    /// struct is built or written out right here.
    pub fn nullability(&self, input: &[bool]) -> bool {
        match self.input.as_ref() {
            Expression::Function(Function::StructPack(packed)) => {
                packed.fields[self.field_idx].nullability(input)
            }
            Expression::Constant(one_struct) => {
                let (one_struct, _) = one_struct.get();
                one_struct.is_null(0) || one_struct.as_struct().column(self.field_idx).is_null(0)
            }
            _ => true,
        }
    }

    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        let input_builder = self.input.compile()?;
        let field_idx = self.field_idx;
        Ok(Box::new(move || {
            let mut input_eval = input_builder();
            Box::new(move |batch: &RecordBatch| match input_eval(batch) {
                ExprResult::Array(structs) => ExprResult::Array(read_field(&structs, field_idx)),
                ExprResult::Scalar(one_struct) => {
                    let field = read_field(&one_struct.into_inner(), field_idx);
                    ExprResult::Scalar(Scalar::new(field))
                }
            }) as ExprEvalFn
        }))
    }
}

/// The child column holding field `field_idx` of every struct in `structs`.
/// The column is shared with the struct, not copied, unless the struct column
/// has NULL rows of its own: a field of a NULL struct is NULL, so those rows
/// are then marked NULL in the field column as well.
fn read_field(structs: &ArrayRef, field_idx: usize) -> ArrayRef {
    let field_column = structs.as_struct().column(field_idx);
    let struct_nulls = match structs.nulls() {
        Some(struct_nulls) if struct_nulls.null_count() > 0 => struct_nulls,
        _ => return field_column.clone(),
    };
    let struct_is_null = BooleanArray::new(!struct_nulls.inner(), None);
    nullif(field_column, &struct_is_null).expect("a struct and its field have the same row count")
}
