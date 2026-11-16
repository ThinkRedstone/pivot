//! [`StructPack`] — builds a struct value from one expression per field.

use super::Expression;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use crate::types::{Type, physical_arrow_type};
use arrow_array::{ArrayRef, RecordBatch, StructArray};
use arrow_schema::DataType;
use std::fmt::{self, Display};
use std::sync::Arc;

/// A struct built from its field values: `{'label': name, 'amount': n}`
/// (DuckDB's `struct_pack`) or the unnamed `(name, n)` (DuckDB's `row`). The
/// field names live in the struct's type, so both forms are the same
/// expression.
#[derive(Debug, Clone)]
pub struct StructPack {
    /// The value of each field, in the order of the struct type's fields.
    pub fields: Vec<Expression>,
    /// The [`Type::Struct`] DuckDB bound for the call.
    pub return_type: Type,
}

impl Display for StructPack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let fields: Vec<String> = self.fields.iter().map(|field| field.to_string()).collect();
        write!(f, "struct_pack({})", fields.join(", "))
    }
}

impl StructPack {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        let DataType::Struct(arrow_fields) = physical_arrow_type(&self.return_type) else {
            return Err(compile::Error::UnsupportedExpression(Expression::Function(
                super::Function::StructPack(self.clone()),
            )));
        };
        let field_builders: Arc<Vec<ExprFn>> = Arc::new(
            self.fields
                .iter()
                .map(|field| field.compile())
                .collect::<Result<Vec<_>, _>>()?,
        );
        Ok(Box::new(move || {
            let mut field_evals: Vec<ExprEvalFn> =
                field_builders.iter().map(|builder| builder()).collect();
            let arrow_fields = arrow_fields.clone();
            Box::new(move |batch: &RecordBatch| {
                // The struct's child columns are the field values themselves, so
                // a field that is a plain column is shared, not copied.
                let row_count = batch.num_rows();
                let field_columns: Vec<ArrayRef> = field_evals
                    .iter_mut()
                    .map(|eval| eval(batch).into_array(row_count))
                    .collect();
                let packed = StructArray::try_new_with_length(
                    arrow_fields.clone(),
                    field_columns,
                    None,
                    row_count,
                )
                .expect("each field evaluates to the type the struct declares for it");
                ExprResult::Array(Arc::new(packed) as ArrayRef)
            }) as ExprEvalFn
        }))
    }
}
