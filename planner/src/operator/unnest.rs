//! [`Unnest`] — repeats each input row once per element of a list.
//!
//! Lowered from an `unnest(...)` in a select list, such as
//! `SELECT unnest([a, b]) AS value, n FROM t`. The list has to be written out
//! in the query, so that it has a known number of elements and each element is
//! an expression over the row. A list read from a column is rejected when the
//! plan is built.

use crate::compile::{Error, ExprEvalFn, ExprFn};
use crate::expression::Expression;
use crate::types::{Type, physical_arrow_type};
use arrow_array::{ArrayRef, RecordBatch, new_null_array};
use arrow_schema::DataType;
use dispatch::RecordBatchOperatorSpec;
use std::fmt;
use std::sync::Arc;

/// Emits the input's columns followed by one element column per list. Each
/// input row comes out once per element of the longest list.
#[derive(Debug)]
pub struct Unnest {
    /// The lists unnested side by side, in output column order. Output row
    /// `k` of an input row carries element `k` of each list, and a list
    /// shorter than the longest one gives NULL past its end.
    pub lists: Vec<UnnestedList>,
}

/// A list written out in the query as one expression per element.
#[derive(Debug)]
pub struct UnnestedList {
    pub element_type: Type,
    /// The elements in list order. Empty for an empty or a NULL list.
    pub elements: Vec<Expression>,
}

impl Unnest {
    /// How many output rows each input row becomes.
    fn longest_list_len(&self) -> usize {
        self.lists
            .iter()
            .map(|list| list.elements.len())
            .max()
            .unwrap_or(0)
    }

    /// The type of each added element column, in output order.
    pub(crate) fn element_types(&self) -> Vec<Type> {
        self.lists
            .iter()
            .map(|list| list.element_type.clone())
            .collect()
    }

    /// Whether each added element column can hold NULL, given the nullability
    /// of each input column: when one of the list's elements can be NULL, or
    /// the list is padded because another list is longer.
    pub(crate) fn element_nullability(&self, input: &[bool]) -> Vec<bool> {
        let longest_list_len = self.longest_list_len();
        self.lists
            .iter()
            .map(|list| {
                list.elements.len() < longest_list_len
                    || list
                        .elements
                        .iter()
                        .any(|element| element.nullability(input))
            })
            .collect()
    }

    pub(crate) fn compile(
        &self,
        input: RecordBatchOperatorSpec,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let element_builders: Arc<Vec<Vec<ExprFn>>> = Arc::new(
            self.lists
                .iter()
                .map(|list| list.elements.iter().map(Expression::compile).collect())
                .collect::<Result<_, _>>()?,
        );
        let element_arrow_types: Arc<Vec<DataType>> = Arc::new(
            self.lists
                .iter()
                .map(|list| physical_arrow_type(&list.element_type))
                .collect(),
        );

        Ok(input.unnest(self.longest_list_len(), move || {
            let mut element_evals: Vec<Vec<ExprEvalFn>> = element_builders
                .iter()
                .map(|list| list.iter().map(|builder| builder()).collect())
                .collect();
            let element_arrow_types = element_arrow_types.clone();
            move |batch: &RecordBatch, element_idx: usize| {
                let row_count = batch.num_rows();
                let element_columns: Vec<ArrayRef> = element_evals
                    .iter_mut()
                    .zip(element_arrow_types.iter())
                    .map(|(list, arrow_type)| match list.get_mut(element_idx) {
                        Some(eval) => eval(batch).into_array(row_count),
                        // Past the end of a list shorter than the longest one.
                        None => new_null_array(arrow_type, row_count),
                    })
                    .collect();
                Ok(element_columns)
            }
        }))
    }
}

impl fmt::Display for Unnest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let lists = self
            .lists
            .iter()
            .map(|list| list.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "Unnest({lists})")
    }
}

impl fmt::Display for UnnestedList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let elements = self
            .elements
            .iter()
            .map(|element| element.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "[{elements}]:{}", self.element_type)
    }
}
