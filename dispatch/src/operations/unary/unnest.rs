//! Unnest of lists with a fixed number of elements: every input row is
//! repeated once per list element, with that element added as a new column.
//!
//! The lists are written out in the query, such as `[a, b]`, so every row's
//! list has the same length and element `k` is computed the same way for every
//! row. The operator therefore never builds the lists. For element `k` it
//! emits the input batch as it is, plus one column per list holding element
//! `k` of each row, which the caller's closure computes for the whole batch.
//! An input batch becomes one output batch per element, and the input's own
//! columns are shared between them, not copied.
//!
//! Several lists are unnested side by side: output row `k` of an input row
//! carries element `k` of each list. The closure pads a list shorter than the
//! longest one with NULLs. With no elements at all, no rows come out.
//!
//! The output rows of one input row are not adjacent, which SQL allows: an
//! unnest promises no row order.

use crate::api::BuildContext;
use crate::operations::channels::Sender;
use crate::operations::unary::{self, Unary, UnaryFactory};
use crate::operations::{Built, BuiltUnary};
use arrow_array::{ArrayRef, RecordBatch, RecordBatchOptions};
use arrow_schema::{Field, Schema, SchemaRef};
use std::sync::Arc;

/// Builds one worker's [`Unnest`] from that worker's element closure.
pub struct UnnestFactory<F> {
    compute_element_columns: F,
    element_count: usize,
}

impl<F> UnnestFactory<F> {
    /// `element_count` is the length of the longest list.
    /// `compute_element_columns` is called with an input batch and an element
    /// position below `element_count`, and returns one column per list holding
    /// that element for every row of the batch.
    pub fn new(compute_element_columns: F, element_count: usize) -> Self {
        Self {
            compute_element_columns,
            element_count,
        }
    }
}

impl<F> UnaryFactory<RecordBatch, RecordBatch> for UnnestFactory<F>
where
    F: FnMut(&RecordBatch, usize) -> unary::Result<Vec<ArrayRef>> + Send + 'static,
{
    type Unary = Unnest<F>;

    fn build_unary(self, _context: &mut BuildContext) -> BuiltUnary<Self::Unary> {
        Built::new(Unnest {
            compute_element_columns: self.compute_element_columns,
            element_count: self.element_count,
            output_schema: None,
        })
    }
}

/// See the [module docs](self).
pub struct Unnest<F> {
    compute_element_columns: F,
    /// How many output rows each input row becomes.
    element_count: usize,
    /// The schema of the last input batch and the output schema built for it.
    /// A stream's batches share one schema, so the output schema is built for
    /// the first batch and reused for the rest.
    output_schema: Option<(SchemaRef, SchemaRef)>,
}

impl<F> Unnest<F> {
    /// The output schema for a batch of `input_schema`: its fields followed by
    /// one field per element column.
    fn output_schema_for(
        &mut self,
        input_schema: &SchemaRef,
        element_columns: &[ArrayRef],
    ) -> SchemaRef {
        if let Some((cached_input, cached_output)) = &self.output_schema
            && Arc::ptr_eq(cached_input, input_schema)
        {
            return cached_output.clone();
        }
        let mut fields = input_schema.fields().to_vec();
        fields.extend(
            element_columns
                .iter()
                .enumerate()
                .map(|(list_idx, column)| {
                    let name = format!("col{}", input_schema.fields().len() + list_idx);
                    Arc::new(Field::new(name, column.data_type().clone(), true))
                }),
        );
        let output_schema = Arc::new(Schema::new(fields));
        self.output_schema = Some((input_schema.clone(), output_schema.clone()));
        output_schema
    }
}

impl<F> Unary<RecordBatch, RecordBatch> for Unnest<F>
where
    F: FnMut(&RecordBatch, usize) -> unary::Result<Vec<ArrayRef>> + Send,
{
    fn consume(
        &mut self,
        batch: RecordBatch,
        sender: &mut dyn Sender<RecordBatch>,
        _io: &mut crate::io::OperatorIO,
    ) -> unary::Result<()> {
        let row_count = batch.num_rows();
        // The row count is stated because the input may have no columns.
        let options = RecordBatchOptions::new().with_row_count(Some(row_count));

        for element_idx in 0..self.element_count {
            let element_columns = (self.compute_element_columns)(&batch, element_idx)?;
            let output_schema = self.output_schema_for(batch.schema_ref(), &element_columns);
            let mut columns = batch.columns().to_vec();
            columns.extend(element_columns);
            sender.send(RecordBatch::try_new_with_options(
                output_schema,
                columns,
                &options,
            )?)?;
        }
        Ok(())
    }
}
