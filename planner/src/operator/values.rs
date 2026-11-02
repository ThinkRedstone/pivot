//! Rows produced by a SQL `VALUES` clause.

use std::fmt;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, RecordBatch, RecordBatchOptions};
use arrow_schema::{Field, Schema};
use dispatch::{DataFlowDispatcher, RECORD_BATCH_SIZE, RecordBatchOperatorSpec, values_input};

use crate::compile::{Error, ExprFn};
use crate::expression::Expression;

#[derive(Debug)]
pub struct Values {
    pub rows: Vec<Vec<Expression>>,
}

impl fmt::Display for Values {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Values(rows: {})", self.rows.len())
    }
}

impl Values {
    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        // Compile on the coordinator so unsupported expressions fail before the
        // dataflow launches. The resulting builder closures are Sync and can be
        // shared; each worker creates private evaluators for the row it steals.
        let rows: Vec<Vec<Arc<ExprFn>>> = self
            .rows
            .iter()
            .map(|row| {
                row.iter()
                    .map(|expression| expression.compile().map(Arc::new))
                    .collect()
            })
            .collect::<Result<_, _>>()?;

        // A single empty row every constant expression evaluates against; the
        // columns are the same regardless of its contents. Built once here and
        // reused for every row instead of rebuilt per row.
        let single_row = RecordBatch::try_new_with_options(
            Arc::new(Schema::empty()),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(1)),
        )
        .expect("one empty row is a valid batch");

        // Steal a record batch of rows at a time, not a single row. Every
        // consumer downstream sizes its work by the batch it is handed, and the
        // heaviest of them, shredding a variant column, builds a fresh set of
        // column builders per call. A batch per row makes that per-row too.
        let chunks: Vec<Vec<Vec<Arc<ExprFn>>>> = rows
            .chunks(RECORD_BATCH_SIZE)
            .map(<[Vec<Arc<ExprFn>>]>::to_vec)
            .collect();

        Ok(values_input(dispatcher, chunks)
            .map_each(move |chunk: Vec<Vec<Arc<ExprFn>>>| {
                let column_count = chunk.first().map_or(0, Vec::len);
                let columns: Vec<ArrayRef> = (0..column_count)
                    .map(|column| {
                        let values: Vec<ArrayRef> = chunk
                            .iter()
                            .map(|row| row[column]()(&single_row).into_array(1))
                            .collect();
                        let values: Vec<&dyn Array> = values.iter().map(AsRef::as_ref).collect();
                        arrow::compute::concat(&values)
                    })
                    .collect::<Result<_, _>>()?;
                // Column names carry no meaning here: the insert matches VALUES
                // columns to the table positionally, so leave them empty.
                let fields = columns
                    .iter()
                    .map(|column| Field::new("", column.data_type().clone(), true))
                    .collect::<Vec<_>>();
                Ok(RecordBatch::try_new(
                    Arc::new(Schema::new(fields)),
                    columns,
                )?)
            })
            .record_batches())
    }
}
