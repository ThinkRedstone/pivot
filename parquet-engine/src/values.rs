//! Typed values and file statistics shared by Parquet writers and table formats.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, Datum, RecordBatch, Scalar, StringArray, StringViewArray, UInt32Array,
};
use arrow_schema::{ArrowError, DataType};
use arrow_select::take::take;

pub fn scalar_values_from_row(
    batch: &RecordBatch,
    columns: &[String],
    row: usize,
) -> Result<HashMap<String, Scalar<ArrayRef>>, ArrowError> {
    if row >= batch.num_rows() {
        return Err(ArrowError::InvalidArgumentError(format!(
            "row {row} is outside a batch of {} rows",
            batch.num_rows()
        )));
    }
    let schema = batch.schema();
    columns
        .iter()
        .map(|name| {
            let index = schema.index_of(name)?;
            Ok((name.clone(), pivot_scalar(batch.column(index), row)))
        })
        .collect()
}

pub fn scalar_values_equal(
    left: &HashMap<String, Scalar<ArrayRef>>,
    right: &HashMap<String, Scalar<ArrayRef>>,
) -> bool {
    left.len() == right.len()
        && left.iter().all(|(name, left_value)| {
            right
                .get(name)
                .is_some_and(|right_value| scalar_equal(left_value, right_value) == Some(true))
        })
}

pub fn scalar_equal(left: &Scalar<ArrayRef>, right: &Scalar<ArrayRef>) -> Option<bool> {
    let left_array = left.get().0;
    let right_array = right.get().0;
    if left_array.data_type() != right_array.data_type() {
        return None;
    }
    match (left_array.is_null(0), right_array.is_null(0)) {
        (true, true) => Some(true),
        (true, false) | (false, true) => Some(false),
        (false, false) => arrow_ord::cmp::eq(left as &dyn Datum, right as &dyn Datum)
            .ok()
            .filter(|result| result.is_valid(0))
            .map(|result| result.value(0)),
    }
}

pub fn pivot_scalar(array: &ArrayRef, row: usize) -> Scalar<ArrayRef> {
    let value: ArrayRef = match array.data_type() {
        DataType::Utf8 => {
            let strings = array
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("Utf8 array has StringArray representation");
            Arc::new(StringViewArray::from(vec![
                (!strings.is_null(row)).then(|| strings.value(row)),
            ]))
        }
        DataType::Utf8View => {
            let strings = array
                .as_any()
                .downcast_ref::<StringViewArray>()
                .expect("Utf8View array has StringViewArray representation");
            Arc::new(StringViewArray::from(vec![
                (!strings.is_null(row)).then(|| strings.value(row)),
            ]))
        }
        _ => take(array, &UInt32Array::from(vec![row as u32]), None)
            .expect("one-row take supports every physical column type"),
    };
    Scalar::new(value)
}

/// A file's partition tuple: column name to the typed value carried by every row.
pub type PartitionValues = HashMap<String, Scalar<ArrayRef>>;

/// Parquet statistics aggregated over every row group in one file.
#[derive(Debug, Clone)]
pub struct FileStats {
    pub num_records: Option<i64>,
    pub min_values: HashMap<String, ArrayRef>,
    pub max_values: HashMap<String, ArrayRef>,
    pub null_counts: HashMap<String, i64>,
}
