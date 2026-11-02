//! One place to reinterpret selected columns of a batch to another arrow type.
//!
//! Used wherever pivot crosses the int/temporal boundary: the scan that
//! introduces `Date32`/`Timestamp`, the late-materialization fetch that does the
//! same for late-read columns, and the aggregate's int-in / temporal-out
//! boundary. The cast between an int and its same-width temporal type is
//! zero-copy (it shares the underlying buffer); only the per-column schema is
//! rebuilt.

use arrow_array::cast::AsArray;
use arrow_array::types::{Decimal64Type, Decimal128Type};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{ArrowError, DataType, Field, Schema};
use std::sync::Arc;

/// Rebuild `batch`, replacing each column `i` whose `target(i, current_type)`
/// returns a different `DataType` with that column reinterpreted to the target.
/// Columns with a `None` target, or whose type already equals the target, pass
/// through untouched (no copy).
pub(crate) fn reinterpret_columns(
    batch: RecordBatch,
    target: impl Fn(usize, &DataType) -> Option<DataType>,
) -> Result<RecordBatch, ArrowError> {
    let schema = batch.schema();
    let mut fields = Vec::with_capacity(batch.num_columns());
    let mut columns = Vec::with_capacity(batch.num_columns());
    for i in 0..batch.num_columns() {
        let col = batch.column(i);
        match target(i, col.data_type()) {
            Some(dt) if col.data_type() != &dt => {
                let field = schema.field(i);
                fields.push(Field::new(field.name(), dt.clone(), field.is_nullable()));
                columns.push(reinterpret_array(col, &dt));
            }
            _ => {
                fields.push(schema.field(i).clone());
                columns.push(col.clone());
            }
        }
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
}

/// Reinterpret one column to `dt` without changing its stored values. A
/// decimal-to-decimal target restamps the precision/scale metadata over the same
/// buffer, because an arrow decimal cast multiplies/divides by powers of ten and
/// would corrupt values that are already at their declared scale. Every other
/// pair goes through the arrow cast, which is zero-copy for the int/temporal
/// reinterpretations this module exists for.
fn reinterpret_array(col: &ArrayRef, dt: &DataType) -> ArrayRef {
    match (col.data_type(), dt) {
        (DataType::Decimal128(_, _), DataType::Decimal128(precision, scale)) => Arc::new(
            col.as_primitive::<Decimal128Type>()
                .clone()
                .with_precision_and_scale(*precision, *scale)
                .expect("declared decimal shape is valid"),
        ),
        (DataType::Decimal64(_, _), DataType::Decimal64(precision, scale)) => Arc::new(
            col.as_primitive::<Decimal64Type>()
                .clone()
                .with_precision_and_scale(*precision, *scale)
                .expect("declared decimal shape is valid"),
        ),
        _ => arrow::compute::cast(col, dt).expect("reinterpret cast"),
    }
}

/// The backing int a temporal arrow column drops to (`Date32 -> Int32`,
/// `Timestamp -> Int64`), or `None` for a non-temporal column. This is the
/// group-by boundary coercion: the only place a temporal column becomes an int,
/// so the row encoder and int reducers can compute on the day/second count.
pub(crate) fn temporal_to_int(data_type: &DataType) -> Option<DataType> {
    match data_type {
        DataType::Date32 => Some(DataType::Int32),
        DataType::Timestamp(_, _) => Some(DataType::Int64),
        _ => None,
    }
}
