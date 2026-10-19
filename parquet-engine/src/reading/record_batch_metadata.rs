use arrow_array::types::Int32Type;
use arrow_array::{Array, Int32Array, RecordBatch, RunArray, UInt32Array};
use arrow_schema::{DataType, Field, Schema};
use std::sync::{Arc, LazyLock};

const GLOBAL_ROW_GROUP_OFFSET_FROM_END: usize = 2;

const ROW_OFFSET_FROM_END: usize = 1;

static GLOBAL_ROW_GROUP_FIELD: LazyLock<Arc<Field>> = LazyLock::new(|| {
    let run_ends = Field::new("run_ends", DataType::Int32, false);
    let values = Field::new("values", DataType::UInt32, true);
    let dt = DataType::RunEndEncoded(Arc::new(run_ends), Arc::new(values));
    Arc::new(Field::new(dispatch::ROW_GROUP_IDX_FIELD, dt, false))
});

static ROW_IDX_FIELD: LazyLock<Arc<Field>> =
    LazyLock::new(|| Arc::new(Field::new(dispatch::ROW_IDX_FIELD, DataType::UInt32, false)));

/// The global row group column for a particular RecordBatch. Note that the global row group column
/// is NOT the row_group within a parquet file, but across
/// the entire table
pub fn global_row_group(batch: &RecordBatch) -> &RunArray<Int32Type> {
    batch
        .column(batch.num_columns() - GLOBAL_ROW_GROUP_OFFSET_FROM_END)
        .as_any()
        .downcast_ref::<RunArray<Int32Type>>()
        .expect("Metadata not set correctly")
}

/// Gets the array for the index of the row within the row group
pub fn row_index(batch: &RecordBatch) -> &UInt32Array {
    batch
        .column(batch.num_columns() - ROW_OFFSET_FROM_END)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .expect("Metadata not set correctly")
}

/// Add row group metadata to the record batch, receiving a new RecordBatch with a column for global
/// row group, and another for index within the group: the batch's rows are
/// the row group's consecutive rows from `offset` on.
///
/// Note that the global row group column is NOT the row_group within a parquet file, but across
/// the entire table
pub fn with_row_group_metadata(batch: RecordBatch, group: usize, offset: usize) -> RecordBatch {
    with_row_group(with_row_index(batch, offset), group)
}

/// [`with_row_group_metadata`] for a batch whose rows are not consecutive in
/// the row group: `positions` is each row's position in it.
pub fn with_row_group_positions(
    batch: RecordBatch,
    group: usize,
    positions: UInt32Array,
) -> RecordBatch {
    with_row_group(with_row_positions(batch, positions), group)
}

/// `batch` with the row index column appended: its rows are the row group's
/// consecutive rows from `offset` on.
pub fn with_row_index(batch: RecordBatch, offset: usize) -> RecordBatch {
    let positions =
        UInt32Array::from_iter_values(offset as u32..offset as u32 + batch.num_rows() as u32);
    with_row_positions(batch, positions)
}

/// `batch` with the row index column appended: `positions` is each row's
/// position in its row group.
pub fn with_row_positions(batch: RecordBatch, positions: UInt32Array) -> RecordBatch {
    let (schema, mut columns, row_count) = batch.into_parts();
    debug_assert_eq!(positions.len(), row_count);
    columns.push(Arc::new(positions));
    let fields: Vec<_> = schema
        .fields()
        .iter()
        .cloned()
        .chain([Arc::clone(&ROW_IDX_FIELD)])
        .collect();
    unsafe { RecordBatch::new_unchecked(Arc::new(Schema::new(fields)), columns, row_count) }
}

/// `batch`, carrying its row index column, with the global row group column
/// added before it: every row is of `group`.
pub fn with_row_group(batch: RecordBatch, group: usize) -> RecordBatch {
    let (schema, mut columns, row_count) = batch.into_parts();

    // We create the row group as a single repeated value for the entire group;
    // a run must end past its start, so a batch with no rows gets no run.
    let (run_ends, values) = if row_count == 0 {
        (
            Int32Array::from(Vec::<i32>::new()),
            UInt32Array::from(Vec::<u32>::new()),
        )
    } else {
        (
            Int32Array::from(vec![row_count as i32]),
            UInt32Array::from(vec![group as u32]),
        )
    };
    let run_array =
        RunArray::<Int32Type>::try_new(&run_ends, &values).expect("Failed to create RunArray");

    let group_column = columns.len() - ROW_OFFSET_FROM_END;
    columns.insert(group_column, Arc::new(run_array));
    let mut fields = schema.fields().to_vec();
    fields.insert(group_column, Arc::clone(&GLOBAL_ROW_GROUP_FIELD));

    unsafe { RecordBatch::new_unchecked(Arc::new(Schema::new(fields)), columns, row_count) }
}
