//! Accumulates rows of many batches into full-size batches.
//!
//! Two callers share one mechanism. A selective operator that forwarded each
//! input batch's survivors directly would emit a runt batch per input batch, and
//! every downstream operator would pay its per-batch costs at that rate. The
//! write path has the mirror-image problem, reshaping `RECORD_BATCH_SIZE` inputs
//! into row groups of a hundred thousand rows. Both append rows as they arrive
//! and take a batch out once enough of them accumulated.
//!
//! A column's values land in slab buffers wherever its type allows, which is the
//! same ring-owned memory the rest of the dataflow runs on, so emitting costs no
//! general-heap round trip and each slab returns to the pool when the downstream
//! consumer drops the batch. How that is done depends on the type, so each
//! strategy is a [`ColumnAccumulator`] of its own: [`fixed_width`] copies raw
//! values, [`view`] gathers the view structs, and [`structs`] accumulates a
//! struct's children. A type none of them covers falls to [`concatenated`],
//! which holds slices of the input and concatenates them through Arrow's own
//! allocator, so those columns are the exception to both the slab memory above
//! and the ownership below.
//!
//! Whether an emitted batch keeps the batches its rows came from alive is decided
//! per append, by the column whose values can live outside its own array, which
//! today means the byte-view columns: a source an append takes a large enough
//! share of is referenced, a source it takes a few rows from has those rows'
//! values copied (see [`view`]). A fixed-width value is copied either way, and a
//! column on the [`concatenated`] path holds slices of its input, so a schema
//! containing one keeps those batches alive regardless.

mod column;
mod concatenated;
mod fixed_width;
mod structs;
mod validity;
mod view;

pub(in crate::arrays) use view::{DataBlock, INLINE_VIEW_LEN, copy_value};

use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{ArrowError, DataType, SchemaRef};

use crate::RECORD_BATCH_SIZE;
use crate::memory::{BUFFER_SIZE, SlabAllocator};
use column::ColumnAccumulator;
pub use column::{ChunkedColumn, EncodedRowIds};
use concatenated::ConcatenatedColumn;
use fixed_width::FixedWidthColumn;
use structs::StructColumn;
use view::ViewColumn;
/// How many rows a coalescing accumulator holds. One full batch to emit, plus
/// room for an input batch that lands on top of an almost-full accumulator.
const COALESCING_CAPACITY: usize = 2 * RECORD_BATCH_SIZE;

/// Accumulates rows of many batches and hands them out as one batch (see the
/// module docs). Rows arrive via
/// [`append_batch_by_indices`](Self::append_batch_by_indices),
/// [`append_from_batches`](Self::append_from_batches), or
/// [`append_range`](Self::append_range); [`take_batch`](Self::take_batch) hands
/// out the accumulated rows and resets.
pub struct BatchAccumulator {
    schema: SchemaRef,
    columns: Vec<Box<dyn ColumnAccumulator>>,
    len: usize,
}

impl BatchAccumulator {
    pub fn new(schema: SchemaRef, allocator: &mut SlabAllocator) -> Self {
        let columns = schema
            .fields()
            .iter()
            .map(|field| {
                create_column_accumulator(field.data_type(), COALESCING_CAPACITY, allocator)
            })
            .collect();
        Self {
            schema,
            columns,
            len: 0,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Rows accumulated so far.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Rows this accumulator can hold before it must be taken.
    pub fn capacity(&self) -> usize {
        COALESCING_CAPACITY
    }

    /// The schema this accumulator coalesces into. A batch of a different schema
    /// cannot be appended (its columns would not concatenate), so a caller
    /// checks this and flushes first.
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// Whether a full batch has accumulated, which is what a coalescing caller
    /// emits on. Not the same as being at [`capacity`](Self::capacity): the
    /// accumulator has room past this point, which is what lets a whole input
    /// batch land on an almost-full one.
    pub fn has_full_batch(&self) -> bool {
        self.len >= RECORD_BATCH_SIZE
    }

    /// Append the rows of `batch` at `indices`, which must be ascending row
    /// positions, in bounds, and no more than the remaining
    /// [`capacity`](Self::capacity). The batch must match the accumulator's
    /// schema.
    ///
    /// Each column copies one row per index instead of detecting and copying
    /// consecutive runs. Selective filters usually produce runs of only one or
    /// two rows, so detecting runs adds branches without reducing copying, while
    /// independent loads let the CPU overlap source cache misses.
    ///
    /// `allocator` is only drawn on by a column that copies values.
    pub fn append_batch_by_indices(
        &mut self,
        batch: &RecordBatch,
        indices: &[u32],
        allocator: &mut SlabAllocator,
    ) {
        assert!(self.len + indices.len() <= COALESCING_CAPACITY);
        for (accumulator, column) in self.columns.iter_mut().zip(batch.columns()) {
            accumulator.append_from_indices(column, indices, self.len, allocator);
        }
        self.len += indices.len();
    }

    /// Append rows stored across many batches at the encoded `ids`
    /// (`batch << shift | row`), which must be no more than the remaining
    /// [`capacity`](Self::capacity). `columns` holds one prepared
    /// [`ChunkedColumn`] per accumulator column; their schema must match the
    /// accumulator's.
    pub fn append_from_batches(
        &mut self,
        columns: &[ChunkedColumn],
        shift: u32,
        ids: EncodedRowIds<'_>,
        allocator: &mut SlabAllocator,
    ) {
        assert!(self.len + ids.len() <= COALESCING_CAPACITY);
        for (accumulator, column) in self.columns.iter_mut().zip(columns) {
            accumulator.append_from_batches(column, ids, shift, self.len, allocator);
        }
        self.len += ids.len();
    }

    /// Append `len` consecutive rows of `batch` from `start`, which must fit in
    /// the remaining [`capacity`](Self::capacity). This is the bulk form of
    /// [`append_batch_by_indices`](Self::append_batch_by_indices), for a caller taking whole
    /// batches. The rows are contiguous in the source as well as the
    /// destination, so each column copies in one run instead of a value at a
    /// time.
    pub fn append_range(
        &mut self,
        batch: &RecordBatch,
        start: usize,
        len: usize,
        allocator: &mut SlabAllocator,
    ) {
        assert!(self.len + len <= COALESCING_CAPACITY);
        for (accumulator, column) in self.columns.iter_mut().zip(batch.columns()) {
            accumulator.append_from_range(column, start, len, self.len, allocator);
        }
        self.len += len;
    }

    /// Append indexed rows of `batch` in room-bounded chunks, handing a full
    /// batch to `flush` whenever one accumulates, so the input's size is
    /// unbounded: the accumulator itself decides where to split, and no
    /// producer upstream has to know this side's capacity.
    pub fn append_batch_by_indices_flushing<E: From<ArrowError>>(
        &mut self,
        batch: &RecordBatch,
        indices: &[u32],
        allocator: &mut SlabAllocator,
        flush: &mut dyn FnMut(RecordBatch) -> Result<(), E>,
    ) -> Result<(), E> {
        let mut remaining = indices;
        while !remaining.is_empty() {
            let room = COALESCING_CAPACITY - self.len;
            let (chunk, rest) = remaining.split_at(room.min(remaining.len()));
            self.append_batch_by_indices(batch, chunk, allocator);
            if self.has_full_batch() {
                let full = self.take_batch(allocator)?;
                flush(full)?;
            }
            remaining = rest;
        }
        Ok(())
    }

    /// Emit the accumulated rows as one batch and reset.
    pub fn take_batch(&mut self, allocator: &mut SlabAllocator) -> Result<RecordBatch, ArrowError> {
        let len = self.len;
        let columns = self
            .columns
            .iter_mut()
            .map(|accumulator| accumulator.take_array(len, allocator))
            .collect::<Result<Vec<ArrayRef>, _>>()?;
        self.len = 0;
        // Row-count options so a column-less accumulation (a join side whose
        // columns all live on the other side) still emits its row count.
        let options = arrow_array::RecordBatchOptions::new().with_row_count(Some(len));
        RecordBatch::try_new_with_options(self.schema.clone(), columns, &options)
    }
}

/// The accumulator a column of this type is built with: raw values where the
/// type has a fixed width, views where it is a byte-view column, children where
/// it is a struct, and slices to concatenate for anything else.
fn create_column_accumulator(
    data_type: &DataType,
    capacity: usize,
    allocator: &mut SlabAllocator,
) -> Box<dyn ColumnAccumulator> {
    match data_type {
        DataType::Utf8View | DataType::BinaryView => {
            Box::new(ViewColumn::new(data_type, capacity, allocator))
        }
        DataType::Struct(fields) => Box::new(StructColumn::new(fields, capacity, allocator)),
        other => match other.primitive_width() {
            Some(width @ (1 | 2 | 4 | 8 | 16)) => {
                Box::new(FixedWidthColumn::new(data_type, width, capacity, allocator))
            }
            _ => Box::new(ConcatenatedColumn::new()),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::init_test_free_pool;
    use arrow::compute::{concat_batches, filter_record_batch};
    use arrow_array::cast::AsArray;
    use arrow_array::{
        BinaryViewArray, BooleanArray, Date32Array, Int64Array, StringViewArray, StructArray,
    };
    use arrow_buffer::{Buffer, NullBuffer};
    use arrow_schema::Fields;
    use arrow_schema::{Field, Schema};
    use std::sync::Arc;

    fn three_column_batch() -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("name", DataType::Utf8View, false),
                Field::new("day", DataType::Date32, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5])),
                Arc::new(StringViewArray::from(vec![
                    "aa",
                    "a longer string that spills out of the view",
                    "cc",
                    "dd",
                    "a second long string kept by the second run",
                ])),
                Arc::new(Date32Array::from(vec![10, 20, 30, 40, 50])),
            ],
        )
        .unwrap()
    }

    fn indices_of(mask: &BooleanArray) -> Vec<u32> {
        mask.values().set_indices_u32().collect()
    }

    #[test]
    fn coalesces_batches() {
        init_test_free_pool(4);
        let batch = three_column_batch();
        let mask = BooleanArray::from(vec![false, true, true, false, true]);
        let mut allocator = SlabAllocator::new(false);
        let mut accumulator = BatchAccumulator::new(batch.schema(), &mut allocator);

        accumulator.append_batch_by_indices(&batch, &indices_of(&mask), &mut allocator);
        let all: Vec<u32> = (0..batch.num_rows() as u32).collect();
        accumulator.append_batch_by_indices(&batch, &all, &mut allocator);
        let emitted = accumulator.take_batch(&mut allocator).unwrap();

        let expected = concat_batches(
            &batch.schema(),
            &[filter_record_batch(&batch, &mask).unwrap(), batch.clone()],
        )
        .unwrap();
        assert_eq!(emitted, expected);
        assert!(accumulator.is_empty());
    }

    #[test]
    fn keeps_null_rows() {
        init_test_free_pool(4);
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, true),
                Field::new("name", DataType::Utf8View, true),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![
                    Some(1),
                    None,
                    Some(3),
                    None,
                    Some(5),
                ])),
                Arc::new(StringViewArray::from(vec![
                    None,
                    Some("a longer string that spills out of the view"),
                    Some("cc"),
                    None,
                    Some("ee"),
                ])),
            ],
        )
        .unwrap();
        let mask = BooleanArray::from(vec![true, true, false, true, true]);
        let mut allocator = SlabAllocator::new(false);
        let mut accumulator = BatchAccumulator::new(batch.schema(), &mut allocator);

        accumulator.append_batch_by_indices(&batch, &indices_of(&mask), &mut allocator);
        let emitted = accumulator.take_batch(&mut allocator).unwrap();

        let expected = filter_record_batch(&batch, &mask).unwrap();
        assert_eq!(emitted, expected);
    }

    /// A `Utf8View` batch whose rows are `(buffer index, value)` pairs, each
    /// value living at offset 0 of the data buffer it names.
    fn view_batch(buffers: Vec<Buffer>, rows: &[(u32, &str)]) -> RecordBatch {
        let views: Vec<u128> = rows
            .iter()
            .map(|(buffer_index, value)| {
                let prefix = u32::from_le_bytes(value.as_bytes()[..4].try_into().unwrap());
                // A view of a value longer than 12 bytes: length, the first
                // four bytes, the data buffer it points into, and last the
                // offset within that buffer, which is 0 for every value here.
                (value.len() as u128) | (prefix as u128) << 32 | (*buffer_index as u128) << 64
            })
            .collect();
        let column = StringViewArray::try_new(views.into(), buffers, None).unwrap();
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "name",
                DataType::Utf8View,
                false,
            )])),
            vec![Arc::new(column)],
        )
        .unwrap()
    }

    /// Two batches can hold the same first data buffer and the same buffer
    /// count while differing in the rest, so a source is only the one already
    /// accumulated when every one of its buffers matches.
    #[test]
    fn appends_sources_that_share_a_first_data_buffer() {
        init_test_free_pool(4);
        let shared = Buffer::from_slice_ref("a value both sources hold".as_bytes());
        let first = view_batch(
            vec![
                shared.clone(),
                Buffer::from_slice_ref("a value only the first source holds".as_bytes()),
            ],
            &[
                (0, "a value both sources hold"),
                (1, "a value only the first source holds"),
            ],
        );
        let second = view_batch(
            vec![
                shared.clone(),
                Buffer::from_slice_ref("a value only the second source holds".as_bytes()),
            ],
            &[
                (0, "a value both sources hold"),
                (1, "a value only the second source holds"),
            ],
        );
        let mut allocator = SlabAllocator::new(false);
        let mut accumulator = BatchAccumulator::new(first.schema(), &mut allocator);

        accumulator.append_batch_by_indices(&first, &[0, 1], &mut allocator);
        accumulator.append_batch_by_indices(&second, &[0, 1], &mut allocator);
        let emitted = accumulator.take_batch(&mut allocator).unwrap();

        let expected = concat_batches(&first.schema(), &[first, second]).unwrap();
        assert_eq!(emitted, expected);
    }

    #[test]
    fn resets_between_emits() {
        init_test_free_pool(4);
        let batch = three_column_batch();
        let mut allocator = SlabAllocator::new(false);
        let mut accumulator = BatchAccumulator::new(batch.schema(), &mut allocator);

        accumulator.append_batch_by_indices(&batch, &[1, 2], &mut allocator);
        let first = accumulator.take_batch(&mut allocator).unwrap();
        accumulator.append_batch_by_indices(&batch, &[4], &mut allocator);
        let second = accumulator.take_batch(&mut allocator).unwrap();

        assert_eq!(first, batch.slice(1, 2));
        assert_eq!(second, batch.slice(4, 1));
    }

    /// A contiguous append takes the same rows as the equivalent selection, so
    /// the bulk path and the gather agree.
    #[test]
    fn appending_a_range_matches_appending_its_positions() {
        init_test_free_pool(4);
        let batch = three_column_batch();
        let mut allocator = SlabAllocator::new(false);
        let mut ranged = BatchAccumulator::new(batch.schema(), &mut allocator);
        let mut selected = BatchAccumulator::new(batch.schema(), &mut allocator);

        ranged.append_range(&batch, 1, 3, &mut allocator);
        selected.append_batch_by_indices(&batch, &[1, 2, 3], &mut allocator);

        assert_eq!(
            ranged.take_batch(&mut allocator).unwrap(),
            selected.take_batch(&mut allocator).unwrap()
        );
    }

    /// The data buffers of `batch`'s view column at `column`, by address.
    fn data_buffer_pointers(batch: &RecordBatch, column: usize) -> Vec<*const u8> {
        batch
            .column(column)
            .as_string_view()
            .data_buffers()
            .iter()
            .map(|buffer| buffer.as_ptr())
            .collect()
    }

    /// Twenty rows of one long value living at the start of `buffers[0]`.
    fn twenty_long_rows(buffers: Vec<Buffer>) -> RecordBatch {
        view_batch(buffers, &[(0, "a value longer than twelve bytes"); 20])
    }

    fn long_value_buffer() -> Buffer {
        Buffer::from_slice_ref("a value longer than twelve bytes".as_bytes())
    }

    /// An append taking a few of a source's rows copies their values, so the
    /// emitted batch names none of the source's data buffers and keeps none of
    /// them alive.
    #[test]
    fn a_selective_append_copies_the_values() {
        init_test_free_pool(8);
        let batch = twenty_long_rows(vec![long_value_buffer()]);
        let mut allocator = SlabAllocator::new(false);
        let mut accumulator = BatchAccumulator::new(batch.schema(), &mut allocator);

        accumulator.append_batch_by_indices(&batch, &[3], &mut allocator);
        let emitted = accumulator.take_batch(&mut allocator).unwrap();

        assert_eq!(emitted, batch.slice(3, 1));
        let source = data_buffer_pointers(&batch, 0);
        assert!(
            data_buffer_pointers(&emitted, 0)
                .iter()
                .all(|buffer| !source.contains(buffer)),
            "a selective append must not hold a source data buffer"
        );
    }

    /// An append taking a large share of a source references its data buffers
    /// instead of copying.
    #[test]
    fn a_broad_append_retains_the_source_buffers() {
        init_test_free_pool(8);
        let batch = twenty_long_rows(vec![long_value_buffer()]);
        let mut allocator = SlabAllocator::new(false);
        let mut accumulator = BatchAccumulator::new(batch.schema(), &mut allocator);

        accumulator.append_batch_by_indices(&batch, &(0..10).collect::<Vec<_>>(), &mut allocator);
        let emitted = accumulator.take_batch(&mut allocator).unwrap();

        assert_eq!(emitted, batch.slice(0, 10));
        assert_eq!(
            data_buffer_pointers(&emitted, 0),
            data_buffer_pointers(&batch, 0)
        );
    }

    /// A selective append from a source whose buffers are all already held
    /// references them again: that keeps nothing extra alive, so there is
    /// nothing to copy.
    #[test]
    fn a_selective_append_from_a_source_already_held_retains_it() {
        init_test_free_pool(8);
        let batch = twenty_long_rows(vec![long_value_buffer()]);
        let mut allocator = SlabAllocator::new(false);
        let mut accumulator = BatchAccumulator::new(batch.schema(), &mut allocator);
        accumulator.append_batch_by_indices(&batch, &(0..10).collect::<Vec<_>>(), &mut allocator);

        accumulator.append_batch_by_indices(&batch.slice(5, 15), &[0], &mut allocator);
        let emitted = accumulator.take_batch(&mut allocator).unwrap();

        assert_eq!(emitted.num_rows(), 11);
        assert_eq!(
            data_buffer_pointers(&emitted, 0),
            data_buffer_pointers(&batch, 0),
            "no block was added for the rebased row"
        );
    }

    /// Holding some of a source's buffers is not holding the source: a
    /// selective append from it still copies, so the buffer it would have added
    /// stays out of the emitted batch.
    #[test]
    fn a_selective_append_copies_when_only_some_buffers_are_held() {
        init_test_free_pool(8);
        let shared = long_value_buffer();
        let held = twenty_long_rows(vec![shared.clone()]);
        let partly_held = twenty_long_rows(vec![shared.clone(), long_value_buffer()]);
        let mut allocator = SlabAllocator::new(false);
        let mut accumulator = BatchAccumulator::new(held.schema(), &mut allocator);
        accumulator.append_batch_by_indices(&held, &(0..10).collect::<Vec<_>>(), &mut allocator);

        accumulator.append_batch_by_indices(&partly_held, &[0], &mut allocator);
        let emitted = accumulator.take_batch(&mut allocator).unwrap();

        let emitted_buffers = data_buffer_pointers(&emitted, 0);
        assert_eq!(
            emitted_buffers.len(),
            2,
            "the held buffer plus one copied block"
        );
        assert!(!emitted_buffers.contains(&data_buffer_pointers(&partly_held, 0)[1]));
        assert_eq!(emitted.num_rows(), 11);
    }

    /// A copied value larger than a slab cannot land in one, so it gets a
    /// buffer of its own rather than a reference to the source's.
    #[test]
    fn a_selective_append_copies_a_value_larger_than_a_slab() {
        init_test_free_pool(8);
        let value = "v".repeat(BUFFER_SIZE + 1024);
        let mut values = vec!["short"; 20];
        values[0] = value.as_str();
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "name",
                DataType::Utf8View,
                false,
            )])),
            vec![Arc::new(StringViewArray::from(values))],
        )
        .unwrap();
        let mut allocator = SlabAllocator::new(false);
        let mut accumulator = BatchAccumulator::new(batch.schema(), &mut allocator);

        accumulator.append_batch_by_indices(&batch, &[0], &mut allocator);
        let emitted = accumulator.take_batch(&mut allocator).unwrap();

        assert_eq!(emitted.column(0).as_string_view().value(0), value);
        let source = data_buffer_pointers(&batch, 0);
        assert!(
            data_buffer_pointers(&emitted, 0)
                .iter()
                .all(|buffer| !source.contains(buffer))
        );
    }

    /// A struct column accumulates through its children rather than falling to
    /// the concatenating path, nulls on the struct itself included. This is the
    /// shape of a variant column.
    #[test]
    fn a_struct_column_accumulates_through_its_children() {
        init_test_free_pool(8);
        let fields = Fields::from(vec![
            Field::new("metadata", DataType::BinaryView, false),
            Field::new("value", DataType::BinaryView, true),
        ]);
        let column = StructArray::new(
            fields.clone(),
            vec![
                Arc::new(BinaryViewArray::from(vec![
                    b"m0".as_slice(),
                    b"m1".as_slice(),
                    b"m2".as_slice(),
                ])),
                Arc::new(BinaryViewArray::from(vec![
                    Some(b"a document body well past twelve bytes".as_slice()),
                    None,
                    Some(b"v2".as_slice()),
                ])),
            ],
            Some(NullBuffer::from(vec![true, false, true])),
        );
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "doc",
                DataType::Struct(fields),
                true,
            )])),
            vec![Arc::new(column)],
        )
        .unwrap();
        let mut allocator = SlabAllocator::new(false);
        let mut accumulator = BatchAccumulator::new(batch.schema(), &mut allocator);

        accumulator.append_range(&batch, 0, 3, &mut allocator);
        let emitted = accumulator.take_batch(&mut allocator).unwrap();

        assert_eq!(emitted, batch);
    }

    /// A one-column batch of `rows` copies of `value`, which must be long
    /// enough to live in a data buffer rather than inline in its view.
    fn long_rows(value: &str, rows: usize) -> RecordBatch {
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "name",
                DataType::Utf8View,
                false,
            )])),
            vec![Arc::new(StringViewArray::from(vec![value; rows]))],
        )
        .unwrap()
    }

    fn shares_no_buffer_with(emitted: &RecordBatch, source: &RecordBatch) -> bool {
        let source = data_buffer_pointers(source, 0);
        data_buffer_pointers(emitted, 0)
            .iter()
            .all(|buffer| !source.contains(buffer))
    }

    #[test]
    fn an_append_at_the_share_threshold_retains() {
        init_test_free_pool(8);
        let batch = long_rows("a value longer than twelve bytes", 100);
        let mut allocator = SlabAllocator::new(false);
        let mut accumulator = BatchAccumulator::new(batch.schema(), &mut allocator);

        accumulator.append_batch_by_indices(&batch, &(0..15).collect::<Vec<_>>(), &mut allocator);
        let emitted = accumulator.take_batch(&mut allocator).unwrap();

        assert_eq!(
            data_buffer_pointers(&emitted, 0),
            data_buffer_pointers(&batch, 0)
        );
    }

    #[test]
    fn an_append_just_below_the_share_threshold_copies() {
        init_test_free_pool(8);
        let batch = long_rows("a value longer than twelve bytes", 100);
        let mut allocator = SlabAllocator::new(false);
        let mut accumulator = BatchAccumulator::new(batch.schema(), &mut allocator);

        accumulator.append_batch_by_indices(&batch, &(0..14).collect::<Vec<_>>(), &mut allocator);
        let emitted = accumulator.take_batch(&mut allocator).unwrap();

        assert!(shares_no_buffer_with(&emitted, &batch));
    }

    #[test]
    fn nulls_survive_a_selective_append() {
        init_test_free_pool(8);
        let mut values = vec![Some("a value longer than twelve bytes"); 20];
        values[3] = None;
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "name",
                DataType::Utf8View,
                true,
            )])),
            vec![Arc::new(StringViewArray::from(values))],
        )
        .unwrap();
        let mut allocator = SlabAllocator::new(false);
        let mut accumulator = BatchAccumulator::new(batch.schema(), &mut allocator);

        accumulator.append_batch_by_indices(&batch, &[3, 4], &mut allocator);
        let emitted = accumulator.take_batch(&mut allocator).unwrap();

        assert_eq!(emitted, batch.slice(3, 2));
        assert!(shares_no_buffer_with(&emitted, &batch));
    }

    /// Copied values past one block's worth open a second block, and the views
    /// written after the rollover name it.
    #[test]
    fn copied_values_open_a_second_block() {
        init_test_free_pool(8);
        let value = "v".repeat(10 * 1024);
        let batch = long_rows(&value, 200);
        let mut allocator = SlabAllocator::new(false);
        let mut accumulator = BatchAccumulator::new(batch.schema(), &mut allocator);

        accumulator.append_batch_by_indices(&batch, &(0..28).collect::<Vec<_>>(), &mut allocator);
        let emitted = accumulator.take_batch(&mut allocator).unwrap();

        assert_eq!(emitted, batch.slice(0, 28));
        assert_eq!(data_buffer_pointers(&emitted, 0).len(), 2);
    }

    #[test]
    fn a_held_source_is_forgotten_after_take_batch() {
        init_test_free_pool(8);
        let batch = long_rows("a value longer than twelve bytes", 20);
        let mut allocator = SlabAllocator::new(false);
        let mut accumulator = BatchAccumulator::new(batch.schema(), &mut allocator);
        accumulator.append_batch_by_indices(&batch, &(0..10).collect::<Vec<_>>(), &mut allocator);
        accumulator.take_batch(&mut allocator).unwrap();

        accumulator.append_batch_by_indices(&batch, &[0], &mut allocator);
        let emitted = accumulator.take_batch(&mut allocator).unwrap();

        assert!(shares_no_buffer_with(&emitted, &batch));
    }

    #[test]
    fn a_selective_range_append_copies() {
        init_test_free_pool(8);
        let batch = long_rows("a value longer than twelve bytes", 20);
        let mut allocator = SlabAllocator::new(false);
        let mut accumulator = BatchAccumulator::new(batch.schema(), &mut allocator);

        accumulator.append_range(&batch, 3, 1, &mut allocator);
        let emitted = accumulator.take_batch(&mut allocator).unwrap();

        assert_eq!(emitted, batch.slice(3, 1));
        assert!(shares_no_buffer_with(&emitted, &batch));
    }

    /// Retained buffers and copied blocks share one list, so a copy between
    /// two retains must leave every view naming the right entry.
    #[test]
    fn values_stay_right_across_retained_and_copied_appends() {
        init_test_free_pool(8);
        let first = long_rows("the first source's long value", 20);
        let second = long_rows("the second source's long value", 20);
        let third = long_rows("the third source's long value", 20);
        let mut allocator = SlabAllocator::new(false);
        let mut accumulator = BatchAccumulator::new(first.schema(), &mut allocator);

        accumulator.append_batch_by_indices(&first, &(0..10).collect::<Vec<_>>(), &mut allocator);
        accumulator.append_batch_by_indices(&second, &[0], &mut allocator);
        accumulator.append_batch_by_indices(&third, &(0..10).collect::<Vec<_>>(), &mut allocator);
        let emitted = accumulator.take_batch(&mut allocator).unwrap();

        let expected = concat_batches(
            &first.schema(),
            &[first.slice(0, 10), second.slice(0, 1), third.slice(0, 10)],
        )
        .unwrap();
        assert_eq!(emitted, expected);
    }

    /// Values copied between retained sources go into one block, rather than a
    /// fresh block each time a retained source came last on the buffer list.
    #[test]
    fn copies_between_retains_share_one_block() {
        init_test_free_pool(8);
        // Half of a source is retained, one row of it is copied.
        let rows_taken = [10, 1, 10, 1, 10, 1];
        let sources: Vec<RecordBatch> = (0..rows_taken.len())
            .map(|i| long_rows(&format!("source {i}'s value, longer than a view"), 20))
            .collect();
        let mut allocator = SlabAllocator::new(false);
        let mut accumulator = BatchAccumulator::new(sources[0].schema(), &mut allocator);

        for (source, &rows) in sources.iter().zip(&rows_taken) {
            let indices: Vec<u32> = (0..rows).collect();
            accumulator.append_batch_by_indices(source, &indices, &mut allocator);
        }
        let emitted = accumulator.take_batch(&mut allocator).unwrap();

        let taken: Vec<RecordBatch> = sources
            .iter()
            .zip(&rows_taken)
            .map(|(source, &rows)| source.slice(0, rows as usize))
            .collect();
        assert_eq!(
            emitted,
            concat_batches(&sources[0].schema(), &taken).unwrap()
        );
        let retained_sources = 3;
        let copy_blocks = 1;
        assert_eq!(
            emitted.column(0).as_string_view().data_buffers().len(),
            retained_sources + copy_blocks
        );
    }

    #[test]
    fn a_struct_child_copies_on_a_selective_append() {
        init_test_free_pool(8);
        let fields = Fields::from(vec![Field::new("value", DataType::BinaryView, false)]);
        let child = BinaryViewArray::from(vec![
            b"a document body well past twelve bytes".as_slice();
            20
        ]);
        let column = StructArray::new(fields.clone(), vec![Arc::new(child)], None);
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "doc",
                DataType::Struct(fields),
                false,
            )])),
            vec![Arc::new(column)],
        )
        .unwrap();
        let mut allocator = SlabAllocator::new(false);
        let mut accumulator = BatchAccumulator::new(batch.schema(), &mut allocator);

        accumulator.append_batch_by_indices(&batch, &[0], &mut allocator);
        let emitted = accumulator.take_batch(&mut allocator).unwrap();

        assert_eq!(emitted, batch.slice(0, 1));
        let child_buffers = |batch: &RecordBatch| -> Vec<*const u8> {
            batch
                .column(0)
                .as_struct()
                .column(0)
                .as_binary_view()
                .data_buffers()
                .iter()
                .map(|buffer| buffer.as_ptr())
                .collect()
        };
        let source = child_buffers(&batch);
        assert!(
            child_buffers(&emitted)
                .iter()
                .all(|buffer| !source.contains(buffer))
        );
    }
}
