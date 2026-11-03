//! Detach a batch from the ring by deep-copying its buffers into heap
//! allocations.
//!
//! Workers produce `RecordBatch`es backed by ring buffers (via
//! [`SlabAllocator`](crate::memory::SlabAllocator)): each Arrow `Buffer` is a
//! window into an `Arc<WriteBuffer>`. As long as that `Arc` is alive
//! *anywhere*, the slot stays held — and when the last reference finally drops
//! (possibly on a non-worker thread), `WriteBuffer::drop` runs against
//! whatever `memory_ctx()` that thread happens to have, which is brittle.
//!
//! [`copy_out`] sits at the boundary where a batch leaves "ring land" and
//! enters "anyone can hold it" land. For every column it emits, every buffer
//! is a fresh `Buffer::from_slice_ref` — owned by a plain `Vec<u8>` whose
//! `Drop` is just `dealloc`. The original `Arc<WriteBuffer>`s drop on *this*
//! worker thread when the conversion returns, so the ring slots return to the
//! right pool immediately.
//!
//! Cost is one memcpy per batch's payload. Cheap compared to producing the
//! batch in the first place, and it buys: no `MemoryContext` requirement on
//! the consumer thread, no cross-thread slot bookkeeping, no surprises in
//! `Drop`.

use crate::operations::unary;
use arrow::array::{Array, ArrayData};
use arrow_array::{RecordBatch, RecordBatchOptions, make_array};
use arrow_buffer::{BooleanBuffer, Buffer, NullBuffer};

/// Deep-copy one ring-backed batch into ordinary heap allocations.
pub fn copy_out(batch: &RecordBatch) -> unary::Result<RecordBatch> {
    let schema = batch.schema();
    let columns = batch
        .columns()
        .iter()
        .map(|column| Ok(make_array(copy_to_malloc(&column.to_data())?)))
        .collect::<unary::Result<Vec<_>>>()?;

    // Carry the row count explicitly: a batch with no columns (e.g. a
    // metadata-only / row-count scan) has no column lengths to infer it from,
    // and `RecordBatch::try_new` would reject it.
    let options = RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
    Ok(RecordBatch::try_new_with_options(
        schema, columns, &options,
    )?)
}

/// Recursively copy every `Buffer`, null buffer, and child `ArrayData` into
/// fresh heap allocations (plain `malloc`-backed `Vec<u8>`s), preserving
/// offsets and lengths exactly.
fn copy_to_malloc(data: &ArrayData) -> unary::Result<ArrayData> {
    let buffers: Vec<Buffer> = data
        .buffers()
        .iter()
        .map(|b| Buffer::from(b.as_slice()))
        .collect();

    let nulls = data.nulls().map(|nb| {
        // The validity bits may start at a non-zero bit offset (a sliced
        // array). `sliced()` re-packs exactly the covered range to start at
        // bit 0, so rebuilding with offset 0 is correct.
        let bytes = Buffer::from(nb.inner().sliced().as_slice());
        NullBuffer::new(BooleanBuffer::new(bytes, 0, nb.len()))
    });

    let children = data
        .child_data()
        .iter()
        .map(copy_to_malloc)
        .collect::<unary::Result<Vec<_>>>()?;

    let mut builder = ArrayData::builder(data.data_type().clone())
        .len(data.len())
        .offset(data.offset())
        .buffers(buffers)
        .child_data(children);
    if let Some(nb) = nulls {
        builder = builder.nulls(Some(nb));
    }
    Ok(builder.build()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{
        Array, ArrayRef, Int32Array, Int64Array, ListArray, StringArray, StringViewArray,
    };
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    fn schema(fields: Vec<Field>) -> Arc<Schema> {
        Arc::new(Schema::new(fields))
    }

    // A batch with no columns carries its row count only in metadata, not in
    // column lengths. Copy-out must preserve it; round-tripping through plain
    // `RecordBatch::try_new` would drop the count and fail to rebuild the batch
    // (e.g. a metadata-only / row-count scan feeding `collect`).
    #[test]
    fn zero_column_batch_preserves_row_count() {
        let input = RecordBatch::try_new_with_options(
            schema(vec![]),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(7)),
        )
        .unwrap();

        let out = copy_out(&input).unwrap();

        assert_eq!(out.num_columns(), 0);
        assert_eq!(out.num_rows(), 7);
    }

    #[test]
    fn primitive_batch_round_trips_with_identical_values() {
        let input = RecordBatch::try_new(
            schema(vec![Field::new("v", DataType::Int32, false)]),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])) as ArrayRef],
        )
        .unwrap();

        let out = copy_out(&input).unwrap();

        assert_eq!(out, input);
    }

    #[test]
    fn copied_buffer_does_not_alias_the_original() {
        let original = Int32Array::from(vec![10, 20, 30]);
        let original_ptr = original.to_data().buffers()[0].as_ptr();
        let input = RecordBatch::try_new(
            schema(vec![Field::new("v", DataType::Int32, false)]),
            vec![Arc::new(original) as ArrayRef],
        )
        .unwrap();

        let out = copy_out(&input).unwrap();

        // Assert: the emitted batch points at a different allocation.
        let emitted_ptr = out.column(0).to_data().buffers()[0].as_ptr();
        assert_ne!(original_ptr, emitted_ptr);
    }

    #[test]
    fn preserves_nulls() {
        let input = RecordBatch::try_new(
            schema(vec![Field::new("v", DataType::Int64, true)]),
            vec![Arc::new(Int64Array::from(vec![Some(1), None, Some(3), None])) as ArrayRef],
        )
        .unwrap();

        let out = copy_out(&input).unwrap();

        assert_eq!(out, input);
    }

    #[test]
    fn preserves_string_view_with_variadic_buffers() {
        let strings = vec!["short", "another medium long string", "x"];
        let input = RecordBatch::try_new(
            schema(vec![Field::new("s", DataType::Utf8View, false)]),
            vec![Arc::new(StringViewArray::from(strings)) as ArrayRef],
        )
        .unwrap();

        let out = copy_out(&input).unwrap();

        assert_eq!(out, input);
    }

    #[test]
    fn recurses_into_child_data() {
        // a ListArray's values live in `child_data`. If `copy_to_malloc`
        // didn't recurse the output would have empty children and not equal
        // the input.
        let values = Int32Array::from(vec![1, 2, 3, 4, 5, 6]);
        let offsets = arrow_buffer::OffsetBuffer::new(vec![0, 2, 5, 6].into());
        let field = Arc::new(Field::new("item", DataType::Int32, false));
        let list = ListArray::new(field.clone(), offsets, Arc::new(values), None);
        let input = RecordBatch::try_new(
            schema(vec![Field::new("l", DataType::List(field), false)]),
            vec![Arc::new(list) as ArrayRef],
        )
        .unwrap();

        let out = copy_out(&input).unwrap();

        assert_eq!(out, input);
    }

    #[test]
    fn preserves_offset_on_sliced_arrays() {
        // an array with an offset > 0 (a slice). The detach must keep
        // the same offset so values still line up.
        let full = StringArray::from(vec!["a", "b", "c", "d", "e"]);
        let sliced = full.slice(2, 2); // values "c", "d"
        let input = RecordBatch::try_new(
            schema(vec![Field::new("s", DataType::Utf8, false)]),
            vec![Arc::new(sliced) as ArrayRef],
        )
        .unwrap();

        let out = copy_out(&input).unwrap();

        assert_eq!(out, input);
    }

    #[test]
    fn preserves_nulls_on_sliced_arrays() {
        // slicing at a non-byte-aligned offset leaves the validity bits
        // starting mid-byte; the copy must keep them lined up with values.
        let full = Int64Array::from(vec![None, None, Some(2), None, Some(4), Some(5)]);
        let sliced = full.slice(2, 3); // Some(2), None, Some(4)
        let input = RecordBatch::try_new(
            schema(vec![Field::new("v", DataType::Int64, true)]),
            vec![Arc::new(sliced) as ArrayRef],
        )
        .unwrap();

        let out = copy_out(&input).unwrap();

        assert_eq!(out, input);
    }

    #[test]
    fn empty_batch_passes_through() {
        let input = RecordBatch::try_new(
            schema(vec![Field::new("v", DataType::Int32, false)]),
            vec![Arc::new(Int32Array::from(Vec::<i32>::new())) as ArrayRef],
        )
        .unwrap();

        let out = copy_out(&input).unwrap();

        assert_eq!(out, input);
    }
}
