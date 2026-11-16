//! Runtime aggregation container.
//!
//! [`Dynamic`] handles signatures whose number of aggregation slots is known
//! while building a query, but not encoded in a Rust type.
//!
//! A table entry stores the accumulator cells inline:
//!
//! ```text
//! hash | key | cell 0 | cell 1 | ... | cell n
//! ```
//!
//! The shared query context supplies `n`; it is not repeated in every entry.
//! `A` selects `i64` or `i128` accumulator storage. `ONLY_ADDITIVE` removes the
//! per-slot operation dispatch when every slot merges with addition.
//!
//! Scatter rows use the same inline cell layout. [`Dynamic`] is the stored
//! cell run itself (an unsized slice newtype); [`OwnedDynamic`] is the sized
//! handle top-k rows hold, pointing at a copy in the value arena.
//!
//! `S` is the [`SeenMask`] storage. A tracking (`u8`) instantiation appends one
//! extra cell to every stored run holding the per-slot seen bits (an `A`-width
//! cell, so up to 64 slots track), and its fold paths check row validity via
//! the reader's side nulls array. The untracked (`()`) instantiation stores and
//! executes exactly the mask-less layout and loops.

use super::super::cell::{F64Cell, IntCell, StringCell, WideCell};
use super::super::{
    AggregationSlot, AggregationValue, ArityBody, SharedContext, ValueColumnBuilder,
};
use super::SeenMask;
use arrow_buffer::{BooleanBuffer, NullBuffer};
mod operations;
mod readers;

use super::super::AggregationKind;
use super::super::fold::{Count, F64Max, F64Min, F64Sum, Fold, Max, Min, StrMax, StrMin, Sum};
use crate::arrays::SlabColumn;
use crate::memory::SlabAllocator;
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::DataType;
use arrow_schema::Field;
use operations::{MergeCells, OperationReader, OperationsReader};
use std::sync::Arc;

/// One group's dynamic aggregation value: the cell run itself, one cell per
/// slot, living inline in a hash entry or scatter row. Unsized; the slot
/// count comes from the storage metadata.
#[repr(transparent)]
pub struct Dynamic<
    A: IntCell + StringCell + F64Cell + WideCell = i64,
    const ONLY_ADDITIVE: bool = false,
    S: SeenMask = (),
>(std::marker::PhantomData<S>, [A]);

/// Owned handle to a dynamic aggregation value copied into the value arena.
///
/// Exists only for top-k heap rows, which must outlive their source table.
pub struct OwnedDynamic<
    A: IntCell + StringCell + F64Cell + WideCell = i64,
    const ONLY_ADDITIVE: bool = false,
> {
    /// Arena allocation containing one cell per slot. The default value is
    /// null and is never read.
    cells: *mut A,
}

// SAFETY: every non-null pointer refers to an independent allocation in the
// shared arena. The arena outlives every handle, and normal table handoffs
// provide synchronization between writers and readers.
unsafe impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool> Send
    for OwnedDynamic<A, ONLY_ADDITIVE>
{
}
unsafe impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool> Sync
    for OwnedDynamic<A, ONLY_ADDITIVE>
{
}

impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool> Copy
    for OwnedDynamic<A, ONLY_ADDITIVE>
{
}
impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool> Clone
    for OwnedDynamic<A, ONLY_ADDITIVE>
{
    fn clone(&self) -> Self {
        *self
    }
}
impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool> Default
    for OwnedDynamic<A, ONLY_ADDITIVE>
{
    fn default() -> Self {
        Self {
            cells: std::ptr::null_mut(),
        }
    }
}

impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool>
    OwnedDynamic<A, ONLY_ADDITIVE>
{
    /// Returns the owned cell block using the slot count from the query context.
    ///
    /// # Safety
    /// `len` must equal the allocation's cell count.
    unsafe fn cells(&self, len: usize) -> &[A] {
        unsafe { std::slice::from_raw_parts(self.cells, len) }
    }
}

impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool, S: SeenMask>
    AggregationValue for Dynamic<A, ONLY_ADDITIVE, S>
{
    type Owned = OwnedDynamic<A, ONLY_ADDITIVE>;
    /// Number of aggregation slots (the tracking mask cell is not counted).
    type StorageMetadata = usize;
    type Reader<'b> = DynReader<'b>;
    /// Slot descriptors and the arena used by strings and owned top-k copies.
    type SharedContext = (Arc<[AggregationSlot]>, Arc<SharedArena>);
    type ColumnBuilder = DynamicColumnBuilder<A, ONLY_ADDITIVE, S>;
    type SortKey = i128;
    /// Per-worker arena handle for winning strings and owned copies.
    type WorkerContext = WorkerArena;

    fn make_reader<'b>(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> DynReader<'b> {
        DynReader::bind(batch, slots)
    }

    fn storage_metadata(ctx: &Self::SharedContext) -> usize {
        ctx.0.len()
    }

    fn metadata_for_arity<const N: usize>() -> usize {
        N
    }

    #[inline(always)]
    fn dispatch_arity<R>(metadata: usize, body: impl ArityBody<R>) -> R {
        // Specialize common arities. The fallback keeps code size bounded for
        // larger signatures.
        match metadata {
            1 => body.run::<1>(),
            2 => body.run::<2>(),
            3 => body.run::<3>(),
            4 => body.run::<4>(),
            _ => body.run::<0>(),
        }
    }

    fn stored_size(metadata: usize) -> usize {
        // A tracking run appends one whole cell holding the seen bits.
        (metadata + S::TRACKING as usize) * size_of::<A>()
    }

    fn stored_align() -> usize {
        align_of::<A>()
    }

    #[inline(always)]
    unsafe fn from_entry<'a>(ptr: *const u8, metadata: usize) -> &'a Self {
        // SAFETY of the cast: `repr(transparent)` over `[A]` gives Self the
        // same layout and slice metadata as the cell run (which includes the
        // trailing mask cell when tracking).
        let len = metadata + S::TRACKING as usize;
        unsafe { &*(std::ptr::slice_from_raw_parts(ptr as *const A, len) as *const Self) }
    }

    #[inline(always)]
    unsafe fn from_entry_mut<'a>(ptr: *mut u8, metadata: usize) -> &'a mut Self {
        let len = metadata + S::TRACKING as usize;
        unsafe { &mut *(std::ptr::slice_from_raw_parts_mut(ptr as *mut A, len) as *mut Self) }
    }

    #[inline(always)]
    fn seed(&mut self, reader: &Self::Reader<'_>, idx: usize, worker_context: &mut WorkerArena) {
        if S::TRACKING {
            return self.seed_tracked(reader, idx, worker_context);
        }
        let operations = reader.operations.as_slice();
        debug_assert_eq!(self.1.len(), operations.len());
        // Constant lengths let LLVM unroll the common small signatures.
        // Larger signatures use the ordinary slice loop to limit generated
        // code and register pressure.
        match self.1.len() {
            1 => self.seed_fixed::<1>(operations, idx, worker_context),
            2 => self.seed_fixed::<2>(operations, idx, worker_context),
            3 => self.seed_fixed::<3>(operations, idx, worker_context),
            4 => self.seed_fixed::<4>(operations, idx, worker_context),
            _ => {
                for (cell, operation) in self.1.iter_mut().zip(operations.iter()) {
                    *cell = operation.seed::<A, ONLY_ADDITIVE>(idx, worker_context);
                }
            }
        }
    }

    #[inline(always)]
    fn update(
        &mut self,
        reader: &Self::Reader<'_>,
        idx: usize,
        worker_context: &mut WorkerArena,
        context: &Self::SharedContext,
    ) {
        if S::TRACKING {
            return self.update_tracked(reader, idx, worker_context, context);
        }
        let (_, shared) = context;
        let operations = reader.operations.as_slice();
        // Match `seed` so repeated-key updates also unroll.
        match self.1.len() {
            1 => self.update_fixed::<1>(operations, idx, worker_context, shared),
            2 => self.update_fixed::<2>(operations, idx, worker_context, shared),
            3 => self.update_fixed::<3>(operations, idx, worker_context, shared),
            4 => self.update_fixed::<4>(operations, idx, worker_context, shared),
            _ => {
                for (cell, operation) in self.1.iter_mut().zip(operations.iter()) {
                    *cell =
                        operation.update::<A, ONLY_ADDITIVE>(*cell, idx, worker_context, shared);
                }
            }
        }
    }

    #[inline(always)]
    fn merge_from(&mut self, source: &Self, context: &Self::SharedContext) {
        if S::TRACKING {
            return self.merge_tracked(source, context);
        }
        // Match `seed` so common merge signatures unroll.
        match self.1.len() {
            1 => self.merge_fixed::<1>(source, context),
            2 => self.merge_fixed::<2>(source, context),
            3 => self.merge_fixed::<3>(source, context),
            4 => self.merge_fixed::<4>(source, context),
            _ => {
                let (slots, shared) = context;
                // Indexing keeps this loop free of function calls. An
                // iterator adapter that the compiler does not inline is a
                // call, and a call here makes every caller's row loop keep
                // its values on the stack and reload them for each row.
                let cell_count = self.1.len();
                let source_cells = &source.1[..cell_count];
                let slots = &slots[..cell_count];
                for slot_index in 0..cell_count {
                    if ONLY_ADDITIVE {
                        self.1[slot_index] = self.1[slot_index] + source_cells[slot_index];
                    } else {
                        self.1[slot_index].merge_cells(
                            source_cells[slot_index],
                            &slots[slot_index],
                            shared,
                        );
                    }
                }
            }
        }
    }

    #[inline(always)]
    fn copy_from(&mut self, source: &Self) {
        // Constant-size copies compile to direct loads and stores. Larger
        // values use the slice implementation.
        match self.1.len() {
            1 => self.clone_fixed::<1>(source),
            2 => self.clone_fixed::<2>(source),
            3 => self.clone_fixed::<3>(source),
            4 => self.clone_fixed::<4>(source),
            _ => self.1.copy_from_slice(&source.1),
        }
    }

    #[inline(always)]
    fn sort_key(&self, slot: usize) -> i128 {
        // The planner only requests top-k sort keys for numeric slots.
        self.1[slot].into()
    }

    fn to_owned(
        &self,
        context: &Self::SharedContext,
        worker_context: &mut Option<WorkerArena>,
    ) -> OwnedDynamic<A, ONLY_ADDITIVE> {
        let stored = &self.1;
        let worker_context = worker_context.get_or_insert_with(|| context.worker());
        let cells: *mut A = worker_context.alloc_cells(stored.len());
        unsafe { std::ptr::copy_nonoverlapping(stored.as_ptr(), cells, stored.len()) };
        OwnedDynamic { cells }
    }
}

/// The per-batch reader for a [`Dynamic`] value: the bound operations plus each
/// slot's null buffer. Validity lives here, in one side array, read solely by a
/// tracking ([`SeenMask`]) instantiation: an untracked one never touches it,
/// so the per-row operation code carries no validity machinery. A `COUNT(*)`
/// slot's entry is `None` (every row counts) whatever its placeholder column
/// holds.
pub struct DynReader<'b> {
    operations: OperationsReader<'b>,
    nulls: Box<[Option<&'b NullBuffer>]>,
}

impl<'b> DynReader<'b> {
    fn bind(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> Self {
        DynReader {
            operations: OperationsReader::bind(batch, slots),
            nulls: slots
                .iter()
                .map(|slot| match slot.kind {
                    AggregationKind::CountStar => None,
                    _ => batch.column(slot.column).nulls(),
                })
                .collect(),
        }
    }
}

impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool, S: SeenMask>
    Dynamic<A, ONLY_ADDITIVE, S>
{
    /// The stored seen bits (the trailing mask cell, tracking only). One whole
    /// cell holds them, so up to 64 slots track per group.
    #[inline(always)]
    fn seen_bits(&self) -> u64 {
        debug_assert!(S::TRACKING);
        Into::<i128>::into(self.1[self.1.len() - 1]) as u64
    }

    #[inline(always)]
    fn set_seen_bits(&mut self, bits: u64) {
        debug_assert!(S::TRACKING);
        let last = self.1.len() - 1;
        self.1[last] = A::from(bits as i64);
    }

    /// Seeds a tracking run: a NULL row seeds the fold's identity and leaves
    /// the slot's seen bit unset; a count sets its bit regardless (its output
    /// is `0`, never NULL).
    fn seed_tracked(
        &mut self,
        reader: &DynReader<'_>,
        idx: usize,
        worker_context: &mut WorkerArena,
    ) {
        let operations = reader.operations.as_slice();
        debug_assert_eq!(self.1.len(), operations.len() + 1);
        let mut bits = 0u64;
        for (slot_index, operation) in operations.iter().enumerate() {
            let valid = reader.nulls[slot_index].is_none_or(|nulls| nulls.is_valid(idx));
            bits |= ((valid || operation.always_seen()) as u64) << slot_index;
            self.1[slot_index] = if valid {
                operation.seed::<A, ONLY_ADDITIVE>(idx, worker_context)
            } else {
                operation.empty::<A, ONLY_ADDITIVE>()
            };
        }
        self.set_seen_bits(bits);
    }

    /// Folds one row into a tracking run: a NULL row keeps every cell
    /// (identity cells make that equivalent to folding nothing); a string
    /// extreme must not resolve an unseen cell through the arena, so its
    /// operation re-seeds on the group's first valid row.
    fn update_tracked(
        &mut self,
        reader: &DynReader<'_>,
        idx: usize,
        worker_context: &mut WorkerArena,
        context: &(Arc<[AggregationSlot]>, Arc<SharedArena>),
    ) {
        let (_, shared) = context;
        let operations = reader.operations.as_slice();
        debug_assert_eq!(self.1.len(), operations.len() + 1);
        let mut bits = self.seen_bits();
        for (slot_index, operation) in operations.iter().enumerate() {
            let valid = reader.nulls[slot_index].is_none_or(|nulls| nulls.is_valid(idx));
            if valid {
                let was_seen = bits & (1 << slot_index) != 0;
                self.1[slot_index] = operation.update_seen::<A, ONLY_ADDITIVE>(
                    self.1[slot_index],
                    idx,
                    was_seen,
                    worker_context,
                    shared,
                );
            }
            bits |= ((valid || operation.always_seen()) as u64) << slot_index;
        }
        self.set_seen_bits(bits);
    }

    /// Merges a tracking partial: numeric identity cells absorb so their merge
    /// stays unconditional; only a string extreme, whose unseen cell must
    /// never resolve through the arena, consults the seen bits. Masks union.
    fn merge_tracked(
        &mut self,
        source: &Self,
        context: &(Arc<[AggregationSlot]>, Arc<SharedArena>),
    ) {
        let (slots, shared) = context;
        let source_bits = source.seen_bits();
        let bits = self.seen_bits();
        debug_assert_eq!(self.1.len(), slots.len() + 1);
        for (slot_index, slot) in slots.iter().enumerate() {
            let source_cell = source.1[slot_index];
            if ONLY_ADDITIVE {
                self.1[slot_index] = self.1[slot_index] + source_cell;
            } else if slot.output_type == DataType::Utf8View {
                let destination_seen = bits & (1 << slot_index) != 0;
                let source_seen = source_bits & (1 << slot_index) != 0;
                match (destination_seen, source_seen) {
                    (true, true) => {
                        self.1[slot_index].merge_cells(source_cell, slot, shared);
                    }
                    (false, true) => self.1[slot_index] = source_cell,
                    (_, false) => {}
                }
            } else {
                self.1[slot_index].merge_cells(source_cell, slot, shared);
            }
        }
        self.set_seen_bits(bits | source_bits);
    }
}

impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool, S: SeenMask>
    Dynamic<A, ONLY_ADDITIVE, S>
{
    /// Seeds a fixed-size cell array so the slot loop can unroll.
    #[inline(always)]
    fn seed_fixed<const N: usize>(
        &mut self,
        operations: &[OperationReader<'_>],
        idx: usize,
        worker_context: &mut WorkerArena,
    ) {
        // SAFETY: the caller dispatched after matching both lengths to `N`.
        debug_assert!(self.1.len() == N && operations.len() == N);
        let destination = unsafe { &mut *(self.1.as_mut_ptr() as *mut [A; N]) };
        let operations = unsafe { &*(operations.as_ptr() as *const [OperationReader<'_>; N]) };
        for operation_index in 0..N {
            destination[operation_index] =
                operations[operation_index].seed::<A, ONLY_ADDITIVE>(idx, worker_context);
        }
    }

    /// Updates a fixed-size cell array so the slot loop can unroll.
    #[inline(always)]
    fn update_fixed<const N: usize>(
        &mut self,
        operations: &[OperationReader<'_>],
        idx: usize,
        worker_context: &mut WorkerArena,
        shared: &Arc<SharedArena>,
    ) {
        // SAFETY: the caller dispatched after matching both lengths to `N`.
        debug_assert!(self.1.len() == N && operations.len() == N);
        let destination = unsafe { &mut *(self.1.as_mut_ptr() as *mut [A; N]) };
        let operations = unsafe { &*(operations.as_ptr() as *const [OperationReader<'_>; N]) };
        for operation_index in 0..N {
            destination[operation_index] = operations[operation_index].update::<A, ONLY_ADDITIVE>(
                destination[operation_index],
                idx,
                worker_context,
                shared,
            );
        }
    }

    /// Merges fixed-size cell arrays so the slot loop can unroll.
    #[inline(always)]
    fn merge_fixed<const N: usize>(
        &mut self,
        source: &Self,
        context: &(Arc<[AggregationSlot]>, Arc<SharedArena>),
    ) {
        let (slots, shared) = context;
        // SAFETY: all three cell runs come from the same `N`-slot signature.
        debug_assert!(self.1.len() == N && source.1.len() == N && slots.len() == N);
        let destination = unsafe { &mut *(self.1.as_mut_ptr() as *mut [A; N]) };
        let source = unsafe { &*(source.1.as_ptr() as *const [A; N]) };
        for slot_index in 0..N {
            if ONLY_ADDITIVE {
                // COUNT and SUM both merge by addition.
                destination[slot_index] = destination[slot_index] + source[slot_index];
            } else {
                destination[slot_index].merge_cells(source[slot_index], &slots[slot_index], shared);
            }
        }
    }

    /// Copies a fixed-size cell array with an array assignment.
    #[inline(always)]
    fn clone_fixed<const N: usize>(&mut self, source: &Self) {
        // SAFETY: the caller dispatched after matching both lengths to `N`.
        debug_assert!(self.1.len() == N && source.1.len() == N);
        let destination = unsafe { &mut *(self.1.as_mut_ptr() as *mut [A; N]) };
        let source = unsafe { &*(source.1.as_ptr() as *const [A; N]) };
        *destination = *source;
    }
}

/// One output-column builder per dynamic aggregation slot, plus (when `S`
/// tracks) one `u64` column buffering each pushed group's seen bits so
/// `finish` can render never-seen slots as SQL NULL.
pub struct DynamicColumnBuilder<
    A: IntCell + StringCell + F64Cell + WideCell,
    const ONLY_ADDITIVE: bool,
    S: SeenMask,
> {
    builders: Vec<SlabColumn<A>>,
    masks: Option<SlabColumn<u64>>,
    all_seen: u64,
    marker: std::marker::PhantomData<S>,
}

impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool, S: SeenMask>
    ValueColumnBuilder for DynamicColumnBuilder<A, ONLY_ADDITIVE, S>
{
    type Value = Dynamic<A, ONLY_ADDITIVE, S>;
    type Context = (Arc<[AggregationSlot]>, Arc<SharedArena>);

    fn with_capacity(allocator: &mut SlabAllocator, rows: usize, context: &Self::Context) -> Self {
        let (slots, _) = context;
        Self {
            builders: slots
                .iter()
                .map(|_| SlabColumn::with_capacity(allocator, rows))
                .collect(),
            masks: S::TRACKING.then(|| SlabColumn::with_capacity(allocator, rows)),
            all_seen: u64::MAX,
            marker: std::marker::PhantomData,
        }
    }

    #[inline(always)]
    fn push(&mut self, value: &OwnedDynamic<A, ONLY_ADDITIVE>) {
        // The query context gives both the allocation and columns the same
        // slot count; a tracking run carries one extra trailing mask cell.
        let cell_count = self.builders.len() + S::TRACKING as usize;
        let cells = unsafe { value.cells(cell_count) };
        for (builder, &cell) in self.builders.iter_mut().zip(cells.iter()) {
            builder.push(cell);
        }
        if S::TRACKING {
            let bits = Into::<i128>::into(cells[cell_count - 1]) as u64;
            self.push_mask(bits);
        }
    }

    #[inline(always)]
    fn push_stored(&mut self, stored: &Dynamic<A, ONLY_ADDITIVE, S>) {
        // `zip` stops at the value columns, so a tracking run's mask cell
        // never lands in a value column.
        for (builder, &cell) in self.builders.iter_mut().zip(stored.1.iter()) {
            builder.push(cell);
        }
        if S::TRACKING {
            self.push_mask(stored.seen_bits());
        }
    }

    fn finish(self, context: &Self::Context) -> (Vec<Field>, Vec<ArrayRef>) {
        let (slots, arena) = context;
        let Self {
            builders,
            masks,
            all_seen,
            marker: _,
        } = self;
        let mask_slices = masks.as_ref().map(|column| column.as_slice());
        // The null buffer for one slot: NULL where the group's seen bit is
        // unset; `None` when every pushed mask has the bit (no group is NULL,
        // and in particular whenever a count keeps the bit always set).
        let nulls_for_slot = |slot: usize| -> Option<NullBuffer> {
            if all_seen & (1 << slot) != 0 {
                return None;
            }
            let masks = mask_slices.expect("a cleared all_seen bit implies a tracking mask column");
            Some(NullBuffer::new(BooleanBuffer::collect_bool(
                masks.len(),
                |i| masks[i] & (1 << slot) != 0,
            )))
        };
        let mut fields = Vec::with_capacity(builders.len());
        let mut arrays = Vec::with_capacity(builders.len());
        for (slot_index, builder) in builders.into_iter().enumerate() {
            let descriptor = &slots[slot_index];
            let name = &format!("v{slot_index}");
            let ty = &descriptor.output_type;
            let nulls = nulls_for_slot(slot_index);
            let (field, array) = match descriptor.kind {
                AggregationKind::CountStar | AggregationKind::Count => {
                    Count::<A>::finish(name, builder, nulls)
                }
                AggregationKind::Sum if ty.is_floating() => {
                    F64Sum::<A>::finish(name, builder, nulls)
                }
                AggregationKind::Sum => Sum::<A>::finish(name, builder, nulls),
                AggregationKind::Min if *ty == DataType::Utf8View => {
                    StrMin::<A>::finish(name, builder, arena, nulls)
                }
                AggregationKind::Min if ty.is_floating() => {
                    F64Min::<A>::finish(name, builder, nulls)
                }
                AggregationKind::Min => Min::<A>::finish(name, builder, nulls),
                AggregationKind::Max if *ty == DataType::Utf8View => {
                    StrMax::<A>::finish(name, builder, arena, nulls)
                }
                AggregationKind::Max if ty.is_floating() => {
                    F64Max::<A>::finish(name, builder, nulls)
                }
                AggregationKind::Max => Max::<A>::finish(name, builder, nulls),
                AggregationKind::First => {
                    unreachable!("the planner rejects FIRST inside a GROUP BY")
                }
            };
            fields.push(field);
            arrays.push(array);
        }
        (fields, arrays)
    }
}

impl<A: IntCell + StringCell + F64Cell + WideCell, const ONLY_ADDITIVE: bool, S: SeenMask>
    DynamicColumnBuilder<A, ONLY_ADDITIVE, S>
{
    #[inline(always)]
    fn push_mask(&mut self, bits: u64) {
        self.masks
            .as_mut()
            .expect("a tracking mask column buffers every pushed mask")
            .push(bits);
        self.all_seen &= bits;
    }
}
