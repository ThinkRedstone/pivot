//! Batch-bound aggregation operations for [`Dynamic`](super::Dynamic).
//!
//! [`OperationReader`] combines an aggregation kind with its input column. The column
//! reader is selected once per batch, so row processing dispatches on one enum
//! instead of repeatedly inspecting Arrow data types.

use super::super::super::cell::{F64Cell, IntCell, StringCell, WideCell};
use super::super::super::fold::Fold;
use super::super::super::read::{Read, StrRead};
use super::super::super::{
    AggregationKind, AggregationSlot, Count, F64Max, F64Min, F64Sum, Max, Min, StrMax, StrMin, Sum,
    U128Max, U128Min, U128Sum,
};
use crate::operations::unary::group::arena::{SharedArena, WorkerArena};
use arrow_array::{RecordBatch, StringViewArray};
use arrow_schema::DataType;
use std::array;
use std::sync::Arc;

use super::readers::{F64Reader, I64Reader, U128Reader};

/// An aggregation operation bound to its input column for one batch.
///
/// Prefixes identify the value returned by the reader: `I64`, `F64`, or
/// `U128` for the engine's `i128` wide values.
pub enum OperationReader<'b> {
    Count,
    I64Sum(I64Reader<'b>),
    I64Min(I64Reader<'b>),
    I64Max(I64Reader<'b>),
    F64Sum(F64Reader<'b>),
    F64Min(F64Reader<'b>),
    F64Max(F64Reader<'b>),
    U128Sum(U128Reader<'b>),
    U128Min(U128Reader<'b>),
    U128Max(U128Reader<'b>),
    StrMin(&'b StringViewArray),
    StrMax(&'b StringViewArray),
}

impl<'b> OperationReader<'b> {
    pub(in super::super) fn bind(batch: &'b RecordBatch, descriptor: &AggregationSlot) -> Self {
        use AggregationKind::*;
        match descriptor.kind {
            CountStar | Count => OperationReader::Count,
            Sum => Self::bind_numeric(
                batch,
                descriptor.column,
                OperationReader::I64Sum,
                OperationReader::F64Sum,
                OperationReader::U128Sum,
            ),
            Min => Self::bind_extreme(
                batch,
                descriptor.column,
                OperationReader::StrMin,
                OperationReader::I64Min,
                OperationReader::F64Min,
                OperationReader::U128Min,
            ),
            Max => Self::bind_extreme(
                batch,
                descriptor.column,
                OperationReader::StrMax,
                OperationReader::I64Max,
                OperationReader::F64Max,
                OperationReader::U128Max,
            ),
            First => unreachable!("the planner rejects FIRST inside a GROUP BY"),
        }
    }

    /// Binds a numeric column to the reader for its value family. An integer or
    /// `Decimal64` column reads as `i64` (a decimal's raw unscaled values fold
    /// like an integer's), a float as `f64`, a `Decimal128` (a decimal column or
    /// a re-read wide partial) as `i128`.
    fn bind_numeric(
        batch: &'b RecordBatch,
        column: usize,
        on_i64: fn(I64Reader<'b>) -> Self,
        on_f64: fn(F64Reader<'b>) -> Self,
        on_u128: fn(U128Reader<'b>) -> Self,
    ) -> Self {
        match batch.column(column).data_type() {
            DataType::Int16 | DataType::Int32 | DataType::Int64 | DataType::Decimal64(_, _) => {
                on_i64(I64Reader::bind(batch, column))
            }
            DataType::Float32 | DataType::Float64 => on_f64(F64Reader::bind(batch, column)),
            DataType::Decimal128(_, _) => on_u128(U128Reader::bind(batch, column)),
            other => panic!("numeric aggregate over unsupported column type {other:?}"),
        }
    }

    /// Binds a string or numeric column for `MIN` and `MAX`.
    fn bind_extreme(
        batch: &'b RecordBatch,
        column: usize,
        on_str: fn(&'b StringViewArray) -> Self,
        on_i64: fn(I64Reader<'b>) -> Self,
        on_f64: fn(F64Reader<'b>) -> Self,
        on_u128: fn(U128Reader<'b>) -> Self,
    ) -> Self {
        if let DataType::Utf8View = batch.column(column).data_type() {
            on_str(StrRead::bind(batch, column))
        } else {
            Self::bind_numeric(batch, column, on_i64, on_f64, on_u128)
        }
    }

    /// Seeds one cell from the first row in a group.
    ///
    /// With `ONLY_ADDITIVE`, non-additive variants are unreachable. The planner
    /// only enables that specialization for COUNT and integer SUM slots.
    #[inline(always)]
    pub(in super::super) fn seed<
        A: IntCell + StringCell + F64Cell + WideCell,
        const ONLY_ADDITIVE: bool,
    >(
        &self,
        row_index: usize,
        worker_context: &mut WorkerArena,
    ) -> A {
        if ONLY_ADDITIVE {
            // Three arms and one shared failure compile to a short chain of
            // compares. One failing arm per other variant makes a switch
            // wide enough that some builds turn it into an indirect jump,
            // which costs every row of the probe loop.
            return match self {
                OperationReader::Count => Count::<A>::seed(()),
                OperationReader::I64Sum(reader) => Sum::<A>::seed(reader.read(row_index)),
                OperationReader::U128Sum(reader) => U128Sum::<A>::seed(reader.read(row_index)),
                _ => fail_on_non_additive_operation(),
            };
        }
        match self {
            OperationReader::Count => Count::<A>::seed(()),
            OperationReader::I64Sum(reader) => Sum::<A>::seed(reader.read(row_index)),
            OperationReader::U128Sum(reader) => U128Sum::<A>::seed(reader.read(row_index)),
            OperationReader::I64Min(reader) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    Min::<A>::seed(reader.read(row_index))
                }
            }
            OperationReader::I64Max(reader) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    Max::<A>::seed(reader.read(row_index))
                }
            }
            OperationReader::F64Sum(reader) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    F64Sum::<A>::seed(reader.read(row_index))
                }
            }
            OperationReader::F64Min(reader) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    F64Min::<A>::seed(reader.read(row_index))
                }
            }
            OperationReader::F64Max(reader) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    F64Max::<A>::seed(reader.read(row_index))
                }
            }
            OperationReader::U128Min(reader) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    U128Min::<A>::seed(reader.read(row_index))
                }
            }
            OperationReader::U128Max(reader) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    U128Max::<A>::seed(reader.read(row_index))
                }
            }
            OperationReader::StrMin(array) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    StrMin::<A>::seed(StrRead::read(array, row_index), worker_context)
                }
            }
            OperationReader::StrMax(array) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    StrMax::<A>::seed(StrRead::read(array, row_index), worker_context)
                }
            }
        }
    }

    /// Whether this operation's seen bit is always set: a count renders `0`
    /// for a group with no counted rows, never SQL NULL.
    #[inline(always)]
    pub(in super::super) fn always_seen(&self) -> bool {
        matches!(self, OperationReader::Count)
    }

    /// The cell of a group whose slot has seen no non-NULL value yet: the
    /// fold's identity, absorbed by any later fold (`0` for a sum or count,
    /// the width's extreme for `MIN`/`MAX`). A string extreme's cell is the
    /// default empty view, guarded by the seen bit so it is never resolved
    /// through the arena.
    #[inline(always)]
    pub(in super::super) fn empty<
        A: IntCell + StringCell + F64Cell + WideCell,
        const ONLY_ADDITIVE: bool,
    >(
        &self,
    ) -> A {
        match self {
            OperationReader::Count => Count::<A>::empty(),
            OperationReader::I64Sum(_) | OperationReader::U128Sum(_) => Sum::<A>::empty(),
            OperationReader::I64Min(_) | OperationReader::U128Min(_) => Min::<A>::empty(),
            OperationReader::I64Max(_) | OperationReader::U128Max(_) => Max::<A>::empty(),
            OperationReader::F64Sum(_) => F64Sum::<A>::empty(),
            OperationReader::F64Min(_) => F64Min::<A>::empty(),
            OperationReader::F64Max(_) => F64Max::<A>::empty(),
            OperationReader::StrMin(_) | OperationReader::StrMax(_) => A::default(),
        }
    }

    /// Folds one valid row into an existing cell of a tracking run. A string
    /// extreme whose slot has not seen a value yet re-seeds instead of
    /// updating: its unseen cell holds no arena key to resolve; every other
    /// operation folds normally (identity cells absorb).
    #[inline(always)]
    pub(in super::super) fn update_seen<
        A: IntCell + StringCell + F64Cell + WideCell,
        const ONLY_ADDITIVE: bool,
    >(
        &self,
        cell: A,
        row_index: usize,
        was_seen: bool,
        worker_context: &mut WorkerArena,
        shared: &Arc<SharedArena>,
    ) -> A {
        match self {
            OperationReader::StrMin(array) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else if !was_seen {
                    StrMin::<A>::seed(StrRead::read(array, row_index), worker_context)
                } else {
                    StrMin::<A>::update(
                        cell,
                        StrRead::read(array, row_index),
                        worker_context,
                        shared,
                    )
                }
            }
            OperationReader::StrMax(array) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else if !was_seen {
                    StrMax::<A>::seed(StrRead::read(array, row_index), worker_context)
                } else {
                    StrMax::<A>::update(
                        cell,
                        StrRead::read(array, row_index),
                        worker_context,
                        shared,
                    )
                }
            }
            _ => self.update::<A, ONLY_ADDITIVE>(cell, row_index, worker_context, shared),
        }
    }

    /// Folds one row into an existing cell.
    #[inline(always)]
    pub(in super::super) fn update<
        A: IntCell + StringCell + F64Cell + WideCell,
        const ONLY_ADDITIVE: bool,
    >(
        &self,
        cell: A,
        row_index: usize,
        worker_context: &mut WorkerArena,
        shared: &Arc<SharedArena>,
    ) -> A {
        if ONLY_ADDITIVE {
            // Kept to three arms and one shared failure for the same reason
            // as in `seed`.
            return match self {
                OperationReader::Count => Count::<A>::update(cell, ()),
                OperationReader::I64Sum(reader) => Sum::<A>::update(cell, reader.read(row_index)),
                OperationReader::U128Sum(reader) => {
                    U128Sum::<A>::update(cell, reader.read(row_index))
                }
                _ => fail_on_non_additive_operation(),
            };
        }
        match self {
            OperationReader::Count => Count::<A>::update(cell, ()),
            OperationReader::I64Sum(reader) => Sum::<A>::update(cell, reader.read(row_index)),
            OperationReader::U128Sum(reader) => U128Sum::<A>::update(cell, reader.read(row_index)),
            OperationReader::I64Min(reader) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    Min::<A>::update(cell, reader.read(row_index))
                }
            }
            OperationReader::I64Max(reader) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    Max::<A>::update(cell, reader.read(row_index))
                }
            }
            OperationReader::F64Sum(reader) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    F64Sum::<A>::update(cell, reader.read(row_index))
                }
            }
            OperationReader::F64Min(reader) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    F64Min::<A>::update(cell, reader.read(row_index))
                }
            }
            OperationReader::F64Max(reader) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    F64Max::<A>::update(cell, reader.read(row_index))
                }
            }
            OperationReader::U128Min(reader) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    U128Min::<A>::update(cell, reader.read(row_index))
                }
            }
            OperationReader::U128Max(reader) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    U128Max::<A>::update(cell, reader.read(row_index))
                }
            }
            OperationReader::StrMin(array) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    StrMin::<A>::update(
                        cell,
                        StrRead::read(array, row_index),
                        worker_context,
                        shared,
                    )
                }
            }
            OperationReader::StrMax(array) => {
                if ONLY_ADDITIVE {
                    unreachable!()
                } else {
                    StrMax::<A>::update(
                        cell,
                        StrRead::read(array, row_index),
                        worker_context,
                        shared,
                    )
                }
            }
        }
    }
}

/// The planner enables the additive specialization only for COUNT and integer
/// SUM slots, so any other operation reaching it is a planning bug.
#[cold]
#[inline(never)]
fn fail_on_non_additive_operation() -> ! {
    panic!("an additive group value holds only COUNT and integer SUM slots")
}

/// Aggregation operations bound to one record batch.
///
/// Up to eight operations stay inline. This lets the compiler keep their column
/// pointers in registers across the probe loop. Larger signatures use a boxed
/// slice to keep the enum reasonably sized.
pub enum OperationsReader<'b> {
    Inline {
        len: usize,
        operations: [OperationReader<'b>; 8],
    },
    Boxed(Box<[OperationReader<'b>]>),
}

impl<'b> OperationsReader<'b> {
    pub(in super::super) fn bind(batch: &'b RecordBatch, slots: &[AggregationSlot]) -> Self {
        if slots.len() <= 8 {
            OperationsReader::Inline {
                len: slots.len(),
                // `as_slice` hides these placeholder entries.
                operations: array::from_fn(|index| {
                    if index < slots.len() {
                        OperationReader::bind(batch, &slots[index])
                    } else {
                        OperationReader::Count
                    }
                }),
            }
        } else {
            OperationsReader::Boxed(
                slots
                    .iter()
                    .map(|descriptor| OperationReader::bind(batch, descriptor))
                    .collect(),
            )
        }
    }

    #[inline(always)]
    pub(in super::super) fn as_slice(&self) -> &[OperationReader<'b>] {
        match self {
            OperationsReader::Inline { len, operations } => &operations[..*len],
            OperationsReader::Boxed(operations) => operations,
        }
    }
}

/// Merging one accumulator cell into another according to a slot descriptor.
pub(in super::super) trait MergeCells: Sized {
    /// Merges `source` into this cell with the slot's operation.
    fn merge_cells(
        &mut self,
        source: Self,
        descriptor: &AggregationSlot,
        shared: &Arc<SharedArena>,
    );
}

impl<A: IntCell + StringCell + F64Cell + WideCell> MergeCells for A {
    #[inline(always)]
    fn merge_cells(&mut self, source: A, descriptor: &AggregationSlot, shared: &Arc<SharedArena>) {
        let ty = &descriptor.output_type;
        *self = match descriptor.kind {
            AggregationKind::CountStar | AggregationKind::Count => Count::<A>::merge(*self, source),
            AggregationKind::Sum if ty.is_floating() => F64Sum::<A>::merge(*self, source),
            AggregationKind::Sum => Sum::<A>::merge(*self, source),
            AggregationKind::Min if *ty == DataType::Utf8View => {
                StrMin::<A>::merge(*self, source, shared)
            }
            AggregationKind::Min if ty.is_floating() => F64Min::<A>::merge(*self, source),
            AggregationKind::Min => Min::<A>::merge(*self, source),
            AggregationKind::Max if *ty == DataType::Utf8View => {
                StrMax::<A>::merge(*self, source, shared)
            }
            AggregationKind::Max if ty.is_floating() => F64Max::<A>::merge(*self, source),
            AggregationKind::Max => Max::<A>::merge(*self, source),
            AggregationKind::First => unreachable!("the planner rejects FIRST inside a GROUP BY"),
        };
    }
}
