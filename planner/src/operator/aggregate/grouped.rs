//! The general grouped aggregate: `GROUP BY k1, k2, ...` with one or more of
//! COUNT(*)/SUM/COUNT/MIN/MAX.
//!
//! DuckDB lowers grouped `AVG(c)` to `sum(c)`+`count(c)` with a downstream
//! divide, so the node here only ever holds count/sum/min/max slots.
//!
//! Lowering selects concrete key and value types independently:
//!
//! * **key**: a single integer/string column gets its dedicated extractor
//!   ([`IntKeyExtractor`]/[`StringKeyExtractor`]); two integer keys pack into
//!   [`IntPairKeyExtractor`]; anything else (3+ keys, mixed types) byte-encodes
//!   the tuple with [`RowKeyExtractor`]. *Computed* keys (`date_trunc(...)`,
//!   `ip - 1`, `CASE ...`) and computed aggregate arguments (`SUM(a * b)`) are
//!   first materialised into leading columns (see
//!   [`Aggregate::materialize_inputs`]), so from the dispatch's view every key
//!   and aggregate argument is a column.
//! * **value**: recognized signatures lower to a branch-free [`Compiled`]
//!   tuple; every other shape folds each slot by kind in the runtime-arity
//!   [`Dynamic`] (numeric, string, or mixed; branch-free `+` when
//!   all-additive), in `i128` when a slot needs the width (see [`Aggregate`]'s
//!   rule) else the narrow `i64`. The slot count is fixed at query build, so
//!   one container covers every arity.

use super::{Aggregate, aggregation_slots, needs_wide_accumulator, row_key_schema};
use crate::compile::{Error, ExprEvalFn, ExprFn};
use crate::expression::{AggregateFunc, Expression, Ref};
use crate::types::MAX_DECIMAL64_PRECISION;
use crate::types::Type;
use arrow_array::types::{
    Decimal64Type, Decimal128Type, Int8Type, Int16Type, Int32Type, Int64Type,
};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use dispatch::{
    AggregationKind, AggregationSlot, Compiled, CountSlot, CountValidSlot, Distinct, Dynamic,
    GroupLimit, IntKeyExtractor, IntPairKeyExtractor, IntStrKeyExtractor, RecordBatchOperatorSpec,
    RowKeyExtractor, StringKeyExtractor, SumSlot,
};
use std::sync::Arc;

impl Aggregate {
    pub(super) fn compile_grouped(
        &self,
        input: RecordBatchOperatorSpec,
        nullability: &[bool],
    ) -> Result<RecordBatchOperatorSpec, Error> {
        // Every group key and aggregate argument is a plain column after
        // `materialize_inputs`, so resolve the keys directly and build the
        // operator with no further column shift.
        let keys = self.resolved_keys();
        build_group_by_operator(
            input,
            &keys,
            &self.expressions,
            self.output_limit,
            nullability,
        )
    }

    /// The group keys as `(column, type)` pairs. Every key is a plain column
    /// reference once [`materialize_inputs`](Self::materialize_inputs) has run,
    /// so this never sees a computed key. Shared by the general grouped path and
    /// the two-level `COUNT(DISTINCT)` lowering.
    pub(super) fn resolved_keys(&self) -> Vec<(usize, Type)> {
        self.groups
            .iter()
            .map(|g| match g {
                Expression::Ref(r) => (r.column_idx, r.return_type.clone()),
                _ => unreachable!("group keys are materialised before lowering"),
            })
            .collect()
    }

    /// Lower every *computed* group key and aggregate argument into a leading
    /// projected column, returning the re-projected input and a resolved
    /// aggregate whose group keys and aggregate arguments are all plain column
    /// references. This is the single entry point to compilation, so every later
    /// slot builder reads a column index and never evaluates an expression.
    ///
    /// The materialised columns take the leading positions in a fixed order: the
    /// computed group keys (in GROUP BY order), then the computed aggregate
    /// arguments (in expression order). Each keeps a canonical type
    /// (`Int64`/`Utf8View`, or `Date`/`Timestamp` for a temporal one) so it is
    /// typed for the extractor and the numeric readers. The original input columns
    /// follow, shifted right by that count, and every surviving plain reference is
    /// shifted to match.
    ///
    /// ```text
    ///   SELECT date_trunc('day', ts), SUM(a * b)
    ///   GROUP BY date_trunc('day', ts)
    ///
    ///   input cols    [ ts    a    b ]            the aggregate's child output
    ///       |
    ///       v  prepend the 1 computed key, then the 1 computed argument,
    ///       |  shifting the original columns right by 2:
    ///       v
    ///   projected     [ trunc(ts) | a*b | ts    a    b ]
    ///                     #0 key    #1    #2   #3   #4
    ///       |
    ///       v  the resolved aggregate now reads only those leading columns:
    ///       v
    ///   GROUP BY #0,  SUM(#1)
    /// ```
    ///
    /// Returns `None` for the resolved aggregate when nothing is computed (the
    /// input is handed back untouched), so a plain-column aggregate keeps its
    /// fast path with no projection and no clone.
    pub(super) fn materialize_inputs(
        &self,
        input: RecordBatchOperatorSpec,
        input_nullability: Vec<bool>,
    ) -> Result<(RecordBatchOperatorSpec, Vec<bool>, Option<Aggregate>), Error> {
        // The computed sub-expressions to materialise and their canonical column
        // types, group keys first then aggregate arguments.
        let mut computed: Vec<&Expression> = Vec::new();
        let mut types: Vec<Type> = Vec::new();
        for g in &self.groups {
            if !matches!(g, Expression::Ref(_)) {
                let result = g.result_type()?;
                let ty =
                    canonical_input_type(&result).ok_or(Error::DataTypeNotSupportedForGroupBy {
                        column: None,
                        data_type: result,
                    })?;
                computed.push(g);
                types.push(ty);
            }
        }
        for e in &self.expressions {
            let Expression::AggregateFunc(func) = e else {
                continue;
            };
            // Only an argument that is not already a plain column needs one.
            for arg in func
                .arguments()
                .filter(|a| !matches!(a, Expression::Ref(_)))
            {
                let ty = canonical_input_type(&arg.result_type()?)
                    .ok_or_else(|| Error::UnsupportedAggregateExpression(e.clone()))?;
                computed.push(arg);
                types.push(ty);
            }
        }

        // Nothing computed: reuse the input untouched and let the caller lower
        // `self` directly (every key and argument is already a column).
        if computed.is_empty() {
            return Ok((input, input_nullability, None));
        }

        // The materialised columns lead, so the nullability vector mirrors the
        // projection: one entry per computed expression, then the original
        // input's entries shifted right.
        let nullability: Vec<bool> = computed
            .iter()
            .map(|e| e.nullability(&input_nullability))
            .chain(input_nullability.iter().copied())
            .collect();

        let shift = computed.len();
        let input = project_leading_columns(input, &computed, &types)?;

        // Draw the leading materialised columns in the same order they were
        // pushed: group keys first, then aggregate arguments. `next` walks them.
        let mut next = 0;
        let groups = self
            .groups
            .iter()
            .map(|g| resolve_to_column(g, shift, &types, &mut next))
            .collect();
        let expressions = self
            .expressions
            .iter()
            .map(|e| rewrite_arguments(e, shift, &types, &mut next))
            .collect::<Result<_, _>>()?;

        Ok((
            input,
            nullability,
            Some(Aggregate {
                groups,
                expressions,
                output_limit: self.output_limit,
            }),
        ))
    }
}

/// Rewrite an aggregate so each of its arguments is a plain column reference over
/// the materialised input (see [`resolve_to_column`]). `COUNT(*)` has no argument
/// and passes through unchanged. `next` continues the column walk from the group
/// keys, in expression then argument order.
fn rewrite_arguments(
    e: &Expression,
    shift: usize,
    types: &[Type],
    next: &mut usize,
) -> Result<Expression, Error> {
    let mut resolved = e.clone();
    let Expression::AggregateFunc(func) = &mut resolved else {
        return Err(Error::UnsupportedAggregateExpression(e.clone()));
    };
    for arg in func.arguments_mut() {
        *arg = resolve_to_column(arg, shift, types, next);
    }
    Ok(resolved)
}

/// Resolve a materialised group key or aggregate argument to the plain column
/// reference that now holds it: a reference shifts right past the `shift` leading
/// columns; a computed expression takes the next leading column (advancing
/// `next`), as a synthesised reference with no source name.
fn resolve_to_column(e: &Expression, shift: usize, types: &[Type], next: &mut usize) -> Expression {
    match e {
        Expression::Ref(r) => Expression::Ref(Ref {
            column_idx: r.column_idx + shift,
            ..r.clone()
        }),
        _ => {
            let column = *next;
            *next += 1;
            Expression::Ref(Ref {
                column_idx: column,
                return_type: types[column].clone(),
                name: None,
            })
        }
    }
}

/// Build a grouped-aggregate operator from the aggregate `exprs`. The semantic
/// layer over [`dispatch_group_by`]: resolve `exprs` to value slots via the shared
/// [`aggregation_slots`] (the single place aggregate semantics map to an
/// [`AggregationKind`]); coalesce duplicate-valued aggregates and re-expand the
/// output around the [`dispatch_group_by`] call. The entry point for any level that
/// has aggregates: the general grouped path and each level of the two-level
/// `COUNT(DISTINCT)` lowering (whose inner/outer levels are *synthetic* expression
/// lists, which is why this takes `exprs` rather than reading `Aggregate`).
pub(super) fn build_group_by_operator(
    input: RecordBatchOperatorSpec,
    keys: &[(usize, Type)],
    exprs: &[Expression],
    output_limit: Option<GroupLimit>,
    nullability: &[bool],
) -> Result<RecordBatchOperatorSpec, Error> {
    // A GROUP BY with no aggregate expressions is a keys-only dedup: the
    // optimizer leaves this shape when nothing reads the aggregate values (e.g.
    // a COUNT(*) over a grouped subquery prunes the subquery's aggregates). Any
    // pushed-down group limit is dropped, not applied early; the LIMIT operator
    // above still trims the output.
    if exprs.is_empty() {
        return build_dedup_operator(input, keys, nullability);
    }

    // Coalesce aggregates that fold to the same value so each is scattered/merged
    // once and the hash entry stays narrow: a COUNT(col) over a NULL-free column
    // is COUNT(*), a SUM/MIN/MAX of the same column is identical. `to_unique[i]`
    // is the deduped slot expression `i` folds into; the value columns are
    // re-expanded below so the output still has one per expression.
    let mut unique: Vec<&Expression> = Vec::new();
    let mut fold_keys: Vec<FoldKey> = Vec::new();
    let mut to_unique: Vec<usize> = Vec::with_capacity(exprs.len());
    for e in exprs {
        let key = fold_key(e, nullability)?;
        let idx = match fold_keys.iter().position(|k| *k == key) {
            Some(i) => i,
            None => {
                fold_keys.push(key);
                unique.push(e);
                unique.len() - 1
            }
        };
        to_unique.push(idx);
    }
    let unique_exprs: Vec<Expression> = unique.iter().map(|e| (*e).clone()).collect();

    let slots = aggregation_slots(&unique_exprs)?;
    let signatures = aggregation_signatures(&unique_exprs, nullability);

    // A pushed-down Top-K sorts by a value slot identified by expression index;
    // coalescing renumbers the slots, so remap it to the unique slot it folds into
    // (identity when nothing coalesced). The sorted expression may itself be a
    // coalesced duplicate, which folds to the same value, so the order is unchanged.
    let output_limit = match output_limit {
        Some(GroupLimit::TopK {
            slot,
            limit,
            with_ties,
        }) => Some(GroupLimit::TopK {
            slot: to_unique[slot],
            limit,
            with_ties,
        }),
        other => other,
    };

    // Cell width: i128 when a string extreme needs its 128-bit `ArenaKey` cell, or
    // when an aggregate reads a column whose values need the wide accumulator (a
    // 64-bit SUM, or any aggregate over a decimal); else the narrow i64 entry.
    let requires_wide_cells = slots.iter().any(AggregationSlot::is_string_extreme)
        || needs_wide_accumulator(&unique_exprs);

    let grouped = dispatch_group_by(
        input,
        keys,
        slots,
        &signatures,
        requires_wide_cells,
        output_limit,
        nullability,
    )?;

    // No two aggregates coalesced: the output already has one value column per
    // expression, in order.
    if unique_exprs.len() == exprs.len() {
        return Ok(grouped);
    }
    Ok(reexpand_coalesced(grouped, keys.len(), to_unique))
}

/// Re-expand a coalesced group output back to one value column per original
/// expression: keep the `key_count` key columns, then select each expression's
/// coalesced value column via `to_unique`. A zero-copy column select on the
/// (low-cardinality) group output, renaming the value columns back to `v0..` so a
/// value mapped to several expressions stays a distinct field, matching the
/// non-coalesced output shape.
fn reexpand_coalesced(
    input: RecordBatchOperatorSpec,
    key_count: usize,
    to_unique: Vec<usize>,
) -> RecordBatchOperatorSpec {
    let cols: Arc<Vec<usize>> = Arc::new(
        (0..key_count)
            .chain(to_unique.iter().map(|&u| key_count + u))
            .collect(),
    );
    input.project(move || {
        let cols = cols.clone();
        move |batch: RecordBatch| {
            let out = batch.project(&cols)?;
            let fields: Vec<Field> = out
                .schema()
                .fields()
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    if i < key_count {
                        f.as_ref().clone()
                    } else {
                        Field::new(format!("v{}", i - key_count), f.data_type().clone(), true)
                    }
                })
                .collect();
            Ok(RecordBatch::try_new(
                Arc::new(Schema::new(fields)),
                out.columns().to_vec(),
            )?)
        }
    })
}

/// Identifies aggregates that fold to the same value, so [`build_group_by_operator`] scatters
/// and merges each once: a `COUNT(col)` over a NULL-free column is `COUNT(*)`
/// (both count every row), so it collapses into [`CountStar`](FoldKey::CountStar);
/// over a nullable column it counts only the non-NULL rows, so it folds per
/// column. A SUM/MIN/MAX of a given column is identical to another of the same
/// column either way (both skip the same NULLs).
#[derive(PartialEq)]
enum FoldKey {
    CountStar,
    Count(usize),
    Sum(usize),
    Min(usize),
    Max(usize),
}

fn fold_key(e: &Expression, nullability: &[bool]) -> Result<FoldKey, Error> {
    let Expression::AggregateFunc(func) = e else {
        return Err(Error::UnsupportedAggregateExpression(e.clone()));
    };
    Ok(match func {
        AggregateFunc::CountStar(_) => FoldKey::CountStar,
        AggregateFunc::Count(a) => {
            let column = a.column().column_idx;
            if column_nullable(nullability, column) {
                FoldKey::Count(column)
            } else {
                FoldKey::CountStar
            }
        }
        AggregateFunc::Sum(a) => FoldKey::Sum(a.column().column_idx),
        AggregateFunc::Min(a) => FoldKey::Min(a.column().column_idx),
        AggregateFunc::Max(a) => FoldKey::Max(a.column().column_idx),
        _ => return Err(Error::UnsupportedAggregateExpression(e.clone())),
    })
}

/// Whether `column` can hold NULLs, conservatively `true` when the vector does
/// not cover it (a synthesised reference into an intermediate output).
pub(super) fn column_nullable(nullability: &[bool], column: usize) -> bool {
    nullability.get(column).copied().unwrap_or(true)
}

/// The canonical column type a computed group key or aggregate argument is
/// materialised as. Strings and temporal values keep their own type (a temporal
/// key/value flows through the same int-ify/restore path a plain temporal column
/// does, so its output is restored to `Date`/`Timestamp` rather than left a raw
/// int); integer widths are widened to the `Int64` the numeric readers consume.
/// Any other result type has no materialised column, so the caller rejects it
/// rather than silently coercing it.
fn canonical_input_type(result_type: &Type) -> Option<Type> {
    match result_type {
        Type::Utf8 => Some(Type::Utf8),
        Type::Date => Some(Type::Date),
        Type::Timestamp => Some(Type::Timestamp),
        Type::TimestampTz => Some(Type::TimestampTz),
        Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64 => Some(Type::Int64),
        // A computed float aggregate argument (`SUM(a * b)`) materialises as the
        // Float64 the float readers consume.
        Type::Float32 | Type::Float64 => Some(Type::Float64),
        // A computed decimal keeps its exact declared shape: its raw unscaled
        // values are read as-is at their carrier width, and any
        // decimal-to-decimal cast would rescale them.
        Type::Decimal { .. } => Some(result_type.clone()),
        _ => None,
    }
}

/// Evaluate each `expr` per batch, cast it to the physical arrow type of its
/// canonical [`canonical_input_type`] (`Utf8View`/`Int64`/`Date32`/`Timestamp`),
/// and prepend them as leading columns `k0, k1, ...`. The original columns follow,
/// shifted right by `exprs.len()`, so the aggregates can still read their value
/// columns.
fn project_leading_columns(
    input: RecordBatchOperatorSpec,
    exprs: &[&Expression],
    types: &[Type],
) -> Result<RecordBatchOperatorSpec, Error> {
    let builders: Arc<Vec<ExprFn>> = Arc::new(
        exprs
            .iter()
            .map(|k| k.compile())
            .collect::<Result<_, _>>()?,
    );
    let targets: Arc<Vec<DataType>> = Arc::new(
        types
            .iter()
            .map(crate::types::physical_arrow_type)
            .collect(),
    );
    Ok(input.project(move || {
        let mut evals: Vec<ExprEvalFn> = builders.iter().map(|b| b()).collect();
        let targets = targets.clone();
        move |batch: RecordBatch| {
            let width = evals.len() + batch.num_columns();
            let mut fields: Vec<Field> = Vec::with_capacity(width);
            let mut columns: Vec<ArrayRef> = Vec::with_capacity(width);
            for (i, eval) in evals.iter_mut().enumerate() {
                // A constant column (e.g. `GROUP BY 1`) is broadcast to the batch
                // length so it can sit beside the per-row columns.
                let arr = eval(&batch).into_array(batch.num_rows());
                let casted = arrow::compute::cast(&arr, &targets[i])?;
                fields.push(Field::new(format!("k{i}"), targets[i].clone(), true));
                columns.push(casted);
            }
            for (field, col) in batch.schema().fields().iter().zip(batch.columns()) {
                fields.push(field.as_ref().clone());
                columns.push(col.clone());
            }
            Ok(RecordBatch::try_new(
                Arc::new(Schema::new(fields)),
                columns,
            )?)
        }
    }))
}

/// Slot information needed to select a compiled value specialization.
///
/// COUNT and COUNT(*) share one signature because they use the same accumulator
/// operation; a COUNT over a nullable column is its own shape, since it must
/// consult the column's validity. SUM retains its input type so the reader type
/// can be selected.
pub(super) enum AggregationSignature {
    Count,
    /// `COUNT(col)` over a nullable column: counts only the non-NULL rows.
    CountNullable,
    Sum(Type),
    /// Any operation without a compiled specialization.
    Other,
}

fn aggregation_signatures(
    expressions: &[Expression],
    nullability: &[bool],
) -> Vec<AggregationSignature> {
    expressions
        .iter()
        .map(|expression| match expression {
            Expression::AggregateFunc(AggregateFunc::Sum(argument)) => {
                AggregationSignature::Sum(argument.column().return_type.clone())
            }
            Expression::AggregateFunc(AggregateFunc::CountStar(_)) => AggregationSignature::Count,
            Expression::AggregateFunc(AggregateFunc::Count(argument)) => {
                if column_nullable(nullability, argument.column().column_idx) {
                    AggregationSignature::CountNullable
                } else {
                    AggregationSignature::Count
                }
            }
            _ => AggregationSignature::Other,
        })
        .collect()
}

/// Returns supported integer pairs for the packed two-key extractor.
fn int_pair_keys(keys: &[(usize, Type)]) -> Option<(Type, Type)> {
    let [(_, a), (_, b)] = keys else { return None };
    let pair = (a.clone(), b.clone());
    matches!(
        pair,
        (Type::Int64, Type::Int32)
            | (Type::Int32, Type::Int32)
            | (Type::Int16, Type::Int32)
            | (Type::Int16, Type::Int16)
            | (Type::Int64, Type::Int64)
            | (Type::Int32, Type::Int64)
    )
    .then_some(pair)
}

/// Selects a key extractor and invokes a continuation with its concrete type.
///
/// A macro continuation is required because a selected Rust type cannot be
/// returned as a runtime value.
macro_rules! select_key_extractor {
    ($keys:expr, $keys_nullable:expr, $with_key:ident) => {{
        // The dedicated extractors read key values with no notion of validity,
        // so they are only sound over NULL-free key columns; nullable keys take
        // the row encoder below, whose blobs carry per-field validity.
        if $keys_nullable.iter().all(|&nullable| !nullable) {
            // Use native extractors for supported single-column keys.
            if let [(_, ty)] = $keys {
                match ty {
                    Type::Int8 => return $with_key!(IntKeyExtractor<Int8Type>, ()),
                    Type::Int16 => return $with_key!(IntKeyExtractor<Int16Type>, ()),
                    Type::Int32 => return $with_key!(IntKeyExtractor<Int32Type>, ()),
                    Type::Int64 => return $with_key!(IntKeyExtractor<Int64Type>, ()),
                    // A decimal keys on its raw unscaled integer at its carrier
                    // width (equal values share a scale, so raw equality is value
                    // equality); the emitted key column gets its declared
                    // precision/scale restamped on by the aggregate's output
                    // restore.
                    Type::Decimal { precision, .. } if *precision <= MAX_DECIMAL64_PRECISION => {
                        return $with_key!(IntKeyExtractor<Decimal64Type>, ());
                    }
                    Type::Decimal { .. } => return $with_key!(IntKeyExtractor<Decimal128Type>, ()),
                    Type::Utf8 => return $with_key!(StringKeyExtractor, ()),
                    // Other single-key types fall through to the row encoder below.
                    _ => {}
                }
            }

            // Pack supported integer pairs into one u128 key.
            if let Some(pair) = int_pair_keys($keys) {
                return match pair {
                    (Type::Int64, Type::Int32) => $with_key!(IntPairKeyExtractor<Int64Type, Int32Type>, ()),
                    (Type::Int32, Type::Int32) => $with_key!(IntPairKeyExtractor<Int32Type, Int32Type>, ()),
                    (Type::Int16, Type::Int32) => $with_key!(IntPairKeyExtractor<Int16Type, Int32Type>, ()),
                    (Type::Int16, Type::Int16) => $with_key!(IntPairKeyExtractor<Int16Type, Int16Type>, ()),
                    (Type::Int64, Type::Int64) => $with_key!(IntPairKeyExtractor<Int64Type, Int64Type>, ()),
                    (Type::Int32, Type::Int64) => $with_key!(IntPairKeyExtractor<Int32Type, Int64Type>, ()),
                    _ => unreachable!("int_pair_keys only returns the arms above"),
                };
            }

            // Keep an integer and string in their native representations. The
            // extractor is integer-first only; a string-first key pair was
            // already reordered by [`reorder_string_first_keys`] before the
            // cascade runs.
            if let [(_, int_ty), (_, Type::Utf8)] = $keys {
                match int_ty {
                    Type::Int8 => return $with_key!(IntStrKeyExtractor<Int8Type>, ()),
                    Type::Int16 => return $with_key!(IntStrKeyExtractor<Int16Type>, ()),
                    Type::Int32 => return $with_key!(IntStrKeyExtractor<Int32Type>, ()),
                    Type::Int64 => return $with_key!(IntStrKeyExtractor<Int64Type>, ()),
                    // Any other pairing falls through to the row encoder below.
                    _ => {}
                }
            }
        }

        // The general fallback: encode the whole key tuple into one byte blob.
        // Handles a single non-int/string key, 3+ keys, mixed types, and any
        // nullable key column.
        let schema = match row_key_schema(
            $keys
                .iter()
                .zip($keys_nullable.iter())
                .map(|((_, t), &nullable)| (t, nullable)),
        ) {
            Ok(schema) => schema,
            Err(unsupported) => {
                return Err(Error::DataTypeNotSupportedForGroupBy {
                    column: None,
                    data_type: unsupported,
                });
            }
        };
        $with_key!(RowKeyExtractor, schema)
    }};
}

/// Selects concrete key and value containers, then builds GROUP BY.
///
/// Selected signatures use [`Compiled`]. All others use [`Dynamic`], with
/// `i128` cells when strings or wide sums require them.
pub(super) fn dispatch_group_by(
    input: RecordBatchOperatorSpec,
    keys: &[(usize, Type)],
    slots: Vec<AggregationSlot>,
    signatures: &[AggregationSignature],
    requires_wide_cells: bool,
    output_limit: Option<GroupLimit>,
    nullability: &[bool],
) -> Result<RecordBatchOperatorSpec, Error> {
    if let Some(reordered) = reorder_string_first_keys(keys, nullability) {
        let grouped = select_group_by_types(
            input,
            &reordered,
            slots,
            signatures,
            requires_wide_cells,
            output_limit,
            nullability,
        )?;
        return Ok(swap_leading_key_columns(grouped));
    }
    select_group_by_types(
        input,
        keys,
        slots,
        signatures,
        requires_wide_cells,
        output_limit,
        nullability,
    )
}

/// A `GROUP BY (string, integer)` runs the group operator integer-first, so
/// [`IntStrKeyExtractor`] exists in one column order. Returns the keys
/// reordered for the operator when that applies: exactly one non-nullable
/// string key followed by one non-nullable extractor-supported integer key.
/// The caller swaps the operator's two leading output columns back with
/// [`swap_leading_key_columns`].
fn reorder_string_first_keys(
    keys: &[(usize, Type)],
    nullability: &[bool],
) -> Option<[(usize, Type); 2]> {
    let [(str_column, Type::Utf8), (int_column, int_ty)] = keys else {
        return None;
    };
    if !matches!(int_ty, Type::Int8 | Type::Int16 | Type::Int32 | Type::Int64) {
        return None;
    }
    if column_nullable(nullability, *str_column) || column_nullable(nullability, *int_column) {
        return None;
    }
    Some([(*int_column, int_ty.clone()), (*str_column, Type::Utf8)])
}

/// Swap the two leading key columns a group operator emitted for keys that
/// [`reorder_string_first_keys`] reordered, restoring the query's declared key
/// order. A per-batch column permutation; the arrays are shared, not copied.
fn swap_leading_key_columns(op: RecordBatchOperatorSpec) -> RecordBatchOperatorSpec {
    op.project(|| {
        |batch: RecordBatch| {
            let mut fields: Vec<_> = batch.schema().fields().iter().cloned().collect();
            fields.swap(0, 1);
            let mut columns = batch.columns().to_vec();
            columns.swap(0, 1);
            Ok(RecordBatch::try_new(
                Arc::new(Schema::new(fields)),
                columns,
            )?)
        }
    })
}

fn select_group_by_types(
    input: RecordBatchOperatorSpec,
    keys: &[(usize, Type)],
    slots: Vec<AggregationSlot>,
    signatures: &[AggregationSignature],
    requires_wide_cells: bool,
    output_limit: Option<GroupLimit>,
    nullability: &[bool],
) -> Result<RecordBatchOperatorSpec, Error> {
    let key_columns: Vec<usize> = keys.iter().map(|(column, _)| *column).collect();
    let keys_nullable: Vec<bool> = keys
        .iter()
        .map(|(column, _)| column_nullable(nullability, *column))
        .collect();

    // COUNT and integer SUM can use the addition-only specialization. Float
    // cells store raw bits, so they must use operation-aware merging.
    let only_additive = slots.iter().all(|slot| {
        matches!(
            slot.kind,
            AggregationKind::CountStar | AggregationKind::Count | AggregationKind::Sum
        ) && !slot.output_type.is_floating()
    });

    // Whether any aggregated column can hold NULLs. When none can, the value
    // containers instantiate with the zero-sized `()` seen mask: no per-group
    // mask storage and no per-row validity checks, so a NULL-free query's
    // group entries and fold loops match a mask-less build exactly.
    // `COUNT(*)` reads no column, so its placeholder column is exempt.
    let values_nullable = slots.iter().any(|slot| {
        !matches!(slot.kind, AggregationKind::CountStar)
            && column_nullable(nullability, slot.column)
    });
    // A tracking `Dynamic` keeps the group's seen bits in one u64 mask cell.
    if values_nullable && slots.len() > 64 {
        return Err(Error::UnsupportedAggregateExpressionAmount(slots.len()));
    }

    // Both type selections meet at this operator-construction leaf.
    macro_rules! build_group_by {
        ($K:ty, $V:ty, $key_config:expr) => {
            Ok(input.group_by_aggregate::<$K, $V>(key_columns, slots, output_limit, $key_config))
        };
    }
    // Dynamic covers every slot count and supported operation mix. The seen
    // mask instantiates `u8` only when an aggregated column can hold NULLs.
    macro_rules! dynamic {
        ($K:ty, $key_config:expr) => {
            match (requires_wide_cells, only_additive, values_nullable) {
                _ if slots.is_empty() => Err(Error::UnsupportedAggregateExpressionAmount(0)),
                (true, true, false) => build_group_by!($K, Dynamic<i128, true>, $key_config),
                (true, false, false) => build_group_by!($K, Dynamic<i128, false>, $key_config),
                (false, true, false) => build_group_by!($K, Dynamic<i64, true>, $key_config),
                (false, false, false) => build_group_by!($K, Dynamic<i64, false>, $key_config),
                (true, true, true) => build_group_by!($K, Dynamic<i128, true, u8>, $key_config),
                (true, false, true) => build_group_by!($K, Dynamic<i128, false, u8>, $key_config),
                (false, true, true) => build_group_by!($K, Dynamic<i64, true, u8>, $key_config),
                (false, false, true) => build_group_by!($K, Dynamic<i64, false, u8>, $key_config),
            }
        };
    }
    // Keep the compiled list small and route every other signature to Dynamic.
    macro_rules! select_value {
        ($K:ty, $key_config:expr) => {
            match signatures {
                // A lone count over no column or a proven NULL-free column.
                [AggregationSignature::Count] => {
                    build_group_by!($K, Compiled<(CountSlot,)>, $key_config)
                }
                // `COUNT(col)` over a nullable column: the same branch-free
                // entry, reading only the column's validity. The `u8` mask
                // keeps the validity checks live (a count's output is still
                // never NULL, so its bits are always set).
                [AggregationSignature::CountNullable] => {
                    build_group_by!($K, Compiled<(CountValidSlot,), u8>, $key_config)
                }
                // Common narrow signature produced when AVG shares a count.
                [
                    AggregationSignature::Count,
                    AggregationSignature::Sum(Type::Int16),
                    AggregationSignature::Sum(Type::Int16),
                ] if !values_nullable => build_group_by!(
                    $K,
                    Compiled<(CountSlot, SumSlot<Int16Type>, SumSlot<Int16Type>)>,
                    $key_config
                ),
                [
                    AggregationSignature::Count,
                    AggregationSignature::Sum(Type::Int16),
                    AggregationSignature::Sum(Type::Int16),
                ] => build_group_by!(
                    $K,
                    Compiled<(CountSlot, SumSlot<Int16Type>, SumSlot<Int16Type>), u8>,
                    $key_config
                ),
                _ => dynamic!($K, $key_config),
            }
        };
    }

    // Key selection invokes value selection as its continuation.
    select_key_extractor!(keys, keys_nullable, select_value)
}

/// Lower a keys-only GROUP BY that dedups the `keys` tuple and emits each distinct
/// key (no aggregates). Shares the [`select_key_extractor!`] extractor cascade with
/// [`dispatch_group_by`], so it supports every group-key shape the aggregating path
/// does. Used as the inner level of the two-level `COUNT(DISTINCT)` lowerings,
/// which then count the deduped rows per group.
pub(in crate::operator) fn build_dedup_operator(
    input: RecordBatchOperatorSpec,
    keys: &[(usize, Type)],
    nullability: &[bool],
) -> Result<RecordBatchOperatorSpec, Error> {
    if let Some(reordered) = reorder_string_first_keys(keys, nullability) {
        let deduped = select_dedup_types(input, &reordered, nullability)?;
        return Ok(swap_leading_key_columns(deduped));
    }
    select_dedup_types(input, keys, nullability)
}

fn select_dedup_types(
    input: RecordBatchOperatorSpec,
    keys: &[(usize, Type)],
    nullability: &[bool],
) -> Result<RecordBatchOperatorSpec, Error> {
    let key_columns: Vec<usize> = keys.iter().map(|(column, _)| *column).collect();
    let keys_nullable: Vec<bool> = keys
        .iter()
        .map(|(column, _)| column_nullable(nullability, *column))
        .collect();
    // Keys-only: `Distinct` holds no accumulator, so the slot list is empty and the
    // group emits the key columns themselves.
    macro_rules! emit_dedup {
        ($K:ty, $key_config:expr) => {
            Ok(
                input.group_by_aggregate::<$K, Distinct>(
                    key_columns,
                    Vec::new(),
                    None,
                    $key_config,
                ),
            )
        };
    }
    select_key_extractor!(keys, keys_nullable, emit_dedup)
}

/// Remap resolved group `keys` onto the leading output columns `0..n` of a
/// preceding group-by (which emits its key columns in order), keeping each key's
/// type. Shared by the two-level `COUNT(DISTINCT)` lowerings to group the deduped
/// inner output by its leading group columns.
pub(super) fn derive_outer_keys(groups: &[(usize, Type)]) -> Vec<(usize, Type)> {
    groups
        .iter()
        .enumerate()
        .map(|(col, (_, ty))| (col, ty.clone()))
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use crate::types::Type;
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Date32Type, Float64Type, Int64Type};
    use arrow_array::{ArrayRef, Float32Array, Float64Array, Int32Array, RecordBatch};
    use arrow_schema::DataType;
    use rstest::rstest;
    use std::sync::Arc;

    #[rstest]
    fn grouped_aggregates_over_double_are_float64(mut testing_planner: TestingPlanner) {
        testing_planner.add_table(
            "metrics",
            &[
                (
                    "g",
                    Type::Int32,
                    Arc::new(Int32Array::from(vec![1, 1, 2])) as ArrayRef,
                ),
                (
                    "v",
                    Type::Float64,
                    Arc::new(Float64Array::from(vec![1.5, 0.5, 4.0])) as ArrayRef,
                ),
            ],
        );

        let batches = run_batches(
            &mut testing_planner,
            "SELECT g, SUM(v), MIN(v), MAX(v) FROM metrics GROUP BY g ORDER BY g",
        );

        let schema = batches[0].schema();
        assert_eq!(schema.field(1).data_type(), &DataType::Float64); // SUM
        assert_eq!(schema.field(2).data_type(), &DataType::Float64); // MIN
        assert_eq!(schema.field(3).data_type(), &DataType::Float64); // MAX
        let sum = batches[0].column(1).as_primitive::<Float64Type>();
        assert_eq!(sum.value(0), 2.0); // group 1: 1.5 + 0.5
        assert_eq!(sum.value(1), 4.0); // group 2
    }

    #[rstest]
    fn grouped_sum_over_real_widens_min_stays_real(mut testing_planner: TestingPlanner) {
        testing_planner.add_table(
            "reals",
            &[
                (
                    "g",
                    Type::Int32,
                    Arc::new(Int32Array::from(vec![1, 1, 2])) as ArrayRef,
                ),
                (
                    "v",
                    Type::Float32,
                    Arc::new(Float32Array::from(vec![1.5f32, 2.5, 4.0])) as ArrayRef,
                ),
            ],
        );

        let batches = run_batches(
            &mut testing_planner,
            "SELECT g, SUM(v), MIN(v) FROM reals GROUP BY g ORDER BY g",
        );

        let schema = batches[0].schema();
        assert_eq!(schema.field(1).data_type(), &DataType::Float64); // SUM(REAL) widens to DOUBLE
        assert_eq!(schema.field(2).data_type(), &DataType::Float32); // MIN(REAL) stays REAL
    }

    #[rstest]
    fn grouped_mixed_int64_and_double_sum(mut testing_planner: TestingPlanner) {
        // A wide (i64) integer SUM forces the i128 cell; the float SUM rides the
        // same cell with its f64 bits bit-punned in. Each renders its own type.
        use arrow_array::Int64Array;
        use arrow_array::types::Decimal128Type;
        testing_planner.add_table(
            "mixed",
            &[
                (
                    "g",
                    Type::Int32,
                    Arc::new(Int32Array::from(vec![1, 1, 2])) as ArrayRef,
                ),
                (
                    "i",
                    Type::Int64,
                    Arc::new(Int64Array::from(vec![10i64, 20, 7])) as ArrayRef,
                ),
                (
                    "d",
                    Type::Float64,
                    Arc::new(Float64Array::from(vec![1.5, 2.5, 4.0])) as ArrayRef,
                ),
            ],
        );

        let batches = run_batches(
            &mut testing_planner,
            "SELECT g, SUM(i), SUM(d) FROM mixed GROUP BY g ORDER BY g",
        );

        let schema = batches[0].schema();
        assert_eq!(schema.field(1).data_type(), &DataType::Decimal128(38, 0)); // SUM(int)
        assert_eq!(schema.field(2).data_type(), &DataType::Float64); // SUM(double)
        assert_eq!(
            batches[0]
                .column(1)
                .as_primitive::<Decimal128Type>()
                .value(0),
            30 // group 1: 10 + 20
        );
        assert_eq!(
            batches[0].column(2).as_primitive::<Float64Type>().value(0),
            4.0
        ); // 1.5 + 2.5
    }

    #[rstest]
    fn global_avg_over_double_is_float64(mut testing_planner: TestingPlanner) {
        testing_planner.add_table(
            "avgd",
            &[(
                "v",
                Type::Float64,
                Arc::new(Float64Array::from(vec![2.0, 4.0])) as ArrayRef,
            )],
        );

        let batches = run_batches(&mut testing_planner, "SELECT AVG(v) FROM avgd");

        assert_eq!(batches[0].column(0).data_type(), &DataType::Float64);
        assert_eq!(
            batches[0].column(0).as_primitive::<Float64Type>().value(0),
            3.0
        );
    }

    /// A `Decimal64` column (the carrier for a declared precision up to 18).
    fn decimal_col(values: Vec<i64>, precision: u8, scale: i8) -> ArrayRef {
        use arrow_array::Decimal64Array;
        Arc::new(
            Decimal64Array::from(values)
                .with_precision_and_scale(precision, scale)
                .unwrap(),
        )
    }

    /// A `Decimal128` column (the carrier for a declared precision of 19+).
    fn wide_decimal_col(values: Vec<i128>, precision: u8, scale: i8) -> ArrayRef {
        use arrow_array::Decimal128Array;
        Arc::new(
            Decimal128Array::from(values)
                .with_precision_and_scale(precision, scale)
                .unwrap(),
        )
    }

    /// g [1, 1, 2] alongside price DECIMAL(10,2) [10.50, 2.50, 4.00].
    fn add_sales_table(testing_planner: &TestingPlanner) {
        testing_planner.add_table(
            "sales",
            &[
                (
                    "g",
                    Type::Int32,
                    Arc::new(Int32Array::from(vec![1, 1, 2])) as ArrayRef,
                ),
                (
                    "price",
                    Type::Decimal {
                        precision: 10,
                        scale: 2,
                    },
                    decimal_col(vec![1050, 250, 400], 10, 2),
                ),
            ],
        );
    }

    fn col_decimal(batches: &[RecordBatch], col: usize, row: usize) -> i128 {
        use arrow_array::types::Decimal128Type;
        batches[0]
            .column(col)
            .as_primitive::<Decimal128Type>()
            .value(row)
    }

    fn col_decimal64(batches: &[RecordBatch], col: usize, row: usize) -> i64 {
        use arrow_array::types::Decimal64Type;
        batches[0]
            .column(col)
            .as_primitive::<Decimal64Type>()
            .value(row)
    }

    #[rstest]
    fn grouped_sum_min_max_over_decimal(mut testing_planner: TestingPlanner) {
        add_sales_table(&testing_planner);

        let batches = run_batches(
            &mut testing_planner,
            "SELECT g, SUM(price), MIN(price), MAX(price) FROM sales GROUP BY g ORDER BY g",
        );

        // SUM(DECIMAL(10,2)) is DECIMAL(38,2) (a Decimal128 column); MIN/MAX
        // keep the input's Decimal64(10,2). The unscaled values must come
        // through untouched (17.00 sums to raw 1700).
        let schema = batches[0].schema();
        assert_eq!(schema.field(1).data_type(), &DataType::Decimal128(38, 2));
        assert_eq!(schema.field(2).data_type(), &DataType::Decimal64(10, 2));
        assert_eq!(schema.field(3).data_type(), &DataType::Decimal64(10, 2));
        assert_eq!(col_decimal(&batches, 1, 0), 1300); // 10.50 + 2.50
        assert_eq!(col_decimal64(&batches, 2, 0), 250); // MIN of group 1
        assert_eq!(col_decimal64(&batches, 3, 0), 1050); // MAX of group 1
        assert_eq!(col_decimal(&batches, 1, 1), 400); // group 2
    }

    #[rstest]
    fn global_sum_min_max_over_decimal(mut testing_planner: TestingPlanner) {
        add_sales_table(&testing_planner);

        let batches = run_batches(
            &mut testing_planner,
            "SELECT SUM(price), MIN(price), MAX(price) FROM sales",
        );

        let schema = batches[0].schema();
        assert_eq!(schema.field(0).data_type(), &DataType::Decimal128(38, 2));
        assert_eq!(schema.field(1).data_type(), &DataType::Decimal64(10, 2));
        assert_eq!(schema.field(2).data_type(), &DataType::Decimal64(10, 2));
        assert_eq!(col_decimal(&batches, 0, 0), 1700); // 17.00
        assert_eq!(col_decimal64(&batches, 1, 0), 250); // 2.50
        assert_eq!(col_decimal64(&batches, 2, 0), 1050); // 10.50
    }

    #[rstest]
    fn grouped_sum_min_max_over_wide_decimal(mut testing_planner: TestingPlanner) {
        // A declared precision of 20 rides the Decimal128 carrier end to end;
        // the raw values are chosen past i64::MAX so only a real i128 path can
        // hold them.
        let huge: i128 = 15_000_000_000_000_000_000;
        testing_planner.add_table(
            "big_sales",
            &[
                (
                    "g",
                    Type::Int32,
                    Arc::new(Int32Array::from(vec![1, 1, 2])) as ArrayRef,
                ),
                (
                    "price",
                    Type::Decimal {
                        precision: 20,
                        scale: 2,
                    },
                    wide_decimal_col(vec![huge, 5_000_000_000_000_000_000, 400], 20, 2),
                ),
            ],
        );

        let batches = run_batches(
            &mut testing_planner,
            "SELECT g, SUM(price), MIN(price), MAX(price) FROM big_sales GROUP BY g ORDER BY g",
        );

        let schema = batches[0].schema();
        assert_eq!(schema.field(1).data_type(), &DataType::Decimal128(38, 2));
        assert_eq!(schema.field(2).data_type(), &DataType::Decimal128(20, 2));
        assert_eq!(schema.field(3).data_type(), &DataType::Decimal128(20, 2));
        assert_eq!(col_decimal(&batches, 1, 0), 20_000_000_000_000_000_000); // past i64::MAX
        assert_eq!(col_decimal(&batches, 2, 0), 5_000_000_000_000_000_000);
        assert_eq!(col_decimal(&batches, 3, 0), huge);
        assert_eq!(col_decimal(&batches, 1, 1), 400); // group 2
    }

    #[rstest]
    fn group_by_wide_decimal_key(mut testing_planner: TestingPlanner) {
        let huge: i128 = 15_000_000_000_000_000_000;
        testing_planner.add_table(
            "big_orders",
            &[(
                "price",
                Type::Decimal {
                    precision: 20,
                    scale: 2,
                },
                wide_decimal_col(vec![huge, 400, huge], 20, 2),
            )],
        );

        let batches = run_batches(
            &mut testing_planner,
            "SELECT price, count(*) FROM big_orders GROUP BY price ORDER BY price",
        );

        // The key column keeps the declared Decimal128(20,2) with its raw i128
        // keys restamped, never rescaled.
        let schema = batches[0].schema();
        assert_eq!(schema.field(0).data_type(), &DataType::Decimal128(20, 2));
        assert_eq!(col_decimal(&batches, 0, 0), 400);
        assert_eq!(col_decimal(&batches, 0, 1), huge);
        let counts = batches[0].column(1).as_primitive::<Int64Type>();
        assert_eq!(counts.value(0), 1);
        assert_eq!(counts.value(1), 2);
    }

    #[rstest]
    fn avg_over_decimal_is_double(mut testing_planner: TestingPlanner) {
        add_sales_table(&testing_planner);

        let batches = run_batches(&mut testing_planner, "SELECT AVG(price) FROM sales");

        // AVG lowers to sum/count with a float divide; the DECIMAL(38,2) sum is
        // cast to DOUBLE (a real division by 10^2), so the average is 17.00 / 3.
        assert_eq!(batches[0].column(0).data_type(), &DataType::Float64);
        let avg = batches[0].column(0).as_primitive::<Float64Type>().value(0);
        assert!((avg - 17.0 / 3.0).abs() < 1e-12);
    }

    #[rstest]
    fn grouped_avg_over_decimal_is_double(mut testing_planner: TestingPlanner) {
        add_sales_table(&testing_planner);

        let batches = run_batches(
            &mut testing_planner,
            "SELECT g, AVG(price) FROM sales GROUP BY g ORDER BY g",
        );

        assert_eq!(batches[0].column(1).data_type(), &DataType::Float64);
        let avg = batches[0].column(1).as_primitive::<Float64Type>();
        assert!((avg.value(0) - 6.5).abs() < 1e-12); // (10.50 + 2.50) / 2
        assert!((avg.value(1) - 4.0).abs() < 1e-12);
    }

    #[rstest]
    fn group_by_decimal_key_with_sum(mut testing_planner: TestingPlanner) {
        testing_planner.add_table(
            "orders",
            &[
                (
                    "price",
                    Type::Decimal {
                        precision: 10,
                        scale: 2,
                    },
                    decimal_col(vec![1050, 400, 1050], 10, 2),
                ),
                (
                    "qty",
                    Type::Int32,
                    Arc::new(Int32Array::from(vec![2, 5, 3])) as ArrayRef,
                ),
            ],
        );

        let batches = run_batches(
            &mut testing_planner,
            "SELECT price, SUM(qty) FROM orders GROUP BY price ORDER BY price",
        );

        // The key column must carry the declared DECIMAL(10,2) shape (a restamp
        // of the raw keys, never a rescale) and the sums fold per distinct key.
        use arrow_array::types::Decimal64Type;
        let schema = batches[0].schema();
        assert_eq!(schema.field(0).data_type(), &DataType::Decimal64(10, 2));
        let keys = batches[0].column(0).as_primitive::<Decimal64Type>();
        assert_eq!(keys.value(0), 400);
        assert_eq!(keys.value(1), 1050);
        assert_eq!(col_decimal(&batches, 1, 0), 5); // SUM(qty) for 4.00
        assert_eq!(col_decimal(&batches, 1, 1), 5); // 2 + 3 for 10.50
    }

    #[rstest]
    fn group_by_decimal_and_int_keys(mut testing_planner: TestingPlanner) {
        // A decimal beside an int takes the row-encoded key path.
        testing_planner.add_table(
            "orders",
            &[
                (
                    "price",
                    Type::Decimal {
                        precision: 10,
                        scale: 2,
                    },
                    decimal_col(vec![1050, 1050, 400], 10, 2),
                ),
                (
                    "g",
                    Type::Int32,
                    Arc::new(Int32Array::from(vec![1, 1, 2])) as ArrayRef,
                ),
            ],
        );

        let batches = run_batches(
            &mut testing_planner,
            "SELECT price, g, count(*) FROM orders GROUP BY price, g ORDER BY price",
        );

        use arrow_array::types::Decimal64Type;
        let schema = batches[0].schema();
        assert_eq!(schema.field(0).data_type(), &DataType::Decimal64(10, 2));
        let keys = batches[0].column(0).as_primitive::<Decimal64Type>();
        assert_eq!(keys.value(0), 400);
        assert_eq!(keys.value(1), 1050);
        let counts = batches[0].column(2).as_primitive::<Int64Type>();
        assert_eq!(counts.value(0), 1);
        assert_eq!(counts.value(1), 2);
    }

    #[rstest]
    fn sum_of_decimal_arithmetic(mut testing_planner: TestingPlanner) {
        add_sales_table(&testing_planner);

        let batches = run_batches(&mut testing_planner, "SELECT SUM(price * 2) FROM sales");

        // price * 2 keeps scale 2, so the sum's raw value is 2 * 1700 = 34.00.
        assert!(matches!(
            batches[0].column(0).data_type(),
            DataType::Decimal128(_, 2)
        ));
        assert_eq!(col_decimal(&batches, 0, 0), 3400);
    }

    #[rstest]
    fn filtered_sum_over_decimal(mut testing_planner: TestingPlanner) {
        add_sales_table(&testing_planner);

        let batches = run_batches(
            &mut testing_planner,
            "SELECT SUM(price) FROM sales WHERE price > 9.99",
        );

        assert_eq!(col_decimal(&batches, 0, 0), 1050); // only 10.50 passes
    }

    #[rstest]
    fn order_by_decimal_limit(mut testing_planner: TestingPlanner) {
        add_sales_table(&testing_planner);

        let batches = run_batches(
            &mut testing_planner,
            "SELECT price FROM sales ORDER BY price DESC LIMIT 2",
        );

        let rows: Vec<i64> = batches
            .iter()
            .flat_map(|b| {
                use arrow_array::types::Decimal64Type;
                let col = b.column(0).as_primitive::<Decimal64Type>();
                (0..b.num_rows()).map(|i| col.value(i)).collect::<Vec<_>>()
            })
            .collect();
        assert_eq!(rows, vec![1050, 400]);
    }

    #[rstest]
    fn group_by_date_emits_dates(mut testing_planner: TestingPlanner) {
        testing_planner.add_table(
            "events",
            &[(
                "d",
                Type::Date,
                Arc::new(Int32Array::from(vec![0, 7, 0])) as ArrayRef,
            )],
        );

        let batches = run_batches(
            &mut testing_planner,
            "SELECT d, count(*) FROM events GROUP BY d",
        );

        // The group-by computes on the int key but the emitted key column must be
        // restored to a real Date32 so it renders as a date.
        assert_eq!(batches[0].schema().field(0).data_type(), &DataType::Date32);
    }

    #[rstest]
    fn group_by_computed_timestamp_emits_a_timestamp(mut testing_planner: TestingPlanner) {
        use arrow_array::TimestampMicrosecondArray;
        use arrow_schema::TimeUnit;

        testing_planner.add_table(
            "events",
            &[(
                "ts",
                Type::Timestamp,
                Arc::new(TimestampMicrosecondArray::from(vec![
                    0i64,
                    3_600_000_000,
                    90_000_000_000,
                ])) as ArrayRef,
            )],
        );

        // A computed temporal group key is materialised into a leading column;
        // it must still be restored to its TIMESTAMP type on output, not left
        // the raw epoch-microseconds int the group-by computes on.
        let batches = run_batches(
            &mut testing_planner,
            "SELECT date_trunc('day', ts) AS d, count(*) FROM events GROUP BY date_trunc('day', ts)",
        );

        assert_eq!(
            batches[0].schema().field(0).data_type(),
            &DataType::Timestamp(TimeUnit::Microsecond, None)
        );
    }

    #[rstest]
    fn grouped_min_of_date_returns_a_date(mut testing_planner: TestingPlanner) {
        testing_planner.add_table(
            "events",
            &[
                (
                    "g",
                    Type::Int32,
                    Arc::new(Int32Array::from(vec![1, 1, 2])) as ArrayRef,
                ),
                (
                    "d",
                    Type::Date,
                    Arc::new(Int32Array::from(vec![10, 3, 7])) as ArrayRef,
                ),
            ],
        );

        let batches = run_batches(
            &mut testing_planner,
            "SELECT g, MIN(d) FROM events GROUP BY g",
        );

        // Key g stays Int32; the MIN(d) value column is restored to Date32.
        assert_eq!(batches[0].schema().field(1).data_type(), &DataType::Date32);
    }

    #[rstest]
    fn global_max_of_date_from_scan_is_a_date(mut testing_planner: TestingPlanner) {
        testing_planner.add_table(
            "events",
            &[(
                "d",
                Type::Date,
                Arc::new(Int32Array::from(vec![10, 3, 7])) as ArrayRef,
            )],
        );

        // The COUNT alongside MAX keeps this off the stats peephole, so it runs
        // the scan-based global aggregate.
        let batches = run_batches(&mut testing_planner, "SELECT MAX(d), COUNT(d) FROM events");

        let col = batches[0].column(0);
        assert_eq!(col.data_type(), &DataType::Date32);
        assert_eq!(col.as_primitive::<Date32Type>().value(0), 10);
    }

    #[rstest]
    fn nine_mixed_aggregates_group_correctly(mut testing_planner: TestingPlanner) {
        testing_planner.add_table(
            "wide",
            &[
                (
                    "g",
                    Type::Int32,
                    Arc::new(Int32Array::from(vec![1, 1, 2])) as ArrayRef,
                ),
                (
                    "a",
                    Type::Int32,
                    Arc::new(Int32Array::from(vec![3, 5, 7])) as ArrayRef,
                ),
                (
                    "b",
                    Type::Float64,
                    Arc::new(Float64Array::from(vec![1.5, 2.5, 4.0])) as ArrayRef,
                ),
            ],
        );

        let batches = run_batches(
            &mut testing_planner,
            "SELECT g, COUNT(*), SUM(a), MIN(a), MAX(a), SUM(b), MIN(b), MAX(b), MIN(g), MAX(g) \
             FROM wide GROUP BY g ORDER BY g",
        );

        use arrow_array::types::{Decimal128Type, Int32Type, Int64Type};
        let batch = &batches[0];
        assert_eq!(batch.num_rows(), 2);
        let counts = batch.column(1).as_primitive::<Int64Type>();
        assert_eq!((counts.value(0), counts.value(1)), (2, 1));
        let sums = batch.column(2).as_primitive::<Decimal128Type>();
        assert_eq!((sums.value(0), sums.value(1)), (8, 7));
        let mins = batch.column(3).as_primitive::<Int32Type>();
        assert_eq!((mins.value(0), mins.value(1)), (3, 7));
        let float_sums = batch.column(5).as_primitive::<Float64Type>();
        assert_eq!((float_sums.value(0), float_sums.value(1)), (4.0, 4.0));
        let float_maxes = batch.column(7).as_primitive::<Float64Type>();
        assert_eq!((float_maxes.value(0), float_maxes.value(1)), (2.5, 4.0));
        let key_maxes = batch.column(9).as_primitive::<Int32Type>();
        assert_eq!((key_maxes.value(0), key_maxes.value(1)), (1, 2));
    }

    #[rstest]
    fn nine_additive_aggregates_group_correctly(mut testing_planner: TestingPlanner) {
        let group = Arc::new(Int32Array::from(vec![1, 1, 2])) as ArrayRef;
        let values = Arc::new(Int32Array::from(vec![3, 5, 7])) as ArrayRef;
        let columns: Vec<(&str, Type, ArrayRef)> = std::iter::once(("g", Type::Int32, group))
            .chain(
                ["c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7"]
                    .map(|name| (name, Type::Int32, values.clone())),
            )
            .collect();
        testing_planner.add_table("adds", &columns);

        let batches = run_batches(
            &mut testing_planner,
            "SELECT g, COUNT(*), SUM(c0), SUM(c1), SUM(c2), SUM(c3), SUM(c4), SUM(c5), SUM(c6), SUM(c7) \
             FROM adds GROUP BY g ORDER BY g",
        );

        use arrow_array::types::{Decimal128Type, Int64Type};
        let batch = &batches[0];
        assert_eq!(batch.num_rows(), 2);
        let counts = batch.column(1).as_primitive::<Int64Type>();
        assert_eq!((counts.value(0), counts.value(1)), (2, 1));
        for column in 2..10 {
            let sums = batch.column(column).as_primitive::<Decimal128Type>();
            assert_eq!((sums.value(0), sums.value(1)), (8, 7));
        }
    }

    #[rstest]
    fn nine_aggregates_with_string_extremes_group_correctly(mut testing_planner: TestingPlanner) {
        use arrow_array::{Int64Array, StringViewArray};
        testing_planner.add_table(
            "mixed",
            &[
                (
                    "g",
                    Type::Int32,
                    Arc::new(Int32Array::from(vec![1, 1, 2])) as ArrayRef,
                ),
                (
                    "i",
                    Type::Int64,
                    Arc::new(Int64Array::from(vec![10i64, 20, 7])) as ArrayRef,
                ),
                (
                    "s",
                    Type::Utf8,
                    Arc::new(StringViewArray::from(vec![
                        "a longer string beyond inlining",
                        "banana",
                        "cherry",
                    ])) as ArrayRef,
                ),
            ],
        );

        let batches = run_batches(
            &mut testing_planner,
            "SELECT g, COUNT(*), SUM(i), MIN(i), MAX(i), MIN(g), MAX(g), SUM(g), MIN(s), MAX(s) \
             FROM mixed GROUP BY g ORDER BY g",
        );

        use arrow_array::types::Decimal128Type;
        let batch = &batches[0];
        assert_eq!(batch.num_rows(), 2);
        let sums = batch.column(2).as_primitive::<Decimal128Type>();
        assert_eq!((sums.value(0), sums.value(1)), (30, 7));
        let string_mins = batch.column(8).as_string_view();
        assert_eq!(string_mins.value(0), "a longer string beyond inlining");
        assert_eq!(string_mins.value(1), "cherry");
        let string_maxes = batch.column(9).as_string_view();
        assert_eq!(string_maxes.value(0), "banana");
        assert_eq!(string_maxes.value(1), "cherry");
    }

    /// A string-then-integer GROUP BY runs the integer-first extractor with its
    /// keys reordered, and the output columns swapped back to the query's order.
    #[rstest]
    fn string_first_group_keys_keep_declared_column_order(mut testing_planner: TestingPlanner) {
        use arrow_array::{Int64Array, StringViewArray};
        testing_planner.add_table(
            "events",
            &[
                (
                    "name",
                    Type::Utf8,
                    Arc::new(StringViewArray::from(vec!["a", "a", "b", "a"])) as ArrayRef,
                ),
                (
                    "id",
                    Type::Int64,
                    Arc::new(Int64Array::from(vec![1i64, 2, 2, 1])) as ArrayRef,
                ),
            ],
        );

        let batches = run_batches(
            &mut testing_planner,
            "SELECT name, id, COUNT(*) FROM events GROUP BY name, id ORDER BY name, id",
        );

        let batch = &batches[0];
        let names = batch.column(0).as_string_view();
        let ids = batch.column(1).as_primitive::<Int64Type>();
        let counts = batch.column(2).as_primitive::<Int64Type>();
        let rows: Vec<(&str, i64, i64)> = (0..batch.num_rows())
            .map(|i| (names.value(i), ids.value(i), counts.value(i)))
            .collect();
        assert_eq!(rows, vec![("a", 1, 2), ("a", 2, 1), ("b", 2, 1)]);
    }

    /// A grouped `COUNT(DISTINCT int)` over a string group key dedups on a
    /// string-then-integer inner key, which takes the same reorder-and-swap
    /// path as a direct string-first GROUP BY.
    #[rstest]
    fn count_distinct_over_string_group_key(mut testing_planner: TestingPlanner) {
        use arrow_array::{Int64Array, StringViewArray};
        testing_planner.add_table(
            "visits",
            &[
                (
                    "site",
                    Type::Utf8,
                    Arc::new(StringViewArray::from(vec!["x", "x", "x", "y"])) as ArrayRef,
                ),
                (
                    "user_id",
                    Type::Int64,
                    Arc::new(Int64Array::from(vec![1i64, 1, 2, 3])) as ArrayRef,
                ),
            ],
        );

        let batches = run_batches(
            &mut testing_planner,
            "SELECT site, COUNT(DISTINCT user_id) FROM visits GROUP BY site ORDER BY site",
        );

        let batch = &batches[0];
        let sites = batch.column(0).as_string_view();
        let counts = batch.column(1).as_primitive::<Int64Type>();
        assert_eq!((sites.value(0), counts.value(0)), ("x", 2));
        assert_eq!((sites.value(1), counts.value(1)), ("y", 1));
    }

    #[rstest]
    fn nine_aggregates_over_many_groups(mut testing_planner: TestingPlanner) {
        // Enough distinct groups to grow the consume tables past their initial
        // capacity and split the merge across partitions.
        let n = 10_000i32;
        let keys = Arc::new(Int32Array::from((0..n).collect::<Vec<_>>())) as ArrayRef;
        let values = Arc::new(Int32Array::from((0..n).collect::<Vec<_>>())) as ArrayRef;
        let floats = Arc::new(Float64Array::from(
            (0..n).map(f64::from).collect::<Vec<_>>(),
        )) as ArrayRef;
        testing_planner.add_table(
            "big",
            &[
                ("g", Type::Int32, keys),
                ("a", Type::Int32, values),
                ("b", Type::Float64, floats),
            ],
        );

        let batches = run_batches(
            &mut testing_planner,
            "SELECT g, COUNT(*), SUM(a), MIN(a), MAX(a), SUM(b), MIN(b), MAX(b), MIN(g), MAX(g) \
             FROM big GROUP BY g",
        );

        use arrow_array::types::{Decimal128Type, Int64Type};
        let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(rows, n as usize);
        let total_count: i64 = batches
            .iter()
            .flat_map(|b| b.column(1).as_primitive::<Int64Type>().iter())
            .flatten()
            .sum();
        assert_eq!(total_count, n as i64);
        let total_sum: i128 = batches
            .iter()
            .flat_map(|b| b.column(2).as_primitive::<Decimal128Type>().iter())
            .flatten()
            .sum();
        assert_eq!(total_sum, i128::from(n) * i128::from(n - 1) / 2);
    }

    #[rstest]
    fn nine_aggregates_with_pushed_top_k(mut testing_planner: TestingPlanner) {
        testing_planner.add_table(
            "ranked",
            &[
                (
                    "g",
                    Type::Int32,
                    Arc::new(Int32Array::from(vec![1, 1, 2, 3, 4])) as ArrayRef,
                ),
                (
                    "a",
                    Type::Int32,
                    Arc::new(Int32Array::from(vec![5, 6, 20, 9, 1])) as ArrayRef,
                ),
                (
                    "b",
                    Type::Float64,
                    Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0])) as ArrayRef,
                ),
            ],
        );

        // ORDER BY an integer SUM slot with LIMIT pushes a per-partition top-k
        // into the group operator.
        let batches = run_batches(
            &mut testing_planner,
            "SELECT g, SUM(a), COUNT(*), MIN(a), MAX(a), SUM(b), MIN(b), MAX(b), MIN(g), MAX(g) \
             FROM ranked GROUP BY g ORDER BY SUM(a) DESC LIMIT 2",
        );

        use arrow_array::types::{Decimal128Type, Int32Type, Int64Type};
        let batch = &batches[0];
        assert_eq!(batch.num_rows(), 2);
        let keys = batch.column(0).as_primitive::<Int32Type>();
        assert_eq!((keys.value(0), keys.value(1)), (2, 1));
        let sums = batch.column(1).as_primitive::<Decimal128Type>();
        assert_eq!((sums.value(0), sums.value(1)), (20, 11));
        let counts = batch.column(2).as_primitive::<Int64Type>();
        assert_eq!((counts.value(0), counts.value(1)), (1, 2));
    }

    #[rstest]
    fn global_max_of_date_from_stats_is_a_date(mut testing_planner: TestingPlanner) {
        testing_planner.add_table(
            "events",
            &[(
                "d",
                Type::Date,
                Arc::new(Int32Array::from(vec![10, 3, 7])) as ArrayRef,
            )],
        );

        // A lone unfiltered MAX over a bare scan is answered from column_min_max
        // (no scan); it must still surface the temporal type, not a bare int.
        let batches = run_batches(&mut testing_planner, "SELECT MAX(d) FROM events");

        let col = batches[0].column(0);
        assert_eq!(col.data_type(), &DataType::Date32);
        assert_eq!(col.as_primitive::<Date32Type>().value(0), 10);
    }

    #[rstest]
    fn global_aggregates_skip_nulls(mut testing_planner: TestingPlanner) {
        let rows = run(
            &mut testing_planner,
            "SELECT COUNT(b) AS cb, COUNT(*) AS ca, SUM(b) AS sb, MIN(b) AS mn, MAX(b) AS mx \
             FROM nullable_table",
        );

        assert_eq!(rows[0]["cb"].as_i64(), Some(3));
        assert_eq!(rows[0]["ca"].as_i64(), Some(6));
        assert_eq!(rows[0]["sb"].as_i64(), Some(90));
        assert_eq!(rows[0]["mn"].as_i64(), Some(10));
        assert_eq!(rows[0]["mx"].as_i64(), Some(50));
    }

    #[rstest]
    fn global_sum_over_only_nulls_is_null(mut testing_planner: TestingPlanner) {
        let rows = run(
            &mut testing_planner,
            "SELECT SUM(b) AS sb FROM nullable_table WHERE b IS NULL",
        );

        assert!(rows[0]["sb"].is_null());
    }

    #[rstest]
    fn grouped_values_skip_nulls_and_all_null_groups_are_null(mut testing_planner: TestingPlanner) {
        let mut rows = run(
            &mut testing_planner,
            "SELECT a, SUM(b) AS sb, COUNT(b) AS cb, MIN(s) AS ms, MIN(b) AS mb \
             FROM nullable_table GROUP BY a",
        );

        rows.sort_by_key(|r| r["a"].as_i64().unwrap());
        assert_eq!(rows.len(), 6);
        assert_eq!(rows[0]["sb"].as_i64(), Some(10));
        assert_eq!(rows[0]["cb"].as_i64(), Some(1));
        assert_eq!(rows[0]["ms"].as_str(), Some("x"));
        // a = 2 saw only NULLs: the sum and extremes are NULL, the count is 0.
        assert!(rows[1]["sb"].is_null());
        assert_eq!(rows[1]["cb"].as_i64(), Some(0));
        assert!(rows[1]["ms"].is_null());
        assert!(rows[1]["mb"].is_null());
        // a = 6 has a NULL b beside a real s.
        assert!(rows[5]["sb"].is_null());
        assert_eq!(rows[5]["ms"].as_str(), Some("z"));
    }

    #[rstest]
    fn lone_grouped_count_of_nullable_column_skips_nulls(mut testing_planner: TestingPlanner) {
        // A lone COUNT(col) is the compiled validity-reading slot.
        let mut rows = run(
            &mut testing_planner,
            "SELECT a, COUNT(b) AS c FROM nullable_table GROUP BY a",
        );

        rows.sort_by_key(|r| r["a"].as_i64().unwrap());
        assert_eq!(
            rows.iter()
                .map(|r| r["c"].as_i64().unwrap())
                .collect::<Vec<_>>(),
            vec![1, 0, 1, 0, 1, 0]
        );
    }

    #[rstest]
    fn null_int_keys_group_together(mut testing_planner: TestingPlanner) {
        let rows = run(
            &mut testing_planner,
            "SELECT b, COUNT(*) AS c FROM nullable_table GROUP BY b",
        );

        assert_eq!(rows.len(), 4);
        let null_group = rows.iter().find(|r| r["b"].is_null()).unwrap();
        assert_eq!(null_group["c"].as_i64(), Some(3));
    }

    #[rstest]
    fn null_string_and_int_keys_group_together(mut testing_planner: TestingPlanner) {
        let rows = run(
            &mut testing_planner,
            "SELECT s, b, COUNT(*) AS c FROM nullable_table GROUP BY s, b",
        );

        // (x, 10), (NULL, NULL) x2, (y, 30), (x, 50), (z, NULL).
        assert_eq!(rows.len(), 5);
        let all_null = rows
            .iter()
            .find(|r| r["s"].is_null() && r["b"].is_null())
            .unwrap();
        assert_eq!(all_null["c"].as_i64(), Some(2));
        let z_group = rows.iter().find(|r| r["s"].as_str() == Some("z")).unwrap();
        assert!(z_group["b"].is_null());
        assert_eq!(z_group["c"].as_i64(), Some(1));
    }
}
