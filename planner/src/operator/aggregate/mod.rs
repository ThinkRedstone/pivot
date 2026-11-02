//! [`Aggregate`] — GROUP BY + aggregate functions.
//!
//! [`Aggregate::compile`] first lowers every computed group key and aggregate
//! argument (e.g. the `column1 * column2` in `SUM(column1 * column2)`) into a
//! leading projected column via [`Aggregate::materialize_inputs`], so the rest
//! of compilation only ever reads column indices. It then routes on the groups
//! and aggregate expressions to one of a handful of lowering strategies, each in
//! its own submodule:
//!
//! | query shape | strategy |
//! |---|---|
//! | `COUNT(DISTINCT x)` alone, no GROUP BY | [`global_distinct`] |
//! | grouped, one `COUNT(DISTINCT)` (alone or with other aggregates) | [`grouped_distinct`] |
//! | no GROUP BY | [`global`] |
//! | any other GROUP BY (incl. a computed key) | [`grouped`] |
//!
//! The shared expression→slot lowering and accumulator-width rule live here,
//! since the global and grouped paths both use them.

mod global;
mod global_distinct;
pub(super) mod grouped;
mod grouped_distinct;
mod reinterpret;

use self::reinterpret::{reinterpret_columns, temporal_to_int};
use crate::compile::Error;
use crate::expression::{AggregateFunc, Expression};
use crate::types::Type;
use arrow_array::RecordBatch;
use arrow_schema::DataType;
use dispatch::{
    AggregationKind, AggregationSlot, GroupLimit, RecordBatchOperatorSpec, RowKeySchema,
};
use std::fmt;
use std::sync::Arc;

/// GROUP BY + aggregate functions.
#[derive(Debug)]
pub struct Aggregate {
    pub groups: Vec<Expression>,
    pub expressions: Vec<Expression>,
    /// A LIMIT pushed into this grouped aggregate by the plan-rewrite passes:
    /// [`GroupLimit::TopK`] for an `ORDER BY <slot> DESC LIMIT k` feeding it (see
    /// `PlanNode::annotate_group_topn`), [`GroupLimit::First`] for a plain
    /// `LIMIT k` (`PlanNode::annotate_group_limit`). The
    /// group operator then emits only each partition's kept rows instead of
    /// every group.
    pub output_limit: Option<GroupLimit>,
}

impl fmt::Display for Aggregate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let groups = render(&self.groups);
        let exprs = render(&self.expressions);
        write!(f, "Aggregate(groups: [{groups}], exprs: [{exprs}])")
    }
}

fn render(exprs: &[Expression]) -> String {
    exprs
        .iter()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

impl Aggregate {
    pub fn compile(
        &self,
        input: RecordBatchOperatorSpec,
        input_nullability: Vec<bool>,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        // Materialise every computed group key and aggregate argument (e.g. the
        // `column1 * column2` in `SUM(column1 * column2)`) into a leading
        // projected column. From here every group key and every aggregate
        // argument is a plain column reference, so the slot builders below only
        // ever read column indices, never run an expression. When nothing needs
        // materialising (the common all-plain-column case) the input and `self`
        // are reused as-is, with no projection and no clone.
        let (input, nullability, resolved) = self.materialize_inputs(input, input_nullability)?;
        match resolved {
            Some(resolved) => resolved.compile_resolved(input, &nullability),
            None => self.compile_resolved(input, &nullability),
        }
    }

    /// Compile a *resolved* aggregate: one whose group keys and aggregate
    /// arguments are all plain column references, because
    /// [`materialize_inputs`](Self::materialize_inputs) has already projected
    /// every computed one into a column. Picks the strategy for the query shape
    /// (global, grouped, or one of the `COUNT(DISTINCT)` paths; see the
    /// module-level table) and builds it.
    fn compile_resolved(
        &self,
        input: RecordBatchOperatorSpec,
        nullability: &[bool],
    ) -> Result<RecordBatchOperatorSpec, Error> {
        // `COUNT(DISTINCT x)` as the sole aggregate with no GROUP BY: a dedicated
        // single-level global fast path.
        if let [Expression::AggregateFunc(AggregateFunc::CountDistinct(a))] =
            self.expressions.as_slice()
            && self.groups.is_empty()
        {
            return self.compile_global_distinct(input, a, nullability);
        }

        // Reject any group key whose type no extractor can encode, naming the
        // offending column so the error points at the right key rather than the
        // first one. Every group key is a plain column reference here (computed
        // keys were materialised into columns), so the source name is to hand.
        for group in &self.groups {
            if let Expression::Ref(r) = group
                && !is_groupable_key_type(&r.return_type)
            {
                return Err(Error::DataTypeNotSupportedForGroupBy {
                    column: r.name.clone(),
                    data_type: r.return_type.clone(),
                });
            }
        }

        // Aggregates compute on the int a temporal column stores (a DATE is Int32
        // days, a TIMESTAMP is Int64 seconds), so reinterpret any temporal value
        // column to that int before the readers see it, and restore the temporal
        // type on the output (group keys, and MIN/MAX of a date) afterwards. The
        // int-ify is gated to a MIN/MAX of a date/timestamp, so ordinary
        // aggregates and a plain `GROUP BY date` (whose key the row reader already
        // casts) pay nothing.
        let input = self.int_ify_temporal_values(input);

        // One `COUNT(DISTINCT)` over a GROUP BY, alone or mixed with non-distinct
        // aggregates, lowers to a two-level GROUP BY.
        let n_distinct = self
            .expressions
            .iter()
            .filter(|e| {
                matches!(
                    e,
                    Expression::AggregateFunc(AggregateFunc::CountDistinct(_))
                )
            })
            .count();
        let grouped = if !self.groups.is_empty() && n_distinct == 1 {
            self.compile_grouped_distinct(input, nullability)?
        } else if self.groups.is_empty() {
            self.compile_global(input)?
        } else {
            // Every GROUP BY (plain column keys, two-int-key, the row fallback, or
            // a single computed key) lowers through the general grouped path.
            self.compile_grouped(input, nullability)?
        };
        Ok(self.restore_declared_output(grouped))
    }

    /// Reinterpret every temporal column of the aggregate input to the int it
    /// stores (`Date32 -> Int32`, `Timestamp -> Int64`), zero-copy, so the value
    /// readers (which only know int columns) can read a `MIN`/`MAX` of a date.
    /// A no-op unless the aggregate actually takes a `MIN`/`MAX` of a date/
    /// timestamp, so the common paths are untouched.
    fn int_ify_temporal_values(&self, input: RecordBatchOperatorSpec) -> RecordBatchOperatorSpec {
        let has_temporal_value = self
            .expressions
            .iter()
            .any(|e| aggregate_value_temporal(e).is_some());
        if !has_temporal_value {
            return input;
        }
        input.project(|| {
            |batch: RecordBatch| Ok(reinterpret_columns(batch, |_, dt| temporal_to_int(dt))?)
        })
    }

    /// Restore the declared arrow type on the grouped/global output's columns:
    /// the group-key columns lead (one per `self.groups`, typed by its
    /// `result_type`), followed by one value column per `self.expressions` (typed
    /// only for a `MIN`/`MAX` of a date). A temporal key/value gets its
    /// `Date32`/`Timestamp` back; a decimal key gets its declared
    /// `Decimal64(p, s)`/`Decimal128(p, s)` restamped over the raw values the
    /// group operator emits (a metadata-only restamp inside
    /// [`reinterpret_columns`], never a rescaling arrow cast). Zero-copy where
    /// the widths match.
    fn restore_declared_output(&self, op: RecordBatchOperatorSpec) -> RecordBatchOperatorSpec {
        let mut targets: Vec<Option<DataType>> = self
            .groups
            .iter()
            .map(|g| g.result_type().ok().and_then(|t| key_output_type(&t)))
            .collect();
        targets.extend(self.expressions.iter().map(aggregate_value_temporal));
        if !targets.iter().any(Option::is_some) {
            return op;
        }
        let targets = Arc::new(targets);
        op.project(move || {
            let targets = targets.clone();
            move |batch: RecordBatch| {
                Ok(reinterpret_columns(batch, |i, _| {
                    targets.get(i).and_then(Option::clone)
                })?)
            }
        })
    }

    fn is_lone_count_star(&self) -> bool {
        matches!(
            self.expressions.as_slice(),
            [Expression::AggregateFunc(AggregateFunc::CountStar(_))]
        )
    }
}

/// Lower each aggregate expression to its [`AggregationSlot`]. Shared by the
/// global and grouped multi-aggregate paths, which build slots identically. A
/// `MIN`/`MAX` over a `Utf8` column folds the byte extreme (`StrMin`/`StrMax`);
/// over an integer column the numeric one. `COUNT(*)` ignores its column, so the
/// index is a placeholder.
///
/// Each slot's output type is DuckDB's declared result type for the call (its
/// physical arrow type: `BIGINT`→`Int64` for a count, `HUGEINT`→`Decimal128` for
/// a sum, the input type for a `MIN`/`MAX`). The accumulator may store it at a
/// different width; the output phase casts to this. A `MIN`/`MAX` of a date keeps
/// its physical int here and is restored to the temporal type by
/// [`Aggregate::restore_declared_output`], the one place dates are coerced.
fn aggregation_slots(exprs: &[Expression]) -> Result<Vec<AggregationSlot>, Error> {
    exprs
        .iter()
        .map(|e| {
            let Expression::AggregateFunc(func) = e else {
                return Err(Error::UnsupportedAggregateExpression(e.clone()));
            };
            let (kind, column) = match func {
                AggregateFunc::CountStar(_) => (AggregationKind::CountStar, 0),
                AggregateFunc::Count(a) => (AggregationKind::Count, a.column().column_idx),
                AggregateFunc::Sum(a) => (AggregationKind::Sum, a.column().column_idx),
                AggregateFunc::Min(a) => (
                    extreme_kind(&a.column().return_type, AggregationKind::Min)
                        .ok_or_else(|| Error::UnsupportedAggregateExpression(e.clone()))?,
                    a.column().column_idx,
                ),
                AggregateFunc::Max(a) => (
                    extreme_kind(&a.column().return_type, AggregationKind::Max)
                        .ok_or_else(|| Error::UnsupportedAggregateExpression(e.clone()))?,
                    a.column().column_idx,
                ),
                AggregateFunc::First(a) => (AggregationKind::First, a.column().column_idx),
                _ => return Err(Error::UnsupportedAggregateExpression(e.clone())),
            };
            Ok(AggregationSlot::new(
                kind,
                column,
                crate::types::physical_arrow_type(func.return_type()),
            ))
        })
        .collect()
}

/// Validate that a column of type `ty` can be a `MIN`/`MAX` argument, returning the
/// `MIN`/`MAX` `kind` if so. The value family (string / integer / float) is chosen
/// by the column type at bind, not encoded in the kind: a `Utf8` column folds the
/// byte extreme through the value arena, an integer/temporal one the numeric extreme
/// (`Date`/`Timestamp` seen as the int they store, after
/// [`Aggregate::int_ify_temporal_values`]), a float one in `f64`, a decimal one at
/// its carrier width (`i64` for `Decimal64`, the wide `i128` for `Decimal128`; its
/// raw unscaled values order exactly as the decimal values, since every value of a
/// column shares its scale). `None` for any other type, so
/// the caller reports a clean `UnsupportedAggregateExpression` rather than a worker
/// panic in the reader.
fn extreme_kind(ty: &Type, kind: AggregationKind) -> Option<AggregationKind> {
    match ty {
        // UInt8/16/32 fold losslessly through the reader's i64 accumulator;
        // UInt64 is excluded (it would wrap above i64::MAX) so it reports a clean
        // unsupported error rather than a silently wrong extreme.
        Type::Utf8
        | Type::Int8
        | Type::Int16
        | Type::Int32
        | Type::Int64
        | Type::UInt8
        | Type::UInt16
        | Type::UInt32
        | Type::Float32
        | Type::Float64
        | Type::Date
        | Type::Timestamp
        | Type::TimestampTz
        | Type::Decimal { .. } => Some(kind),
        _ => None,
    }
}

/// The arrow output type of a temporal `Type`, or `None` for a non-temporal one.
/// A temporal type is one whose canonical arrow type has a backing int form
/// ([`temporal_to_int`]); that int is what the group-by computes on, and this
/// canonical type is restored on its output. The one place dates are coerced.
pub(super) fn temporal_output_type(t: &Type) -> Option<DataType> {
    let arrow = crate::types::physical_arrow_type(t);
    temporal_to_int(&arrow).is_some().then_some(arrow)
}

/// The arrow type a group-key column must be restored to on output, or `None`
/// when the group operator already emits the declared type. A temporal key is
/// restored to its `Date32`/`Timestamp`; a decimal key to its declared
/// `Decimal64(p, s)`/`Decimal128(p, s)`, because the single-key extractor emits
/// it at the extractor's default decimal shape with correct raw values.
fn key_output_type(t: &Type) -> Option<DataType> {
    match t {
        Type::Decimal { .. } => Some(crate::types::physical_arrow_type(t)),
        _ => temporal_output_type(t),
    }
}

/// The temporal arrow type a `MIN`/`MAX` aggregate emits, or `None` for any other
/// expression (or a `MIN`/`MAX` over a non-temporal column). Used to gate the
/// int-ify of value columns and to restore the temporal type on the output.
fn aggregate_value_temporal(e: &Expression) -> Option<DataType> {
    match e {
        Expression::AggregateFunc(AggregateFunc::Min(a) | AggregateFunc::Max(a)) => {
            temporal_output_type(&a.column().return_type)
        }
        _ => None,
    }
}

/// Map group-key types (each paired with its column's nullability) to the
/// [`RowKeyExtractor`](dispatch::RowKeyExtractor) encoding schema, in key
/// order. Shared by the general grouped path ([`grouped`]) and the
/// `COUNT(DISTINCT)` two-level lowering ([`grouped_distinct`]). Returns `Err`
/// with the first key type the row encoding doesn't support, so the caller can
/// name the offending key rather than panicking in `RowKeySchema::new`.
pub(super) fn row_key_schema<'a>(
    keys: impl IntoIterator<Item = (&'a Type, bool)>,
) -> Result<RowKeySchema, Type> {
    let (arrow, nullable): (Vec<_>, Vec<_>) = keys
        .into_iter()
        .map(|(t, nullable)| {
            row_key_arrow_type(t)
                .map(|dt| (dt, nullable))
                .ok_or_else(|| t.clone())
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .unzip();
    Ok(RowKeySchema::new(arrow, nullable))
}

/// The arrow type the row encoder uses for one group-key column, or `None` for a
/// type it can't encode.
///
/// Only the fixed-width types (integer widths and `Decimal128`) and `Utf8View`
/// are byte-packable into a row key, exactly the rule
/// [`RowKeySchema::new`](dispatch::RowKeySchema) enforces. A `DATE`/`TIMESTAMP`
/// key drops to its backing int ([`temporal_to_int`]) and rides along for free;
/// the row reader casts each key column to this type, so grouping on the integer
/// day/second count is lossless (the temporal type is restored on the output). A
/// decimal key keeps its exact `Decimal64(p, s)`/`Decimal128(p, s)` shape, so the
/// row reader never casts it (a decimal-to-decimal arrow cast would rescale the
/// values). `Boolean`/`Float64` can't pack, so they return `None` for a clean
/// "unsupported" rather than a panic in `RowKeySchema::new`.
fn row_key_arrow_type(t: &Type) -> Option<DataType> {
    let dt = crate::types::physical_arrow_type(t);
    let dt = temporal_to_int(&dt).unwrap_or(dt);
    (dt.is_integer()
        || matches!(dt, DataType::Decimal64(_, _) | DataType::Decimal128(_, _))
        || dt == DataType::Utf8View)
        .then_some(dt)
}

/// Whether any key extractor can group on this type. The dedicated single/pair/
/// int-string extractors handle exactly the integer widths, the decimals, and
/// `Utf8`; every other groupable type (`Date`/`Timestamp`) rides the row
/// encoder. A type this rejects (`Float`/`Boolean`) has no grouping path at all,
/// so the caller can reject it up front and name the offending column.
pub(super) fn is_groupable_key_type(t: &Type) -> bool {
    row_key_arrow_type(t).is_some()
}

/// The accumulator-width rule shared by the global and grouped paths: `i128` when
/// a `SUM` reads a 64-bit or 128-bit column (whose total can overflow `i64`, and
/// any decimal `SUM` declares a `DECIMAL(38, s)` result), or when a `MIN`/`MAX`
/// reads a wide (`Decimal128`-carried) decimal column whose raw unscaled `i128`
/// values must be held losslessly; else `i64`. A `MIN`/`MAX` over a
/// `Decimal64`-carried column stays narrow: its raw values are `i64` and its
/// extreme is one of them.
fn needs_wide_accumulator(exprs: &[Expression]) -> bool {
    exprs.iter().any(|e| match e {
        Expression::AggregateFunc(AggregateFunc::Sum(a)) => {
            matches!(
                a.column().return_type,
                Type::Int64 | Type::Int128 | Type::Decimal { .. }
            )
        }
        Expression::AggregateFunc(AggregateFunc::Min(a) | AggregateFunc::Max(a)) => {
            matches!(
                a.column().return_type,
                Type::Decimal { precision, .. } if precision > crate::types::MAX_DECIMAL64_PRECISION
            )
        }
        _ => false,
    })
}
