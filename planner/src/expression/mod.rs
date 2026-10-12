//! Expressions used inside [`Operator`](crate::operator::Operator)s — column
//! references, comparisons, aggregates, scalar functions, and constants.
//!
//! Each expression *kind* lives in its own submodule co-locating the AST type,
//! its `TryFrom` from the DuckDB expression, its `Display`, and (for the
//! evaluable kinds) its `compile` impl producing an
//! [`ExprFn`]. This module holds the cross-cutting
//! pieces: the [`Expression`] / [`Function`] enums that tie the kinds together,
//! the conversion [`enum@Error`], and the small `Display`/constant helpers
//! shared across kinds.

mod aggregate;
mod arithmetic;
mod between;
mod case;
mod cast;
mod compare;
mod concat;
mod conjunction;
mod contains;
mod convert;
mod date_part;
mod date_trunc;
mod divide;
mod format_bytes;
mod function;
mod in_list;
mod interval;
mod is_null;
mod length;
mod like;
mod maybe_error;
mod normalized_interval;
mod not;
mod prefix;
mod reference;
mod regexp;
mod regexp_jit;
mod substring;
mod suffix;
mod variant_get;

pub use aggregate::{AggregateFunc, CountStar, NumericAggregate};
pub use arithmetic::{Arithmetic, ArithmeticOp};
pub use between::Between;
pub use case::{Case, CaseCheck};
pub use cast::Cast;
pub use cast::cast_variant_array;
pub use cast::extract_variant_path;
pub use cast::json_to_canonical_variant;
pub use compare::{Compare, CompareType};
pub use concat::Concat;
pub use conjunction::{Conjunction, ConjunctionOp};
pub use contains::Contains;
pub use convert::TemporalConvert;
pub use date_part::{DatePart, DatePartKind};
pub use date_trunc::DateTrunc;
pub use divide::Divide;
pub use format_bytes::FormatBytes;
pub use function::{Function, ScalarFunctionSignature, builtin_scalar_function};
pub use in_list::InList;
pub use interval::IntervalArithmetic;
pub use is_null::IsNull;
pub use length::Length;
pub use like::Like;
pub use maybe_error::MaybeError;
pub use normalized_interval::NormalizedInterval;
pub use not::Not;
pub use prefix::Prefix;
pub use reference::Ref;
pub use regexp::{RegexpFullMatch, RegexpReplace};
pub use regexp_jit::RegexpJitReplace;
pub use substring::Substring;
pub use suffix::Suffix;
pub use variant_get::{JsonPath, VariantGet};

use crate::compile::{self, ExprFn, ExprResult, stateless_expr};
use crate::types::{self, Type};
use arrow::util::display::{ArrayFormatter, FormatOptions};
use arrow_array::{ArrayRef, Datum, RecordBatch, Scalar};
use duckdb_planner::duckdb_bridge::duckdb_types::{ExpressionType, OrderByNullType};
use std::fmt::{self, Display};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("{0}")]
    TypeError(#[from] types::Error),
    #[error("Unsupported comparison type: {0:?}")]
    UnsupportedComparisonType(ExpressionType),
    #[error("Unsupported aggregate function: {0}")]
    UnsupportedAggregateFunction(String),
    #[error("Unsupported scalar function: {0}")]
    UnsupportedScalarFunction(String),
    #[error("TRY_CAST is not supported; use CAST, which fails on unconvertible values")]
    TryCastUnsupported,
    #[error("Invalid parameter count for {function}: expected {expected}, got {actual}")]
    InvalidParameterCount {
        function: String,
        expected: usize,
        actual: usize,
    },
    #[error("Unsupported interval arithmetic: {0}")]
    UnsupportedInterval(String),
    #[error("Unsupported expression type: {0:?}")]
    UnsupportedExpressionType(ExpressionType),
    /// A sort key whose null placement the binder left unresolved.
    #[error("Unsupported null order: {0}")]
    UnsupportedNullOrder(OrderByNullType),
    /// A DuckDB exception surfaced while reading the plan across the bridge.
    #[error("{0}")]
    Bridge(#[from] duckdb_planner::BridgeError),
}

/// Format an arrow `Scalar<ArrayRef>` constant as `value:Type` for plan
/// display. Falls back to `?:DataType` for unsupported arrow types.
fn format_constant(s: &Scalar<ArrayRef>) -> String {
    let (arr, _is_scalar) = s.get();
    // Arrow renders a null as the empty string by default, which reads as a
    // missing value rather than a NULL one.
    let options = FormatOptions::default().with_null("NULL");
    let formatter = ArrayFormatter::try_new(arr, &options);
    let value = match formatter {
        Ok(f) => f.value(0).to_string(),
        Err(_) => "?".to_string(),
    };
    format!("{value}:{}", arr.data_type())
}

/// An expression in the logical plan. Discriminated by DuckDB's [`ExpressionType`].
#[derive(Debug, Clone)]
pub enum Expression {
    Ref(Ref),
    Compare(Compare),
    Between(Between),
    Constant(Scalar<ArrayRef>),
    AggregateFunc(AggregateFunc),
    Function(Function),
    InList(InList),
    Conjunction(Conjunction),
    Case(Case),
    MaybeError(MaybeError),
    Not(Not),
    IsNull(IsNull),
    Cast(Cast),
}

impl Expression {
    /// Rough count of the vectorized kernels evaluating this expression runs
    /// per batch - a cost proxy for deciding whether shrinking a batch before
    /// evaluating it pays for the row gather (see the `Filter` operator).
    pub fn count_kernels(&self) -> usize {
        match self {
            Expression::Ref(_) | Expression::Constant(_) => 0,
            Expression::Compare(c) => 1 + c.left.count_kernels() + c.right.count_kernels(),
            Expression::Between(b) => 2 + b.input.count_kernels(),
            Expression::AggregateFunc(_) => 1,
            // Function operands are almost always plain column refs; count the
            // function itself as one kernel without recursing into variants.
            Expression::Function(_) => 1,
            Expression::InList(l) => l.values.len() + l.input.count_kernels(),
            Expression::Conjunction(c) => {
                c.children.iter().map(Self::count_kernels).sum::<usize>()
                    + c.children.len().saturating_sub(1)
            }
            Expression::Case(c) => {
                c.checks
                    .iter()
                    .map(|check| 1 + check.when.count_kernels() + check.then.count_kernels())
                    .sum::<usize>()
                    + c.else_expr.count_kernels()
            }
            // The message only evaluates on failure, so it costs nothing here.
            Expression::MaybeError(m) => m.check.count_kernels() + m.value.count_kernels(),
            Expression::Not(n) => 1 + n.input.count_kernels(),
            Expression::IsNull(n) => 1 + n.input.count_kernels(),
            Expression::Cast(c) => 1 + c.source.count_kernels(),
        }
    }

    /// Collect the column indices this expression reads into `out`. Exact
    /// for every expression shape: columns are only ever read through `Ref`
    /// leaves, so visiting all children covers everything.
    pub fn collect_column_refs(&self, out: &mut Vec<usize>) {
        match self {
            Expression::Ref(r) => out.push(r.column_idx),
            Expression::Constant(_) => {}
            Expression::Compare(c) => {
                c.left.collect_column_refs(out);
                c.right.collect_column_refs(out);
            }
            Expression::Between(b) => {
                b.input.collect_column_refs(out);
                b.lower.collect_column_refs(out);
                b.upper.collect_column_refs(out);
            }
            Expression::AggregateFunc(a) => {
                for argument in a.arguments() {
                    argument.collect_column_refs(out);
                }
            }
            Expression::Function(f) => {
                f.for_each_argument(&mut |argument| argument.collect_column_refs(out));
            }
            Expression::InList(l) => {
                l.input.collect_column_refs(out);
                for value in &l.values {
                    value.collect_column_refs(out);
                }
            }
            Expression::Conjunction(c) => {
                for child in &c.children {
                    child.collect_column_refs(out);
                }
            }
            Expression::Case(c) => {
                for check in &c.checks {
                    check.when.collect_column_refs(out);
                    check.then.collect_column_refs(out);
                }
                c.else_expr.collect_column_refs(out);
            }
            Expression::MaybeError(m) => {
                m.check.collect_column_refs(out);
                m.message.collect_column_refs(out);
                m.value.collect_column_refs(out);
            }
            Expression::Not(n) => n.input.collect_column_refs(out),
            Expression::IsNull(n) => n.input.collect_column_refs(out),
            Expression::Cast(c) => c.source.collect_column_refs(out),
        }
    }

    /// Visit every column reference in this expression, in expression order.
    /// An exhaustive walk keeps transformations over bound references in one
    /// place instead of making each transformation know every expression
    /// shape.
    pub fn visit_column_refs_mut(&mut self, visit: &mut impl FnMut(&mut Ref)) {
        match self {
            Expression::Ref(r) => visit(r),
            Expression::Constant(_) => {}
            Expression::Compare(c) => {
                c.left.visit_column_refs_mut(visit);
                c.right.visit_column_refs_mut(visit);
            }
            Expression::Between(b) => {
                b.input.visit_column_refs_mut(visit);
                b.lower.visit_column_refs_mut(visit);
                b.upper.visit_column_refs_mut(visit);
            }
            Expression::AggregateFunc(a) => {
                for argument in a.arguments_mut() {
                    argument.visit_column_refs_mut(visit);
                }
            }
            Expression::Function(f) => {
                f.for_each_argument_mut(&mut |argument| argument.visit_column_refs_mut(visit));
            }
            Expression::InList(l) => {
                l.input.visit_column_refs_mut(visit);
                for value in &mut l.values {
                    value.visit_column_refs_mut(visit);
                }
            }
            Expression::Conjunction(c) => {
                for child in &mut c.children {
                    child.visit_column_refs_mut(visit);
                }
            }
            Expression::Case(c) => {
                for check in &mut c.checks {
                    check.when.visit_column_refs_mut(visit);
                    check.then.visit_column_refs_mut(visit);
                }
                c.else_expr.visit_column_refs_mut(visit);
            }
            Expression::MaybeError(m) => {
                m.check.visit_column_refs_mut(visit);
                m.message.visit_column_refs_mut(visit);
                m.value.visit_column_refs_mut(visit);
            }
            Expression::Not(n) => n.input.visit_column_refs_mut(visit),
            Expression::IsNull(n) => n.input.visit_column_refs_mut(visit),
            Expression::Cast(c) => c.source.visit_column_refs_mut(visit),
        }
    }

    /// Static result type of a computed expression. Used to pick a group-key
    /// extractor when grouping on it (e.g. `GROUP BY CASE …`) and to derive
    /// each operator's output types (see `PlanNode::output_types`).
    ///
    /// Constants with unsupported Arrow types return an error.
    pub fn result_type(&self) -> Result<Type, compile::Error> {
        match self {
            Expression::Ref(r) => Ok(r.return_type.clone()),
            Expression::Constant(s) => types::type_from_physical(s.get().0.data_type())
                .ok_or_else(|| compile::Error::IndeterminateResultType(self.clone())),
            // Comparisons and the boolean combinators all yield booleans.
            Expression::Compare(_)
            | Expression::Between(_)
            | Expression::InList(_)
            | Expression::Conjunction(_)
            | Expression::Not(_)
            | Expression::IsNull(_) => Ok(Type::Boolean),
            // A CASE's branches are unified to one type by DuckDB, so the ELSE
            // branch's type is the whole expression's type.
            Expression::Case(c) => c.else_expr.result_type(),
            // A tripped guard fails the query instead of yielding a value, so
            // the value branch's type is the whole expression's type.
            Expression::MaybeError(m) => m.value.result_type(),
            // A cast yields its target type.
            Expression::Cast(c) => Ok(c.target.clone()),
            // An aggregate carries DuckDB's bound result type.
            Expression::AggregateFunc(a) => Ok(a.return_type().clone()),
            Expression::Function(f) => Ok(f.result_type()),
        }
    }

    /// Whether this expression can evaluate to SQL NULL, given the nullability
    /// of each input column (`input[i]` = column `i` can hold NULLs). The
    /// planner uses this to route between the branch-free and the null-aware
    /// execution paths, so `false` must be sound: a `true` merely costs the
    /// fast path. Functions are folded conservatively (any nullable input
    /// column makes a function nullable) since a function of an entirely
    /// NULL-free input produces no NULLs (an operation that introduces its own
    /// NULLs, if one is ever added, must be special-cased here).
    pub fn nullability(&self, input: &[bool]) -> bool {
        match self {
            Expression::Ref(r) => input.get(r.column_idx).copied().unwrap_or(true),
            Expression::Constant(c) => c.get().0.is_null(0),
            // `IS NULL` is the one predicate that never yields NULL itself.
            Expression::IsNull(_) => false,
            Expression::Compare(c) => c.left.nullability(input) || c.right.nullability(input),
            Expression::Between(b) => {
                b.input.nullability(input)
                    || b.lower.nullability(input)
                    || b.upper.nullability(input)
            }
            Expression::InList(l) => {
                l.input.nullability(input) || l.values.iter().any(|v| v.nullability(input))
            }
            Expression::Conjunction(c) => c.children.iter().any(|e| e.nullability(input)),
            Expression::Not(n) => n.input.nullability(input),
            // A CASE's value comes from its THEN/ELSE branches.
            Expression::Case(c) => {
                c.checks.iter().any(|check| check.then.nullability(input))
                    || c.else_expr.nullability(input)
            }
            Expression::MaybeError(m) => m.value.nullability(input),
            Expression::Cast(c) => c.source.nullability(input),
            // A count is never NULL; the other aggregates are NULL over zero
            // (non-NULL) rows.
            Expression::AggregateFunc(a) => !matches!(
                a,
                AggregateFunc::CountStar(_)
                    | AggregateFunc::Count(_)
                    | AggregateFunc::CountDistinct(_)
            ),
            Expression::Function(_) => input.iter().any(|&nullable| nullable),
        }
    }

    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        match self {
            Expression::Ref(r) => r.compile(),
            Expression::Compare(c) => c.compile(),
            Expression::Constant(c) => {
                let scalar = c.clone();
                Ok(stateless_expr(move |_batch: &RecordBatch| {
                    ExprResult::Scalar(scalar.clone())
                }))
            }
            Expression::Function(f) => f.compile(),
            Expression::Between(b) => b.compile(),
            Expression::InList(i) => i.compile(),
            Expression::Conjunction(c) => c.compile(),
            Expression::Case(c) => c.compile(),
            Expression::MaybeError(m) => m.compile(),
            Expression::Not(n) => n.compile(),
            Expression::IsNull(n) => n.compile(),
            Expression::Cast(c) => c.compile(),
            _ => Err(compile::Error::UnsupportedExpression(self.clone())),
        }
    }
}

impl Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Ref(r) => write!(f, "{r}"),
            Expression::Compare(c) => write!(f, "{c}"),
            Expression::Between(b) => write!(f, "{b}"),
            Expression::Constant(c) => f.write_str(&format_constant(c)),
            Expression::AggregateFunc(a) => write!(f, "{a}"),
            Expression::Function(fun) => write!(f, "{fun}"),
            Expression::InList(i) => write!(f, "{i}"),
            Expression::Conjunction(c) => write!(f, "{c}"),
            Expression::Case(c) => write!(f, "{c}"),
            Expression::MaybeError(m) => write!(f, "{m}"),
            Expression::Not(n) => write!(f, "{n}"),
            Expression::IsNull(n) => write!(f, "{n}"),
            Expression::Cast(c) => write!(f, "{c}"),
        }
    }
}
