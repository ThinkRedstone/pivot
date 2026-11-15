//! [`Function`] — the scalar-function enum dispatching DuckDB `BOUND_FUNCTION`s
//! to the per-function expression types.

use super::{
    Arithmetic, Concat, Contains, DatePart, DateTrunc, Divide, FormatBytes, IntervalArithmetic,
    Length, Like, NormalizedInterval, Prefix, RegexpFullMatch, RegexpJitReplace, RegexpReplace,
    Substring, Suffix, TemporalConvert, VariantGet,
};
use crate::compile::{self, ExprFn, ExprResult, stateless_expr};
use crate::expression::Expression;
use crate::types::Type;
use arrow_array::{Int64Array, RecordBatch};
use std::fmt::{self, Display};
use std::sync::Arc;

/// The binding signature of a pivot-defined scalar function: argument types,
/// return type, and whether DuckDB must leave the call un-folded. The bridge
/// reads this to register the function for DuckDB's binder, so a pivot scalar
/// function needs no C++ registration.
pub struct ScalarFunctionSignature {
    pub arguments: Vec<Type>,
    pub return_type: Type,
    /// `true` marks the function `VOLATILE` so DuckDB can't constant-fold the
    /// call away before pivot re-plans it (e.g. a no-arg `drop_cache()`).
    pub volatile: bool,
}

/// The binding signature for a pivot-defined scalar function `name`, or `None`
/// for names pivot doesn't define (DuckDB's own built-ins like `+`/`length`,
/// which DuckDB binds itself and pivot only intercepts at compile time). The
/// compile-time mapping of the same names lives in `Function::from_handle`.
pub fn builtin_scalar_function(name: &str) -> Option<ScalarFunctionSignature> {
    match name {
        "drop_cache" => Some(ScalarFunctionSignature {
            arguments: vec![],
            return_type: Type::Int64,
            volatile: true,
        }),
        // Not a DuckDB built-in (unlike `regexp_replace`), so its signature is
        // registered here for DuckDB's binder.
        "regexp_jit_replace" => Some(ScalarFunctionSignature {
            arguments: vec![Type::Utf8, Type::Utf8, Type::Utf8],
            return_type: Type::Utf8,
            volatile: false,
        }),
        // DuckDB binds `doc->'key'` as `json_extract`. The stub is a no-op, so
        // mark it volatile to prevent DuckDB from evaluating or folding it.
        "json_extract" => Some(ScalarFunctionSignature {
            arguments: vec![Type::Variant, Type::Utf8],
            return_type: Type::Variant,
            volatile: true,
        }),
        _ => None,
    }
}

/// A scalar function call (e.g. `year`, `substring`).
#[derive(Debug, Clone)]
pub enum Function {
    Contains(Contains),
    Prefix(Prefix),
    Suffix(Suffix),
    /// `LIKE` / `NOT LIKE` against a constant pattern, for the patterns DuckDB's
    /// optimizer leaves alone (several wildcards, or a `_`).
    Like(Like),
    Arithmetic(Arithmetic),
    Length(Length),
    /// `format_bytes(bytes)` — a byte count rendered in binary units.
    FormatBytes(FormatBytes),
    RegexpReplace(RegexpReplace),
    /// `regexp_full_match` — also the `~` / `!~` operators and `SIMILAR TO`.
    RegexpFullMatch(RegexpFullMatch),
    /// `regexp_jit_replace` — like `RegexpReplace` but always PCRE2 JIT-compiled.
    RegexpJitReplace(RegexpJitReplace),
    Divide(Divide),
    /// `left || right` string concatenation.
    Concat(Concat),
    /// `substring(string, start[, length])` with constant positions.
    Substring(Substring),
    DateTrunc(DateTrunc),
    DatePart(DatePart),
    /// `date`/`timestamp` ± `INTERVAL` (e.g. `now() - interval '5 days'`).
    IntervalArithmetic(IntervalArithmetic),
    /// `make_date(days)` / `make_timestamp(seconds)` — read an integer column as a
    /// real `DATE` / `TIMESTAMP`.
    TemporalConvert(TemporalConvert),
    /// `drop_cache()` — evict pivot's compressed cache, returning the regions dropped.
    /// A side-effecting admin function; evaluated once over the [`DummyScan`]
    /// row of a `FROM`-less `SELECT`. See its compile impl.
    ///
    /// [`DummyScan`]: crate::operator::DummyScan
    DropCache,
    /// `normalized_interval(span)` — an interval carried into the canonical
    /// form its fields must take before they compare. DuckDB's binder wraps
    /// every interval it orders, compares or hashes in this.
    NormalizedInterval(NormalizedInterval),
    /// A variant (JSON) path read: `doc->'key'` chains, optionally typed by a
    /// fused `CAST`.
    VariantGet(VariantGet),
}

impl Function {
    /// Visit every child expression of this function, in argument order. An
    /// exhaustive match with no wildcard, so adding a variant forces the
    /// author to declare its children here.
    pub fn for_each_argument(&self, visit: &mut impl FnMut(&Expression)) {
        match self {
            Function::Contains(c) => {
                visit(&c.needle);
                visit(&c.haystack);
            }
            Function::Prefix(p) => {
                visit(&p.haystack);
                visit(&p.prefix);
            }
            Function::Suffix(s) => {
                visit(&s.haystack);
                visit(&s.suffix);
            }
            Function::Like(l) => visit(&l.input),
            Function::Arithmetic(a) => {
                visit(&a.left);
                visit(&a.right);
            }
            Function::Length(l) => visit(&l.input),
            Function::FormatBytes(b) => visit(&b.input),
            Function::RegexpReplace(r) => visit(&r.input),
            // The pattern is a plan-time constant, not a child expression.
            Function::RegexpFullMatch(r) => visit(&r.input),
            Function::RegexpJitReplace(r) => visit(&r.input),
            Function::Divide(d) => {
                visit(&d.left);
                visit(&d.right);
            }
            Function::Concat(c) => {
                visit(&c.left);
                visit(&c.right);
            }
            Function::Substring(s) => visit(&s.input),
            Function::DateTrunc(d) => visit(&d.source),
            Function::DatePart(d) => visit(&d.source),
            Function::IntervalArithmetic(i) => visit(&i.operand),
            Function::TemporalConvert(t) => visit(&t.source),
            Function::NormalizedInterval(n) => visit(&n.input),
            Function::DropCache => {}
            Function::VariantGet(v) => visit(&v.input),
        }
    }

    /// The mutable twin of [`for_each_argument`](Self::for_each_argument),
    /// for passes that rewrite child expressions in place.
    pub fn for_each_argument_mut(&mut self, visit: &mut impl FnMut(&mut Expression)) {
        match self {
            Function::Contains(c) => {
                visit(&mut c.needle);
                visit(&mut c.haystack);
            }
            Function::Prefix(p) => {
                visit(&mut p.haystack);
                visit(&mut p.prefix);
            }
            Function::Suffix(s) => {
                visit(&mut s.haystack);
                visit(&mut s.suffix);
            }
            Function::Like(l) => visit(&mut l.input),
            Function::Arithmetic(a) => {
                visit(&mut a.left);
                visit(&mut a.right);
            }
            Function::Length(l) => visit(&mut l.input),
            Function::FormatBytes(b) => visit(&mut b.input),
            Function::RegexpReplace(r) => visit(&mut r.input),
            // The pattern is a plan-time constant, not a child expression.
            Function::RegexpFullMatch(r) => visit(&mut r.input),
            Function::RegexpJitReplace(r) => visit(&mut r.input),
            Function::Divide(d) => {
                visit(&mut d.left);
                visit(&mut d.right);
            }
            Function::Concat(c) => {
                visit(&mut c.left);
                visit(&mut c.right);
            }
            Function::Substring(s) => visit(&mut s.input),
            Function::DateTrunc(d) => visit(&mut d.source),
            Function::DatePart(d) => visit(&mut d.source),
            Function::IntervalArithmetic(i) => visit(&mut i.operand),
            Function::TemporalConvert(t) => visit(&mut t.source),
            Function::NormalizedInterval(n) => visit(&mut n.input),
            Function::DropCache => {}
            Function::VariantGet(v) => visit(&mut v.input),
        }
    }
}

impl Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Function::Contains(c) => write!(f, "{c}"),
            Function::Prefix(p) => write!(f, "{p}"),
            Function::Suffix(s) => write!(f, "{s}"),
            Function::Like(l) => write!(f, "{l}"),
            Function::Arithmetic(a) => write!(f, "{a}"),
            Function::Length(l) => write!(f, "{l}"),
            Function::FormatBytes(b) => write!(f, "{b}"),
            Function::RegexpReplace(r) => write!(f, "{r}"),
            Function::RegexpFullMatch(r) => write!(f, "{r}"),
            Function::RegexpJitReplace(r) => write!(f, "{r}"),
            Function::Divide(d) => write!(f, "{d}"),
            Function::Concat(c) => write!(f, "{c}"),
            Function::Substring(s) => write!(f, "{s}"),
            Function::DateTrunc(dt) => write!(f, "{dt}"),
            Function::DatePart(d) => write!(f, "{d}"),
            Function::IntervalArithmetic(i) => write!(f, "{i}"),
            Function::TemporalConvert(c) => write!(f, "{c}"),
            Function::NormalizedInterval(n) => write!(f, "{n}"),
            Function::DropCache => write!(f, "drop_cache()"),
            Function::VariantGet(v) => write!(f, "{v}"),
        }
    }
}

impl Function {
    /// The type a call to this function yields.
    pub fn result_type(&self) -> Type {
        match self {
            // The string matchers yield booleans.
            Function::Contains(_)
            | Function::Prefix(_)
            | Function::Suffix(_)
            | Function::Like(_)
            | Function::RegexpFullMatch(_) => Type::Boolean,
            // Arithmetic and the extractors carry DuckDB's bound result type.
            Function::Arithmetic(a) => a.return_type.clone(),
            Function::Length(l) => l.return_type.clone(),
            Function::DatePart(d) => d.return_type.clone(),
            // The regex replacers, substring and concatenation rewrite
            // strings, and a formatted byte count is one.
            Function::RegexpReplace(_)
            | Function::RegexpJitReplace(_)
            | Function::Substring(_)
            | Function::Concat(_)
            | Function::FormatBytes(_) => Type::Utf8,
            // `/` computes a float quotient, single- or double-precision.
            Function::Divide(d) => d.return_type.clone(),
            // `date_trunc` keeps its source's timestamp type (with or without
            // a zone), as DuckDB bound it.
            Function::DateTrunc(d) => d.result.clone(),
            // Canonicalising a span rewrites its fields, not its type.
            Function::NormalizedInterval(_) => Type::Interval,
            // `date`/`timestamp` ± interval keeps the temporal operand's type,
            // and `make_date`/`make_timestamp` produce the type they convert to.
            Function::IntervalArithmetic(i) => i.result.clone(),
            Function::TemporalConvert(c) => c.result.clone(),
            // A variant path read yields its cast's type (the sub-variant when
            // bare); a JSON-text render always yields a string.
            Function::VariantGet(v) => v.result_type(),
            // `drop_cache()` returns the evicted-entry count.
            Function::DropCache => Type::Int64,
        }
    }

    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        match self {
            Function::Contains(c) => c.compile(),
            Function::Prefix(p) => p.compile(),
            Function::Suffix(s) => s.compile(),
            Function::Like(l) => l.compile(),
            Function::Arithmetic(a) => a.compile(),
            Function::Length(l) => l.compile(),
            Function::FormatBytes(b) => b.compile(),
            Function::RegexpReplace(r) => r.compile(),
            Function::RegexpFullMatch(r) => r.compile(),
            Function::RegexpJitReplace(r) => r.compile(),
            Function::Divide(d) => d.compile(),
            Function::Concat(c) => c.compile(),
            Function::Substring(s) => s.compile(),
            Function::DateTrunc(dt) => dt.compile(),
            Function::DatePart(d) => d.compile(),
            Function::IntervalArithmetic(i) => i.compile(),
            Function::TemporalConvert(c) => c.compile(),
            Function::NormalizedInterval(n) => n.compile(),
            Function::VariantGet(v) => v.compile(),
            // `drop_cache()` evicts every in-memory cache *and* the on-disk cache
            // (so remote reads go cold to the network) as a side effect, then
            // returns the total entries dropped. Evaluated over the single
            // `DummyScan` row on a worker thread (where `memory_ctx` and the
            // worker's disk-cache handle are valid), so the eviction happens
            // exactly once; the returned array matches the (one-row) batch.
            Function::DropCache => Ok(stateless_expr(|batch: &RecordBatch| {
                let in_memory = dispatch::memory_ctx().caches().clear_all();
                let objects = dispatch::io::clear_disk_cache();
                let evicted = (in_memory + objects) as i64;
                ExprResult::Array(Arc::new(Int64Array::from(vec![evicted; batch.num_rows()])))
            })),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use rstest::rstest;

    #[rstest]
    fn drop_cache_returns_one_row(mut testing_planner: TestingPlanner) {
        // FROM-less SELECT runs over the DummyScan; drop_cache() evicts and
        // returns the (non-negative) region count.
        let rows = run(&mut testing_planner, "SELECT drop_cache()");

        assert_eq!(rows.len(), 1);
        assert!(only_column(&rows[0]).as_i64().unwrap() >= 0);
    }
}
