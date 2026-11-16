//! Constructs Pivot [`Expression`]s by walking DuckDB's borrowed bound-expression
//! handles. This is the expression half of the plan build (the operator half is
//! in [`operator`](super::operator)): every expression kind's `from_handle` is
//! implemented here, so the `crate::expression` modules hold only Pivot IR (the
//! AST types, their `Display`, and their `compile`).
//!
//! [`Expression::from_handle`] is the single recursive entry point; it dispatches
//! each DuckDB expression variant to its kind's `from_handle`. Scalar-function
//! calls go through [`Function::from_handle`], which routes on the function name.

use arrow_array::Datum;
use arrow_array::cast::AsArray;
use arrow_schema::{DataType, TimeUnit};
use duckdb_planner::duckdb_bridge::duckdb_types::ExpressionType;
use duckdb_planner::handle::{
    AggregateFunc as AggregateFuncHandle, Between as BetweenHandle, Case as CaseHandle,
    Cast as CastHandle, Compare as CompareHandle, Conjunction as ConjunctionHandle,
    Expression as DuckExpression, Function as FunctionHandle, InList as InListHandle,
    Not as NotHandle, Ref as RefHandle,
};
use duckdb_planner::{Expr, LogicalTypeId, ScalarValue};

use crate::expression::{
    AggregateFunc, Arithmetic, ArithmeticOp, Between, BucketGrid, Case, CaseCheck, Cast, Compare,
    Concat, Conjunction, ConjunctionOp, Contains, CountStar, DatePart, DatePartKind, DateTrunc,
    Divide, Error, Expression, FormatBytes, Function, InList, IntervalArithmetic, IsNull, Length,
    Like, MaybeError, NormalizedInterval, Not, NumericAggregate, Prefix, Ref, RegexpFullMatch,
    RegexpJitReplace, RegexpReplace, Round, Rounding, StructExtract, StructPack, Substring, Suffix,
    TemporalConvert, TimeBucket, VariantGet, count_epoch_months,
};
use crate::types::{
    MICROS_PER_DAY, StructField, Type, build_scalar_value, physical_arrow_type, type_from_logical,
};

impl Expression {
    /// Build a Pivot [`Expression`] (and its whole subtree) from a borrowed DuckDB
    /// bound expression handle. The single recursive entry point the operator walk
    /// and the per-kind builders share; each DuckDB expression variant dispatches
    /// to its kind's `from_handle`.
    pub(crate) fn from_handle(e: Expr<'_>) -> Result<Expression, Error> {
        Ok(match e.expression()? {
            DuckExpression::Ref(r) => Expression::Ref(Ref::from_handle(r)?),
            DuckExpression::Compare(c) => Expression::Compare(Compare::from_handle(c)?),
            DuckExpression::Between(b) => Expression::Between(Between::from_handle(b)?),
            DuckExpression::Constant(c) => Expression::Constant(build_scalar_value(c.value()?)?),
            DuckExpression::AggregateFunc(a) => {
                Expression::AggregateFunc(AggregateFunc::from_handle(a)?)
            }
            DuckExpression::Function(f) => Expression::Function(Function::from_handle(f)?),
            // `NOT IN` is the same operator negated; SQL's three-valued NOT
            // rides the [`Not`] wrapper.
            DuckExpression::InList(i) if i.negated()? => Expression::Not(Not {
                input: Box::new(Expression::InList(InList::from_handle(i)?)),
            }),
            DuckExpression::InList(i) => Expression::InList(InList::from_handle(i)?),
            DuckExpression::Conjunction(c) => Expression::Conjunction(Conjunction::from_handle(c)?),
            DuckExpression::Case(c) => Case::from_handle(c)?,
            DuckExpression::Not(n) => Expression::Not(Not::from_handle(n)?),
            DuckExpression::IsNull(n) => Expression::IsNull(IsNull {
                negated: n.negated()?,
                input: Box::new(Expression::from_handle(n.input()?)?),
            }),
            DuckExpression::Cast(c) => Cast::from_handle(c)?,
            DuckExpression::Unsupported(t) => return Err(Error::UnsupportedExpressionType(t)),
        })
    }
}

impl Ref {
    pub(crate) fn from_handle(view: RefHandle<'_>) -> Result<Ref, Error> {
        Ok(Ref {
            column_idx: view.column_index()?,
            return_type: type_from_logical(view.return_type()?)?,
            // Drop DuckDB's positional aliases (e.g. "0" for an unnamed computed
            // group key): an all-digit name carries no more than the index does
            // and reads as a constant in a plan dump, so fall back to `#idx`.
            name: view
                .alias()?
                .filter(|n| !n.is_empty() && !n.bytes().all(|b| b.is_ascii_digit())),
        })
    }
}

/// Refuse `operation` over an operand that is a whole struct. The kernels
/// behind comparisons and casts do not take a struct, and would fail on the
/// first batch instead of when the query is planned.
fn reject_struct_operand(operand: Expr<'_>, operation: &str) -> Result<(), Error> {
    if operand.return_type()?.id == LogicalTypeId::STRUCT {
        return Err(Error::UnsupportedStructOperation(operation.to_string()));
    }
    Ok(())
}

impl Compare {
    pub(crate) fn from_handle(view: CompareHandle<'_>) -> Result<Compare, Error> {
        reject_struct_operand(view.left()?, "A comparison")?;
        Ok(Compare {
            left: Box::new(Expression::from_handle(view.left()?)?),
            right: Box::new(Expression::from_handle(view.right()?)?),
            compare_type: view.comparison_type()?.try_into()?,
            return_type: type_from_logical(view.return_type()?)?,
        })
    }
}

impl Between {
    pub(crate) fn from_handle(view: BetweenHandle<'_>) -> Result<Between, Error> {
        reject_struct_operand(view.input()?, "BETWEEN")?;
        Ok(Between {
            input: Box::new(Expression::from_handle(view.input()?)?),
            lower: Box::new(Expression::from_handle(view.lower()?)?),
            upper: Box::new(Expression::from_handle(view.upper()?)?),
            lower_inclusive: view.lower_inclusive()?,
            upper_inclusive: view.upper_inclusive()?,
        })
    }
}

impl Not {
    pub(crate) fn from_handle(view: NotHandle<'_>) -> Result<Not, Error> {
        Ok(Not {
            input: Box::new(Expression::from_handle(view.input()?)?),
        })
    }
}

impl Conjunction {
    pub(crate) fn from_handle(view: ConjunctionHandle<'_>) -> Result<Conjunction, Error> {
        let op = if view.conjunction_type()? == ExpressionType::CONJUNCTION_OR {
            ConjunctionOp::Or
        } else {
            ConjunctionOp::And
        };
        Ok(Conjunction {
            op,
            children: view
                .children()?
                .into_iter()
                .map(Expression::from_handle)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

impl InList {
    pub(crate) fn from_handle(view: InListHandle<'_>) -> Result<InList, Error> {
        // Child 0 is the tested expression, children 1.. are the list values.
        let children: Vec<Expr<'_>> = view.children()?;
        reject_struct_operand(children[0], "IN")?;
        Ok(InList {
            input: Box::new(Expression::from_handle(children[0])?),
            values: children[1..]
                .iter()
                .copied()
                .map(Expression::from_handle)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

impl Case {
    pub(crate) fn from_handle(view: CaseHandle<'_>) -> Result<Expression, Error> {
        let arms = view.checks()?;

        // DuckDB guards every scalar subquery with `CASE WHEN too_many_rows
        // THEN error('…') ELSE value END`. The CASE kernel evaluates all of
        // its branches eagerly, which would raise the error on every batch, so
        // this shape lowers to [`MaybeError`], which only raises when a row's
        // check is actually true.
        if let [arm] = arms.as_slice()
            && let Some(message) = error_call_message(arm.then)?
        {
            return Ok(Expression::MaybeError(MaybeError {
                check: Box::new(Expression::from_handle(arm.when)?),
                message: Box::new(message),
                value: Box::new(Expression::from_handle(view.else_expr()?)?),
            }));
        }

        let checks = arms
            .into_iter()
            .map(|arm| {
                Ok(CaseCheck {
                    when: Box::new(Expression::from_handle(arm.when)?),
                    then: Box::new(Expression::from_handle(arm.then)?),
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(Expression::Case(Case {
            checks,
            else_expr: Box::new(Expression::from_handle(view.else_expr()?)?),
        }))
    }
}

/// The message argument of an `error(…)` call, or `None` for any other
/// expression.
fn error_call_message(e: Expr<'_>) -> Result<Option<Expression>, Error> {
    let DuckExpression::Function(func) = e.expression()? else {
        return Ok(None);
    };
    if func.name()? != "error" {
        return Ok(None);
    }
    let message = function_args(func, 1)?[0];
    Ok(Some(Expression::from_handle(message)?))
}

impl Cast {
    pub(crate) fn from_handle(view: CastHandle<'_>) -> Result<Expression, Error> {
        // DuckDB's TRY_CAST converts cast failures into NULLs; pivot casts
        // strictly and does not implement that contract, so reject the query
        // rather than quietly running the cast strict.
        if view.is_try()? {
            return Err(Error::TryCastUnsupported);
        }
        // A `BoundCastExpression`'s target is its own result type.
        let target = type_from_logical(view.return_type()?)?;
        // A struct casts to a struct field by field; nothing converts a
        // whole struct to or from another type.
        if !matches!(target, Type::Struct(_)) {
            reject_struct_operand(view.child()?, "A cast to another type")?;
        }
        let source = Expression::from_handle(view.child()?)?;

        // Fold a cast and its `->` chain into one typed read. This lets the
        // kernel read a shredded leaf without building intermediate variants.
        if matches!(source.result_type(), Ok(Type::Variant)) {
            if !VariantGet::supports_cast_to(&target) {
                return Err(Error::UnsupportedScalarFunction(format!(
                    "cannot cast a VARIANT to {target}"
                )));
            }
            let (input, path) = collapse_extractions(source);
            return Ok(Expression::Function(Function::VariantGet(VariantGet {
                input: Box::new(input),
                path,
                as_type: Some(target),
            })));
        }

        // Every other cast, including text into a variant (which `Cast::compile`
        // parses as JSON), is a plain `Cast`. A constant `'{...}'::VARIANT` never
        // reaches here: DuckDB folds it into a variant constant, parsed in
        // `build_scalar_value`.
        Ok(Expression::Cast(Cast {
            target_arrow: physical_arrow_type(&target),
            target,
            source: Box::new(source),
        }))
    }
}

/// Peel a chain of bare `->` extractions, returning the base expression and the
/// concatenated path (empty when `e` isn't an extraction, e.g. a cast directly
/// over a variant column).
fn collapse_extractions(e: Expression) -> (Expression, Vec<String>) {
    match e {
        Expression::Function(Function::VariantGet(extraction)) if extraction.as_type.is_none() => {
            let (input, mut path) = collapse_extractions(*extraction.input);
            path.extend(extraction.path);
            (input, path)
        }
        other => (other, Vec::new()),
    }
}

impl AggregateFunc {
    pub(crate) fn from_handle(view: AggregateFuncHandle<'_>) -> Result<AggregateFunc, Error> {
        let function = view.name()?;
        let distinct = view.distinct()?;
        let return_type = type_from_logical(view.return_type()?)?;
        let params = view
            .children()?
            .into_iter()
            .map(Expression::from_handle)
            .collect::<Result<Vec<_>, _>>()?;

        // DISTINCT is only supported for `COUNT` so far; reject `SUM(DISTINCT)`
        // etc. rather than silently computing the non-distinct aggregate.
        if distinct && function != "count" {
            return Err(Error::UnsupportedAggregateFunction(format!(
                "DISTINCT {function}"
            )));
        }

        // `FILTER (WHERE …)` is not supported; reject it rather than silently
        // aggregating over every row.
        if view.has_filter()? {
            return Err(Error::UnsupportedAggregateFunction(format!(
                "{function} with a FILTER clause"
            )));
        }

        match function.as_str() {
            "count_star" => Ok(AggregateFunc::CountStar(CountStar {
                params,
                return_type,
            })),
            // DuckDB plans SUM over integers and decimals as sum_no_overflow
            // when column statistics prove the accumulation cannot overflow;
            // the computation is plain SUM either way.
            "sum" | "sum_no_overflow" => Ok(AggregateFunc::Sum(numeric_aggregate(
                params,
                return_type,
                function,
            )?)),
            "avg" => Ok(AggregateFunc::Avg(numeric_aggregate(
                params,
                return_type,
                function,
            )?)),
            "min" => Ok(AggregateFunc::Min(numeric_aggregate(
                params,
                return_type,
                function,
            )?)),
            "max" => Ok(AggregateFunc::Max(numeric_aggregate(
                params,
                return_type,
                function,
            )?)),
            "count" if distinct => Ok(AggregateFunc::CountDistinct(numeric_aggregate(
                params,
                return_type,
                function,
            )?)),
            "count" => Ok(AggregateFunc::Count(numeric_aggregate(
                params,
                return_type,
                function,
            )?)),
            "first" => Ok(AggregateFunc::First(numeric_aggregate(
                params,
                return_type,
                function,
            )?)),
            _ => Err(Error::UnsupportedAggregateFunction(function)),
        }
    }
}

/// Build a single-argument numeric aggregate's payload, validating the arity.
fn numeric_aggregate(
    params: Vec<Expression>,
    return_type: Type,
    function: String,
) -> Result<NumericAggregate, Error> {
    if params.len() != 1 {
        return Err(Error::InvalidParameterCount {
            function,
            expected: 1,
            actual: params.len(),
        });
    }
    Ok(NumericAggregate {
        argument: Box::new(params.into_iter().next().unwrap()),
        return_type,
    })
}

impl Function {
    pub(crate) fn from_handle(func: FunctionHandle<'_>) -> Result<Function, Error> {
        let name = func.name()?;
        match name.as_str() {
            // DuckDB's binder wraps every interval it has to order, compare or
            // hash in this, and pivot has to honour it: an interval's fields
            // only compare once carried into their canonical form.
            "normalized_interval" => Ok(Function::NormalizedInterval(NormalizedInterval {
                input: Box::new(Expression::from_handle(function_args(func, 1)?[0])?),
            })),
            "contains" => Ok(Function::Contains(Contains::from_handle(func)?)),
            // DuckDB's optimizer rewrites `LIKE 'foo%'` into `prefix(col, 'foo')`
            // and `LIKE '%foo'` into `suffix(col, 'foo')`.
            "prefix" => Ok(Function::Prefix(Prefix::from_handle(func)?)),
            "suffix" => Ok(Function::Suffix(Suffix::from_handle(func)?)),
            // The patterns that same optimizer leaves alone stay `LIKE` calls:
            // `~~` is the operator's function name, `!~~` the negated form.
            "~~" => Ok(Function::Like(Like::from_handle(func, false)?)),
            "!~~" => Ok(Function::Like(Like::from_handle(func, true)?)),
            // `date`/`timestamp` ± `INTERVAL` carries an INTERVAL constant operand;
            // plain numeric `+`/`-` does not and stays `Arithmetic`.
            "+" | "-" => match interval_operand_index(func)? {
                Some(idx) => Ok(Function::IntervalArithmetic(
                    IntervalArithmetic::from_handle(func, idx)?,
                )),
                None => Ok(Function::Arithmetic(Arithmetic::from_handle(func)?)),
            },
            "*" => Ok(Function::Arithmetic(Arithmetic::from_handle(func)?)),
            "||" => Ok(Function::Concat(Concat::from_handle(func)?)),
            "length" | "strlen" | "len" => Ok(Function::Length(Length::from_handle(func)?)),
            "format_bytes" => Ok(Function::FormatBytes(FormatBytes::from_handle(func)?)),
            // The `substring(x FROM a FOR b)` syntax binds to the same call.
            "substring" | "substr" => Ok(Function::Substring(Substring::from_handle(func)?)),
            "regexp_replace" => Ok(Function::RegexpReplace(RegexpReplace::from_handle(func)?)),
            "regexp_full_match" => Ok(Function::RegexpFullMatch(RegexpFullMatch::from_handle(
                func,
            )?)),
            "regexp_jit_replace" => Ok(Function::RegexpJitReplace(RegexpJitReplace::from_handle(
                func,
            )?)),
            "/" => Ok(Function::Divide(Divide::from_handle(func)?)),
            "floor" => Ok(Function::Round(Round::from_floor_call(func)?)),
            "round" => Ok(Function::Round(Round::from_round_call(func)?)),
            "date_trunc" => Ok(Function::DateTrunc(DateTrunc::from_handle(func)?)),
            "time_bucket" => Ok(Function::TimeBucket(TimeBucket::from_handle(func)?)),
            "make_date" => Ok(Function::TemporalConvert(TemporalConvert::make_date(func)?)),
            "make_timestamp" => Ok(Function::TemporalConvert(TemporalConvert::make_timestamp(
                func,
            )?)),
            // Variant field access: DuckDB's binder rewrites `d.age` to
            // `variant_extract` (its native VARIANT function) and `d->'age'` to
            // `json_extract` (which resolves against pivot's registry). Both are
            // one extracted field.
            "variant_extract" | "json_extract" => {
                Ok(Function::VariantGet(VariantGet::from_handle(func)?))
            }
            // `{'a': x, 'b': y}` binds to `struct_pack` and the unnamed
            // `(x, y)` to `row`. The field names are in the result type.
            "struct_pack" | "row" => Ok(Function::StructPack(StructPack::from_handle(func)?)),
            // `pair.a` binds to `struct_extract` with the field's name. `pair[1]`
            // binds to `array_extract` with the field's position from 1, which
            // DuckDB's optimizer may rewrite to `struct_extract_at`. Only a
            // struct is supported as the value `array_extract` reads from.
            "struct_extract" | "struct_extract_at" | "array_extract" => {
                Ok(Function::StructExtract(StructExtract::from_handle(func)?))
            }
            "drop_cache" => {
                function_args(func, 0)?;
                Ok(Function::DropCache)
            }
            // `extract(<part> FROM ts)` lowers to a function named after the part
            // (`minute`, `year`, `dayofweek`, …).
            _ => match DatePartKind::from_function_name(&name) {
                Some(kind) => Ok(Function::DatePart(DatePart::from_handle(kind, func)?)),
                None => Err(Error::UnsupportedScalarFunction(name)),
            },
        }
    }
}

/// The position of the `INTERVAL` constant operand of a `+`/`-` call, or `None`
/// when neither operand is one (plain numeric arithmetic).
fn interval_operand_index(func: FunctionHandle<'_>) -> Result<Option<usize>, Error> {
    for (idx, param) in func.children()?.into_iter().enumerate() {
        if let DuckExpression::Constant(c) = param.expression()?
            && c.return_type()?.id == LogicalTypeId::INTERVAL
        {
            return Ok(Some(idx));
        }
    }
    Ok(None)
}

impl VariantGet {
    /// A bare `doc->'key'`: one extracted field, yielding the sub-variant. A
    /// cast above it types the read ([`Cast::from_handle`] fuses the chain).
    pub(crate) fn from_handle(func: FunctionHandle<'_>) -> Result<VariantGet, Error> {
        let params = function_args(func, 2)?;
        let field = constant_string(Expression::from_handle(params[1])?)?;
        Ok(VariantGet {
            input: Box::new(Expression::from_handle(params[0])?),
            path: vec![field],
            as_type: None,
        })
    }
}

impl Contains {
    pub(crate) fn from_handle(func: FunctionHandle<'_>) -> Result<Contains, Error> {
        let params = function_args(func, 2)?;
        Ok(Contains {
            needle: Box::new(Expression::from_handle(params[1])?),
            haystack: Box::new(Expression::from_handle(params[0])?),
        })
    }
}

impl Like {
    pub(crate) fn from_handle(func: FunctionHandle<'_>, negated: bool) -> Result<Like, Error> {
        let params = function_args(func, 2)?;
        Ok(Like {
            input: Box::new(Expression::from_handle(params[0])?),
            pattern: constant_string(Expression::from_handle(params[1])?)?,
            negated,
        })
    }
}

impl Prefix {
    pub(crate) fn from_handle(func: FunctionHandle<'_>) -> Result<Prefix, Error> {
        let params = function_args(func, 2)?;
        Ok(Prefix {
            haystack: Box::new(Expression::from_handle(params[0])?),
            prefix: Box::new(Expression::from_handle(params[1])?),
        })
    }
}

impl Substring {
    pub(crate) fn from_handle(func: FunctionHandle<'_>) -> Result<Substring, Error> {
        let params: Vec<Expr<'_>> = func.children()?;
        if params.len() != 2 && params.len() != 3 {
            return Err(Error::InvalidParameterCount {
                function: func.name()?,
                expected: 3,
                actual: params.len(),
            });
        }
        let start = constant_int64(Expression::from_handle(params[1])?)?;
        if start < 0 {
            return Err(Error::UnsupportedScalarFunction(format!(
                "substring start positions below 0 are not supported (got {start})"
            )));
        }
        let mut length = match params.get(2) {
            Some(&length) => {
                let length = constant_int64(Expression::from_handle(length)?)?;
                if length < 0 {
                    return Err(Error::UnsupportedScalarFunction(format!(
                        "substring lengths below 0 are not supported (got {length})"
                    )));
                }
                Some(length as u64)
            }
            None => None,
        };
        // DuckDB treats zero as one position before the first character: the
        // slice starts at the first character, but that virtual position uses
        // one unit of an explicit length. Canonicalize it here so the hot
        // execution path remains the same branch-free positive-start path.
        let start = if start == 0 {
            length = length.map(|length| length.saturating_sub(1));
            1
        } else {
            start as u64
        };
        Ok(Substring {
            input: Box::new(Expression::from_handle(params[0])?),
            start,
            length,
        })
    }
}

impl Suffix {
    pub(crate) fn from_handle(func: FunctionHandle<'_>) -> Result<Suffix, Error> {
        let params = function_args(func, 2)?;
        Ok(Suffix {
            haystack: Box::new(Expression::from_handle(params[0])?),
            suffix: Box::new(Expression::from_handle(params[1])?),
        })
    }
}

impl Arithmetic {
    pub(crate) fn from_handle(func: FunctionHandle<'_>) -> Result<Arithmetic, Error> {
        let name = func.name()?;
        let op = match name.as_str() {
            "+" => ArithmeticOp::Add,
            "-" => ArithmeticOp::Sub,
            "*" => ArithmeticOp::Mul,
            _ => return Err(Error::UnsupportedScalarFunction(name)),
        };
        // Unary forms (e.g. `-x`) bind to the same function names with one
        // parameter; only the binary forms are supported.
        let params = function_args(func, 2)?;
        Ok(Arithmetic {
            op,
            left: Box::new(Expression::from_handle(params[0])?),
            right: Box::new(Expression::from_handle(params[1])?),
            return_type: type_from_logical(func.return_type()?)?,
        })
    }
}

impl Concat {
    pub(crate) fn from_handle(func: FunctionHandle<'_>) -> Result<Concat, Error> {
        // `||` also concatenates blobs and lists; only the string form (whose
        // operands DuckDB's binder casts to VARCHAR) is supported.
        let return_type = type_from_logical(func.return_type()?)?;
        if return_type != Type::Utf8 {
            return Err(Error::UnsupportedScalarFunction(format!(
                "|| over {return_type} (only strings can be concatenated)"
            )));
        }
        let params = function_args(func, 2)?;
        Ok(Concat {
            left: Box::new(Expression::from_handle(params[0])?),
            right: Box::new(Expression::from_handle(params[1])?),
        })
    }
}

impl Divide {
    pub(crate) fn from_handle(func: FunctionHandle<'_>) -> Result<Divide, Error> {
        let params = function_args(func, 2)?;
        Ok(Divide {
            left: Box::new(Expression::from_handle(params[0])?),
            right: Box::new(Expression::from_handle(params[1])?),
            return_type: type_from_logical(func.return_type()?)?,
        })
    }
}

impl Round {
    pub(crate) fn from_floor_call(func: FunctionHandle<'_>) -> Result<Round, Error> {
        let params = function_args(func, 1)?;
        Ok(Round {
            rounding: Rounding::Floor,
            input: Box::new(Expression::from_handle(params[0])?),
            return_type: type_from_logical(func.return_type()?)?,
        })
    }

    pub(crate) fn from_round_call(func: FunctionHandle<'_>) -> Result<Round, Error> {
        let has_digits = func.children()?.len() == 2;
        let params = function_args(func, if has_digits { 2 } else { 1 })?;
        let digits = match has_digits {
            true => {
                let digits = constant_int64(Expression::from_handle(params[1])?)?;
                i32::try_from(digits).map_err(|_| {
                    Error::UnsupportedScalarFunction(format!("round to {digits} digits"))
                })?
            }
            false => 0,
        };
        Ok(Round {
            rounding: Rounding::Nearest { digits },
            input: Box::new(Expression::from_handle(params[0])?),
            // DuckDB types the call: a decimal's result keeps only the scale
            // the digits leave it.
            return_type: type_from_logical(func.return_type()?)?,
        })
    }
}

impl StructPack {
    pub(crate) fn from_handle(func: FunctionHandle<'_>) -> Result<StructPack, Error> {
        let return_type = type_from_logical(func.return_type()?)?;
        // A variant column's arrow layout differs per file, so no single
        // struct type describes a field holding one.
        if let Type::Struct(fields) = &return_type
            && fields.iter().any(|field| field.field_type == Type::Variant)
        {
            return Err(Error::UnsupportedScalarFunction(
                "a struct with a VARIANT field".to_string(),
            ));
        }
        Ok(StructPack {
            return_type,
            fields: func
                .children()?
                .into_iter()
                .map(Expression::from_handle)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

impl StructExtract {
    pub(crate) fn from_handle(func: FunctionHandle<'_>) -> Result<StructExtract, Error> {
        let name = func.name()?;
        let params = function_args(func, 2)?;
        let input = Expression::from_handle(params[0])?;
        let Type::Struct(fields) = type_from_logical(params[0].return_type()?)? else {
            return Err(Error::UnsupportedScalarFunction(format!(
                "{name} of a value that is not a struct"
            )));
        };
        let DuckExpression::Constant(field_key) = params[1].expression()? else {
            return Err(Error::UnsupportedScalarFunction(format!(
                "{name} with a field that is not a constant"
            )));
        };
        let field_idx = find_struct_field(&fields, field_key.value()?).ok_or_else(|| {
            Error::UnsupportedScalarFunction(format!("{name} of a field the struct does not have"))
        })?;
        Ok(StructExtract {
            input: Box::new(input),
            field_idx,
            return_type: type_from_logical(func.return_type()?)?,
        })
    }
}

/// The position of the struct field a `struct_extract` names: a string is a
/// field name, matched ignoring case as DuckDB's binder does, and an integer is
/// a position counted from 1. `None` when the struct has no such field, which
/// the binder has already ruled out for a query that reached planning.
fn find_struct_field(fields: &[StructField], field_key: ScalarValue) -> Option<usize> {
    match field_key {
        ScalarValue::Utf8(name) => fields
            .iter()
            .position(|field| field.name.eq_ignore_ascii_case(&name)),
        ScalarValue::Int64(position) => usize::try_from(position)
            .ok()
            .filter(|position| (1..=fields.len()).contains(position))
            .map(|position| position - 1),
        _ => None,
    }
}

impl Length {
    pub(crate) fn from_handle(func: FunctionHandle<'_>) -> Result<Length, Error> {
        let params = function_args(func, 1)?;
        Ok(Length {
            return_type: type_from_logical(func.return_type()?)?,
            input: Box::new(Expression::from_handle(params[0])?),
        })
    }
}

impl FormatBytes {
    pub(crate) fn from_handle(func: FunctionHandle<'_>) -> Result<FormatBytes, Error> {
        let params = function_args(func, 1)?;
        Ok(FormatBytes {
            input: Box::new(Expression::from_handle(params[0])?),
        })
    }
}

impl DateTrunc {
    pub(crate) fn from_handle(func: FunctionHandle<'_>) -> Result<DateTrunc, Error> {
        let params = function_args(func, 2)?;
        let unit = constant_string(Expression::from_handle(params[0])?)?.to_ascii_lowercase();
        // DuckDB types the call: the result tracks the source's timestamp
        // type, zone marker included.
        let result = type_from_logical(func.return_type()?)?;
        Ok(DateTrunc {
            unit,
            source: Box::new(Expression::from_handle(params[1])?),
            result,
        })
    }
}

impl TimeBucket {
    pub(crate) fn from_handle(func: FunctionHandle<'_>) -> Result<TimeBucket, Error> {
        let has_third_argument = func.children()?.len() == 3;
        let params = function_args(func, if has_third_argument { 3 } else { 2 })?;
        let ScalarValue::Interval {
            months,
            days,
            micros,
        } = read_constant_value(params[0])?
        else {
            return Err(reject_time_bucket("the bucket width has to be an INTERVAL"));
        };
        // DuckDB types the call, and casts the source to the result type.
        let result = type_from_logical(func.return_type()?)?;
        // A bucket is either a fixed length or a whole number of calendar
        // months, never a blend of the two.
        let is_month_width = months != 0;
        if is_month_width && (days != 0 || micros != 0) {
            return Err(reject_time_bucket(
                "a bucket width in months cannot have a day or time component",
            ));
        }
        // A zoned day is a calendar day, not a fixed 24 hours, so it does not
        // blend with a time component either.
        if result == Type::TimestampTz && days != 0 && micros != 0 {
            return Err(reject_time_bucket(
                "a bucket width in days cannot have a time component on a TIMESTAMP WITH TIME ZONE",
            ));
        }
        let width_micros = count_interval_micros(days, micros)?;
        if months < 0 || (!is_month_width && width_micros <= 0) {
            return Err(reject_time_bucket(
                "the bucket width must be greater than 0",
            ));
        }
        let third_argument = match has_third_argument {
            true => Some(read_constant_value(params[2])?),
            false => None,
        };
        let grid = match is_month_width {
            true => BucketGrid::from_months(i64::from(months), read_origin_months(third_argument)?),
            false => BucketGrid::from_micros(width_micros, read_origin_micros(third_argument)?),
        };
        Ok(TimeBucket {
            grid,
            source: Box::new(Expression::from_handle(params[1])?),
            result,
        })
    }
}

/// The start of the fixed-length bucket grid `time_bucket` uses when the query
/// names none: Monday 2000-01-03, so week-long buckets start on Mondays.
const DEFAULT_BUCKET_ORIGIN_MICROS: i64 = 10_959 * MICROS_PER_DAY;

/// The start of the month bucket grid `time_bucket` uses when the query names
/// none: January 2000, counted in months since January 1970.
const DEFAULT_BUCKET_ORIGIN_MONTHS: i64 = 360;

fn reject_time_bucket(reason: &str) -> Error {
    Error::InvalidTimeBucket(reason.to_string())
}

/// Count the microseconds in an interval's day and time fields, taking a day
/// as a fixed 24 hours.
fn count_interval_micros(days: i32, micros: i64) -> Result<i64, Error> {
    i64::from(days)
        .checked_mul(MICROS_PER_DAY)
        .and_then(|day_micros| day_micros.checked_add(micros))
        .ok_or_else(|| reject_time_bucket("the interval is too long to count in microseconds"))
}

/// Read a `time_bucket` origin `DATE` as days since the epoch. An infinite
/// date starts no bucket.
fn read_origin_days(days: i32) -> Result<i64, Error> {
    if days == i32::MAX || days == -i32::MAX {
        return Err(reject_time_bucket("the origin cannot be infinite"));
    }
    Ok(i64::from(days))
}

/// Read a `time_bucket` origin timestamp as microseconds since the epoch. An
/// infinite timestamp starts no bucket.
fn read_origin_instant(micros: i64) -> Result<i64, Error> {
    if micros == i64::MAX || micros == -i64::MAX {
        return Err(reject_time_bucket("the origin cannot be infinite"));
    }
    Ok(micros)
}

/// The instant a fixed-length bucket grid starts at, in microseconds since the
/// epoch, given `time_bucket`'s optional third argument: an `origin` to start
/// at, or an `offset` that shifts the default start.
fn read_origin_micros(third_argument: Option<ScalarValue>) -> Result<i64, Error> {
    match third_argument {
        None => Ok(DEFAULT_BUCKET_ORIGIN_MICROS),
        Some(ScalarValue::Date(days)) => read_origin_days(days)?
            .checked_mul(MICROS_PER_DAY)
            .ok_or_else(|| reject_time_bucket("the origin is out of range for a TIMESTAMP")),
        Some(ScalarValue::Timestamp(micros) | ScalarValue::TimestampTz(micros)) => {
            read_origin_instant(micros)
        }
        Some(ScalarValue::Interval {
            months: 0,
            days,
            micros,
        }) => count_interval_micros(days, micros)?
            .checked_add(DEFAULT_BUCKET_ORIGIN_MICROS)
            .ok_or_else(|| reject_time_bucket("the offset is too long to count in microseconds")),
        Some(ScalarValue::Interval { .. }) => Err(reject_time_bucket(
            "an offset in months needs a bucket width in months",
        )),
        Some(other) => Err(reject_bucket_origin(other)),
    }
}

/// The month a calendar-month bucket grid starts at, in months since January
/// 1970, given `time_bucket`'s optional third argument. An `origin` counts by
/// the month it falls in.
fn read_origin_months(third_argument: Option<ScalarValue>) -> Result<i64, Error> {
    match third_argument {
        None => Ok(DEFAULT_BUCKET_ORIGIN_MONTHS),
        Some(ScalarValue::Date(days)) => Ok(count_epoch_months(read_origin_days(days)?)),
        Some(ScalarValue::Timestamp(micros) | ScalarValue::TimestampTz(micros)) => Ok(
            count_epoch_months(read_origin_instant(micros)?.div_euclid(MICROS_PER_DAY)),
        ),
        Some(ScalarValue::Interval {
            months,
            days: 0,
            micros: 0,
        }) => Ok(DEFAULT_BUCKET_ORIGIN_MONTHS + i64::from(months)),
        Some(ScalarValue::Interval { .. }) => Err(reject_time_bucket(
            "an offset in days or time needs a bucket width in days or time",
        )),
        Some(other) => Err(reject_bucket_origin(other)),
    }
}

fn reject_bucket_origin(third_argument: ScalarValue) -> Error {
    Error::InvalidTimeBucket(format!(
        "{third_argument:?} is neither an offset nor an origin"
    ))
}

impl DatePart {
    pub(crate) fn from_handle(
        kind: DatePartKind,
        func: FunctionHandle<'_>,
    ) -> Result<DatePart, Error> {
        let params = function_args(func, 1)?;
        Ok(DatePart {
            kind,
            source: Box::new(Expression::from_handle(params[0])?),
            return_type: Type::Int64,
        })
    }
}

impl TemporalConvert {
    /// `make_date(days)` → `Date32`.
    pub(crate) fn make_date(func: FunctionHandle<'_>) -> Result<TemporalConvert, Error> {
        Self::build(
            "make_date",
            Type::Date,
            DataType::Date32,
            DataType::Int32,
            func,
        )
    }

    /// `make_timestamp(microseconds)` → `Timestamp(Microsecond)`, DuckDB's own
    /// reading of the function.
    pub(crate) fn make_timestamp(func: FunctionHandle<'_>) -> Result<TemporalConvert, Error> {
        Self::build(
            "make_timestamp",
            Type::Timestamp,
            DataType::Timestamp(TimeUnit::Microsecond, None),
            DataType::Int64,
            func,
        )
    }

    fn build(
        name: &'static str,
        result: Type,
        target: DataType,
        via: DataType,
        func: FunctionHandle<'_>,
    ) -> Result<TemporalConvert, Error> {
        let params = function_args(func, 1)?;
        Ok(TemporalConvert {
            name,
            result,
            target,
            via,
            source: Box::new(Expression::from_handle(params[0])?),
        })
    }
}

impl IntervalArithmetic {
    /// Build a `date`/`timestamp` ± `INTERVAL` from a DuckDB `+`/`-` whose
    /// `params[interval_idx]` is an `INTERVAL` constant; the other operand is the
    /// temporal value. The caller ([`Function::from_handle`]) has already located
    /// the interval operand.
    pub(crate) fn from_handle(
        func: FunctionHandle<'_>,
        interval_idx: usize,
    ) -> Result<IntervalArithmetic, Error> {
        let params = function_args(func, 2)?;
        let op = match func.name()?.as_str() {
            "+" => ArithmeticOp::Add,
            "-" => ArithmeticOp::Sub,
            other => return Err(Error::UnsupportedScalarFunction(other.to_string())),
        };
        // DuckDB types the `+`/`-` itself, so its return type is the temporal
        // result (DATE for whole-day intervals, TIMESTAMP otherwise).
        let result = type_from_logical(func.return_type()?)?;
        let interval = match params[interval_idx].expression()? {
            DuckExpression::Constant(c) => match c.value()? {
                ScalarValue::Interval {
                    months,
                    days,
                    micros,
                } => IntervalParts {
                    months,
                    days,
                    micros,
                },
                _ => unreachable!("caller selected an interval constant"),
            },
            _ => unreachable!("caller selected an interval constant"),
        };
        let offset = interval_offset(&interval, &result)?;
        Ok(IntervalArithmetic {
            op,
            operand: Box::new(Expression::from_handle(params[1 - interval_idx])?),
            offset,
            result,
        })
    }
}

/// A DuckDB `INTERVAL`'s three independent components (months are
/// calendar-variable, so they stay separate from the fixed day/microsecond parts).
struct IntervalParts {
    months: i32,
    days: i32,
    micros: i64,
}

const SECS_PER_DAY: i64 = 86_400;
const MICROS_PER_SEC: i64 = 1_000_000;

/// Reduce an interval to a constant offset in the result's unit. Months/years are
/// calendar-variable (rejected); a `DATE` only takes whole-day intervals, and a
/// `TIMESTAMP` takes days plus the sub-day part as-is, both counted in the
/// microseconds a timestamp stores.
fn interval_offset(interval: &IntervalParts, result: &Type) -> Result<i64, Error> {
    if interval.months != 0 {
        return Err(Error::UnsupportedInterval(
            "month and year intervals require calendar arithmetic".to_string(),
        ));
    }
    match result {
        Type::Date => {
            if interval.micros != 0 {
                return Err(Error::UnsupportedInterval(
                    "sub-day interval applied to a DATE".to_string(),
                ));
            }
            Ok(interval.days as i64)
        }
        // A day is a fixed 24 hours here even for TIMESTAMPTZ: the session
        // zone is UTC, which has no DST boundary to stretch a day across.
        Type::Timestamp | Type::TimestampTz => {
            Ok(interval.days as i64 * SECS_PER_DAY * MICROS_PER_SEC + interval.micros)
        }
        other => Err(Error::UnsupportedInterval(format!(
            "interval arithmetic on a non-temporal {other}"
        ))),
    }
}

/// Extract the `(input, pattern, replacement)` shared by `regexp_replace` and
/// `regexp_jit_replace`. Only the three-argument first-match form is accepted: a
/// fourth `options` argument (e.g. 'g' for replace-all) would change the
/// semantics.
fn regex_replace_args(
    func: FunctionHandle<'_>,
) -> Result<(Box<Expression>, String, String), Error> {
    let params = function_args(func, 3)?;
    let replacement = constant_string(Expression::from_handle(params[2])?)?;
    let pattern = constant_string(Expression::from_handle(params[1])?)?;
    let input = Box::new(Expression::from_handle(params[0])?);
    Ok((input, pattern, replacement))
}

impl RegexpReplace {
    pub(crate) fn from_handle(func: FunctionHandle<'_>) -> Result<RegexpReplace, Error> {
        let (input, pattern, replacement) = regex_replace_args(func)?;
        Ok(RegexpReplace {
            input,
            pattern,
            replacement,
        })
    }
}

impl RegexpFullMatch {
    pub(crate) fn from_handle(func: FunctionHandle<'_>) -> Result<RegexpFullMatch, Error> {
        // Only the two-argument overload; DuckDB's third `options` argument
        // (`'i'`, `'s'`, …) is rejected as a wrong parameter count.
        let params = function_args(func, 2)?;
        let pattern = constant_string(Expression::from_handle(params[1])?)?;
        let input = Box::new(Expression::from_handle(params[0])?);
        Ok(RegexpFullMatch { input, pattern })
    }
}

impl RegexpJitReplace {
    pub(crate) fn from_handle(func: FunctionHandle<'_>) -> Result<RegexpJitReplace, Error> {
        let (input, pattern, replacement) = regex_replace_args(func)?;
        Ok(RegexpJitReplace {
            input,
            pattern,
            replacement,
        })
    }
}

// ---- Shared argument helpers ----

/// Collect a bound function's argument handles, rejecting a wrong argument count
/// the uniform way every scalar-function builder needs. Each builder then indexes
/// the returned `params` positionally.
fn function_args(func: FunctionHandle<'_>, expected: usize) -> Result<Vec<Expr<'_>>, Error> {
    let params: Vec<Expr<'_>> = func.children()?;
    if params.len() != expected {
        return Err(Error::InvalidParameterCount {
            function: func.name()?,
            expected,
            actual: params.len(),
        });
    }
    Ok(params)
}

/// Read the value of an argument that has to be a constant, such as
/// `time_bucket`'s bucket width.
fn read_constant_value(param: Expr<'_>) -> Result<ScalarValue, Error> {
    match param.expression()? {
        DuckExpression::Constant(constant) => Ok(constant.value()?),
        _ => Err(Error::UnsupportedScalarFunction(format!(
            "expected a constant, got {}",
            Expression::from_handle(param)?
        ))),
    }
}

/// Extract a constant string argument (e.g. a regex pattern or a `date_trunc`
/// unit) from a built expression. The error names the offending argument, so a
/// caller needs to supply nothing beyond the expression itself.
fn constant_string(e: Expression) -> Result<String, Error> {
    match e {
        Expression::Constant(scalar) => {
            let (arr, _) = scalar.get();
            Ok(arr
                .as_string_view_opt()
                .ok_or_else(|| {
                    Error::UnsupportedScalarFunction(format!(
                        "expected a string constant, got a {} constant",
                        arr.data_type()
                    ))
                })?
                .value(0)
                .to_string())
        }
        other => Err(Error::UnsupportedScalarFunction(format!(
            "expected a string constant, got {other}"
        ))),
    }
}

/// Extract a constant integer argument (e.g. a substring position) from a
/// built expression, whichever integer width DuckDB bound it at.
fn constant_int64(e: Expression) -> Result<i64, Error> {
    let Expression::Constant(scalar) = e else {
        return Err(Error::UnsupportedScalarFunction(format!(
            "expected an integer constant, got {e}"
        )));
    };
    let (arr, _) = scalar.get();
    if arr.is_null(0) {
        return Err(Error::UnsupportedScalarFunction(
            "expected an integer constant, got NULL".to_string(),
        ));
    }
    if let Some(values) = arr.as_primitive_opt::<arrow_array::types::Int64Type>() {
        return Ok(values.value(0));
    }
    if let Some(values) = arr.as_primitive_opt::<arrow_array::types::Int32Type>() {
        return Ok(values.value(0) as i64);
    }
    Err(Error::UnsupportedScalarFunction(format!(
        "expected an integer constant, got a {} constant",
        arr.data_type()
    )))
}
