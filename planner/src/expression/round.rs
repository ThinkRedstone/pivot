//! [`Round`] — SQL `floor(value)` and `round(value[, digits])`.

use super::{Expression, Function};
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use crate::types::{MAX_DECIMAL64_PRECISION, Type, physical_arrow_type};
use arrow_array::cast::AsArray;
use arrow_array::types::{
    ArrowPrimitiveType, Decimal64Type, Decimal128Type, Float32Type, Float64Type, Int8Type,
    Int16Type, Int32Type, Int64Type,
};
use arrow_array::{ArrayRef, ArrowNativeTypeOp, PrimitiveArray, RecordBatch, Scalar};
use arrow_buffer::ArrowNativeType;
use std::fmt::{self, Display};
use std::sync::Arc;

/// The message a query fails with when rounding a whole number to tens,
/// hundreds and so on lands on a value its type cannot hold.
const ROUNDED_VALUE_OUT_OF_RANGE: &str =
    "round: the rounded value is out of range for the value's type";

/// Where a [`Round`] moves each value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rounding {
    /// `floor(value)`: the largest whole number at or below the value.
    Floor,
    /// `round(value, digits)`: the nearest value with `digits` digits after
    /// the decimal point, a tie going away from zero. Negative `digits` round
    /// left of the decimal point, so `-2` rounds to hundreds.
    Nearest { digits: i32 },
}

/// SQL `floor(value)` and `round(value[, digits])` over a float, a decimal or
/// an integer. The digit count is a constant, and a `round` without one keeps
/// no digits after the decimal point.
#[derive(Debug, Clone)]
pub struct Round {
    pub rounding: Rounding,
    pub input: Box<Expression>,
    /// The bound result type. A float or an integer keeps its type; a decimal
    /// keeps its precision, and of its scale only the digits that survive.
    pub return_type: Type,
}

impl Display for Round {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.rounding {
            Rounding::Floor => write!(f, "floor({})", self.input),
            Rounding::Nearest { digits } => write!(f, "round({}, {digits})", self.input),
        }
    }
}

impl Round {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        match (self.input.result_type()?, self.rounding) {
            (Type::Float32, Rounding::Floor) => {
                self.compile_kernel::<Float32Type>(|values| values.unary(f32::floor))
            }
            (Type::Float32, Rounding::Nearest { digits: 0 }) => {
                self.compile_kernel::<Float32Type>(|values| values.unary(f32::round))
            }
            (Type::Float32, Rounding::Nearest { digits }) => self
                .compile_float_digits::<Float32Type>(digits, f64::from, |rounded| rounded as f32),
            (Type::Float64, Rounding::Floor) => {
                self.compile_kernel::<Float64Type>(|values| values.unary(f64::floor))
            }
            (Type::Float64, Rounding::Nearest { digits: 0 }) => {
                self.compile_kernel::<Float64Type>(|values| values.unary(f64::round))
            }
            (Type::Float64, Rounding::Nearest { digits }) => {
                self.compile_float_digits::<Float64Type>(digits, |value| value, |rounded| rounded)
            }
            (Type::Decimal { precision, scale }, _) if precision <= MAX_DECIMAL64_PRECISION => {
                self.compile_fixed_point::<Decimal64Type>(precision - scale as u8, scale)
            }
            (Type::Decimal { precision, scale }, _) => {
                self.compile_fixed_point::<Decimal128Type>(precision - scale as u8, scale)
            }
            // An integer has as many whole digits as the largest value of its
            // type.
            (Type::Int8, _) => self.compile_fixed_point::<Int8Type>(3, 0),
            (Type::Int16, _) => self.compile_fixed_point::<Int16Type>(5, 0),
            (Type::Int32, _) => self.compile_fixed_point::<Int32Type>(10, 0),
            (Type::Int64, _) => self.compile_fixed_point::<Int64Type>(19, 0),
            // A 128-bit integer is carried as a decimal with no scale.
            (Type::Int128, _) => self.compile_fixed_point::<Decimal128Type>(39, 0),
            _ => Err(compile::Error::UnsupportedExpression(Expression::Function(
                Function::Round(self.clone()),
            ))),
        }
    }

    /// Compile `round(value, digits)` over a column of the float `T`, a tie
    /// going away from zero. The digits are scaled in double precision, which
    /// `widen` and `narrow` carry a single-precision value into and out of.
    ///
    /// The results that are not ordinary follow DuckDB's own `round`. DuckDB
    /// folds a call over constants before the plan gets here, so a column has
    /// to round the way it does for the two to agree.
    fn compile_float_digits<T: ArrowPrimitiveType>(
        &self,
        digits: i32,
        widen: impl Fn(T::Native) -> f64 + Send + Sync + Copy + 'static,
        narrow: impl Fn(f64) -> T::Native + Send + Sync + Copy + 'static,
    ) -> Result<ExprFn, compile::Error> {
        let scale = 10f64.powf(f64::from(digits.unsigned_abs()));
        if digits >= 0 {
            self.compile_kernel::<T>(move |values| {
                values.unary(|value| {
                    let value = widen(value);
                    let rounded = (value * scale).round() / scale;
                    // A value too large to scale has no digits that far right
                    // of the decimal point, and an infinity or a NaN has no
                    // digits at all: each stays itself.
                    narrow(if rounded.is_finite() { rounded } else { value })
                })
            })
        } else {
            self.compile_kernel::<T>(move |values| {
                values.unary(|value| {
                    let rounded = (widen(value) / scale).round() * scale;
                    // A step beyond what a float holds is further than any
                    // value reaches halfway to, so it rounds to zero. Every
                    // other result that is not finite is zero as well: an
                    // infinite or NaN value, and one rounded past the
                    // largest float.
                    narrow(if rounded.is_finite() { rounded } else { 0.0 })
                })
            })
        }
    }

    /// Compile the rounding of a column of integers that count units of
    /// `10^-scale`: a decimal's unscaled values, or plain integers with a
    /// `scale` of 0. `whole_digits` is how many digits a value of the type
    /// can have left of the decimal point.
    fn compile_fixed_point<T: ArrowPrimitiveType>(
        &self,
        whole_digits: u8,
        scale: i8,
    ) -> Result<ExprFn, compile::Error> {
        let kept_digits = match self.rounding {
            Rounding::Floor => 0,
            Rounding::Nearest { digits } => digits,
        };
        let dropped_digits = i32::from(scale).saturating_sub(kept_digits);
        if dropped_digits <= 0 {
            // The value has no digit past the ones kept, so it is already
            // rounded.
            return self.input.compile();
        }
        // How many digits left of the decimal point become zeros.
        let zeroed_digits = kept_digits.min(0).unsigned_abs();
        if zeroed_digits >= u32::from(whole_digits) {
            // Every whole digit the type has is rounded away, which leaves
            // zero. A value past half of such a step is zero too, not the
            // step, which is how DuckDB rounds it: it folds a call over
            // constants before the plan gets here, so a column has to agree.
            return self.compile_kernel::<T>(|values| values.unary(|_| T::Native::ZERO));
        }
        let ten = T::Native::usize_as(10);
        // Rounding drops the low digits by dividing by this step, which the
        // type holds: the check above bounds the digits it drops left of the
        // decimal point, and a decimal's scale the ones right of it.
        let step = ten.pow_wrapping(dropped_digits.unsigned_abs());
        let Rounding::Nearest { .. } = self.rounding else {
            return self.compile_kernel::<T>(move |values| {
                values.unary(|unscaled| {
                    let quotient = unscaled.div_wrapping(step);
                    // Integer division truncates toward zero, which leaves a
                    // negative value that has a remainder one above its floor.
                    if unscaled.mod_wrapping(step) < T::Native::ZERO {
                        quotient.sub_wrapping(T::Native::ONE)
                    } else {
                        quotient
                    }
                })
            });
        };
        let half_step = step.div_wrapping(T::Native::usize_as(2));
        if zeroed_digits == 0 {
            return self.compile_kernel::<T>(move |values| {
                values.unary(|unscaled| Round::divide_to_nearest(unscaled, step, half_step))
            });
        }
        // Digits dropped left of the decimal point are filled back with zeros.
        let restored_zeros = ten.pow_wrapping(zeroed_digits);
        if scale > 0 {
            // A value with digits after the decimal point rounds to at most
            // `10^whole_digits`, which its precision holds.
            return self.compile_kernel::<T>(move |values| {
                values.unary(|unscaled| {
                    Round::divide_to_nearest(unscaled, step, half_step).mul_wrapping(restored_zeros)
                })
            });
        }
        // A whole number can round up to one digit more than its type holds.
        let largest = match self.return_type {
            Type::Decimal { precision, .. } => ten
                .pow_wrapping(u32::from(precision))
                .sub_wrapping(T::Native::ONE),
            _ => T::Native::MAX_TOTAL_ORDER,
        };
        self.compile_kernel::<T>(move |values| {
            values
                .try_unary(|unscaled| {
                    Round::divide_to_nearest(unscaled, step, half_step)
                        .mul_checked(restored_zeros)
                        .ok()
                        .filter(|rounded| *rounded <= largest && *rounded >= largest.neg_wrapping())
                        .ok_or(ROUNDED_VALUE_OUT_OF_RANGE)
                })
                // The overflow is data-dependent and a compiled expression
                // has no error channel, so it panics and the dataflow fails
                // the query with the message.
                .unwrap_or_else(|message| panic!("{message}"))
        })
    }

    /// Divide by `step`, rounding the quotient to the nearest whole number
    /// with a tie going away from zero. `half_step` is half of `step`.
    #[inline(always)]
    fn divide_to_nearest<Native: ArrowNativeTypeOp>(
        unscaled: Native,
        step: Native,
        half_step: Native,
    ) -> Native {
        let quotient = unscaled.div_wrapping(step);
        let remainder = unscaled.mod_wrapping(step);
        if remainder >= half_step {
            quotient.add_wrapping(Native::ONE)
        } else if remainder <= half_step.neg_wrapping() {
            quotient.sub_wrapping(Native::ONE)
        } else {
            quotient
        }
    }

    /// Compile to `kernel` run over each batch's input column, which is
    /// carried as an array of `T`. The result takes the bound result type,
    /// which for a decimal names the scale the rounding left it with.
    fn compile_kernel<T: ArrowPrimitiveType>(
        &self,
        kernel: impl Fn(&PrimitiveArray<T>) -> PrimitiveArray<T> + Send + Sync + Clone + 'static,
    ) -> Result<ExprFn, compile::Error> {
        let result_arrow = physical_arrow_type(&self.return_type);
        let input_builder = self.input.compile()?;
        Ok(Box::new(move || {
            let mut input_expr = input_builder();
            let kernel = kernel.clone();
            let result_arrow = result_arrow.clone();
            Box::new(move |batch: &RecordBatch| {
                let input = input_expr(batch);
                let (values, is_scalar) = input.as_datum().get();
                let rounded = kernel(values.as_primitive::<T>());
                let rounded = Arc::new(rounded.with_data_type(result_arrow.clone())) as ArrayRef;
                // A constant input stays a constant, so whatever reads the
                // result still broadcasts it across the batch.
                if is_scalar {
                    ExprResult::Scalar(Scalar::new(rounded))
                } else {
                    ExprResult::Array(rounded)
                }
            }) as ExprEvalFn
        }))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use crate::types::Type;
    use arrow_array::cast::AsArray;
    use arrow_array::types::{Float64Type, Int32Type};
    use arrow_array::{
        ArrayRef, Decimal64Array, Decimal128Array, Float32Array, Float64Array, Int8Array,
        Int64Array,
    };
    use rstest::rstest;
    use std::sync::Arc;

    /// Register `numbers` with one row whose `x` column holds a DOUBLE.
    fn add_double(testing_planner: &TestingPlanner, value: f64) {
        let column = Arc::new(Float64Array::from(vec![value])) as ArrayRef;
        testing_planner.add_table("numbers", &[("x", Type::Float64, column)]);
    }

    /// Register `numbers` with one row whose `x` column holds a DECIMAL(15, 2)
    /// of `hundredths` hundredths.
    fn add_decimal(testing_planner: &TestingPlanner, hundredths: i64) {
        let column = Decimal64Array::from(vec![hundredths])
            .with_precision_and_scale(15, 2)
            .unwrap();
        let column_type = Type::Decimal {
            precision: 15,
            scale: 2,
        };
        testing_planner.add_table("numbers", &[("x", column_type, Arc::new(column))]);
    }

    #[rstest]
    #[case::down("floor(x)", 2.7, 2.0)]
    #[case::down_below_zero("floor(x)", -2.1, -3.0)]
    #[case::nearest("round(x)", 2.4, 2.0)]
    #[case::tie_away_from_zero("round(x)", 2.5, 3.0)]
    #[case::tie_away_from_zero_below_zero("round(x)", -2.5, -3.0)]
    #[case::digits_after_the_point("round(x, 2)", 1.23456, 1.23)]
    #[case::digits_before_the_point("round(x, -2)", 1250.0, 1300.0)]
    #[case::more_digits_than_a_double_has("round(x, 400)", 1.5, 1.5)]
    #[case::a_step_beyond_every_double("round(x, -400)", 1.5, 0.0)]
    fn rounds_a_double(
        mut testing_planner: TestingPlanner,
        #[case] call: &str,
        #[case] value: f64,
        #[case] expected: f64,
    ) {
        add_double(&testing_planner, value);

        let rows = run(&mut testing_planner, &format!("SELECT {call} FROM numbers"));

        assert_eq!(only_column(&rows[0]).as_f64().unwrap(), expected);
    }

    #[rstest]
    #[case::whole_number("round(x)", f64::INFINITY, f64::INFINITY)]
    #[case::digits_after_the_point("round(x, 2)", f64::NEG_INFINITY, f64::NEG_INFINITY)]
    #[case::digits_before_the_point("round(x, -1)", f64::INFINITY, 0.0)]
    #[case::not_a_number_before_the_point("round(x, -1)", f64::NAN, 0.0)]
    fn rounds_a_double_that_is_not_finite(
        mut testing_planner: TestingPlanner,
        #[case] call: &str,
        #[case] value: f64,
        #[case] expected: f64,
    ) {
        add_double(&testing_planner, value);

        let batches = run_batches(&mut testing_planner, &format!("SELECT {call} FROM numbers"));

        let expected = Arc::new(Float64Array::from(vec![expected])) as ArrayRef;
        assert_eq!(batches[0].column(0), &expected);
    }

    #[rstest]
    #[case::infinity(f64::INFINITY, "'inf'", -1)]
    #[case::not_a_number(f64::NAN, "'nan'", -1)]
    #[case::past_the_largest_double(1.5e308, "1.5e308", -308)]
    fn rounds_a_double_column_like_a_constant(
        mut testing_planner: TestingPlanner,
        #[case] value: f64,
        #[case] constant: &str,
        #[case] digits: i32,
    ) {
        add_double(&testing_planner, value);

        let batches = run_batches(
            &mut testing_planner,
            &format!(
                "SELECT round(x, {digits}), round(CAST({constant} AS DOUBLE), {digits})
                 FROM numbers"
            ),
        );

        assert_eq!(batches[0].column(0), batches[0].column(1));
    }

    #[rstest]
    #[case::down("floor(x)", 1.75, 1.0)]
    #[case::nearest("round(x)", 2.5, 3.0)]
    #[case::digits_after_the_point("round(x, 1)", 0.25, 0.3)]
    fn rounding_a_real_yields_a_real(
        mut testing_planner: TestingPlanner,
        #[case] call: &str,
        #[case] value: f32,
        #[case] expected: f32,
    ) {
        let column = Arc::new(Float32Array::from(vec![value])) as ArrayRef;
        testing_planner.add_table("numbers", &[("x", Type::Float32, column)]);

        let batches = run_batches(&mut testing_planner, &format!("SELECT {call} FROM numbers"));

        let expected = Arc::new(Float32Array::from(vec![expected])) as ArrayRef;
        assert_eq!(batches[0].column(0), &expected);
    }

    #[rstest]
    #[case::down("floor(x)", 1250, 12, 0)]
    #[case::down_below_zero("floor(x)", -1201, -13, 0)]
    #[case::down_from_a_whole_number_below_zero("floor(x)", -1200, -12, 0)]
    #[case::nearest("round(x)", 1249, 12, 0)]
    #[case::tie_away_from_zero("round(x)", 1250, 13, 0)]
    #[case::tie_away_from_zero_below_zero("round(x)", -1250, -13, 0)]
    #[case::digits_after_the_point("round(x, 1)", 1255, 126, 1)]
    #[case::every_digit_it_has("round(x, 2)", 1255, 1255, 2)]
    #[case::more_digits_than_it_has("round(x, 5)", 1255, 1255, 2)]
    #[case::digits_before_the_point("round(x, -1)", 1500, 20, 0)]
    #[case::a_step_beyond_its_precision("round(x, -30)", 1500, 0, 0)]
    fn rounds_a_decimal(
        mut testing_planner: TestingPlanner,
        #[case] call: &str,
        #[case] hundredths: i64,
        #[case] expected_unscaled: i64,
        #[case] expected_scale: i8,
    ) {
        add_decimal(&testing_planner, hundredths);

        let batches = run_batches(&mut testing_planner, &format!("SELECT {call} FROM numbers"));

        // The result keeps the precision, and the scale the rounding left.
        let expected = Decimal64Array::from(vec![expected_unscaled])
            .with_precision_and_scale(15, expected_scale)
            .unwrap();
        assert_eq!(batches[0].column(0), &(Arc::new(expected) as ArrayRef));
    }

    #[rstest]
    fn rounds_a_decimal_column_like_a_constant(mut testing_planner: TestingPlanner) {
        add_decimal(&testing_planner, 999_999_999_999_999);

        // Every one of the 13 whole digits is rounded away.
        let batches = run_batches(
            &mut testing_planner,
            "SELECT round(x, -13), round(CAST(9999999999999.99 AS DECIMAL(15, 2)), -13)
             FROM numbers",
        );

        assert_eq!(batches[0].column(0), batches[0].column(1));
    }

    /// Register `numbers` with one row whose `x` column holds a DECIMAL(5, 0).
    fn add_whole_decimal(testing_planner: &TestingPlanner, value: i64) {
        let column = Decimal64Array::from(vec![value])
            .with_precision_and_scale(5, 0)
            .unwrap();
        let column_type = Type::Decimal {
            precision: 5,
            scale: 0,
        };
        testing_planner.add_table("numbers", &[("x", column_type, Arc::new(column))]);
    }

    #[rstest]
    fn rounds_a_whole_decimal_to_hundreds(mut testing_planner: TestingPlanner) {
        add_whole_decimal(&testing_planner, 12_350);

        let batches = run_batches(&mut testing_planner, "SELECT round(x, -2) FROM numbers");

        let expected = Decimal64Array::from(vec![12_400i64])
            .with_precision_and_scale(5, 0)
            .unwrap();
        assert_eq!(batches[0].column(0), &(Arc::new(expected) as ArrayRef));
    }

    #[rstest]
    fn fails_the_query_when_a_rounded_decimal_outgrows_its_precision(
        mut testing_planner: TestingPlanner,
    ) {
        add_whole_decimal(&testing_planner, 99_999);

        let error = run_expecting_error(&mut testing_planner, "SELECT round(x, -1) FROM numbers");

        assert!(error.contains("out of range"), "{error}");
    }

    #[rstest]
    fn rounds_a_decimal_wider_than_18_digits(mut testing_planner: TestingPlanner) {
        let column = Decimal128Array::from(vec![-12_345_678_901_234_567_890_555i128])
            .with_precision_and_scale(30, 3)
            .unwrap();
        let column_type = Type::Decimal {
            precision: 30,
            scale: 3,
        };
        testing_planner.add_table("numbers", &[("x", column_type, Arc::new(column))]);

        let batches = run_batches(&mut testing_planner, "SELECT round(x, 2) FROM numbers");

        let expected = Decimal128Array::from(vec![-1_234_567_890_123_456_789_056i128])
            .with_precision_and_scale(30, 2)
            .unwrap();
        assert_eq!(batches[0].column(0), &(Arc::new(expected) as ArrayRef));
    }

    #[rstest]
    #[case::whole_number("round(a)", vec![1, 2, 3, 4, 5])]
    #[case::digits_after_the_point("round(a, 2)", vec![1, 2, 3, 4, 5])]
    #[case::tens("round(c + a, -1)", vec![100, 200, 300, 400, 510])]
    #[case::thousands("round(c, -3)", vec![0, 0, 0, 0, 1000])]
    #[case::a_step_beyond_every_integer("round(c, -10)", vec![0, 0, 0, 0, 0])]
    fn rounding_an_integer_yields_an_integer(
        mut testing_planner: TestingPlanner,
        #[case] call: &str,
        #[case] expected: Vec<i32>,
    ) {
        let batches = run_batches(
            &mut testing_planner,
            &format!("SELECT {call} FROM example_table"),
        );

        let mut rounded = batches[0]
            .column(0)
            .as_primitive::<Int32Type>()
            .values()
            .to_vec();
        rounded.sort();
        assert_eq!(rounded, expected);
    }

    #[rstest]
    fn flooring_an_integer_yields_a_double(mut testing_planner: TestingPlanner) {
        let batches = run_batches(&mut testing_planner, "SELECT floor(a) FROM example_table");

        let mut floored = batches[0]
            .column(0)
            .as_primitive::<Float64Type>()
            .values()
            .to_vec();
        floored.sort_by(f64::total_cmp);
        assert_eq!(floored, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[rstest]
    fn rounds_the_integers_around_a_null(mut testing_planner: TestingPlanner) {
        let column = Arc::new(Int8Array::from(vec![Some(4), None, Some(15)])) as ArrayRef;
        testing_planner.add_table("numbers", &[("x", Type::Int8, column)]);

        let batches = run_batches(&mut testing_planner, "SELECT round(x, -1) FROM numbers");

        let expected = Arc::new(Int8Array::from(vec![Some(0), None, Some(20)])) as ArrayRef;
        assert_eq!(batches[0].column(0), &expected);
    }

    #[rstest]
    #[case::tie_away_from_zero(-25, -30)]
    #[case::nearest(-24, -20)]
    fn rounds_an_integer_below_zero(
        mut testing_planner: TestingPlanner,
        #[case] value: i64,
        #[case] expected: i64,
    ) {
        let column = Arc::new(Int64Array::from(vec![value])) as ArrayRef;
        testing_planner.add_table("numbers", &[("x", Type::Int64, column)]);

        let rows = run(&mut testing_planner, "SELECT round(x, -1) FROM numbers");

        assert_eq!(only_column(&rows[0]).as_i64().unwrap(), expected);
    }

    #[rstest]
    fn fails_the_query_when_a_rounded_integer_is_out_of_range(mut testing_planner: TestingPlanner) {
        let column = Arc::new(Int8Array::from(vec![127i8])) as ArrayRef;
        testing_planner.add_table("numbers", &[("x", Type::Int8, column)]);

        let error = run_expecting_error(&mut testing_planner, "SELECT round(x, -1) FROM numbers");

        assert!(error.contains("out of range"), "{error}");
    }

    #[rstest]
    #[case::floor("floor(b)")]
    #[case::round("round(b, -1)")]
    fn rounding_a_null_yields_a_null(mut testing_planner: TestingPlanner, #[case] call: &str) {
        let rows = run(
            &mut testing_planner,
            &format!("SELECT {call} AS rounded FROM nullable_table WHERE a = 2"),
        );

        assert!(rows[0].get("rounded").is_none(), "{rows:?}");
    }

    #[rstest]
    fn groups_by_a_rounded_value(mut testing_planner: TestingPlanner) {
        let mut rows = run(
            &mut testing_planner,
            "SELECT round(c, -3) AS thousands, count(*) AS row_count
             FROM example_table GROUP BY round(c, -3)",
        );

        rows.sort_by_key(|row| row["thousands"].as_i64().unwrap());
        assert_eq!(
            rows,
            serde_json::json!([
                {"thousands": 0, "row_count": 4},
                {"thousands": 1000, "row_count": 1}
            ])
            .as_array()
            .unwrap()
            .clone()
        );
    }

    #[rstest]
    fn rejects_digits_that_are_not_a_constant(mut testing_planner: TestingPlanner) {
        add_double(&testing_planner, 1.5);

        let error = testing_planner
            .plan("SELECT round(x, CAST(x AS INTEGER)) FROM numbers")
            .err()
            .unwrap()
            .to_string();

        assert!(error.contains("expected an integer constant"), "{error}");
    }
}
