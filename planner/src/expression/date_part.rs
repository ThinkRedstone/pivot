//! [`DatePart`] — SQL `extract(<part> FROM ts)` and its [`DatePartKind`].

use super::Expression;
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use crate::types::Type;
use arrow_array::cast::AsArray;
use arrow_array::types::{Date32Type, Int64Type};
use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::DataType;
use std::fmt::{self, Display};
use std::sync::Arc;

/// Seconds in a day / hour, for the time-of-day parts.
const SECS_PER_DAY: i64 = 86_400;
const SECS_PER_HOUR: i64 = 3_600;
/// Microseconds in a second, the unit a timestamp counts in.
const MICROS_PER_SEC: i64 = 1_000_000;

/// Convert a day count relative to the Unix epoch (1970-01-01) into a
/// `(year, month, day)` civil date. Howard Hinnant's `civil_from_days`
/// (<http://howardhinnant.github.io/date_algorithms.html>); valid for the full
/// proleptic Gregorian range, including negative (pre-epoch) day counts.
pub(super) fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // day of era, [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    (y + i64::from(m <= 2), m, d)
}

/// Inverse of [`civil_from_days`]: the epoch-relative day count for a civil
/// date. Used to derive day-of-year and ISO week.
pub(super) fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400; // [0, 399]
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy; // [0, 146096]
    era * 146_097 + doe - 719_468
}

/// Number of ISO 8601 weeks (52 or 53) in the given year. A year has 53 weeks
/// iff it starts on a Thursday, or it is a leap year starting on a Wednesday.
fn iso_weeks_in_year(y: i64) -> i64 {
    let p = |y: i64| (y + y.div_euclid(4) - y.div_euclid(100) + y.div_euclid(400)).rem_euclid(7);
    if p(y) == 4 || p(y - 1) == 3 { 53 } else { 52 }
}

/// ISO 8601 week-of-year (1–53) for an epoch-relative day count. Week 1 is the
/// week containing the year's first Thursday; days before it belong to the
/// prior year's last week, and the year's tail can roll into week 1.
fn iso_week(days: i64) -> i64 {
    let (y, ..) = civil_from_days(days);
    let ordinal = days - days_from_civil(y, 1, 1) + 1;
    let iso_dow = (days + 3).rem_euclid(7) + 1;
    let week = (ordinal - iso_dow + 10).div_euclid(7);
    if week < 1 {
        iso_weeks_in_year(y - 1)
    } else if week > iso_weeks_in_year(y) {
        1
    } else {
        week
    }
}

/// Which field of a timestamp a [`DatePart`] extracts. DuckDB lowers
/// `extract(<part> FROM ts)` to a scalar function named after the part (e.g.
/// `minute`, `year`); this enumerates the parts we evaluate from a timestamp's
/// Int64 epoch-microseconds representation. See [`DatePart`]'s compile impl for
/// the per-part arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatePartKind {
    /// Whole seconds since the Unix epoch.
    Epoch,
    /// Second of minute, 0–59.
    Second,
    /// Millisecond of minute, 0–59999: the whole seconds plus the stored
    /// fraction, as DuckDB reports it.
    Millisecond,
    /// Microsecond of minute, 0–59999999, likewise carrying the fraction.
    Microsecond,
    /// Minute of hour, 0–59.
    Minute,
    /// Hour of day, 0–23.
    Hour,
    /// Day of month, 1–31.
    Day,
    /// Month of year, 1–12.
    Month,
    /// Quarter of year, 1–4.
    Quarter,
    /// Full year (e.g. 2024).
    Year,
    /// Decade — `year / 10` (e.g. 202 for 2024).
    Decade,
    /// Century — e.g. 21 for years 2001–2100.
    Century,
    /// Millennium — e.g. 3 for years 2001–3000.
    Millennium,
    /// Day of week, 0 (Sunday)–6 (Saturday).
    DayOfWeek,
    /// ISO day of week, 1 (Monday)–7 (Sunday).
    IsoDayOfWeek,
    /// Day of year, 1–366.
    DayOfYear,
    /// ISO 8601 week of year, 1–53.
    Week,
}

impl DatePartKind {
    /// Resolve a DuckDB scalar-function name (as `extract` lowers it) to a part.
    /// Includes the common DuckDB aliases (`dow`, `doy`, `weekofyear`).
    pub fn from_function_name(name: &str) -> Option<Self> {
        Some(match name {
            "epoch" => DatePartKind::Epoch,
            "second" => DatePartKind::Second,
            "millisecond" => DatePartKind::Millisecond,
            "microsecond" => DatePartKind::Microsecond,
            "minute" => DatePartKind::Minute,
            "hour" => DatePartKind::Hour,
            "day" => DatePartKind::Day,
            "month" => DatePartKind::Month,
            "quarter" => DatePartKind::Quarter,
            "year" => DatePartKind::Year,
            "decade" => DatePartKind::Decade,
            "century" => DatePartKind::Century,
            "millennium" => DatePartKind::Millennium,
            "dayofweek" | "dow" => DatePartKind::DayOfWeek,
            "isodow" => DatePartKind::IsoDayOfWeek,
            "dayofyear" | "doy" => DatePartKind::DayOfYear,
            "week" | "weekofyear" => DatePartKind::Week,
            _ => return None,
        })
    }

    /// The canonical DuckDB function name for this part, used for plan display.
    pub fn name(&self) -> &'static str {
        match self {
            DatePartKind::Epoch => "epoch",
            DatePartKind::Second => "second",
            DatePartKind::Millisecond => "millisecond",
            DatePartKind::Microsecond => "microsecond",
            DatePartKind::Minute => "minute",
            DatePartKind::Hour => "hour",
            DatePartKind::Day => "day",
            DatePartKind::Month => "month",
            DatePartKind::Quarter => "quarter",
            DatePartKind::Year => "year",
            DatePartKind::Decade => "decade",
            DatePartKind::Century => "century",
            DatePartKind::Millennium => "millennium",
            DatePartKind::DayOfWeek => "dayofweek",
            DatePartKind::IsoDayOfWeek => "isodow",
            DatePartKind::DayOfYear => "dayofyear",
            DatePartKind::Week => "week",
        }
    }
}

/// SQL `extract(<part> FROM source)` — a timestamp field accessor. DuckDB
/// lowers each part to a scalar function (`minute`, `year`, …); a timestamp is
/// stored as Int64 epoch *seconds* (see [`Type::Timestamp`]), so every part is
/// a pure integer computation. A `DATE` source (see [`Type::Date`]) stores days
/// instead, and is scaled to those seconds before the parts are read. See its
/// compile impl.
///
/// [`Type::Timestamp`]: crate::types::Type::Timestamp
/// [`Type::Date`]: crate::types::Type::Date
#[derive(Debug, Clone)]
pub struct DatePart {
    pub kind: DatePartKind,
    pub source: Box<Expression>,
    pub return_type: Type,
}

impl Display for DatePart {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.kind.name(), self.source)
    }
}

impl DatePart {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        // A timestamp is stored as Int64 epoch *microseconds* (UTC), so every
        // part is a pure integer computation. Euclidean div/rem keep the
        // time-of-day and calendar fields well-defined for pre-epoch (negative)
        // timestamps, matching DuckDB's `extract(<part> FROM ...)`.
        let kind = self.kind;
        let source_builder = self.source.compile()?;
        Ok(Box::new(move || {
            let mut source_expr = source_builder();
            Box::new(move |batch: &RecordBatch| {
                let src = source_expr(batch);
                let (arr, _) = src.as_datum().get();
                // A DATE column stores whole days since the epoch, so scale it
                // to the epoch microseconds every arm below computes on. The
                // time-of-day parts of a date then read as midnight, which is
                // what DuckDB returns for them.
                let micros: Int64Array = match arr.data_type() {
                    DataType::Date32 => arr
                        .as_primitive::<Date32Type>()
                        .unary(|days: i32| i64::from(days) * SECS_PER_DAY * MICROS_PER_SEC),
                    _ => arrow::compute::cast(arr, &DataType::Int64)
                        .unwrap()
                        .as_primitive::<Int64Type>()
                        .clone(),
                };
                let vals = &micros;
                // Dispatch on the part ONCE per batch, then run a single
                // monomorphic, branch-free row loop per arm — so e.g. `minute`
                // compiles to exactly its two-op loop with no per-row `kind`
                // test (no reliance on the optimizer hoisting a loop-invariant
                // branch). The civil-date parts share `civil_from_days`.
                use crate::expression::DatePartKind::*;
                macro_rules! map_part {
                    ($f:expr) => {{
                        let out: Int64Array = vals.iter().map(|v| v.map($f)).collect();
                        out
                    }};
                }
                // Each arm divides the stored microseconds by its own field's
                // width in one step rather than reducing to whole seconds
                // first: flooring twice is flooring once by the product, so
                // `(t / 1e6) / 60` is `t / 6e7`, and one division per row is
                // what the field costs.
                let second = |t: i64| t.div_euclid(MICROS_PER_SEC);
                let day = |t: i64| t.div_euclid(SECS_PER_DAY * MICROS_PER_SEC);
                let out = match kind {
                    Epoch => map_part!(|t: i64| second(t)),
                    Second => map_part!(|t: i64| second(t).rem_euclid(60)),
                    Millisecond => map_part!(|t: i64| t.div_euclid(1_000).rem_euclid(60_000)),
                    Microsecond => map_part!(|t: i64| t.rem_euclid(60 * MICROS_PER_SEC)),
                    Minute => map_part!(|t: i64| t.div_euclid(60 * MICROS_PER_SEC).rem_euclid(60)),
                    Hour => {
                        map_part!(|t: i64| t
                            .div_euclid(SECS_PER_HOUR * MICROS_PER_SEC)
                            .rem_euclid(24))
                    }
                    // 0 = Sunday … 6 = Saturday. The epoch day (1970-01-01) was a
                    // Thursday (4), so `(days + 4) mod 7` rebases to Sunday = 0.
                    DayOfWeek => map_part!(|t: i64| (day(t) + 4).rem_euclid(7)),
                    // 1 = Monday … 7 = Sunday.
                    IsoDayOfWeek => map_part!(|t: i64| (day(t) + 3).rem_euclid(7) + 1),
                    Day => map_part!(|t: i64| civil_from_days(day(t)).2),
                    Month => map_part!(|t: i64| civil_from_days(day(t)).1),
                    Quarter => map_part!(|t: i64| (civil_from_days(day(t)).1 - 1) / 3 + 1),
                    Year => map_part!(|t: i64| civil_from_days(day(t)).0),
                    Decade => map_part!(|t: i64| civil_from_days(day(t)).0.div_euclid(10)),
                    Century => {
                        map_part!(|t: i64| (civil_from_days(day(t)).0 - 1).div_euclid(100) + 1)
                    }
                    Millennium => {
                        map_part!(|t: i64| (civil_from_days(day(t)).0 - 1).div_euclid(1000) + 1)
                    }
                    DayOfYear => map_part!(|t: i64| {
                        let d = day(t);
                        d - days_from_civil(civil_from_days(d).0, 1, 1) + 1
                    }),
                    Week => map_part!(|t: i64| iso_week(day(t))),
                };
                ExprResult::Array(Arc::new(out) as ArrayRef)
            }) as ExprEvalFn
        }))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use crate::types::Type;
    use arrow_array::{ArrayRef, Int32Array, Int64Array};
    use rstest::rstest;
    use std::sync::Arc;

    #[rstest]
    fn extracts_the_calendar_parts_of_a_date(mut testing_planner: TestingPlanner) {
        testing_planner.add_table(
            "orders",
            &[(
                "o_orderdate",
                Type::Date,
                // 9204 days after the epoch is 1995-03-15.
                Arc::new(Int32Array::from(vec![9204i32])) as ArrayRef,
            )],
        );

        let rows = run(
            &mut testing_planner,
            "SELECT extract(year FROM o_orderdate) AS y, \
                    extract(month FROM o_orderdate) AS m, \
                    extract(day FROM o_orderdate) AS d, \
                    extract(hour FROM o_orderdate) AS h \
             FROM orders",
        );

        assert_eq!(rows[0]["y"], 1995);
        assert_eq!(rows[0]["m"], 3);
        assert_eq!(rows[0]["d"], 15);
        assert_eq!(rows[0]["h"], 0);
    }

    #[rstest]
    fn extracts_year(mut testing_planner: TestingPlanner) {
        testing_planner.add_table(
            "ts",
            &[(
                "EventTime",
                Type::Timestamp,
                // 1704067200s = 2024-01-01 00:00:00 UTC, in microseconds.
                Arc::new(Int64Array::from(vec![1_704_067_200_000_000i64])) as ArrayRef,
            )],
        );

        let rows = run(
            &mut testing_planner,
            "SELECT extract(year FROM EventTime) FROM ts",
        );

        assert_eq!(*only_column(&rows[0]), 2024);
    }

    #[rstest]
    fn group_by_epoch_groups_on_the_int(mut testing_planner: TestingPlanner) {
        testing_planner.add_table(
            "ts",
            &[(
                "EventTime",
                Type::Timestamp,
                Arc::new(Int64Array::from(vec![0i64, 0, 3_600_000_000])) as ArrayRef,
            )],
        );

        // `extract(epoch …)` is the stored epoch-seconds int (DuckDB types it
        // DOUBLE, but pivot computes Int64), so it must be groupable as an
        // integer key: the two rows at second 0 collapse into one group.
        let rows = run(
            &mut testing_planner,
            "SELECT extract(epoch FROM EventTime) AS e, count(*) AS n \
             FROM ts GROUP BY extract(epoch FROM EventTime)",
        );

        let zero = rows.iter().find(|r| r["e"] == 0).unwrap();
        assert_eq!(zero["n"], 2);
    }
}
