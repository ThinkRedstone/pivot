//! [`TimeBucket`] — SQL `time_bucket(bucket_width, source[, offset | origin])`.

use super::Expression;
use super::date_part::{civil_from_days, days_from_civil};
use crate::compile::{self, ExprEvalFn, ExprFn, ExprResult};
use crate::types::{MICROS_PER_DAY, Type, physical_arrow_type};
use arrow_array::cast::AsArray;
use arrow_array::{ArrayRef, Date32Array, Int64Array, RecordBatch, TimestampMicrosecondArray};
use arrow_schema::DataType;
use std::fmt::{self, Display};
use std::sync::Arc;

/// The message a query fails with when a bucket starts before the earliest
/// value its type can hold.
const BUCKET_START_OUT_OF_RANGE: &str =
    "time_bucket: the start of the bucket is out of range for the value's type";

/// Count the whole months from January 1970 to the month that holds a day,
/// itself counted in days since the epoch.
pub(crate) fn count_epoch_months(days: i64) -> i64 {
    let (year, month, _) = civil_from_days(days);
    (year - 1970) * 12 + month - 1
}

/// Floor `value` to the start of its bucket among the buckets of a fixed
/// `width` that start at `origin + k * width`, with `origin` in `[0, width)`.
/// The value, width and origin share one unit. Returns `None` when the bucket
/// starts below what an `i64` holds.
#[inline(always)]
fn floor_to_fixed_width(value: i64, width: i64, origin: i64) -> Option<i64> {
    // Both terms sit in `[0, width)`, so their difference is the distance
    // past the bucket start, or one width short of it.
    let past_start = value.rem_euclid(width) - origin;
    value.checked_sub(if past_start < 0 {
        past_start + width
    } else {
        past_start
    })
}

/// Floor a day, counted in days since the epoch, to the first day of its
/// bucket among the buckets of `width` months that start at the month
/// `origin + k * width`, with `origin` in `[0, width)`.
#[inline(always)]
fn floor_to_month_bucket(days: i64, width: i64, origin: i64) -> i64 {
    let months = count_epoch_months(days);
    let bucket_start = months - (months - origin).rem_euclid(width);
    days_from_civil(
        1970 + bucket_start.div_euclid(12),
        bucket_start.rem_euclid(12) + 1,
        1,
    )
}

/// The buckets a [`TimeBucket`] floors onto: every instant `origin + k * width`.
/// The origin is kept reduced into `[0, width)`, which names the same buckets
/// as any other origin a whole number of widths away.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BucketGrid {
    /// Buckets of a fixed length, both fields counted in microseconds since
    /// the epoch.
    Micros { width: i64, origin: i64 },
    /// Buckets of whole calendar months, both fields counted in months since
    /// January 1970. Every bucket starts at midnight on the first of a month.
    Months { width: i64, origin: i64 },
}

impl BucketGrid {
    /// Build the grid of fixed-length buckets that has a bucket starting at
    /// `origin`. `width` must be positive.
    pub fn from_micros(width: i64, origin: i64) -> BucketGrid {
        BucketGrid::Micros {
            width,
            origin: origin.rem_euclid(width),
        }
    }

    /// Build the grid of calendar-month buckets that has a bucket starting at
    /// the month `origin`. `width` must be positive.
    pub fn from_months(width: i64, origin: i64) -> BucketGrid {
        BucketGrid::Months {
            width,
            origin: origin.rem_euclid(width),
        }
    }

    /// Floor each timestamp, in microseconds since the epoch, to its bucket
    /// start. The two infinite timestamps have no bucket and pass through
    /// unchanged.
    ///
    /// The grid is matched once for the whole array, so each row loop runs
    /// only its own kind of bucket.
    fn floor_timestamps(
        self,
        micros: &Int64Array,
    ) -> Result<TimestampMicrosecondArray, &'static str> {
        let is_infinite = |instant: i64| instant == i64::MAX || instant == -i64::MAX;
        match self {
            BucketGrid::Micros { width, origin } => micros.try_unary(|instant| {
                if is_infinite(instant) {
                    return Ok(instant);
                }
                floor_to_fixed_width(instant, width, origin).ok_or(BUCKET_START_OUT_OF_RANGE)
            }),
            BucketGrid::Months { width, origin } => micros.try_unary(|instant| {
                if is_infinite(instant) {
                    return Ok(instant);
                }
                floor_to_month_bucket(instant.div_euclid(MICROS_PER_DAY), width, origin)
                    .checked_mul(MICROS_PER_DAY)
                    .ok_or(BUCKET_START_OUT_OF_RANGE)
            }),
        }
    }

    /// Floor each date, in days since the epoch, to the day its bucket starts
    /// on, reading the date as its midnight. The two infinite dates have no
    /// bucket and pass through unchanged.
    fn floor_dates(self, days: &Date32Array) -> Result<Date32Array, &'static str> {
        let is_infinite = |day: i32| day == i32::MAX || day == -i32::MAX;
        let narrow_to_date = |day: i64| i32::try_from(day).ok();
        match self {
            // Buckets that start and end on midnights are floored in days, so
            // a date never has to fit in a timestamp's microseconds.
            BucketGrid::Micros { width, origin }
                if width % MICROS_PER_DAY == 0 && origin % MICROS_PER_DAY == 0 =>
            {
                let width_days = width / MICROS_PER_DAY;
                let origin_days = origin / MICROS_PER_DAY;
                days.try_unary(|day| {
                    if is_infinite(day) {
                        return Ok(day);
                    }
                    floor_to_fixed_width(i64::from(day), width_days, origin_days)
                        .and_then(narrow_to_date)
                        .ok_or(BUCKET_START_OUT_OF_RANGE)
                })
            }
            BucketGrid::Micros { width, origin } => days.try_unary(|day| {
                if is_infinite(day) {
                    return Ok(day);
                }
                i64::from(day)
                    .checked_mul(MICROS_PER_DAY)
                    .and_then(|midnight| floor_to_fixed_width(midnight, width, origin))
                    .and_then(|bucket_start| {
                        narrow_to_date(bucket_start.div_euclid(MICROS_PER_DAY))
                    })
                    .ok_or(BUCKET_START_OUT_OF_RANGE)
            }),
            BucketGrid::Months { width, origin } => days.try_unary(|day| {
                if is_infinite(day) {
                    return Ok(day);
                }
                narrow_to_date(floor_to_month_bucket(i64::from(day), width, origin))
                    .ok_or(BUCKET_START_OUT_OF_RANGE)
            }),
        }
    }
}

impl Display for BucketGrid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BucketGrid::Micros { width, origin } => {
                write!(f, "{width} microseconds from {origin}")
            }
            BucketGrid::Months { width, origin } => write!(f, "{width} months from {origin}"),
        }
    }
}

/// SQL `time_bucket(bucket_width, source[, offset | origin])` — floor a date or
/// timestamp to the start of its bucket. The width and the optional third
/// argument are constants, folded into the [`BucketGrid`] when the call is
/// built.
#[derive(Debug, Clone)]
pub struct TimeBucket {
    pub grid: BucketGrid,
    pub source: Box<Expression>,
    /// The bound result type ([`Type::Date`], [`Type::Timestamp`] or
    /// [`Type::TimestampTz`]), which DuckDB has already cast the source to.
    pub result: Type,
}

impl Display for TimeBucket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "time_bucket({}, {})", self.grid, self.source)
    }
}

impl TimeBucket {
    pub fn compile(&self) -> Result<ExprFn, compile::Error> {
        let grid = self.grid;
        let result_arrow = physical_arrow_type(&self.result);
        let source_builder = self.source.compile()?;
        Ok(Box::new(move || {
            let mut source_expr = source_builder();
            let result_arrow = result_arrow.clone();
            Box::new(move |batch: &RecordBatch| {
                let source = source_expr(batch);
                let (array, _) = source.as_datum().get();
                // A bucket start the type cannot hold is data-dependent and a
                // compiled expression has no error channel, so it panics and
                // the dataflow fails the query with the message.
                let bucketed: ArrayRef = match &result_arrow {
                    DataType::Date32 => Arc::new(
                        grid.floor_dates(array.as_primitive())
                            .unwrap_or_else(|message| panic!("{message}")),
                    ),
                    DataType::Timestamp(_, zone) => {
                        // Bucketing the UTC instant is zone-independent under
                        // the UTC session, so a zoned source keeps its marker.
                        let micros = arrow::compute::cast(array, &DataType::Int64).unwrap();
                        Arc::new(
                            grid.floor_timestamps(micros.as_primitive())
                                .unwrap_or_else(|message| panic!("{message}"))
                                .with_timezone_opt(zone.clone()),
                        )
                    }
                    other => unreachable!("time_bucket result is a date or timestamp, not {other}"),
                };
                ExprResult::Array(bucketed)
            }) as ExprEvalFn
        }))
    }
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use crate::types::Type;
    use arrow_array::cast::AsArray;
    use arrow_array::types::Date32Type;
    use arrow_array::{ArrayRef, Int32Array, Int64Array};
    use rstest::rstest;
    use std::sync::Arc;

    /// 2024-03-05 10:22:33.5, a Tuesday, in microseconds since the epoch.
    const TUESDAY_MORNING: i64 = 1_709_634_153_500_000;

    /// A day past the last one a timestamp can hold, in days since the epoch:
    /// the 15th of its month, and a Thursday.
    const FAR_FUTURE_DAY: i32 = 109_000_003;

    /// Register `events` with one row whose `ts` column of `column_type`
    /// holds `micros`.
    fn add_event(testing_planner: &TestingPlanner, column_type: Type, micros: i64) {
        let column = Arc::new(Int64Array::from(vec![micros])) as ArrayRef;
        testing_planner.add_table("events", &[("ts", column_type, column)]);
    }

    #[rstest]
    #[case::minutes("INTERVAL '15 minutes', ts", "2024-03-05T10:15:00")]
    #[case::weeks_start_on_monday("INTERVAL '7 days', ts", "2024-03-04T00:00:00")]
    #[case::months("INTERVAL '3 months', ts", "2024-01-01T00:00:00")]
    #[case::years("INTERVAL '5 years', ts", "2020-01-01T00:00:00")]
    #[case::offset("INTERVAL '1 hour', ts, INTERVAL '10 minutes'", "2024-03-05T10:10:00")]
    #[case::month_offset("INTERVAL '3 months', ts, INTERVAL '1 month'", "2024-02-01T00:00:00")]
    #[case::origin(
        "INTERVAL '1 hour', ts, TIMESTAMP '2024-01-01 00:20:00'",
        "2024-03-05T10:20:00"
    )]
    fn floors_a_timestamp_to_its_bucket(
        mut testing_planner: TestingPlanner,
        #[case] arguments: &str,
        #[case] expected: &str,
    ) {
        add_event(&testing_planner, Type::Timestamp, TUESDAY_MORNING);

        let rows = run(
            &mut testing_planner,
            &format!("SELECT time_bucket({arguments}) FROM events"),
        );

        assert_eq!(only_column(&rows[0]), expected);
    }

    #[rstest]
    fn floors_a_timestamp_before_the_epoch(mut testing_planner: TestingPlanner) {
        let half_an_hour_before_the_epoch = -1_800_000_000;
        add_event(
            &testing_planner,
            Type::Timestamp,
            half_an_hour_before_the_epoch,
        );

        let rows = run(
            &mut testing_planner,
            "SELECT time_bucket(INTERVAL '1 day', ts) FROM events",
        );

        assert_eq!(only_column(&rows[0]), "1969-12-31T00:00:00");
    }

    #[rstest]
    fn bucketing_a_timestamptz_keeps_its_zone(mut testing_planner: TestingPlanner) {
        add_event(&testing_planner, Type::TimestampTz, TUESDAY_MORNING);

        let rows = run(
            &mut testing_planner,
            "SELECT time_bucket(INTERVAL '1 hour', ts) FROM events",
        );

        assert_eq!(only_column(&rows[0]), "2024-03-05T10:00:00Z");
    }

    #[rstest]
    fn bucketing_a_date_yields_a_date(mut testing_planner: TestingPlanner) {
        let tuesday = Arc::new(Int32Array::from(vec![19_787])) as ArrayRef;
        testing_planner.add_table("events", &[("day", Type::Date, tuesday)]);

        let rows = run(
            &mut testing_planner,
            "SELECT time_bucket(INTERVAL '7 days', day) FROM events",
        );

        assert_eq!(only_column(&rows[0]), "2024-03-04");
    }

    #[rstest]
    #[case::month_bucket("INTERVAL '1 month'", FAR_FUTURE_DAY - 14)]
    #[case::week_bucket("INTERVAL '7 days'", FAR_FUTURE_DAY - 3)]
    fn floors_a_date_beyond_the_timestamp_range(
        mut testing_planner: TestingPlanner,
        #[case] bucket_width: &str,
        #[case] expected_day: i32,
    ) {
        let far_future = Arc::new(Int32Array::from(vec![FAR_FUTURE_DAY])) as ArrayRef;
        testing_planner.add_table("events", &[("day", Type::Date, far_future)]);

        let batches = run_batches(
            &mut testing_planner,
            &format!("SELECT time_bucket({bucket_width}, day) FROM events"),
        );

        let bucket = batches[0].column(0).as_primitive::<Date32Type>().value(0);
        assert_eq!(bucket, expected_day);
    }

    #[rstest]
    fn fails_the_query_when_the_bucket_starts_out_of_range(mut testing_planner: TestingPlanner) {
        add_event(&testing_planner, Type::Timestamp, i64::MIN + 2);

        let error = run_expecting_error(
            &mut testing_planner,
            "SELECT time_bucket(INTERVAL '1 hour', ts) FROM events",
        );

        assert!(error.contains("out of range"), "{error}");
    }

    #[rstest]
    #[case::months_with_days(
        Type::Timestamp,
        "INTERVAL '1 month 1 day', ts",
        "cannot have a day or time component"
    )]
    #[case::months_with_cancelling_days_and_time(
        Type::Timestamp,
        "INTERVAL '1 month 1 day -24 hours', ts",
        "cannot have a day or time component"
    )]
    #[case::zoned_days_with_time(
        Type::TimestampTz,
        "INTERVAL '1 day 12 hours', ts",
        "cannot have a time component"
    )]
    #[case::negative_width(Type::Timestamp, "INTERVAL '-1 hour', ts", "greater than 0")]
    #[case::overflowing_width(Type::Timestamp, "INTERVAL '2147483647 days', ts", "too long")]
    #[case::overflowing_offset(
        Type::Timestamp,
        "INTERVAL '1 hour', ts, INTERVAL '2147483647 days'",
        "too long"
    )]
    #[case::infinite_origin(
        Type::Timestamp,
        "INTERVAL '1 hour', ts, TIMESTAMP 'infinity'",
        "cannot be infinite"
    )]
    fn rejects_an_invalid_call(
        mut testing_planner: TestingPlanner,
        #[case] column_type: Type,
        #[case] arguments: &str,
        #[case] expected_reason: &str,
    ) {
        add_event(&testing_planner, column_type, TUESDAY_MORNING);

        let error = testing_planner
            .plan(&format!("SELECT time_bucket({arguments}) FROM events"))
            .err()
            .unwrap()
            .to_string();

        assert!(error.contains(expected_reason), "{error}");
    }
}
