---
title: "Dates & times"
description: Construct timestamps, extract fields, and truncate or bucket time values.
sidebar:
  order: 3
---

Pivot's `TIMESTAMP` is timezone-free and uses microsecond precision. Normalize
timezone-sensitive input before loading it.

| Function | Result |
| --- | --- |
| [now](#now) | Timestamp fixed for the statement. |
| [date_trunc](#date_trunc) | Timestamp truncated to a supported unit. |
| [time_bucket](#time_bucket) | Date or timestamp floored to the start of its bucket. |
| [extract](#extract) | A date or timestamp field. |
| [make_date](#make_date) | Date from days since the epoch. |
| [make_timestamp](#make_timestamp) | Timestamp from microseconds since the epoch. |

## now

```sql
now()
```

Takes no arguments. Returns a timezone-free `TIMESTAMP` fixed for the duration
of the statement.

```sql
SELECT now() AS statement_time;
```

`now()` returns the time the statement started, not the time the function is
evaluated.

## date_trunc

```sql
date_trunc(part, timestamp)
```

Returns a timestamp truncated to the specified unit. `part` must be a constant:
`second`, `minute`, `hour`, or `day`.

```sql
SELECT date_trunc('hour', ts) AS hour
FROM (VALUES (TIMESTAMP '2026-09-09 12:34:56')) AS input(ts);
-- hour: 2026-09-09 12:00:00
```

## time_bucket

```sql
time_bucket(bucket_width, value)
time_bucket(bucket_width, value, offset)
time_bucket(bucket_width, value, origin)
```

Floors a date or timestamp to the start of the bucket that holds it.
`bucket_width` must be a constant, positive
`INTERVAL`, made either of days and time (`INTERVAL '15 minutes'`,
`INTERVAL '7 days'`) or of whole months (`INTERVAL '3 months'`,
`INTERVAL '1 year'`), not both.

```sql
SELECT time_bucket(INTERVAL '15 minutes', ts) AS bucket
FROM (VALUES (TIMESTAMP '2026-09-09 12:34:56')) AS input(ts);
-- bucket: 2026-09-09 12:30:00
```

The result has the type of `value`, unless an `origin` of a wider type is
given: a `DATE` bucketed from a `TIMESTAMP` origin yields a `TIMESTAMP`, and
either bucketed from a `TIMESTAMPTZ` origin yields a `TIMESTAMPTZ`. On a
`TIMESTAMPTZ`, a width in days cannot also have a time component.

Day and time buckets are aligned to Monday 2000-01-03 00:00:00, so 7-day
buckets start on Mondays. Month buckets are aligned to 2000-01-01.

The optional third argument, also a constant, moves that alignment. An
`origin` date or timestamp is an instant a bucket starts at, and cannot be
infinite. An `offset`
interval shifts every bucket by that much; it must be made of days and time
for a day or time width, and of whole months for a month width.

```sql
SELECT time_bucket(INTERVAL '1 hour', ts, INTERVAL '10 minutes') AS bucket
FROM (VALUES (TIMESTAMP '2026-09-09 12:34:56')) AS input(ts);
-- bucket: 2026-09-09 12:10:00
```

## extract

```sql
extract(part FROM value)
```

Extracts a field from a date or timestamp and returns its numeric value.

```sql
SELECT extract(year FROM ts) AS year
FROM (VALUES (TIMESTAMP '2026-09-09 12:34:56')) AS input(ts);
-- year: 2026
```

Supported parts are `epoch`, `second`, `millisecond`, `microsecond`, `minute`,
`hour`, `day`, `month`, `quarter`, `year`, `decade`, `century`, `millennium`,
`dayofweek` (`dow`), `isodow`, `dayofyear` (`doy`), and `week` (`weekofyear`).

## make_date

```sql
make_date(days)
```

Interprets an integer as signed days since 1970-01-01 and returns a `DATE`.

```sql
SELECT make_date(days) AS date
FROM (VALUES (1)) AS input(days);
-- date: 1970-01-02
```

## make_timestamp

```sql
make_timestamp(microseconds)
```

Interprets an integer as microseconds since 1970-01-01 and returns a
`TIMESTAMP`.

```sql
SELECT make_timestamp(microseconds) AS ts
FROM (VALUES (1000000::BIGINT)) AS input(microseconds);
-- ts: 1970-01-01 00:00:01
```

## Related

- [Date and time arithmetic](/docs/reference/functions/arithmetic/#date-and-time-arithmetic)
- [Date and time types](/docs/reference/data-types/#dates-times-and-intervals)
