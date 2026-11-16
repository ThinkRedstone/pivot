---
title: "Functions & operators"
description: Browse functions and operators by the values they work with.
---

Browse by category to find signatures, return values, examples, and argument
restrictions. Function names are case-insensitive.

## Aggregate functions

[Aggregate functions](/docs/reference/functions/aggregates/) combine rows into
counts, sums, averages, and other summaries: `avg`, `count`, `first`, `min`,
`max`, and `sum`.

## Scalar functions

- [Strings and regular expressions](/docs/reference/functions/strings/) - search, measure, slice, and replace text.
- [Dates and times](/docs/reference/functions/datetime/) - construct timestamps, truncate them, and extract fields.
- [Arithmetic](/docs/reference/functions/arithmetic/) - numeric operators, rounding, and supported interval arithmetic.
- [VARIANT access](/docs/reference/functions/variant/) - read nested values and cast fields to SQL types.
- [Utility functions](/docs/reference/functions/utilities/) - format byte counts and clear Pivot's data caches.

## Table functions

[Table functions](/docs/reference/functions/table-functions/) produce rows for
use in `FROM`: `read_parquet` reads Parquet files, and `range` and
`generate_series` generate integers.

## Compatibility

The documented forms and argument restrictions define the implemented
surface, even when the SQL binder recognizes additional overloads.
