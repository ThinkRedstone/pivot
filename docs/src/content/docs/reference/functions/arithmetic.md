---
title: "Arithmetic"
description: Add, subtract, multiply, divide, round, and adjust dates and timestamps.
sidebar:
  order: 4
---

Pivot supports binary arithmetic and rounding on numeric expressions, and
limited interval arithmetic on dates and timestamps.

| Operator | Result |
| --- | --- |
| [Addition](#addition) | Sum of two values. |
| [Subtraction](#subtraction) | Difference between two values. |
| [Multiplication](#multiplication) | Product of two values. |
| [Division](#division) | Floating-point quotient. |

| Function | Result |
| --- | --- |
| [floor](#floor) | Largest whole number at or below a value. |
| [round](#round) | Nearest value with a given number of decimal digits. |

## Addition

```sql
left_value + right_value
```

Adds two numeric values.

```sql
SELECT range + 2 AS result FROM range(1, 2);
-- result: 3
```

## Subtraction

```sql
left_value - right_value
```

Subtracts the right numeric value from the left.

```sql
SELECT range - 2 AS result FROM range(5, 6);
-- result: 3
```

## Multiplication

```sql
left_value * right_value
```

Multiplies two numeric values.

```sql
SELECT range * 3 AS result FROM range(2, 3);
-- result: 6
```

## Division

```sql
left_value / right_value
```

Returns a floating-point quotient. Dividing two `REAL` values returns `REAL`;
other numeric pairs use `DOUBLE`.

```sql
SELECT range / 2 AS result FROM range(5, 6);
-- result: 2.5
```

## floor

```sql
floor(value)
```

Returns the largest whole number at or below a `REAL`, `DOUBLE`, or `DECIMAL`
value. A `REAL` or `DOUBLE` keeps its type, and a `DECIMAL(p, s)` returns
`DECIMAL(p, 0)`. An integer argument is converted to `DOUBLE`.

```sql
SELECT floor(x) AS result FROM (VALUES (-2.5::DOUBLE)) AS input(x);
-- result: -3.0
```

## round

```sql
round(value)
round(value, digits)
```

Rounds a numeric value to the nearest value with `digits` digits after the
decimal point. A tie rounds away from zero. Without `digits`, the value is
rounded to a whole number.

```sql
SELECT round(x) AS result FROM (VALUES (2.5::DOUBLE)) AS input(x);
-- result: 3.0

SELECT round(x, 2) AS result FROM (VALUES (3.14159::DOUBLE)) AS input(x);
-- result: 3.14
```

A negative `digits` rounds to the left of the decimal point.

```sql
SELECT round(range, -2) AS result FROM range(1250, 1251);
-- result: 1300
```

A `REAL`, `DOUBLE`, or signed integer keeps its type. A `DECIMAL(p, s)` keeps
as many of its `s` digits as `digits` asks for, and none for a negative
`digits`: `round(x, 1)` over a `DECIMAL(15, 2)` returns `DECIMAL(15, 1)`.

`digits` must be an integer constant.

A negative `digits` has these edge cases:

- Rounding away every digit a type can hold left of the decimal point returns
  0: `round(x, -3)` over a `TINYINT` or a `DECIMAL(5, 2)` is always 0.
- Rounding an integer or a `DECIMAL(p, 0)` column fails the query when a
  rounded value does not fit in the column's type, such as 127 in a `TINYINT`
  rounded to tens.
- An infinite or NaN `REAL` or `DOUBLE` rounds to 0.

## Date and time arithmetic

A `DATE` or `TIMESTAMP` can be adjusted by a constant `INTERVAL`.

```sql
SELECT ts + INTERVAL '1 hour' AS later
FROM (VALUES (TIMESTAMP '2026-09-09 12:00:00')) AS input(ts);
-- later: 2026-09-09 13:00:00
```

Month and year offsets are not supported. A `DATE` accepts only whole-day
offsets.

## Related

- [Numeric types](/docs/reference/data-types/#numeric-types)
- [Dates and times](/docs/reference/functions/datetime/)
