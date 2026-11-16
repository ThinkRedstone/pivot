---
title: "SELECT"
description: Read, filter, join, and aggregate rows.
sidebar:
  order: 1
---

`SELECT` reads rows from tables or table functions and transforms them into a result.

## Example

List the ten largest tables by row count:

```sql
SELECT name, total_rows
FROM system.tables
WHERE datastore <> 'system'
ORDER BY total_rows DESC
LIMIT 10;
```

## Syntax

```sql
[WITH name AS (query) [, ...]]
SELECT [DISTINCT] expression [AS alias] [, ...]
[FROM source]
[WHERE condition]
[GROUP BY expression [, ...]]
[HAVING condition]
[ORDER BY expression [ASC | DESC] [, ...]]
[LIMIT count]
[OFFSET count];
```

`source` can be a table, a supported join, a subquery, or a
[table function](/docs/reference/functions/table-functions/).

## Columns and expressions

Select named columns, use `*` for all columns, or compute expressions. `AS`
names an output column. `DISTINCT` removes duplicate result rows.

```sql
SELECT DISTINCT datastore FROM system.tables;
```

Tables can be qualified as `datastore.schema.table`. An unqualified table
name uses the default datastore and schema. System tables use two-part names,
such as `system.tables`.

## Unnesting a list

`unnest` in the select list repeats each input row once per element of a
list, with the element as an output column. The row order of the result is
not defined.

```sql
SELECT unnest([width, height]) AS dimension, name FROM products;
```

The list must be written out in the query, as `[element, ...]`. Each element
can be any expression over the row. A list read from a column is not
supported, and neither is `unnest` in the `FROM` clause.

An element can be a struct, whose fields are then read by name. A query
cannot return the struct itself, only its fields. This turns several columns
of a row into one row per column:

```sql
SELECT pair.label, pair.amount
FROM (
  SELECT unnest([{'label': 'width', 'amount': width},
                 {'label': 'height', 'amount': height}]) AS pair
  FROM products
);
```

With several `unnest` calls in one select list, the lists are stepped through
together, and a list shorter than the longest one gives `NULL` past its end.
An empty or `NULL` list produces no rows.

## Filtering

`WHERE` filters input rows before aggregation.

```sql
SELECT name, total_rows
FROM system.tables
WHERE total_rows > 1000;
```

See [functions and operators](/docs/reference/functions/) for string matching,
date expressions, and other filters.

## Joins

Use `JOIN` in the `FROM` clause to combine rows from two tables. The `ON`
clause specifies which rows match.

### Inner joins

An `INNER JOIN` returns a row for each matching pair. `JOIN` without a type
means `INNER JOIN`.

List customers and their orders:

```sql
SELECT c.name, o.id AS order_id
FROM customers AS c
JOIN orders AS o ON c.id = o.customer_id;
```

### Left joins

A `LEFT JOIN` also includes rows from the left table that have no match.
For those rows, columns from the right table are `NULL`.

List all customers, including those without orders:

```sql
SELECT c.name, o.id AS order_id
FROM customers AS c
LEFT JOIN orders AS o ON c.id = o.customer_id;
```

### Semi joins

A `SEMI JOIN` keeps rows from the left table that have at least one match.
It returns only columns from the left table, without repeating a left row
for multiple matches.

List customers who have placed an order:

```sql
SELECT c.id, c.name
FROM customers AS c
SEMI JOIN orders AS o ON c.id = o.customer_id;
```

### Anti joins

An `ANTI JOIN` keeps rows from the left table that have no match. It returns
only columns from the left table.

List customers who have never placed an order:

```sql
SELECT c.id, c.name
FROM customers AS c
ANTI JOIN orders AS o ON c.id = o.customer_id;
```

### Join conditions

Inner, left, semi, and anti joins support equality conditions such as
`c.id = o.customer_id`. Use `AND` to match on multiple keys or add filters
to matching pairs, for example `ON c.id = o.customer_id AND o.total > 100`.

An inner join can also match rows using an inequality instead of equality.
This is called a *range join*. For example, given a `discount_tiers` table
with `name` and `min_total` columns, find every discount tier each order
qualifies for:

```sql
SELECT o.id AS order_id, d.name AS discount_tier
FROM orders AS o
JOIN discount_tiers AS d ON o.total >= d.min_total;
```

Range joins require exactly one `<`, `<=`, `>`, or `>=` comparison between
numeric, date, or timestamp expressions. Additional join conditions and
left, semi, or anti range joins are not supported.

## Grouping and aggregates

`GROUP BY` combines rows with matching keys. `HAVING` filters the resulting
groups, after aggregation.

```sql
SELECT table_id, count(*) AS column_count
FROM system.columns
GROUP BY table_id
HAVING count(*) > 5
ORDER BY column_count DESC;
```

See [aggregate functions](/docs/reference/functions/aggregates/) for the
supported aggregates and their result types.

## Ordering and limits

`ORDER BY` sorts the result; `ASC` is ascending and `DESC` is descending.
`LIMIT` caps the number of rows, and `OFFSET` skips rows before returning them.
Use an explicit order when the choice of returned rows matters.

```sql
SELECT name FROM system.tables ORDER BY name LIMIT 10 OFFSET 10;
```

## Common table expressions

`WITH` names a query result for use in the same statement.

```sql
WITH user_tables AS (
  SELECT name, total_rows
  FROM system.tables
  WHERE datastore <> 'system'
)
SELECT * FROM user_tables WHERE total_rows > 0;
```

## Compatibility

PostgreSQL wire compatibility does not imply support for every PostgreSQL
query form. The clauses and join forms above describe the supported surface.

## Related

- [EXPLAIN](/docs/reference/statements/explain/) - inspect a query plan.
- [VALUES](/docs/reference/statements/values/) - construct rows directly.
- [System tables](/docs/reference/system-tables/) - query server metadata.
