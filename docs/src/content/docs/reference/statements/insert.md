---
title: "INSERT"
description: Append literal rows or query results to a table.
sidebar:
  order: 2
---

`INSERT` appends rows to an existing table. Each statement commits individually.

## Example

```sql
CREATE TABLE events (id BIGINT, region VARCHAR);
INSERT INTO events (id, region) VALUES (1, 'eu'), (2, 'us');
```

## Syntax

```sql
INSERT INTO table_name [(column [, ...])]
VALUES (expression [, ...]) [, ...];

INSERT INTO table_name [(column [, ...])]
SELECT ...;

INSERT INTO table_name BY NAME
SELECT ...;
```

## Target columns

An explicit column list maps input values to those columns in order. Columns
omitted from that list are filled with `NULL`.

```sql
INSERT INTO events (id) VALUES (3);
```

This adds a row whose `region` is `NULL`.

## Insert from a query

Use a query to produce the input rows. Its output must match the target
columns in number and compatible types.

```sql
INSERT INTO events (id, region)
SELECT range, 'eu' FROM range(4, 7);
```

`BY NAME` matches query output names to target columns rather than using
position. Omitted target columns are filled with `NULL`.

```sql
INSERT INTO events BY NAME
SELECT 'us' AS region, 7 AS id;
```

## Limitations

`DEFAULT VALUES` is not supported. `BEGIN` and `ROLLBACK` do not group or undo
inserts; see [transaction behavior](/docs/reference/statements/transactions/).

## Related

- [CREATE TABLE](/docs/reference/statements/create-table/) - define the target table.
- [COPY](/docs/reference/statements/copy/) - stream text or Arrow IPC rows from a client.
- [SELECT](/docs/reference/statements/select/) - construct an input query.
