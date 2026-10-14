---
title: "SQL statements"
description: Find query, write, schema, maintenance, and session commands.
---

Each command page includes examples, supported syntax, and usage notes.

## Queries

- [SELECT](/docs/reference/statements/select/) - filter, join, aggregate, and sort rows.
- [VALUES](/docs/reference/statements/values/) - construct rows from expressions.
- [EXPLAIN](/docs/reference/statements/explain/) - inspect a query's physical plan.

## Writing data

- [INSERT](/docs/reference/statements/insert/) - append values or query results.
- [COPY](/docs/reference/statements/copy/) - stream text or Arrow IPC rows from a client.

## Schemas and tables

- [CREATE TABLE](/docs/reference/statements/create-table/) - define columns, configure layout, or adopt Parquet files.
- [CREATE SCHEMA](/docs/reference/statements/create-schema/) - create a namespace for tables.
- [DROP TABLE](/docs/reference/statements/drop-table/) - remove a table from the catalog.

## Maintenance

- [COMPACT](/docs/reference/statements/compact/) - merge and re-sort Parquet files.

## Users and sessions

- [CREATE USER](/docs/reference/statements/create-user/) - add a login user.
- [SET / RESET](/docs/reference/statements/set-reset/) - control query statistics for a connection.
- [Transaction behavior](/docs/reference/statements/transactions/) - understand the behavior of `BEGIN`, `COMMIT`, and `ROLLBACK`.

## Compatibility

PostgreSQL wire compatibility lets PostgreSQL clients connect. It does not
imply support for every PostgreSQL statement or statement variant.

`DELETE` is not supported yet. Planned additions are tracked in the
[roadmap](/docs/database/roadmap/).
