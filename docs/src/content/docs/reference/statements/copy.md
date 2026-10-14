---
title: "COPY"
description: Stream rows from a client into a table, as PostgreSQL text or Arrow IPC.
sidebar:
  order: 3
---

`COPY ... FROM STDIN` ingests rows sent over the PostgreSQL wire protocol's
copy interface, in PostgreSQL's text format (the default, what `psql` sends)
or as an Arrow IPC stream.

## Example

In `psql`, load a tab-separated file from the client machine into an
existing `events` table with `id` and `region` columns:

```text
\copy events (id, region) FROM 'events.tsv'
```

Or type the rows in, ending them with a line holding only `\.`:

```text
COPY events (id, region) FROM STDIN;
1	us-east
2	\N
\.
```

An Arrow client starts its copy session with:

```sql
COPY events (id, region) FROM STDIN WITH (FORMAT arrow);
```

and then sends Arrow IPC bytes through the copy interface. In every form the
SQL statement opens the copy session; the row data is sent separately.

## Syntax

```sql
COPY table_name [(column [, ...])]
FROM STDIN [WITH (option [, ...])]
```

| Option | Default | Description |
| --- | --- | --- |
| `FORMAT { text \| arrow }` | `text` | The data format. |
| `DELIMITER 'c'` | tab | Text only: the single one-byte character between fields. It cannot be a backslash, a dot, a lowercase letter, or a digit. |
| `NULL 'string'` | `\N` | Text only: the spelling of a NULL field. |

The older `WITH DELIMITER 'c'` and `WITH NULL 'string'` spellings are
accepted too.

## Columns

An explicit column list chooses the target columns, in the order the rows
carry them. Columns omitted from the list are filled with `NULL`.

## Text format

Each line is one row, with fields separated by the delimiter. A backslash
escapes the next character, so a field may contain a tab, a newline, a
backslash, or the delimiter as `\t`, `\n`, `\\`, or `\<delimiter>`; the
C-style `\b`, `\f`, `\r`, `\v`, octal `\ddd`, and hex `\xhh` escapes are
accepted as well. A field spelled exactly as the NULL string (before its
escapes resolve) is NULL. Lines may end in `\n` or `\r\n`, and a line holding
only `\.` ends the data; `\.` anywhere else in a line is an error. This is the format `COPY ... TO` writes in
PostgreSQL, so a PostgreSQL text dump loads as is.

Values are parsed as the target column's type, ignoring whitespace around
any value that is not text: integers, floats, decimals,
dates (`2024-02-29`), timestamps (`2024-02-29 13:45:00`, with an optional
fraction and zone offset), and JSON documents for `VARIANT` columns. A value
that does not parse fails the whole copy.

## Arrow format

Batches conform to the table by position: the stream carries one column per
targeted column, and each is cast to the column's type. A `VARIANT` column
accepts a variant-encoded column or a string column of JSON documents.

## Failure

A copy is one transaction. A bad value, a row of the wrong width, a stream
cut mid-message, or a client abort rolls back everything the copy loaded.

## Loading a local file in the shell

In [`pivot open`](/docs/reference/cli/#loading-a-file), `\copy` runs the same
`COPY` with rows read from a local file:

```text
\copy events (id, region) FROM 'events.tsv'
\copy events (id, region) FROM 'events.arrow' WITH (FORMAT arrow)
```

## Limitations

CSV input and the `HEADER` option are not supported. This form receives data
from a client; it does not read a named server-side file.

## Related

- [CREATE TABLE](/docs/reference/statements/create-table/) - create the target table.
- [INSERT](/docs/reference/statements/insert/) - append SQL values or query results.
- [Adopt existing Parquet files](/docs/reference/statements/create-table/#adopt-existing-parquet-files) - register files already in storage.
