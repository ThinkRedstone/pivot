---
title: "Command-line interface"
description: The pivot executable's commands, flags, environment variables, and SQL shell.
---

Pivot ships as a single executable, `pivot`. It either opens one datastore
in an interactive SQL shell, or runs the PostgreSQL-compatible server.

## Usage

```sh
pivot open [--kind pivotlake] [--memory <SIZE>] [--workers <COUNT>] <DIRECTORY | URI>
pivot open --kind iceberg [--warehouse <WAREHOUSE>] [--memory <SIZE>] [--workers <COUNT>] <CATALOG_URI>
pivot server --config <FILE>
pivot --help
pivot --version
```

| Command | Description |
| --- | --- |
| [`pivot open`](#pivot-open) | Open one datastore in an interactive SQL shell. The datastore runs inside the shell's process, with no server. |
| [`pivot server`](#pivot-server) | Run the server in the foreground, serving every configured datastore to PostgreSQL clients. |

Every command accepts `--help`, which prints its flags.

## `pivot open`

Opens a single datastore and starts the [SQL shell](#sql-shell) over it.

### Flags

| Flag | Description |
| --- | --- |
| `<DATASTORE_LOCATION>` | **Required.** Where the datastore is: a local directory or object-store URI for `--kind pivotlake`, or the REST catalog's `http(s)` base URI for `--kind iceberg`. See [Datastore locations](#datastore-locations). |
| `--kind <KIND>` | The datastore implementation, `pivotlake` (also accepted as `pivot`) or `iceberg`.<br />**Default:** `pivotlake` |
| `--warehouse <WAREHOUSE>` | The warehouse to serve, for an Iceberg catalog that serves several. Only valid with `--kind iceberg`. |
| `--memory <SIZE>` | The buffer-pool budget, as a size such as `8g` or a share of memory such as `50%`. See [Memory budget](#memory-budget).<br />**Default:** `80%` |
| `--workers <COUNT>` | The number of worker threads that execute queries. Must be at least 1.<br />**Default:** the number of available cores |

### Datastore locations

| Location | Kind | Example |
| --- | --- | --- |
| Local directory | `pivotlake` | `./pivot-data`, `/var/lib/pivot` |
| Local file URI | `pivotlake` | `file:///var/lib/pivot` |
| Amazon S3 or S3-compatible storage | `pivotlake` | `s3://bucket/prefix` (also `s3a://`) |
| Google Cloud Storage | `pivotlake` | `gs://bucket/prefix` |
| Iceberg REST catalog | `iceberg` | `https://catalog.example.com/api` |

A local directory is created if it does not exist. Object-store credentials
come from the [environment](#object-store-credentials).

An Iceberg catalog is served read-only, and each catalog namespace appears
as a schema. Table files are read from wherever the catalog says they are,
using the credentials the catalog vends, or the
[object-store variables](#object-store-credentials) when it vends none.

:::caution[One process per local datastore]
A local datastore can be open in only one process at a time. The process
that opens it holds a lock file, `.pivot.lock`, in the directory. Opening it
from a second shell, or from a shell while a `pivot server` serves it, fails
with an error that names the owning process ID.
:::

### Refresh and maintenance

The shell reloads its tables every 30 seconds, so data that another process
commits to a shared datastore, or a table added to an Iceberg catalog,
becomes visible without restarting the shell.

The shell never compacts or vacuums a datastore. Table maintenance belongs to
the one process that owns the datastore, normally a `pivot server`. See
[Datastores](/docs/reference/server/datastores/).

### Examples

Open a local datastore, creating it if needed:

```sh
pivot open ./pivot-data
```

Open a datastore in S3:

```sh
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-east-1
pivot open s3://example-bucket/pivot/
```

Open a datastore on an S3-compatible server such as MinIO:

```sh
export AWS_ENDPOINT_URL=http://localhost:9000
pivot open s3://example-bucket/pivot/
```

Open a datastore in Google Cloud Storage, with the credentials from
`gcloud auth application-default login`:

```sh
pivot open gs://example-bucket/pivot/
```

Open the tables of an Iceberg REST catalog, authenticating with a bearer
token:

```sh
export PIVOT_ICEBERG_TOKEN=...
pivot open --kind iceberg https://catalog.example.com/api \
  --warehouse s3://lake/warehouse
```

Limit the shell to 2 GiB of buffer pool and 4 worker threads:

```sh
pivot open --memory 2g --workers 4 ./pivot-data
```

## `pivot server`

Runs the Pivot server in the foreground until it receives `SIGINT`
(Ctrl-C) or `SIGTERM`, then shuts down.

### Flags

| Flag | Description |
| --- | --- |
| `--config <FILE>` | **Required.** The YAML file that configures the server. See [Configuration file](/docs/reference/configuration/). |

Everything else, including the memory and worker budgets, the listen
address, the datastores, and users, is set in the configuration file. The
server reads the file once, at startup.

The server writes its log to stdout, where `journald` and `docker logs`
collect it. The log's format and level are set in the configuration file.

### Examples

Run a server in the foreground:

```sh
pivot server --config pivot.yaml
```

Then connect with any PostgreSQL client:

```sh
psql -h 127.0.0.1 -p 5432 -U pivot
```

## SQL shell

`pivot open` starts an interactive shell. It requires a terminal, and does
not read SQL piped into stdin.

```text
$ pivot open ./pivot-data
pivot shell (0.1.0)
Type "\help" for help.
pivot=> CREATE TABLE events (id BIGINT, name TEXT);
CREATE TABLE
pivot=> SELECT count(*)
pivot-> FROM events;
```

### Entering statements

Statements end with `;` and may span several lines. The prompt is
`pivot=>` at the start of a statement and `pivot->` while one is
incomplete. Several statements on one line run in order.

Query results are printed as a table. When a statement fails, the shell
prints `ERROR:` and the message, and skips any statements that followed it
in the same input.

### Shell commands

Shell commands start with a backslash and take no `;`.

| Command | Description |
| --- | --- |
| `\q` | Quit the shell. |
| `\h`, `\help` | Show the list of shell commands. |
| `\timing` | Toggle printing the elapsed time after each statement. |
| `\timing on \| off` | Turn statement timing on or off. `true`/`false` and `1`/`0` are also accepted. |
| `\copy table [(columns)] FROM 'file' WITH (FORMAT arrow)` | Load a local Arrow IPC file into a table. See [Loading a file](#loading-a-file). |

### Loading a file

`\copy` loads a file from the machine running the shell into a table. It
takes the same table, column list, and options as
[`COPY ... FROM STDIN`](/docs/reference/statements/copy/), with a file name in
place of `STDIN`:

```text
pivot=> \copy events (id, region) FROM 'events.arrow' WITH (FORMAT arrow)
COPY 1000
```

The file must be an Arrow IPC stream. The file name can be bare or
single-quoted; quote it if it contains spaces. The shell prints the number of
rows loaded. The load is all-or-nothing: if the file is incomplete or you
press Ctrl-C, no rows are added.

### Execution stats

To print each statement's execution stats, set `pivot_stats`. See
[`SET` and `RESET`](/docs/reference/statements/set-reset/).

```sql
SET pivot_stats = true;
```

### Keyboard shortcuts

| Key | Description |
| --- | --- |
| Ctrl-C | While typing, discard the current statement. While a statement runs, cancel it. |
| Ctrl-D | Quit the shell. |
| Up, Down | Recall statements entered earlier in the session. |

### Exiting the shell

Any of these quit the shell:

```text
\q
quit;
Ctrl-D
```

## Memory budget

Both commands allocate a fixed buffer pool at startup. `pivot open` takes
its budget from `--memory`, and `pivot server` from the `memory` key of its
configuration file. The value takes one of two forms:

| Form | Example | Meaning |
| --- | --- | --- |
| Size | `512m`, `32g`, `1t` | Exactly this many bytes. The suffixes `k`, `m`, `g`, and `t`, each optionally followed by `b`, are base-1024 and case-insensitive. A number without a suffix is bytes. Fractions such as `1.5g` are rejected. |
| Percentage | `50%` | This share of the machine's physical memory, minus 4 GiB reserved for memory used outside the pool. From `1%` to `100%`. |

The default is `80%`.

The pool is divided into 2 MiB slots, so the budget must be at least 2 MiB.
Every slot is allocated as the process starts, so Pivot refuses to start when
the budget exceeds the memory currently available, rather than being killed
by the operating system partway through startup.

:::note
On a machine with little memory, the 4 GiB reserve can leave nothing for the
pool, which is an error. Pass an absolute size instead, such as
`--memory 1g`.
:::

## Environment variables

### Object-store credentials

`pivot open` reads these variables when opening a datastore in a bucket.

| Variable | Description |
| --- | --- |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | S3 credentials. Set both, or neither for anonymous access. |
| `AWS_SESSION_TOKEN` | A session token for temporary S3 credentials. Only used together with the keys above. |
| `AWS_REGION`, `AWS_DEFAULT_REGION` | The bucket's region. When neither is set, the region is discovered from the bucket. |
| `AWS_ENDPOINT_URL` | An S3-compatible endpoint, such as MinIO. Requests use path-style addressing. |
| `GOOGLE_APPLICATION_CREDENTIALS` | A GCS service-account key file. |

GCS follows Application Default Credentials: it tries
`GOOGLE_APPLICATION_CREDENTIALS`, then the `gcloud` login file, then the
instance metadata server.

### Iceberg catalog credentials

`pivot open --kind iceberg` authenticates to the catalog with these
variables, or without authentication when none is set.

| Variable | Description |
| --- | --- |
| `PIVOT_ICEBERG_TOKEN` | A bearer token. |
| `PIVOT_ICEBERG_CREDENTIAL` | An OAuth2 client credential, written `client_id:client_secret`. |
| `PIVOT_ICEBERG_OAUTH2_SERVER_URI` | The OAuth2 token endpoint. Only valid with `PIVOT_ICEBERG_CREDENTIAL`. |
| `PIVOT_ICEBERG_OAUTH2_SCOPE` | The OAuth2 scope to request. Only valid with `PIVOT_ICEBERG_CREDENTIAL`. |

Set at most one of `PIVOT_ICEBERG_TOKEN` and `PIVOT_ICEBERG_CREDENTIAL`. A
variable that is set but empty is an error.

## Exit status

| Status | Meaning |
| --- | --- |
| `0` | The command finished normally: the shell was quit, or the server shut down after a signal. Statements that fail inside the shell do not change the exit status. |
| `1` | The command failed, for example because the datastore could not be opened or the configuration file is invalid. The error is printed to stderr as `pivot: <message>`. |
| `2` | The command line is invalid, such as an unknown flag or a missing argument. |

## Known limitations

- A plain `COPY ... FROM STDIN` is refused in the shell. Use
  [`\copy`](#loading-a-file) to load a local file instead.
- The shell serves exactly one datastore. To query several datastores
  together, configure them in a `pivot server`.
- The shell runs only interactively. It does not execute SQL from a file or
  from stdin.

## See also

- [Configuration file](/docs/reference/configuration/)
- [Datastores](/docs/reference/server/datastores/)
- [Authentication](/docs/reference/server/authentication/)
- [`SET` and `RESET`](/docs/reference/statements/set-reset/)
