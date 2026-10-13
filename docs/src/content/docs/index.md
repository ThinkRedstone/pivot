---
title: Introduction
description: What pivotdb is and how the pieces fit together.
---

Pivot is a high-performance analytics database that runs on open data formats. It delivers low-latency queries and high concurrency without replicating data into a dedicated database or proprietary format.

<figure class="arch-figure">
<svg viewBox="0 0 920 578" role="img" aria-labelledby="arch-title arch-desc">
<title id="arch-title">Pivot architecture</title>
<desc id="arch-desc">Object storage holds pivotlake tables in Delta Lake and Iceberg formats. A Pivot cluster reads and writes them and serves SQL clients over the Postgres wire. Agents, each embedding its own pivot open, and third-party engines such as Snowflake, read and write the same files directly without going through the cluster.</desc>
<defs>
<marker id="arch-head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
<path d="M0,0 L10,5 L0,10 z" class="arch-arrowhead" />
</marker>
</defs>
<rect x="40" y="24" width="840" height="156" rx="3" class="arch-panel" />
<text x="64" y="54" class="arch-title">Object storage</text>
<text x="856" y="54" text-anchor="end" class="arch-muted">S3 · GCS</text>
<text x="64" y="74" class="arch-muted">pivotlake</text>
<rect x="64" y="88" width="250" height="76" rx="3" class="arch-inner" />
<rect x="65" y="89" width="248" height="24" class="arch-strip" />
<line x1="65" y1="113" x2="313" y2="113" class="arch-rule" />
<text x="76" y="106" class="arch-label">events</text>
<text x="302" y="106" text-anchor="end" class="arch-tiny">Delta Lake format</text>
<text x="76" y="132" class="arch-tiny">part-00000-3f7a….parquet</text>
<text x="76" y="150" class="arch-tiny">part-00001-9c21….parquet</text>
<rect x="335" y="88" width="250" height="76" rx="3" class="arch-inner" />
<rect x="336" y="89" width="248" height="24" class="arch-strip" />
<line x1="336" y1="113" x2="584" y2="113" class="arch-rule" />
<text x="347" y="106" class="arch-label">orders</text>
<text x="573" y="106" text-anchor="end" class="arch-tiny">Delta Lake format</text>
<text x="347" y="132" class="arch-tiny">part-00000-3f7a….parquet</text>
<text x="347" y="150" class="arch-tiny">part-00001-9c21….parquet</text>
<rect x="606" y="88" width="250" height="76" rx="3" class="arch-inner" />
<rect x="607" y="89" width="248" height="24" class="arch-strip" />
<line x1="607" y1="113" x2="855" y2="113" class="arch-rule" />
<text x="618" y="106" class="arch-label">sessions</text>
<text x="844" y="106" text-anchor="end" class="arch-tiny">Iceberg format</text>
<text x="618" y="132" class="arch-tiny">part-00000-3f7a….parquet</text>
<text x="618" y="150" class="arch-tiny">part-00001-9c21….parquet</text>
<line x1="182" y1="250" x2="182" y2="182" class="arch-line" marker-start="url(#arch-head)" marker-end="url(#arch-head)" />
<text x="196" y="220" class="arch-muted">read + write</text>
<line x1="479" y1="250" x2="479" y2="182" class="arch-line" marker-start="url(#arch-head)" marker-end="url(#arch-head)" />
<text x="493" y="220" class="arch-muted">read + write</text>
<line x1="757" y1="250" x2="757" y2="182" class="arch-line" marker-start="url(#arch-head)" marker-end="url(#arch-head)" />
<text x="771" y="220" class="arch-muted">read + write</text>
<rect x="40" y="250" width="284" height="186" rx="3" class="arch-panel" />
<text x="64" y="280" class="arch-title">Pivot cluster</text>
<rect x="64" y="296" width="72" height="100" rx="2" class="arch-inner" />
<rect x="65" y="297" width="70" height="23" class="arch-strip" />
<line x1="65" y1="320" x2="135" y2="320" class="arch-rule" />
<text x="100" y="313" text-anchor="middle" class="arch-tiny">node 1</text>
<rect x="70" y="332" width="16" height="16" rx="1" class="arch-cell" />
<rect x="92" y="332" width="16" height="16" rx="1" class="arch-cell" />
<rect x="114" y="332" width="16" height="16" rx="1" class="arch-cell" />
<rect x="70" y="354" width="16" height="16" rx="1" class="arch-cell" />
<rect x="92" y="354" width="16" height="16" rx="1" class="arch-cell" />
<rect x="114" y="354" width="16" height="16" rx="1" class="arch-cell" />
<rect x="70" y="376" width="16" height="16" rx="1" class="arch-cell" />
<rect x="92" y="376" width="16" height="16" rx="1" class="arch-cell" />
<rect x="114" y="376" width="16" height="16" rx="1" class="arch-cell" />
<rect x="146" y="296" width="72" height="100" rx="2" class="arch-inner" />
<rect x="147" y="297" width="70" height="23" class="arch-strip" />
<line x1="147" y1="320" x2="217" y2="320" class="arch-rule" />
<text x="182" y="313" text-anchor="middle" class="arch-tiny">node 2</text>
<rect x="152" y="332" width="16" height="16" rx="1" class="arch-cell" />
<rect x="174" y="332" width="16" height="16" rx="1" class="arch-cell" />
<rect x="196" y="332" width="16" height="16" rx="1" class="arch-cell" />
<rect x="152" y="354" width="16" height="16" rx="1" class="arch-cell" />
<rect x="174" y="354" width="16" height="16" rx="1" class="arch-cell" />
<rect x="196" y="354" width="16" height="16" rx="1" class="arch-cell" />
<rect x="152" y="376" width="16" height="16" rx="1" class="arch-cell" />
<rect x="174" y="376" width="16" height="16" rx="1" class="arch-cell" />
<rect x="196" y="376" width="16" height="16" rx="1" class="arch-cell" />
<rect x="228" y="296" width="72" height="100" rx="2" class="arch-inner" />
<rect x="229" y="297" width="70" height="23" class="arch-strip" />
<line x1="229" y1="320" x2="299" y2="320" class="arch-rule" />
<text x="264" y="313" text-anchor="middle" class="arch-tiny">node 3</text>
<rect x="234" y="332" width="16" height="16" rx="1" class="arch-cell" />
<rect x="256" y="332" width="16" height="16" rx="1" class="arch-cell" />
<rect x="278" y="332" width="16" height="16" rx="1" class="arch-cell" />
<rect x="234" y="354" width="16" height="16" rx="1" class="arch-cell" />
<rect x="256" y="354" width="16" height="16" rx="1" class="arch-cell" />
<rect x="278" y="354" width="16" height="16" rx="1" class="arch-cell" />
<rect x="234" y="376" width="16" height="16" rx="1" class="arch-cell" />
<rect x="256" y="376" width="16" height="16" rx="1" class="arch-cell" />
<rect x="278" y="376" width="16" height="16" rx="1" class="arch-cell" />
<rect x="344" y="250" width="270" height="186" rx="3" class="arch-panel" />
<text x="479" y="280" text-anchor="middle" class="arch-title">Agents</text>
<rect x="360" y="296" width="238" height="44" rx="2" class="arch-inner" />
<circle cx="374" cy="310" r="3.5" class="arch-bullet" />
<text x="384" y="314" class="arch-cmd">Bash(pivot open s3://pivotlake)</text>
<text x="384" y="330" class="arch-tiny">└ opened 3 tables</text>
<rect x="360" y="352" width="238" height="44" rx="2" class="arch-inner" />
<circle cx="374" cy="366" r="3.5" class="arch-bullet" />
<text x="384" y="370" class="arch-cmd">Bash(pivot open s3://pivotlake)</text>
<text x="384" y="386" class="arch-tiny">└ opened 3 tables</text>
<rect x="634" y="250" width="246" height="186" rx="3" class="arch-panel" />
<text x="757" y="280" text-anchor="middle" class="arch-title">Other engines</text>
<rect x="648" y="296" width="104" height="32" rx="2" class="arch-inner" />
<rect x="657" y="305" width="14" height="14" rx="1" class="arch-cell" />
<text x="677" y="317" class="arch-tiny">Snowflake</text>
<rect x="762" y="296" width="104" height="32" rx="2" class="arch-inner" />
<rect x="771" y="305" width="14" height="14" rx="1" class="arch-cell" />
<text x="791" y="317" class="arch-tiny">Databricks</text>
<rect x="648" y="336" width="104" height="32" rx="2" class="arch-inner" />
<rect x="657" y="345" width="14" height="14" rx="1" class="arch-cell" />
<text x="677" y="357" class="arch-tiny">DuckDB</text>
<rect x="762" y="336" width="104" height="32" rx="2" class="arch-inner" />
<rect x="771" y="345" width="14" height="14" rx="1" class="arch-cell" />
<text x="791" y="357" class="arch-tiny">Spark</text>
<rect x="648" y="376" width="104" height="32" rx="2" class="arch-inner" />
<rect x="657" y="385" width="14" height="14" rx="1" class="arch-cell" />
<text x="677" y="397" class="arch-tiny">Trino</text>
<rect x="762" y="376" width="104" height="32" rx="2" class="arch-inner" />
<rect x="771" y="385" width="14" height="14" rx="1" class="arch-cell" />
<text x="791" y="397" class="arch-tiny">Polars</text>
<line x1="182" y1="486" x2="182" y2="438" class="arch-line" marker-end="url(#arch-head)" />
<text x="196" y="466" class="arch-muted">Postgres wire</text>
<rect x="40" y="486" width="284" height="72" rx="3" class="arch-panel" />
<text x="64" y="510" class="arch-title">SQL clients</text>
<rect x="64" y="520" width="72" height="26" rx="2" class="arch-inner" />
<text x="100" y="537" text-anchor="middle" class="arch-muted">backend</text>
<rect x="146" y="520" width="56" height="26" rx="2" class="arch-inner" />
<text x="174" y="537" text-anchor="middle" class="arch-muted">psql</text>
<rect x="212" y="520" width="80" height="26" rx="2" class="arch-inner" />
<text x="252" y="537" text-anchor="middle" class="arch-muted">BI tools</text>
</svg>
</figure>

#### Key features
- **Fast** - Written in Rust and built on state-of-the-art columnar execution techniques, including [morsel-driven parallelism](https://db.in.tum.de/~leis/papers/morsels.pdf), [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data), [NUMA](https://en.wikipedia.org/wiki/Non-uniform_memory_access)-aware execution, and cache-conscious aggregation and joins, in addition to a few novel additions ([Why is Pivot fast?](/docs/database/why-pivot-is-fast/)).
- **Scalable** - With object storage as its backing store, Pivot can be scaled up, down, or to zero almost instantly.
- **Portable** -     Pivot can run both as a server serving backends and clients, or as a local engine where users and agents query the source of truth directly, allowing local ad-hoc and agentic analytics to share the same engine and architecture as traditional dashboards and in-app analytics
- **Open** - Pivot is open source and built on open data formats (Delta Lake and Apache Iceberg). This means you can use Pivot with data already stored in your data warehouse, while data ingested by Pivot remains accessible to other query engines.

### Project status

Pivot is in early development and is not production ready. It is meant for
evaluation, experiments and local analysis today.

- **Breaking changes.** SQL surface, configuration keys, wire behaviour and
  the on-disk layout change between releases, with no migration path and no
  deprecation period. Expect to recreate a deployment rather than upgrade it.
- **Gaps in coverage.** Whole areas of SQL are unimplemented. What is
  supported today is listed under [SQL reference](/docs/reference/); assume
  anything not listed there is missing.

### Where to start

- [Quickstart](/docs/quickstart/) runs a server and issues a first query.
- [Architecture](/docs/database/architecture/) explains how a query becomes
  work across the Dispatch pool.
- [Roadmap](/docs/database/roadmap/) shows what is available, in progress, and planned.
- [SQL reference](/docs/reference/) documents commands, functions, operators,
  and data types with examples.
- [CLI & server reference](/docs/reference/cli/) covers CLI commands,
  configuration, datastores, authentication, and system tables.
