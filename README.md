<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset=".github/assets/pivot-wordmark-dark.png">
    <img alt="Pivot" src="docs/src/assets/pivot-wordmark.png" width="240">
  </picture>

  <p><strong>A high-performance analytics engine on open data formats.</strong></p>

  <a href="https://github.com/pivotlake/pivot/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/pivotlake/pivot/actions/workflows/ci.yml/badge.svg?branch=main"></a>
  <a href="#license"><img alt="License: MIT OR Apache-2.0" src="https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue"></a>

  <p>
    <a href="https://pivotlake.io/docs">Docs</a> ·
    <a href="https://pivotlake.io/docs/quickstart/">Quickstart</a> ·
    <a href="https://github.com/pivotlake/pivot/discussions">Discussions</a>
  </p>
</div>

Pivot is a high-performance analytics database that runs on open data formats. It delivers low-latency queries and high concurrency without replicating data into a dedicated real-time analytics database such as ClickHouse / Druid.

<p align="center">
  <a href="https://github.com/pivotlake/benchmarks">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset=".github/assets/benchmarks-dark.svg">
      <img alt="Relative query time on ClickBench, TPC-H SF1000 and the Star Schema Benchmark SF1000: Pivot is the fastest engine on all three" src=".github/assets/benchmarks.svg">
    </picture>
  </a>
</p>

## Key features

- **Fast** - Written in Rust and built on state-of-the-art columnar execution techniques, including morsel-driven parallelism, SIMD, NUMA-aware execution, and cache-conscious aggregation and joins, in addition to a few novel additions.
- **Scalable** - With object storage as its backing store, Pivot can be scaled up, down, or to zero almost instantly.
- **Portable** -     Pivot can run both as a server serving backends and clients, or as a local engine where users and agents query the source of truth directly—allowing local ad-hoc and agentic analytics to share the same engine and architecture as traditional dashboards and in-app analytics
- **Open** - Pivot is open source and built on open data formats (Delta Lake and Iceberg). This means you can use Pivot with data already stored in your data warehouse, while data ingested by Pivot remains accessible to other query engines.

## Getting started

```sh
curl https://pivotlake.io | sh
```

See the [Quickstart](https://pivotlake.io/docs/quickstart/) for Docker, the
Debian package and a first query.

> [!NOTE]
> Pivot is in early development and not yet production ready. Expect breaking
> changes between releases.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md), and ask
questions in [Discussions](https://github.com/pivotlake/pivot/discussions).

## License

Licensed under either of the [Apache License, Version 2.0](LICENSE-APACHE) or
the [MIT license](LICENSE-MIT), at your option.
