//! Public inputs, outputs, statistics, and errors for statement execution.

use std::fmt;
use std::time::Duration;

use arrow_schema::DataType;
use dispatch::DataFlowStats;
use thiserror::Error;
use tokio::task::JoinError;

use super::CopyIngest;

/// The session variable every frontend reads to toggle per-statement
/// execution stats: `SET pivot_stats = true` turns them on for the session,
/// `RESET pivot_stats` (or a falsy value) turns them off. The executor only
/// parses the statement; each frontend applies the toggle to its own session.
pub const STATS_VARIABLE: &str = "pivot_stats";

/// Whether a `SET` value spells a true boolean, the way PostgreSQL reads one.
pub fn is_truthy(value: &str) -> bool {
    matches!(
        value.to_ascii_lowercase().as_str(),
        "true" | "t" | "1" | "on" | "yes"
    )
}

/// Options that affect execution but not the statement's result.
#[derive(Clone, Copy, Debug, Default)]
pub struct ExecuteOptions {
    /// Collect the dispatch IO and CPU breakdown.
    pub collect_stats: bool,
    /// Mark the statement's dataflows for exclusive perf profiling. This is a
    /// no-op unless the crate is built with its `perf` feature.
    pub profile: bool,
}

/// One result column resolved during planning. This remains available for an
/// empty result, where no output batch exists to carry a schema.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResultColumn {
    pub name: String,
    pub data_type: DataType,
}

/// A statement that completed without returning a row set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Command {
    Insert {
        rows: usize,
    },
    CreateTable,
    CreateSchema,
    CreateUser,
    DropTable,
    Compact,
    Vacuum,
    /// `BEGIN`, `COMMIT` or `ROLLBACK`, answered without doing anything:
    /// every statement commits individually, so there is no transaction to
    /// open or resolve. Accepted so PostgreSQL drivers that wrap statements
    /// in a transaction by default can work.
    Begin,
    Commit,
    Rollback,
}

impl Command {
    /// PostgreSQL-style command-completion text.
    pub fn tag(&self) -> String {
        match self {
            Self::Insert { rows } => format!("INSERT 0 {rows}"),
            Self::CreateTable => "CREATE TABLE".to_string(),
            Self::CreateSchema => "CREATE SCHEMA".to_string(),
            Self::CreateUser => "CREATE USER".to_string(),
            Self::DropTable => "DROP TABLE".to_string(),
            Self::Compact => "COMPACT".to_string(),
            Self::Vacuum => "VACUUM".to_string(),
            Self::Begin => "BEGIN".to_string(),
            Self::Commit => "COMMIT".to_string(),
            Self::Rollback => "ROLLBACK".to_string(),
        }
    }
}

impl fmt::Display for Command {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.tag())
    }
}

/// Semantic output from one statement. `T` is the frontend-selected material
/// produced for each query batch on a dispatch worker.
#[derive(Debug)]
pub enum StatementOutput<T> {
    Rows {
        columns: Vec<ResultColumn>,
        batches: Vec<T>,
    },
    Command(Command),
    /// A parsed `SET` or `RESET`. Frontends own session variables, so the
    /// executor returns the name and optional value rather than applying it.
    Set {
        name: String,
        value: Option<String>,
    },
    /// A `COPY ... FROM STDIN` with its ingest dataflow already running: the
    /// rows arrive over the frontend's own protocol, so the frontend feeds
    /// [`CopyIngest::push`] and then either [`CopyIngest::finish`]es (which
    /// commits) or drops the ingest (which aborts and rolls back). A frontend
    /// with no copy-in channel just drops it.
    CopyFromStdin(Box<CopyIngest>),
}

/// Phase timings and aggregate dataflow work for one statement.
#[derive(Clone, Debug, Default)]
pub struct ExecutionStats {
    pub plan: Duration,
    pub compile: Duration,
    pub execute: Duration,
    pub flow: DataFlowStats,
}

impl ExecutionStats {
    /// Render the server's compact `pivot_stats` notice.
    pub fn summary(&self) -> String {
        let ms = |duration: Duration| duration.as_secs_f64() * 1e3;
        let mib = |bytes: u64| bytes as f64 / (1024.0 * 1024.0);
        format!(
            "stats: plan={:.1}ms compile={:.1}ms exec={:.1}ms | \
             disk={} ops/{:.1}MiB/read={:.1}ms/write={:.1}ms  \
             http={} ops/{:.1}MiB/get={:.1}ms/reissued={}/upload={:.1}ms  \
             http-disk-cache={} ops/{:.1}MiB/{:.1}ms  cpu={:.1}ms",
            ms(self.plan),
            ms(self.compile),
            ms(self.execute),
            self.flow.disk_requests,
            mib(self.flow.disk_bytes),
            ms(self.flow.disk_read_time),
            ms(self.flow.disk_write_time),
            self.flow.http_requests,
            mib(self.flow.http_bytes),
            ms(self.flow.http_get_time),
            self.flow.http_reissues,
            ms(self.flow.http_upload_time),
            self.flow.disk_cache_requests,
            mib(self.flow.disk_cache_bytes),
            ms(self.flow.disk_cache_time),
            ms(self.flow.cpu),
        )
    }
}

/// A completed statement and its execution measurements.
#[derive(Debug)]
pub struct Execution<T> {
    pub output: StatementOutput<T>,
    pub stats: ExecutionStats,
}

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    Plan(#[from] planner::Error),
    #[error("compile error: {0}")]
    Compile(#[from] planner::compile::Error),
    #[error(transparent)]
    DataFlow(#[from] dispatch::DataFlowError),
    #[error(transparent)]
    Catalog(#[from] planner::catalog::Error),
    #[error("waiter thread panicked: {0}")]
    WorkerPanic(JoinError),
    #[error("planner thread panicked: {0}")]
    PlannerPanic(JoinError),
    #[error("invalid INSERT row-count result: {0}")]
    InvalidInsertResult(String),
    #[error("{0}")]
    Copy(String),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
