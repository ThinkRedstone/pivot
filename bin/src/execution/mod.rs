//! Transport-neutral SQL execution for Pivot.
//!
//! [`Executor`] owns the planner cache and coordinates one statement
//! transaction across planning, dataflow execution, cancellation, and commit
//! or rollback. Frontends pass the conversion each query batch goes through
//! before it leaves a dispatch worker. The PostgreSQL frontend turns them into wire rows,
//! while a local frontend can copy Arrow batches or turn cells into terminal
//! text.

mod copy;
mod dataflow;
mod planning;
mod types;

use std::sync::Arc;
use std::time::Instant;

use arrow_array::RecordBatch;

use dataflow::{
    StatementHandle, StatementKind, StatementResults, affected_rows,
    affected_rows_from_record_batch, collect_handle, execute_compact, result_columns,
};
use planner::operator::TransactionStatement;
use planning::{PlanCache, PlannerPool, plan_query};

pub use copy::CopyIngest;
pub use types::{
    Command, Error, ExecuteOptions, Execution, ExecutionStats, Result, ResultColumn,
    STATS_VARIABLE, StatementOutput, is_truthy,
};

/// Shared SQL execution state for every frontend of one Pivot instance.
#[derive(Clone)]
pub struct Executor {
    catalog: Arc<catalog::PivotCatalog>,
    dispatcher: dispatch::DataFlowDispatcher,
    plan_cache: Arc<PlanCache>,
    planner_pool: Arc<PlannerPool>,
}

impl Executor {
    pub fn new(
        catalog: Arc<catalog::PivotCatalog>,
        dispatcher: dispatch::DataFlowDispatcher,
    ) -> Result<Self> {
        Ok(Self {
            planner_pool: Arc::new(PlannerPool::new(catalog.clone())?),
            catalog,
            dispatcher,
            plan_cache: Arc::new(PlanCache::default()),
        })
    }

    /// Execute a statement and copy its Arrow batches out of dispatch memory.
    pub async fn execute(
        &self,
        sql: String,
        options: ExecuteOptions,
    ) -> Result<Execution<RecordBatch>> {
        self.execute_with_output(sql, options, dispatch::copy_out)
            .await
    }

    /// Execute a statement, converting each result batch with `convert`.
    ///
    /// `convert` runs on each dispatch worker before the result leaves
    /// dispatch memory; an `Err` fails the statement.
    pub async fn execute_with_output<T, F>(
        &self,
        sql: String,
        options: ExecuteOptions,
        convert: F,
    ) -> Result<Execution<T>>
    where
        T: Send + 'static,
        F: dispatch::OutputConvert<T>,
    {
        let transaction = self.catalog.begin_transaction();
        let result = self
            .execute_in_transaction(sql, options, transaction.clone(), convert)
            .await;
        match result {
            // A COPY FROM STDIN hands the transaction to whoever drives the
            // ingest instead of resolving it here.
            Ok(execution) if matches!(execution.output, StatementOutput::CopyFromStdin(_)) => {
                Ok(execution)
            }
            Ok(execution) => {
                transaction.commit().await?;
                Ok(execution)
            }
            Err(error) => {
                // A rollback removes what the statement already uploaded:
                // blocking store I/O, kept off the runtime.
                if let Err(panic) =
                    tokio::task::spawn_blocking(move || transaction.rollback()).await
                {
                    tracing::error!(%panic, "rollback thread panicked");
                }
                Err(error)
            }
        }
    }

    /// Plan a statement without executing it and report its result shape:
    /// `Some` with the row description for statements that return rows,
    /// `None` for those that do not (commands, SET, COMPACT, VACUUM, COPY FROM
    /// STDIN). Planning runs in its own transaction, rolled back before
    /// returning.
    pub async fn describe(&self, sql: &str) -> Result<Option<Vec<ResultColumn>>> {
        let transaction = self.catalog.begin_transaction();
        let plan = plan_query(
            &self.planner_pool,
            transaction.clone(),
            self.plan_cache.as_ref(),
            sql,
        )
        .await;
        transaction.rollback();
        let plan = plan?;

        if plan.as_set_variable().is_some()
            || plan.as_compact().is_some()
            || plan.is_vacuum()
            || plan.as_copy_from_stdin().is_some()
            || plan.as_transaction_stmt().is_some()
        {
            return Ok(None);
        }
        match StatementKind::from_plan(&plan) {
            StatementKind::Query => Ok(Some(result_columns(&plan)?)),
            _ => Ok(None),
        }
    }

    async fn execute_in_transaction<T, F>(
        &self,
        sql: String,
        options: ExecuteOptions,
        transaction: Arc<dyn planner::catalog::CatalogTransaction>,
        convert: F,
    ) -> Result<Execution<T>>
    where
        T: Send + 'static,
        F: dispatch::OutputConvert<T>,
    {
        let started = Instant::now();
        let plan = plan_query(
            &self.planner_pool,
            transaction.clone(),
            self.plan_cache.as_ref(),
            &sql,
        )
        .await?;
        let plan_time = started.elapsed();

        if let Some(set) = plan.as_set_variable() {
            return Ok(Execution {
                output: StatementOutput::Set {
                    name: set.name.clone(),
                    value: set.value.clone(),
                },
                stats: ExecutionStats {
                    plan: plan_time,
                    ..ExecutionStats::default()
                },
            });
        }

        // BEGIN/COMMIT/ROLLBACK are answered without doing anything: every
        // statement commits individually, so there is no transaction to open
        // or resolve. PostgreSQL drivers that wrap statements in a transaction
        // by default get their expected tags, but no statement grouping: work
        // done between a BEGIN and a ROLLBACK is already committed and stays.
        if let Some(statement) = plan.as_transaction_stmt() {
            return Ok(Execution {
                output: StatementOutput::Command(match statement {
                    TransactionStatement::Begin => Command::Begin,
                    TransactionStatement::Commit => Command::Commit,
                    TransactionStatement::Rollback => Command::Rollback,
                }),
                stats: ExecutionStats {
                    plan: plan_time,
                    ..ExecutionStats::default()
                },
            });
        }

        if let Some(request) = plan.as_compact() {
            let started = Instant::now();
            execute_compact(&self.catalog, transaction.as_ref(), request).await?;
            return Ok(Execution {
                output: StatementOutput::Command(Command::Compact),
                stats: ExecutionStats {
                    plan: plan_time,
                    execute: started.elapsed(),
                    ..ExecutionStats::default()
                },
            });
        }

        if plan.is_vacuum() {
            let started = Instant::now();
            transaction
                .vacuum(self.catalog.default_datastore_name())
                .await?;
            return Ok(Execution {
                output: StatementOutput::Command(Command::Vacuum),
                stats: ExecutionStats {
                    plan: plan_time,
                    execute: started.elapsed(),
                    ..ExecutionStats::default()
                },
            });
        }

        // A COPY FROM STDIN never compiles through the plan: its rows arrive
        // later over the frontend's protocol. Launch its ingest dataflow here
        // and hand the running exchange back; the frontend feeds it and
        // resolves it (or drops it, which aborts).
        if let Some(statement) = plan.as_copy_from_stdin() {
            let ingest = self
                .launch_copy_ingest(statement.clone(), transaction)
                .await?;
            return Ok(Execution {
                output: StatementOutput::CopyFromStdin(Box::new(ingest)),
                stats: ExecutionStats {
                    plan: plan_time,
                    ..ExecutionStats::default()
                },
            });
        }

        let statement_kind = StatementKind::from_plan(&plan);
        let columns = result_columns(&plan)?;
        let dispatcher = self.dispatcher.clone();
        #[cfg(feature = "perf")]
        let dispatcher = dispatcher.with_profiling(options.profile);

        let started = Instant::now();
        let compile_transaction = transaction;
        let handle = tokio::task::spawn_blocking(move || -> Result<StatementHandle<T>> {
            let spec = plan.compile(&dispatcher, compile_transaction.as_ref())?;
            Ok(match (statement_kind, options.collect_stats) {
                (StatementKind::Insert, true) => {
                    let output = spec.map(|| affected_rows_from_record_batch);
                    StatementHandle::Insert(output.execute_with_stats())
                }
                (StatementKind::Insert, false) => {
                    let output = spec.map(|| affected_rows_from_record_batch);
                    StatementHandle::Insert(output.execute())
                }
                (_, true) => StatementHandle::Batches(spec.execute_with_stats(convert)),
                (_, false) => StatementHandle::Batches(spec.execute(convert)),
            })
        })
        .await
        .map_err(Error::PlannerPanic)??;
        let compile_time = started.elapsed();

        let (results, flow, execute_time) = collect_handle(handle).await?;
        let output = match (statement_kind, results) {
            (StatementKind::Query, StatementResults::Batches(batches)) => {
                StatementOutput::Rows { columns, batches }
            }
            (StatementKind::Insert, StatementResults::Insert(outputs)) => {
                StatementOutput::Command(Command::Insert {
                    rows: affected_rows(outputs)?,
                })
            }
            (StatementKind::CreateTable, StatementResults::Batches(_)) => {
                StatementOutput::Command(Command::CreateTable)
            }
            (StatementKind::CreateSchema, StatementResults::Batches(_)) => {
                StatementOutput::Command(Command::CreateSchema)
            }
            (StatementKind::CreateUser, StatementResults::Batches(_)) => {
                StatementOutput::Command(Command::CreateUser)
            }
            (StatementKind::DropTable, StatementResults::Batches(_)) => {
                StatementOutput::Command(Command::DropTable)
            }
            _ => unreachable!("statement kind and worker output must agree"),
        };
        Ok(Execution {
            output,
            stats: ExecutionStats {
                plan: plan_time,
                compile: compile_time,
                execute: execute_time,
                flow,
            },
        })
    }
}
