//! PostgreSQL wire adapters around the transport-neutral query executor.

use std::fmt::Debug;
use std::sync::Arc;

use crate::execution::{Command, ExecuteOptions, Execution, StatementOutput, is_truthy};
use crate::server::arrow_to_pgwire::{
    PGBinaryRowBatch, PGRowBatch, PGRowEncoding, PGTextRowBatch, build_field_info,
};
use crate::server::auth::Authenticator;
use crate::server::copy_session;
use arrow_schema::{Field, Schema};
use async_trait::async_trait;
use catalog::metastore::Metastore;
use futures::{Sink, SinkExt, stream};
use pgwire::api::auth::StartupHandler;
use pgwire::api::cancel::{CancelHandler, DefaultCancelHandler};
use pgwire::api::copy::CopyHandler;
use pgwire::api::portal::{Format, Portal};
use pgwire::api::query::{ExtendedQueryHandler, SimpleQueryHandler};
use pgwire::api::results::{
    CopyResponse, DescribePortalResponse, DescribeStatementResponse, FieldFormat, FieldInfo,
    QueryResponse, Response, Tag,
};
use pgwire::api::stmt::{NoopQueryParser, StoredStatement};
use pgwire::api::store::PortalStore;
use pgwire::api::{
    ClientInfo, ClientPortalStore, ConnectionManager, PgWireConnectionState, PgWireServerHandlers,
};
use pgwire::error::{ErrorInfo, PgWireError, PgWireResult};
use pgwire::messages::PgWireBackendMessage;
use pgwire::messages::copy::{CopyData, CopyDone, CopyFail};
use pgwire::messages::response::NoticeResponse;
use tracing::{info, warn};

/// Per-connection flag toggled with `SET pivot_stats = true`.
const STATS_FLAG: &str = crate::execution::STATS_VARIABLE;

/// Per-connection flag toggled with `SET perf = 1`.
#[cfg(feature = "perf")]
const PERF_FLAG: &str = "perf";

fn into_pgwire_error(error: crate::execution::Error) -> PgWireError {
    user_error(error.to_string())
}

fn user_error(message: String) -> PgWireError {
    let info = ErrorInfo::new("ERROR".to_string(), "XX000".to_string(), message);
    PgWireError::UserError(Box::new(info))
}

fn fields_for_columns(
    columns: Vec<crate::execution::ResultColumn>,
    format: FieldFormat,
) -> Arc<Vec<FieldInfo>> {
    let schema = Arc::new(Schema::new(
        columns
            .into_iter()
            .map(|column| Field::new(column.name, column.data_type, true))
            .collect::<Vec<_>>(),
    ));
    build_field_info(&schema, format)
}

fn build_query_response<T: PGRowEncoding>(
    columns: Vec<crate::execution::ResultColumn>,
    batches: Vec<T>,
) -> Response {
    let batches: Vec<PGRowBatch> = batches.into_iter().map(Into::into).collect();
    let fields = batches.first().map_or_else(
        || fields_for_columns(columns, T::FORMAT),
        |batch| batch.fields.clone(),
    );
    Response::Query(QueryResponse::new(
        fields,
        stream::iter(batches.into_iter().flat_map(|batch| batch.rows).map(Ok)),
    ))
}

fn build_command_response(command: Command) -> Response {
    match command {
        Command::Insert { rows } => {
            Response::Execution(Tag::new("INSERT").with_oid(0).with_rows(rows))
        }
        Command::CreateTable => Response::Execution(Tag::new("CREATE TABLE")),
        Command::CreateSchema => Response::Execution(Tag::new("CREATE SCHEMA")),
        Command::CreateUser => Response::Execution(Tag::new("CREATE USER")),
        Command::DropTable => Response::Execution(Tag::new("DROP TABLE")),
        Command::Compact => Response::Execution(Tag::new("COMPACT")),
        Command::Vacuum => Response::Execution(Tag::new("VACUUM")),
        Command::Begin => Response::Execution(Tag::new("BEGIN")),
        Command::Commit => Response::Execution(Tag::new("COMMIT")),
        Command::Rollback => Response::Execution(Tag::new("ROLLBACK")),
    }
}

/// The pgwire simple-query frontend for one shared executor.
pub struct PivotQueryHandler {
    executor: Arc<crate::execution::Executor>,
}

impl PivotQueryHandler {
    pub fn new(executor: Arc<crate::execution::Executor>) -> Self {
        Self { executor }
    }

    async fn run_query<T: PGRowEncoding>(
        &self,
        query: &str,
        collect_stats: bool,
        with_perf: bool,
    ) -> crate::execution::Result<Execution<T>> {
        #[cfg(feature = "perf")]
        let mut perf_guard = if with_perf {
            let sql = query.to_string();
            tokio::task::spawn_blocking(move || crate::server::perf::start(&sql))
                .await
                .ok()
                .flatten()
        } else {
            None
        };
        #[cfg(not(feature = "perf"))]
        let _ = with_perf;
        #[cfg(feature = "perf")]
        let profile = perf_guard.is_some();
        #[cfg(not(feature = "perf"))]
        let profile = false;

        let result = self
            .executor
            .execute_with_output::<T>(
                query.to_string(),
                ExecuteOptions {
                    collect_stats,
                    profile,
                },
            )
            .await;

        #[cfg(feature = "perf")]
        if let Some(perf) = perf_guard.take() {
            let _ = tokio::task::spawn_blocking(move || drop(perf)).await;
        }
        result
    }
}

fn apply_set<C: ClientInfo>(
    client: &mut C,
    name: &str,
    value: Option<&str>,
) -> PgWireResult<Response> {
    // The session time zone is fixed to UTC: every timestamptz renders,
    // parses, and truncates in UTC. A `SET TimeZone` to a UTC spelling is
    // accepted (drivers send one at startup), anything else is refused so a
    // client never gets UTC answers while believing another zone applies.
    // `RESET TimeZone` restores the default, which is UTC already.
    if name.eq_ignore_ascii_case("timezone")
        && let Some(zone) = value
        && !is_utc_zone(zone)
    {
        return Err(user_error(format!(
            "unsupported TimeZone {zone:?}: this server runs sessions in UTC only"
        )));
    }
    if name.eq_ignore_ascii_case(STATS_FLAG) {
        if value.is_some_and(is_truthy) {
            client
                .metadata_mut()
                .insert(STATS_FLAG.to_string(), "on".to_string());
        } else {
            client.metadata_mut().remove(STATS_FLAG);
        }
    }
    #[cfg(feature = "perf")]
    if name.eq_ignore_ascii_case(PERF_FLAG) {
        if value.is_some_and(is_truthy) {
            client
                .metadata_mut()
                .insert(PERF_FLAG.to_string(), "on".to_string());
        } else {
            client.metadata_mut().remove(PERF_FLAG);
        }
    }
    Ok(Response::Execution(Tag::new(if value.is_none() {
        "RESET"
    } else {
        "SET"
    })))
}

/// The `TimeZone` spellings that mean UTC. Offset spellings are not expanded
/// beyond the common zero offsets; a session asking for any other zone is
/// refused rather than quietly served UTC.
fn is_utc_zone(zone: &str) -> bool {
    matches!(
        zone.to_ascii_lowercase().trim_matches('\''),
        "utc" | "etc/utc" | "gmt" | "etc/gmt" | "+00" | "+00:00" | "00:00"
    )
}

fn stats_on<C: ClientInfo>(client: &C) -> bool {
    client
        .metadata()
        .get(STATS_FLAG)
        .is_some_and(|value| value == "on")
}

#[cfg(feature = "perf")]
fn perf_on<C: ClientInfo>(client: &C) -> bool {
    client
        .metadata()
        .get(PERF_FLAG)
        .is_some_and(|value| value == "on")
}

#[async_trait]
impl CopyHandler for PivotQueryHandler {
    async fn on_copy_data<C>(&self, client: &mut C, data: CopyData) -> PgWireResult<()>
    where
        C: ClientInfo + Sink<PgWireBackendMessage> + Unpin + Send + Sync,
        C::Error: Debug,
        PgWireError: From<<C as Sink<PgWireBackendMessage>>::Error>,
    {
        copy_session::push_copy(client, data.data)
            .await
            .map_err(copy_session::Error::into_pgwire)
    }

    async fn on_copy_done<C>(&self, client: &mut C, _done: CopyDone) -> PgWireResult<()>
    where
        C: ClientInfo + Sink<PgWireBackendMessage> + Unpin + Send + Sync,
        C::Error: Debug,
        PgWireError: From<<C as Sink<PgWireBackendMessage>>::Error>,
    {
        let count = copy_session::finish_copy(client).await.map_err(|error| {
            warn!(%error, "copy from stdin failed");
            error.into_pgwire()
        })?;
        info!(rows = count, "copy from stdin committed");
        client
            .send(PgWireBackendMessage::CommandComplete(
                Tag::new("COPY").with_rows(count).into(),
            ))
            .await?;
        // In the extended protocol pgwire leaves the connection in copy-in
        // state; step out of it so the client's trailing Sync reaches the
        // extended handler and gets its ReadyForQuery. (The simple-protocol
        // loop resets the state itself.)
        if matches!(client.state(), PgWireConnectionState::CopyInProgress(true)) {
            client.set_state(PgWireConnectionState::ReadyForQuery);
        }
        Ok(())
    }

    async fn on_copy_fail<C>(&self, client: &mut C, fail: CopyFail) -> PgWireError
    where
        C: ClientInfo + Sink<PgWireBackendMessage> + Unpin + Send + Sync,
        C::Error: Debug,
        PgWireError: From<<C as Sink<PgWireBackendMessage>>::Error>,
    {
        copy_session::abort_copy(client).await;
        copy_session::Error::ClientAbort(fail.message).into_pgwire()
    }
}

impl PivotQueryHandler {
    /// Execute one statement and build its wire response, its rows encoded
    /// as `T` dictates. Shared by the simple and extended protocols; a COPY
    /// FROM STDIN switches the connection into copy-in mode on the way out.
    async fn execute_statement<T, C>(&self, client: &mut C, query: &str) -> PgWireResult<Response>
    where
        T: PGRowEncoding,
        C: ClientInfo + Sink<PgWireBackendMessage> + Unpin + Send + Sync,
        C::Error: Debug,
        PgWireError: From<<C as Sink<PgWireBackendMessage>>::Error>,
    {
        let with_stats = stats_on(client);
        #[cfg(feature = "perf")]
        let with_perf = perf_on(client);
        #[cfg(not(feature = "perf"))]
        let with_perf = false;

        info!(sql = %query, "query received");
        let execution = self
            .run_query::<T>(query, with_stats, with_perf)
            .await
            .map_err(|error| {
                warn!(%error, sql = %query, "query failed");
                into_pgwire_error(error)
            })?;

        if with_stats {
            let notice = NoticeResponse::from(ErrorInfo::new(
                "INFO".to_string(),
                "00000".to_string(),
                execution.stats.summary(),
            ));
            client
                .send(PgWireBackendMessage::NoticeResponse(notice))
                .await?;
        }

        let response = match execution.output {
            StatementOutput::Rows { columns, batches } => build_query_response(columns, batches),
            StatementOutput::Command(command) => build_command_response(command),
            StatementOutput::Set { name, value } => apply_set(client, &name, value.as_deref())?,
            StatementOutput::CopyFromStdin(ingest) => {
                // The CopyInResponse advertises a binary payload (the Arrow
                // IPC stream), so clients know not to apply text escaping to
                // the bytes they send.
                const BINARY_FORMAT: i8 = 1;
                let columns = copy_session::begin_copy(client, *ingest)
                    .await
                    .map_err(|error| {
                        warn!(%error, sql = %query, "copy from stdin failed to start");
                        error.into_pgwire()
                    })?;
                info!(sql = %query, "copy from stdin started");
                Response::CopyIn(CopyResponse::new(BINARY_FORMAT, columns, stream::empty()))
            }
        };

        info!(sql = %query, "query succeeded");
        Ok(response)
    }
}

#[async_trait]
impl SimpleQueryHandler for PivotQueryHandler {
    async fn do_query<C>(&self, client: &mut C, query: &str) -> PgWireResult<Vec<Response>>
    where
        C: ClientInfo + ClientPortalStore + Sink<PgWireBackendMessage> + Unpin + Send + Sync,
        C::PortalStore: PortalStore,
        C::Error: Debug,
        PgWireError: From<<C as Sink<PgWireBackendMessage>>::Error>,
    {
        Ok(vec![
            self.execute_statement::<PGTextRowBatch, _>(client, query)
                .await?,
        ])
    }
}

/// The one wire format a Bind's result-column format request asks every
/// column to use. Rows are encoded in a single format, so a request mixing
/// text and binary columns is refused.
fn resolve_result_format(format: &Format) -> PgWireResult<FieldFormat> {
    const TEXT: i16 = 0;
    const BINARY: i16 = 1;
    match format {
        Format::UnifiedText => Ok(FieldFormat::Text),
        Format::UnifiedBinary => Ok(FieldFormat::Binary),
        Format::Individual(codes) if codes.iter().all(|&code| code == TEXT) => {
            Ok(FieldFormat::Text)
        }
        Format::Individual(codes) if codes.iter().all(|&code| code == BINARY) => {
            Ok(FieldFormat::Binary)
        }
        Format::Individual(codes) => Err(user_error(format!(
            "result format codes {codes:?} mix text and binary columns, which is not supported"
        ))),
    }
}

#[async_trait]
impl ExtendedQueryHandler for PivotQueryHandler {
    type Statement = String;
    type QueryParser = NoopQueryParser;

    fn query_parser(&self) -> Arc<Self::QueryParser> {
        Arc::new(NoopQueryParser)
    }

    async fn do_describe_statement<C>(
        &self,
        _client: &mut C,
        target: &StoredStatement<Self::Statement>,
    ) -> PgWireResult<DescribeStatementResponse>
    where
        C: ClientInfo + ClientPortalStore + Sink<PgWireBackendMessage> + Unpin + Send + Sync,
        C::PortalStore: PortalStore<Statement = Self::Statement>,
        C::Error: Debug,
        PgWireError: From<<C as Sink<PgWireBackendMessage>>::Error>,
    {
        if !target.parameter_types.is_empty() {
            return Err(user_error(
                "prepared-statement parameters are not supported yet".to_string(),
            ));
        }
        let columns = self
            .executor
            .describe(&target.statement)
            .await
            .map_err(into_pgwire_error)?;
        Ok(match columns {
            // Result formats are only chosen at Bind, so a statement
            // describes its columns as text, as a Postgres server does.
            Some(columns) => DescribeStatementResponse::new(
                Vec::new(),
                fields_for_columns(columns, FieldFormat::Text)
                    .as_ref()
                    .clone(),
            ),
            None => DescribeStatementResponse::new(Vec::new(), Vec::new()),
        })
    }

    async fn do_describe_portal<C>(
        &self,
        _client: &mut C,
        target: &Portal<Self::Statement>,
    ) -> PgWireResult<DescribePortalResponse>
    where
        C: ClientInfo + ClientPortalStore + Sink<PgWireBackendMessage> + Unpin + Send + Sync,
        C::PortalStore: PortalStore<Statement = Self::Statement>,
        C::Error: Debug,
        PgWireError: From<<C as Sink<PgWireBackendMessage>>::Error>,
    {
        let format = resolve_result_format(&target.result_column_format)?;
        let columns = self
            .executor
            .describe(&target.statement.statement)
            .await
            .map_err(into_pgwire_error)?;
        Ok(match columns {
            Some(columns) => {
                DescribePortalResponse::new(fields_for_columns(columns, format).as_ref().clone())
            }
            None => DescribePortalResponse::new(Vec::new()),
        })
    }

    async fn do_query<C>(
        &self,
        client: &mut C,
        portal: &Portal<Self::Statement>,
        _max_rows: usize,
    ) -> PgWireResult<Response>
    where
        C: ClientInfo + ClientPortalStore + Sink<PgWireBackendMessage> + Unpin + Send + Sync,
        C::PortalStore: PortalStore<Statement = Self::Statement>,
        C::Error: Debug,
        PgWireError: From<<C as Sink<PgWireBackendMessage>>::Error>,
    {
        if !portal.parameters.is_empty() {
            return Err(user_error(
                "prepared-statement parameters are not supported yet".to_string(),
            ));
        }
        // Resolved before executing, so a refused request has no effects.
        let format = resolve_result_format(&portal.result_column_format)?;
        let query = &portal.statement.statement;
        match format {
            FieldFormat::Text => {
                self.execute_statement::<PGTextRowBatch, _>(client, query)
                    .await
            }
            FieldFormat::Binary => {
                self.execute_statement::<PGBinaryRowBatch, _>(client, query)
                    .await
            }
        }
    }
}

/// The startup, query, cancel, and unsupported-protocol handlers handed to
/// pgwire for each connection.
pub struct PivotHandlers {
    query_handler: Arc<PivotQueryHandler>,
    cancel_handler: Arc<DefaultCancelHandler>,
    authenticator: Authenticator,
}

impl PivotHandlers {
    pub fn new(
        query_executor: Arc<crate::execution::Executor>,
        metastore: Arc<dyn Metastore>,
    ) -> Self {
        let manager = Arc::new(ConnectionManager::new());
        Self {
            query_handler: Arc::new(PivotQueryHandler::new(query_executor)),
            cancel_handler: Arc::new(DefaultCancelHandler::new(manager.clone())),
            authenticator: Authenticator::new(metastore, manager),
        }
    }
}

impl PgWireServerHandlers for PivotHandlers {
    fn simple_query_handler(&self) -> Arc<impl SimpleQueryHandler> {
        self.query_handler.clone()
    }

    fn startup_handler(&self) -> Arc<impl StartupHandler> {
        self.authenticator.startup_handler()
    }

    fn cancel_handler(&self) -> Arc<impl CancelHandler> {
        self.cancel_handler.clone()
    }

    fn extended_query_handler(&self) -> Arc<impl pgwire::api::query::ExtendedQueryHandler> {
        self.query_handler.clone()
    }

    fn copy_handler(&self) -> Arc<impl pgwire::api::copy::CopyHandler> {
        self.query_handler.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::is_truthy;

    #[test]
    fn truthy_set_values_are_case_insensitive() {
        assert!(is_truthy("TRUE"));
        assert!(is_truthy("yes"));
        assert!(!is_truthy("off"));
    }
}
