//! duckdb-planner: a thin FFI wrapper around DuckDB's C++ query planner.
//!
//! This crate sends SQL strings to an embedded DuckDB instance and exposes the
//! resolved logical plan as safe borrowed handles ([`LogicalOp`] / [`Expr`]).
//! It does *not* materialize the plan into owned Rust structs: walking the
//! handles and building whatever IR a caller needs is the caller's job.
//!
//! The main entry point is [`PlannerContext`], which owns the DuckDB state and
//! exposes [`plan`](PlannerContext::plan) (returns a [`Plan`] handle).
//!
//! # Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use duckdb_planner::{
//!     DuckDBColumn, DuckDBTable, DuckDBTransaction, LogicalTypeId, PlannerContext,
//! };
//! use duckdb_planner::duckdb_bridge::duckdb_types::LogicalOperatorType;
//!
//! struct UsersTable;
//!
//! impl DuckDBTable for UsersTable {
//!     fn clone_box(&self) -> Box<dyn DuckDBTable> { Box::new(UsersTable) }
//!     fn duckdb_typed_columns(&self) -> Vec<DuckDBColumn> {
//!         vec![DuckDBColumn {
//!             name: "name".to_string(),
//!             duckdb_logical_type_id: LogicalTypeId::VARCHAR as u8,
//!             decimal_width: 0,
//!             decimal_scale: 0,
//!         }]
//!     }
//! }
//!
//! // BoundTable names resolve through a per-plan transaction (a snapshot of the
//! // catalog).
//! struct MyTransaction;
//!
//! impl DuckDBTransaction for MyTransaction {
//!     fn does_schema_exist(&self, _datastore: &str, schema: &str) -> duckdb_planner::catalog_provider::Result<bool> {
//!         Ok(schema == "main")
//!     }
//!     fn bind_table(&self, _datastore: &str, _schema: &str, name: &str) -> duckdb_planner::catalog_provider::Result<Option<Box<dyn DuckDBTable>>> {
//!         Ok(match name {
//!             "users" => Some(Box::new(UsersTable)),
//!             _ => None,
//!         })
//!     }
//! }
//!
//! let mut ctx = PlannerContext::new(Vec::new(), vec!["db".to_string()], "db".to_string())
//!     .unwrap();
//!
//! // Plan a query inside a transaction; walk the root handle.
//! let plan = ctx.plan("SELECT name FROM users", Vec::new(), Arc::new(MyTransaction)).unwrap();
//! let root = plan.root().unwrap();
//! assert_eq!(root.op_type().unwrap(), LogicalOperatorType::LOGICAL_PROJECTION);
//! ```

#![allow(clippy::upper_case_acronyms)]

pub mod catalog_provider;
pub mod duckdb_bridge;
pub mod handle;
mod types;

use std::sync::Arc;

use duckdb_bridge::ffi;
use thiserror::Error;

pub use catalog_provider::{DuckDBTable, DuckDBTransaction};
pub use duckdb_bridge::duckdb_types::LogicalTypeId;
pub use duckdb_bridge::ffi::DuckDBColumn;
pub use handle::{BridgeError, Expr, LogicalOp, Plan};
pub use types::{BoundLogicalType, ExtraTypeInfo, ScalarValue};

/// Top-level error type for the planner.
#[derive(Error, Debug)]
pub enum Error {
    #[error(transparent)]
    DuckDBPlanning(#[from] PlanningError),
    #[error("{0}")]
    UnsupportedPlan(String),
    #[error("Bridge error: {0}")]
    Bridge(String),
    #[error("a parameter of type {0:?} cannot be bound")]
    UnsupportedParameterType(LogicalTypeId),
    #[error("a parameter of type {0:?} is missing its element or field types")]
    IncompleteParameterType(LogicalTypeId),
}

impl From<cxx::Exception> for Error {
    fn from(exception: cxx::Exception) -> Self {
        Error::Bridge(exception.what().to_string())
    }
}

impl From<BridgeError> for Error {
    fn from(error: BridgeError) -> Self {
        Error::Bridge(error.0)
    }
}

/// Error returned by DuckDB when it cannot produce a plan for a query.
#[derive(Debug, Error)]
pub struct PlanningError {
    pub exception_message: String,
    pub position: Option<String>,
}

impl std::fmt::Display for PlanningError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.exception_message)
    }
}

/// Translate the error fields the bridge reports into a typed [`enum@Error`].
fn bridge_error(error: &ffi::BridgeErrorInfo) -> Error {
    let position = error.has_position.then(|| error.position.clone());
    match error.kind.as_str() {
        "duckdb_planning" => Error::DuckDBPlanning(PlanningError {
            exception_message: error.message.clone(),
            position,
        }),
        "unsupported_plan" => Error::UnsupportedPlan(error.message.clone()),
        "bridge_error" => Error::Bridge(error.message.clone()),
        other => Error::Bridge(format!(
            "Unknown bridge error kind `{other}`: {}",
            error.message
        )),
    }
}

/// Owns an in-process DuckDB instance and exposes SQL planning.
pub struct PlannerContext {
    cxx_context: cxx::UniquePtr<ffi::DuckPlannerContext>,
}

impl PlannerContext {
    /// Create a planner context with the given static table-function overloads,
    /// attaching one DuckDB database per name in `database_names` and making
    /// `default_name` the current database. Function invocations and per-query
    /// catalog lookups are routed through the transaction handed to
    /// [`plan`](Self::plan).
    /// Creating the context attaches every datastore as a DuckDB database, and
    /// any of those steps can fail inside DuckDB; the C++ exception comes back
    /// as [`Error::Bridge`].
    pub fn new(
        table_functions: Vec<catalog_provider::TableFunctionDef>,
        database_names: Vec<String>,
        default_name: String,
    ) -> Result<Self, Error> {
        let ctx = Box::new(catalog_provider::CatalogContext::new(
            table_functions,
            database_names,
            default_name,
        ));
        Ok(Self {
            cxx_context: ffi::new_context(ctx)?,
        })
    }

    /// Plan a SQL query inside `transaction`: sends the query to DuckDB and
    /// returns a [`Plan`] handle owning the resolved logical plan. The caller
    /// walks it through [`Plan::root`] and the [`LogicalOp`]/[`Expr`] accessors.
    ///
    /// Every table the query references is bound through `transaction`, so one
    /// plan sees one consistent snapshot of the catalog. The transaction is
    /// published to the bridge only for the duration of this call.
    ///
    /// `parameters` are the values of the statement's `$1..$n` placeholders, in
    /// placeholder order. DuckDB binds each as a constant of the value's type,
    /// so the plan holds no placeholders; a statement whose placeholder count
    /// differs from `parameters.len()` is an error.
    pub fn plan(
        &mut self,
        query: &str,
        parameters: Vec<ScalarValue>,
        transaction: Arc<dyn DuckDBTransaction>,
    ) -> Result<Plan, Error> {
        let mut parameter_values = ffi::new_value_list()?;
        for parameter in parameters {
            ffi::push_value(parameter_values.pin_mut(), parameter.into_duckdb_value()?)?;
        }
        let transaction_ctx = catalog_provider::TransactionContext::new(transaction);
        let result = ffi::extract_plan(
            self.cxx_context.pin_mut(),
            query,
            &parameter_values,
            &transaction_ctx,
        )?;
        if !result.error.kind.is_empty() {
            return Err(bridge_error(&result.error));
        }

        let ffi::ExtractPlanResult {
            plan,
            output_names,
            requires_rebind,
            ..
        } = result;
        Ok(Plan::new(plan, output_names, requires_rebind))
    }

    /// The type DuckDB infers for each of the statement's `$1..$n`
    /// placeholders, in placeholder order, from the context each one appears
    /// in. A placeholder no context gives a type to (e.g. `SELECT $1`) is
    /// `UNKNOWN`.
    pub fn infer_parameter_types(
        &mut self,
        query: &str,
        transaction: Arc<dyn DuckDBTransaction>,
    ) -> Result<Vec<BoundLogicalType>, Error> {
        let transaction_ctx = catalog_provider::TransactionContext::new(transaction);
        let result = ffi::extract_param_types(self.cxx_context.pin_mut(), query, &transaction_ctx)?;
        if !result.error.kind.is_empty() {
            return Err(bridge_error(&result.error));
        }
        let parameter_types = (0..ffi::parameter_type_count(&result.parameter_types)?)
            .map(|i| {
                handle::bound_type_from_logical_type(ffi::parameter_type(
                    &result.parameter_types,
                    i,
                )?)
            })
            .collect::<Result<Vec<_>, BridgeError>>()?;
        Ok(parameter_types)
    }
}
