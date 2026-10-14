//! Operators in a Pivot [`Plan`](crate::Plan).
//!
//! Each variant of [`Operator`] corresponds to one stage of a plan
//! (scan a table, project columns, filter rows, aggregate, sort, top-N,
//! create a table). Operators are produced by convertion from
//! a [`duckdb_planner::handle::Operator`].
//!
//! Each operator *kind* lives in its own submodule co-locating the AST type,
//! its `TryFrom` from the DuckDB operator, its `Display`, and its `compile`
//! impl lowering it into a
//! [`RecordBatchOperatorSpec`](dispatch::RecordBatchOperatorSpec). This module
//! holds the cross-cutting pieces: the [`Operator`] enum that ties the kinds
//! together, the conversion [`enum@Error`], the shared ORDER BY key types, and
//! the dynamic-filter slot helper used by more than one operator.

mod aggregate;
mod compact;
mod copy_arrow;
mod copy_from_stdin;
mod copy_text;
mod create_schema;
mod create_table;
mod create_user;
mod cte;
mod distinct;
mod drop_table;
mod dummy_scan;
mod explain;
mod filter;
mod input;
mod insert;
mod join;
mod limit;
mod materialize;
mod order_by;
mod projection;
mod set_variable;
mod table_function;
mod top_n;
mod transaction;
mod unnest;
mod values;

pub use aggregate::Aggregate;
pub use compact::Compact;
pub use copy_from_stdin::{CopyChunk, CopyFormat, CopyFromStdin, FrameDecoder};
pub use copy_text::TextFormat;
pub use create_schema::CreateSchema;
pub use create_table::CreateTable;
pub use create_user::CreateUser;
pub use cte::{Cte, CteScan};
pub use distinct::Distinct;
pub use drop_table::DropTable;
pub use dummy_scan::DummyScan;
pub use explain::Explain;
pub use filter::Filter;
pub use input::Input;
pub use insert::Insert;
pub use join::{Join, JoinKind};
pub use limit::Limit;
pub use materialize::Materialize;
pub use order_by::{OrderBy, OrderByDirection, OrderByNode};
pub use projection::Projection;
pub use set_variable::SetVariable;
pub use table_function::{TableFunction, TableFunctionScan, TableFunctionSignature};
pub(crate) use table_function::{built_in_table_function, built_in_table_function_defs};
pub use top_n::TopN;
pub use transaction::TransactionStatement;
pub use unnest::{Unnest, UnnestedList};
pub use values::Values;

use crate::compile;
use crate::expression::{self, Expression};
use crate::types::Type;
use std::fmt;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("{0}")]
    Expression(#[from] expression::Error),
    #[error("{0}")]
    Type(#[from] crate::types::Error),
    /// A plan shape pivot doesn't support yet (an unmapped operator, a
    /// non-constant LIMIT, a table function with named parameters, ...).
    #[error("{0}")]
    Unsupported(String),
    /// A statement whose arguments fail validation (e.g. a COPY option with
    /// an out-of-spec value).
    #[error("{0}")]
    InvalidStatement(String),
    /// A subtree's output columns couldn't be typed while shaping the plan
    /// (the join projection-map replay needs each side's width and types).
    #[error("{0}")]
    Typing(#[from] compile::Error),
    /// A DuckDB exception surfaced while reading the plan across the bridge.
    #[error("{0}")]
    Bridge(#[from] duckdb_planner::BridgeError),
}

/// An operator in the query plan.
#[derive(Debug)]
pub enum Operator {
    Input(Input),
    Values(Values),
    Insert(Insert),
    /// A scan over a table-valued function (e.g. `generate_series`).
    TableFunctionScan(TableFunctionScan),
    Projection(Projection),
    OrderBy(OrderBy),
    Aggregate(Aggregate),
    Filter(Filter),
    TopN(TopN),
    Limit(Limit),
    /// Inner hash equi-join: probe (first input) against build (second input).
    Join(Join),
    CreateTable(CreateTable),
    CreateSchema(CreateSchema),
    /// `DROP TABLE <name>` — compiles into a dataflow that stages the drop;
    /// the transaction's commit removes the table.
    DropTable(DropTable),
    /// `CREATE USER <name> [PASSWORD '<password>']` — compiles into a dataflow
    /// that stages the user; the transaction's commit creates it.
    CreateUser(CreateUser),
    DummyScan(DummyScan),
    /// `SET`/`RESET` of a session variable — handled by the server, not compiled.
    SetVariable(SetVariable),
    /// `COMPACT <table> [FINAL]` — handled by the server, not compiled.
    Compact(Compact),
    /// `VACUUM`: handled by the executor without compiling a dataflow.
    Vacuum,
    /// `COPY <table> FROM STDIN` — handled by the server, not compiled.
    CopyFromStdin(CopyFromStdin),
    /// `BEGIN`/`COMMIT`/`ROLLBACK`, answered by the server, not compiled.
    Transaction(TransactionStatement),
    /// Late-materialization fetch (synthesized by the rewrite, see [`Materialize`]).
    Materialize(Materialize),
    /// `EXPLAIN <query>`: renders its child plan as text (see [`Explain`]).
    Explain(Explain),
    /// A CTE: its definition (first input) feeding the query reading it (second).
    Cte(Cte),
    /// One place a CTE's rows are read (see [`CteScan`]).
    CteScan(CteScan),
    /// The distinct values of a tuple of input columns (see [`Distinct`]).
    Distinct(Distinct),
    /// Each input row repeated once per element of a list written out in the
    /// query (see [`Unnest`]).
    Unnest(Unnest),
}

impl Operator {
    /// The pivot [`Type`] of each column this operator emits, in output order,
    /// given its inputs' column types (one `Vec` per child, in child order).
    /// This is what lets a pass over the plan ask what any subtree produces
    /// (e.g. rendering variant output columns as JSON text) without knowing
    /// operator shapes.
    pub fn output_types(&self, inputs: &[Vec<Type>]) -> Result<Vec<Type>, compile::Error> {
        match self {
            // Sources: a scan's outputs are its column expressions. (The extra
            // row-group metadata columns an `emit_row_group_metadata` scan
            // appends are plumbing between that scan and its Materialize, not
            // part of the logical schema.)
            Operator::Input(input) => expression_types(&input.columns),
            Operator::TableFunctionScan(scan) => scan.output_types(),
            // A materialize re-fetches columns like a scan: its outputs are
            // its column expressions.
            Operator::Materialize(materialize) => expression_types(&materialize.columns),
            // A projection reshapes its input into its expressions.
            Operator::Projection(projection) => expression_types(&projection.projections),
            // An aggregate emits its group keys, then one column per aggregate.
            Operator::Aggregate(aggregate) => {
                let mut types = expression_types(&aggregate.groups)?;
                types.extend(expression_types(&aggregate.expressions)?);
                Ok(types)
            }
            // These only reorder or trim rows; the columns pass through.
            Operator::Filter(_) | Operator::OrderBy(_) | Operator::TopN(_) | Operator::Limit(_) => {
                Ok(inputs[0].clone())
            }
            // A VALUES source emits its expression columns; every row shares
            // the same types, so the first row speaks for all.
            Operator::Values(values) => {
                expression_types(values.rows.first().map_or(&[][..], Vec::as_slice))
            }
            // An INSERT emits one row: the inserted-row count.
            Operator::Insert(_) => Ok(vec![Type::Int64]),
            // A join emits its listed probe columns followed by its listed
            // build columns; a mark join appends its boolean marker column.
            Operator::Join(join) => {
                let mut types: Vec<Type> = join
                    .probe_output
                    .iter()
                    .map(|&i| inputs[0][i].clone())
                    .chain(join.build_output.iter().map(|&i| inputs[1][i].clone()))
                    .collect();
                if matches!(join.kind, JoinKind::ProbeMark) {
                    types.push(Type::Boolean);
                }
                Ok(types)
            }
            // A FROM-less SELECT's one-row source emits one placeholder
            // column, as DuckDB's dummy scan declares.
            Operator::DummyScan(_) => Ok(vec![Type::Int32]),
            // EXPLAIN renders its child plan as text, one line per row.
            Operator::Explain(_) => Ok(vec![Type::Utf8]),
            // A CTE emits what the query reading it emits; the definition
            // under its first input is only a source for the scans.
            Operator::Cte(_) => Ok(inputs[1].clone()),
            // A scan of a CTE emits what that CTE's definition produces,
            // recorded when the definition was walked.
            Operator::CteScan(scan) => Ok(scan.types.clone()),
            // A distinct emits its key columns, in key order.
            Operator::Distinct(distinct) => {
                Ok(distinct.keys.iter().map(|(_, ty)| ty.clone()).collect())
            }
            // An unnest passes its input through and appends one element
            // column per list.
            Operator::Unnest(unnest) => {
                let mut types = inputs[0].clone();
                types.extend(unnest.element_types());
                Ok(types)
            }
            // Statements, not queries: no result columns.
            Operator::CreateTable(_)
            | Operator::CreateSchema(_)
            | Operator::DropTable(_)
            | Operator::CreateUser(_)
            | Operator::SetVariable(_)
            | Operator::Compact(_)
            | Operator::Vacuum
            | Operator::CopyFromStdin(_)
            | Operator::Transaction(_) => Ok(Vec::new()),
        }
    }

    /// Whether each column this operator emits can hold SQL NULLs, in output
    /// order: the nullability companion to [`output_types`](Self::output_types),
    /// resolved per expression via [`Expression::nullability`]. The planner uses
    /// it to route between the branch-free and the null-aware execution paths,
    /// so `false` must be sound while `true` merely costs the fast path.
    pub fn output_nullability(&self, inputs: &[Vec<bool>]) -> Vec<bool> {
        match self {
            // A scan's nullability comes from the table binding, which reports
            // whether each column's data can actually hold NULLs.
            Operator::Input(input) => {
                let table = input.table.nullability();
                input
                    .columns
                    .iter()
                    .map(|e| e.nullability(&table))
                    .collect()
            }
            // Table functions do not report nullability; stay conservative.
            Operator::TableFunctionScan(scan) => {
                vec![true; scan.output_types().map_or(0, |t| t.len())]
            }
            Operator::Materialize(materialize) => {
                let table = materialize.table.nullability();
                materialize
                    .columns
                    .iter()
                    .map(|e| e.nullability(&table))
                    .collect()
            }
            Operator::Projection(projection) => projection
                .projections
                .iter()
                .map(|e| e.nullability(&inputs[0]))
                .collect(),
            Operator::Aggregate(aggregate) => aggregate
                .groups
                .iter()
                .chain(aggregate.expressions.iter())
                .map(|e| e.nullability(&inputs[0]))
                .collect(),
            Operator::Filter(_) | Operator::OrderBy(_) | Operator::TopN(_) | Operator::Limit(_) => {
                inputs[0].clone()
            }
            Operator::DummyScan(_) => vec![true],
            Operator::Explain(_) => vec![false],
            // A VALUES row may hold NULL literals; stay conservative.
            Operator::Values(values) => {
                vec![true; values.rows.first().map_or(0, Vec::len)]
            }
            // An INSERT emits one row: the inserted-row count, never NULL.
            Operator::Insert(_) => vec![false],
            // An inner join only emits rows built from both inputs, so each
            // output column keeps its own side's nullability; so does a semi
            // join, whose output is probe columns the build side matched. An
            // outer join also emits its preserved side's unmatched rows,
            // filling the other side's columns with NULL regardless of that
            // input's declared nullability.
            Operator::Join(join) => {
                let probe_nullable = matches!(join.kind, JoinKind::BuildOuter);
                let build_nullable = matches!(join.kind, JoinKind::ProbeOuter);
                let mut nullable: Vec<bool> = join
                    .probe_output
                    .iter()
                    .map(|&i| probe_nullable || inputs[0][i])
                    .chain(
                        join.build_output
                            .iter()
                            .map(|&i| build_nullable || inputs[1][i]),
                    )
                    .collect();
                // A mark join's marker column is NULL where three-valued `IN`
                // leaves a miss unknown.
                if matches!(join.kind, JoinKind::ProbeMark) {
                    nullable.push(true);
                }
                nullable
            }
            Operator::Cte(_) => inputs[1].clone(),
            Operator::CteScan(scan) => scan.nullable.clone(),
            Operator::Distinct(distinct) => distinct
                .keys
                .iter()
                .map(|&(idx, _)| inputs[0][idx])
                .collect(),
            Operator::Unnest(unnest) => {
                let mut nullable = inputs[0].clone();
                nullable.extend(unnest.element_nullability(&inputs[0]));
                nullable
            }
            Operator::CreateTable(_)
            | Operator::CreateSchema(_)
            | Operator::DropTable(_)
            | Operator::CreateUser(_)
            | Operator::SetVariable(_)
            | Operator::Compact(_)
            | Operator::Vacuum
            | Operator::CopyFromStdin(_)
            | Operator::Transaction(_) => Vec::new(),
        }
    }
}

/// The result type of each expression, in order.
fn expression_types(expressions: &[Expression]) -> Result<Vec<Type>, compile::Error> {
    expressions.iter().map(Expression::result_type).collect()
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operator::Input(i) => write!(f, "{i}"),
            Operator::Values(v) => write!(f, "{v}"),
            Operator::Insert(i) => write!(f, "{i}"),
            Operator::TableFunctionScan(t) => write!(f, "{t}"),
            Operator::Projection(p) => write!(f, "{p}"),
            Operator::OrderBy(o) => write!(f, "{o}"),
            Operator::Aggregate(a) => write!(f, "{a}"),
            Operator::Filter(fl) => write!(f, "{fl}"),
            Operator::TopN(t) => write!(f, "{t}"),
            Operator::Limit(l) => write!(f, "{l}"),
            Operator::Join(j) => write!(f, "{j}"),
            Operator::CreateTable(c) => write!(f, "{c}"),
            Operator::CreateSchema(c) => write!(f, "{c}"),
            Operator::DropTable(d) => write!(f, "{d}"),
            Operator::CreateUser(c) => write!(f, "{c}"),
            Operator::DummyScan(d) => write!(f, "{d}"),
            Operator::SetVariable(s) => write!(f, "{s}"),
            Operator::Compact(c) => write!(f, "{c}"),
            Operator::Vacuum => f.write_str("Vacuum"),
            Operator::CopyFromStdin(c) => write!(f, "{c}"),
            Operator::Transaction(t) => write!(f, "{t}"),
            Operator::Materialize(m) => write!(f, "{m}"),
            Operator::Explain(e) => write!(f, "{e}"),
            Operator::Cte(c) => write!(f, "{c}"),
            Operator::CteScan(c) => write!(f, "{c}"),
            Operator::Distinct(d) => write!(f, "{d}"),
            Operator::Unnest(u) => write!(f, "{u}"),
        }
    }
}
