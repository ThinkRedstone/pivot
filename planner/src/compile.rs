//! Compile a Pivot [`Plan`] into an executable
//! [`RecordBatchOperatorSpec`].
//!
//! Compilation is a recursive walk: each [`PlanNode`] compiles its inputs
//! first, then dispatches to the per-operator `compile` impl (e.g. a
//! [`Filter`](crate::operator::Filter) compiles into
//! [`RecordBatchOperatorSpec::filter`]). Expressions compile into **builder
//! closures** ([`ExprFn`]): a `Fn()` returning a per-worker
//! `FnMut(&RecordBatch) -> ExprResult`, matching dispatch's two-level
//! closure pattern so each worker gets its own state without synchronization.
//!
//! # Organization
//!
//! This module holds the cross-cutting compile infrastructure: the compile
//! [`enum@Error`], the [`ExprResult`]/[`ExprFn`] closure types, the
//! `RuntimeFilterSlots` registry, and the recursive [`Plan`]/[`PlanNode`]
//! walk. The per-operator `compile` impls live alongside their AST types in the
//! [`operator`](crate::operator) submodules, and the per-expression `compile`
//! impls producing [`ExprFn`]s live in the [`expression`](crate::expression)
//! submodules, not here.
//!

use crate::catalog::CatalogTransaction;
use crate::expression::Expression;
use crate::types::Type;
use crate::{Plan, PlanNode};
use arrow_array::{ArrayRef, Datum, RecordBatch, Scalar, UInt32Array};
use dispatch::{BoundarySlot, DataFlowDispatcher, KeyBitsetSlot, RecordBatchOperatorSpec};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

/// Resolves the logical slot IDs in a plan to shared runtime cells for one
/// execution.
///
/// Producers and consumers are stored in different parts of the plan and are
/// compiled independently. The planner connects them with a small integer
/// `slot_id`; this registry is their rendezvous point. Whichever end compiles
/// first creates the cell, and the other end receives a clone of the same
/// `Arc`:
///
/// ```text
///     logical plan                         compiled execution
///
///     TopN(slot_id = 4) ─────┐             producer ─────┐
///                             ├─ compile ─►                ├─ Arc<BoundarySlot>
///     Input(slot_id = 4) ────┘             consumer ─────┘
/// ```
///
/// Boundary slots and key-bitset slots have separate ID spaces because they
/// hold different concrete types; boundary slot `0` and key-bitset slot `0`
/// are unrelated cells.
///
/// The registry belongs to compilation, not to [`Plan`]. Compiling a cached
/// plan therefore creates fresh, unarmed slots for every execution instead of
/// retaining a prior query's boundary or build-key bitset.
#[derive(Default)]
pub(crate) struct RuntimeFilterSlots {
    /// Boundary-value slots (Top-N limits, join key bounds), by slot id.
    boundary: HashMap<usize, Arc<BoundarySlot>>,
    /// Key-bitset slots, by slot id — a separate id space
    /// from `boundary`.
    key_bitset: HashMap<usize, Arc<KeyBitsetSlot>>,
}

impl RuntimeFilterSlots {
    /// Return the boundary cell named by `slot_id`, creating it if this is the
    /// first producer or consumer for that ID to compile.
    pub(crate) fn boundary_slot(&mut self, slot_id: usize) -> Arc<BoundarySlot> {
        Arc::clone(
            self.boundary
                .entry(slot_id)
                .or_insert_with(|| Arc::new(BoundarySlot::new())),
        )
    }

    /// Return the key-bitset cell named by `slot_id`, creating it if this is the
    /// first producer or consumer for that ID to compile.
    pub(crate) fn key_bitset_slot(&mut self, slot_id: usize) -> Arc<KeyBitsetSlot> {
        Arc::clone(
            self.key_bitset
                .entry(slot_id)
                .or_insert_with(|| Arc::new(KeyBitsetSlot::new())),
        )
    }
}

#[derive(Debug, Error)]
pub enum Error {
    #[error("Unexpected input Expression {0}")]
    UnexpectedInputExpression(Expression),
    #[error("Unsupported projection expression: {0:?}")]
    UnsupportedProjectionExpression(Expression),
    #[error("Unsupported order by expression: {0:?}")]
    UnsupportedOrderByExpression(Expression),
    #[error("Unsupported top k expression: {0:?}")]
    UnsupportedTopKExpression(Expression),
    #[error("Unsupported aggregate expression: {0:?}")]
    UnsupportedAggregateExpression(Expression),
    #[error("Unsupported aggregate expression amount: {0}")]
    UnsupportedAggregateExpressionAmount(usize),
    #[error("Unsupported expression: {0:?}")]
    UnsupportedExpression(Expression),
    #[error("Unsupported type for group by{}: {data_type:?}", .column.as_deref().map(|c| format!(" of column \"{c}\"")).unwrap_or_default())]
    DataTypeNotSupportedForGroupBy {
        column: Option<String>,
        data_type: Type,
    },
    #[error("Cannot statically determine the result type of expression: {0:?}")]
    IndeterminateResultType(Expression),
    #[error("Unsupported expression for contains: {0:?}")]
    UnsupportedExpressionForContainsNeedle(Expression),
    #[error("Unsupported haystack expression for contains: {0:?}")]
    UnsupportedExpressionForContainsHaystack(Expression),
    #[error("Unsupported pattern expression for prefix: {0:?}")]
    UnsupportedExpressionForPrefixPattern(Expression),
    #[error("Unsupported input expression for LIKE: {0:?}")]
    UnsupportedExpressionForLikeInput(Expression),
    #[error("Invalid LIKE pattern '{pattern}': {source}")]
    InvalidLikePattern {
        pattern: String,
        #[source]
        source: regex::Error,
    },
    #[error("Unsupported haystack expression for prefix: {0:?}")]
    UnsupportedExpressionForPrefixHaystack(Expression),
    #[error("Unsupported pattern expression for suffix: {0:?}")]
    UnsupportedExpressionForSuffixPattern(Expression),
    #[error("Unsupported haystack expression for suffix: {0:?}")]
    UnsupportedExpressionForSuffixHaystack(Expression),
    #[error("Failed to downcast scalar into string: {0:?}")]
    FailedToDowncastScalarIntoString(Scalar<ArrayRef>),
    #[error("Invalid regex pattern '{pattern}': {source}")]
    InvalidRegexPattern {
        pattern: String,
        #[source]
        source: regex::Error,
    },
    #[error("Invalid regexp_jit_replace pattern '{pattern}': {source}")]
    InvalidJitRegexPattern {
        pattern: String,
        #[source]
        source: pcre2::Error,
    },
    #[error("CREATE TABLE does not support OR REPLACE yet")]
    UnsupportedCreateTableOrReplace,
    #[error("CREATE TEMPORARY TABLE is not supported yet")]
    UnsupportedTemporaryCreateTable,
    #[error("CREATE TABLE AS SELECT is not supported yet")]
    UnsupportedCreateTableAs,
    #[error("CREATE TABLE with constraints is not supported yet ({0} constraint(s))")]
    UnsupportedCreateTableConstraints(usize),
    #[error("CREATE TABLE nodes should not have input operators")]
    UnexpectedCreateTableInputs,
    #[error("compiling table scan: {0}")]
    TableScan(#[source] crate::catalog::Error),
    #[error("creating table: {0}")]
    CreateTable(#[source] crate::catalog::Error),
    #[error("DROP TABLE does not support CASCADE: there are no dependent objects to cascade over")]
    UnsupportedDropTableCascade,
    #[error("DROP TABLE does not take inputs")]
    UnexpectedDropTableInputs,
    #[error("dropping table: {0}")]
    DropTable(#[source] crate::catalog::Error),
    #[error("CREATE OR REPLACE SCHEMA is not supported")]
    UnsupportedCreateSchemaOrReplace,
    #[error("CREATE SCHEMA does not take inputs")]
    UnexpectedCreateSchemaInputs,
    #[error("create schema: {0}")]
    CreateSchema(#[source] crate::catalog::Error),
    #[error("inserting rows: {0}")]
    Insert(#[source] crate::catalog::Error),
    #[error("SET/RESET is a session command, not a compilable query")]
    SetVariableNotCompilable,
    #[error("COMPACT runs as its own statement and cannot be part of a query")]
    CompactNotCompilable,
    #[error("VACUUM runs as its own statement and cannot be part of a query")]
    VacuumNotCompilable,
    #[error(
        "COPY FROM STDIN compiles to no dataflow; the server drives the ingest from the protocol"
    )]
    CopyFromStdinNotCompilable,
    #[error("BEGIN/COMMIT/ROLLBACK is a session command, not a compilable query")]
    TransactionNotCompilable,
    #[error("CREATE USER does not take inputs")]
    UnexpectedCreateUserInputs,
    #[error("create user: {0}")]
    CreateUser(#[source] crate::catalog::Error),
    #[error("Unsupported table function: {0}")]
    UnsupportedTableFunction(String),
    #[error("Invalid argument to table function {function}: {message}")]
    InvalidTableFunctionArgument { function: String, message: String },
}

impl Plan {
    /// Lower this plan into an executable
    /// [`dispatch::RecordBatchOperatorSpec`] on the given dispatcher. Base-table
    /// scans and table functions use self-contained bindings captured while
    /// planning; DDL resolves through `transaction`.
    pub fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
        transaction: &dyn CatalogTransaction,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        // DDL resolves through `transaction`. Scans carry their planning
        // snapshot. A cached plan reaches this compile only after the server
        // verifies that every recorded table revision matches this transaction.
        let mut slots = RuntimeFilterSlots::default();
        let compiled = self.root.compile(dispatcher, transaction, &mut slots)?;
        Ok(stamp_output_names(compiled, &self.output_names))
    }
}

/// Stamp DuckDB's client-facing result column names onto the compiled plan's
/// output. Naming happens once here, at the root, rather than inside each
/// operator: the operators are free to use whatever intermediate field names
/// are convenient, and the query's actual output schema is fixed up last.
///
/// The names are DuckDB's select-list order, which the output columns follow.
/// Only the leading `names.len()` fields are renamed, so a wider output (an
/// operator that appends trailing bookkeeping columns) keeps its extra fields;
/// a narrower one than DuckDB reported is left untouched rather than risk
/// mislabeling.
fn stamp_output_names(spec: RecordBatchOperatorSpec, names: &[String]) -> RecordBatchOperatorSpec {
    if names.is_empty() {
        return spec;
    }
    let names = Arc::new(names.to_vec());
    spec.project(move || {
        let names = names.clone();
        move |batch: RecordBatch| {
            let schema = batch.schema();
            let fields = schema.fields();
            // Fewer columns than DuckDB reported names (e.g. EXPLAIN emits one
            // column where DuckDB resolves two): leave the batch untouched.
            if fields.len() < names.len() {
                return Ok(batch);
            }
            // Already correctly named (a plain `SELECT a, b` or `SELECT *`):
            // skip the rebuild and pass the batch through zero-copy.
            if names
                .iter()
                .zip(fields.iter())
                .all(|(name, field)| field.name() == name)
            {
                return Ok(batch);
            }
            // `with_name` preserves each field's data type, nullability, and
            // metadata (e.g. an extension-type annotation); only the leading
            // `names.len()` fields are renamed, so trailing fields are reused
            // by their `Arc` rather than rebuilt.
            let renamed: Vec<arrow_schema::FieldRef> = fields
                .iter()
                .enumerate()
                .map(|(i, field)| match names.get(i) {
                    Some(name) => Arc::new(field.as_ref().clone().with_name(name.clone())),
                    None => field.clone(),
                })
                .collect();
            let new_schema = Arc::new(arrow_schema::Schema::new_with_metadata(
                renamed,
                schema.metadata().clone(),
            ));
            // We own `batch`, so move its column arrays out (there is at least
            // one, since `fields.len() >= names.len() >= 1`) instead of cloning.
            let (_, columns, _) = batch.into_parts();
            Ok(RecordBatch::try_new(new_schema, columns)?)
        }
    })
}

impl PlanNode {
    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
        transaction: &dyn CatalogTransaction,
        slots: &mut RuntimeFilterSlots,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        // Peephole: an unfiltered global MIN/MAX or COUNT(*) over a bare scan is
        // fully determined by table metadata (e.g. parquet row-group
        // statistics). It must run before the child scan is compiled, since
        // succeeding means no scan happens at all. `try_compile_from_stats`
        // checks the rest of the shape (single bare-scan child, no predicates).
        if let crate::Operator::Aggregate(agg) = &self.operator
            && let Some(spec) = agg.try_compile_from_stats(&self.inputs, dispatcher)?
        {
            return Ok(spec);
        }

        // EXPLAIN renders its (already-optimized) child plan as text and emits
        // that. The child is formatted here, not compiled, so the explained
        // query never runs. This must happen before the input-compile loop below.
        if let crate::Operator::Explain(explain) = &self.operator {
            let plan_text = self
                .inputs
                .first()
                .map(PlanNode::to_string)
                .unwrap_or_default();
            return explain.compile(dispatcher, plan_text);
        }

        let mut inputs = Vec::with_capacity(self.inputs.len());
        for input in &self.inputs {
            inputs.push(input.compile(dispatcher, transaction, slots)?);
        }

        match &self.operator {
            crate::Operator::Input(o) => o.compile(dispatcher, slots),
            crate::Operator::Values(o) => o.compile(dispatcher),
            crate::Operator::TableFunctionScan(o) => o.compile(dispatcher, slots),
            crate::Operator::Projection(o) => o.compile(inputs.remove(0)),
            crate::Operator::Filter(o) => o.compile(inputs.remove(0)),
            crate::Operator::Aggregate(o) => {
                o.compile(inputs.remove(0), self.inputs[0].output_nullability())
            }
            crate::Operator::OrderBy(o) => {
                let normalize_variants = self.inputs[0]
                    .output_types()?
                    .iter()
                    .any(|ty| matches!(ty, crate::types::Type::Variant));
                o.compile(inputs.remove(0), normalize_variants)
            }
            crate::Operator::TopN(o) => o.compile(inputs.remove(0), slots),
            crate::Operator::Limit(o) => o.compile(inputs.remove(0)),
            crate::Operator::Join(o) => {
                let normalize_build_variants = self.inputs[1]
                    .output_types()?
                    .iter()
                    .any(|ty| matches!(ty, crate::types::Type::Variant));
                let probe = inputs.remove(0);
                let build = inputs.remove(0);
                o.compile(probe, build, normalize_build_variants, slots)
            }
            crate::Operator::Cte(o) => {
                let definition = inputs.remove(0);
                let body = inputs.remove(0);
                o.compile(definition, body)
            }
            crate::Operator::CteScan(o) => o.compile(dispatcher),
            crate::Operator::Distinct(o) => {
                o.compile(inputs.remove(0), self.inputs[0].output_nullability())
            }
            crate::Operator::Materialize(o) => o.compile(inputs.remove(0)),
            crate::Operator::CreateTable(o) => {
                if !inputs.is_empty() {
                    return Err(Error::UnexpectedCreateTableInputs);
                }
                o.compile(dispatcher, transaction)
            }
            crate::Operator::CreateSchema(o) => {
                if !inputs.is_empty() {
                    return Err(Error::UnexpectedCreateSchemaInputs);
                }
                o.compile(dispatcher, transaction)
            }
            crate::Operator::DropTable(o) => {
                if !inputs.is_empty() {
                    return Err(Error::UnexpectedDropTableInputs);
                }
                o.compile(dispatcher, transaction)
            }
            crate::Operator::Insert(o) => o.compile(inputs.remove(0), dispatcher),
            crate::Operator::DummyScan(o) => o.compile(dispatcher),
            // EXPLAIN is handled above, before inputs are compiled.
            crate::Operator::Explain(_) => unreachable!("Explain is compiled before its inputs"),
            // SET/RESET is intercepted by the server after planning (it toggles
            // session state, not data), so it should never reach compilation.
            crate::Operator::SetVariable(_) => Err(Error::SetVariableNotCompilable),
            // So is COMPACT: its sweeps drive dataflows of their own, so the
            // server runs them on the coordinator, never inside a dataflow.
            crate::Operator::Compact(_) => Err(Error::CompactNotCompilable),
            crate::Operator::Vacuum => Err(Error::VacuumNotCompilable),
            crate::Operator::CopyFromStdin(_) => Err(Error::CopyFromStdinNotCompilable),
            // And BEGIN/COMMIT/ROLLBACK: the server answers them directly
            // without compiling anything.
            crate::Operator::Transaction(_) => Err(Error::TransactionNotCompilable),
            crate::Operator::CreateUser(o) => {
                if !inputs.is_empty() {
                    return Err(Error::UnexpectedCreateUserInputs);
                }
                o.compile(dispatcher, transaction)
            }
        }
    }
}

/// What an evaluated expression produces for one input batch: either a
/// per-row [`Array`](ExprResult::Array) (column refs, comparisons, scalar
/// functions) or a single [`Scalar`](ExprResult::Scalar) (constants). Both
/// variants implement arrow's [`Datum`] so consumers can pass them straight
/// into kernels.
pub enum ExprResult {
    Array(ArrayRef),
    Scalar(Scalar<ArrayRef>),
}

impl ExprResult {
    pub fn as_datum(&self) -> &dyn Datum {
        match self {
            ExprResult::Array(a) => a,
            ExprResult::Scalar(s) => s,
        }
    }

    /// The result as a full column of `num_rows` rows: an [`Array`](Self::Array)
    /// passes through; a [`Scalar`](Self::Scalar) (e.g. `SELECT 1`) is broadcast
    /// to that length so it can sit beside the per-row columns.
    pub fn into_array(self, num_rows: usize) -> ArrayRef {
        match self {
            ExprResult::Array(a) => a,
            ExprResult::Scalar(s) => {
                let inner = s.into_inner();
                // A `Scalar` already holds a length-1 array; when a single row is
                // all that's asked for, hand it back directly rather than copying
                // it through `take` (e.g. one `VALUES` row).
                if inner.len() == num_rows {
                    return inner;
                }
                let indices = UInt32Array::from(vec![0u32; num_rows]);
                arrow::compute::take(&inner, &indices, None).unwrap()
            }
        }
    }
}

/// A per-batch evaluator: takes a [`RecordBatch`] and produces an
/// [`ExprResult`]. `FnMut` so the closure can hold mutable per-worker state
/// (e.g. a [`dispatch::Contains`] cache).
pub type ExprEvalFn = Box<dyn FnMut(&RecordBatch) -> ExprResult + Send>;

/// Builder for [`ExprEvalFn`]: called once per worker thread to produce that
/// worker's private evaluator. Mirrors the two-level closure pattern used by
/// dispatch's [`filter`](dispatch::RecordBatchOperatorSpec::filter) /
/// [`project`](dispatch::RecordBatchOperatorSpec::project) APIs.
pub type ExprFn = Box<dyn Fn() -> ExprEvalFn + Send + Sync>;

/// Wraps a stateless expression closure into the builder pattern (closure returning closure).
pub(crate) fn stateless_expr<F>(f: F) -> ExprFn
where
    F: Fn(&RecordBatch) -> ExprResult + Send + Sync + Clone + 'static,
{
    Box::new(move || {
        let f = f.clone();
        Box::new(move |batch: &RecordBatch| f(batch)) as ExprEvalFn
    })
}
