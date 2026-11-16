//! [`DummyScan`] — the single-row source under a `FROM`-less `SELECT`.
//!
//! The operator (DuckDB's single-row input under e.g. `SELECT 1`,
//! `SELECT drop_cache()`) compiles to the [`Nullary`] source below, which emits
//! exactly one row on a single worker so the projection above it evaluates
//! its expressions once and produces one output row.
//!
//! The row has one placeholder column, a NULL integer, because DuckDB's dummy
//! scan declares one. An operator above that passes its input's columns
//! through (an unnest) is bound against that layout, so the columns it adds
//! start at position 1.

use crate::compile::Error;
use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use dispatch::{Built, BuiltNullary};
use dispatch::{
    DataFlowDispatcher, Nullary, NullaryFactory, NullaryResult, RecordBatchOperatorSpec, Sender,
    WorkStatus,
};
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// The single-row source under a `FROM`-less `SELECT`.
#[derive(Debug)]
pub struct DummyScan;

impl fmt::Display for DummyScan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DummyScan")
    }
}

impl DummyScan {
    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        // One nullary per worker sharing a flag, so exactly one emits the single
        // dummy row the parent projection runs over.
        let emitted = Arc::new(AtomicBool::new(false));
        Ok(RecordBatchOperatorSpec::from_nullary(
            dispatcher,
            (0..dispatcher.worker_count()).map(|_| DummyScanNullaryFactory::new(emitted.clone())),
        ))
    }
}

struct DummyScanNullaryFactory {
    /// Shared across every worker's nullary so exactly one emits the row.
    emitted: Arc<AtomicBool>,
}

impl DummyScanNullaryFactory {
    fn new(emitted: Arc<AtomicBool>) -> Self {
        Self { emitted }
    }
}

impl NullaryFactory<RecordBatch> for DummyScanNullaryFactory {
    type Nullary = DummyScanDispatchOperator;

    fn build_nullary(self) -> BuiltNullary<Self::Nullary> {
        Built::new(DummyScanDispatchOperator {
            emitted: self.emitted,
        })
    }
}

struct DummyScanDispatchOperator {
    emitted: Arc<AtomicBool>,
}

impl Nullary<RecordBatch> for DummyScanDispatchOperator {
    fn run(
        &mut self,
        sender: &mut dyn Sender<RecordBatch>,
        _io: &mut dispatch::OperatorIO,
    ) -> NullaryResult<WorkStatus> {
        if !self.emitted.swap(true, Ordering::Relaxed) {
            let placeholder_field = Field::new("placeholder", DataType::Int32, true);
            let batch = RecordBatch::try_new(
                Arc::new(Schema::new(vec![placeholder_field])),
                vec![Arc::new(Int32Array::new_null(1))],
            )
            .expect("a single NULL integer row is always well-formed");
            sender.send(batch)?;
            Ok(WorkStatus::Ran)
        } else {
            Ok(WorkStatus::Pending)
        }
    }

    fn finish(&mut self, _sender: &mut dyn Sender<RecordBatch>) -> NullaryResult<bool> {
        Ok(self.emitted.load(Ordering::Relaxed))
    }
}
