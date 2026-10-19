//! The per-query binding of one loaded table: the snapshot the query bound
//! plus the filters pushed into it, compiled into the Parquet scan.

use std::sync::{Arc, Mutex};

use arrow_array::{ArrayRef, Scalar};
use dispatch::{DataFlowDispatcher, Projection, RecordBatchOperatorSpec};
use parquet_engine::{
    ParquetTable, PushedPredicate, equality_predicates, materialize, prune_parquet,
    row_group_filter_from, scan_order_from, table_input_with_filter_and_eq_predicates,
};
use planner::catalog::{
    BoundTable, Column, DynamicScanPredicate, Result as CatalogResult, TableReference,
    TableRevision,
};
use planner::expression::Expression;

use crate::table::LoadedTable;

/// One reference to a table in a query. Filter pushdown finishes before the
/// binding is cloned for materialization, and clones share the pruned scan
/// metadata.
#[derive(Clone)]
pub(crate) struct IcebergTableBinding {
    reference: TableReference,
    table: Arc<LoadedTable>,
    /// The filters pushed into this scan. Manifests and data files are pruned
    /// by them, row groups and dictionaries by the comparisons among them.
    filters: Vec<Expression>,
    /// Statistics, scanning, and late materialization reuse this binding's load.
    cached_pruned_parquet: Arc<Mutex<Option<Arc<ParquetTable>>>>,
}

impl std::fmt::Debug for IcebergTableBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IcebergTableBinding")
            .field("reference", &self.reference)
            .field("metadata_location", &self.table.metadata_location)
            .field("filters", &self.filters)
            .finish()
    }
}

impl IcebergTableBinding {
    pub(crate) fn new(reference: TableReference, table: Arc<LoadedTable>) -> Self {
        Self {
            reference,
            table,
            filters: Vec::new(),
            cached_pruned_parquet: Arc::default(),
        }
    }

    /// The row groups a scan reads: those of the files the pushed filters
    /// cannot rule out, less the ones their own statistics rule out. Serialize
    /// preparation across clones and retain only successful loads.
    fn load_pruned_parquet(&self) -> CatalogResult<Arc<ParquetTable>> {
        let mut cached = self.cached_pruned_parquet.lock().unwrap();
        if let Some(table) = cached.as_ref() {
            return Ok(table.clone());
        }
        let parquet = self.table.load_parquet(&self.filters)?;
        let predicates: Vec<PushedPredicate> = self
            .filters
            .iter()
            .flat_map(PushedPredicate::from_filter)
            .collect();
        let table = Arc::new(prune_parquet(&parquet, &predicates));
        *cached = Some(table.clone());
        Ok(table)
    }

    /// Whether the scan this binding stands for reads the whole table.
    fn is_unfiltered(&self) -> bool {
        self.filters.is_empty()
    }
}

impl BoundTable for IcebergTableBinding {
    fn table_reference(&self) -> TableReference {
        self.reference.clone()
    }

    fn table_revision(&self) -> TableRevision {
        self.table.revision()
    }

    fn supports_late_materialization(&self) -> bool {
        true
    }

    fn compile_scan(
        &self,
        dispatcher: &DataFlowDispatcher,
        projection: Projection,
        dynamic_filters: Vec<DynamicScanPredicate>,
        emit_row_group_metadata: bool,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        let predicates: Vec<PushedPredicate> = self
            .filters
            .iter()
            .flat_map(PushedPredicate::from_filter)
            .collect();
        let eq_predicates = equality_predicates(&predicates);
        let scan_order = scan_order_from(&dynamic_filters);
        Ok(table_input_with_filter_and_eq_predicates(
            dispatcher,
            &self.load_pruned_parquet()?,
            projection,
            emit_row_group_metadata,
            row_group_filter_from(dynamic_filters),
            scan_order,
            Arc::new(eq_predicates),
        ))
    }

    fn columns(&self) -> Vec<Column> {
        self.table.columns().to_vec()
    }

    fn nullability(&self) -> Vec<bool> {
        self.table.nullability()
    }

    fn clone_box(&self) -> Box<dyn BoundTable> {
        Box::new(self.clone())
    }

    fn materialize(
        &self,
        input: RecordBatchOperatorSpec,
        projection: Projection,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        Ok(materialize(input, self.load_pruned_parquet()?, projection))
    }

    fn pushdown_filter(&mut self, filter: Expression) -> CatalogResult<bool> {
        self.filters.push(filter);
        // Metadata loaded for fewer filters is not this binding's view.
        self.cached_pruned_parquet = Arc::default();
        // Pruning only excludes work; the SQL filter still evaluates rows.
        Ok(false)
    }

    fn column_min_max(&self, column: usize) -> Option<(Scalar<ArrayRef>, Scalar<ArrayRef>)> {
        if !self.is_unfiltered() {
            return None;
        }
        // Manifest bounds can be loose. Exact extrema come from the prepared
        // Parquet view; any preparation error is reported by the ensuing scan.
        self.load_pruned_parquet().ok()?.column_min_max(column)
    }

    fn row_count(&self) -> Option<i64> {
        self.is_unfiltered()
            .then(|| self.table.row_count())
            .flatten()
    }

    fn estimate_row_count(&self) -> Option<u64> {
        self.table.estimated_row_count().map(|rows| rows as u64)
    }
}
