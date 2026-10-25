//! The per-query **binding**: the independently-mutable [`TableBinding`] every
//! transaction resolve derives from its snapshot, so a query's filter pushdown
//! prunes its own view without affecting anyone else.

use std::sync::Arc;

use crate::manifest::{ColumnStatFilter, PartitionEqFilter};
use arrow_array::{ArrayRef, Scalar};
use crossbeam_deque::Injector;
use dispatch::{DataFlowDispatcher, Projection, RecordBatchOperatorSpec};
use parquet_engine::{
    ParquetTable, PushedPredicate, equality_predicates, materialize, prune_parquet,
    row_group_filter_from, scan_order_from, table_input_with_filter_and_eq_predicates,
};
use planner::catalog::{
    BoundTable, Column, DynamicScanPredicate, Error as CatalogError, Result as CatalogResult,
    TableReference, TableRevision,
};
use planner::expression::{CompareType, Expression};

use super::insert_sink::build_insert_spec;
use super::{CatalogTable, CommitAction, PendingInsert};

/// A catalog table resolved from one transaction: it captures the snapshot's own
/// copy of the [`CatalogTable`] (schema, files, and their row groups), this
/// query's pushed-down predicates, and the transaction's uploaded-files injector
/// for the write path. Because the frozen snapshot copy travels with the binding,
/// every compile-time method resolves its file set with no transaction handle.
/// Cloned per-binding so each query accumulates its own predicates.
#[derive(Clone)]
pub struct TableBinding {
    /// The qualified name this binding was resolved under, as the catalog routed
    /// it. Recorded from the bind call rather than read back off `table`, so a
    /// later rename cannot drift it away from the name the query bound.
    reference: TableReference,
    /// The snapshot's frozen copy of this table: its schema and the row groups of
    /// every committed file at the version the transaction opened. Reads build
    /// their scan view straight from it; a write resolves the live table by its
    /// durable id ([`CatalogTable::id`]) at commit, so it survives a concurrent
    /// rename.
    table: CatalogTable,
    pub columns: Vec<Column>,
    /// Per-column footer-derived nullability, in `columns` order (see
    /// [`CatalogTable::nullability`](super::table::CatalogTable)).
    nullability: Vec<bool>,
    /// Single-column predicates pushed down for this binding (recorded here
    /// because the `BoundTable` trait gives no channel from `pushdown_filter` to
    /// `compile`); applied as a filter when the scan is compiled.
    predicates: Vec<PushedPredicate>,
    /// The transaction's staging queue, shared (`Arc`) with the
    /// [`PivotlakeTransaction`](super::PivotlakeTransaction) that produced this
    /// binding, so the append this binding's
    /// [`compile_insert`](BoundTable::compile_insert) stages is applied by that
    /// transaction's commit.
    commit_actions: Arc<Injector<Box<dyn CommitAction>>>,
}

impl std::fmt::Debug for TableBinding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TableBinding")
            .field("reference", &self.reference)
            .field("id", &self.table.id())
            .field("predicates", &self.predicates)
            .finish_non_exhaustive()
    }
}

impl TableBinding {
    /// A binding over the snapshot's `table` copy, sharing the transaction's
    /// staging queue, with no predicates pushed yet.
    pub(super) fn new(
        reference: TableReference,
        table: CatalogTable,
        commit_actions: Arc<Injector<Box<dyn CommitAction>>>,
    ) -> Self {
        let columns = table.columns();
        let nullability = table.nullability();
        Self {
            reference,
            table,
            columns,
            nullability,
            predicates: Vec::new(),
            commit_actions,
        }
    }

    /// This table's row groups in the captured snapshot, partition-pruned by the
    /// pushed equality predicates. Pure in-memory; the snapshot copy is
    /// immutable, so a scan and its late materialize (same filters) build
    /// identical views addressing the same global row-group indices.
    fn resolve_files(&self) -> CatalogResult<Arc<ParquetTable>> {
        let partition_filters: Vec<PartitionEqFilter> =
            self.partition_filter_candidates().collect();
        let stat_filters: Vec<ColumnStatFilter> = self.stat_filter_candidates().collect();
        self.table
            .build_scan_view(&partition_filters, &stat_filters)
            .map_err(|e| CatalogError::Other(Box::new(e)))
    }

    /// This binding's pushed equality predicates as partition-filter candidates:
    /// the column name and the typed scalar. The catalog intersects these with
    /// the table's partition columns, so yielding every equality predicate (not
    /// just ones on partition columns, which the binding can't tell apart) is fine;
    /// a non-partition column prunes no files.
    fn partition_filter_candidates(&self) -> impl Iterator<Item = PartitionEqFilter> + '_ {
        self.predicates
            .iter()
            .filter(|p| matches!(p.compare_type, CompareType::Equal))
            .filter_map(|p| {
                Some(PartitionEqFilter {
                    column: self.columns.get(p.column_idx)?.name.clone(),
                    value: p.value.clone(),
                })
            })
    }

    /// This binding's pushed predicates as file-level stat-filter candidates: a
    /// plain top-level column comparison (no variant path) paired with its typed
    /// constant. Variant paths are excluded — their stats live in a shredded leaf,
    /// pruned per row group, not in the file's column stats. A column the file's
    /// stats don't bound simply prunes no files.
    fn stat_filter_candidates(&self) -> impl Iterator<Item = ColumnStatFilter> + '_ {
        self.predicates
            .iter()
            .filter(|p| p.path.is_empty())
            .filter_map(|p| {
                Some(ColumnStatFilter {
                    column: self.columns.get(p.column_idx)?.name.clone(),
                    compare_type: p.compare_type,
                    value: p.value.clone(),
                })
            })
    }
}

impl BoundTable for TableBinding {
    fn table_reference(&self) -> TableReference {
        self.reference.clone()
    }

    fn table_revision(&self) -> TableRevision {
        TableRevision {
            identity: self.table.id().to_string(),
            version: self.table.version().to_string(),
        }
    }

    fn supports_late_materialization(&self) -> bool {
        true
    }

    /// Each row group resolves a pushed path against its own shredding layout,
    /// so the scan reads only the leaves the path needs.
    fn applies_variant_extracts(&self) -> bool {
        true
    }

    fn compile_scan(
        &self,
        dispatcher: &DataFlowDispatcher,
        projection: Projection,
        dynamic_filters: Vec<DynamicScanPredicate>,
        emit_row_group_metadata: bool,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        // The captured snapshot copy hands back the file set this query was
        // bound against, partition-pruned. All in-memory: the background
        // refresh already materialized every footer.
        let current = self.resolve_files()?;

        // A variant path predicate reaches its shredded typed leaf only in the
        // files that shred it, so each row group resolves the path itself.
        let eq_predicates = equality_predicates(&self.predicates);

        // Prune the row groups by the pushed-down predicates' stats.
        let parquet = Arc::new(self.pruned_parquet(&current));
        // Order the scan by the Top-N's key so its boundary tightens after the
        // first row group and the rest get pruned, instead of racing file order.
        let scan_order = scan_order_from(&dynamic_filters);
        Ok(table_input_with_filter_and_eq_predicates(
            dispatcher,
            &parquet,
            projection,
            emit_row_group_metadata,
            row_group_filter_from(dynamic_filters),
            scan_order,
            Arc::new(eq_predicates),
        ))
    }

    fn compile_insert(
        &self,
        input: RecordBatchOperatorSpec,
        _dispatcher: &DataFlowDispatcher,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        // The captured snapshot copy stamps the durable schema onto every
        // written file. The files land on this statement's own queue as the
        // dataflow runs; the append staged here hands that queue to the
        // transaction's commit, which resolves the live table by its durable id
        // (surviving a concurrent rename).
        let uploaded_files = Arc::new(Injector::new());
        let spec = build_insert_spec(&self.table, uploaded_files.clone(), input)?;
        self.commit_actions.push(Box::new(PendingInsert {
            table_id: self.table.id(),
            uploaded_files,
        }));
        Ok(spec)
    }

    fn columns(&self) -> Vec<Column> {
        self.columns.clone()
    }

    fn nullability(&self) -> Vec<bool> {
        self.nullability.clone()
    }

    fn clone_box(&self) -> Box<dyn BoundTable> {
        Box::new(self.clone())
    }

    fn materialize(
        &self,
        input: RecordBatchOperatorSpec,
        projection: Projection,
    ) -> CatalogResult<RecordBatchOperatorSpec> {
        // A row reference is a position in the *scanning* view's flat row-group
        // list, so this has to build the identical view: same captured
        // snapshot, same partition pruning, and the same stats pruning
        // [`compile_scan`](BoundTable::compile_scan) applies. Re-reading from a
        // view that kept even one row group the scan dropped shifts every later
        // index and silently returns another row group's rows.
        let current = self.resolve_files()?;
        Ok(materialize(
            input,
            Arc::new(self.pruned_parquet(&current)),
            projection,
        ))
    }

    fn pushdown_filter(&mut self, filter: Expression) -> CatalogResult<bool> {
        // Just record it. The actual pruning (min/max row-group elimination and
        // equality/dictionary pruning) happens in `compile`, once the row-group
        // metadata exists. The upstream `Filter` is kept (we return `Ok(false)`),
        // so this is purely an optimization and never affects correctness.
        self.predicates
            .extend(PushedPredicate::from_filter(&filter));

        Ok(false)
    }

    fn column_min_max(&self, column: usize) -> Option<(Scalar<ArrayRef>, Scalar<ArrayRef>)> {
        // Only sound for the whole, unfiltered table: a pushed-down predicate
        // means the scan this binding stands for excludes rows.
        if !self.predicates.is_empty() {
            return None;
        }
        // Read the captured snapshot files, the same view a scan would see; an
        // unresolvable view means "scan".
        self.resolve_files().ok()?.column_min_max(column)
    }

    fn row_count(&self) -> Option<i64> {
        // Only sound for the whole, unfiltered table: a pushed-down predicate
        // means the scan this binding stands for excludes rows.
        if !self.predicates.is_empty() {
            return None;
        }
        // The parquet footer carries each row group's exact row count, so the
        // table's count is their sum, with no data pages read.
        let parquet = self.resolve_files().ok()?;
        Some(parquet.total_rows())
    }

    fn estimate_row_count(&self) -> Option<u64> {
        // A planning estimate wants the base table's full size: pushed
        // predicates and partition pruning deliberately don't apply, since the
        // cost model accounts for filter selectivity itself. Build the whole
        // (unfiltered) view over the captured snapshot and sum the footers'
        // row-group counts.
        let parquet = self.table.build_scan_view(&[], &[]).ok()?;
        Some(
            parquet
                .row_groups()
                .iter()
                .map(|rg| rg.num_rows as u64)
                .sum(),
        )
    }
}

impl TableBinding {
    /// Clone `parquet`'s row groups and keep only those that survive this
    /// binding's pushed-down predicates — i.e. what [`BoundTable::compile_scan`] actually
    /// scans over the table's current files. A min/max stat that proves no row in
    /// a group can match drops it; a stats-comparison error means "can't prune"
    /// (kept) — never wrong, just unoptimized. No footer I/O. Exposed so pruning
    /// can be asserted directly.
    pub fn pruned_parquet(&self, parquet: &ParquetTable) -> ParquetTable {
        prune_parquet(parquet, &self.predicates)
    }
}
