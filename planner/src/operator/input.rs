//! [`Input`] — scans a [`BoundTable`] from the catalog.

use crate::catalog::{BoundTable, DynamicScanPredicate};
use crate::compile::{Error, RuntimeFilterSlots};
use crate::dynamic_filter::{DynamicFilter, JoinFilterScanInfo, KeyBitsetFilter};
use crate::expression::{Expression, Function, Ref, VariantGet};
use crate::operator::Projection;
use crate::types::{Type, physical_arrow_type};
use dispatch::{
    DataFlowDispatcher, Projection as DispatchProjection, RecordBatchOperatorSpec, VariantExtract,
};
use std::fmt;

/// Scans a [`BoundTable`] from the catalog. `columns` lists the requested output
/// columns (each as a [`Ref`] into the table schema)
/// and `filters` are predicates pushed down into the scan.
#[derive(Debug)]
pub struct Input {
    pub table: Box<dyn BoundTable>,
    pub columns: Vec<Expression>,
    /// Runtime-populated predicates the scan reads from shared slots — installed
    /// by a Top-N or a hash join elsewhere in the plan. Each prunes row groups
    /// against the producer's live boundary value.
    pub dynamic_filters: Vec<DynamicFilter>,
    /// Key-bitset filters installed by the build sides of the hash joins above this input. This
    /// allows early filtering of probe rows
    pub key_bitset_filters: Vec<KeyBitsetFilter>,
    /// The DuckDB binding table index of the get this scan was built from,
    /// which plan-level references (e.g. a join's filter-pushdown targets)
    /// name the scan by. `None` for a scan that has no DuckDB origin (built
    /// directly rather than from a plan walk).
    pub duckdb_table_binding_index: Option<usize>,
    /// Join-filter-pushdown wiring for this scan, when a join above narrows
    /// it; the join's builder consumes this to append consumer entries to
    /// `dynamic_filters`. `None` when no join pushes filters here.
    pub join_filter_info: Option<JoinFilterScanInfo>,
    /// When `true`, the scan tags each emitted row with its global row-group ID
    /// and per-row index (extra metadata columns appended after the data
    /// columns). Set on the narrow scan of a late-materialized query so a
    /// downstream [`Materialize`](crate::operator::Materialize) can fetch the
    /// remaining columns for only the surviving rows. Always `false` for
    /// ordinary scans.
    pub emit_row_group_metadata: bool,
}

impl fmt::Display for Input {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cols = describe_scan_columns(self.table.as_ref(), &self.columns);
        write!(f, "Input([{cols}]")?;
        if !self.dynamic_filters.is_empty() {
            let filters = self
                .dynamic_filters
                .iter()
                .map(|filter| filter.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            write!(f, ", dynamic: [{filters}]")?;
        }
        if !self.key_bitset_filters.is_empty() {
            let filters = self
                .key_bitset_filters
                .iter()
                .map(|filter| format!("#{} in build keys", filter.column_idx))
                .collect::<Vec<_>>()
                .join(", ");
            write!(f, ", key bitsets: [{filters}]")?;
        }
        write!(f, ")")
    }
}

impl Input {
    pub(crate) fn compile(
        &self,
        dispatcher: &DataFlowDispatcher,
        slots: &mut RuntimeFilterSlots,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let (projection, extract_projection) =
            plan_scan_projection(&self.columns, self.table.as_ref())?;
        let dynamic_filters = build_dynamic_scan_predicates(&self.dynamic_filters, slots);
        let scan = self
            .table
            .compile_scan(
                dispatcher,
                projection,
                dynamic_filters,
                self.emit_row_group_metadata,
            )
            .map_err(Error::TableScan)?;
        let scan = self.attach_key_bitset_filters(scan, slots);
        match extract_projection {
            Some(extract_projection) => extract_projection.compile(scan),
            None => Ok(scan),
        }
    }

    /// Chain each key-bitset filter directly above the scan: a row whose key
    /// the sealed build set does not hold dies here, before any other operator
    /// sees it. Until the producing join's build seals, the slot is unarmed
    /// and every row passes, so the filter is never a correctness event.
    fn attach_key_bitset_filters(
        &self,
        mut scan: RecordBatchOperatorSpec,
        slots: &mut RuntimeFilterSlots,
    ) -> RecordBatchOperatorSpec {
        for filter in &self.key_bitset_filters {
            // The filter reads the scan's output positionally; find where the
            // storage column landed, counting only entries that emit an
            // output (the COUNT(*) sentinel refs emit none). A plan naming a
            // column the scan does not emit as a plain read has nothing to
            // test against.
            let mut output_position = 0;
            let mut position = None;
            for column in &self.columns {
                match column {
                    Expression::Ref(r) if r.column_idx == usize::MAX => {}
                    Expression::Ref(r) if r.column_idx == filter.column_idx => {
                        position = Some(output_position);
                        break;
                    }
                    _ => output_position += 1,
                }
            }
            let Some(position) = position else {
                continue;
            };
            let slot = slots.key_bitset_slot(filter.slot_id);
            scan = scan.filter(move || {
                let slot = std::sync::Arc::clone(&slot);
                move |batch: &arrow_array::RecordBatch,
                      _slab: &mut dispatch::memory::SlabAllocator,
                      scratch: &mut dispatch::SelectionScratch| {
                    match slot.get() {
                        None => dispatch::RowSelection::All,
                        Some(bitset) => {
                            scratch.positions.clear();
                            bitset.select(batch.column(position), &mut scratch.positions);
                            if scratch.positions.len() == batch.num_rows() {
                                dispatch::RowSelection::All
                            } else {
                                dispatch::RowSelection::Indices
                            }
                        }
                    }
                }
            });
        }
        scan
    }
}

/// Turns a table read's output expressions into the [`DispatchProjection`] to
/// hand the table and, for a table that does not apply variant extracts
/// itself, the projection to run above the emitted columns. Shared by
/// [`Input`] (the scan) and [`Materialize`](super::Materialize) (the
/// late-materialization fetch), which read the same column expression shape.
///
/// Each output column is either a plain column read (`Ref`) or a pushed
/// variant field extract (`VariantGet` over a `Ref`), where DuckDB's
/// projection-pushdown pushed the path into the read so it emits only the
/// referenced leaf.
pub(crate) fn plan_scan_projection(
    columns: &[Expression],
    table: &dyn BoundTable,
) -> Result<(DispatchProjection, Option<Projection>), Error> {
    // `column_indices` and `extracts` stay parallel: the extract at each
    // position, or `None` for a plain read.
    let mut column_indices: Vec<usize> = Vec::with_capacity(columns.len());
    let mut extracts: Vec<Option<VariantExtract>> = Vec::with_capacity(columns.len());
    // For a table that does not apply extracts itself: the type the read
    // emits at each position, and the extraction to run over it. Kept in
    // step with `column_indices`.
    let mut above_scan: Vec<(Type, Option<&VariantGet>)> = Vec::with_capacity(columns.len());
    for expr in columns {
        match expr {
            // DuckDB emits column_idx == usize::MAX as a sentinel for "no
            // column needed" (e.g. COUNT(*) scans). Skip these.
            Expression::Ref(r) if r.column_idx == usize::MAX => {}
            Expression::Ref(r) => {
                column_indices.push(r.column_idx);
                extracts.push(None);
                above_scan.push((r.return_type.clone(), None));
            }
            Expression::Function(Function::VariantGet(vg)) => {
                let Expression::Ref(r) = vg.input.as_ref() else {
                    return Err(Error::UnexpectedInputExpression(expr.clone()));
                };
                // A cast pushes a typed scalar read; a bare extract (no
                // cast) pushes a sub-variant read. The read resolves either
                // against each file's shredding.
                column_indices.push(r.column_idx);
                extracts.push(Some(VariantExtract {
                    path: vg.path.clone(),
                    as_type: vg.as_type.as_ref().map(physical_arrow_type),
                }));
                above_scan.push((r.return_type.clone(), Some(vg)));
            }
            _ => return Err(Error::UnexpectedInputExpression(expr.clone())),
        }
    }

    let pushed = !extracts.iter().all(Option::is_none);
    // A table that does not resolve paths itself reads the whole variant
    // column, and the extraction runs as an ordinary projection above the
    // read. The answer is the same either way; only the bytes read differ.
    let extract_above_scan = pushed && !table.applies_variant_extracts();
    let projection = if pushed && !extract_above_scan {
        DispatchProjection::columns_with_extracts(column_indices, extracts)
    } else {
        DispatchProjection::columns(column_indices)
    };
    if !extract_above_scan {
        return Ok((projection, None));
    }
    // Each projection entry reads the emitted output positionally, which is
    // where the variant column landed regardless of its index in the table's
    // schema.
    let projections = above_scan
        .into_iter()
        .enumerate()
        .map(|(position, (return_type, extraction))| {
            let column = Expression::Ref(Ref {
                column_idx: position,
                return_type,
                name: None,
            });
            match extraction {
                None => column,
                Some(vg) => Expression::Function(Function::VariantGet(VariantGet {
                    input: Box::new(column),
                    path: vg.path.clone(),
                    as_type: vg.as_type.clone(),
                })),
            }
        })
        .collect();
    Ok((projection, Some(Projection { projections })))
}

/// Renders a table read's output expressions for a plan dump, resolving each
/// column reference back to its real name from the table schema (e.g.
/// `URL:Utf8` rather than `#2:Utf8`). Falls back to the bare expression for
/// the COUNT(*) sentinel index or any non-`Ref`. Shared by the [`Input`] and
/// [`Materialize`](super::Materialize) displays.
pub(crate) fn describe_scan_columns(table: &dyn BoundTable, columns: &[Expression]) -> String {
    let schema = table.columns();
    columns
        .iter()
        .map(|c| match c {
            Expression::Ref(r) => match schema.get(r.column_idx) {
                Some(column) => format!("{}:{}", column.name, r.return_type),
                None => r.to_string(),
            },
            _ => c.to_string(),
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Lower the Top-N's dynamic filters into logical [`DynamicScanPredicate`]s the
/// table can use for pruning. Each carries the column, comparison, and the
/// shared slot the Top-N fills with its live boundary; turning that into actual
/// (e.g. row-group) elimination is the storage backend's job.
pub(crate) fn build_dynamic_scan_predicates(
    filters: &[DynamicFilter],
    slots: &mut RuntimeFilterSlots,
) -> Vec<DynamicScanPredicate> {
    filters
        .iter()
        .map(|df| DynamicScanPredicate {
            column_idx: df.column_idx,
            compare_type: df.compare_type,
            slot: slots.boundary_slot(df.slot_id),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use crate::test_support::*;
    use crate::types::Type;
    use arrow_array::{ArrayRef, Int32Array};
    use arrow_schema::DataType;
    use rstest::rstest;
    use std::sync::Arc;

    #[rstest]
    fn date_column_scans_as_a_real_date(mut testing_planner: TestingPlanner) {
        testing_planner.add_table(
            "events",
            &[(
                "d",
                Type::Date,
                Arc::new(Int32Array::from(vec![0, 7])) as ArrayRef,
            )],
        );

        let batches = run_batches(&mut testing_planner, "SELECT d FROM events");

        // The DATE column is stored as Int32 days but must surface as a real
        // Date32 (reinterpreted zero-copy at the scan), so it renders as a date.
        assert_eq!(batches[0].schema().field(0).data_type(), &DataType::Date32);
    }
}
