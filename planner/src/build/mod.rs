//! Builds the Pivot [`PlanNode`] tree by walking DuckDB's plan handles.
//!
//! [`duckdb_planner::Plan`] hands back the root [`LogicalOp`]; this module walks
//! that live tree through the handle accessors ([`LogicalOp::operator`] dispatches
//! each node to a typed [`Operator`](duckdb_planner::handle::Operator) view) and
//! assembles the typed Pivot [`Operator`]/[`Expression`] values directly. There is
//! no intermediate IR: a handle is one borrowed reference and reading a field is
//! one FFI call.
//!
//! Each node's construction is a `from_handle` on its Pivot type, implemented in
//! the [`operator`] and [`expression`] submodules; this module drives the walk and
//! the DuckDB-specific tree shaping that no single node owns: building the
//! late-materialization node into a [`Materialize`], lifting a scan's
//! pushed-down `table_filters` back into a [`Filter`], and replaying a
//! filter's `projection_map`.

use dispatch::RangeCompare;
use dispatch::RowDelivery;
use duckdb_planner::BoundLogicalType;
use duckdb_planner::Expr;
use duckdb_planner::LogicalOp;
use duckdb_planner::duckdb_bridge::duckdb_types::{ExpressionType, JoinType, TransactionType};
use duckdb_planner::handle::{
    ComparisonJoin as ComparisonJoinView, DelimGet as DelimGetView, DelimJoin as DelimJoinView,
    JoinCondition, JoinConditionEntry, Operator as DuckOperator, PushedConditions,
};

use crate::catalog::BoundTable;
use crate::dynamic_filter::{DynamicFilter, JoinProducedFilter, KeyBitsetFilter};
use crate::expression::{
    Cast, Compare, CompareType, Error as ExpressionError, Expression, Function, Ref, VariantGet,
};
use crate::operator::{
    Aggregate, Compact, CopyFromStdin, CreateSchema, CreateTable, CreateUser, Cte, CteScan,
    Distinct, DropTable, DummyScan, Error as OperatorError, Explain, Filter, Input, Insert, Join,
    JoinKind, Limit, Materialize, Operator, OrderBy, Projection, SetVariable, TableFunctionScan,
    TopN, TransactionStatement, Unnest, Values,
};
use crate::plan::{self, PlanNode};
use crate::types::{Type, physical_arrow_type, type_from_logical};

mod context;
mod expression;
mod operator;

use context::BuildCtx;

/// Build the whole Pivot plan tree from DuckDB's root operator handle.
pub(crate) fn build_plan(root: LogicalOp<'_>) -> Result<PlanNode, plan::Error> {
    let mut ctx = BuildCtx::default();
    let plan = build_node(root, &mut ctx)?;
    reject_struct_outputs(&plan)?;
    Ok(render_variant_outputs(plan)?)
}

/// Refuse a query whose result has a struct column. A struct exists only
/// inside a query, and the result encoders have no form for one.
fn reject_struct_outputs(plan: &PlanNode) -> Result<(), plan::Error> {
    if plan
        .output_types()?
        .iter()
        .any(|column_type| matches!(column_type, Type::Struct(_)))
    {
        return Err(OperatorError::Unsupported(
            "A query cannot return a struct; select its fields instead".to_string(),
        )
        .into());
    }
    Ok(())
}

/// Render the query's variant-typed output columns as JSON text.
///
/// A variant's physical layout can differ per file (each file shreds by its own
/// data), so handing the raw struct to the client would mean result batches of
/// varying shape, and an unreadable binary value even when they don't vary.
/// When the plan's output contains a variant column, wrap the whole plan in one
/// more projection that renders those columns and passes the rest through, so
/// the client always sees uniform JSON text. Asking the plan for its
/// [`output_types`](PlanNode::output_types) makes this work for any root
/// operator; below the added projection, everything still operates on the raw
/// variant.
fn render_variant_outputs(plan: PlanNode) -> Result<PlanNode, crate::compile::Error> {
    let types = plan.output_types()?;
    if !types.contains(&Type::Variant) {
        return Ok(plan);
    }

    let projections = types
        .iter()
        .enumerate()
        .map(|(column_idx, column_type)| {
            let column = Expression::Ref(Ref {
                column_idx,
                return_type: column_type.clone(),
                name: None,
            });
            match column_type {
                // Serialize a variant output column as JSON text: a cast to
                // text, which `Cast::compile` renders each document with.
                Type::Variant => Expression::Cast(Cast {
                    target: Type::Utf8,
                    target_arrow: physical_arrow_type(&Type::Utf8),
                    source: Box::new(column),
                }),
                _ => column,
            }
        })
        .collect();
    Ok(PlanNode {
        name: "render variant outputs".to_string(),
        inputs: vec![plan],
        operator: Operator::Projection(Projection { projections }),
    })
}

fn build_node(op: LogicalOp<'_>, ctx: &mut BuildCtx) -> Result<PlanNode, OperatorError> {
    let kind = op.operator()?;

    // DuckDB's late_materialization optimizer emits an explicit materialize
    // node (see the fork's LogicalPivotMaterialize): child 0 is the wide Get
    // describing the fetch, child 1 the narrow pipeline. Its children walk
    // specially (the wide Get is data, not a scan to execute), so it cannot go
    // through the generic loop below.
    if matches!(kind, DuckOperator::PivotMaterialize) {
        return build_late_materialization(op, ctx);
    }

    // A CTE's two children are walked in order for a reason, so it can't go
    // through the generic loop below: the definition's output shape is what the
    // scans in the body declare as their own.
    if let DuckOperator::MaterializedCte(cte) = kind {
        return build_cte(op, cte.cte_index()?, ctx);
    }

    // A delim join walks its children in order too: the subquery side's delim
    // scans declare the de-duplicated columns of the outer side, which must
    // have been walked first.
    if let DuckOperator::DelimJoin(delim) = kind {
        return build_delim_join(op, delim, ctx);
    }

    let mut inputs = op
        .children()?
        .into_iter()
        .map(|child| build_node(child, ctx))
        .collect::<Result<Vec<_>, _>>()?;

    // DuckDB's optimizer pushes simple `column <op> constant` predicates down
    // into the scan itself (its `table_filters`), so the `Filter` operator that
    // would sit above the scan disappears. Pivot wants that explicit `Filter`
    // back, so every Pivot-backed scan collects the pushed predicates here and,
    // after the node is built (below), wraps it in a synthetic `Filter`, restoring
    // the same shape as if pushdown were off. Empty for every other operator.
    let mut pushed_conditions: Vec<Expression> = Vec::new();

    let operator = match kind {
        DuckOperator::Projection(p) => Operator::Projection(Projection::from_handle(p)?),
        DuckOperator::Values(v) => Operator::Values(Values::from_handle(v)?),
        DuckOperator::ChunkGet(c) => Operator::Values(Values::from_chunk_get(c)?),
        DuckOperator::Insert(i) => {
            let (insert, column_list) = Insert::from_handle(i)?;
            // An INSERT that named its target columns binds a child emitting
            // just those, in the order written. Widen it to the table's full
            // column list here, so the insert itself always writes a child row
            // straight through, column for column.
            if let Some(projection) = column_list {
                let child = inputs
                    .pop()
                    .expect("an INSERT with a column list has a source of rows");
                inputs.push(PlanNode {
                    name: "INSERT_COLUMN_LIST".to_string(),
                    inputs: vec![child],
                    operator: Operator::Projection(projection),
                });
            }
            Operator::Insert(insert)
        }
        DuckOperator::Filter(f) => Operator::Filter(Filter::from_handle(f)?),
        DuckOperator::Aggregate(a) => Operator::Aggregate(Aggregate::from_handle(a)?),
        DuckOperator::Distinct(d) => {
            let child = inputs.first().expect("a DISTINCT dedups a child's rows");
            Operator::Distinct(Distinct::from_handle(d, child.output_types()?.len())?)
        }
        DuckOperator::Unnest(u) => Operator::Unnest(Unnest::from_handle(u)?),
        DuckOperator::OrderBy(o) => Operator::OrderBy(OrderBy::from_handle(o)?),
        DuckOperator::TopN(t) => Operator::TopN(TopN::from_handle(t, ctx)?),
        DuckOperator::Limit(l) => Operator::Limit(Limit::from_handle(l)?),
        DuckOperator::TableScan(scan) => {
            // The pushdown lift is a walk-level transform (it wraps the scan in a
            // Filter below), so it stays here rather than in `Input::from_handle`.
            pushed_conditions = build_pushed_conditions(scan.pushed_conditions()?)?;
            Operator::Input(Input::from_handle(scan, ctx)?)
        }
        DuckOperator::TableFunctionScan(view) => {
            pushed_conditions = build_pushed_conditions(view.pushed_conditions()?)?;
            Operator::TableFunctionScan(TableFunctionScan::from_handle(view, ctx)?)
        }
        DuckOperator::CreateTable(c) => Operator::CreateTable(CreateTable::from_handle(c)?),
        DuckOperator::CreateSchema(c) => Operator::CreateSchema(CreateSchema::from_handle(c)?),
        DuckOperator::Drop(d) => {
            // Tables are the only entry kind pivot can drop; any other kind
            // reports what was asked rather than a generic unsupported-operator
            // error.
            if !d.is_table()? {
                return Err(OperatorError::Unsupported(format!(
                    "DROP {} is not supported",
                    d.entry_kind()?
                )));
            }
            Operator::DropTable(DropTable::from_handle(d)?)
        }
        DuckOperator::Set(s) => Operator::SetVariable(SetVariable::from_set(s)?),
        DuckOperator::Reset(r) => Operator::SetVariable(SetVariable::from_reset(r)?),
        DuckOperator::Transaction(t) => Operator::Transaction(match t.transaction_type()? {
            TransactionType::BEGIN_TRANSACTION => TransactionStatement::Begin,
            TransactionType::COMMIT => TransactionStatement::Commit,
            TransactionType::ROLLBACK => TransactionStatement::Rollback,
            other => {
                return Err(OperatorError::Unsupported(format!(
                    "unsupported transaction statement type {}",
                    other as u8
                )));
            }
        }),
        DuckOperator::Compact(c) => Operator::Compact(Compact::from_handle(c)?),
        DuckOperator::Vacuum => Operator::Vacuum,
        DuckOperator::CopyFromStdin(c) => Operator::CopyFromStdin(CopyFromStdin::from_handle(c)?),
        DuckOperator::CreateUser(c) => Operator::CreateUser(CreateUser::from_handle(c)?),
        // No view to construct from: this carries no kind-specific payload.
        DuckOperator::DummyScan => Operator::DummyScan(DummyScan),
        DuckOperator::Explain(explain) => {
            // EXPLAIN ANALYZE would print the plan without ever running the
            // query; reject it rather than silently answer as plain EXPLAIN.
            if explain.is_analyze()? {
                return Err(OperatorError::Unsupported(
                    "EXPLAIN ANALYZE is not supported".to_string(),
                ));
            }
            Operator::Explain(Explain)
        }
        DuckOperator::ComparisonJoin(join) => {
            return build_join(op, join, inputs, ctx);
        }
        DuckOperator::CteRef(cte_ref) => {
            Operator::CteScan(ctx.create_cte_scan(cte_ref.cte_index()?)?)
        }
        DuckOperator::DelimGet(get) => Operator::CteScan(build_delim_get_scan(get, ctx)?),
        // A delim join is walked above, before its children.
        DuckOperator::DelimJoin(_) => {
            unreachable!("a delim join walks its own children")
        }
        // A CTE is walked above, before its children.
        DuckOperator::MaterializedCte(_) => {
            unreachable!("a CTE walks its own children")
        }
        DuckOperator::PivotMaterialize => {
            unreachable!("a materialize walks its own children")
        }
        DuckOperator::Unsupported => {
            return Err(OperatorError::Unsupported(format!(
                "Unsupported operator type: {}",
                op.name()?
            )));
        }
    };

    let mut node = PlanNode {
        name: op.name()?,
        inputs,
        operator,
    };

    // A filter directly above a Pivot-backed scan folds the scan's lifted
    // pushed-down conditions into itself (see `absorb_scan_pushdown_filter`).
    // The peek at the DuckDB child is what grounds the fold: only a scan child
    // means the built input's Filter can be the synthetic wrapper.
    if matches!(kind, DuckOperator::Filter(_))
        && match op.children()?.first() {
            Some(child) => matches!(
                child.operator()?,
                DuckOperator::TableScan(_) | DuckOperator::TableFunctionScan(_)
            ),
            None => false,
        }
    {
        node = absorb_scan_pushdown_filter(node);
    }

    // Reattach the scan's static pushed-down filters as a Filter above it, so the
    // Rust side keeps seeing `Filter -> Input` exactly as with filter_pushdown off.
    // Only Pivot-backed scans populate `pushed_conditions`.
    if !pushed_conditions.is_empty() {
        return Ok(PlanNode {
            name: "PUSHDOWN_FILTER".to_string(),
            inputs: vec![node],
            operator: Operator::Filter(Filter {
                conditions: pushed_conditions,
                delivery: RowDelivery::Coalesced,
            }),
        });
    }

    // A LogicalFilter may carry a projection_map: it outputs only the listed
    // subset/reordering of its child's columns. Replay it by wrapping the filter
    // in a Projection that selects exactly projection_map, positionally, so refs
    // above the filter line up.
    if let DuckOperator::Filter(f) = kind {
        // A filter's projection_map is a plain positional reorder of its child's
        // columns, never a variant field extract, so each carries an empty path.
        let projections = build_scan_columns(
            f.projection_map()?
                .into_iter()
                .map(|(idx, ty)| (idx, ty, Vec::new()))
                .collect(),
        )?;
        if !projections.is_empty() {
            return Ok(PlanNode {
                name: "FILTER_PROJECTION".to_string(),
                inputs: vec![node],
                operator: Operator::Projection(Projection { projections }),
            });
        }
    }

    Ok(node)
}

/// Fold the synthetic pushdown filter under `node` into `node` itself.
///
/// Called only for a filter whose DuckDB child is a base-table scan. By
/// construction (see the `TableScan` arm) the built input is then either the
/// raw `Input` or the synthetic Filter carrying the scan's lifted pushed-down
/// conditions - never a user filter - so a Filter input is always the
/// wrapper, and it is safe to dissolve. The wrapper's conditions go first:
/// they sit closer to the scan, where DuckDB puts the typically most
/// selective predicates. One filter evaluates its conditions progressively
/// and materializes survivors once, where a stack of filters would
/// re-materialize at every level - the lower one at its own (much weaker)
/// selectivity.
fn absorb_scan_pushdown_filter(node: PlanNode) -> PlanNode {
    if node.inputs.len() != 1 || !matches!(node.inputs[0].operator, Operator::Filter(_)) {
        // The scan had no pushed-down conditions, so there is no wrapper.
        return node;
    }
    let PlanNode {
        name,
        mut inputs,
        operator,
    } = node;
    let Operator::Filter(filter) = operator else {
        unreachable!("the caller matched a filter");
    };
    let wrapper = inputs.remove(0);
    let Operator::Filter(mut merged) = wrapper.operator else {
        unreachable!("matched as a filter above");
    };
    merged.conditions.extend(filter.conditions);
    PlanNode {
        name,
        inputs: wrapper.inputs,
        operator: Operator::Filter(merged),
    }
}

/// Translate a general comparison join into pivot's [`Join`]. The supported
/// shape is an INNER, RIGHT or SEMI join with one or more equality conditions
/// whose sides are plain column refs of key types the dispatch join handles;
/// anything else reports the specific gap. The probe is DuckDB's left child
/// and the build its right, matching its hash-join convention (the cost model
/// puts the smaller relation on the right).
///
/// A join may additionally carry expression-form conditions (a predicate
/// referencing both sides that is not a bare comparison, e.g. an OR of
/// per-side conjunctions). Those become the [`Join`]'s residual: part of the
/// join itself, evaluated on key-matched pairs, so a pair the predicate
/// rejects is not a match. That placement is what keeps outer and semi joins
/// correct, where the predicate takes part in the match decision (an outer
/// build row all of whose pairs are rejected is emitted null-filled). DuckDB
/// resolved such a predicate against the concatenation of both children's
/// full outputs (see its `ColumnBindingResolver`), which is the layout the
/// dispatch join evaluates it in.
///
/// RIGHT is the outer join whose preserved side is that right child, so it
/// maps onto the dispatch join's build-side outer mode, and LEFT onto its
/// probe-side outer mode. DuckDB writes whichever puts the preserved relation
/// on the side its cost model wants it: a written `LEFT JOIN` lands here as
/// RIGHT whenever its preserved relation is the smaller one, and stays LEFT
/// when it is the larger.
///
/// SEMI keeps rows of the left child, so it is semi on the *probe* side. The
/// mirrored RIGHT_SEMI, which DuckDB's build-probe-side optimizer produces
/// when it would rather build the hash table from the preserved relation (an
/// `IN` whose outer table is far smaller than the subquery's), is semi on the
/// build side.
///
/// ANTI is SEMI's negation, one output row per left-child row with no
/// surviving match, so it is anti on the probe side. Here the mirrored
/// RIGHT_ANTI *is* supported, as anti on the build side: the same optimizer
/// rewrites an ANTI into it whenever the preserved relation is the cheaper
/// side to build the hash table from, which for anti joins (preserving the
/// rows a typically larger side fails to match) is routine rather than rare.
///
/// MARK keeps every left-child row and appends a boolean marker column, the
/// three-valued result of `IN (...)`: what DuckDB plans an uncorrelated `IN`
/// subquery, or the constant chunk its optimizer rewrites a long `IN` list
/// into, as. The marker is TRUE on a match, FALSE on a miss, and NULL where
/// a null key leaves the miss unknown.
///
/// DuckDB's join projection maps (which trim the join's output to the columns
/// actually used above it) are folded into the join's own output lists, so
/// the dispatch probe never materializes trimmed columns. An empty map means
/// every column of that side is kept. (DuckDB also emits an empty map when NO
/// column of a side is referenced above the join, so a bare `COUNT(*)` join
/// still materializes both keys; distinguishing that case needs information
/// the plan does not carry.)
fn build_join(
    op: LogicalOp<'_>,
    join: ComparisonJoinView<'_>,
    mut inputs: Vec<PlanNode>,
    ctx: &mut BuildCtx,
) -> Result<PlanNode, OperatorError> {
    let probe_types = inputs[0].output_types()?;
    let build_types = inputs[1].output_types()?;
    let mut probe_keys = Vec::new();
    let mut build_keys = Vec::new();
    let mut key_types = Vec::new();
    let mut computed_probe_keys = Vec::new();
    let mut computed_build_keys = Vec::new();
    let mut residual_predicates = Vec::new();
    let mut inequalities = Vec::new();
    // The equality-key position of each DuckDB condition index, for resolving
    // join-filter-pushdown targets (which name conditions by that index).
    let mut key_position_of_condition = Vec::new();
    for entry in join.conditions()? {
        let condition = match entry {
            JoinConditionEntry::Comparison(condition) => condition,
            JoinConditionEntry::Expression(predicate) => {
                residual_predicates.push(predicate);
                key_position_of_condition.push(None);
                continue;
            }
        };
        if condition.comparison != ExpressionType::COMPARE_EQUAL {
            inequalities.push(condition);
            key_position_of_condition.push(None);
            continue;
        }
        key_position_of_condition.push(Some(key_types.len()));
        let (probe_key, probe_key_type) = join_key_expression(
            Expression::from_handle(condition.left)?,
            probe_types.len(),
            &mut computed_probe_keys,
        )?;
        let (build_key, build_key_type) = join_key_expression(
            Expression::from_handle(condition.right)?,
            build_types.len(),
            &mut computed_build_keys,
        )?;
        if probe_key_type != build_key_type {
            return Err(OperatorError::Unsupported(format!(
                "join key types differ: {probe_key_type:?} vs {build_key_type:?} \
                 (the planner casts both sides of a condition to a common type, so this plan \
                 shape is unexpected)"
            )));
        }
        probe_keys.push(probe_key);
        build_keys.push(build_key);
        key_types.push(probe_key_type);
    }
    // With no equality to hash on, a single comparison condition runs as a
    // range join instead.
    if key_types.is_empty() {
        return build_range_join(op, join, inputs, inequalities, residual_predicates);
    }
    let computed_probe_key_count = computed_probe_keys.len();
    if !computed_probe_keys.is_empty() {
        let probe = inputs.remove(0);
        inputs.insert(0, append_computed_keys(probe, computed_probe_keys)?);
    }
    if !computed_build_keys.is_empty() {
        let build = inputs.pop().expect("a join has two inputs");
        inputs.push(append_computed_keys(build, computed_build_keys)?);
    }
    let mut residual_filters = residual_predicates
        .into_iter()
        .map(Expression::from_handle)
        .collect::<Result<Vec<_>, _>>()?;
    // An expression-form residual was resolved against the original combined
    // layout [probe, build]. Appended probe keys move the build block right;
    // appended build keys sit after it and move nothing. Rebase only the refs
    // that named the original build block. This is a plan-time walk and adds
    // no work to residual evaluation.
    if computed_probe_key_count != 0 {
        for residual in &mut residual_filters {
            residual.visit_column_refs_mut(&mut |reference| {
                if reference.column_idx >= probe_types.len() {
                    reference.column_idx += computed_probe_key_count;
                }
            });
        }
    }
    let residual_probe_width = probe_types.len() + computed_probe_key_count;
    // A non-equality comparison condition beside the hash equalities (e.g.
    // `l = r AND a <> b`) rides the join as one more residual over the
    // key-matched pairs. Its left operand was resolved against the probe
    // input alone and its right against the build input alone, so the build
    // side's column refs shift past the probe columns into the combined
    // probe-then-build layout every residual is evaluated in.
    for condition in inequalities {
        let message = format!(
            "Unsupported join comparison type: {:?}",
            condition.comparison
        );
        let compare_type = CompareType::try_from(condition.comparison)
            .map_err(|_| OperatorError::Unsupported(message))?;
        let left = Expression::from_handle(condition.left)?;
        let mut right = Expression::from_handle(condition.right)?;
        right.visit_column_refs_mut(&mut |reference| {
            reference.column_idx += residual_probe_width;
        });
        residual_filters.push(Expression::Compare(Compare {
            left: Box::new(left),
            right: Box::new(right),
            compare_type,
            return_type: Type::Boolean,
        }));
    }
    let (residual_probe_columns, residual_build_columns) =
        compact_join_residuals(&mut residual_filters, residual_probe_width);
    let left_map: Vec<usize> = join.left_projection_map()?;
    let right_map: Vec<usize> = join.right_projection_map()?;

    // Refs above the join were resolved against the trimmed output (kept left
    // columns, then kept right columns), which is exactly the layout the join
    // emits with these lists.
    let kept_probe_output: Vec<usize> = if left_map.is_empty() {
        (0..probe_types.len()).collect()
    } else {
        left_map
    };
    let kept_build_output: Vec<usize> = if right_map.is_empty() {
        (0..build_types.len()).collect()
    } else {
        right_map
    };
    let (kind, probe_output, build_output) = match join.join_type()? {
        JoinType::INNER => (JoinKind::Inner, kept_probe_output, kept_build_output),
        JoinType::RIGHT => (JoinKind::BuildOuter, kept_probe_output, kept_build_output),
        JoinType::LEFT => (JoinKind::ProbeOuter, kept_probe_output, kept_build_output),
        // A semi or anti join emits no build column at all. DuckDB agrees, and
        // says so by returning the left bindings alone for such a join rather
        // than through the right projection map, which it never reads here -
        // so the map's "empty means keep every column" reading must not be
        // applied.
        JoinType::SEMI => (JoinKind::ProbeSemi, kept_probe_output, Vec::new()),
        JoinType::ANTI => (JoinKind::ProbeAnti, kept_probe_output, Vec::new()),
        // The mirror images: a right semi or anti join emits build columns
        // alone, and DuckDB never reads the left projection map for either.
        JoinType::RIGHT_SEMI => (JoinKind::BuildSemi, Vec::new(), kept_build_output),
        JoinType::RIGHT_ANTI => (JoinKind::BuildAnti, Vec::new(), kept_build_output),
        // A mark join emits the left bindings plus its marker column, which
        // the dispatch join appends after the probe columns. The marker's
        // three-valued semantics are defined per key comparison, so only the
        // single-equality shape (what `IN` compiles to) translates.
        JoinType::MARK => {
            if probe_keys.len() != 1 || !residual_filters.is_empty() {
                return Err(OperatorError::Unsupported(
                    "mark joins are supported on exactly one equality condition".to_string(),
                ));
            }
            (JoinKind::ProbeMark, kept_probe_output, Vec::new())
        }
        other => {
            return Err(OperatorError::Unsupported(format!(
                "Unsupported join type: {other:?}"
            )));
        }
    };
    let probe_nullability = inputs[0].output_nullability();
    let build_nullability = inputs[1].output_nullability();
    let probe_column_types = probe_output
        .iter()
        .map(|&i| (probe_types[i].clone(), probe_nullability[i]))
        .collect();
    let build_column_types = build_output
        .iter()
        .map(|&i| (build_types[i].clone(), build_nullability[i]))
        .collect();

    // Join-filter pushdown: wire each target the optimizer recorded to its
    // probe-side scan. The scan gets a `>= min` / `<= max` consumer pair on
    // the probed column, and the join records the matching producer entry so
    // its build publishes the key bounds into the slots when it seals. The
    // optimizer only records targets on join types where a non-matching probe
    // row produces no output, so pruning such rows early is safe.
    let mut produced_filters = Vec::new();
    for target in join.pushdown_targets()? {
        for column in &target.columns {
            let Some(&Some(key_position)) = key_position_of_condition.get(column.condition_index)
            else {
                continue;
            };
            let Some(scan) = find_join_filter_scan(
                &mut inputs[0],
                target.filter_set_id,
                column.probe_table_index,
            ) else {
                continue;
            };
            let info = scan
                .join_filter_info
                .as_ref()
                .expect("a matched scan carries its join-filter wiring");
            let Some(&storage_column) = info.proj_to_storage.get(column.probe_column) else {
                continue;
            };
            let min_slot_id = ctx.allocate_join_filter_slot();
            let max_slot_id = ctx.allocate_join_filter_slot();
            let key_bitset_slot_id = ctx.allocate_key_bitset_slot();
            scan.dynamic_filters.push(DynamicFilter {
                slot_id: min_slot_id,
                column_idx: storage_column,
                compare_type: CompareType::GreaterEqual,
            });
            scan.dynamic_filters.push(DynamicFilter {
                slot_id: max_slot_id,
                column_idx: storage_column,
                compare_type: CompareType::LessEqual,
            });
            scan.key_bitset_filters.push(KeyBitsetFilter {
                slot_id: key_bitset_slot_id,
                column_idx: storage_column,
            });
            produced_filters.push(JoinProducedFilter {
                key_position,
                min_slot_id,
                max_slot_id,
                key_bitset_slot_id,
            });
        }
    }

    Ok(PlanNode {
        name: op.name()?,
        inputs,
        operator: Operator::Join(Join {
            probe_keys,
            build_keys,
            key_types,
            probe_output,
            build_output,
            probe_column_types,
            build_column_types,
            residual_filters,
            residual_probe_columns,
            residual_build_columns,
            kind,
            produced_filters,
        }),
    })
}

/// Find the probe-subtree scan a join-filter target names, by the shared-set
/// identity and table index its wiring carries. Anything can sit between the
/// join and its scan (filters, projections, other joins), so the whole
/// subtree is searched.
fn find_join_filter_scan(
    node: &mut PlanNode,
    filter_set_id: usize,
    table_index: usize,
) -> Option<&mut Input> {
    match &mut node.operator {
        Operator::Input(input) => (input.duckdb_table_binding_index == Some(table_index)
            && input
                .join_filter_info
                .as_ref()
                .is_some_and(|info| info.filter_set_id == filter_set_id))
        .then_some(input),
        _ => node
            .inputs
            .iter_mut()
            .find_map(|child| find_join_filter_scan(child, filter_set_id, table_index)),
    }
}

/// Narrow residual evaluation from the join's full `[probe, build]` input to
/// just the columns its expressions read. Returns the input-column projections
/// and rewrites refs in place to address their compact concatenation. This runs
/// once while building the plan; every execution then gathers the minimal
/// candidate-pair batch directly.
fn compact_join_residuals(
    residuals: &mut [Expression],
    probe_width: usize,
) -> (Vec<usize>, Vec<usize>) {
    let mut referenced_columns = Vec::new();
    for residual in residuals.iter() {
        residual.collect_column_refs(&mut referenced_columns);
    }

    let mut probe_columns = Vec::new();
    let mut build_columns = Vec::new();
    for column_idx in referenced_columns {
        if column_idx < probe_width {
            probe_columns.push(column_idx);
        } else {
            build_columns.push(column_idx - probe_width);
        }
    }
    probe_columns.sort_unstable();
    probe_columns.dedup();
    build_columns.sort_unstable();
    build_columns.dedup();

    for residual in residuals {
        residual.visit_column_refs_mut(&mut |reference| {
            reference.column_idx = if reference.column_idx < probe_width {
                probe_columns
                    .binary_search(&reference.column_idx)
                    .expect("every residual probe ref was collected")
            } else {
                let build_column_idx = reference.column_idx - probe_width;
                probe_columns.len()
                    + build_columns
                        .binary_search(&build_column_idx)
                        .expect("every residual build ref was collected")
            };
        });
    }

    (probe_columns, build_columns)
}

/// Build a [`JoinKind::Range`] join: an inner join whose one condition is a
/// `<`/`<=`/`>`/`>=` comparison, with no equality to hash on.
fn build_range_join(
    op: LogicalOp<'_>,
    join: ComparisonJoinView<'_>,
    mut inputs: Vec<PlanNode>,
    inequalities: Vec<JoinCondition<'_>>,
    residual_predicates: Vec<Expr<'_>>,
) -> Result<PlanNode, OperatorError> {
    let [condition] = inequalities.as_slice() else {
        return Err(OperatorError::Unsupported(format!(
            "joins must have an equality condition or exactly one comparison, got {} comparisons",
            inequalities.len()
        )));
    };
    if !residual_predicates.is_empty() {
        return Err(OperatorError::Unsupported(
            "a range join does not support extra conditions".to_string(),
        ));
    }
    if join.join_type()? != JoinType::INNER {
        return Err(OperatorError::Unsupported(format!(
            "Unsupported range join type: {:?}",
            join.join_type()?
        )));
    }
    let compare = match &condition.comparison {
        &ExpressionType::COMPARE_LESSTHAN => RangeCompare::Less,
        &ExpressionType::COMPARE_LESSTHANOREQUALTO => RangeCompare::LessEq,
        &ExpressionType::COMPARE_GREATERTHAN => RangeCompare::Greater,
        &ExpressionType::COMPARE_GREATERTHANOREQUALTO => RangeCompare::GreaterEq,
        other => {
            return Err(OperatorError::Unsupported(format!(
                "Unsupported join comparison type: {other:?}"
            )));
        }
    };

    let probe_key_expr = Expression::from_handle(condition.left)?;
    let build_key_expr = Expression::from_handle(condition.right)?;
    let key_type = probe_key_expr.result_type()?;
    if build_key_expr.result_type()? != key_type {
        return Err(OperatorError::Unsupported(format!(
            "join key types differ: {key_type:?} vs {:?} \
             (the planner casts both sides of a condition to a common type, so this plan \
             shape is unexpected)",
            build_key_expr.result_type()?
        )));
    }
    // The sorted build side compares keys by a fixed-width total order
    // (floats by IEEE totalOrder, NaN greatest, matching the build sort);
    // strings are not fixed-width and stay unsupported.
    match key_type {
        Type::Int8
        | Type::Int16
        | Type::Int32
        | Type::Int64
        | Type::UInt8
        | Type::UInt16
        | Type::UInt32
        | Type::UInt64
        | Type::Int128
        | Type::Decimal { .. }
        | Type::Float32
        | Type::Float64
        | Type::Date
        | Type::Timestamp
        | Type::TimestampTz => {}
        other => {
            return Err(OperatorError::Unsupported(format!(
                "Unsupported range join key type: {other:?}"
            )));
        }
    }

    let build_node = inputs.pop().expect("a join has two inputs");
    let probe_node = inputs.pop().expect("a join has two inputs");
    // The output lists below index the sides' original columns, so they are
    // sized before a computed key gets materialized as an extra column.
    let probe_width = probe_node.output_types()?.len();
    let build_width = build_node.output_types()?.len();
    let (probe_node, probe_key) = materialize_key_column(probe_node, probe_key_expr)?;
    let (build_node, build_key) = materialize_key_column(build_node, build_key_expr)?;

    let left_map: Vec<usize> = join.left_projection_map()?;
    let right_map: Vec<usize> = join.right_projection_map()?;
    let probe_output: Vec<usize> = if left_map.is_empty() {
        (0..probe_width).collect()
    } else {
        left_map
    };
    let build_output: Vec<usize> = if right_map.is_empty() {
        (0..build_width).collect()
    } else {
        right_map
    };

    let probe_types = probe_node.output_types()?;
    let build_types = build_node.output_types()?;
    let probe_nullability = probe_node.output_nullability();
    let build_nullability = build_node.output_nullability();
    let probe_column_types = probe_output
        .iter()
        .map(|&i| (probe_types[i].clone(), probe_nullability[i]))
        .collect();
    let build_column_types = build_output
        .iter()
        .map(|&i| (build_types[i].clone(), build_nullability[i]))
        .collect();
    Ok(PlanNode {
        name: op.name()?,
        inputs: vec![probe_node, build_node],
        operator: Operator::Join(Join {
            probe_keys: vec![probe_key],
            build_keys: vec![build_key],
            key_types: vec![key_type],
            probe_output,
            build_output,
            probe_column_types,
            build_column_types,
            residual_filters: Vec::new(),
            residual_probe_columns: Vec::new(),
            residual_build_columns: Vec::new(),
            kind: JoinKind::Range(compare),
            produced_filters: Vec::new(),
        }),
    })
}

/// The column a join key expression is read from. A plain column ref is its
/// own answer; a computed key (e.g. the cast the planner wraps one side in to
/// unify types) is materialized by a projection that passes every existing
/// column through and appends the key as one more.
fn materialize_key_column(
    input: PlanNode,
    key: Expression,
) -> Result<(PlanNode, usize), OperatorError> {
    if let Expression::Ref(key) = &key {
        return Ok((input, key.column_idx));
    }
    let types = input.output_types()?;
    let mut projections: Vec<Expression> = types
        .iter()
        .enumerate()
        .map(|(column_idx, column_type)| {
            Expression::Ref(Ref {
                column_idx,
                return_type: column_type.clone(),
                name: None,
            })
        })
        .collect();
    let key_column = projections.len();
    projections.push(key);
    Ok((
        PlanNode {
            name: "materialize join key".to_string(),
            inputs: vec![input],
            operator: Operator::Projection(Projection { projections }),
        },
        key_column,
    ))
}

/// The column ref a join key must be, with the key type restriction the
/// dispatch join imposes: any fixed-width hashable type, or text. Floats
/// hash and compare by SQL semantics, not bit patterns, so they are out.
fn join_key_ref(key: Expression) -> Result<(usize, Type), OperatorError> {
    let Expression::Ref(key) = key else {
        return Err(OperatorError::Unsupported(format!(
            "join keys must be plain columns, got: {key:?}"
        )));
    };
    Ok((key.column_idx, joinable_key_type(key.return_type)?))
}

/// The key's type if a hash join can key on it, an error otherwise.
fn joinable_key_type(key_type: Type) -> Result<Type, OperatorError> {
    match key_type {
        Type::Int8
        | Type::Int16
        | Type::Int32
        | Type::Int64
        | Type::UInt8
        | Type::UInt16
        | Type::UInt32
        | Type::UInt64
        | Type::Int128
        | Type::Decimal { .. }
        | Type::Date
        | Type::Timestamp
        | Type::TimestampTz
        | Type::Utf8 => Ok(key_type),
        _ => Err(OperatorError::Unsupported(format!(
            "Unsupported join key type: {key_type:?}"
        ))),
    }
}

/// One side of a join equality: a plain column keys the join directly, and a
/// computed expression (e.g. `substring(phone, 1, 2) IN (...)` arriving as a
/// mark join keyed on the substring) is appended to `computed_keys`, to be
/// projected onto the end of that side's input, and keys the join at the
/// appended position.
fn join_key_expression(
    key: Expression,
    input_width: usize,
    computed_keys: &mut Vec<Expression>,
) -> Result<(usize, Type), OperatorError> {
    if matches!(key, Expression::Ref(_)) {
        return join_key_ref(key);
    }
    let key_type = joinable_key_type(key.result_type().map_err(|error| {
        OperatorError::Unsupported(format!("cannot type the join key {key}: {error}"))
    })?)?;
    let index = input_width + computed_keys.len();
    computed_keys.push(key);
    Ok((index, key_type))
}

/// Wrap `input` in a projection that passes its columns through and appends
/// `computed_keys` after them, so a join can key on the appended columns
/// while every existing column index stays valid.
fn append_computed_keys(
    input: PlanNode,
    computed_keys: Vec<Expression>,
) -> Result<PlanNode, OperatorError> {
    let types = input.output_types()?;
    let projections = types
        .iter()
        .enumerate()
        .map(|(column_idx, column_type)| {
            Expression::Ref(Ref {
                column_idx,
                return_type: column_type.clone(),
                name: None,
            })
        })
        .chain(computed_keys)
        .collect();
    Ok(PlanNode {
        name: "JOIN_KEY_COMPUTE".to_string(),
        inputs: vec![input],
        operator: Operator::Projection(Projection { projections }),
    })
}

/// A scan's projected output columns (and a filter's `projection_map`), each a
/// positional `BOUND_REF` over storage column indices. Shared by the walk's
/// `projection_map` replay and the scan constructors in [`operator`].
fn build_scan_columns(
    columns: Vec<(usize, BoundLogicalType, Vec<String>)>,
) -> Result<Vec<Expression>, ExpressionError> {
    columns
        .into_iter()
        .map(|(column_idx, col_type, path)| {
            let return_type = type_from_logical(col_type)?;
            if path.is_empty() {
                return Ok(Expression::Ref(Ref {
                    column_idx,
                    return_type,
                    name: None,
                }));
            }
            // DuckDB's projection-pushdown pushed a variant field extract into
            // this scan output. Read it as a `VariantGet` over the variant
            // column so the reader resolves the path to a single shredded leaf
            // instead of materializing the whole variant. A `VARIANT` output is
            // a bare extraction (the sub-variant); any other type is a typed read.
            let as_type = match return_type {
                Type::Variant => None,
                typed => Some(typed),
            };
            Ok(Expression::Function(Function::VariantGet(VariantGet {
                input: Box::new(Expression::Ref(Ref {
                    column_idx,
                    return_type: Type::Variant,
                    name: None,
                })),
                path,
                as_type,
            })))
        })
        .collect()
}

fn build_pushed_conditions(list: PushedConditions) -> Result<Vec<Expression>, OperatorError> {
    Ok(list
        .exprs()?
        .into_iter()
        .map(Expression::from_handle)
        .collect::<Result<Vec<_>, _>>()?)
}

/// Walk a CTE: its definition first, so the scans reading it in the body know
/// what shape the rows arrive in, then the body.
fn build_cte(
    op: LogicalOp<'_>,
    cte_index: usize,
    ctx: &mut BuildCtx,
) -> Result<PlanNode, OperatorError> {
    let definition = build_node(op.child(0)?, ctx)?;
    let types = definition.output_types()?;
    let nullable = definition.output_nullability();
    ctx.register_cte_output(cte_index, types, nullable);

    let body = build_node(op.child(1)?, ctx)?;
    let sites = ctx.take_cte_sites(cte_index);
    if sites == 0 {
        return Err(OperatorError::Unsupported(format!(
            "CTE #{cte_index} is materialized but never read"
        )));
    }

    Ok(PlanNode {
        name: op.name()?,
        inputs: vec![definition, body],
        operator: Operator::Cte(Cte { cte_index, sites }),
    })
}

/// One scan of the innermost enclosing delim join's de-duplicated values.
fn build_delim_get_scan(
    get: DelimGetView<'_>,
    ctx: &mut BuildCtx,
) -> Result<CteScan, OperatorError> {
    let Some(dedup_keys_cte) = ctx.delim_target() else {
        return Err(OperatorError::Unsupported(
            "DELIM_GET outside a delim join".to_string(),
        ));
    };
    let scan = ctx.create_cte_scan(dedup_keys_cte)?;
    let declared: Vec<Type> = get
        .column_types()?
        .into_iter()
        .map(type_from_logical)
        .collect::<Result<_, _>>()
        .map_err(ExpressionError::from)?;
    if declared != scan.types {
        return Err(OperatorError::Unsupported(format!(
            "DELIM_GET declares {declared:?} but the delim join de-duplicates {:?}",
            scan.types
        )));
    }
    Ok(scan)
}

/// Translate DuckDB's delim join into ordinary Pivot operators.
///
/// A delim join is how DuckDB evaluates many correlated subqueries without
/// running the subquery separately for every outer row. For example:
///
/// ```sql
/// SELECT ...
/// FROM sales, parts
/// WHERE p_key = s_part
///   AND p_flag = 'a'
///   AND s_qty < (
///       SELECT avg(s2.s_qty)
///       FROM sales s2
///       WHERE s2.s_part = p_key
///   )
/// ```
///
/// The inner query is correlated because its `p_key` comes from the outer
/// query. Another way to read it is: "for each relevant `p_key`, calculate an
/// average, then attach that average to every outer row with that key."
///
/// DuckDB represents this as a delim join with two ordinary children and one
/// implicit data dependency between them:
///
/// * Child 0 is the **outer side**. It produces the candidate rows from
///   `sales` and `parts`, before applying the predicate that uses the scalar
///   subquery. These rows can contain the same correlation key many times.
/// * Child 1 is the **subquery side**. It produces one `(key, answer)` row for
///   each relevant correlation key. Child 1 has no normal input edge from
///   child 0, so a `DELIM_GET` leaf supplies it with the distinct keys that
///   child 0 produced. The delim join's `duplicate_eliminated_columns` say
///   which child-0 columns make up those keys. A key may be a tuple of several
///   columns, not just one column.
///
/// The dataflow is therefore:
///
/// 1. Run the outer side once.
/// 2. Keep its complete rows for the final join, and separately take the
///    distinct correlation keys from those rows.
/// 3. Feed the distinct keys to every `DELIM_GET` in the subquery side.
/// 4. Run the subquery once for that set of keys, producing `(key, answer)`
///    rows.
/// 5. Join those answers back to the complete outer rows. Duplicate outer
///    keys deliberately reappear here: every original row receives the answer
///    for its key.
///
/// For a concrete example, suppose the outer side produces:
///
/// ```text
/// p_key  s_qty
///      1      4
///      1      8
///      2      9
/// ```
///
/// `DELIM_DEDUP` reduces the correlation-key column to `{1, 2}`. The
/// subquery's `DELIM_GET` reads those two keys, joins them to `sales s2`, and
/// produces `(1, avg 6)` and `(2, avg 9)`. The final delim join attaches the
/// answers to the original rows:
///
/// ```text
/// p_key  s_qty  avg
///      1      4    6
///      1      8    6
///      2      9    9
/// ```
///
/// The `s_qty < avg` filter above the delim join then keeps only the first
/// row. One batched subquery plan computed the answers for the distinct key
/// set rather than rerunning the subquery for every outer row.
///
/// Pivot has no native delim-join operator, so this function expresses the
/// implicit dependency with two synthetic CTEs. Here a CTE means "run this
/// definition once, make its rows available to the scans in its body, and
/// return the body's output."
///
/// * `outer_rows_cte` publishes all outer rows. It has two readers: the
///   distinct operation and the final join.
/// * `dedup_keys_cte` publishes only the distinct correlation keys. Each
///   translated `DELIM_GET` becomes a scan of this CTE.
///
/// The complete plan produced here is:
///
/// ```text
/// DELIM_JOIN: Cte(outer_rows_cte)
/// ├── definition: (outer side)
/// │                 Produces the complete candidate rows once.
/// └── body: DELIM_KEY_FEED: Cte(dedup_keys_cte)
///     ├── definition: DELIM_DEDUP
///     │   └── DELIM_OUTER_ROWS: CteScan(outer_rows_cte)
///     │                 First reader of the complete outer rows; emits
///     │                 only the distinct correlation-key tuples.
///     └── body: DELIM_SUBQUERY_JOIN
///         ├── DELIM_OUTER_ROWS: CteScan(outer_rows_cte)
///         │                 Second reader of the complete outer rows; probes
///         │                 for its subquery answer.
///         └── (subquery side)
///             └── ... DELIM_GET -> CteScan(dedup_keys_cte)
///                       Reads the keys published by DELIM_DEDUP and
///                       computes one answer per key, the join's build side.
/// ```
///
/// Every kind keeps that orientation: the subquery side's answer table (one
/// row per distinct key) is the small side, so it is the one worth building a
/// hash table from, and the outer rows probe it. LEFT preserves the outer
/// rows through [`JoinKind::ProbeOuter`], which pads an unmatched probe row
/// with NULL build columns; SEMI keeps a probing row only when an answer
/// exists, and ANTI (a decorrelated `NOT EXISTS`) only when none does.
///
/// A *flipped* delim join is DuckDB's mirror image of all of the above: its
/// build-probe-side optimizer swaps the children whenever the outer side is
/// the cheaper one to build the hash table from, marks the join
/// `delim_flipped`, and mirrors the join type. The outer rows (and with them
/// the dedup source) are then the plan's RIGHT child, the subquery side with
/// the DELIM_GETs the LEFT, and the join arrives as RIGHT, RIGHT_SEMI,
/// RIGHT_ANTI, or INNER. The translation here is the same CTE structure with
/// the join's sides swapped: the subquery side probes and the outer rows are
/// the build, so the mirrored types land on the build-side kinds
/// ([`JoinKind::BuildOuter`], [`JoinKind::BuildSemi`],
/// [`JoinKind::BuildAnti`]), each of which preserves (or drops) build rows
/// exactly as its probe-side twin treats probe rows. No reorder is needed:
/// probe-then-build columns is LHS-then-RHS, DuckDB's output order.
///
/// DuckDB may express the final delim-join condition as `IS NOT DISTINCT FROM`
/// because the distinct key set can contain NULL. For the supported shapes,
/// the subquery reaches this final join through its own ordinary equality
/// predicate against the key set; that equality does not produce a subquery
/// key for NULL. We can therefore use plain equality here: non-NULL keys behave
/// identically, while a NULL-keyed outer row remains unmatched and is padded
/// for LEFT, discarded for SEMI/INNER, and kept for ANTI, matching DuckDB's
/// result (a `NOT EXISTS` whose correlated equality compares against NULL
/// finds no row, so it holds). The flipped kinds inherit the argument: a
/// NULL-keyed outer row is then a build row without a hash-table tuple, which
/// the build-side outer join pads, the build-side semi join drops, and the
/// build-side anti join emits.
fn build_delim_join(
    op: LogicalOp<'_>,
    delim: DelimJoinView<'_>,
    ctx: &mut BuildCtx,
) -> Result<PlanNode, OperatorError> {
    let flipped = delim.is_flipped()?;
    let join = delim.join();

    // The outer side (the dedup source) is the LHS, or the RHS of a flipped
    // join; the subquery side holding the DELIM_GETs is the other child.
    let (outer_child, subquery_child) = if flipped { (1, 0) } else { (0, 1) };
    let outer_side = build_node(op.child(outer_child)?, ctx)?;
    let outer_types = outer_side.output_types()?;
    let outer_nullability = outer_side.output_nullability();

    let mut correlation_columns: Vec<(usize, Type)> = Vec::new();
    for column in delim.delim_columns()? {
        let column = Expression::from_handle(column)?;
        let Expression::Ref(reference) = column else {
            return Err(OperatorError::Unsupported(format!(
                "delim columns must be plain columns, got: {column:?}"
            )));
        };
        correlation_columns.push((reference.column_idx, reference.return_type));
    }
    if correlation_columns.is_empty() {
        return Err(OperatorError::Unsupported(
            "a delim join must de-duplicate at least one column".to_string(),
        ));
    }

    // The two shared streams, as synthetic CTE channels: the outer side's
    // rows, and the dedup'd correlation keys the subquery side scans.
    let outer_rows_cte = ctx.create_synthetic_cte(outer_types.clone(), outer_nullability.clone());
    let dedup_keys_cte = ctx.create_synthetic_cte(
        correlation_columns
            .iter()
            .map(|(_, ty)| ty.clone())
            .collect(),
        correlation_columns
            .iter()
            .map(|&(idx, _)| outer_nullability[idx])
            .collect(),
    );

    let subquery_child = op.child(subquery_child)?;
    let subquery_side =
        ctx.with_delim_target(dedup_keys_cte, |ctx| build_node(subquery_child, ctx))?;
    let delim_scan_sites = ctx.take_cte_sites(dedup_keys_cte);
    if delim_scan_sites == 0 {
        return Err(OperatorError::Unsupported(
            "a delim join whose subquery side reads no DELIM_GET".to_string(),
        ));
    }

    let mut outer_keys = Vec::new();
    let mut subquery_keys = Vec::new();
    let mut key_types = Vec::new();
    for entry in join.conditions()? {
        let condition = match entry {
            JoinConditionEntry::Comparison(condition) => condition,
            JoinConditionEntry::Expression(_) => {
                return Err(OperatorError::Unsupported(
                    "delim joins with expression-form conditions are not supported".to_string(),
                ));
            }
        };
        if !matches!(
            condition.comparison,
            ExpressionType::COMPARE_EQUAL | ExpressionType::COMPARE_NOT_DISTINCT_FROM
        ) {
            return Err(OperatorError::Unsupported(format!(
                "Unsupported delim join comparison type: {:?}",
                condition.comparison
            )));
        }
        // A condition's left ref binds to the plan's left child, so on a
        // flipped join the outer key is the right ref.
        let (outer_ref, subquery_ref) = if flipped {
            (condition.right, condition.left)
        } else {
            (condition.left, condition.right)
        };
        let (outer_key, outer_key_type) = join_key_ref(Expression::from_handle(outer_ref)?)?;
        let (subquery_key, subquery_key_type) =
            join_key_ref(Expression::from_handle(subquery_ref)?)?;
        if outer_key_type != subquery_key_type {
            return Err(OperatorError::Unsupported(format!(
                "join key types differ: {outer_key_type:?} vs {subquery_key_type:?} \
                 (the planner casts both sides of a condition to a common type, so this plan \
                 shape is unexpected)"
            )));
        }
        outer_keys.push(outer_key);
        subquery_keys.push(subquery_key);
        key_types.push(outer_key_type);
    }
    if key_types.is_empty() {
        return Err(OperatorError::Unsupported(
            "joins must have at least one equality condition".to_string(),
        ));
    }

    let subquery_types = subquery_side.output_types()?;
    let subquery_nullability = subquery_side.output_nullability();
    let left_map: Vec<usize> = join.left_projection_map()?;
    let right_map: Vec<usize> = join.right_projection_map()?;
    let (outer_map, subquery_map) = if flipped {
        (right_map, left_map)
    } else {
        (left_map, right_map)
    };
    let outer_output: Vec<usize> = if outer_map.is_empty() {
        (0..outer_types.len()).collect()
    } else {
        outer_map
    };
    let subquery_output: Vec<usize> = if subquery_map.is_empty() {
        (0..subquery_types.len()).collect()
    } else {
        subquery_map
    };

    let read_outer_rows = || PlanNode {
        name: "DELIM_OUTER_ROWS".to_string(),
        inputs: Vec::new(),
        operator: Operator::CteScan(CteScan {
            cte_index: outer_rows_cte,
            types: outer_types.clone(),
            nullable: outer_nullability.clone(),
        }),
    };
    let column_types = |output: &[usize], types: &[Type], nullability: &[bool]| {
        output
            .iter()
            .map(|&i| (types[i].clone(), nullability[i]))
            .collect::<Vec<_>>()
    };

    let outer_column_types = column_types(&outer_output, &outer_types, &outer_nullability);
    let subquery_column_types =
        column_types(&subquery_output, &subquery_types, &subquery_nullability);
    // Natural orientation: the outer side probes the subquery side's (small,
    // one row per distinct key) answer table, LEFT preserving the outer side
    // through the probe-side outer mode. A flipped join swaps the sides, so
    // the subquery side probes the outer rows and the mirrored types land on
    // the build-side kinds. Either way a semi or anti join emits its
    // preserved side's columns alone, and the other side's projection map
    // must not be read (see `build_join`).
    let (join_inputs, join_operator) = if flipped {
        let kind = match join.join_type()? {
            JoinType::RIGHT => JoinKind::BuildOuter,
            JoinType::INNER => JoinKind::Inner,
            JoinType::RIGHT_SEMI => JoinKind::BuildSemi,
            JoinType::RIGHT_ANTI => JoinKind::BuildAnti,
            other => {
                return Err(OperatorError::Unsupported(format!(
                    "Unsupported flipped delim join type: {other:?}"
                )));
            }
        };
        let (probe_output, probe_column_types) = match kind {
            JoinKind::BuildSemi | JoinKind::BuildAnti => (Vec::new(), Vec::new()),
            _ => (subquery_output, subquery_column_types),
        };
        (
            vec![subquery_side, read_outer_rows()],
            Join {
                probe_keys: subquery_keys,
                build_keys: outer_keys,
                key_types,
                probe_output,
                build_output: outer_output,
                probe_column_types,
                build_column_types: outer_column_types,
                residual_filters: Vec::new(),
                residual_probe_columns: Vec::new(),
                residual_build_columns: Vec::new(),
                kind,
                produced_filters: Vec::new(),
            },
        )
    } else {
        let kind = match join.join_type()? {
            JoinType::LEFT => JoinKind::ProbeOuter,
            JoinType::INNER => JoinKind::Inner,
            JoinType::SEMI => JoinKind::ProbeSemi,
            JoinType::ANTI => JoinKind::ProbeAnti,
            other => {
                return Err(OperatorError::Unsupported(format!(
                    "Unsupported delim join type: {other:?}"
                )));
            }
        };
        let (build_output, build_column_types) = match kind {
            JoinKind::ProbeSemi | JoinKind::ProbeAnti => (Vec::new(), Vec::new()),
            _ => (subquery_output, subquery_column_types),
        };
        (
            vec![read_outer_rows(), subquery_side],
            Join {
                probe_keys: outer_keys,
                build_keys: subquery_keys,
                key_types,
                probe_output: outer_output,
                build_output,
                probe_column_types: outer_column_types,
                build_column_types,
                residual_filters: Vec::new(),
                residual_probe_columns: Vec::new(),
                residual_build_columns: Vec::new(),
                kind,
                produced_filters: Vec::new(),
            },
        )
    };
    let join_node = PlanNode {
        name: "DELIM_SUBQUERY_JOIN".to_string(),
        inputs: join_inputs,
        operator: Operator::Join(join_operator),
    };

    let dedup = PlanNode {
        name: "DELIM_DEDUP".to_string(),
        inputs: vec![read_outer_rows()],
        operator: Operator::Distinct(Distinct {
            keys: correlation_columns,
        }),
    };
    let key_feed = PlanNode {
        name: "DELIM_KEY_FEED".to_string(),
        inputs: vec![dedup, join_node],
        operator: Operator::Cte(Cte {
            cte_index: dedup_keys_cte,
            sites: delim_scan_sites,
        }),
    };
    Ok(PlanNode {
        name: op.name()?,
        inputs: vec![outer_side, key_feed],
        operator: Operator::Cte(Cte {
            cte_index: outer_rows_cte,
            sites: 2,
        }),
    })
}

/// Collapse DuckDB's late-materialization SEMI join into a pivot Materialize: the
/// narrow pipeline (RHS) runs as-is with its row-id column stripped, and the
/// materializer re-reads the LHS columns for the surviving rows.
fn build_late_materialization(
    op: LogicalOp<'_>,
    ctx: &mut BuildCtx,
) -> Result<PlanNode, OperatorError> {
    // Child 0 is the wide Get describing the fetch. Read its full output
    // description (storage index, type, pushed extract path) so a variant
    // field extract DuckDB pushed into the Get survives; fetching by bare
    // storage index would hand back the whole document column where the query
    // asked for a single field.
    let columns = match op.child(0)?.operator()? {
        DuckOperator::TableScan(wide_scan) => build_scan_columns(wide_scan.output_columns()?)?,
        DuckOperator::TableFunctionScan(wide_scan) => {
            build_scan_columns(wide_scan.output_columns()?)?
        }
        _ => {
            return Err(OperatorError::Unsupported(
                "late materialization without a Pivot-backed wide scan is not supported"
                    .to_string(),
            ));
        }
    };

    // Child 1 is the narrow pipeline; translate it normally.
    let mut child = build_node(op.child(1)?, ctx)?;
    // Flag the narrow scan to emit row-group metadata, and clone its table for the
    // Materialize so both read (a clone of) the same table.
    let table = prepare_narrow_scan(&mut child).ok_or_else(|| {
        OperatorError::Unsupported(
            "late materialization without a Pivot-backed narrow scan is not supported".to_string(),
        )
    })?;

    Ok(PlanNode {
        name: "Materialize".to_string(),
        inputs: vec![child],
        operator: Operator::Materialize(Materialize { table, columns }),
    })
}

/// Walk the late-mat narrow subtree to its Pivot-backed scan, flag it to emit
/// row-group metadata, and return a clone of its table (which the Materialize
/// re-reads). The narrow subtree is a straight projection/filter/limit chain
/// over one scan, so the first catalog input or table-function scan is the one.
fn prepare_narrow_scan(node: &mut PlanNode) -> Option<Box<dyn BoundTable>> {
    match &mut node.operator {
        Operator::Input(input) => {
            input.emit_row_group_metadata = true;
            return Some(input.table.clone_box());
        }
        Operator::TableFunctionScan(scan) => {
            return Some(scan.prepare_late_materialization());
        }
        _ => {}
    }
    prepare_narrow_scan(node.inputs.first_mut()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::Aggregate;

    fn variant_ref(column_idx: usize) -> Expression {
        Expression::Ref(Ref {
            column_idx,
            return_type: Type::Variant,
            name: None,
        })
    }

    /// A variant output is rendered even when the root isn't a projection:
    /// `output_types` sees through any operator, and the render wraps the
    /// whole plan rather than rewriting inside it.
    #[test]
    fn renders_a_variant_output_under_an_aggregate_root() {
        let plan = PlanNode {
            name: "aggregate".to_string(),
            inputs: Vec::new(),
            operator: Operator::Aggregate(Aggregate {
                groups: vec![variant_ref(0)],
                expressions: Vec::new(),
                output_limit: None,
            }),
        };

        let rendered = render_variant_outputs(plan).unwrap();

        assert_eq!(rendered.output_types().unwrap(), vec![Type::Utf8]);
        let Operator::Projection(projection) = &rendered.operator else {
            panic!("expected a render projection above the aggregate root");
        };
        assert!(matches!(
            &projection.projections[0],
            Expression::Cast(cast) if cast.target == Type::Utf8
        ));
    }

    /// A plan without variant outputs is returned untouched, with no extra
    /// projection.
    #[test]
    fn leaves_variant_free_outputs_alone() {
        let plan = PlanNode {
            name: "projection".to_string(),
            inputs: Vec::new(),
            operator: Operator::Projection(Projection {
                projections: vec![Expression::Ref(Ref {
                    column_idx: 0,
                    return_type: Type::Int64,
                    name: None,
                })],
            }),
        };

        let rendered = render_variant_outputs(plan).unwrap();

        assert_eq!(rendered.output_types().unwrap(), vec![Type::Int64]);
        assert!(rendered.inputs.is_empty(), "no wrapper was added");
    }
}
