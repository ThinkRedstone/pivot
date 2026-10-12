//! Constructs Pivot [`Operator`](crate::operator::Operator)s by walking DuckDB's
//! borrowed logical-operator handles. Each operator's `from_handle` is implemented
//! here (the walk in [`super`] dispatches to them), so the `crate::operator`
//! modules hold only Pivot IR (the operator types, their `Display`, and their
//! `compile`). Expression construction is the sibling
//! [`expression`](super::expression).

use std::any::Any;

use arrow_array::{Scalar, new_null_array};
use dispatch::RowDelivery;
use duckdb_planner::DuckDBTable;
use duckdb_planner::catalog_provider::OptionalTableWrapper;
use duckdb_planner::duckdb_bridge::duckdb_types::{LimitNodeType, OrderByNullType};
use duckdb_planner::handle::{
    Aggregate as AggregateView, BridgeError, ChunkGet as ChunkGetView, Compact as CompactView,
    CopyFromStdin as CopyFromStdinView, CreateSchema as CreateSchemaView,
    CreateTable as CreateTableView, CreateUser as CreateUserView, Distinct as DistinctView,
    Drop as DropView, Filter as FilterView, Insert as InsertView, Limit as LimitView,
    OrderBy as OrderByView, OrderKey, Projection as ProjectionView, Reset as ResetView,
    Set as SetView, TableFunctionScan as TableFunctionScanView, TableScan as TableScanView,
    TopN as TopNView, Values as ValuesView,
};

use super::{BuildCtx, build_scan_columns};
use crate::catalog::{
    BoundTable, Column, CreateSchemaRequest, CreateTableRequest, DropTableRequest,
    DuckDBTableAdapter,
};
use crate::dynamic_filter::JoinFilterScanInfo;
use crate::expression::{Error as ExpressionError, Expression, Ref};
use crate::operator::{
    Aggregate, Compact, CopyFormat, CopyFromStdin, CreateSchema, CreateTable, CreateUser, Distinct,
    DropTable, Error as OperatorError, Filter, Input, Insert, Limit, OrderBy, OrderByNode,
    Projection, SetVariable, TableFunctionScan, TopN, Values,
};
use crate::types::{build_scalar_value, physical_arrow_type, type_from_logical};

impl Projection {
    pub(crate) fn from_handle(view: ProjectionView<'_>) -> Result<Projection, OperatorError> {
        Ok(Projection {
            projections: view
                .exprs()?
                .into_iter()
                .map(Expression::from_handle)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

impl Values {
    pub(crate) fn from_handle(view: ValuesView<'_>) -> Result<Values, OperatorError> {
        let rows = (0..view.row_count()?)
            .map(|row| {
                (0..view.column_count()?)
                    .map(|column| {
                        Expression::from_handle(view.expression(row, column)?)
                            .map_err(OperatorError::from)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Values { rows })
    }

    /// A CHUNK_GET, DuckDB's scan of an in-memory constant chunk (what its
    /// optimizer rewrites a long `IN` list into), becomes a VALUES source
    /// whose cells are the chunk's constants.
    pub(crate) fn from_chunk_get(view: ChunkGetView<'_>) -> Result<Values, OperatorError> {
        let rows = (0..view.row_count()?)
            .map(|row| {
                (0..view.column_count()?)
                    .map(|column| {
                        let cell = view.value(column, row)?;
                        Ok(Expression::Constant(build_scalar_value(cell)?))
                    })
                    .collect::<Result<Vec<_>, OperatorError>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Values { rows })
    }
}

impl Insert {
    /// Build the insert, along with the projection that reshapes its child when
    /// the statement named its target columns (see
    /// [`build_column_list_projection`]). The walk in [`super`] puts that
    /// projection between the two.
    pub(crate) fn from_handle(
        view: InsertView<'_>,
    ) -> Result<(Insert, Option<Projection>), OperatorError> {
        if view.returns_rows()? {
            return Err(OperatorError::Unsupported(
                "INSERT ... RETURNING is not supported".to_string(),
            ));
        }
        let column_map = view.column_map()?;
        let table = bind_table(*view.take_table()?);
        let column_list = build_column_list_projection(&column_map, &table.columns())?;
        Ok((Insert { table }, column_list))
    }
}

/// Reshape the child of an `INSERT INTO t (b, a) ...`, which emits only the
/// listed columns in the order they were written, into the table's full column
/// list: every table column reads the child column that fills it, and a column
/// the statement leaves out becomes a NULL of that column's type. Unfilled can
/// only mean NULL here, since the catalog carries no DEFAULT values.
///
/// `None` when the statement named no columns: its child already emits the
/// table's columns in order, which is what the insert path expects.
fn build_column_list_projection(
    column_map: &[Option<usize>],
    columns: &[Column],
) -> Result<Option<Projection>, OperatorError> {
    if column_map.is_empty() {
        return Ok(None);
    }
    if column_map.len() != columns.len() {
        return Err(OperatorError::InvalidStatement(format!(
            "INSERT names {} columns of a table that has {}",
            column_map.len(),
            columns.len()
        )));
    }
    let listed = column_map.iter().flatten().count();
    if listed == 0 {
        // The column map is all-unfilled only for DEFAULT VALUES: a written
        // column list always fills at least one column.
        return Err(OperatorError::Unsupported(
            "INSERT ... DEFAULT VALUES is not supported".to_string(),
        ));
    }
    // The binder resolved the names, so each source is a distinct child column;
    // check anyway, so a disagreement fails loudly instead of scrambling
    // columns or reading past the child's width.
    let mut filled = vec![false; listed];
    let projections = columns
        .iter()
        .zip(column_map)
        .map(|(column, source)| {
            let Some(&column_idx) = source.as_ref() else {
                let null = new_null_array(&physical_arrow_type(&column.col_type), 1);
                return Ok(Expression::Constant(Scalar::new(null)));
            };
            let taken = filled.get_mut(column_idx).ok_or_else(|| {
                OperatorError::InvalidStatement(format!(
                    "INSERT column \"{}\" reads value {column_idx} of {listed}",
                    column.name
                ))
            })?;
            if std::mem::replace(taken, true) {
                return Err(OperatorError::InvalidStatement(format!(
                    "INSERT fills more than one column from value {column_idx}"
                )));
            }
            Ok(Expression::Ref(Ref {
                column_idx,
                return_type: column.col_type.clone(),
                name: Some(column.name.clone()),
            }))
        })
        .collect::<Result<Vec<_>, OperatorError>>()?;
    Ok(Some(Projection { projections }))
}

impl Filter {
    pub(crate) fn from_handle(view: FilterView<'_>) -> Result<Filter, OperatorError> {
        Ok(Filter {
            conditions: view
                .exprs()?
                .into_iter()
                .map(Expression::from_handle)
                .collect::<Result<Vec<_>, _>>()?,
            // `annotate_filter_delivery` revisits this once the tree is built
            // and the operator below this filter is known.
            delivery: RowDelivery::Coalesced,
        })
    }
}

impl Aggregate {
    pub(crate) fn from_handle(view: AggregateView<'_>) -> Result<Aggregate, OperatorError> {
        Ok(Aggregate {
            output_limit: None,
            groups: view
                .groups()?
                .into_iter()
                .map(Expression::from_handle)
                .collect::<Result<Vec<_>, _>>()?,
            expressions: view
                .expressions()?
                .into_iter()
                .map(Expression::from_handle)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }
}

impl Distinct {
    /// Build a [`Distinct`] from a `SELECT DISTINCT` node. The node's declared
    /// output is the child's columns unchanged, while the dedup operator emits
    /// its key columns in key order; the two agree only when the targets are
    /// exactly the child's columns, so anything else (a `DISTINCT ON`, whose
    /// non-target columns take an arbitrary row's values) is rejected. A key
    /// tuple dedups the same in any order, so sorting the keys into column
    /// order makes the output the child's layout.
    pub(crate) fn from_handle(
        view: DistinctView<'_>,
        child_width: usize,
    ) -> Result<Distinct, OperatorError> {
        if view.is_distinct_on()? {
            return Err(OperatorError::Unsupported(
                "DISTINCT ON is not supported".to_string(),
            ));
        }
        let mut keys = view
            .targets()?
            .into_iter()
            .map(|target| match Expression::from_handle(target)? {
                Expression::Ref(r) => Ok((r.column_idx, r.return_type)),
                other => Err(OperatorError::Unsupported(format!(
                    "DISTINCT over a computed expression is not supported: {other}"
                ))),
            })
            .collect::<Result<Vec<_>, OperatorError>>()?;
        keys.sort_unstable_by_key(|(column, _)| *column);
        let covers_child = keys.len() == child_width
            && keys.iter().enumerate().all(|(i, (column, _))| i == *column);
        if !covers_child {
            return Err(OperatorError::Unsupported(
                "DISTINCT over a subset of the output columns is not supported".to_string(),
            ));
        }
        Ok(Distinct { keys })
    }
}

impl OrderBy {
    pub(crate) fn from_handle(view: OrderByView<'_>) -> Result<OrderBy, OperatorError> {
        Ok(OrderBy {
            order_bys: build_orders(view.keys()?)?,
        })
    }
}

impl TopN {
    pub(super) fn from_handle(
        view: TopNView<'_>,
        ctx: &mut BuildCtx,
    ) -> Result<TopN, OperatorError> {
        let produces_dynamic_filter = match view.dynamic_filter()? {
            Some(df) => Some(ctx.dynamic_filter(df)?),
            None => None,
        };
        Ok(TopN {
            order_bys: build_orders(view.keys()?)?,
            limit: view.limit()?,
            offset: view.offset()?,
            produces_dynamic_filter,
        })
    }
}

impl Limit {
    pub(crate) fn from_handle(view: LimitView<'_>) -> Result<Limit, OperatorError> {
        Ok(Limit {
            limit: limit_bound(view.value_kind()?, || view.value(), "value")?,
            offset: limit_bound(view.offset_kind()?, || view.offset(), "offset")?.unwrap_or(0),
        })
    }
}

impl Input {
    pub(super) fn from_handle(
        scan: TableScanView<'_>,
        ctx: &mut BuildCtx,
    ) -> Result<Input, OperatorError> {
        let join_filter_info = match scan.join_filter_set_id()? {
            Some(filter_set_id) => Some(JoinFilterScanInfo {
                filter_set_id,
                proj_to_storage: scan.storage_columns()?,
            }),
            None => None,
        };
        Ok(Input {
            table: bind_table(*scan.take_table()?),
            columns: build_scan_columns(scan.output_columns()?)?,
            dynamic_filters: scan
                .dynamic_filters()?
                .into_iter()
                .map(|df| ctx.dynamic_filter(df))
                .collect::<Result<Vec<_>, _>>()?,
            key_bitset_filters: Vec::new(),
            duckdb_table_binding_index: Some(scan.table_index()?),
            join_filter_info,
            emit_row_group_metadata: false,
        })
    }
}

impl TableFunctionScan {
    pub(super) fn from_handle(
        view: TableFunctionScanView<'_>,
        ctx: &mut BuildCtx,
    ) -> Result<TableFunctionScan, OperatorError> {
        // Only positional parameters are supported; reject named parameters rather
        // than silently drop a bound argument.
        if view.has_named_params()? {
            return Err(OperatorError::Unsupported(format!(
                "table function {} with named parameters is not supported",
                view.function_name()?
            )));
        }
        if !view.has_bound_table()? {
            return Err(OperatorError::Unsupported(format!(
                "table function {} was not registered by Pivot",
                view.function_name()?
            )));
        }
        let bound_table = bind_table(*view.bound_table()?);
        Ok(TableFunctionScan::new(
            view.function_name()?,
            view.params()?,
            build_scan_columns(view.output_columns()?)?,
            bound_table,
            view.dynamic_filters()?
                .into_iter()
                .map(|df| ctx.dynamic_filter(df))
                .collect::<Result<Vec<_>, _>>()?,
        ))
    }
}

impl CreateSchema {
    pub(crate) fn from_handle(view: CreateSchemaView<'_>) -> Result<CreateSchema, OperatorError> {
        Ok(CreateSchema {
            request: CreateSchemaRequest {
                datastore_name: view.datastore()?,
                name: view.name()?,
                if_not_exists: view.if_not_exists()?,
            },
            or_replace: view.or_replace()?,
        })
    }
}

impl DropTable {
    pub(crate) fn from_handle(view: DropView<'_>) -> Result<DropTable, OperatorError> {
        Ok(DropTable {
            request: DropTableRequest {
                datastore_name: view.datastore()?,
                schema_name: view.schema()?,
                name: view.name()?,
                if_exists: view.if_exists()?,
            },
            cascade: view.cascade()?,
        })
    }
}

impl CreateTable {
    pub(crate) fn from_handle(view: CreateTableView<'_>) -> Result<CreateTable, OperatorError> {
        Ok(CreateTable {
            request: CreateTableRequest {
                datastore_name: view.datastore()?,
                schema_name: view.schema()?,
                name: view.name()?,
                columns: view
                    .columns()?
                    .into_iter()
                    .map(|(name, col_type)| {
                        Ok(Column {
                            name,
                            col_type: type_from_logical(col_type)?,
                        })
                    })
                    .collect::<Result<Vec<_>, OperatorError>>()?,
                options: view.options()?.into_iter().collect(),
                if_not_exists: view.if_not_exists()?,
            },
            or_replace: view.or_replace()?,
            temporary: view.temporary()?,
            has_query: view.has_query()?,
            constraint_count: view.constraint_count()?,
        })
    }
}

impl SetVariable {
    pub(crate) fn from_set(view: SetView<'_>) -> Result<SetVariable, BridgeError> {
        Ok(SetVariable {
            name: view.name()?,
            value: Some(view.value()?),
        })
    }

    /// `RESET name` is modelled as a `SET` with no value (the consumer reads "no
    /// value" as "off / default").
    pub(crate) fn from_reset(view: ResetView<'_>) -> Result<SetVariable, BridgeError> {
        Ok(SetVariable {
            name: view.name()?,
            value: None,
        })
    }
}

impl Compact {
    pub(crate) fn from_handle(view: CompactView<'_>) -> Result<Compact, BridgeError> {
        Ok(Compact {
            datastore: view.datastore()?,
            schema: view.schema()?,
            table: view.table()?,
            final_sweep: view.final_sweep()?,
        })
    }
}

impl CopyFromStdin {
    pub(crate) fn from_handle(view: CopyFromStdinView<'_>) -> Result<CopyFromStdin, OperatorError> {
        let table = bind_table(*view.take_table()?);
        let columns = view.column_indexes()?;
        // Formats are case-insensitive; canonicalize before interpreting.
        let format = view.format()?.map(|format| format.to_lowercase());
        let format = CopyFormat::resolve(format.as_deref(), &view.options()?)
            .map_err(OperatorError::InvalidStatement)?;

        Ok(CopyFromStdin {
            table,
            columns,
            format,
        })
    }
}

impl CreateUser {
    pub(crate) fn from_handle(view: CreateUserView<'_>) -> Result<CreateUser, BridgeError> {
        Ok(CreateUser {
            name: view.name()?,
            password: view.password()?,
        })
    }
}

/// Lower a sequence of handle sort keys into Pivot [`OrderByNode`]s. Shared by the
/// `OrderBy` and `TopN` constructors.
fn build_orders(keys: Vec<OrderKey<'_>>) -> Result<Vec<OrderByNode>, ExpressionError> {
    keys.into_iter()
        .map(|key| {
            let nulls_first = match key.null_order {
                OrderByNullType::NULLS_FIRST => true,
                OrderByNullType::NULLS_LAST => false,
                unresolved => return Err(ExpressionError::UnsupportedNullOrder(unresolved)),
            };
            Ok(OrderByNode {
                direction: key.direction.into(),
                nulls_first,
                expression: Expression::from_handle(key.expression)?,
            })
        })
        .collect()
}

/// Resolve a bound catalog entry into the Pivot [`BoundTable`] it wraps.
fn bind_table(wrapper: OptionalTableWrapper) -> Box<dyn BoundTable> {
    let duck: Box<dyn DuckDBTable> = wrapper.table.expect("planner returned an unbound table");
    let any: Box<dyn Any> = duck;
    let adapter: Box<DuckDBTableAdapter> = any
        .downcast::<DuckDBTableAdapter>()
        .expect("scan table should be the planner's table adapter");
    adapter.table
}

/// Interpret a `LIMIT`/`OFFSET` bound's [`LimitNodeType`] into an `Option<usize>`:
/// an unset bound is unbounded (`None`), a constant is the row count, and a
/// percentage/expression bound is rejected (pivot only handles a fixed row count).
///
/// `value` is read lazily: DuckDB's constant accessor throws for a non-constant
/// bound, so it must only be called once the kind is known to be a constant.
fn limit_bound(
    kind: LimitNodeType,
    value: impl FnOnce() -> Result<usize, BridgeError>,
    what: &str,
) -> Result<Option<usize>, OperatorError> {
    match kind {
        LimitNodeType::UNSET => Ok(None),
        LimitNodeType::CONSTANT_VALUE => Ok(Some(value()?)),
        _ => Err(OperatorError::Unsupported(format!(
            "Unsupported non-constant LIMIT {what}"
        ))),
    }
}
