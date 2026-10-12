//! Safe borrowed handles over DuckDB's live plan objects.
//!
//! `extract_plan` hands back a [`Plan`] that owns DuckDB's resolved
//! `LogicalOperator` tree. Consumers walk that tree through [`LogicalOp`] and
//! [`Expr`], borrowed handles tethered by their lifetime to the owning [`Plan`]
//! (like a `rusqlite::Row` borrowed from its `Statement`). Every `unsafe` enum
//! decode and raw FFI accessor stays behind these methods, so callers read the
//! plan in safe, typed Rust without ever touching a discriminant byte or an
//! opaque pointer.
//!
//! Every accessor is fallible: the C++ side reads fields off DuckDB's own
//! objects and throws a DuckDB exception when a shape assumption does not hold
//! (a cast to the wrong subtype, a value of an unexpected physical type). The
//! bridge converts those throws into [`BridgeError`]s, so a wrong assumption
//! fails the one query instead of terminating the process.
//!
//! [`LogicalOp`] carries the accessors shared by every operator (`op_type`,
//! `name`, `children`); the kind-specific ones live on the typed views of
//! [`Operator`], which [`LogicalOp::operator`] dispatches to. The views are
//! zero-cost borrowed references, not owned data: there is no intermediate
//! representation here, and materializing the tree into owned structs is the
//! consumer's job.

use cxx::UniquePtr;

use crate::catalog_provider::OptionalTableWrapper;
use crate::duckdb_bridge::duckdb_types::{
    ExpressionType, JoinType, LimitNodeType, LogicalOperatorType, LogicalTypeId, OrderByNullType,
    OrderType, TransactionType,
};
use crate::duckdb_bridge::ffi;
use crate::types::{BoundLogicalType, ScalarValue};

/// A DuckDB C++ exception caught at the FFI boundary while reading plan
/// objects. Reaching one means a shape assumption in the plan walk was wrong;
/// the query fails with this message instead of the process aborting.
#[derive(Debug, thiserror::Error)]
#[error("plan walk failed: {0}")]
pub struct BridgeError(pub String);

impl From<cxx::Exception> for BridgeError {
    fn from(exception: cxx::Exception) -> Self {
        BridgeError(exception.what().to_string())
    }
}

type Result<T> = std::result::Result<T, BridgeError>;

/// DuckDB's virtual row-id column identifier, used by late-materialization
/// row-id stripping to recognise the threaded-up row-id column.
pub fn rowid_column_id() -> usize {
    ffi::rowid_column_id().expect("rowid_column_id returns a constant")
}

/// Rebuild a 128-bit integer from the two halves the FFI carries it as.
fn i128_from_halves(hi: i64, lo: u64) -> i128 {
    ((hi as i128) << 64) | (lo as i128)
}

/// Decode the FFI type struct into a [`BoundLogicalType`].
fn bound_type_from(raw: ffi::BridgeLogicalType) -> BoundLogicalType {
    BoundLogicalType::from_bridge(raw)
}

/// Decode a DuckDB [`Value`](ffi::Value) into a typed [`ScalarValue`]: read its
/// type, then the matching typed accessor. Shared by query constants and
/// table-function arguments. A type the bridge doesn't decode lands in
/// [`ScalarValue::Other`].
///
/// The NULL check comes first: a NULL value carries a full logical type but no
/// payload, and the typed accessors are only defined on a value that has one.
pub(crate) fn scalar_from_value(v: &ffi::Value) -> Result<ScalarValue> {
    use LogicalTypeId as L;
    let value_type = bound_type_from(ffi::value_type(v)?);
    if ffi::value_is_null(v)? {
        return Ok(ScalarValue::Null(value_type));
    }
    Ok(match value_type.id {
        L::BOOLEAN => ScalarValue::Boolean(ffi::value_bool(v)?),
        L::TINYINT => ScalarValue::Int8(ffi::value_i8(v)?),
        L::SMALLINT => ScalarValue::Int16(ffi::value_i16(v)?),
        L::INTEGER => ScalarValue::Int32(ffi::value_i32(v)?),
        L::BIGINT => ScalarValue::Int64(ffi::value_i64(v)?),
        L::UTINYINT => ScalarValue::UInt8(ffi::value_u8(v)?),
        L::USMALLINT => ScalarValue::UInt16(ffi::value_u16(v)?),
        L::UINTEGER => ScalarValue::UInt32(ffi::value_u32(v)?),
        L::UBIGINT => ScalarValue::UInt64(ffi::value_u64(v)?),
        L::HUGEINT => {
            let raw = ffi::value_hugeint(v)?;
            ScalarValue::Int128(i128_from_halves(raw.hi, raw.lo))
        }
        L::FLOAT => ScalarValue::Float32(ffi::value_f32(v)?),
        L::DOUBLE => ScalarValue::Float64(ffi::value_f64(v)?),
        L::DECIMAL => {
            let raw = ffi::value_decimal(v)?;
            ScalarValue::Decimal {
                value: i128_from_halves(raw.hi, raw.lo),
                width: raw.width,
                scale: raw.scale,
            }
        }
        L::VARCHAR => ScalarValue::Utf8(ffi::value_string(v)?),
        // A variant Value casts to VARCHAR as its raw text (DuckDB's
        // variant-to-VARCHAR is a direct string conversion, not a JSON render),
        // so `value_string` recovers the document the variant was built from.
        L::VARIANT => ScalarValue::Variant(ffi::value_string(v)?),
        L::DATE => ScalarValue::Date(ffi::value_date(v)?),
        L::TIMESTAMP => ScalarValue::Timestamp(ffi::value_timestamp(v)?),
        L::TIMESTAMP_TZ => ScalarValue::TimestampTz(ffi::value_timestamp_tz(v)?),
        L::INTERVAL => ScalarValue::Interval {
            months: ffi::value_interval_months(v)?,
            days: ffi::value_interval_days(v)?,
            micros: ffi::value_interval_micros(v)?,
        },
        other => ScalarValue::Other(other),
    })
}

/// Owns DuckDB's resolved plan tree (kept alive while it is walked) plus the
/// binder-resolved client column names. Hand out the root with [`Plan::root`].
pub struct Plan {
    handle: UniquePtr<ffi::PlanHandle>,
    output_names: Vec<String>,
    requires_rebind: bool,
}

impl Plan {
    pub(crate) fn new(
        handle: UniquePtr<ffi::PlanHandle>,
        output_names: Vec<String>,
        requires_rebind: bool,
    ) -> Self {
        Self {
            handle,
            output_names,
            requires_rebind,
        }
    }

    /// The root operator of the plan, borrowed for the lifetime of this [`Plan`].
    pub fn root(&self) -> Result<LogicalOp<'_>> {
        Ok(LogicalOp {
            raw: ffi::plan_root(&self.handle)?,
        })
    }

    /// The result column names DuckDB resolved for the client, in output order
    /// (e.g. `["hour", "count_star()"]`). Empty when the bridge could not recover
    /// them.
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    /// Whether DuckDB requires this statement to be rebound before a later
    /// execution because optimization embedded query-dependent state.
    pub fn requires_rebind(&self) -> bool {
        self.requires_rebind
    }

    /// Consume the plan, returning the resolved output names.
    pub fn into_output_names(self) -> Vec<String> {
        self.output_names
    }
}

/// A borrowed DuckDB `LogicalOperator`, tethered to the [`Plan`] that owns it.
/// Cheap to copy (it is one reference). Carries the accessors common to every
/// operator; reach the kind-specific ones through [`operator`](Self::operator).
#[derive(Clone, Copy)]
pub struct LogicalOp<'plan> {
    raw: &'plan ffi::LogicalOperator,
}

impl<'plan> LogicalOp<'plan> {
    /// The operator's [`LogicalOperatorType`].
    pub fn op_type(self) -> Result<LogicalOperatorType> {
        Ok(LogicalOperatorType::from_u8(ffi::lo_type(self.raw)?))
    }

    /// The operator's display name (`LogicalOperator::GetName`).
    pub fn name(self) -> Result<String> {
        Ok(ffi::lo_name(self.raw)?)
    }

    pub fn child_count(self) -> Result<usize> {
        Ok(ffi::lo_child_count(self.raw)?)
    }

    pub fn child(self, index: usize) -> Result<LogicalOp<'plan>> {
        Ok(LogicalOp {
            raw: ffi::lo_child(self.raw, index)?,
        })
    }

    pub fn children(self) -> Result<Vec<LogicalOp<'plan>>> {
        (0..self.child_count()?).map(|i| self.child(i)).collect()
    }

    /// Dispatch on the operator's [`LogicalOperatorType`] into a typed [`Operator`]
    /// view that exposes only that operator's accessors. A `LOGICAL_GET` splits
    /// into [`Operator::TableScan`] / [`Operator::TableFunctionScan`] by whether it
    /// reads a base table; any kind the consumer doesn't handle is
    /// [`Operator::Unsupported`].
    pub fn operator(self) -> Result<Operator<'plan>> {
        use LogicalOperatorType as L;
        Ok(match self.op_type()? {
            L::LOGICAL_PROJECTION => Operator::Projection(Projection { raw: self.raw }),
            L::LOGICAL_EXPRESSION_GET => Operator::Values(Values { raw: self.raw }),
            L::LOGICAL_INSERT => Operator::Insert(Insert { raw: self.raw }),
            L::LOGICAL_FILTER => Operator::Filter(Filter { raw: self.raw }),
            L::LOGICAL_AGGREGATE_AND_GROUP_BY => Operator::Aggregate(Aggregate { raw: self.raw }),
            L::LOGICAL_DISTINCT => Operator::Distinct(Distinct { raw: self.raw }),
            L::LOGICAL_ORDER_BY => Operator::OrderBy(OrderBy { raw: self.raw }),
            L::LOGICAL_TOP_N => Operator::TopN(TopN { raw: self.raw }),
            L::LOGICAL_LIMIT => Operator::Limit(Limit { raw: self.raw }),
            L::LOGICAL_GET if ffi::lo_get_has_table(self.raw)? => {
                Operator::TableScan(TableScan { raw: self.raw })
            }
            L::LOGICAL_GET => Operator::TableFunctionScan(TableFunctionScan { raw: self.raw }),
            L::LOGICAL_CREATE_TABLE => Operator::CreateTable(CreateTable { raw: self.raw }),
            L::LOGICAL_CREATE_SCHEMA => Operator::CreateSchema(CreateSchema { raw: self.raw }),
            L::LOGICAL_DROP => Operator::Drop(Drop { raw: self.raw }),
            L::LOGICAL_SET => Operator::Set(Set { raw: self.raw }),
            L::LOGICAL_RESET => Operator::Reset(Reset { raw: self.raw }),
            L::LOGICAL_TRANSACTION => Operator::Transaction(Transaction { raw: self.raw }),
            L::LOGICAL_COMPACT => Operator::Compact(Compact { raw: self.raw }),
            L::LOGICAL_VACUUM => {
                ffi::lo_vacuum_validate(self.raw)?;
                Operator::Vacuum
            }
            L::LOGICAL_COPY_FROM_STDIN => Operator::CopyFromStdin(CopyFromStdin { raw: self.raw }),
            L::LOGICAL_CREATE_USER => Operator::CreateUser(CreateUser { raw: self.raw }),
            L::LOGICAL_COMPARISON_JOIN => {
                Operator::ComparisonJoin(ComparisonJoin { raw: self.raw })
            }
            L::LOGICAL_PIVOT_MATERIALIZE => Operator::PivotMaterialize,
            L::LOGICAL_DELIM_JOIN => Operator::DelimJoin(DelimJoin { raw: self.raw }),
            L::LOGICAL_DELIM_GET => Operator::DelimGet(DelimGet { raw: self.raw }),
            L::LOGICAL_MATERIALIZED_CTE => {
                Operator::MaterializedCte(MaterializedCte { raw: self.raw })
            }
            L::LOGICAL_CTE_REF => Operator::CteRef(CteRef { raw: self.raw }),
            L::LOGICAL_CHUNK_GET => Operator::ChunkGet(ChunkGet { raw: self.raw }),
            L::LOGICAL_DUMMY_SCAN => Operator::DummyScan,
            L::LOGICAL_EXPLAIN => Operator::Explain(Explain { raw: self.raw }),
            _ => Operator::Unsupported,
        })
    }
}

/// A borrowed, typed view of a [`LogicalOp`], discriminated by operator kind.
/// Each variant exposes only the accessors valid for that kind, so a projection
/// accessor can't be called on a filter. Obtained via [`LogicalOp::operator`].
/// Every variant is a zero-cost borrowed view (one reference), not owned data.
#[derive(Clone, Copy)]
pub enum Operator<'plan> {
    Projection(Projection<'plan>),
    Values(Values<'plan>),
    Insert(Insert<'plan>),
    Filter(Filter<'plan>),
    Aggregate(Aggregate<'plan>),
    /// A `DISTINCT` over the child's rows.
    Distinct(Distinct<'plan>),
    OrderBy(OrderBy<'plan>),
    TopN(TopN<'plan>),
    Limit(Limit<'plan>),
    /// A base-table scan (`LOGICAL_GET` over a table).
    TableScan(TableScan<'plan>),
    /// A scan over a table-valued function (`LOGICAL_GET` with no base table).
    TableFunctionScan(TableFunctionScan<'plan>),
    CreateTable(CreateTable<'plan>),
    CreateSchema(CreateSchema<'plan>),
    /// `DROP <kind> name` of any catalog entry kind; the consumer checks the
    /// kind and supports only table drops.
    Drop(Drop<'plan>),
    /// `SET name = value`.
    Set(Set<'plan>),
    /// `RESET name`.
    Reset(Reset<'plan>),
    /// `BEGIN`, `COMMIT` or `ROLLBACK`.
    Transaction(Transaction<'plan>),
    /// `COMPACT <table> [FINAL]`.
    Compact(Compact<'plan>),
    /// `VACUUM`: one sweep of the current datastore.
    Vacuum,
    /// `COPY <table> [(columns)] FROM STDIN [WITH (...)]`.
    CopyFromStdin(CopyFromStdin<'plan>),
    /// `CREATE USER <name> [PASSWORD '<password>']`.
    CreateUser(CreateUser<'plan>),
    /// A user IN/EXISTS comparison join.
    ComparisonJoin(ComparisonJoin<'plan>),
    /// The late-materialization rewrite's fetch: child 0 is the wide get
    /// describing the columns to fetch for the rows surviving child 1 (the
    /// narrow pipeline).
    PivotMaterialize,
    /// A comparison join that additionally de-duplicates correlation columns
    /// from one side and publishes them to the [`DelimGet`]s under the other.
    DelimJoin(DelimJoin<'plan>),
    /// A scan of the distinct correlation values its enclosing [`DelimJoin`]
    /// de-duplicated.
    DelimGet(DelimGet<'plan>),
    /// A CTE: its definition, then the query reading it.
    MaterializedCte(MaterializedCte<'plan>),
    /// One place a CTE's rows are read.
    CteRef(CteRef<'plan>),
    /// A scan of an in-memory constant chunk (a long `IN` list's rewrite).
    ChunkGet(ChunkGet<'plan>),
    /// The single-row source under a `FROM`-less `SELECT`.
    DummyScan,
    /// `EXPLAIN [ANALYZE] <query>`; the consumer checks
    /// [`Explain::is_analyze`] and supports only plain `EXPLAIN`.
    Explain(Explain<'plan>),
    /// Any operator type the consumer doesn't handle.
    Unsupported,
}

/// Define zero-cost borrowed view newtypes over a raw FFI object: each wraps one
/// `&'plan $raw` reference and is `Copy`. The doc comment on each name carries
/// through. Used for both the [`Operator`] and [`Expression`] views.
macro_rules! define_handles {
    ($raw:ty; $($(#[$attr:meta])* $name:ident),+ $(,)?) => {
        $(
            $(#[$attr])*
            #[derive(Clone, Copy)]
            pub struct $name<'plan> {
                raw: &'plan $raw,
            }
        )+
    };
}

define_handles! { ffi::LogicalOperator;
    /// A `LogicalProjection`: a list of output expressions.
    Projection,
    /// A `LogicalExpressionGet`: rows of bound expressions from `VALUES`.
    Values,
    /// A `LogicalInsert`: the target table above its value-producing child.
    Insert,
    /// A `LogicalFilter`: boolean conditions plus an optional projection map.
    Filter,
    /// A `LogicalAggregate`: GROUP BY keys plus aggregate expressions.
    Aggregate,
    /// A `LogicalDistinct`: dedup of the child's rows on the distinct targets.
    Distinct,
    /// A `LogicalOrder`: a list of sort keys.
    OrderBy,
    /// A `LogicalTopN`: ORDER BY + LIMIT, optionally a dynamic-filter producer.
    TopN,
    /// A `LogicalLimit`: a bare `LIMIT`/`OFFSET` with no ORDER BY.
    Limit,
    /// A `LogicalGet` over a base table.
    TableScan,
    /// A `LogicalGet` over a table-valued function.
    TableFunctionScan,
    /// A `LogicalCreateTable` with an explicit column list.
    CreateTable,
    /// A `LogicalCreate` for `CREATE SCHEMA`.
    CreateSchema,
    /// A `LogicalSimple` for `DROP <kind> name` carrying DuckDB's `DropInfo`.
    Drop,
    /// A `LogicalSet`: `SET name = value`.
    Set,
    /// A `LogicalReset`: `RESET name`.
    Reset,
    /// A `LogicalSimple` for `BEGIN`/`COMMIT`/`ROLLBACK` carrying DuckDB's
    /// `TransactionInfo`.
    Transaction,
    /// A `LogicalCompact`: `COMPACT <table> [FINAL]`.
    Compact,
    /// A `LogicalCopyFromStdin`: `COPY <table> [(columns)] FROM STDIN [WITH (...)]`.
    CopyFromStdin,
    /// A `LogicalCreateUser`: `CREATE USER <name> [PASSWORD '<password>']`.
    CreateUser,
    /// A `LogicalComparisonJoin`.
    ComparisonJoin,
    /// A `LogicalComparisonJoin` whose operator type is `LOGICAL_DELIM_JOIN`.
    DelimJoin,
    /// A `LogicalDelimGet`: a scan of a delim join's de-duplicated values.
    DelimGet,
    /// A `LogicalMaterializedCTE`: the CTE's definition above the query using it.
    MaterializedCte,
    /// A `LogicalCTERef`: one place a CTE's rows are read.
    CteRef,
    /// A `LogicalColumnDataGet` (CHUNK_GET): a scan of an in-memory constant
    /// chunk, e.g. what DuckDB rewrites a long `IN` list into.
    ChunkGet,
    /// A `LogicalExplain`: `EXPLAIN [ANALYZE] <query>`.
    Explain,
}

impl<'plan> MaterializedCte<'plan> {
    /// The index this CTE publishes its rows under, which every
    /// [`CteRef`] reading it carries.
    pub fn cte_index(self) -> Result<usize> {
        Ok(ffi::lo_cte_table_index(self.raw)?)
    }
}

impl<'plan> CteRef<'plan> {
    /// The CTE these rows come from.
    pub fn cte_index(self) -> Result<usize> {
        Ok(ffi::lo_cte_ref_index(self.raw)?)
    }
}

impl<'plan> Projection<'plan> {
    /// The projection's output expressions.
    pub fn exprs(self) -> Result<Vec<Expr<'plan>>> {
        (0..ffi::lo_projection_expr_count(self.raw)?)
            .map(|i| {
                Ok(Expr {
                    raw: ffi::lo_projection_expr(self.raw, i)?,
                })
            })
            .collect()
    }
}

impl<'plan> Values<'plan> {
    pub fn row_count(self) -> Result<usize> {
        Ok(ffi::lo_values_row_count(self.raw)?)
    }

    pub fn column_count(self) -> Result<usize> {
        Ok(ffi::lo_values_column_count(self.raw)?)
    }

    pub fn expression(self, row: usize, column: usize) -> Result<Expr<'plan>> {
        Ok(Expr {
            raw: ffi::lo_values_expr(self.raw, row, column)?,
        })
    }
}

impl<'plan> ChunkGet<'plan> {
    pub fn row_count(self) -> Result<usize> {
        Ok(ffi::lo_chunk_get_row_count(self.raw)?)
    }

    pub fn column_count(self) -> Result<usize> {
        Ok(ffi::lo_chunk_get_column_count(self.raw)?)
    }

    /// One cell of the constant chunk.
    pub fn value(self, column: usize, row: usize) -> Result<ScalarValue> {
        let value = ffi::lo_chunk_get_value(self.raw, column, row)?;
        scalar_from_value(&value)
    }
}

impl<'plan> Insert<'plan> {
    /// Take the table binding DuckDB resolved for this INSERT target.
    pub fn take_table(self) -> Result<Box<OptionalTableWrapper>> {
        Ok(ffi::lo_insert_take_table(self.raw)?)
    }

    /// Where each physical table column takes its values from in the child's
    /// output: `None` for a column the statement leaves out. An empty result
    /// means DuckDB bound an insert into every physical column by position, so
    /// the child already emits the table's columns in order.
    pub fn column_map(self) -> Result<Vec<Option<usize>>> {
        (0..ffi::lo_insert_column_map_count(self.raw)?)
            .map(
                |index| match ffi::lo_insert_column_map_has_source(self.raw, index)? {
                    true => Ok(Some(ffi::lo_insert_column_map_source(self.raw, index)?)),
                    false => Ok(None),
                },
            )
            .collect()
    }

    pub fn returns_rows(self) -> Result<bool> {
        Ok(ffi::lo_insert_returns_rows(self.raw)?)
    }
}

impl<'plan> Filter<'plan> {
    /// The filter's boolean conditions (implicitly ANDed).
    pub fn exprs(self) -> Result<Vec<Expr<'plan>>> {
        (0..ffi::lo_filter_expr_count(self.raw)?)
            .map(|i| {
                Ok(Expr {
                    raw: ffi::lo_filter_expr(self.raw, i)?,
                })
            })
            .collect()
    }

    /// The filter's `projection_map`: each entry is the child-output column index
    /// it keeps, paired with that column's type. Empty when the filter passes all
    /// child columns through.
    pub fn projection_map(self) -> Result<Vec<(usize, BoundLogicalType)>> {
        (0..ffi::lo_filter_projection_map_count(self.raw)?)
            .map(|i| {
                Ok((
                    ffi::lo_filter_projection_map_index(self.raw, i)?,
                    bound_type_from(ffi::lo_filter_type_id(self.raw, i)?),
                ))
            })
            .collect()
    }
}

impl<'plan> Aggregate<'plan> {
    pub fn groups(self) -> Result<Vec<Expr<'plan>>> {
        (0..ffi::lo_aggregate_group_count(self.raw)?)
            .map(|i| {
                Ok(Expr {
                    raw: ffi::lo_aggregate_group(self.raw, i)?,
                })
            })
            .collect()
    }

    pub fn expressions(self) -> Result<Vec<Expr<'plan>>> {
        (0..ffi::lo_aggregate_expr_count(self.raw)?)
            .map(|i| {
                Ok(Expr {
                    raw: ffi::lo_aggregate_expr(self.raw, i)?,
                })
            })
            .collect()
    }
}

impl<'plan> Distinct<'plan> {
    /// Whether this is a `DISTINCT ON (...)` rather than a plain `DISTINCT`
    /// over the whole row.
    pub fn is_distinct_on(self) -> Result<bool> {
        Ok(ffi::lo_distinct_is_distinct_on(self.raw)?)
    }

    /// The expressions the node deduplicates on. The binding resolver has run,
    /// so each is a reference into the child's output columns.
    pub fn targets(self) -> Result<Vec<Expr<'plan>>> {
        (0..ffi::lo_distinct_target_count(self.raw)?)
            .map(|i| {
                Ok(Expr {
                    raw: ffi::lo_distinct_target(self.raw, i)?,
                })
            })
            .collect()
    }
}

impl<'plan> OrderBy<'plan> {
    pub fn keys(self) -> Result<Vec<OrderKey<'plan>>> {
        (0..ffi::lo_orderby_count(self.raw)?)
            .map(|i| {
                Ok(OrderKey {
                    direction: OrderType::from_u8(ffi::lo_orderby_direction(self.raw, i)?),
                    null_order: OrderByNullType::from_u8(ffi::lo_orderby_null_order(self.raw, i)?),
                    expression: Expr {
                        raw: ffi::lo_orderby_expr(self.raw, i)?,
                    },
                })
            })
            .collect()
    }
}

impl<'plan> TopN<'plan> {
    pub fn keys(self) -> Result<Vec<OrderKey<'plan>>> {
        (0..ffi::lo_topn_order_count(self.raw)?)
            .map(|i| {
                Ok(OrderKey {
                    direction: OrderType::from_u8(ffi::lo_topn_order_direction(self.raw, i)?),
                    null_order: OrderByNullType::from_u8(ffi::lo_topn_order_null_order(
                        self.raw, i,
                    )?),
                    expression: Expr {
                        raw: ffi::lo_topn_order_expr(self.raw, i)?,
                    },
                })
            })
            .collect()
    }

    pub fn limit(self) -> Result<usize> {
        Ok(ffi::lo_topn_limit(self.raw)?)
    }

    pub fn offset(self) -> Result<usize> {
        Ok(ffi::lo_topn_offset(self.raw)?)
    }

    /// The dynamic filter this Top-N publishes as a producer, if its optimizer
    /// installed one. At runtime it pushes its running boundary into the
    /// referenced shared cell so consumer scans elsewhere can prune.
    pub fn dynamic_filter(self) -> Result<Option<DynamicFilterRef>> {
        if !ffi::lo_topn_has_dynamic_filter(self.raw)? {
            return Ok(None);
        }
        Ok(Some(DynamicFilterRef {
            data_id: ffi::lo_topn_dynamic_filter_data_id(self.raw)?,
            column: ffi::lo_topn_dynamic_filter_column(self.raw)?,
            comparison: ExpressionType::from_u8(ffi::lo_topn_dynamic_filter_comparison(self.raw)?),
        }))
    }
}

impl<'plan> Limit<'plan> {
    pub fn value_kind(self) -> Result<LimitNodeType> {
        Ok(LimitNodeType::from_u8(ffi::lo_limit_value_kind(self.raw)?))
    }

    pub fn value(self) -> Result<usize> {
        Ok(ffi::lo_limit_value(self.raw)?)
    }

    pub fn offset_kind(self) -> Result<LimitNodeType> {
        Ok(LimitNodeType::from_u8(ffi::lo_limit_offset_kind(self.raw)?))
    }

    pub fn offset(self) -> Result<usize> {
        Ok(ffi::lo_limit_offset(self.raw)?)
    }
}

/// The projected output columns shared by base-table and table-function scans:
/// each a storage column index, its output type, and, when DuckDB's
/// projection-pushdown pushed a field extract into this column, the referenced
/// field path (empty means the whole column is read). The path lets the reader
/// fetch only the referenced variant leaf.
fn scan_output_columns(
    raw: &ffi::LogicalOperator,
) -> Result<Vec<(usize, BoundLogicalType, Vec<String>)>> {
    (0..ffi::lo_get_output_count(raw)?)
        .map(|i| {
            let depth = ffi::lo_get_output_extract_depth(raw, i)?;
            let path: Vec<String> = (0..depth)
                .map(|s| ffi::lo_get_output_extract_field(raw, i, s))
                .collect::<std::result::Result<_, _>>()?;
            Ok((
                ffi::lo_get_output_column(raw, i)?,
                bound_type_from(ffi::lo_get_output_type(raw, i)?),
                path,
            ))
        })
        .collect()
}

fn scan_dynamic_filters(raw: &ffi::LogicalOperator) -> Result<Vec<DynamicFilterRef>> {
    (0..ffi::lo_get_dynamic_filter_count(raw)?)
        .map(|i| {
            Ok(DynamicFilterRef {
                data_id: ffi::lo_get_dynamic_filter_data_id(raw, i)?,
                column: ffi::lo_get_dynamic_filter_column(raw, i)?,
                comparison: ExpressionType::from_u8(ffi::lo_get_dynamic_filter_comparison(raw, i)?),
            })
        })
        .collect()
}

impl<'plan> TableScan<'plan> {
    /// Move the pivot table handle out of this scan's catalog entry. Call once
    /// per base-table scan during the walk.
    pub fn take_table(self) -> Result<Box<OptionalTableWrapper>> {
        Ok(ffi::lo_get_take_table(self.raw)?)
    }

    /// The scan's projected output columns, each a storage column index paired
    /// with its type and pushed extract path.
    pub fn output_columns(self) -> Result<Vec<(usize, BoundLogicalType, Vec<String>)>> {
        scan_output_columns(self.raw)
    }

    /// The static `col op const` predicates DuckDB pushed into `table_filters`,
    /// rebuilt as expressions owned by the returned [`PushedConditions`]. Empty
    /// when there are none.
    pub fn pushed_conditions(self) -> Result<PushedConditions> {
        Ok(PushedConditions {
            list: ffi::lo_get_pushed_conditions(self.raw)?,
        })
    }

    /// The get's DuckDB binding table index, which plan-level references
    /// (e.g. a join's filter-pushdown targets) name the scan by.
    pub fn table_index(self) -> Result<usize> {
        Ok(ffi::lo_get_table_index(self.raw)?)
    }

    /// The pointer identity of the shared filter set a join above pushes
    /// filters into, matching a [`JoinPushdownTarget`]. `None` when no join
    /// pushes filters into this scan.
    pub fn join_filter_set_id(self) -> Result<Option<usize>> {
        let filter_set_id = ffi::lo_get_join_filter_set_id(self.raw)?;
        Ok((filter_set_id != 0).then_some(filter_set_id))
    }

    /// The get's `column_ids` index -> storage column mapping, the space
    /// join-pushdown probe columns arrive in.
    pub fn storage_columns(self) -> Result<Vec<usize>> {
        (0..ffi::lo_get_column_ids_count(self.raw)?)
            .map(|i| Ok(ffi::lo_get_storage_column(self.raw, i)?))
            .collect()
    }

    /// Dynamic-filter consumers attached to this scan's `table_filters`.
    pub fn dynamic_filters(self) -> Result<Vec<DynamicFilterRef>> {
        scan_dynamic_filters(self.raw)
    }
}

impl<'plan> TableFunctionScan<'plan> {
    pub fn function_name(self) -> Result<String> {
        Ok(ffi::lo_get_function_name(self.raw)?)
    }

    pub fn has_named_params(self) -> Result<bool> {
        Ok(ffi::lo_get_has_named_params(self.raw)?)
    }

    /// The function's bound constant arguments.
    pub fn params(self) -> Result<Vec<ScalarValue>> {
        (0..ffi::lo_get_param_count(self.raw)?)
            .map(|i| scalar_from_value(ffi::lo_get_param(self.raw, i)?))
            .collect()
    }

    /// Whether this Pivot table-function invocation bound to a regular table.
    pub fn has_bound_table(self) -> Result<bool> {
        Ok(ffi::lo_get_has_bound_table_function(self.raw)?)
    }

    /// Clone the regular table produced while binding this invocation.
    pub fn bound_table(self) -> Result<Box<OptionalTableWrapper>> {
        Ok(ffi::lo_get_clone_bound_table_function(self.raw)?)
    }

    /// The scan's projected output columns, each a generated column index paired
    /// with its type and pushed extract path.
    pub fn output_columns(self) -> Result<Vec<(usize, BoundLogicalType, Vec<String>)>> {
        scan_output_columns(self.raw)
    }

    /// Dynamic-filter consumers attached to this invocation's `table_filters`.
    pub fn dynamic_filters(self) -> Result<Vec<DynamicFilterRef>> {
        scan_dynamic_filters(self.raw)
    }

    /// The static predicates DuckDB lowered into this invocation's
    /// `table_filters`. Pivot replays them above the bound table's scan.
    pub fn pushed_conditions(self) -> Result<PushedConditions> {
        Ok(PushedConditions {
            list: ffi::lo_get_pushed_conditions(self.raw)?,
        })
    }
}

impl<'plan> CreateSchema<'plan> {
    pub fn name(self) -> Result<String> {
        Ok(ffi::lo_create_schema_name(self.raw)?)
    }

    /// The target database (datastore) of `CREATE SCHEMA db.s`, or `None` when
    /// the statement is unqualified (routes to the default datastore).
    pub fn datastore(self) -> Result<Option<String>> {
        let datastore = ffi::lo_create_schema_datastore(self.raw)?;
        Ok((!datastore.is_empty()).then_some(datastore))
    }

    pub fn if_not_exists(self) -> Result<bool> {
        Ok(ffi::lo_create_schema_if_not_exists(self.raw)?)
    }

    pub fn or_replace(self) -> Result<bool> {
        Ok(ffi::lo_create_schema_or_replace(self.raw)?)
    }
}

impl<'plan> Explain<'plan> {
    /// Whether the statement is `EXPLAIN ANALYZE` (vs plain `EXPLAIN`).
    pub fn is_analyze(self) -> Result<bool> {
        Ok(ffi::lo_explain_is_analyze(self.raw)?)
    }
}

impl<'plan> Drop<'plan> {
    /// Whether the drop targets a table (the only kind the consumer supports).
    pub fn is_table(self) -> Result<bool> {
        Ok(ffi::lo_drop_is_table(self.raw)?)
    }

    /// The kind of catalog entry the drop targets, as DuckDB spells it
    /// (`TABLE`, `SCHEMA`, `VIEW`, ...): for the unsupported-kind error.
    pub fn entry_kind(self) -> Result<String> {
        Ok(ffi::lo_drop_entry_kind(self.raw)?)
    }

    pub fn name(self) -> Result<String> {
        Ok(ffi::lo_drop_name(self.raw)?)
    }

    /// The schema holding the dropped entry, or `None` when the statement named
    /// none and nothing resolved (an `IF EXISTS` drop of a missing entry).
    pub fn schema(self) -> Result<Option<String>> {
        let schema = ffi::lo_drop_schema(self.raw)?;
        Ok((!schema.is_empty()).then_some(schema))
    }

    /// The target database (datastore), or `None` when the statement was
    /// unqualified and nothing resolved (routes to the default datastore).
    pub fn datastore(self) -> Result<Option<String>> {
        let datastore = ffi::lo_drop_datastore(self.raw)?;
        Ok((!datastore.is_empty()).then_some(datastore))
    }

    pub fn if_exists(self) -> Result<bool> {
        Ok(ffi::lo_drop_if_exists(self.raw)?)
    }

    pub fn cascade(self) -> Result<bool> {
        Ok(ffi::lo_drop_cascade(self.raw)?)
    }
}

impl<'plan> CreateTable<'plan> {
    pub fn name(self) -> Result<String> {
        Ok(ffi::lo_create_table_name(self.raw)?)
    }

    /// The target database (datastore) of `CREATE TABLE db.schema.t`, or `None`
    /// when the statement is unqualified (routes to the default datastore).
    pub fn datastore(self) -> Result<Option<String>> {
        let datastore = ffi::lo_create_table_datastore(self.raw)?;
        Ok((!datastore.is_empty()).then_some(datastore))
    }

    /// The target schema of `CREATE TABLE db.schema.t`, or `None` when the
    /// statement named none (routes to the default schema).
    pub fn schema(self) -> Result<Option<String>> {
        let schema = ffi::lo_create_table_schema(self.raw)?;
        Ok((!schema.is_empty()).then_some(schema))
    }

    pub fn columns(self) -> Result<Vec<(String, BoundLogicalType)>> {
        (0..ffi::lo_create_column_count(self.raw)?)
            .map(|i| {
                Ok((
                    ffi::lo_create_column_name(self.raw, i)?,
                    bound_type_from(ffi::lo_create_column_type(self.raw, i)?),
                ))
            })
            .collect()
    }

    pub fn options(self) -> Result<Vec<(String, String)>> {
        (0..ffi::lo_create_option_count(self.raw)?)
            .map(|i| {
                Ok((
                    ffi::lo_create_option_key(self.raw, i)?,
                    ffi::lo_create_option_value(self.raw, i)?,
                ))
            })
            .collect()
    }

    pub fn if_not_exists(self) -> Result<bool> {
        Ok(ffi::lo_create_if_not_exists(self.raw)?)
    }

    pub fn or_replace(self) -> Result<bool> {
        Ok(ffi::lo_create_or_replace(self.raw)?)
    }

    pub fn temporary(self) -> Result<bool> {
        Ok(ffi::lo_create_temporary(self.raw)?)
    }

    pub fn has_query(self) -> Result<bool> {
        Ok(ffi::lo_create_has_query(self.raw)?)
    }

    pub fn constraint_count(self) -> Result<usize> {
        Ok(ffi::lo_create_constraint_count(self.raw)?)
    }
}

impl<'plan> Set<'plan> {
    pub fn name(self) -> Result<String> {
        Ok(ffi::lo_set_name(self.raw)?)
    }

    pub fn value(self) -> Result<String> {
        Ok(ffi::lo_set_value(self.raw)?)
    }
}

impl<'plan> Reset<'plan> {
    pub fn name(self) -> Result<String> {
        Ok(ffi::lo_reset_name(self.raw)?)
    }
}

impl<'plan> Transaction<'plan> {
    /// Which transaction statement this is.
    pub fn transaction_type(self) -> Result<TransactionType> {
        Ok(TransactionType::from_u8(ffi::lo_transaction_type(
            self.raw,
        )?))
    }
}

impl<'plan> Compact<'plan> {
    /// The datastore the statement named, or `None` when unqualified (routes
    /// to the default datastore).
    pub fn datastore(self) -> Result<Option<String>> {
        let datastore = ffi::lo_compact_datastore(self.raw)?;
        Ok((!datastore.is_empty()).then_some(datastore))
    }

    /// The schema the statement named, or `None` when unqualified (routes to
    /// the default schema).
    pub fn schema(self) -> Result<Option<String>> {
        let schema = ffi::lo_compact_schema(self.raw)?;
        Ok((!schema.is_empty()).then_some(schema))
    }

    pub fn table(self) -> Result<String> {
        Ok(ffi::lo_compact_table(self.raw)?)
    }

    /// `COMPACT ... FINAL`: keep sweeping until a pass merges nothing.
    pub fn final_sweep(self) -> Result<bool> {
        Ok(ffi::lo_compact_final(self.raw)?)
    }
}

impl<'plan> CopyFromStdin<'plan> {
    /// Take the table binding DuckDB resolved for this COPY target.
    pub fn take_table(self) -> Result<Box<OptionalTableWrapper>> {
        Ok(ffi::lo_copy_stdin_take_table(self.raw)?)
    }

    /// The explicit column list resolved to physical column positions; empty
    /// when the statement targets every table column.
    pub fn column_indexes(self) -> Result<Vec<usize>> {
        (0..ffi::lo_copy_stdin_column_count(self.raw)?)
            .map(|index| Ok(ffi::lo_copy_stdin_column_index(self.raw, index)?))
            .collect()
    }

    /// The FORMAT option, or `None` when the statement gave none.
    pub fn format(self) -> Result<Option<String>> {
        let format = ffi::lo_copy_stdin_format(self.raw)?;
        Ok((!format.is_empty()).then_some(format))
    }

    /// The remaining `WITH (...)` options in name order, each with its bound
    /// constant values rendered as text (none for a bare flag like `HEADER`).
    pub fn options(self) -> Result<Vec<(String, Vec<String>)>> {
        (0..ffi::lo_copy_stdin_option_count(self.raw)?)
            .map(|index| {
                let name = ffi::lo_copy_stdin_option_name(self.raw, index)?;
                let values = (0..ffi::lo_copy_stdin_option_value_count(self.raw, index)?)
                    .map(|value_index| {
                        Ok(ffi::lo_copy_stdin_option_value(
                            self.raw,
                            index,
                            value_index,
                        )?)
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok((name, values))
            })
            .collect()
    }
}

impl<'plan> CreateUser<'plan> {
    /// The name of the user to create.
    pub fn name(self) -> Result<String> {
        Ok(ffi::lo_create_user_name(self.raw)?)
    }

    /// The password, or `None` when the PASSWORD clause was omitted (the user
    /// authenticates by trust).
    pub fn password(self) -> Result<Option<String>> {
        if ffi::lo_create_user_has_password(self.raw)? {
            Ok(Some(ffi::lo_create_user_password(self.raw)?))
        } else {
            Ok(None)
        }
    }
}

impl<'plan> ComparisonJoin<'plan> {
    /// The join's [`JoinType`] (INNER/SEMI/...).
    pub fn join_type(self) -> Result<JoinType> {
        Ok(JoinType::from_u8(ffi::lo_join_type(self.raw)?))
    }

    /// The join's conditions, each in one of DuckDB's two forms (see
    /// [`JoinConditionEntry`]).
    pub fn conditions(self) -> Result<Vec<JoinConditionEntry<'plan>>> {
        (0..ffi::lo_join_condition_count(self.raw)?)
            .map(|i| {
                if !ffi::lo_join_condition_is_comparison(self.raw, i)? {
                    return Ok(JoinConditionEntry::Expression(Expr {
                        raw: ffi::lo_join_condition_expression(self.raw, i)?,
                    }));
                }
                Ok(JoinConditionEntry::Comparison(JoinCondition {
                    left: Expr {
                        raw: ffi::lo_join_condition_left(self.raw, i)?,
                    },
                    right: Expr {
                        raw: ffi::lo_join_condition_right(self.raw, i)?,
                    },
                    comparison: ExpressionType::from_u8(ffi::lo_join_condition_comparison(
                        self.raw, i,
                    )?),
                }))
            })
            .collect()
    }

    /// Which LHS child output columns survive in the join's output, in order.
    /// Empty means all of them. Filled by DuckDB's column-lifetime pass, e.g.
    /// to drop a key column only referenced by the join condition.
    pub fn left_projection_map(self) -> Result<Vec<usize>> {
        (0..ffi::lo_join_left_projection_map_count(self.raw)?)
            .map(|i| Ok(ffi::lo_join_left_projection_map_index(self.raw, i)?))
            .collect()
    }

    /// The RHS twin of [`left_projection_map`](Self::left_projection_map).
    pub fn right_projection_map(self) -> Result<Vec<usize>> {
        (0..ffi::lo_join_right_projection_map_count(self.raw)?)
            .map(|i| Ok(ffi::lo_join_right_projection_map_index(self.raw, i)?))
            .collect()
    }

    /// The optimizer's join-filter-pushdown record: which probe-side scans can
    /// be narrowed by filters derived from this join's build-side key values.
    /// Empty when the optimizer created none (unsafe join type, no eligible
    /// conditions, or no reachable scan).
    pub fn pushdown_targets(self) -> Result<Vec<JoinPushdownTarget>> {
        (0..ffi::lo_join_pushdown_target_count(self.raw)?)
            .map(|t| {
                let columns = (0..ffi::lo_join_pushdown_column_count(self.raw, t)?)
                    .map(|i| {
                        Ok(JoinPushdownColumn {
                            probe_table_index: ffi::lo_join_pushdown_probe_table(self.raw, t, i)?,
                            probe_column: ffi::lo_join_pushdown_probe_column(self.raw, t, i)?,
                            condition_index: ffi::lo_join_pushdown_condition_index(self.raw, i)?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(JoinPushdownTarget {
                    filter_set_id: ffi::lo_join_pushdown_set_id(self.raw, t)?,
                    columns,
                })
            })
            .collect()
    }
}

impl<'plan> DelimJoin<'plan> {
    /// The underlying comparison join: a delim join is stored as a
    /// `LogicalComparisonJoin`, so the type, conditions and projection maps all
    /// read through the plain join view.
    pub fn join(self) -> ComparisonJoin<'plan> {
        ComparisonJoin { raw: self.raw }
    }

    /// The expressions (over the de-duplicated side's output) whose distinct
    /// values every [`DelimGet`] under the other side scans.
    pub fn delim_columns(self) -> Result<Vec<Expr<'plan>>> {
        (0..ffi::lo_delim_join_column_count(self.raw)?)
            .map(|i| {
                Ok(Expr {
                    raw: ffi::lo_delim_join_column(self.raw, i)?,
                })
            })
            .collect()
    }

    /// False: the LHS is de-duplicated and the [`DelimGet`]s sit under the RHS.
    /// True: the join was flipped and the roles reverse.
    pub fn is_flipped(self) -> Result<bool> {
        Ok(ffi::lo_delim_join_is_flipped(self.raw)?)
    }
}

impl<'plan> DelimGet<'plan> {
    /// The types of the de-duplicated values this scan produces, in column
    /// order (matching the enclosing delim join's `delim_columns`).
    pub fn column_types(self) -> Result<Vec<BoundLogicalType>> {
        (0..ffi::lo_delim_get_column_count(self.raw)?)
            .map(|i| Ok(bound_type_from(ffi::lo_delim_get_column_type(self.raw, i)?)))
            .collect()
    }
}

/// One condition of a comparison join, in one of DuckDB's two stored forms.
pub enum JoinConditionEntry<'plan> {
    /// `left <comparison> right`, each side bound positionally to the
    /// corresponding child's output.
    Comparison(JoinCondition<'plan>),
    /// A predicate that is not a bare comparison (e.g. an OR referencing both
    /// sides), bound positionally against the two children's concatenated
    /// outputs: all LHS columns, then all RHS columns.
    Expression(Expr<'plan>),
}

/// A comparison-form join condition: `left <comparison> right`, each side
/// bound positionally to the corresponding child's output.
pub struct JoinCondition<'plan> {
    pub left: Expr<'plan>,
    pub right: Expr<'plan>,
    pub comparison: ExpressionType,
}

/// One sort key of an ORDER BY / TopN: a direction, where NULLs go, and the
/// keyed expression. The binder resolves a key written without `NULLS FIRST`
/// / `NULLS LAST` to one of the two, so `null_order` is never `ORDER_DEFAULT`.
pub struct OrderKey<'plan> {
    pub direction: OrderType,
    pub null_order: OrderByNullType,
    pub expression: Expr<'plan>,
}

/// A dynamic-filter wiring read off a scan or Top-N: the shared cell it
/// references (by address), the column it constrains, and the comparison shape.
/// The constant itself is published into the cell at runtime.
pub struct DynamicFilterRef {
    pub data_id: usize,
    pub column: usize,
    pub comparison: ExpressionType,
}

/// One join-filter-pushdown target read off a comparison join: the probe-side
/// scan it narrows (paired by `filter_set_id` with
/// [`TableScan::join_filter_set_id`]) and the pushed columns.
pub struct JoinPushdownTarget {
    pub filter_set_id: usize,
    pub columns: Vec<JoinPushdownColumn>,
}

/// One pushed column of a [`JoinPushdownTarget`].
pub struct JoinPushdownColumn {
    /// The target scan's table index (a second pairing check besides the set).
    pub probe_table_index: usize,
    /// Index into the target get's `column_ids`; resolve to a storage column
    /// via [`TableScan::storage_columns`].
    pub probe_column: usize,
    /// The join condition whose build-side values bound this column.
    pub condition_index: usize,
}

/// An owning list of synthesized expressions (a scan's pushed-down filter
/// conditions). Borrow the entries as [`Expr`] handles via [`PushedConditions::exprs`].
pub struct PushedConditions {
    list: UniquePtr<ffi::ExpressionList>,
}

impl PushedConditions {
    pub fn exprs(&self) -> Result<Vec<Expr<'_>>> {
        (0..ffi::expr_list_count(&self.list)?)
            .map(|i| {
                Ok(Expr {
                    raw: ffi::expr_list_get(&self.list, i)?,
                })
            })
            .collect()
    }
}

/// A borrowed DuckDB bound `Expression`, tethered to the [`Plan`] that owns it.
#[derive(Clone, Copy)]
pub struct Expr<'plan> {
    raw: &'plan ffi::Expression,
}

impl<'plan> Expr<'plan> {
    /// Wrap a raw bound expression. Used by the catalog pushdown callback, which
    /// receives a borrowed `Expression` straight from C++.
    pub(crate) fn from_raw(raw: &'plan ffi::Expression) -> Self {
        Expr { raw }
    }

    /// Dispatch on the expression's [`ExpressionType`] into a typed [`Expression`]
    /// view that exposes only that kind's accessors. Any kind the consumer doesn't
    /// handle is [`Expression::Unsupported`], carrying the raw type.
    pub fn expression(self) -> Result<Expression<'plan>> {
        use ExpressionType as T;
        Ok(match ExpressionType::from_u8(ffi::expr_type(self.raw)?) {
            T::BOUND_REF | T::BOUND_COLUMN_REF => Expression::Ref(Ref { raw: self.raw }),
            T::COMPARE_EQUAL
            | T::COMPARE_NOTEQUAL
            | T::COMPARE_LESSTHAN
            | T::COMPARE_GREATERTHAN
            | T::COMPARE_LESSTHANOREQUALTO
            | T::COMPARE_GREATERTHANOREQUALTO => Expression::Compare(Compare { raw: self.raw }),
            T::COMPARE_BETWEEN => Expression::Between(Between { raw: self.raw }),
            T::VALUE_CONSTANT => Expression::Constant(Constant { raw: self.raw }),
            T::BOUND_AGGREGATE => Expression::AggregateFunc(AggregateFunc { raw: self.raw }),
            T::BOUND_FUNCTION => Expression::Function(Function { raw: self.raw }),
            T::COMPARE_IN | T::COMPARE_NOT_IN => Expression::InList(InList { raw: self.raw }),
            T::CONJUNCTION_AND | T::CONJUNCTION_OR => {
                Expression::Conjunction(Conjunction { raw: self.raw })
            }
            T::CASE_EXPR => Expression::Case(Case { raw: self.raw }),
            T::OPERATOR_NOT => Expression::Not(Not { raw: self.raw }),
            T::OPERATOR_IS_NULL | T::OPERATOR_IS_NOT_NULL => {
                Expression::IsNull(IsNull { raw: self.raw })
            }
            T::OPERATOR_CAST => Expression::Cast(Cast { raw: self.raw }),
            other => Expression::Unsupported(other),
        })
    }
}

/// A borrowed, typed view of an [`Expr`], discriminated by expression kind. Each
/// variant exposes only the accessors valid for that kind, so a comparison
/// accessor can't be called on a constant. Obtained via [`Expr::expression`].
/// Every variant is a zero-cost borrowed view (one reference), not owned data.
pub enum Expression<'plan> {
    /// A bound column reference (`BOUND_REF` / `BOUND_COLUMN_REF`).
    Ref(Ref<'plan>),
    /// A binary comparison (`=`, `<>`, `<`, …).
    Compare(Compare<'plan>),
    Between(Between<'plan>),
    Constant(Constant<'plan>),
    /// An aggregate function call (`SUM`, `COUNT`, …).
    AggregateFunc(AggregateFunc<'plan>),
    /// A scalar function call (`BOUND_FUNCTION`).
    Function(Function<'plan>),
    /// `input IN (v0, v1, …)`.
    InList(InList<'plan>),
    /// A boolean `AND`/`OR`.
    Conjunction(Conjunction<'plan>),
    Case(Case<'plan>),
    /// Logical negation (`NOT expr`).
    Not(Not<'plan>),
    /// An `expr IS NULL` / `expr IS NOT NULL` test.
    IsNull(IsNull<'plan>),
    /// A type cast.
    Cast(Cast<'plan>),
    /// Any expression type the consumer doesn't handle, carrying the raw type.
    Unsupported(ExpressionType),
}

define_handles! { ffi::Expression;
    /// A bound column reference.
    Ref,
    /// A binary comparison expression.
    Compare,
    /// A `BETWEEN` range test.
    Between,
    /// A constant value.
    Constant,
    /// An aggregate function call.
    AggregateFunc,
    /// A scalar function call.
    Function,
    /// An `input IN (v0, v1, …)` membership test.
    InList,
    /// A boolean `AND`/`OR` over child predicates.
    Conjunction,
    /// A `CASE WHEN … THEN … ELSE … END` expression.
    Case,
    /// Logical negation (`NOT expr`).
    Not,
    /// An `expr IS NULL` / `expr IS NOT NULL` test.
    IsNull,
    /// A type cast (`CAST(child AS return_type)`).
    Cast,
}

impl<'plan> Ref<'plan> {
    /// The referenced column's positional index. A `BOUND_REF` indexes the child
    /// operator's output; a `BOUND_COLUMN_REF` carries its own column index.
    pub fn column_index(self) -> Result<usize> {
        if ExpressionType::from_u8(ffi::expr_type(self.raw)?) == ExpressionType::BOUND_REF {
            Ok(ffi::expr_ref_index(self.raw)?)
        } else {
            Ok(ffi::expr_columnref_index(self.raw)?)
        }
    }

    /// The column's logical type.
    pub fn return_type(self) -> Result<BoundLogicalType> {
        Ok(bound_type_from(ffi::expr_return_type(self.raw)?))
    }

    /// The column's source name from DuckDB's binding, or `None`. Display-only.
    pub fn alias(self) -> Result<Option<String>> {
        if ffi::expr_has_alias(self.raw)? {
            Ok(Some(ffi::expr_alias(self.raw)?))
        } else {
            Ok(None)
        }
    }
}

impl<'plan> Compare<'plan> {
    /// The specific comparison (one of the `COMPARE_*` [`ExpressionType`]s).
    pub fn comparison_type(self) -> Result<ExpressionType> {
        Ok(ExpressionType::from_u8(ffi::expr_type(self.raw)?))
    }

    pub fn left(self) -> Result<Expr<'plan>> {
        Ok(Expr {
            raw: ffi::expr_comparison_left(self.raw)?,
        })
    }

    pub fn right(self) -> Result<Expr<'plan>> {
        Ok(Expr {
            raw: ffi::expr_comparison_right(self.raw)?,
        })
    }

    /// The comparison's result type.
    pub fn return_type(self) -> Result<BoundLogicalType> {
        Ok(bound_type_from(ffi::expr_return_type(self.raw)?))
    }
}

impl<'plan> Between<'plan> {
    pub fn input(self) -> Result<Expr<'plan>> {
        Ok(Expr {
            raw: ffi::expr_between_input(self.raw)?,
        })
    }

    pub fn lower(self) -> Result<Expr<'plan>> {
        Ok(Expr {
            raw: ffi::expr_between_lower(self.raw)?,
        })
    }

    pub fn upper(self) -> Result<Expr<'plan>> {
        Ok(Expr {
            raw: ffi::expr_between_upper(self.raw)?,
        })
    }

    pub fn lower_inclusive(self) -> Result<bool> {
        Ok(ffi::expr_between_lower_inclusive(self.raw)?)
    }

    pub fn upper_inclusive(self) -> Result<bool> {
        Ok(ffi::expr_between_upper_inclusive(self.raw)?)
    }
}

impl<'plan> Constant<'plan> {
    /// The constant's typed value.
    pub fn value(self) -> Result<ScalarValue> {
        scalar_from_value(ffi::expr_constant(self.raw)?)
    }

    /// The constant's logical type, without decoding its value.
    pub fn return_type(self) -> Result<BoundLogicalType> {
        Ok(bound_type_from(ffi::expr_return_type(self.raw)?))
    }
}

impl<'plan> AggregateFunc<'plan> {
    pub fn name(self) -> Result<String> {
        Ok(ffi::expr_aggregate_name(self.raw)?)
    }

    pub fn distinct(self) -> Result<bool> {
        Ok(ffi::expr_aggregate_distinct(self.raw)?)
    }

    pub fn children(self) -> Result<Vec<Expr<'plan>>> {
        (0..ffi::expr_aggregate_child_count(self.raw)?)
            .map(|i| {
                Ok(Expr {
                    raw: ffi::expr_aggregate_child(self.raw, i)?,
                })
            })
            .collect()
    }

    /// Whether the call carries a `FILTER (WHERE …)` clause.
    pub fn has_filter(self) -> Result<bool> {
        Ok(ffi::expr_aggregate_has_filter(self.raw)?)
    }

    /// DuckDB's declared result type for the call.
    pub fn return_type(self) -> Result<BoundLogicalType> {
        Ok(bound_type_from(ffi::expr_return_type(self.raw)?))
    }
}

impl<'plan> Function<'plan> {
    pub fn name(self) -> Result<String> {
        Ok(ffi::expr_function_name(self.raw)?)
    }

    pub fn children(self) -> Result<Vec<Expr<'plan>>> {
        (0..ffi::expr_function_child_count(self.raw)?)
            .map(|i| {
                Ok(Expr {
                    raw: ffi::expr_function_child(self.raw, i)?,
                })
            })
            .collect()
    }

    /// The function's result type.
    pub fn return_type(self) -> Result<BoundLogicalType> {
        Ok(bound_type_from(ffi::expr_return_type(self.raw)?))
    }
}

impl<'plan> InList<'plan> {
    /// The `BoundOperatorExpression` children: child 0 is the tested expression,
    /// children 1.. are the list values.
    pub fn children(self) -> Result<Vec<Expr<'plan>>> {
        (0..ffi::expr_operator_child_count(self.raw)?)
            .map(|i| {
                Ok(Expr {
                    raw: ffi::expr_operator_child(self.raw, i)?,
                })
            })
            .collect()
    }

    /// `true` for `NOT IN`, `false` for `IN`.
    pub fn negated(self) -> Result<bool> {
        Ok(ExpressionType::from_u8(ffi::expr_type(self.raw)?) == ExpressionType::COMPARE_NOT_IN)
    }
}

impl<'plan> Conjunction<'plan> {
    /// `CONJUNCTION_AND` or `CONJUNCTION_OR`.
    pub fn conjunction_type(self) -> Result<ExpressionType> {
        Ok(ExpressionType::from_u8(ffi::expr_type(self.raw)?))
    }

    pub fn children(self) -> Result<Vec<Expr<'plan>>> {
        (0..ffi::expr_conjunction_child_count(self.raw)?)
            .map(|i| {
                Ok(Expr {
                    raw: ffi::expr_conjunction_child(self.raw, i)?,
                })
            })
            .collect()
    }
}

impl<'plan> Case<'plan> {
    pub fn checks(self) -> Result<Vec<CaseArm<'plan>>> {
        (0..ffi::expr_case_check_count(self.raw)?)
            .map(|i| {
                Ok(CaseArm {
                    when: Expr {
                        raw: ffi::expr_case_when(self.raw, i)?,
                    },
                    then: Expr {
                        raw: ffi::expr_case_then(self.raw, i)?,
                    },
                })
            })
            .collect()
    }

    pub fn else_expr(self) -> Result<Expr<'plan>> {
        Ok(Expr {
            raw: ffi::expr_case_else(self.raw)?,
        })
    }
}

impl<'plan> Not<'plan> {
    /// The negated operand (`OPERATOR_NOT` has a single child).
    pub fn input(self) -> Result<Expr<'plan>> {
        Ok(Expr {
            raw: ffi::expr_operator_child(self.raw, 0)?,
        })
    }
}

impl<'plan> IsNull<'plan> {
    /// The tested operand (both IS NULL forms have a single child).
    pub fn input(self) -> Result<Expr<'plan>> {
        Ok(Expr {
            raw: ffi::expr_operator_child(self.raw, 0)?,
        })
    }

    /// `true` for `IS NOT NULL`, `false` for `IS NULL`.
    pub fn negated(self) -> Result<bool> {
        Ok(ExpressionType::from_u8(ffi::expr_type(self.raw)?)
            == ExpressionType::OPERATOR_IS_NOT_NULL)
    }
}

impl<'plan> Cast<'plan> {
    pub fn child(self) -> Result<Expr<'plan>> {
        Ok(Expr {
            raw: ffi::expr_cast_child(self.raw)?,
        })
    }

    /// The cast's target type (a `BoundCastExpression`'s own result type).
    pub fn return_type(self) -> Result<BoundLogicalType> {
        Ok(bound_type_from(ffi::expr_return_type(self.raw)?))
    }

    /// Whether this is a `TRY_CAST`, which yields NULL for a value the target
    /// type cannot represent instead of failing the query.
    pub fn is_try(self) -> Result<bool> {
        Ok(ffi::expr_cast_is_try(self.raw)?)
    }
}

/// One `WHEN when THEN then` arm of a `CASE` expression.
pub struct CaseArm<'plan> {
    pub when: Expr<'plan>,
    pub then: Expr<'plan>,
}
