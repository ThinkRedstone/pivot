//! FFI layer to the DuckDB C++ planner.
//!
//! * [`ffi`] — `cxx`-generated bridge exposing `DuckPlannerContext`,
//!   `new_context`, `extract_plan`, and the accessor functions the Rust plan
//!   builder uses to read DuckDB's own plan objects.
//! * [`duckdb_types`] — `autocxx`-generated Rust mirrors of DuckDB enums
//!   (`LogicalOperatorType`, `ExpressionType`, etc.).

pub mod duckdb_types;

use crate::catalog_provider::{
    CatalogContext, OptionalTableWrapper, TransactionContext, catalog_bind_table_function,
    catalog_context_default, catalog_context_names, catalog_does_schema_exist,
    catalog_get_scalar_function, catalog_get_table, catalog_table_functions,
    clone_table_function_table, pushdown_filter, table_estimate_row_count,
    table_supports_late_materialization,
};

/// CXX bridge to the hand-written C++ glue in `bridge.cpp` / `bridge.h`.
///
/// `extract_plan` hands back an opaque [`PlanHandle`] that owns DuckDB's
/// resolved `LogicalOperator` tree. The Rust plan builder walks that live tree
/// directly through the accessor functions below (`lo_*` for operators, `expr_*`
/// for expressions): there is no intermediate representation, the Rust side just
/// reads fields straight off the C++ objects.
#[cxx::bridge]
pub mod ffi {
    /// Column definition for FFI, carrying the type as a `u8` discriminant
    /// because CXX cannot pass Rust enums across the bridge. When the type is
    /// `DECIMAL` the width and scale complete it; they are zero otherwise.
    struct DuckDBColumn {
        pub name: String,
        pub duckdb_logical_type_id: u8,
        pub decimal_width: u8,
        pub decimal_scale: u8,
    }

    /// A DuckDB `LogicalType` read off a bound plan object: the
    /// `LogicalTypeId` discriminant, plus the width and scale that complete a
    /// `DECIMAL` (zero for every other id, which the id alone fully describes).
    struct BridgeLogicalType {
        pub id: u8,
        pub decimal_width: u8,
        pub decimal_scale: u8,
    }

    /// A `DECIMAL` constant: the unscaled 128-bit integer split into halves
    /// (CXX has no 128-bit type), plus the value's width and scale.
    struct BridgeDecimalValue {
        pub hi: i64,
        pub lo: u64,
        pub width: u8,
        pub scale: u8,
    }

    /// A `HUGEINT` constant split into halves (CXX has no 128-bit type).
    struct BridgeHugeint {
        pub hi: i64,
        pub lo: u64,
    }

    /// Result of a catalog table lookup, allowing C++ to inspect the outcome.
    struct CatalogGetTableResult {
        pub found: bool,
        pub columns: Vec<DuckDBColumn>,
        /// Opaque handle to the Rust `DuckDBTable` object that was looked up.
        pub table: Box<OptionalTableWrapper>,
    }

    /// One globally registered table-function overload. Several entries may
    /// share a name and are grouped into one DuckDB function set.
    struct TableFunctionDef {
        pub name: String,
        pub arg_type_ids: Vec<u8>,
        /// Whether DuckDB may split this function's scan into a narrow scan
        /// followed by a late-materialization fetch.
        pub supports_late_materialization: bool,
    }

    /// One invocation-specific table-function binding: its footer-derived
    /// columns and the regular table object carried into the logical scan.
    struct CatalogBindTableFunctionResult {
        pub columns: Vec<DuckDBColumn>,
        pub table: Box<OptionalTableWrapper>,
    }

    /// A scalar function the current transaction defines for DuckDB's binder:
    /// its argument and return types (DuckDB logical type id discriminants),
    /// plus whether it must be marked `VOLATILE` so the optimizer can't fold
    /// the call away before pivot re-plans it.
    struct ScalarFunctionDef {
        pub arg_type_ids: Vec<u8>,
        pub return_type_id: u8,
        pub is_volatile: bool,
    }

    /// Result of a catalog scalar-function lookup, or `found = false` if the
    /// transaction has no such function.
    struct CatalogGetScalarFunctionResult {
        pub found: bool,
        pub function: ScalarFunctionDef,
    }

    /// A table's estimated total row count for DuckDB's cost model.
    /// `known = false` when the backend has no metadata-only answer; DuckDB
    /// then uses its own defaults.
    struct CardinalityEstimate {
        pub known: bool,
        pub rows: u64,
    }

    /// Outcome of `extract_plan`.
    ///
    /// On success `error_kind` is empty and `plan` owns DuckDB's resolved plan
    /// tree (kept alive for the duration of the Rust walk). On failure `plan` is
    /// null and the `error_*` fields describe what went wrong (`error_kind` is
    /// one of `duckdb_planning`, `unsupported_plan`, `bridge_error`).
    struct ExtractPlanResult {
        pub error_kind: String,
        pub error_message: String,
        pub has_error_position: bool,
        pub error_position: String,
        pub plan: UniquePtr<PlanHandle>,
        /// The binder-resolved result column names DuckDB would hand a client,
        /// in select order (e.g. `["hour", "count_star()"]`). Empty when the
        /// bridge could not recover them.
        pub output_names: Vec<String>,
        /// DuckDB found query-dependent state (such as its native `now()`)
        /// whose optimized value requires a fresh plan for another execution.
        pub requires_rebind: bool,
    }

    extern "Rust" {
        type CatalogContext;
        type TransactionContext;
        type OptionalTableWrapper;
        /// Datastore names to attach (one DuckDB database each), and which is the
        /// current database, read by the C++ context constructor.
        fn catalog_context_names(ctx: &CatalogContext) -> Vec<String>;
        fn catalog_context_default(ctx: &CatalogContext) -> String;
        fn catalog_table_functions(ctx: &CatalogContext) -> Vec<TableFunctionDef>;
        /// Whether `datastore` defines `schema`, asked when the binder looks a
        /// schema up so an unknown schema is reported as one.
        fn catalog_does_schema_exist(
            transaction: &TransactionContext,
            datastore: &str,
            schema: &str,
        ) -> Result<bool>;
        /// BoundTable lookups are routed by `datastore` (the datastore / database
        /// name) to that datastore's snapshot in the transaction; the table is then
        /// resolved within `schema` of that datastore.
        fn catalog_get_table(
            transaction: &TransactionContext,
            datastore: &str,
            schema: &str,
            name: &str,
        ) -> Result<CatalogGetTableResult>;
        fn catalog_bind_table_function(
            transaction: &TransactionContext,
            name: &str,
            arguments: &CxxVector<Value>,
        ) -> Result<CatalogBindTableFunctionResult>;
        fn clone_table_function_table(table: &OptionalTableWrapper) -> Box<OptionalTableWrapper>;
        fn catalog_get_scalar_function(
            transaction: &TransactionContext,
            name: &str,
        ) -> CatalogGetScalarFunctionResult;
        fn pushdown_filter(table: &mut OptionalTableWrapper, expr: &Expression) -> Result<bool>;
        fn table_estimate_row_count(table: &OptionalTableWrapper) -> CardinalityEstimate;
        fn table_supports_late_materialization(table: &OptionalTableWrapper) -> bool;
    }

    unsafe extern "C++" {
        include!("duckdb-planner/src/duckdb_bridge/cpp/bridge.h");
        type DuckPlannerContext;

        /// Owns the resolved DuckDB plan tree returned by `extract_plan` (and
        /// frees the per-plan catalog entries when dropped); must stay alive
        /// while Rust walks it.
        type PlanHandle;
        /// A DuckDB `LogicalOperator`. Opaque; read via the `lo_*` accessors.
        type LogicalOperator;
        /// A DuckDB bound `Expression`. Opaque; read via the `expr_*` accessors.
        type Expression;
        /// An owning list of synthesized expressions (a `LogicalGet`'s
        /// pushed-down filter conditions). Read via `expr_list_*`.
        type ExpressionList;
        /// A DuckDB `Value` (a query constant or a table-function argument).
        /// Opaque; read via the `value_*` accessors after `value_type`
        /// and `value_is_null`.
        type Value;

        fn new_context(catalog: Box<CatalogContext>) -> Result<UniquePtr<DuckPlannerContext>>;
        /// Plan `query` with `transaction` published for its duration: every
        /// table lookup during binding resolves through it (see
        /// `PivotSchemaCatalogEntry::LookupEntry`). The reference only
        /// needs to outlive this call; the C++ side clears its pointer before
        /// returning.
        fn extract_plan(
            ctx: Pin<&mut DuckPlannerContext>,
            query: &str,
            transaction: &TransactionContext,
        ) -> Result<ExtractPlanResult>;

        fn plan_root(plan: &PlanHandle) -> Result<&LogicalOperator>;

        /// DuckDB's virtual row-id column identifier, used by the late-materialization
        /// row-id stripping to recognise the threaded-up row-id column.
        fn rowid_column_id() -> Result<usize>;

        /// DuckDB's own name for a discriminant of each mirrored enum, e.g.
        /// `"TIME"` for a logical type id. Each errors for a discriminant
        /// DuckDB does not know.
        fn logical_type_id_name(id: u8) -> Result<String>;
        fn expression_type_name(type_id: u8) -> Result<String>;
        fn join_type_name(type_id: u8) -> Result<String>;
        fn logical_operator_type_name(type_id: u8) -> Result<String>;
        fn limit_node_type_name(type_id: u8) -> Result<String>;
        fn order_type_name(type_id: u8) -> Result<String>;
        fn order_by_null_type_name(type_id: u8) -> Result<String>;

        // ---- LogicalOperator: shared structure ----
        /// DuckDB `LogicalOperatorType` discriminant.
        fn lo_type(op: &LogicalOperator) -> Result<u8>;
        /// The operator's display name (`LogicalOperator::GetName`).
        fn lo_name(op: &LogicalOperator) -> Result<String>;
        fn lo_child_count(op: &LogicalOperator) -> Result<usize>;
        fn lo_child(op: &LogicalOperator, index: usize) -> Result<&LogicalOperator>;

        // ---- Projection ----
        fn lo_projection_expr_count(op: &LogicalOperator) -> Result<usize>;
        fn lo_projection_expr(op: &LogicalOperator, index: usize) -> Result<&Expression>;

        // ---- ExpressionGet (VALUES) ----
        fn lo_values_row_count(op: &LogicalOperator) -> Result<usize>;
        fn lo_values_column_count(op: &LogicalOperator) -> Result<usize>;
        fn lo_values_expr(op: &LogicalOperator, row: usize, column: usize) -> Result<&Expression>;

        // ---- Insert ----
        fn lo_insert_take_table(op: &LogicalOperator) -> Result<Box<OptionalTableWrapper>>;
        /// A `LogicalInsert`'s `column_index_map`: one entry per physical table
        /// column, empty when the statement gave no column list. Each entry is
        /// the child column that fills that table column, or "no source" for a
        /// column the statement leaves out.
        fn lo_insert_column_map_count(op: &LogicalOperator) -> Result<usize>;
        fn lo_insert_column_map_has_source(op: &LogicalOperator, index: usize) -> Result<bool>;
        fn lo_insert_column_map_source(op: &LogicalOperator, index: usize) -> Result<usize>;
        fn lo_insert_returns_rows(op: &LogicalOperator) -> Result<bool>;

        // ---- Filter ----
        fn lo_filter_expr_count(op: &LogicalOperator) -> Result<usize>;
        fn lo_filter_expr(op: &LogicalOperator, index: usize) -> Result<&Expression>;
        /// A `LogicalFilter`'s `projection_map` (empty when it passes all child
        /// columns through). Each entry is a child-output column index to keep.
        fn lo_filter_projection_map_count(op: &LogicalOperator) -> Result<usize>;
        fn lo_filter_projection_map_index(op: &LogicalOperator, index: usize) -> Result<usize>;
        /// The type of the filter's `index`th output column, paired with
        /// `lo_filter_projection_map_index` for the replay projection.
        fn lo_filter_type_id(op: &LogicalOperator, index: usize) -> Result<BridgeLogicalType>;

        // ---- OrderBy ----
        fn lo_orderby_count(op: &LogicalOperator) -> Result<usize>;
        fn lo_orderby_direction(op: &LogicalOperator, index: usize) -> Result<u8>;
        fn lo_orderby_null_order(op: &LogicalOperator, index: usize) -> Result<u8>;
        fn lo_orderby_expr(op: &LogicalOperator, index: usize) -> Result<&Expression>;

        // ---- Aggregate ----
        fn lo_aggregate_group_count(op: &LogicalOperator) -> Result<usize>;
        fn lo_aggregate_group(op: &LogicalOperator, index: usize) -> Result<&Expression>;
        fn lo_aggregate_expr_count(op: &LogicalOperator) -> Result<usize>;
        fn lo_aggregate_expr(op: &LogicalOperator, index: usize) -> Result<&Expression>;

        // ---- Distinct ----
        /// Whether the node is `DISTINCT ON (...)` rather than a plain
        /// `DISTINCT` over the whole row.
        fn lo_distinct_is_distinct_on(op: &LogicalOperator) -> Result<bool>;
        fn lo_distinct_target_count(op: &LogicalOperator) -> Result<usize>;
        fn lo_distinct_target(op: &LogicalOperator, index: usize) -> Result<&Expression>;

        // ---- TopN ----
        fn lo_topn_order_count(op: &LogicalOperator) -> Result<usize>;
        fn lo_topn_order_direction(op: &LogicalOperator, index: usize) -> Result<u8>;
        fn lo_topn_order_null_order(op: &LogicalOperator, index: usize) -> Result<u8>;
        fn lo_topn_order_expr(op: &LogicalOperator, index: usize) -> Result<&Expression>;
        fn lo_topn_limit(op: &LogicalOperator) -> Result<usize>;
        fn lo_topn_offset(op: &LogicalOperator) -> Result<usize>;
        fn lo_topn_has_dynamic_filter(op: &LogicalOperator) -> Result<bool>;
        /// Pointer identity of the producer's shared `DynamicFilterData` cell,
        /// used to correlate it with consumer scans (the Rust side assigns slots).
        fn lo_topn_dynamic_filter_data_id(op: &LogicalOperator) -> Result<usize>;
        /// The child-output column index whose boundary the TopN publishes.
        fn lo_topn_dynamic_filter_column(op: &LogicalOperator) -> Result<usize>;
        fn lo_topn_dynamic_filter_comparison(op: &LogicalOperator) -> Result<u8>;

        // ---- Limit ----
        /// DuckDB `LimitNodeType` discriminant of a `LIMIT`/`OFFSET` bound
        /// (`UNSET` / `CONSTANT_VALUE` / a percentage or expression form). The
        /// constant value, when applicable, is read via `lo_limit_value` /
        /// `lo_limit_offset`.
        fn lo_limit_value_kind(op: &LogicalOperator) -> Result<u8>;
        fn lo_limit_value(op: &LogicalOperator) -> Result<usize>;
        fn lo_limit_offset_kind(op: &LogicalOperator) -> Result<u8>;
        fn lo_limit_offset(op: &LogicalOperator) -> Result<usize>;

        // ---- Get: base table ----
        /// Whether this `LogicalGet` reads a base table (vs a table function).
        fn lo_get_has_table(op: &LogicalOperator) -> Result<bool>;
        /// Move the pivot `DuckDBTable` handle out of the scan's catalog entry.
        /// Called once per base-table scan during the walk.
        fn lo_get_take_table(op: &LogicalOperator) -> Result<Box<OptionalTableWrapper>>;
        /// The scan's projected output columns (respecting `projection_ids`),
        /// each a storage column index + its type.
        fn lo_get_output_count(op: &LogicalOperator) -> Result<usize>;
        fn lo_get_output_column(op: &LogicalOperator, index: usize) -> Result<usize>;
        fn lo_get_output_type(op: &LogicalOperator, index: usize) -> Result<BridgeLogicalType>;
        /// For an output column carrying a pushed field extract (`variant_extract`
        /// / `struct_extract`), the referenced path: `depth` segments, each a
        /// field name via `lo_get_output_extract_field`. `depth` 0 means the whole
        /// column is read (no pushdown).
        fn lo_get_output_extract_depth(op: &LogicalOperator, index: usize) -> Result<usize>;
        fn lo_get_output_extract_field(
            op: &LogicalOperator,
            index: usize,
            seg: usize,
        ) -> Result<String>;
        /// The static `col op const` predicates DuckDB pushed into `table_filters`,
        /// rebuilt as expressions (owned by the returned list). Empty list when
        /// there are none.
        fn lo_get_pushed_conditions(op: &LogicalOperator) -> Result<UniquePtr<ExpressionList>>;
        /// Dynamic-filter consumers attached to the scan's `table_filters`.
        fn lo_get_dynamic_filter_count(op: &LogicalOperator) -> Result<usize>;
        fn lo_get_dynamic_filter_data_id(op: &LogicalOperator, index: usize) -> Result<usize>;
        fn lo_get_dynamic_filter_column(op: &LogicalOperator, index: usize) -> Result<usize>;
        fn lo_get_dynamic_filter_comparison(op: &LogicalOperator, index: usize) -> Result<u8>;
        /// Join-filter-pushdown pairing data: the pointer identity of the
        /// scan's shared filter set (0 when no join pushes filters into this
        /// scan), the scan's table index, and the column_ids -> storage column
        /// mapping the join's probe column indexes resolve through.
        fn lo_get_join_filter_set_id(op: &LogicalOperator) -> Result<usize>;
        fn lo_get_table_index(op: &LogicalOperator) -> Result<usize>;
        fn lo_get_column_ids_count(op: &LogicalOperator) -> Result<usize>;
        fn lo_get_storage_column(op: &LogicalOperator, column_ids_index: usize) -> Result<usize>;

        // ---- Get: table function ----
        fn lo_get_function_name(op: &LogicalOperator) -> Result<String>;
        fn lo_get_has_named_params(op: &LogicalOperator) -> Result<bool>;
        fn lo_get_param_count(op: &LogicalOperator) -> Result<usize>;
        fn lo_get_param(op: &LogicalOperator, index: usize) -> Result<&Value>;
        fn lo_get_has_bound_table_function(op: &LogicalOperator) -> Result<bool>;
        fn lo_get_clone_bound_table_function(
            op: &LogicalOperator,
        ) -> Result<Box<OptionalTableWrapper>>;

        // ---- CreateTable ----
        fn lo_create_table_name(op: &LogicalOperator) -> Result<String>;
        /// The target database (datastore) of `CREATE TABLE db.schema.t`, or empty
        /// when unqualified. Routes the create to the right datastore.
        fn lo_create_table_datastore(op: &LogicalOperator) -> Result<String>;
        /// The target schema of `CREATE TABLE db.schema.t`, or empty when the
        /// statement named none. Routes the create to the right schema.
        fn lo_create_table_schema(op: &LogicalOperator) -> Result<String>;
        fn lo_create_column_count(op: &LogicalOperator) -> Result<usize>;
        fn lo_create_column_name(op: &LogicalOperator, index: usize) -> Result<String>;
        fn lo_create_column_type(op: &LogicalOperator, index: usize) -> Result<BridgeLogicalType>;
        fn lo_create_option_count(op: &LogicalOperator) -> Result<usize>;
        fn lo_create_option_key(op: &LogicalOperator, index: usize) -> Result<String>;
        fn lo_create_option_value(op: &LogicalOperator, index: usize) -> Result<String>;
        fn lo_create_if_not_exists(op: &LogicalOperator) -> Result<bool>;
        fn lo_create_or_replace(op: &LogicalOperator) -> Result<bool>;
        fn lo_create_temporary(op: &LogicalOperator) -> Result<bool>;
        fn lo_create_has_query(op: &LogicalOperator) -> Result<bool>;
        fn lo_create_constraint_count(op: &LogicalOperator) -> Result<usize>;

        // ---- CreateSchema ----
        fn lo_create_schema_name(op: &LogicalOperator) -> Result<String>;
        /// The target database (datastore) of `CREATE SCHEMA db.s`, or empty when
        /// unqualified. Routes the create to the right datastore.
        fn lo_create_schema_datastore(op: &LogicalOperator) -> Result<String>;
        fn lo_create_schema_if_not_exists(op: &LogicalOperator) -> Result<bool>;
        fn lo_create_schema_or_replace(op: &LogicalOperator) -> Result<bool>;

        // ---- Drop ----
        /// Whether the `LOGICAL_DROP` targets a table (vs. a schema, view, ...).
        fn lo_drop_is_table(op: &LogicalOperator) -> Result<bool>;
        /// The kind of catalog entry the drop targets, as DuckDB spells it
        /// (`TABLE`, `SCHEMA`, `VIEW`, ...): for the unsupported-kind error.
        fn lo_drop_entry_kind(op: &LogicalOperator) -> Result<String>;
        fn lo_drop_name(op: &LogicalOperator) -> Result<String>;
        /// The schema of the dropped entry, or empty when unresolved (an
        /// `IF EXISTS` drop of a missing entry keeps whatever was written).
        fn lo_drop_schema(op: &LogicalOperator) -> Result<String>;
        /// The target database (datastore), or empty when unresolved.
        fn lo_drop_datastore(op: &LogicalOperator) -> Result<String>;
        fn lo_drop_if_exists(op: &LogicalOperator) -> Result<bool>;
        fn lo_drop_cascade(op: &LogicalOperator) -> Result<bool>;

        // ---- Explain ----
        /// Whether the `LOGICAL_EXPLAIN` is `EXPLAIN ANALYZE` (vs plain
        /// `EXPLAIN`).
        fn lo_explain_is_analyze(op: &LogicalOperator) -> Result<bool>;

        // ---- Set / Reset ----
        fn lo_set_name(op: &LogicalOperator) -> Result<String>;
        fn lo_set_value(op: &LogicalOperator) -> Result<String>;
        fn lo_reset_name(op: &LogicalOperator) -> Result<String>;

        // ---- Transaction ----
        /// The `TransactionType` discriminant carried by the
        /// `LOGICAL_TRANSACTION`'s `TransactionInfo` (BEGIN, COMMIT or
        /// ROLLBACK).
        fn lo_transaction_type(op: &LogicalOperator) -> Result<u8>;

        // ---- Vacuum ----
        /// Reject table-specific vacuum and ANALYZE, which Pivot does not implement.
        fn lo_vacuum_validate(op: &LogicalOperator) -> Result<()>;

        // ---- Compact ----
        /// The datastore `COMPACT db.s.t` named, or empty when unqualified.
        fn lo_compact_datastore(op: &LogicalOperator) -> Result<String>;
        /// The schema the statement named, or empty when unqualified.
        fn lo_compact_schema(op: &LogicalOperator) -> Result<String>;
        fn lo_compact_table(op: &LogicalOperator) -> Result<String>;
        /// `COMPACT ... FINAL`: keep sweeping until a pass merges nothing.
        fn lo_compact_final(op: &LogicalOperator) -> Result<bool>;

        // ---- CopyFromStdin ----
        /// Take the table binding DuckDB resolved for this COPY target.
        fn lo_copy_stdin_take_table(op: &LogicalOperator) -> Result<Box<OptionalTableWrapper>>;
        /// The explicit column list resolved to physical column positions;
        /// empty when the statement targets every table column.
        fn lo_copy_stdin_column_count(op: &LogicalOperator) -> Result<usize>;
        fn lo_copy_stdin_column_index(op: &LogicalOperator, index: usize) -> Result<usize>;
        /// The FORMAT option as written, empty when the statement gave none.
        fn lo_copy_stdin_format(op: &LogicalOperator) -> Result<String>;
        /// The remaining `WITH (...)` options: bound constant values rendered
        /// as text, in name order; a bare flag (e.g. HEADER) has no values.
        fn lo_copy_stdin_option_count(op: &LogicalOperator) -> Result<usize>;
        fn lo_copy_stdin_option_name(op: &LogicalOperator, index: usize) -> Result<String>;
        fn lo_copy_stdin_option_value_count(op: &LogicalOperator, index: usize) -> Result<usize>;
        fn lo_copy_stdin_option_value(
            op: &LogicalOperator,
            index: usize,
            value_index: usize,
        ) -> Result<String>;

        // ---- CreateUser ----
        fn lo_create_user_name(op: &LogicalOperator) -> Result<String>;
        /// Whether a PASSWORD clause was given.
        fn lo_create_user_has_password(op: &LogicalOperator) -> Result<bool>;
        /// The password, meaningful only when a PASSWORD clause was given.
        fn lo_create_user_password(op: &LogicalOperator) -> Result<String>;

        // ---- ComparisonJoin: general accessors ----
        /// DuckDB `JoinType` discriminant.
        /// The index a materialized CTE publishes its rows under.
        fn lo_cte_table_index(op: &LogicalOperator) -> Result<usize>;

        /// The CTE index a reference reads.
        fn lo_cte_ref_index(op: &LogicalOperator) -> Result<usize>;

        /// Rows in a CHUNK_GET's constant collection.
        fn lo_chunk_get_row_count(op: &LogicalOperator) -> Result<usize>;
        fn lo_chunk_get_column_count(op: &LogicalOperator) -> Result<usize>;
        /// One cell of the collection, owned: DuckDB hands the value out by
        /// value rather than by reference.
        fn lo_chunk_get_value(
            op: &LogicalOperator,
            column: usize,
            row: usize,
        ) -> Result<UniquePtr<Value>>;

        fn lo_join_type(op: &LogicalOperator) -> Result<u8>;
        fn lo_join_condition_count(op: &LogicalOperator) -> Result<usize>;
        /// Whether the condition is a left/right comparison. When false, the
        /// whole predicate is a single expression over both sides, read via
        /// `lo_join_condition_expression`; the left/right/comparison accessors
        /// throw for it.
        fn lo_join_condition_is_comparison(op: &LogicalOperator, index: usize) -> Result<bool>;
        /// The single-expression form's predicate, bound against the two
        /// children's concatenated outputs (all LHS columns, then all RHS).
        fn lo_join_condition_expression(op: &LogicalOperator, index: usize) -> Result<&Expression>;
        fn lo_join_condition_left(op: &LogicalOperator, index: usize) -> Result<&Expression>;
        fn lo_join_condition_right(op: &LogicalOperator, index: usize) -> Result<&Expression>;
        /// DuckDB `ExpressionType` discriminant of the condition's comparison.
        fn lo_join_condition_comparison(op: &LogicalOperator, index: usize) -> Result<u8>;

        /// Join filter pushdown: the optimizer's record of which probe-side
        /// scans can be narrowed by filters derived from this join's build-side
        /// key values. Targets pair with their scan by the pointer identity of
        /// the shared filter set (`lo_get_join_filter_set_id`); each target
        /// column names the get's column_ids index to filter plus (via
        /// `lo_join_pushdown_condition_index`) the join condition whose build
        /// values bound it.
        fn lo_join_pushdown_target_count(op: &LogicalOperator) -> Result<usize>;
        fn lo_join_pushdown_set_id(op: &LogicalOperator, target: usize) -> Result<usize>;
        fn lo_join_pushdown_column_count(op: &LogicalOperator, target: usize) -> Result<usize>;
        fn lo_join_pushdown_probe_table(
            op: &LogicalOperator,
            target: usize,
            index: usize,
        ) -> Result<usize>;
        fn lo_join_pushdown_probe_column(
            op: &LogicalOperator,
            target: usize,
            index: usize,
        ) -> Result<usize>;
        fn lo_join_pushdown_condition_index(op: &LogicalOperator, index: usize) -> Result<usize>;
        /// The join's projection maps: which of each child's output columns
        /// survive in the join's output (empty = all of them).
        fn lo_join_left_projection_map_count(op: &LogicalOperator) -> Result<usize>;
        fn lo_join_left_projection_map_index(op: &LogicalOperator, index: usize) -> Result<usize>;
        fn lo_join_right_projection_map_count(op: &LogicalOperator) -> Result<usize>;
        fn lo_join_right_projection_map_index(op: &LogicalOperator, index: usize) -> Result<usize>;

        // ---- DelimJoin / DelimGet ----
        /// The expressions (over the de-duplicated side's output) whose distinct
        /// values every DELIM_GET under the other side scans.
        fn lo_delim_join_column_count(op: &LogicalOperator) -> Result<usize>;
        fn lo_delim_join_column(op: &LogicalOperator, index: usize) -> Result<&Expression>;
        /// False: the LHS is de-duplicated and the DELIM_GETs sit under the RHS.
        /// True: the join was flipped and the roles reverse.
        fn lo_delim_join_is_flipped(op: &LogicalOperator) -> Result<bool>;
        fn lo_delim_get_column_count(op: &LogicalOperator) -> Result<usize>;
        fn lo_delim_get_column_type(
            op: &LogicalOperator,
            index: usize,
        ) -> Result<BridgeLogicalType>;

        // ---- ExpressionList (owned synthesized expressions) ----
        fn expr_list_count(list: &ExpressionList) -> Result<usize>;
        fn expr_list_get(list: &ExpressionList, index: usize) -> Result<&Expression>;

        // ---- Expression: shared ----
        /// DuckDB `ExpressionType` discriminant.
        fn expr_type(expr: &Expression) -> Result<u8>;
        /// The type of the expression's result.
        fn expr_return_type(expr: &Expression) -> Result<BridgeLogicalType>;
        fn expr_has_alias(expr: &Expression) -> Result<bool>;
        fn expr_alias(expr: &Expression) -> Result<String>;

        // BoundReferenceExpression / BoundColumnRefExpression
        fn expr_ref_index(expr: &Expression) -> Result<usize>;
        fn expr_columnref_index(expr: &Expression) -> Result<usize>;

        // BoundComparisonExpression
        fn expr_comparison_left(expr: &Expression) -> Result<&Expression>;
        fn expr_comparison_right(expr: &Expression) -> Result<&Expression>;

        // BoundBetweenExpression
        fn expr_between_input(expr: &Expression) -> Result<&Expression>;
        fn expr_between_lower(expr: &Expression) -> Result<&Expression>;
        fn expr_between_upper(expr: &Expression) -> Result<&Expression>;
        fn expr_between_lower_inclusive(expr: &Expression) -> Result<bool>;
        fn expr_between_upper_inclusive(expr: &Expression) -> Result<bool>;

        // BoundConstantExpression
        fn expr_constant(expr: &Expression) -> Result<&Value>;

        // ---- Value: typed accessors (shared by constants and table-function
        // arguments). Read `value_type` and `value_is_null` first, then the
        // matching accessor: the typed accessors are only valid on a non-NULL
        // value.
        fn value_type(v: &Value) -> Result<BridgeLogicalType>;
        fn value_is_null(v: &Value) -> Result<bool>;
        fn value_bool(v: &Value) -> Result<bool>;
        fn value_i8(v: &Value) -> Result<i8>;
        fn value_i16(v: &Value) -> Result<i16>;
        fn value_i32(v: &Value) -> Result<i32>;
        fn value_i64(v: &Value) -> Result<i64>;
        fn value_u8(v: &Value) -> Result<u8>;
        fn value_u16(v: &Value) -> Result<u16>;
        fn value_u32(v: &Value) -> Result<u32>;
        fn value_u64(v: &Value) -> Result<u64>;
        fn value_f32(v: &Value) -> Result<f32>;
        fn value_f64(v: &Value) -> Result<f64>;
        fn value_string(v: &Value) -> Result<String>;
        fn value_decimal(v: &Value) -> Result<BridgeDecimalValue>;
        fn value_hugeint(v: &Value) -> Result<BridgeHugeint>;
        fn value_date(v: &Value) -> Result<i32>;
        fn value_timestamp(v: &Value) -> Result<i64>;
        fn value_timestamp_tz(v: &Value) -> Result<i64>;
        fn value_interval_months(v: &Value) -> Result<i32>;
        fn value_interval_days(v: &Value) -> Result<i32>;
        fn value_interval_micros(v: &Value) -> Result<i64>;

        // BoundAggregateExpression
        fn expr_aggregate_name(expr: &Expression) -> Result<String>;
        fn expr_aggregate_distinct(expr: &Expression) -> Result<bool>;
        fn expr_aggregate_child_count(expr: &Expression) -> Result<usize>;
        fn expr_aggregate_child(expr: &Expression, index: usize) -> Result<&Expression>;
        fn expr_aggregate_has_filter(expr: &Expression) -> Result<bool>;

        // BoundFunctionExpression
        fn expr_function_name(expr: &Expression) -> Result<String>;
        fn expr_function_child_count(expr: &Expression) -> Result<usize>;
        fn expr_function_child(expr: &Expression, index: usize) -> Result<&Expression>;

        // BoundOperatorExpression (COMPARE_IN, OPERATOR_NOT)
        fn expr_operator_child_count(expr: &Expression) -> Result<usize>;
        fn expr_operator_child(expr: &Expression, index: usize) -> Result<&Expression>;

        // BoundConjunctionExpression
        fn expr_conjunction_child_count(expr: &Expression) -> Result<usize>;
        fn expr_conjunction_child(expr: &Expression, index: usize) -> Result<&Expression>;

        // BoundCaseExpression
        fn expr_case_check_count(expr: &Expression) -> Result<usize>;
        fn expr_case_when(expr: &Expression, index: usize) -> Result<&Expression>;
        fn expr_case_then(expr: &Expression, index: usize) -> Result<&Expression>;
        fn expr_case_else(expr: &Expression) -> Result<&Expression>;

        // BoundCastExpression
        fn expr_cast_child(expr: &Expression) -> Result<&Expression>;
        fn expr_cast_is_try(expr: &Expression) -> Result<bool>;
    }
}
