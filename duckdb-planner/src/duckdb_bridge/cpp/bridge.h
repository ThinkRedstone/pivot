#pragma once
#include "rust/cxx.h"
#include "duckdb.hpp"
#include "duckdb/planner/table_filter_set.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include <memory>
#include <vector>

struct CatalogContext;
struct TransactionContext;
struct OptionalTableWrapper;
struct ExtractPlanResult;
struct BridgeLogicalType;
struct BridgeDecimalValue;
struct BridgeHugeint;

// DuckDB exceptions serialize themselves as JSON; returns the embedded
// exception type and message when present, the raw what() otherwise.
std::string bridge_exception_message(const std::exception &e);

// Every bridge function below is declared with a Result return on the Rust
// side, so cxx routes it through this handler: a C++ exception becomes a Rust
// error instead of unwinding into the (noexcept) FFI frame and terminating the
// process. The catch-all clause covers throws that are not std::exceptions.
namespace rust {
namespace behavior {
template <typename Try, typename Fail>
static void trycatch(Try &&func, Fail &&fail) noexcept try {
	func();
} catch (const std::exception &e) {
	fail(bridge_exception_message(e).c_str());
} catch (...) {
	fail("unknown C++ exception");
}
} // namespace behavior
} // namespace rust

// The Rust plan builder reads DuckDB's own objects directly, so expose them to
// CXX as opaque types by their real names.
using LogicalOperator = duckdb::LogicalOperator;
using Expression = duckdb::Expression;
using Value = duckdb::Value;

struct DuckPlannerContext {
	rust::Box<CatalogContext> catalog;  // owns the CatalogContext; must outlive db
	duckdb::DBConfig config;
	duckdb::DuckDB db;
	duckdb::Connection con;

	explicit DuckPlannerContext(rust::Box<CatalogContext> catalog);
};

// Owns DuckDB's resolved plan tree so it stays alive while Rust walks it. The
// pivot `DuckDBTable` handles are moved out of the catalog entries during that
// walk, so the per-plan catalog entries are only cleared when this handle is
// dropped (after the walk), not at the end of `extract_plan`.
struct PlanHandle {
	duckdb::unique_ptr<duckdb::LogicalOperator> root;
	DuckPlannerContext *ctx;
	~PlanHandle();
};

// Owns the expressions synthesized for a LogicalGet's pushed-down filters (the
// `ToExpression` results), so they outlive the per-condition Rust read.
struct ExpressionList {
	std::vector<duckdb::unique_ptr<duckdb::Expression>> exprs;
};

std::unique_ptr<DuckPlannerContext> new_context(rust::Box<CatalogContext> catalog);
ExtractPlanResult extract_plan(DuckPlannerContext &ctx, rust::Str query,
                               const TransactionContext &transaction);

const LogicalOperator &plan_root(const PlanHandle &plan);
size_t rowid_column_id();

// DuckDB's own name for a discriminant of each enum the Rust side mirrors (a
// logical type id of 27 is "INTERVAL"), so a message naming one of them prints
// the name and not a number. The first three have a DuckDB spelling meant for
// messages; the rest are named after their enumerator. Throw for a
// discriminant DuckDB does not know.
rust::String logical_type_id_name(uint8_t id);
rust::String expression_type_name(uint8_t type);
rust::String join_type_name(uint8_t type);
rust::String logical_operator_type_name(uint8_t type);
rust::String limit_node_type_name(uint8_t type);
rust::String order_type_name(uint8_t type);
rust::String order_by_null_type_name(uint8_t type);


// ---- LogicalOperator: shared structure ----
uint8_t lo_type(const LogicalOperator &op);
rust::String lo_name(const LogicalOperator &op);
size_t lo_child_count(const LogicalOperator &op);
const LogicalOperator &lo_child(const LogicalOperator &op, size_t index);

// ---- Projection ----
size_t lo_projection_expr_count(const LogicalOperator &op);
const Expression &lo_projection_expr(const LogicalOperator &op, size_t index);

// ---- ExpressionGet (VALUES) ----
size_t lo_values_row_count(const LogicalOperator &op);
size_t lo_values_column_count(const LogicalOperator &op);
const Expression &lo_values_expr(const LogicalOperator &op, size_t row, size_t column);

// ---- Insert ----
rust::Box<OptionalTableWrapper> lo_insert_take_table(const LogicalOperator &op);
size_t lo_insert_column_map_count(const LogicalOperator &op);
bool lo_insert_column_map_has_source(const LogicalOperator &op, size_t index);
size_t lo_insert_column_map_source(const LogicalOperator &op, size_t index);
bool lo_insert_returns_rows(const LogicalOperator &op);

// ---- Filter ----
size_t lo_filter_expr_count(const LogicalOperator &op);
const Expression &lo_filter_expr(const LogicalOperator &op, size_t index);
size_t lo_filter_projection_map_count(const LogicalOperator &op);
size_t lo_filter_projection_map_index(const LogicalOperator &op, size_t index);
BridgeLogicalType lo_filter_type_id(const LogicalOperator &op, size_t index);

// ---- OrderBy ----
size_t lo_orderby_count(const LogicalOperator &op);
uint8_t lo_orderby_direction(const LogicalOperator &op, size_t index);
uint8_t lo_orderby_null_order(const LogicalOperator &op, size_t index);
const Expression &lo_orderby_expr(const LogicalOperator &op, size_t index);

// ---- Aggregate ----
size_t lo_aggregate_group_count(const LogicalOperator &op);
const Expression &lo_aggregate_group(const LogicalOperator &op, size_t index);
size_t lo_aggregate_expr_count(const LogicalOperator &op);
const Expression &lo_aggregate_expr(const LogicalOperator &op, size_t index);

// ---- Distinct ----
bool lo_distinct_is_distinct_on(const LogicalOperator &op);
size_t lo_distinct_target_count(const LogicalOperator &op);
const Expression &lo_distinct_target(const LogicalOperator &op, size_t index);

// ---- TopN ----
size_t lo_topn_order_count(const LogicalOperator &op);
uint8_t lo_topn_order_direction(const LogicalOperator &op, size_t index);
uint8_t lo_topn_order_null_order(const LogicalOperator &op, size_t index);
const Expression &lo_topn_order_expr(const LogicalOperator &op, size_t index);
size_t lo_topn_limit(const LogicalOperator &op);
size_t lo_topn_offset(const LogicalOperator &op);
bool lo_topn_has_dynamic_filter(const LogicalOperator &op);
size_t lo_topn_dynamic_filter_data_id(const LogicalOperator &op);
size_t lo_topn_dynamic_filter_column(const LogicalOperator &op);
uint8_t lo_topn_dynamic_filter_comparison(const LogicalOperator &op);

// ---- Limit ----
uint8_t lo_limit_value_kind(const LogicalOperator &op);
size_t lo_limit_value(const LogicalOperator &op);
uint8_t lo_limit_offset_kind(const LogicalOperator &op);
size_t lo_limit_offset(const LogicalOperator &op);

// ---- Get: base table ----
bool lo_get_has_table(const LogicalOperator &op);
rust::Box<OptionalTableWrapper> lo_get_take_table(const LogicalOperator &op);
size_t lo_get_output_count(const LogicalOperator &op);
size_t lo_get_output_column(const LogicalOperator &op, size_t index);
BridgeLogicalType lo_get_output_type(const LogicalOperator &op, size_t index);
size_t lo_get_output_extract_depth(const LogicalOperator &op, size_t index);
rust::String lo_get_output_extract_field(const LogicalOperator &op, size_t index, size_t seg);
std::unique_ptr<ExpressionList> lo_get_pushed_conditions(const LogicalOperator &op);
size_t lo_get_dynamic_filter_count(const LogicalOperator &op);
size_t lo_get_dynamic_filter_data_id(const LogicalOperator &op, size_t index);
size_t lo_get_dynamic_filter_column(const LogicalOperator &op, size_t index);
uint8_t lo_get_dynamic_filter_comparison(const LogicalOperator &op, size_t index);
size_t lo_get_join_filter_set_id(const LogicalOperator &op);
size_t lo_get_table_index(const LogicalOperator &op);
size_t lo_get_column_ids_count(const LogicalOperator &op);
size_t lo_get_storage_column(const LogicalOperator &op, size_t column_ids_index);

// ---- Get: table function ----
rust::String lo_get_function_name(const LogicalOperator &op);
bool lo_get_has_named_params(const LogicalOperator &op);
size_t lo_get_param_count(const LogicalOperator &op);
const Value &lo_get_param(const LogicalOperator &op, size_t index);
bool lo_get_has_bound_table_function(const LogicalOperator &op);
rust::Box<OptionalTableWrapper> lo_get_clone_bound_table_function(const LogicalOperator &op);

// ---- CreateTable ----
rust::String lo_create_table_name(const LogicalOperator &op);
rust::String lo_create_table_datastore(const LogicalOperator &op);
rust::String lo_create_table_schema(const LogicalOperator &op);
rust::String lo_create_schema_name(const LogicalOperator &op);
rust::String lo_create_schema_datastore(const LogicalOperator &op);
bool lo_create_schema_if_not_exists(const LogicalOperator &op);
bool lo_create_schema_or_replace(const LogicalOperator &op);
bool lo_drop_is_table(const LogicalOperator &op);
rust::String lo_drop_entry_kind(const LogicalOperator &op);
rust::String lo_drop_name(const LogicalOperator &op);
rust::String lo_drop_schema(const LogicalOperator &op);
rust::String lo_drop_datastore(const LogicalOperator &op);
bool lo_drop_if_exists(const LogicalOperator &op);
bool lo_drop_cascade(const LogicalOperator &op);
bool lo_explain_is_analyze(const LogicalOperator &op);
size_t lo_create_column_count(const LogicalOperator &op);
rust::String lo_create_column_name(const LogicalOperator &op, size_t index);
BridgeLogicalType lo_create_column_type(const LogicalOperator &op, size_t index);
size_t lo_create_option_count(const LogicalOperator &op);
rust::String lo_create_option_key(const LogicalOperator &op, size_t index);
rust::String lo_create_option_value(const LogicalOperator &op, size_t index);
bool lo_create_if_not_exists(const LogicalOperator &op);
bool lo_create_or_replace(const LogicalOperator &op);
bool lo_create_temporary(const LogicalOperator &op);
bool lo_create_has_query(const LogicalOperator &op);
size_t lo_create_constraint_count(const LogicalOperator &op);

// ---- Set / Reset ----
rust::String lo_set_name(const LogicalOperator &op);
rust::String lo_set_value(const LogicalOperator &op);
rust::String lo_reset_name(const LogicalOperator &op);

// ---- Transaction ----
uint8_t lo_transaction_type(const LogicalOperator &op);

// ---- Vacuum ----
void lo_vacuum_validate(const LogicalOperator &op);

// ---- Compact ----
rust::String lo_compact_datastore(const LogicalOperator &op);
rust::String lo_compact_schema(const LogicalOperator &op);
rust::String lo_compact_table(const LogicalOperator &op);
bool lo_compact_final(const LogicalOperator &op);

// ---- CopyFromStdin ----
rust::Box<OptionalTableWrapper> lo_copy_stdin_take_table(const LogicalOperator &op);
size_t lo_copy_stdin_column_count(const LogicalOperator &op);
size_t lo_copy_stdin_column_index(const LogicalOperator &op, size_t index);
rust::String lo_copy_stdin_format(const LogicalOperator &op);
size_t lo_copy_stdin_option_count(const LogicalOperator &op);
rust::String lo_copy_stdin_option_name(const LogicalOperator &op, size_t index);
size_t lo_copy_stdin_option_value_count(const LogicalOperator &op, size_t index);
rust::String lo_copy_stdin_option_value(const LogicalOperator &op, size_t index, size_t value_index);

// ---- CreateUser ----
rust::String lo_create_user_name(const LogicalOperator &op);
bool lo_create_user_has_password(const LogicalOperator &op);
rust::String lo_create_user_password(const LogicalOperator &op);

// ---- CTE ----
size_t lo_cte_table_index(const LogicalOperator &op);
size_t lo_cte_ref_index(const LogicalOperator &op);

// ---- ColumnDataGet (CHUNK_GET) ----
size_t lo_chunk_get_row_count(const LogicalOperator &op);
size_t lo_chunk_get_column_count(const LogicalOperator &op);
std::unique_ptr<Value> lo_chunk_get_value(const LogicalOperator &op, size_t column, size_t row);

// ---- ComparisonJoin: general accessors ----
uint8_t lo_join_type(const LogicalOperator &op);
size_t lo_join_condition_count(const LogicalOperator &op);
bool lo_join_condition_is_comparison(const LogicalOperator &op, size_t index);
const Expression &lo_join_condition_expression(const LogicalOperator &op, size_t index);
const Expression &lo_join_condition_left(const LogicalOperator &op, size_t index);
const Expression &lo_join_condition_right(const LogicalOperator &op, size_t index);
uint8_t lo_join_condition_comparison(const LogicalOperator &op, size_t index);
size_t lo_join_pushdown_target_count(const LogicalOperator &op);
size_t lo_join_pushdown_set_id(const LogicalOperator &op, size_t target);
size_t lo_join_pushdown_column_count(const LogicalOperator &op, size_t target);
size_t lo_join_pushdown_probe_table(const LogicalOperator &op, size_t target, size_t index);
size_t lo_join_pushdown_probe_column(const LogicalOperator &op, size_t target, size_t index);
size_t lo_join_pushdown_condition_index(const LogicalOperator &op, size_t index);
size_t lo_join_left_projection_map_count(const LogicalOperator &op);
size_t lo_join_left_projection_map_index(const LogicalOperator &op, size_t index);
size_t lo_join_right_projection_map_count(const LogicalOperator &op);
size_t lo_join_right_projection_map_index(const LogicalOperator &op, size_t index);

// ---- DelimJoin / DelimGet ----
size_t lo_delim_join_column_count(const LogicalOperator &op);
const Expression &lo_delim_join_column(const LogicalOperator &op, size_t index);
bool lo_delim_join_is_flipped(const LogicalOperator &op);
size_t lo_delim_get_column_count(const LogicalOperator &op);
BridgeLogicalType lo_delim_get_column_type(const LogicalOperator &op, size_t index);

// ---- ExpressionList ----
size_t expr_list_count(const ExpressionList &list);
const Expression &expr_list_get(const ExpressionList &list, size_t index);

// ---- Expression: shared ----
uint8_t expr_type(const Expression &expr);
BridgeLogicalType expr_return_type(const Expression &expr);
bool expr_has_alias(const Expression &expr);
rust::String expr_alias(const Expression &expr);

const Value &expr_constant(const Expression &expr);

BridgeLogicalType value_type(const Value &v);
bool value_is_null(const Value &v);
bool value_bool(const Value &v);
int8_t value_i8(const Value &v);
int16_t value_i16(const Value &v);
int32_t value_i32(const Value &v);
int64_t value_i64(const Value &v);
uint8_t value_u8(const Value &v);
uint16_t value_u16(const Value &v);
uint32_t value_u32(const Value &v);
uint64_t value_u64(const Value &v);
float value_f32(const Value &v);
double value_f64(const Value &v);
rust::String value_string(const Value &v);
BridgeDecimalValue value_decimal(const Value &v);
BridgeHugeint value_hugeint(const Value &v);
int32_t value_date(const Value &v);
int64_t value_timestamp(const Value &v);
int64_t value_timestamp_tz(const Value &v);
int32_t value_interval_months(const Value &v);
int32_t value_interval_days(const Value &v);
int64_t value_interval_micros(const Value &v);

size_t expr_ref_index(const Expression &expr);
size_t expr_columnref_index(const Expression &expr);

const Expression &expr_comparison_left(const Expression &expr);
const Expression &expr_comparison_right(const Expression &expr);

const Expression &expr_between_input(const Expression &expr);
const Expression &expr_between_lower(const Expression &expr);
const Expression &expr_between_upper(const Expression &expr);
bool expr_between_lower_inclusive(const Expression &expr);
bool expr_between_upper_inclusive(const Expression &expr);

rust::String expr_aggregate_name(const Expression &expr);
bool expr_aggregate_distinct(const Expression &expr);
size_t expr_aggregate_child_count(const Expression &expr);
const Expression &expr_aggregate_child(const Expression &expr, size_t index);
bool expr_aggregate_has_filter(const Expression &expr);

rust::String expr_function_name(const Expression &expr);
size_t expr_function_child_count(const Expression &expr);
const Expression &expr_function_child(const Expression &expr, size_t index);

size_t expr_operator_child_count(const Expression &expr);
const Expression &expr_operator_child(const Expression &expr, size_t index);

size_t expr_conjunction_child_count(const Expression &expr);
const Expression &expr_conjunction_child(const Expression &expr, size_t index);

size_t expr_case_check_count(const Expression &expr);
const Expression &expr_case_when(const Expression &expr, size_t index);
const Expression &expr_case_then(const Expression &expr, size_t index);
const Expression &expr_case_else(const Expression &expr);

const Expression &expr_cast_child(const Expression &expr);
bool expr_cast_is_try(const Expression &expr);
