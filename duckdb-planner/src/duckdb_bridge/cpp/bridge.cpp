#include "duckdb-planner/src/duckdb_bridge/mod.rs.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/bridge.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/storage_info.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/catalog/table_entry.h"
#include "duckdb-planner/src/duckdb_bridge/cpp/catalog/schema_entry.h"
#include "duckdb/common/enum_util.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/main/client_data.hpp"
#include "duckdb/catalog/catalog_search_path.hpp"
#include "duckdb/parser/keyword_helper.hpp"
#include "duckdb/parser/parsed_data/create_table_info.hpp"
#include "duckdb/planner/operator/logical_copy_from_stdin.hpp"
#include "duckdb/planner/operator/logical_explain.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_order.hpp"
#include "duckdb/planner/operator/logical_aggregate.hpp"
#include "duckdb/planner/operator/logical_distinct.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_top_n.hpp"
#include "duckdb/planner/operator/logical_limit.hpp"
#include "duckdb/planner/operator/logical_insert.hpp"
#include "duckdb/planner/operator/logical_expression_get.hpp"
#include "duckdb/planner/operator/logical_create_table.hpp"
#include "duckdb/planner/operator/logical_create.hpp"
#include "duckdb/parser/parsed_data/create_schema_info.hpp"
#include "duckdb/planner/operator/logical_simple.hpp"
#include "duckdb/parser/parsed_data/drop_info.hpp"
#include "duckdb/parser/parsed_data/transaction_info.hpp"
#include "duckdb/execution/operator/join/join_filter_pushdown.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_delim_get.hpp"
#include "duckdb/planner/operator/logical_materialized_cte.hpp"
#include "duckdb/planner/operator/logical_cteref.hpp"
#include "duckdb/planner/operator/logical_column_data_get.hpp"
#include "duckdb/planner/operator/logical_set.hpp"
#include "duckdb/planner/operator/logical_reset.hpp"
#include "duckdb/planner/operator/logical_compact.hpp"
#include "duckdb/planner/operator/logical_vacuum.hpp"
#include "duckdb/planner/operator/logical_create_user.hpp"
#include "duckdb/catalog/catalog_entry/table_catalog_entry.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression/bound_comparison_expression.hpp"
#include "duckdb/planner/expression/bound_between_expression.hpp"
#include "duckdb/planner/expression/bound_constant_expression.hpp"
#include "duckdb/planner/expression/bound_aggregate_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_operator_expression.hpp"
#include "duckdb/planner/expression/bound_conjunction_expression.hpp"
#include "duckdb/planner/expression/bound_case_expression.hpp"
#include "duckdb/planner/expression/bound_cast_expression.hpp"
#include "duckdb/common/types/interval.hpp"
#include "duckdb/execution/column_binding_resolver.hpp"
#include "duckdb/planner/planner.hpp"
#include "duckdb/optimizer/optimizer.hpp"
#include "duckdb/planner/filter/expression_filter.hpp"
#include "duckdb/planner/filter/constant_filter.hpp"
#include "duckdb/planner/filter/conjunction_filter.hpp"
#include "duckdb/planner/filter/dynamic_filter.hpp"
#include "duckdb/planner/filter/optional_filter.hpp"

// nlohmann/json is used only to parse DuckDB's own exception text (which DuckDB
// emits as JSON); the C++/Rust bridge itself no longer serializes anything.
#include <nlohmann/json.hpp>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

using std::string;

struct UnsupportedPlanError : public std::runtime_error {
	using std::runtime_error::runtime_error;
};

// Cast a const LogicalOperator/Expression to the concrete DuckDB subtype. The
// accessors take const references (cxx passes shared references as const), but
// some reads (taking the table handle, ToExpression) need a mutable object;
// const_cast keeps the accessor signatures uniform.
template <typename T>
static T &as(const LogicalOperator &op) {
	return const_cast<LogicalOperator &>(op).Cast<T>();
}
template <typename T>
static const T &as_expr(const Expression &expr) {
	return expr.Cast<T>();
}

std::string bridge_exception_message(const std::exception &e) {
	using json = nlohmann::json;
	auto parsed = json::parse(e.what(), nullptr, false);
	if (parsed.is_object()) {
		auto message_it = parsed.find("exception_message");
		if (message_it != parsed.end() && message_it->is_string()) {
			auto type_it = parsed.find("exception_type");
			if (type_it != parsed.end() && type_it->is_string()) {
				return type_it->get<string>() + ": " + message_it->get<string>();
			}
			return message_it->get<string>();
		}
	}
	return e.what();
}

static ExtractPlanResult make_error(const string &kind, const string &message,
                                    std::optional<string> position = std::nullopt) {
	ExtractPlanResult result;
	result.error_kind = rust::String::lossy(kind);
	result.error_message = rust::String::lossy(message);
	result.has_error_position = position.has_value();
	result.error_position = rust::String::lossy(position.value_or(""));
	return result;
}

static ExtractPlanResult build_duckdb_error(const duckdb::Exception &e) {
	using json = nlohmann::json;
	auto parsed = json::parse(e.what(), nullptr, false);
	string message = e.what();
	std::optional<string> position;

	if (parsed.is_object()) {
		if (auto message_it = parsed.find("exception_message");
		    message_it != parsed.end() && message_it->is_string()) {
			message = message_it->get<string>();
		}
		if (auto position_it = parsed.find("position"); position_it != parsed.end()) {
			if (position_it->is_string()) {
				position = position_it->get<string>();
			} else if (position_it->is_number_integer()) {
				position = std::to_string(position_it->get<int64_t>());
			} else if (position_it->is_number_unsigned()) {
				position = std::to_string(position_it->get<uint64_t>());
			} else if (!position_it->is_null()) {
				position = position_it->dump();
			}
		}
	}

	return make_error("duckdb_planning", message, position);
}

// Create new context for an in-memory DB. The context only binds and optimizes
// statements and never executes one, so it runs with a single thread: by
// default DuckDB starts a worker per core, which makes creating a context
// cost milliseconds and leaves those workers idle for the context's lifetime.
DuckPlannerContext::DuckPlannerContext(rust::Box<CatalogContext> catalog)
    : catalog(std::move(catalog)),
      config({{"threads", duckdb::Value::BIGINT(1)}}, false),
      db(nullptr, &this->config),
      con(db) {
        auto timezone_result = con.Query("SET TimeZone='UTC'");
        if (timezone_result->HasError()) {
                throw std::runtime_error(timezone_result->GetError());
        }

        auto disable_result = con.Query(
            "SET disabled_optimizers='compressed_materialization,empty_result_pullup,regex_range'");
        if (disable_result->HasError()) {
                throw std::runtime_error(disable_result->GetError());
        }

	// Point the pivotdb storage extension at our catalog context, then attach one
	// DuckDB database per datastore. All attached databases share this one
	// PivotStorageInfo; each ATTACH's alias is the datastore name, which the
	// schema-entry lookups pass back so a table lookup routes to that datastore's
	// snapshot in the current transaction. Finally make the default datastore the
	// current database, so unqualified names resolve against it.
	auto ext = duckdb::StorageExtension::Find(
	    duckdb::DBConfig::GetConfig(*db.instance), "pivotdb");
	ext->storage_info = duckdb::make_shared_ptr<PivotStorageInfo>();
	RegisterPivotTableFunctions(*con.context, *this->catalog);

	for (const auto &name : catalog_context_names(*this->catalog)) {
		std::string db_name(name);
		std::string sql = "ATTACH " + duckdb::KeywordHelper::WriteQuoted(db_name, '\'') +
		                  " AS " + duckdb::KeywordHelper::WriteQuoted(db_name, '"') +
		                  " (TYPE pivotdb)";
		auto attach_result = con.Query(sql);
		if (attach_result->HasError()) {
			throw std::runtime_error(attach_result->GetError());
		}
	}

	// Make the default datastore the current database. Set the search path
	// directly rather than through `USE`: the binder verifies a `USE` target's
	// schema through the catalog, and schema lookups need a published
	// transaction, which only exists while a plan is being extracted. The
	// unverified path is checked at bind time by every plan that resolves an
	// unqualified name through it.
	std::string default_name(catalog_context_default(*this->catalog));
	duckdb::ClientData::Get(*con.context)
	    .catalog_search_path->Set(
	        duckdb::CatalogSearchEntry(default_name, DEFAULT_SCHEMA),
	        duckdb::CatalogSetPathType::SET_DIRECTLY);
}

std::unique_ptr<DuckPlannerContext> new_context(rust::Box<CatalogContext> catalog) {
	return std::unique_ptr<DuckPlannerContext>(
	    new DuckPlannerContext(std::move(catalog)));
}

// The Rust walk takes the pivot table handles out of the catalog entries, so the
// entries are only cleared once the walk is done and this handle drops. Destroy
// the plan tree first (it references the entries), then clear them.
PlanHandle::~PlanHandle() {
	root.reset();
	if (ctx) {
		PivotStorageInfo::Get(*ctx->db.instance).ClearTableEntries();
	}
}

// Replicates `duckdb::ClientContext::ExtractPlan`, additionally returning the
// binder-resolved result column names (in select order) via `result_names`.
// The stock `ExtractPlan` computes those names on its local `Planner` and then
// discards them when it returns only the plan; we reproduce its body here (the
// binder/optimizer/resolver steps are all public API) so the names can be
// captured without patching the bundled DuckDB.
static duckdb::unique_ptr<duckdb::LogicalOperator>
extract_plan_with_names(duckdb::Connection &con, const std::string &query,
                        duckdb::vector<std::string> &result_names, bool &requires_rebind) {
	auto statements = con.ExtractStatements(query);
	if (statements.size() != 1) {
		throw duckdb::InvalidInputException("ExtractPlan can only prepare a single statement");
	}
	auto &context = *con.context;
	duckdb::unique_ptr<duckdb::LogicalOperator> plan;
	context.RunFunctionInTransaction([&]() {
		duckdb::Planner planner(context);
		planner.CreatePlan(std::move(statements[0]));
		// The binder resolves the client-facing result column names before
		// optimization rewrites the plan; capture them while still intact.
		result_names = planner.names;
		requires_rebind = planner.properties.always_require_rebind;
		plan = std::move(planner.plan);
		if (context.config.enable_optimizer) {
			duckdb::Optimizer optimizer(*planner.binder, context);
			plan = optimizer.Optimize(std::move(plan));
		}
		plan->ResolveOperatorTypes();
		duckdb::ColumnBindingResolver resolver;
		resolver.Verify(*plan);
		resolver.VisitOperator(*plan);
	});
	return plan;
}

// Publishes the pivot transaction for the duration of one plan: every table
// lookup during binding reads it off the storage info (see
// `PivotSchemaCatalogEntry::LookupEntry`) and the destructor clears it, so the
// pointer never outlives the `extract_plan` call that owns the referent.
struct CurrentTransactionScope {
	PivotStorageInfo &storage_info;

	CurrentTransactionScope(PivotStorageInfo &storage_info, const TransactionContext &transaction)
	    : storage_info(storage_info) {
		storage_info.current_transaction = &transaction;
	}
	~CurrentTransactionScope() {
		storage_info.current_transaction = nullptr;
	}
};

ExtractPlanResult extract_plan(DuckPlannerContext &ctx, rust::Str query,
                               const TransactionContext &transaction) {
	duckdb::unique_ptr<duckdb::LogicalOperator> plan;
	duckdb::vector<std::string> name_list;
	bool requires_rebind = false;
	std::optional<ExtractPlanResult> error;
	CurrentTransactionScope transaction_scope(PivotStorageInfo::Get(*ctx.db.instance), transaction);

	try {
		std::string query_str(query.data(), query.size());
		// The result column names DuckDB would hand a client, in select order
		// (e.g. `["hour", "count_star()"]` for `SELECT f(t) AS hour, COUNT(*)`).
		plan = extract_plan_with_names(ctx.con, query_str, name_list, requires_rebind);
		// Rewrite DuckDB's column *bindings* (table_index, column_index) into
		// positional BoundReference indices against each operator's actual child
		// output. This is the standard resolution DuckDB runs before execution;
		// the Rust walk then reads those positional indices directly.
		duckdb::ColumnBindingResolver resolver;
		resolver.VisitOperator(*plan);
	} catch (duckdb::Exception &e) {
		error = build_duckdb_error(e);
	} catch (const UnsupportedPlanError &e) {
		error = make_error("unsupported_plan", e.what());
	} catch (const std::exception &e) {
		error = make_error("bridge_error", e.what());
	} catch (...) {
		error = make_error("bridge_error", "unknown C++ exception");
	}

	if (error) {
		// No plan was produced, so nothing took the table handles; free the
		// per-plan catalog entries now.
		PivotStorageInfo::Get(*ctx.db.instance).ClearTableEntries();
		return std::move(*error);
	}

	ExtractPlanResult result;
	auto handle = std::make_unique<PlanHandle>();
	handle->root = std::move(plan);
	handle->ctx = &ctx;
	result.plan = std::move(handle);
	result.requires_rebind = requires_rebind;
	for (auto &name : name_list) {
		result.output_names.push_back(rust::String::lossy(name));
	}
	return result;
}

const LogicalOperator &plan_root(const PlanHandle &plan) {
	return *plan.root;
}

size_t rowid_column_id() {
	return static_cast<size_t>(duckdb::COLUMN_IDENTIFIER_ROW_ID);
}

rust::String logical_type_id_name(uint8_t id) {
	return rust::String::lossy(duckdb::LogicalTypeIdToString(static_cast<duckdb::LogicalTypeId>(id)));
}

rust::String expression_type_name(uint8_t type) {
	return rust::String::lossy(duckdb::ExpressionTypeToString(static_cast<duckdb::ExpressionType>(type)));
}

rust::String join_type_name(uint8_t type) {
	return rust::String::lossy(duckdb::JoinTypeToString(static_cast<duckdb::JoinType>(type)));
}

rust::String logical_operator_type_name(uint8_t type) {
	return rust::String::lossy(duckdb::EnumUtil::ToString(static_cast<duckdb::LogicalOperatorType>(type)));
}

rust::String limit_node_type_name(uint8_t type) {
	return rust::String::lossy(duckdb::EnumUtil::ToString(static_cast<duckdb::LimitNodeType>(type)));
}

rust::String order_type_name(uint8_t type) {
	return rust::String::lossy(duckdb::EnumUtil::ToString(static_cast<duckdb::OrderType>(type)));
}

rust::String order_by_null_type_name(uint8_t type) {
	return rust::String::lossy(duckdb::EnumUtil::ToString(static_cast<duckdb::OrderByNullType>(type)));
}

// ---- LogicalOperator: shared structure ----

uint8_t lo_type(const LogicalOperator &op) {
	return static_cast<uint8_t>(op.type);
}

rust::String lo_name(const LogicalOperator &op) {
	return rust::String::lossy(const_cast<LogicalOperator &>(op).GetName());
}

size_t lo_child_count(const LogicalOperator &op) {
	return op.children.size();
}

const LogicalOperator &lo_child(const LogicalOperator &op, size_t index) {
	return *op.children[index];
}

// ---- Projection ----

size_t lo_projection_expr_count(const LogicalOperator &op) {
	return as<duckdb::LogicalProjection>(op).expressions.size();
}

const Expression &lo_projection_expr(const LogicalOperator &op, size_t index) {
	return *as<duckdb::LogicalProjection>(op).expressions[index];
}

// ---- ExpressionGet (VALUES) ----

size_t lo_values_row_count(const LogicalOperator &op) {
	return as<duckdb::LogicalExpressionGet>(op).expressions.size();
}

size_t lo_values_column_count(const LogicalOperator &op) {
	return as<duckdb::LogicalExpressionGet>(op).expr_types.size();
}

const Expression &lo_values_expr(const LogicalOperator &op, size_t row, size_t column) {
	return *as<duckdb::LogicalExpressionGet>(op).expressions[row][column];
}

// ---- Insert ----

rust::Box<OptionalTableWrapper> lo_insert_take_table(const LogicalOperator &op) {
	auto &insert = as<duckdb::LogicalInsert>(op);
	auto &pivot_entry = insert.table.Cast<PivotTableCatalogEntry>();
	return std::move(pivot_entry.table);
}

size_t lo_insert_column_map_count(const LogicalOperator &op) {
	return as<duckdb::LogicalInsert>(op).column_index_map.size();
}

bool lo_insert_column_map_has_source(const LogicalOperator &op, size_t index) {
	auto source = as<duckdb::LogicalInsert>(op).column_index_map[duckdb::PhysicalIndex(index)];
	return source != duckdb::DConstants::INVALID_INDEX;
}

size_t lo_insert_column_map_source(const LogicalOperator &op, size_t index) {
	return as<duckdb::LogicalInsert>(op).column_index_map[duckdb::PhysicalIndex(index)];
}

bool lo_insert_returns_rows(const LogicalOperator &op) {
	return as<duckdb::LogicalInsert>(op).return_chunk;
}

// ---- Filter ----

size_t lo_filter_expr_count(const LogicalOperator &op) {
	return as<duckdb::LogicalFilter>(op).expressions.size();
}

const Expression &lo_filter_expr(const LogicalOperator &op, size_t index) {
	return *as<duckdb::LogicalFilter>(op).expressions[index];
}

size_t lo_filter_projection_map_count(const LogicalOperator &op) {
	return as<duckdb::LogicalFilter>(op).projection_map.size();
}

size_t lo_filter_projection_map_index(const LogicalOperator &op, size_t index) {
	return as<duckdb::LogicalFilter>(op).projection_map[index];
}

// Read a DuckDB LogicalType into the FFI struct: the id discriminant plus the
// width/scale that complete a DECIMAL (zero for every other id).
static BridgeLogicalType bridge_logical_type(const duckdb::LogicalType &type) {
	BridgeLogicalType out;
	out.id = static_cast<uint8_t>(type.id());
	out.decimal_width = 0;
	out.decimal_scale = 0;
	if (type.id() == duckdb::LogicalTypeId::DECIMAL) {
		out.decimal_width = duckdb::DecimalType::GetWidth(type);
		out.decimal_scale = duckdb::DecimalType::GetScale(type);
	}
	return out;
}

BridgeLogicalType lo_filter_type_id(const LogicalOperator &op, size_t index) {
	return bridge_logical_type(as<duckdb::LogicalFilter>(op).types[index]);
}

// ---- OrderBy ----

size_t lo_orderby_count(const LogicalOperator &op) {
	return as<duckdb::LogicalOrder>(op).orders.size();
}

uint8_t lo_orderby_direction(const LogicalOperator &op, size_t index) {
	return static_cast<uint8_t>(as<duckdb::LogicalOrder>(op).orders[index].type);
}

uint8_t lo_orderby_null_order(const LogicalOperator &op, size_t index) {
	return static_cast<uint8_t>(as<duckdb::LogicalOrder>(op).orders[index].null_order);
}

const Expression &lo_orderby_expr(const LogicalOperator &op, size_t index) {
	return *as<duckdb::LogicalOrder>(op).orders[index].expression;
}

// ---- Aggregate ----

size_t lo_aggregate_group_count(const LogicalOperator &op) {
	return as<duckdb::LogicalAggregate>(op).groups.size();
}

const Expression &lo_aggregate_group(const LogicalOperator &op, size_t index) {
	return *as<duckdb::LogicalAggregate>(op).groups[index];
}

size_t lo_aggregate_expr_count(const LogicalOperator &op) {
	return as<duckdb::LogicalAggregate>(op).expressions.size();
}

const Expression &lo_aggregate_expr(const LogicalOperator &op, size_t index) {
	return *as<duckdb::LogicalAggregate>(op).expressions[index];
}

// ---- Distinct ----

bool lo_distinct_is_distinct_on(const LogicalOperator &op) {
	return as<duckdb::LogicalDistinct>(op).distinct_type == duckdb::DistinctType::DISTINCT_ON;
}

size_t lo_distinct_target_count(const LogicalOperator &op) {
	return as<duckdb::LogicalDistinct>(op).distinct_targets.size();
}

const Expression &lo_distinct_target(const LogicalOperator &op, size_t index) {
	return *as<duckdb::LogicalDistinct>(op).distinct_targets[index];
}

// ---- TopN ----

size_t lo_topn_order_count(const LogicalOperator &op) {
	return as<duckdb::LogicalTopN>(op).orders.size();
}

uint8_t lo_topn_order_direction(const LogicalOperator &op, size_t index) {
	return static_cast<uint8_t>(as<duckdb::LogicalTopN>(op).orders[index].type);
}

uint8_t lo_topn_order_null_order(const LogicalOperator &op, size_t index) {
	return static_cast<uint8_t>(as<duckdb::LogicalTopN>(op).orders[index].null_order);
}

const Expression &lo_topn_order_expr(const LogicalOperator &op, size_t index) {
	return *as<duckdb::LogicalTopN>(op).orders[index].expression;
}

size_t lo_topn_limit(const LogicalOperator &op) {
	return as<duckdb::LogicalTopN>(op).limit;
}

size_t lo_topn_offset(const LogicalOperator &op) {
	return as<duckdb::LogicalTopN>(op).offset;
}

bool lo_topn_has_dynamic_filter(const LogicalOperator &op) {
	return static_cast<bool>(as<duckdb::LogicalTopN>(op).dynamic_filter);
}

size_t lo_topn_dynamic_filter_data_id(const LogicalOperator &op) {
	return reinterpret_cast<uintptr_t>(as<duckdb::LogicalTopN>(op).dynamic_filter.get());
}

size_t lo_topn_dynamic_filter_column(const LogicalOperator &op) {
	// By plan-extract time ColumnBindingResolver has rewritten orders[0] into a
	// BoundReferenceExpression whose `index` is the column's position in the
	// TopN's child output.
	auto &topn = as<duckdb::LogicalTopN>(op);
	return topn.orders[0].expression->Cast<duckdb::BoundReferenceExpression>().index;
}

uint8_t lo_topn_dynamic_filter_comparison(const LogicalOperator &op) {
	auto &topn = as<duckdb::LogicalTopN>(op);
	return static_cast<uint8_t>(topn.dynamic_filter->filter->comparison_type);
}

// ---- Limit ----

uint8_t lo_limit_value_kind(const LogicalOperator &op) {
	return static_cast<uint8_t>(as<duckdb::LogicalLimit>(op).limit_val.Type());
}

size_t lo_limit_value(const LogicalOperator &op) {
	return as<duckdb::LogicalLimit>(op).limit_val.GetConstantValue();
}

uint8_t lo_limit_offset_kind(const LogicalOperator &op) {
	return static_cast<uint8_t>(as<duckdb::LogicalLimit>(op).offset_val.Type());
}

size_t lo_limit_offset(const LogicalOperator &op) {
	return as<duckdb::LogicalLimit>(op).offset_val.GetConstantValue();
}

// ---- Get: base table ----

bool lo_get_has_table(const LogicalOperator &op) {
	return static_cast<bool>(as<duckdb::LogicalGet>(op).GetTable());
}

rust::Box<OptionalTableWrapper> lo_get_take_table(const LogicalOperator &op) {
	auto &get = as<duckdb::LogicalGet>(op);
	auto &pivot_entry = get.GetTable()->Cast<PivotTableCatalogEntry>();
	return std::move(pivot_entry.table);
}

bool lo_get_has_bound_table_function(const LogicalOperator &op) {
	auto &get = as<duckdb::LogicalGet>(op);
	return dynamic_cast<PivotTableFunctionBindData *>(get.bind_data.get()) != nullptr;
}

rust::Box<OptionalTableWrapper> lo_get_clone_bound_table_function(const LogicalOperator &op) {
	auto &get = as<duckdb::LogicalGet>(op);
	auto *bound = dynamic_cast<PivotTableFunctionBindData *>(get.bind_data.get());
	if (!bound) {
		throw std::runtime_error("table function has no pivot bound table");
	}
	return clone_table_function_table(*bound->table);
}

// A LogicalGet reads `column_ids` off disk but may *output* only a subset /
// reordering of them, given by `projection_ids` (indices into column_ids). When
// `projection_ids` is set, the scan's output order, and therefore the positional
// indices ColumnBindingResolver assigns to every ref above the scan, follow
// projection_ids, so we report columns in that order.
static size_t get_output_source(const duckdb::LogicalGet &get, size_t index) {
	return get.projection_ids.empty() ? index : get.projection_ids[index];
}

size_t lo_get_output_count(const LogicalOperator &op) {
	auto &get = as<duckdb::LogicalGet>(op);
	return get.projection_ids.empty() ? get.GetColumnIds().size() : get.projection_ids.size();
}

size_t lo_get_output_column(const LogicalOperator &op, size_t index) {
	auto &get = as<duckdb::LogicalGet>(op);
	return get.GetColumnIds()[get_output_source(get, index)].GetPrimaryIndex();
}

BridgeLogicalType lo_get_output_type(const LogicalOperator &op, size_t index) {
	auto &get = as<duckdb::LogicalGet>(op);
	return bridge_logical_type(get.types[get_output_source(get, index)]);
}

// Depth of the field path DuckDB's pushdown-extract pass attached to a projected
// output column: 0 = the whole column is read; N = an N-segment field path
// (`commit.collection` = 2) is pushed, so we read only that leaf. Pushed paths
// are linear (one child per level), one output column per extracted path.
size_t lo_get_output_extract_depth(const LogicalOperator &op, size_t index) {
	auto &get = as<duckdb::LogicalGet>(op);
	const duckdb::ColumnIndex *cur = &get.GetColumnIds()[get_output_source(get, index)];
	size_t depth = 0;
	while (cur->ChildIndexCount() > 0) {
		cur = &cur->GetChildIndex(0);
		depth++;
	}
	return depth;
}

// The field name at 0-based segment `seg` of an output column's pushed extract
// path (segment 0 is the first field below the column, e.g. `commit`).
rust::String lo_get_output_extract_field(const LogicalOperator &op, size_t index, size_t seg) {
	auto &get = as<duckdb::LogicalGet>(op);
	const duckdb::ColumnIndex *cur = &get.GetColumnIds()[get_output_source(get, index)];
	for (size_t i = 0; i <= seg; i++) {
		cur = &cur->GetChildIndex(0);
	}
	return rust::String::lossy(cur->GetFieldName());
}

// Find a DynamicFilter inside a TableFilter, peeling off a single OPTIONAL_FILTER
// wrapper if present. Returns nullptr when there's no DynamicFilter at the top
// (or one level inside an OptionalFilter). More exotic shapes are intentionally
// left unrecognised and dropped by the skip path in collect_filter.
static duckdb::DynamicFilter *extract_dynamic_filter(duckdb::TableFilter &filter) {
	if (filter.filter_type == duckdb::TableFilterType::DYNAMIC_FILTER) {
		return &filter.Cast<duckdb::DynamicFilter>();
	}
	if (filter.filter_type == duckdb::TableFilterType::OPTIONAL_FILTER) {
		auto &opt = filter.Cast<duckdb::OptionalFilter>();
		if (opt.child_filter && opt.child_filter->filter_type == duckdb::TableFilterType::DYNAMIC_FILTER) {
			return &opt.child_filter->Cast<duckdb::DynamicFilter>();
		}
	}
	return nullptr;
}

struct DynFilterInfo {
	uint64_t data_id;
	uint64_t column_idx;
	uint8_t comparison;
};

// Recursively split one column's TableFilter into static-predicate conditions
// (rebuilt with `ToExpression`, appended to `conditions`) and dynamic-filter
// consumer entries (appended to `dynamic_filters`). Either output may be null
// when the caller only wants the other half.
//
// CONJUNCTION_AND is unwrapped and its children processed independently at the
// same column. A standalone OPTIONAL/DYNAMIC we couldn't peel is dropped (it only
// forgoes an optimization). Anything else flows through `ToExpression`.
static void collect_filter(duckdb::TableFilter &filter, duckdb::idx_t proj_idx, duckdb::LogicalGet &get,
                           std::vector<duckdb::unique_ptr<duckdb::Expression>> *conditions,
                           std::vector<DynFilterInfo> *dynamic_filters) {
	if (filter.filter_type == duckdb::TableFilterType::CONJUNCTION_AND) {
		auto &conj = filter.Cast<duckdb::ConjunctionAndFilter>();
		for (auto &child : conj.child_filters) {
			collect_filter(*child, proj_idx, get, conditions, dynamic_filters);
		}
		return;
	}

	if (auto *df = extract_dynamic_filter(filter)) {
		if (dynamic_filters) {
			auto *filter_data = df->filter_data.get();
			auto storage_idx = get.GetColumnIds()[proj_idx].GetPrimaryIndex();
			dynamic_filters->push_back(DynFilterInfo{
			    reinterpret_cast<uintptr_t>(filter_data),
			    storage_idx,
			    static_cast<uint8_t>(filter_data->filter->comparison_type),
			});
		}
		return;
	}

	if (filter.filter_type == duckdb::TableFilterType::OPTIONAL_FILTER ||
	    filter.filter_type == duckdb::TableFilterType::DYNAMIC_FILTER) {
		return;
	}

	if (conditions) {
		duckdb::ColumnBinding binding(get.table_index, duckdb::ProjectionIndex(proj_idx));
		auto col_ref = duckdb::make_uniq<duckdb::BoundColumnRefExpression>(get.types[proj_idx], binding);
		// Tag the synthetic ref with the table's column name so a pushed-down
		// `col op const` filter renders as `col = 5`, not `#idx = 5`.
		auto storage_idx = get.GetColumnIds()[proj_idx].GetPrimaryIndex();
		if (storage_idx < get.names.size()) {
			col_ref->SetAlias(get.names[storage_idx]);
		}
		conditions->push_back(filter.ToExpression(*col_ref));
	}
}

std::unique_ptr<ExpressionList> lo_get_pushed_conditions(const LogicalOperator &op) {
	auto &get = as<duckdb::LogicalGet>(op);
	auto list = std::make_unique<ExpressionList>();
	for (auto &entry : get.table_filters) {
		collect_filter(entry.Filter(), entry.GetIndex().GetIndex(), get, &list->exprs, nullptr);
	}
	return list;
}

static std::vector<DynFilterInfo> collect_get_dynamic_filters(duckdb::LogicalGet &get) {
	std::vector<DynFilterInfo> out;
	for (auto &entry : get.table_filters) {
		collect_filter(entry.Filter(), entry.GetIndex().GetIndex(), get, nullptr, &out);
	}
	return out;
}

size_t lo_get_dynamic_filter_count(const LogicalOperator &op) {
	return collect_get_dynamic_filters(as<duckdb::LogicalGet>(op)).size();
}

size_t lo_get_dynamic_filter_data_id(const LogicalOperator &op, size_t index) {
	return collect_get_dynamic_filters(as<duckdb::LogicalGet>(op))[index].data_id;
}

size_t lo_get_dynamic_filter_column(const LogicalOperator &op, size_t index) {
	return collect_get_dynamic_filters(as<duckdb::LogicalGet>(op))[index].column_idx;
}

uint8_t lo_get_dynamic_filter_comparison(const LogicalOperator &op, size_t index) {
	return collect_get_dynamic_filters(as<duckdb::LogicalGet>(op))[index].comparison;
}

// Join-filter-pushdown pairing data: the pointer identity of the scan's shared
// DynamicTableFilterSet (0 when no join pushes filters into this scan), the
// scan's table index, and the column_ids -> storage column mapping the join's
// probe column indexes resolve through.
size_t lo_get_join_filter_set_id(const LogicalOperator &op) {
	return reinterpret_cast<uintptr_t>(as<duckdb::LogicalGet>(op).dynamic_filters.get());
}

size_t lo_get_table_index(const LogicalOperator &op) {
	return as<duckdb::LogicalGet>(op).table_index.index;
}

size_t lo_get_column_ids_count(const LogicalOperator &op) {
	return as<duckdb::LogicalGet>(op).GetColumnIds().size();
}

size_t lo_get_storage_column(const LogicalOperator &op, size_t column_ids_index) {
	return as<duckdb::LogicalGet>(op).GetColumnIds()[column_ids_index].GetPrimaryIndex();
}

// ---- Get: table function ----

rust::String lo_get_function_name(const LogicalOperator &op) {
	return rust::String::lossy(as<duckdb::LogicalGet>(op).function.name);
}

bool lo_get_has_named_params(const LogicalOperator &op) {
	return !as<duckdb::LogicalGet>(op).named_parameters.empty();
}

size_t lo_get_param_count(const LogicalOperator &op) {
	return as<duckdb::LogicalGet>(op).parameters.size();
}

const Value &lo_get_param(const LogicalOperator &op, size_t index) {
	return as<duckdb::LogicalGet>(op).parameters[index];
}

// ---- CreateTable ----

static duckdb::CreateTableInfo &create_table_info(const LogicalOperator &op) {
	return as<duckdb::LogicalCreateTable>(op).info->Base();
}

rust::String lo_create_table_name(const LogicalOperator &op) {
	return rust::String::lossy(create_table_info(op).table);
}

rust::String lo_create_table_datastore(const LogicalOperator &op) {
	// DuckDB stores the resolved datastore/database for
	// `CREATE TABLE db.schema.t` in its native `catalog` field. It is empty when
	// the statement is unqualified.
	return rust::String::lossy(create_table_info(op).catalog);
}

rust::String lo_create_table_schema(const LogicalOperator &op) {
	// DuckDB resolves the schema of `CREATE TABLE db.schema.t` into its native
	// `schema` field. It is empty when the statement named no schema.
	return rust::String::lossy(create_table_info(op).schema);
}

// ---- CreateSchema ----

static duckdb::CreateSchemaInfo &create_schema_info(const LogicalOperator &op) {
	return as<duckdb::LogicalCreate>(op).info->Cast<duckdb::CreateSchemaInfo>();
}

rust::String lo_create_schema_name(const LogicalOperator &op) {
	return rust::String::lossy(create_schema_info(op).schema);
}

rust::String lo_create_schema_datastore(const LogicalOperator &op) {
	// DuckDB stores the resolved datastore/database for `CREATE SCHEMA db.s` in
	// its native `catalog` field. It is empty when the statement is unqualified.
	return rust::String::lossy(create_schema_info(op).catalog);
}

bool lo_create_schema_if_not_exists(const LogicalOperator &op) {
	return create_schema_info(op).on_conflict == duckdb::OnCreateConflict::IGNORE_ON_CONFLICT;
}

bool lo_create_schema_or_replace(const LogicalOperator &op) {
	return create_schema_info(op).on_conflict == duckdb::OnCreateConflict::REPLACE_ON_CONFLICT;
}

// ---- Drop ----

static duckdb::DropInfo &drop_info(const LogicalOperator &op) {
	return as<duckdb::LogicalSimple>(op).info->Cast<duckdb::DropInfo>();
}

bool lo_drop_is_table(const LogicalOperator &op) {
	return drop_info(op).type == duckdb::CatalogType::TABLE_ENTRY;
}

rust::String lo_drop_entry_kind(const LogicalOperator &op) {
	return rust::String::lossy(duckdb::CatalogTypeToString(drop_info(op).type));
}

rust::String lo_drop_name(const LogicalOperator &op) {
	return rust::String::lossy(drop_info(op).name);
}

rust::String lo_drop_schema(const LogicalOperator &op) {
	// The binder fills `schema` with the resolved entry's parent schema; for an
	// `IF EXISTS` drop of a missing entry it stays as written, possibly empty.
	return rust::String::lossy(drop_info(op).schema);
}

rust::String lo_drop_datastore(const LogicalOperator &op) {
	// DuckDB stores the resolved datastore/database in its native `catalog`
	// field. It is empty when the statement is unqualified and nothing resolved.
	return rust::String::lossy(drop_info(op).catalog);
}

bool lo_drop_if_exists(const LogicalOperator &op) {
	return drop_info(op).if_not_found == duckdb::OnEntryNotFound::RETURN_NULL;
}

bool lo_drop_cascade(const LogicalOperator &op) {
	return drop_info(op).cascade;
}

// ---- Explain ----

bool lo_explain_is_analyze(const LogicalOperator &op) {
	return as<duckdb::LogicalExplain>(op).explain_type == duckdb::ExplainType::EXPLAIN_ANALYZE;
}

size_t lo_create_column_count(const LogicalOperator &op) {
	return create_table_info(op).columns.LogicalColumnCount();
}

rust::String lo_create_column_name(const LogicalOperator &op, size_t index) {
	auto &col = create_table_info(op).columns.GetColumn(duckdb::LogicalIndex(index));
	return rust::String::lossy(col.GetName());
}

BridgeLogicalType lo_create_column_type(const LogicalOperator &op, size_t index) {
	auto &col = create_table_info(op).columns.GetColumn(duckdb::LogicalIndex(index));
	return bridge_logical_type(col.Type());
}

static string create_table_option_to_string(duckdb::ParsedExpression &expr) {
	if (expr.GetExpressionClass() == duckdb::ExpressionClass::CONSTANT) {
		auto &value = expr.Cast<duckdb::ConstantExpression>().value;
		if (value.IsNull()) {
			return "true";
		}
		return value.ToString();
	}
	return expr.ToString();
}

size_t lo_create_option_count(const LogicalOperator &op) {
	return create_table_info(op).options.size();
}

// CreateTableInfo::options is an unordered map; index into it deterministically
// by iteration order so key and value accessors agree within one plan.
static std::pair<const string &, duckdb::ParsedExpression &> create_option_at(const LogicalOperator &op,
                                                                              size_t index) {
	auto &options = create_table_info(op).options;
	auto it = options.begin();
	std::advance(it, index);
	return {it->first, *it->second};
}

rust::String lo_create_option_key(const LogicalOperator &op, size_t index) {
	return rust::String::lossy(create_option_at(op, index).first);
}

rust::String lo_create_option_value(const LogicalOperator &op, size_t index) {
	return rust::String::lossy(create_table_option_to_string(create_option_at(op, index).second));
}

bool lo_create_if_not_exists(const LogicalOperator &op) {
	return create_table_info(op).on_conflict == duckdb::OnCreateConflict::IGNORE_ON_CONFLICT;
}

bool lo_create_or_replace(const LogicalOperator &op) {
	return create_table_info(op).on_conflict == duckdb::OnCreateConflict::REPLACE_ON_CONFLICT;
}

bool lo_create_temporary(const LogicalOperator &op) {
	return create_table_info(op).temporary;
}

bool lo_create_has_query(const LogicalOperator &op) {
	return as<duckdb::LogicalCreateTable>(op).info->query != nullptr;
}

size_t lo_create_constraint_count(const LogicalOperator &op) {
	return as<duckdb::LogicalCreateTable>(op).info->constraints.size();
}

// ---- Set / Reset ----

rust::String lo_set_name(const LogicalOperator &op) {
	return rust::String::lossy(as<duckdb::LogicalSet>(op).name);
}

rust::String lo_set_value(const LogicalOperator &op) {
	return rust::String::lossy(as<duckdb::LogicalSet>(op).value.ToString());
}

rust::String lo_reset_name(const LogicalOperator &op) {
	return rust::String::lossy(as<duckdb::LogicalReset>(op).name);
}

// ---- Transaction ----

uint8_t lo_transaction_type(const LogicalOperator &op) {
	return static_cast<uint8_t>(as<duckdb::LogicalSimple>(op).info->Cast<duckdb::TransactionInfo>().type);
}

// ---- Vacuum ----

void lo_vacuum_validate(const LogicalOperator &op) {
	const auto &info = *as<duckdb::LogicalVacuum>(op).info;
	if (!info.options.vacuum || info.options.analyze || info.has_table) {
		throw duckdb::NotImplementedException("Only VACUUM without a table or ANALYZE is supported");
	}
}

// ---- Compact ----

rust::String lo_compact_datastore(const LogicalOperator &op) {
	return rust::String::lossy(as<duckdb::LogicalCompact>(op).table.catalog);
}

rust::String lo_compact_schema(const LogicalOperator &op) {
	return rust::String::lossy(as<duckdb::LogicalCompact>(op).table.schema);
}

rust::String lo_compact_table(const LogicalOperator &op) {
	return rust::String::lossy(as<duckdb::LogicalCompact>(op).table.name);
}

bool lo_compact_final(const LogicalOperator &op) {
	return as<duckdb::LogicalCompact>(op).final_sweep;
}

// ---- CreateUser ----

rust::String lo_create_user_name(const LogicalOperator &op) {
	return rust::String::lossy(as<duckdb::LogicalCreateUser>(op).user);
}

bool lo_create_user_has_password(const LogicalOperator &op) {
	return as<duckdb::LogicalCreateUser>(op).has_password;
}

// Not lossy: replacing invalid bytes would store a different password than
// the one the user typed. The strict conversion throws on invalid UTF-8,
// which surfaces as an error on the Rust side.
rust::String lo_create_user_password(const LogicalOperator &op) {
	return rust::String(as<duckdb::LogicalCreateUser>(op).password);
}

// ---- CopyFromStdin ----

rust::Box<OptionalTableWrapper> lo_copy_stdin_take_table(const LogicalOperator &op) {
	auto &copy = as<duckdb::LogicalCopyFromStdin>(op);
	auto &pivot_entry = copy.table.Cast<PivotTableCatalogEntry>();
	return std::move(pivot_entry.table);
}

size_t lo_copy_stdin_column_count(const LogicalOperator &op) {
	return as<duckdb::LogicalCopyFromStdin>(op).column_indexes.size();
}

size_t lo_copy_stdin_column_index(const LogicalOperator &op, size_t index) {
	return as<duckdb::LogicalCopyFromStdin>(op).column_indexes[index];
}

rust::String lo_copy_stdin_format(const LogicalOperator &op) {
	return rust::String::lossy(as<duckdb::LogicalCopyFromStdin>(op).format);
}

size_t lo_copy_stdin_option_count(const LogicalOperator &op) {
	return as<duckdb::LogicalCopyFromStdin>(op).options.size();
}

rust::String lo_copy_stdin_option_name(const LogicalOperator &op, size_t index) {
	return rust::String::lossy(as<duckdb::LogicalCopyFromStdin>(op).options[index].first);
}

size_t lo_copy_stdin_option_value_count(const LogicalOperator &op, size_t index) {
	return as<duckdb::LogicalCopyFromStdin>(op).options[index].second.size();
}

rust::String lo_copy_stdin_option_value(const LogicalOperator &op, size_t index, size_t value_index) {
	return rust::String::lossy(as<duckdb::LogicalCopyFromStdin>(op).options[index].second[value_index].ToString());
}

// ---- CTE ----

// The index a materialized CTE publishes its rows under, which every reference
// to it carries (see `lo_cte_ref_index`).
size_t lo_cte_table_index(const LogicalOperator &op) {
	return as<duckdb::LogicalMaterializedCTE>(op).table_index.index;
}

// The CTE a reference reads, as the index that CTE was published under.
size_t lo_cte_ref_index(const LogicalOperator &op) {
	return as<duckdb::LogicalCTERef>(op).cte_index.index;
}

// ---- ColumnDataGet (CHUNK_GET): a scan of an in-memory constant chunk ----

size_t lo_chunk_get_row_count(const LogicalOperator &op) {
	return as<duckdb::LogicalColumnDataGet>(op).collection->Count();
}

size_t lo_chunk_get_column_count(const LogicalOperator &op) {
	return as<duckdb::LogicalColumnDataGet>(op).chunk_types.size();
}

// One cell of the collection, owned because the row view hands values out by
// value. Materializing the row view per call is fine: these chunks are small
// by construction (an IN list's constants).
std::unique_ptr<Value> lo_chunk_get_value(const LogicalOperator &op, size_t column, size_t row) {
	duckdb::ColumnDataRowCollection rows(*as<duckdb::LogicalColumnDataGet>(op).collection);
	return std::make_unique<duckdb::Value>(rows.GetValue(column, row));
}

// ---- ComparisonJoin: general accessors ----

uint8_t lo_join_type(const LogicalOperator &op) {
	return static_cast<uint8_t>(as<duckdb::LogicalComparisonJoin>(op).join_type);
}

size_t lo_join_condition_count(const LogicalOperator &op) {
	return as<duckdb::LogicalComparisonJoin>(op).conditions.size();
}

bool lo_join_condition_is_comparison(const LogicalOperator &op, size_t index) {
	return as<duckdb::LogicalComparisonJoin>(op).conditions[index].IsComparison();
}

const Expression &lo_join_condition_expression(const LogicalOperator &op, size_t index) {
	return as<duckdb::LogicalComparisonJoin>(op).conditions[index].GetJoinExpression();
}

const Expression &lo_join_condition_left(const LogicalOperator &op, size_t index) {
	return as<duckdb::LogicalComparisonJoin>(op).conditions[index].GetLHS();
}

const Expression &lo_join_condition_right(const LogicalOperator &op, size_t index) {
	return as<duckdb::LogicalComparisonJoin>(op).conditions[index].GetRHS();
}

uint8_t lo_join_condition_comparison(const LogicalOperator &op, size_t index) {
	return static_cast<uint8_t>(as<duckdb::LogicalComparisonJoin>(op).conditions[index].GetComparisonType());
}

// The join's projection maps: which of each child's output columns survive in
// the join's output (empty = all of them). Filled by DuckDB's column-lifetime
// pass, e.g. to drop a build-side key only referenced by the join condition.
size_t lo_join_left_projection_map_count(const LogicalOperator &op) {
	return as<duckdb::LogicalComparisonJoin>(op).left_projection_map.size();
}

size_t lo_join_left_projection_map_index(const LogicalOperator &op, size_t index) {
	return as<duckdb::LogicalComparisonJoin>(op).left_projection_map[index];
}

size_t lo_join_right_projection_map_count(const LogicalOperator &op) {
	return as<duckdb::LogicalComparisonJoin>(op).right_projection_map.size();
}

size_t lo_join_right_projection_map_index(const LogicalOperator &op, size_t index) {
	return as<duckdb::LogicalComparisonJoin>(op).right_projection_map[index];
}

// ---- ComparisonJoin: join filter pushdown ----

// DuckDB's join-filter-pushdown optimizer records on the join which probe-side
// scans can be narrowed by filters derived from the build side's key values.
// Each target is one probe scan, paired with its LogicalGet by the pointer
// identity of the shared DynamicTableFilterSet both hold; each target column
// names the get's column_ids index to filter and the join condition whose
// build-side values bound it.

static const duckdb::JoinFilterPushdownInfo *join_pushdown(const LogicalOperator &op) {
	return as<duckdb::LogicalComparisonJoin>(op).filter_pushdown.get();
}

size_t lo_join_pushdown_target_count(const LogicalOperator &op) {
	auto *info = join_pushdown(op);
	return info ? info->probe_info.size() : 0;
}

size_t lo_join_pushdown_set_id(const LogicalOperator &op, size_t target) {
	return reinterpret_cast<uintptr_t>(join_pushdown(op)->probe_info[target].dynamic_filters.get());
}

size_t lo_join_pushdown_column_count(const LogicalOperator &op, size_t target) {
	auto *info = join_pushdown(op);
	auto count = info->probe_info[target].columns.size();
	// The per-target column list mirrors the join_condition list; a mismatch
	// would desynchronise the condition lookup below, so report none instead.
	return count == info->join_condition.size() ? count : 0;
}

size_t lo_join_pushdown_probe_table(const LogicalOperator &op, size_t target, size_t index) {
	return join_pushdown(op)->probe_info[target].columns[index].probe_column_index.table_index.index;
}

size_t lo_join_pushdown_probe_column(const LogicalOperator &op, size_t target, size_t index) {
	return join_pushdown(op)->probe_info[target].columns[index].probe_column_index.column_index.GetIndex();
}

size_t lo_join_pushdown_condition_index(const LogicalOperator &op, size_t index) {
	return join_pushdown(op)->join_condition[index];
}

// ---- DelimJoin / DelimGet ----

// A delim join is stored as a LogicalComparisonJoin whose operator type is
// LOGICAL_DELIM_JOIN, so the general join accessors above work on it too.
// These read the delim-specific fields.

// The expressions (over the de-duplicated side's output) whose distinct values
// every DELIM_GET under the other side scans.
size_t lo_delim_join_column_count(const LogicalOperator &op) {
	return as<duckdb::LogicalComparisonJoin>(op).duplicate_eliminated_columns.size();
}

const Expression &lo_delim_join_column(const LogicalOperator &op, size_t index) {
	return *as<duckdb::LogicalComparisonJoin>(op).duplicate_eliminated_columns[index];
}

// False: the LHS is de-duplicated and the DELIM_GETs sit under the RHS.
// True: the join was flipped and the roles reverse.
bool lo_delim_join_is_flipped(const LogicalOperator &op) {
	return as<duckdb::LogicalComparisonJoin>(op).delim_flipped;
}

size_t lo_delim_get_column_count(const LogicalOperator &op) {
	return as<duckdb::LogicalDelimGet>(op).chunk_types.size();
}

BridgeLogicalType lo_delim_get_column_type(const LogicalOperator &op, size_t index) {
	return bridge_logical_type(as<duckdb::LogicalDelimGet>(op).chunk_types[index]);
}

// ---- ExpressionList ----

size_t expr_list_count(const ExpressionList &list) {
	return list.exprs.size();
}

const Expression &expr_list_get(const ExpressionList &list, size_t index) {
	return *list.exprs[index];
}

// ---- Expression: shared ----

uint8_t expr_type(const Expression &expr) {
	return static_cast<uint8_t>(expr.type);
}

BridgeLogicalType expr_return_type(const Expression &expr) {
	return bridge_logical_type(expr.return_type);
}

bool expr_has_alias(const Expression &expr) {
	return !expr.GetAlias().empty();
}

rust::String expr_alias(const Expression &expr) {
	return rust::String::lossy(expr.GetAlias());
}

const Value &expr_constant(const Expression &expr) {
	return as_expr<duckdb::BoundConstantExpression>(expr).value;
}

// Typed accessors for a DuckDB `Value`, shared by query constants and
// table-function arguments. The Rust side reads `value_type` and `value_is_null`
// first, then calls the matching accessor; `GetValue<T>` returns the stored
// value untouched when the requested type matches the value's own.
//
// A NULL value still carries the type the binder resolved for it (the target
// column's type for an INSERT), and none of the typed accessors below may be
// called on one: they either throw or, for `value_string`, hand back the text
// "NULL".
BridgeLogicalType value_type(const Value &v) {
	return bridge_logical_type(v.type());
}
bool value_is_null(const Value &v) {
	return v.IsNull();
}
bool value_bool(const Value &v) {
	return v.GetValue<bool>();
}
int8_t value_i8(const Value &v) {
	return v.GetValue<int8_t>();
}
int16_t value_i16(const Value &v) {
	return v.GetValue<int16_t>();
}
int32_t value_i32(const Value &v) {
	return v.GetValue<int32_t>();
}
int64_t value_i64(const Value &v) {
	return v.GetValue<int64_t>();
}
uint8_t value_u8(const Value &v) {
	return v.GetValue<uint8_t>();
}
uint16_t value_u16(const Value &v) {
	return v.GetValue<uint16_t>();
}
uint32_t value_u32(const Value &v) {
	return v.GetValue<uint32_t>();
}
uint64_t value_u64(const Value &v) {
	return v.GetValue<uint64_t>();
}
float value_f32(const Value &v) {
	return v.GetValue<float>();
}
double value_f64(const Value &v) {
	return v.GetValue<double>();
}
rust::String value_string(const Value &v) {
	return rust::String::lossy(v.GetValue<std::string>());
}
// A DECIMAL Value stores its unscaled integer in the physical width its
// precision requires; widen the raw (unrescaled) integer to 128 bits.
BridgeDecimalValue value_decimal(const Value &v) {
	uint8_t width, scale;
	v.type().GetDecimalProperties(width, scale);
	duckdb::hugeint_t raw;
	switch (v.type().InternalType()) {
	case duckdb::PhysicalType::INT16:
		raw = duckdb::hugeint_t(v.GetValueUnsafe<int16_t>());
		break;
	case duckdb::PhysicalType::INT32:
		raw = duckdb::hugeint_t(v.GetValueUnsafe<int32_t>());
		break;
	case duckdb::PhysicalType::INT64:
		raw = duckdb::hugeint_t(v.GetValueUnsafe<int64_t>());
		break;
	case duckdb::PhysicalType::INT128:
		raw = v.GetValueUnsafe<duckdb::hugeint_t>();
		break;
	default:
		throw duckdb::InternalException("DECIMAL value with unexpected physical type");
	}
	BridgeDecimalValue out;
	out.hi = raw.upper;
	out.lo = raw.lower;
	out.width = width;
	out.scale = scale;
	return out;
}
BridgeHugeint value_hugeint(const Value &v) {
	auto raw = v.GetValueUnsafe<duckdb::hugeint_t>();
	BridgeHugeint out;
	out.hi = raw.upper;
	out.lo = raw.lower;
	return out;
}
// DATE is days since the Unix epoch; TIMESTAMP is microseconds since the epoch.
int32_t value_date(const Value &v) {
	return v.GetValue<duckdb::date_t>().days;
}
int64_t value_timestamp(const Value &v) {
	return v.GetValue<duckdb::timestamp_t>().value;
}
// TIMESTAMP WITH TIME ZONE holds the same microsecond count, taken as UTC.
int64_t value_timestamp_tz(const Value &v) {
	return v.GetValue<duckdb::timestamp_tz_t>().value;
}
// INTERVAL keeps its three independent components (months are calendar-variable,
// so they stay separate from the fixed day/microsecond parts).
int32_t value_interval_months(const Value &v) {
	return v.GetValue<duckdb::interval_t>().months;
}
int32_t value_interval_days(const Value &v) {
	return v.GetValue<duckdb::interval_t>().days;
}
int64_t value_interval_micros(const Value &v) {
	return v.GetValue<duckdb::interval_t>().micros;
}

size_t expr_ref_index(const Expression &expr) {
	return as_expr<duckdb::BoundReferenceExpression>(expr).index;
}

size_t expr_columnref_index(const Expression &expr) {
	return as_expr<duckdb::BoundColumnRefExpression>(expr).binding.column_index.GetIndex();
}

const Expression &expr_comparison_left(const Expression &expr) {
	return *as_expr<duckdb::BoundComparisonExpression>(expr).left;
}

const Expression &expr_comparison_right(const Expression &expr) {
	return *as_expr<duckdb::BoundComparisonExpression>(expr).right;
}

const Expression &expr_between_input(const Expression &expr) {
	return *as_expr<duckdb::BoundBetweenExpression>(expr).input;
}

const Expression &expr_between_lower(const Expression &expr) {
	return *as_expr<duckdb::BoundBetweenExpression>(expr).lower;
}

const Expression &expr_between_upper(const Expression &expr) {
	return *as_expr<duckdb::BoundBetweenExpression>(expr).upper;
}

bool expr_between_lower_inclusive(const Expression &expr) {
	return as_expr<duckdb::BoundBetweenExpression>(expr).lower_inclusive;
}

bool expr_between_upper_inclusive(const Expression &expr) {
	return as_expr<duckdb::BoundBetweenExpression>(expr).upper_inclusive;
}

rust::String expr_aggregate_name(const Expression &expr) {
	return rust::String::lossy(as_expr<duckdb::BoundAggregateExpression>(expr).function.name);
}

bool expr_aggregate_distinct(const Expression &expr) {
	return as_expr<duckdb::BoundAggregateExpression>(expr).IsDistinct();
}

size_t expr_aggregate_child_count(const Expression &expr) {
	return as_expr<duckdb::BoundAggregateExpression>(expr).children.size();
}

const Expression &expr_aggregate_child(const Expression &expr, size_t index) {
	return *as_expr<duckdb::BoundAggregateExpression>(expr).children[index];
}

// Whether the aggregate call carries a `FILTER (WHERE …)` clause.
bool expr_aggregate_has_filter(const Expression &expr) {
	return as_expr<duckdb::BoundAggregateExpression>(expr).filter != nullptr;
}

rust::String expr_function_name(const Expression &expr) {
	return rust::String::lossy(as_expr<duckdb::BoundFunctionExpression>(expr).function.name);
}

size_t expr_function_child_count(const Expression &expr) {
	return as_expr<duckdb::BoundFunctionExpression>(expr).children.size();
}

const Expression &expr_function_child(const Expression &expr, size_t index) {
	return *as_expr<duckdb::BoundFunctionExpression>(expr).children[index];
}

size_t expr_operator_child_count(const Expression &expr) {
	return as_expr<duckdb::BoundOperatorExpression>(expr).children.size();
}

const Expression &expr_operator_child(const Expression &expr, size_t index) {
	return *as_expr<duckdb::BoundOperatorExpression>(expr).children[index];
}

size_t expr_conjunction_child_count(const Expression &expr) {
	return as_expr<duckdb::BoundConjunctionExpression>(expr).children.size();
}

const Expression &expr_conjunction_child(const Expression &expr, size_t index) {
	return *as_expr<duckdb::BoundConjunctionExpression>(expr).children[index];
}

size_t expr_case_check_count(const Expression &expr) {
	return as_expr<duckdb::BoundCaseExpression>(expr).case_checks.size();
}

const Expression &expr_case_when(const Expression &expr, size_t index) {
	return *as_expr<duckdb::BoundCaseExpression>(expr).case_checks[index].when_expr;
}

const Expression &expr_case_then(const Expression &expr, size_t index) {
	return *as_expr<duckdb::BoundCaseExpression>(expr).case_checks[index].then_expr;
}

const Expression &expr_case_else(const Expression &expr) {
	return *as_expr<duckdb::BoundCaseExpression>(expr).else_expr;
}

const Expression &expr_cast_child(const Expression &expr) {
	return *as_expr<duckdb::BoundCastExpression>(expr).child;
}

bool expr_cast_is_try(const Expression &expr) {
	return as_expr<duckdb::BoundCastExpression>(expr).try_cast;
}
