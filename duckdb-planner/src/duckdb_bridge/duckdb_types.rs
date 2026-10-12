//! Auto-generated Rust mirrors of DuckDB C++ enums via `autocxx`.
//!
//! The [`include_cpp!`] macro pulls in the DuckDB enum headers and generates
//! Rust types that can be used for discriminant matching when building the plan.

use autocxx::prelude::*;

include_cpp! {
    #include "duckdb/common/enums/logical_operator_type.hpp"
    #include "duckdb/common/types.hpp"
    #include "duckdb/common/enums/expression_type.hpp"
    #include "duckdb/common/enums/order_type.hpp"
    #include "duckdb/common/enums/join_type.hpp"
    #include "duckdb/planner/table_filter.hpp"
    #include "duckdb/planner/bound_result_modifier.hpp"
    #include "duckdb/parser/parsed_data/transaction_info.hpp"
    safety!(unsafe)
    generate!("duckdb::LogicalOperatorType")
    generate!("duckdb::LogicalTypeId")
    generate!("duckdb::ExpressionType")
    generate!("duckdb::OrderType")
    generate!("duckdb::OrderByNullType")
    generate!("duckdb::LogicalTypeId")
    generate!("duckdb::TableFilterType")
    generate!("duckdb::LimitNodeType")
    generate!("duckdb::JoinType")
    generate!("duckdb::TransactionType")
}

pub use ffi::duckdb::ExpressionType;
pub use ffi::duckdb::JoinType;
pub use ffi::duckdb::LimitNodeType;
pub use ffi::duckdb::LogicalOperatorType;
pub use ffi::duckdb::LogicalTypeId;
pub use ffi::duckdb::OrderByNullType;
pub use ffi::duckdb::OrderType;
pub use ffi::duckdb::TableFilterType;
pub use ffi::duckdb::TransactionType;

use std::fmt;

/// Give each DuckDB enum a `from_u8` decoder. DuckDB defines them as
/// `enum class : uint8_t`, so the `u8` discriminant the bridge reports
/// round-trips exactly through a transmute. `from_u8` is `pub(crate)` on
/// purpose: only the handle layer decodes raw bytes, and callers get typed
/// accessors rather than a way to forge an invalid enum.
macro_rules! impl_from_u8 {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl $ty {
                pub(crate) fn from_u8(value: u8) -> Self {
                    unsafe { std::mem::transmute::<u8, $ty>(value) }
                }
            }
        )+
    };
}

/// Render a DuckDB enum as the name DuckDB itself prints for it (`TIME`,
/// `COMPARE_DISTINCT_FROM`, `SINGLE`, ...), read back through `$name_fn`, so
/// a message about something the planner rejects names it instead of printing
/// a discriminant. A discriminant DuckDB has no name for falls back to the
/// number: formatting cannot fail, and the number still identifies the value.
macro_rules! impl_named_format {
    ($($ty:ty => $name_fn:ident),+ $(,)?) => {
        $(
            impl fmt::Display for $ty {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    let discriminant = self.clone() as u8;
                    // Fully qualified: `include_cpp!` above defines its own
                    // `ffi` module in this one.
                    match crate::duckdb_bridge::ffi::$name_fn(discriminant) {
                        Ok(name) => f.write_str(&name),
                        Err(_) => write!(f, "{discriminant}"),
                    }
                }
            }

            impl fmt::Debug for $ty {
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    fmt::Display::fmt(self, f)
                }
            }
        )+
    };
}

impl_from_u8!(
    LogicalOperatorType,
    ExpressionType,
    JoinType,
    LimitNodeType,
    LogicalTypeId,
    OrderType,
    OrderByNullType,
    TransactionType,
);

impl_named_format!(
    LogicalTypeId => logical_type_id_name,
    ExpressionType => expression_type_name,
    JoinType => join_type_name,
    LogicalOperatorType => logical_operator_type_name,
    LimitNodeType => limit_node_type_name,
    OrderType => order_type_name,
    OrderByNullType => order_by_null_type_name,
);
