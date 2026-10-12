//! [`TopN`] — combined ORDER BY + LIMIT (returns the top N rows).

use super::{OrderByDirection, OrderByNode};
use crate::compile::{Error, RuntimeFilterSlots};
use crate::dynamic_filter::DynamicFilter;
use crate::expression::Expression;
use dispatch::{OrderBy as DispatchOrderBy, RecordBatchOperatorSpec};
use std::fmt;

/// Combined ORDER BY + LIMIT (returns the top N rows).
#[derive(Debug)]
pub struct TopN {
    pub order_bys: Vec<OrderByNode>,
    pub limit: usize,
    pub offset: usize,
    /// When set, this Top-N is a dynamic-filter producer: at runtime it
    /// publishes its current boundary value into the shared slot so consumer
    /// scans elsewhere in the plan can prune row groups against it.
    pub produces_dynamic_filter: Option<DynamicFilter>,
}

impl fmt::Display for TopN {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let orders = self
            .order_bys
            .iter()
            .map(|o| o.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        write!(
            f,
            "TopN(limit: {}, offset: {}, order: {orders})",
            self.limit, self.offset
        )
    }
}

impl TopN {
    pub(crate) fn compile(
        &self,
        input: RecordBatchOperatorSpec,
        slots: &mut RuntimeFilterSlots,
    ) -> Result<RecordBatchOperatorSpec, Error> {
        let orders = self
            .order_bys
            .iter()
            .map(|node| {
                let col = match &node.expression {
                    Expression::Ref(r) => Ok(r.column_idx),
                    expr => Err(Error::UnsupportedTopKExpression(expr.clone())),
                }?;
                let descending = matches!(node.direction, OrderByDirection::Desc);
                Ok(DispatchOrderBy::new(col, descending, node.nulls_first))
            })
            .collect::<Result<Vec<_>, _>>()?;
        // If DuckDB's Top-N optimizer marked this node as a dynamic-filter
        // producer, hand it the shared slot (minted fresh per compile, shared
        // with the consumer scan by `slot_id`) so it publishes its running
        // boundary into it, tightening row-group pruning at sibling scans.
        let dynamic_filter = self
            .produces_dynamic_filter
            .as_ref()
            .map(|df| slots.boundary_slot(df.slot_id));
        Ok(input.order_by_limit_offset(orders, self.limit, self.offset, dynamic_filter))
    }
}
