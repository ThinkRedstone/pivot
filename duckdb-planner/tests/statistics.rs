use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use duckdb_planner::{DuckDBColumn, DuckDBTable, DuckDBTransaction, LogicalTypeId, PlannerContext};

struct CountingTable {
    rows: u64,
    asked: Arc<AtomicUsize>,
}

impl DuckDBTable for CountingTable {
    fn clone_box(&self) -> Box<dyn DuckDBTable> {
        Box::new(CountingTable {
            rows: self.rows,
            asked: self.asked.clone(),
        })
    }

    fn duckdb_typed_columns(&self) -> Vec<DuckDBColumn> {
        vec![DuckDBColumn {
            name: "id".to_string(),
            duckdb_logical_type_id: LogicalTypeId::INTEGER as u8,
            decimal_width: 0,
            decimal_scale: 0,
        }]
    }

    fn estimate_row_count(&self) -> Option<u64> {
        self.asked.fetch_add(1, Ordering::Relaxed);
        Some(self.rows)
    }
}

struct StatsTransaction {
    asked: Arc<AtomicUsize>,
}

impl DuckDBTransaction for StatsTransaction {
    fn does_schema_exist(
        &self,
        _datastore: &str,
        schema: &str,
    ) -> duckdb_planner::catalog_provider::Result<bool> {
        Ok(schema == "main")
    }

    fn bind_table(
        &self,
        _datastore: &str,
        _schema: &str,
        table_name: &str,
    ) -> duckdb_planner::catalog_provider::Result<Option<Box<dyn DuckDBTable>>> {
        let rows = match table_name {
            "small" => 10,
            "big" => 1_000_000,
            _ => return Ok(None),
        };
        Ok(Some(Box::new(CountingTable {
            rows,
            asked: self.asked.clone(),
        })))
    }
}

#[test]
fn join_planning_consults_table_row_counts() {
    let asked = Arc::new(AtomicUsize::new(0));
    let mut planner =
        PlannerContext::new(Vec::new(), vec!["db".to_string()], "db".to_string()).unwrap();

    planner
        .plan(
            "SELECT small.id FROM small JOIN big ON small.id = big.id",
            Vec::new(),
            Arc::new(StatsTransaction {
                asked: asked.clone(),
            }),
        )
        .unwrap();

    assert!(asked.load(Ordering::Relaxed) > 0);
}
