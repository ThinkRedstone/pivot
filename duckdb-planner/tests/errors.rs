use duckdb_planner::catalog_provider::ScalarFunctionDef;
use duckdb_planner::{
    DuckDBColumn, DuckDBTable, DuckDBTransaction, Error, LogicalTypeId, PlannerContext,
};
use std::sync::Arc;

struct TTable;

impl DuckDBTable for TTable {
    fn clone_box(&self) -> Box<dyn DuckDBTable> {
        Box::new(TTable)
    }

    fn duckdb_typed_columns(&self) -> Vec<DuckDBColumn> {
        vec![
            DuckDBColumn {
                name: "id".to_string(),
                duckdb_logical_type_id: LogicalTypeId::INTEGER as u8,
                decimal_width: 0,
                decimal_scale: 0,
            },
            DuckDBColumn {
                name: "name".to_string(),
                duckdb_logical_type_id: LogicalTypeId::VARCHAR as u8,
                decimal_width: 0,
                decimal_scale: 0,
            },
        ]
    }
}

struct TestTransaction;

impl DuckDBTransaction for TestTransaction {
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
        Ok(match table_name {
            "t" => Some(Box::new(TTable)),
            _ => None,
        })
    }
}

struct ScalarTransaction {
    defines_function: bool,
}

impl DuckDBTransaction for ScalarTransaction {
    fn scalar_function(&self, name: &str) -> Option<ScalarFunctionDef> {
        (self.defines_function && name == "transaction_scalar").then(|| ScalarFunctionDef {
            arg_type_ids: Vec::new(),
            return_type_id: LogicalTypeId::BIGINT as u8,
            is_volatile: false,
        })
    }

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
        _table_name: &str,
    ) -> duckdb_planner::catalog_provider::Result<Option<Box<dyn DuckDBTable>>> {
        Ok(None)
    }
}

fn create_simple_context() -> PlannerContext {
    PlannerContext::new(Vec::new(), vec!["db".to_string()], "db".to_string()).unwrap()
}

fn plan(p: &mut PlannerContext, query: &str) -> Result<duckdb_planner::Plan, Error> {
    p.plan(query, Vec::new(), Arc::new(TestTransaction))
}

#[test]
fn scalar_functions_resolve_through_the_current_transaction() {
    let mut planner = create_simple_context();

    planner
        .plan(
            "SELECT transaction_scalar()",
            Vec::new(),
            Arc::new(ScalarTransaction {
                defines_function: true,
            }),
        )
        .unwrap();

    let result = planner.plan(
        "SELECT transaction_scalar()",
        Vec::new(),
        Arc::new(ScalarTransaction {
            defines_function: false,
        }),
    );
    assert!(matches!(result, Err(Error::DuckDBPlanning(_))));
}

#[test]
fn invalid_syntax() {
    let mut p = create_simple_context();
    let result = plan(&mut p, "SELECTTT * FROM t");
    match result {
        Ok(_) => panic!("Expected error"),
        Err(Error::DuckDBPlanning(e)) => {
            assert!(e.exception_message.contains("syntax error at or near"))
        }
        Err(e) => panic!("Unexpected error: {e}"),
    }
}

#[test]
fn nonexistent_column() {
    let mut p = create_simple_context();
    let result = plan(&mut p, "SELECT nonexistent_column FROM t");
    match result {
        Ok(_) => panic!("Expected error"),
        Err(Error::DuckDBPlanning(e)) => {
            assert!(e.exception_message.contains("nonexistent_column"))
        }
        Err(e) => panic!("Unexpected error: {e}"),
    }
}

#[test]
fn nonexistent_table() {
    let mut p = create_simple_context();
    let result = plan(&mut p, "SELECT * FROM nonexistent_table");
    match result {
        Ok(_) => panic!("Expected error"),
        Err(Error::DuckDBPlanning(e)) => assert!(e.exception_message.contains("nonexistent_table")),
        Err(e) => panic!("Unexpected error: {e}"),
    }
}

/// The catalog holding DuckDB's built-in functions and types carries an
/// internal name in our fork, so `system` names nothing but Pivot's virtual
/// schema and no query reaches the built-in catalog by naming it.
#[test]
fn system_is_not_a_catalog() {
    let mut p = create_simple_context();

    for query in [
        "SELECT * FROM system.information_schema.tables",
        "SELECT * FROM system.main.duckdb_tables()",
        "SELECT system.main.current_database()",
        "DROP TABLE system.main.not_allowed",
    ] {
        match plan(&mut p, query) {
            Ok(_) => panic!("expected `{query}` to be rejected"),
            Err(Error::DuckDBPlanning(e)) => assert!(
                e.exception_message.contains("system"),
                "unexpected message for `{query}`: {}",
                e.exception_message
            ),
            Err(e) => panic!("unexpected error for `{query}`: {e}"),
        }
    }

    // Built-ins still resolve through that catalog internally: this macro
    // expands to a call qualified with it.
    plan(&mut p, "SELECT current_database()").unwrap();
}

/// Pivot cannot install or load DuckDB extensions, so stock DuckDB's "it
/// exists in the json extension, run INSTALL/LOAD" error steers users to a
/// dead end. Our duckdb fork strips those suggestions; this pins the plain
/// missing-function error so an upstream merge cannot quietly bring them back.
#[test]
fn function_known_only_to_extensions_errors_without_install_hint() {
    let mut p = create_simple_context();

    // `->>` exists in DuckDB's json extension, which pivot never ships.
    let result = plan(&mut p, "SELECT name->>'x' FROM t");

    match result {
        Ok(_) => panic!("Expected error"),
        Err(Error::DuckDBPlanning(e)) => assert_eq!(
            e.exception_message,
            "Scalar Function with name ->> does not exist!\nDid you mean \"-\"?"
        ),
        Err(e) => panic!("Unexpected error: {e}"),
    }
}

#[test]
fn exception_location() {
    let mut p = create_simple_context();
    let result = plan(&mut p, "SELECT * FROM nonexistent_table");
    match result {
        Ok(_) => panic!("Expected error"),
        Err(Error::DuckDBPlanning(e)) => assert_eq!(e.position.unwrap(), "14"),
        Err(e) => panic!("Unexpected error: {e}"),
    }
}
