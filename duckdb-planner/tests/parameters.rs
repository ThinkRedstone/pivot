use duckdb_planner::handle::{Expression, Operator};
use duckdb_planner::{
    BoundLogicalType, DuckDBTable, DuckDBTransaction, Error, ExtraTypeInfo, LogicalTypeId,
    PlannerContext, ScalarValue,
};
use std::sync::Arc;

struct EmptyTransaction;

impl DuckDBTransaction for EmptyTransaction {
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

fn create_context() -> PlannerContext {
    PlannerContext::new(Vec::new(), vec!["db".to_string()], "db".to_string()).unwrap()
}

#[test]
fn plan_binds_parameter_as_constant() {
    let mut context = create_context();

    let plan = context
        .plan(
            "SELECT $1",
            vec![ScalarValue::Int32(7)],
            Arc::new(EmptyTransaction),
        )
        .unwrap();

    let Operator::Projection(projection) = plan.root().unwrap().operator().unwrap() else {
        panic!("expected a projection");
    };
    let Expression::Constant(constant) = projection.exprs().unwrap()[0].expression().unwrap()
    else {
        panic!("expected a constant");
    };
    assert!(matches!(constant.value().unwrap(), ScalarValue::Int32(7)));
}

#[test]
fn plan_reuses_parameter_among_several() {
    let mut context = create_context();

    let plan = context
        .plan(
            "SELECT $1 + $2 + $1",
            vec![ScalarValue::Int32(3), ScalarValue::Int32(10)],
            Arc::new(EmptyTransaction),
        )
        .unwrap();

    let Operator::Projection(projection) = plan.root().unwrap().operator().unwrap() else {
        panic!("expected a projection");
    };
    let Expression::Constant(constant) = projection.exprs().unwrap()[0].expression().unwrap()
    else {
        panic!("expected the sum folded into a constant");
    };
    assert!(matches!(constant.value().unwrap(), ScalarValue::Int32(16)));
}

fn build_integer_list(elements: Vec<ScalarValue>) -> ScalarValue {
    ScalarValue::List {
        element_type: BoundLogicalType::plain(LogicalTypeId::INTEGER),
        elements,
    }
}

#[test]
fn plan_binds_list_parameter_as_constant() {
    let mut context = create_context();
    let list = build_integer_list(vec![ScalarValue::Int32(1), ScalarValue::Int32(2)]);

    let plan = context
        .plan("SELECT unnest($1)", vec![list], Arc::new(EmptyTransaction))
        .unwrap();

    let projection = plan.root().unwrap();
    let Operator::Unnest(unnest) = projection.child(0).unwrap().operator().unwrap() else {
        panic!("expected an unnest below the projection");
    };
    let Expression::Constant(constant) = unnest.lists().unwrap()[0].list.expression().unwrap()
    else {
        panic!("expected a constant list");
    };
    assert!(matches!(
        constant.list_elements().unwrap()[..],
        [ScalarValue::Int32(1), ScalarValue::Int32(2)]
    ));
}

#[test]
fn plan_rejects_list_parameter_with_unconvertible_element() {
    let mut context = create_context();
    let list = build_integer_list(vec![ScalarValue::Utf8("abc".to_string())]);

    let result = context.plan("SELECT unnest($1)", vec![list], Arc::new(EmptyTransaction));

    assert!(matches!(result, Err(Error::Bridge(_))));
}

#[test]
fn plan_rejects_list_parameter_without_element_type() {
    let mut context = create_context();
    let list = ScalarValue::Null(BoundLogicalType::plain(LogicalTypeId::LIST));

    let result = context.plan("SELECT unnest($1)", vec![list], Arc::new(EmptyTransaction));

    assert!(matches!(
        result,
        Err(Error::IncompleteParameterType(LogicalTypeId::LIST))
    ));
}

#[test]
fn plan_rejects_missing_parameter() {
    let mut context = create_context();

    let result = context.plan("SELECT $1", Vec::new(), Arc::new(EmptyTransaction));

    let Err(Error::DuckDBPlanning(error)) = result else {
        panic!("expected a planning error");
    };
    assert!(
        error
            .exception_message
            .contains("statement takes 1 parameters, 0 were supplied")
    );
}

#[test]
fn plan_rejects_surplus_parameter() {
    let mut context = create_context();

    let result = context.plan(
        "SELECT 1",
        vec![ScalarValue::Int32(7)],
        Arc::new(EmptyTransaction),
    );

    let Err(Error::DuckDBPlanning(error)) = result else {
        panic!("expected a planning error");
    };
    assert!(
        error
            .exception_message
            .contains("statement takes 0 parameters, 1 were supplied")
    );
}

#[test]
fn plan_rejects_parameter_without_value() {
    let mut context = create_context();

    let result = context.plan(
        "SELECT $1",
        vec![ScalarValue::Other(LogicalTypeId::BLOB)],
        Arc::new(EmptyTransaction),
    );

    assert!(matches!(
        result,
        Err(Error::UnsupportedParameterType(LogicalTypeId::BLOB))
    ));
}

#[test]
fn infer_parameter_types_reads_casts() {
    let mut context = create_context();

    let types = context
        .infer_parameter_types(
            "SELECT $1::DOUBLE, $2::INTEGER[]",
            Arc::new(EmptyTransaction),
        )
        .unwrap();

    let integer_list = BoundLogicalType {
        id: LogicalTypeId::LIST,
        extra: ExtraTypeInfo::List(Box::new(BoundLogicalType::plain(LogicalTypeId::INTEGER))),
    };
    assert_eq!(
        types,
        vec![BoundLogicalType::plain(LogicalTypeId::DOUBLE), integer_list]
    );
}

#[test]
fn infer_parameter_types_reports_unresolved_placeholder_as_unknown() {
    let mut context = create_context();

    let types = context
        .infer_parameter_types("SELECT $1", Arc::new(EmptyTransaction))
        .unwrap();

    assert_eq!(types, vec![BoundLogicalType::plain(LogicalTypeId::UNKNOWN)]);
}

#[test]
fn infer_parameter_types_lists_reused_placeholder_once() {
    let mut context = create_context();

    let types = context
        .infer_parameter_types("SELECT $1::BIGINT + $1", Arc::new(EmptyTransaction))
        .unwrap();

    assert_eq!(types, vec![BoundLogicalType::plain(LogicalTypeId::BIGINT)]);
}

#[test]
fn infer_parameter_types_of_statement_without_placeholders_is_empty() {
    let mut context = create_context();

    let types = context
        .infer_parameter_types("SELECT 1", Arc::new(EmptyTransaction))
        .unwrap();

    assert!(types.is_empty());
}

#[test]
fn infer_parameter_types_rejects_unknown_table() {
    let mut context = create_context();

    let result = context.infer_parameter_types(
        "SELECT * FROM missing_table WHERE x = $1",
        Arc::new(EmptyTransaction),
    );

    assert!(matches!(result, Err(Error::DuckDBPlanning(_))));
}
