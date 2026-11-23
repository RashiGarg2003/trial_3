"""
Standalone tests for SQLAgent prompt and output handling.
"""

from agents.sql_agent import SQLAgent


def test_extract_sql_truncates_at_first_statement() -> bool:
    agent = SQLAgent()
    model_output = """
```sql
SELECT ProductName
FROM Products
ORDER BY ProductName ASC
LIMIT 5;
```
Additional prose should not be included.
"""

    sql = agent._extract_sql(model_output)

    if "Additional prose" in sql:
        raise AssertionError(
            "SQL extraction retained text after the statement."
        )
    if not sql.upper().startswith("SELECT"):
        raise AssertionError(
            "SQL extraction did not locate the SELECT statement."
        )
    if not sql.endswith(";"):
        raise AssertionError(
            "SQL extraction did not retain the statement terminator."
        )
    return True


def test_compact_prompt_preserves_retry_context() -> bool:
    agent = SQLAgent()
    prompt = """
BUSINESS QUESTION:

Show the top 5 products by revenue.

ACTUAL SQLITE SCHEMA:

TABLE: [Order Details]
COLUMNS:
- UnitPrice
- Quantity

Generate one read-only SQLite query.

The previous SQL attempt failed.

Previous SQL:

SELECT Orders.UnitPrice FROM Orders;

Validation or database error:

no such column: Orders.UnitPrice

Generate a corrected complete query using the supplied schema.
"""

    compact = agent._compact_prompt(prompt)

    if "Show the top 5 products" not in compact:
        raise AssertionError("Compact prompt lost the question.")
    if "Order Details" not in compact:
        raise AssertionError("Compact prompt lost the schema.")
    if "no such column" not in compact:
        raise AssertionError("Compact prompt lost the error.")
    if len(compact) >= len(prompt):
        raise AssertionError("Compact prompt was not compact.")
    return True


def test_canonicalize_order_details_name() -> bool:
    agent = SQLAgent()
    database_schema = """
TABLE: [Order Details]
COLUMNS:
- OrderID
- ProductID
- UnitPrice
- Quantity

TABLE: Products
COLUMNS:
- ProductID
- ProductName
"""
    sql = """
SELECT Products.ProductName,
       SUM(order_details.UnitPrice * order_details.Quantity)
FROM order_details
JOIN Products ON order_details.ProductID = Products.ProductID
GROUP BY Products.ProductName
LIMIT 5;
"""

    canonical = agent._canonicalize_schema_identifiers(
        sql,
        database_schema,
    )

    if "order_details" in canonical.lower():
        raise AssertionError(
            "Snake-case table name was not canonicalized."
        )
    if '"Order Details"' not in canonical:
        raise AssertionError(
            "Exact quoted schema table name was not restored."
        )
    return True


def test_canonicalize_misspelled_unit_price() -> bool:
    agent = SQLAgent()
    database_schema = """
TABLE: [Order Details]
COLUMNS:
- UnitPrice NUMERIC
- Quantity INTEGER
"""
    sql = (
        'SELECT SUM("Order Details".Unitity '
        '* "Order Details".Quantity) '
        'FROM "Order Details";'
    )

    canonical = agent._canonicalize_schema_identifiers(
        sql,
        database_schema,
    )

    if "Unitity" in canonical:
        raise AssertionError(
            "Misspelled column was not corrected."
        )
    if "UnitPrice" not in canonical:
        raise AssertionError(
            "Exact UnitPrice column was not restored."
        )
    return True


def test_schema_validation_rejects_column_as_table() -> bool:
    agent = SQLAgent()
    database_schema = """
TABLE: [Order Details]
COLUMNS:
- OrderID
- ProductID
- UnitPrice
- Quantity

TABLE: Products
COLUMNS:
- ProductID
- ProductName
"""
    sql = (
        'SELECT p.ProductName FROM "ProductID" AS p '
        'JOIN "Order Details" AS od '
        'ON od.ProductID = p.ProductID'
    )

    errors = agent._validate_schema_names(
        sql,
        database_schema,
    )

    if not any(
        "ProductID" in error
        for error in errors
    ):
        raise AssertionError(
            "Column-as-table SQL was not rejected."
        )
    return True


def test_schema_revenue_ranking_fallback() -> bool:
    from models.schemas import IntentResult, TimePeriod

    agent = SQLAgent()
    intent = IntentResult(
        original_question=(
            "Show the top 5 products by revenue."
        ),
        rewritten_question=(
            "Rank the top five products by revenue."
        ),
        analysis_type="ranking",
        metrics=["revenue"],
        dimensions=["product"],
        filters=[],
        time_period=TimePeriod(),
        sort_order="descending",
        limit=5,
        clarification_required=False,
        clarification_question=None,
    )
    database_schema = """
TABLE: [Order Details]
COLUMNS:
- OrderID
- ProductID
- UnitPrice
- Quantity

TABLE: Products
COLUMNS:
- ProductID
- ProductName
"""

    sql = agent._build_schema_revenue_ranking_sql(
        intent,
        database_schema,
        5,
    )

    if sql is None:
        raise AssertionError(
            "Revenue ranking fallback was not built."
        )
    if '"Order Details"' not in sql:
        raise AssertionError(
            "Fallback did not quote the physical table."
        )
    if "UnitPrice * od.Quantity" not in sql:
        raise AssertionError(
            "Fallback did not calculate revenue."
        )
    if "LIMIT 5" not in sql:
        raise AssertionError(
            "Fallback did not apply the requested limit."
        )
    return True


def test_generate_sql_returns_deterministic_ranking() -> bool:
    from agents.sql_agent import SQLAgent
    from models.schemas import (
        AnalysisPlan,
        IntentResult,
        PlanStep,
        TimePeriod,
    )
    from database.db_manager import DatabaseManager
    from database.schema_manager import SchemaManager

    agent = SQLAgent()
    intent = IntentResult(
        original_question="Show the top 5 products by revenue.",
        rewritten_question="Rank products by revenue.",
        analysis_type="ranking",
        metrics=["revenue"],
        dimensions=["product"],
        filters=[],
        time_period=TimePeriod(),
        sort_order="descending",
        limit=5,
        clarification_required=False,
        clarification_question=None,
    )
    plan = AnalysisPlan(
        objective="Rank products by revenue",
        strategy="Calculate product revenue and sort descending",
        steps=[
            PlanStep(
                step_id=1,
                title="Calculate product revenue",
                description=(
                    "Join Order Details to Products "
                    "and sum revenue."
                ),
                step_type="data_retrieval",
                depends_on=[],
                required_metrics=["revenue"],
                required_dimensions=["product"],
                required_data=["Order Details", "Products"],
                expected_output="Ranked product revenue",
                requires_sql=True,
            )
        ],
    )
    schema = SchemaManager(
        DatabaseManager()
    ).get_schema_prompt(include_views=False)

    result = agent.generate_sql(
        user_question=intent.original_question,
        intent=intent,
        plan=plan,
        database_schema=schema,
    )

    if '"Order Details"' not in result.sql:
        raise AssertionError(
            "generate_sql did not return deterministic "
            "schema-derived SQL."
        )
    if "LIMIT 5" not in result.sql:
        raise AssertionError(
            "generate_sql did not apply the limit."
        )
    return True


def main() -> None:
    tests = [
        test_extract_sql_truncates_at_first_statement,
        test_compact_prompt_preserves_retry_context,
        test_canonicalize_order_details_name,
        test_canonicalize_misspelled_unit_price,
        test_schema_validation_rejects_column_as_table,
        test_schema_revenue_ranking_fallback,
        test_generate_sql_returns_deterministic_ranking,
    ]
    passed = 0

    for test in tests:
        try:
            test()
        except Exception as exc:
            print(f"[FAILED] {test.__name__}: {exc}")
        else:
            print(f"[PASSED] {test.__name__}")
            passed += 1

    print(f"Passed: {passed}/{len(tests)}")
    if passed != len(tests):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
