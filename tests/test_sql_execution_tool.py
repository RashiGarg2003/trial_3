"""
Standalone tests for Agent 3b: SQL Validation and Execution.

These tests verify:

1. Valid SELECT execution.
2. Aggregate-query execution.
3. Parameter-free read-only execution.
4. Unknown-table rejection.
5. Unknown-column detection through EXPLAIN QUERY PLAN.
6. INSERT, UPDATE, DELETE, DROP, and CREATE rejection.
7. Multiple-statement rejection.
8. Empty and non-string input rejection.
9. Maximum-row enforcement.
10. Structured retryable error feedback.

Run from the project root:

    python -m tests.test_sql_execution_tool
"""

import sys
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from database.db_manager import DatabaseManager
from models.schemas import QueryExecutionResult
from tools.sql_execution_tool import (
    SQLExecutionOutcome,
    SQLExecutionTool,
)
from tools.sql_validator import SQLValidator
from security.models import UserRole
from security.schema_policy import SchemaPolicyValidator


def print_test_header(
    test_number: int,
    test_name: str,
) -> None:
    """
    Print a readable test header.
    """

    print("\n" + "=" * 80)
    print(f"TEST {test_number}: {test_name}")
    print("=" * 80)


def quote_identifier(
    identifier: str,
) -> str:
    """
    Safely quote an SQLite identifier used by the tests.

    The identifiers passed here come from SQLite metadata.
    """

    escaped_identifier = identifier.replace(
        '"',
        '""',
    )

    return f'"{escaped_identifier}"'


def print_outcome(
    outcome: SQLExecutionOutcome,
) -> None:
    """
    Print the outcome returned by Agent 3b.
    """

    print(f"Success           : {outcome.success}")
    print(f"Stage             : {outcome.stage}")
    print(f"Retryable         : {outcome.retryable}")
    print(
        f"Validation errors : "
        f"{outcome.validation_errors}"
    )
    print(
        f"Execution error   : "
        f"{outcome.execution_error}"
    )

    if outcome.validated_sql:
        print("\nValidated SQL:")
        print(outcome.validated_sql)

    if outcome.query_result is not None:
        print("\nQuery result:")
        print(
            f"Columns        : "
            f"{outcome.query_result.columns}"
        )
        print(
            f"Rows returned  : "
            f"{outcome.query_result.row_count}"
        )
        print(
            f"Truncated      : "
            f"{outcome.query_result.truncated}"
        )
        print(
            f"Execution time : "
            f"{outcome.query_result.execution_time_ms} ms"
        )

        for row in outcome.query_result.rows:
            print(row)


def require_success(
    outcome: SQLExecutionOutcome,
) -> None:
    """
    Confirm that an execution outcome succeeded.
    """

    if not outcome.success:
        raise AssertionError(
            "Expected SQL execution to succeed.\n"
            f"Stage: {outcome.stage}\n"
            f"Validation errors: "
            f"{outcome.validation_errors}\n"
            f"Execution error: "
            f"{outcome.execution_error}"
        )

    if outcome.stage != "completed":
        raise AssertionError(
            "Successful execution should have "
            f"stage='completed', received "
            f"'{outcome.stage}'."
        )

    if outcome.query_result is None:
        raise AssertionError(
            "Successful execution returned no query result."
        )

    if outcome.retryable:
        raise AssertionError(
            "Successful execution should not be retryable."
        )


def require_failure(
    outcome: SQLExecutionOutcome,
) -> None:
    """
    Confirm that an execution outcome failed safely.
    """

    if outcome.success:
        raise AssertionError(
            "Expected execution to fail, "
            "but the query succeeded."
        )

    if outcome.query_result is not None:
        raise AssertionError(
            "Failed execution should not contain "
            "a query result."
        )

    has_validation_error = bool(
        outcome.validation_errors
    )

    has_execution_error = (
        isinstance(
            outcome.execution_error,
            str,
        )
        and bool(
            outcome.execution_error.strip()
        )
    )

    if not (
        has_validation_error
        or has_execution_error
    ):
        raise AssertionError(
            "Failed execution did not provide "
            "error feedback."
        )


def get_test_table(
    database_manager: DatabaseManager,
) -> str:
    """
    Return one real database table dynamically.
    """

    tables = database_manager.get_tables()

    if not tables:
        raise RuntimeError(
            "The database contains no tables."
        )

    return tables[0]


def get_test_column(
    database_manager: DatabaseManager,
    table_name: str,
) -> str:
    """
    Return one real column from a discovered table.
    """

    columns = database_manager.get_columns(
        table_name
    )

    if not columns:
        raise RuntimeError(
            f"Table '{table_name}' contains no columns."
        )

    return str(columns[0]["name"])


def test_basic_select_execution(
    execution_tool: SQLExecutionTool,
) -> bool:
    """
    Execute a valid SELECT against a real table.
    """

    database_manager = (
        execution_tool.database_manager
    )

    table_name = get_test_table(
        database_manager
    )

    quoted_table = quote_identifier(
        table_name
    )

    sql = f"""
        SELECT *
        FROM {quoted_table}
        LIMIT 5;
    """

    outcome = (
        execution_tool.validate_and_execute(
            sql
        )
    )

    print_outcome(outcome)
    require_success(outcome)

    if outcome.query_result.row_count > 5:
        raise AssertionError(
            "The query returned more than five rows."
        )

    return True


def test_selected_column_execution(
    execution_tool: SQLExecutionTool,
) -> bool:
    """
    Execute a query using a dynamically discovered column.
    """

    database_manager = (
        execution_tool.database_manager
    )

    table_name = get_test_table(
        database_manager
    )

    column_name = get_test_column(
        database_manager,
        table_name,
    )

    quoted_table = quote_identifier(
        table_name
    )

    quoted_column = quote_identifier(
        column_name
    )

    sql = f"""
        SELECT {quoted_column}
        FROM {quoted_table}
        LIMIT 3;
    """

    outcome = (
        execution_tool.validate_and_execute(
            sql
        )
    )

    print_outcome(outcome)
    require_success(outcome)

    if column_name not in (
        outcome.query_result.columns
    ):
        raise AssertionError(
            "The selected column is missing from "
            "the query result."
        )

    return True


def test_aggregate_execution(
    execution_tool: SQLExecutionTool,
) -> bool:
    """
    Execute a generic aggregate query against a real table.
    """

    database_manager = (
        execution_tool.database_manager
    )

    table_name = get_test_table(
        database_manager
    )

    quoted_table = quote_identifier(
        table_name
    )

    sql = f"""
        SELECT COUNT(*) AS record_count
        FROM {quoted_table};
    """

    outcome = (
        execution_tool.validate_and_execute(
            sql
        )
    )

    print_outcome(outcome)
    require_success(outcome)

    if outcome.query_result.row_count != 1:
        raise AssertionError(
            "COUNT query should return exactly one row."
        )

    first_row = outcome.query_result.rows[0]

    if "record_count" not in first_row:
        raise AssertionError(
            "COUNT result does not contain "
            "'record_count'."
        )

    return True


def test_result_model_validation(
    execution_tool: SQLExecutionTool,
) -> bool:
    """
    Confirm successful results use the shared Pydantic model.
    """

    database_manager = (
        execution_tool.database_manager
    )

    table_name = get_test_table(
        database_manager
    )

    quoted_table = quote_identifier(
        table_name
    )

    outcome = (
        execution_tool.validate_and_execute(
            f"""
            SELECT *
            FROM {quoted_table}
            LIMIT 1;
            """
        )
    )

    print_outcome(outcome)
    require_success(outcome)

    if not isinstance(
        outcome.query_result,
        QueryExecutionResult,
    ):
        raise AssertionError(
            "Agent 3b did not return "
            "QueryExecutionResult."
        )

    return True


def test_result_row_cap(
    execution_tool: SQLExecutionTool,
) -> bool:
    """
    Confirm DatabaseManager enforces its maximum row limit.
    """

    database_manager = (
        execution_tool.database_manager
    )

    limited_manager = DatabaseManager(
        database_path=(
            database_manager.database_path
        ),
        max_rows=3,
    )

    limited_tool = SQLExecutionTool(
        database_manager=limited_manager,
        sql_validator=SQLValidator(
            dialect="sqlite"
        ),
        explain_before_execution=True,
    )

    table_name = get_test_table(
        limited_manager
    )

    quoted_table = quote_identifier(
        table_name
    )

    outcome = (
        limited_tool.validate_and_execute(
            f"""
            SELECT *
            FROM {quoted_table};
            """
        )
    )

    print_outcome(outcome)
    require_success(outcome)

    if outcome.query_result.row_count > 3:
        raise AssertionError(
            "Agent 3b returned more than "
            "the configured maximum rows."
        )

    return True


def test_unknown_table_rejection(
    execution_tool: SQLExecutionTool,
) -> bool:
    """
    Reject a query referencing a table outside the schema.
    """

    outcome = (
        execution_tool.validate_and_execute(
            """
            SELECT *
            FROM table_that_does_not_exist;
            """
        )
    )

    print_outcome(outcome)
    require_failure(outcome)

    if outcome.stage != "schema_validation":
        raise AssertionError(
            "Unknown table should fail during "
            f"schema validation, received "
            f"stage '{outcome.stage}'."
        )

    if not outcome.retryable:
        raise AssertionError(
            "Unknown-table SQL should be retryable "
            "by Agent 3a."
        )

    return True


def test_unknown_column_rejection(
    execution_tool: SQLExecutionTool,
) -> bool:
    """
    Detect a missing column through EXPLAIN QUERY PLAN.
    """

    database_manager = (
        execution_tool.database_manager
    )

    table_name = get_test_table(
        database_manager
    )

    quoted_table = quote_identifier(
        table_name
    )

    sql = f"""
        SELECT column_that_does_not_exist
        FROM {quoted_table};
    """

    outcome = (
        execution_tool.validate_and_execute(
            sql
        )
    )

    print_outcome(outcome)
    require_failure(outcome)

    if outcome.stage != "explain_validation":
        raise AssertionError(
            "Unknown column should fail during "
            f"EXPLAIN validation, received "
            f"stage '{outcome.stage}'."
        )

    if not outcome.execution_error:
        raise AssertionError(
            "Unknown-column failure should contain "
            "SQLite error feedback."
        )

    return True


def test_missing_column_reports_actual_owner(
    execution_tool: SQLExecutionTool,
) -> bool:
    """
    Point retry feedback to the table that declares a missing column.
    """
    outcome = (
        execution_tool.validate_and_execute(
            """
            SELECT SUM(orders.UnitPrice)
            FROM Orders AS orders;
            """
        )
    )

    print_outcome(outcome)
    require_failure(outcome)

    error_text = outcome.execution_error or ""
    if "no such column" not in error_text.lower():
        raise AssertionError(
            "Missing-column SQL did not preserve SQLite feedback."
        )

    if (
        "Order Details" not in error_text
        or "UnitPrice" not in error_text
    ):
        raise AssertionError(
            "Missing-column feedback did not identify "
            "the actual column owner."
        )

    return True


def test_empty_sql_rejection(
    execution_tool: SQLExecutionTool,
) -> bool:
    """
    Reject empty SQL before parsing.
    """

    outcome = (
        execution_tool.validate_and_execute(
            "   "
        )
    )

    print_outcome(outcome)
    require_failure(outcome)

    if outcome.stage != "input_validation":
        raise AssertionError(
            "Empty SQL should fail during "
            "input validation."
        )

    return True


def test_non_string_sql_rejection(
    execution_tool: SQLExecutionTool,
) -> bool:
    """
    Reject non-string SQL safely.
    """

    outcome = (
        execution_tool.validate_and_execute(
            123  # type: ignore[arg-type]
        )
    )

    print_outcome(outcome)
    require_failure(outcome)

    if outcome.stage != "input_validation":
        raise AssertionError(
            "Non-string SQL should fail during "
            "input validation."
        )

    return True


def test_multiple_statements_rejection(
    execution_tool: SQLExecutionTool,
) -> bool:
    """
    Reject multiple statements.
    """

    database_manager = (
        execution_tool.database_manager
    )

    table_name = get_test_table(
        database_manager
    )

    quoted_table = quote_identifier(
        table_name
    )

    sql = f"""
        SELECT * FROM {quoted_table};
        SELECT * FROM {quoted_table};
    """

    outcome = (
        execution_tool.validate_and_execute(
            sql
        )
    )

    print_outcome(outcome)
    require_failure(outcome)

    if outcome.stage != "structural_validation":
        raise AssertionError(
            "Multiple statements should fail during "
            "structural validation."
        )

    return True


def test_insert_rejection(
    execution_tool: SQLExecutionTool,
) -> bool:
    """
    Reject INSERT statements.
    """

    outcome = (
        execution_tool.validate_and_execute(
            """
            INSERT INTO forbidden_table (
                value
            )
            VALUES (
                1
            );
            """
        )
    )

    print_outcome(outcome)
    require_failure(outcome)

    if outcome.stage != "structural_validation":
        raise AssertionError(
            "INSERT should fail during "
            "structural validation."
        )

    return True


def test_update_rejection(
    execution_tool: SQLExecutionTool,
) -> bool:
    """
    Reject UPDATE statements.
    """

    database_manager = (
        execution_tool.database_manager
    )

    table_name = get_test_table(
        database_manager
    )

    quoted_table = quote_identifier(
        table_name
    )

    outcome = (
        execution_tool.validate_and_execute(
            f"""
            UPDATE {quoted_table}
            SET rowid = rowid;
            """
        )
    )

    print_outcome(outcome)
    require_failure(outcome)

    return True


def test_delete_rejection(
    execution_tool: SQLExecutionTool,
) -> bool:
    """
    Reject DELETE statements.
    """

    database_manager = (
        execution_tool.database_manager
    )

    table_name = get_test_table(
        database_manager
    )

    quoted_table = quote_identifier(
        table_name
    )

    outcome = (
        execution_tool.validate_and_execute(
            f"""
            DELETE FROM {quoted_table};
            """
        )
    )

    print_outcome(outcome)
    require_failure(outcome)

    return True


def test_drop_rejection(
    execution_tool: SQLExecutionTool,
) -> bool:
    """
    Reject DROP statements.
    """

    database_manager = (
        execution_tool.database_manager
    )

    table_name = get_test_table(
        database_manager
    )

    quoted_table = quote_identifier(
        table_name
    )

    outcome = (
        execution_tool.validate_and_execute(
            f"""
            DROP TABLE {quoted_table};
            """
        )
    )

    print_outcome(outcome)
    require_failure(outcome)

    return True


def test_create_rejection(
    execution_tool: SQLExecutionTool,
) -> bool:
    """
    Reject CREATE statements.
    """

    outcome = (
        execution_tool.validate_and_execute(
            """
            CREATE TABLE forbidden_table (
                id INTEGER PRIMARY KEY
            );
            """
        )
    )

    print_outcome(outcome)
    require_failure(outcome)

    return True


def test_markdown_select_execution(
    execution_tool: SQLExecutionTool,
) -> bool:
    """
    Accept a safe SELECT wrapped in a Markdown SQL block.
    """

    database_manager = (
        execution_tool.database_manager
    )

    table_name = get_test_table(
        database_manager
    )

    quoted_table = quote_identifier(
        table_name
    )

    sql = f"""```sql
SELECT *
FROM {quoted_table}
LIMIT 2;
```"""

    outcome = (
        execution_tool.validate_and_execute(
            sql
        )
    )

    print_outcome(outcome)
    require_success(outcome)

    return True


def run_test(
    test_number: int,
    test_name: str,
    test_function: Callable[
        [SQLExecutionTool],
        bool,
    ],
    execution_tool: SQLExecutionTool,
) -> bool:
    """
    Execute one Agent 3b test.
    """

    print_test_header(
        test_number=test_number,
        test_name=test_name,
    )

    try:
        test_function(
            execution_tool
        )

    except Exception as exc:
        print("\n[FAILED]")
        print(
            f"{type(exc).__name__}: {exc}"
        )

        return False

    print("\n[PASSED]")
    return True


def main() -> None:
    """
    Run all Agent 3b tests.
    """

    print("\nAGENT 3B TEST SUITE")
    print(
        "Component: SQL Validation and Execution"
    )

    try:
        database_manager = DatabaseManager(
            max_rows=100,
        )

        sql_validator = SQLValidator(
            dialect="sqlite"
        )

        execution_tool = SQLExecutionTool(
            database_manager=database_manager,
            sql_validator=sql_validator,
            explain_before_execution=True,
        )

    except Exception as exc:
        print(
            "\nCould not initialize Agent 3b."
        )
        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise SystemExit(1) from exc

    tests: list[
        tuple[
            str,
            Callable[
                [SQLExecutionTool],
                bool,
            ],
        ]
    ] = [
        (
            "Basic SELECT execution",
            test_basic_select_execution,
        ),
        (
            "Selected-column execution",
            test_selected_column_execution,
        ),
        (
            "Aggregate execution",
            test_aggregate_execution,
        ),
        (
            "Shared result-model validation",
            test_result_model_validation,
        ),
        (
            "Maximum result-row cap",
            test_result_row_cap,
        ),
        (
            "Unknown-table rejection",
            test_unknown_table_rejection,
        ),
        (
            "Unknown-column rejection",
            test_unknown_column_rejection,
        ),
        (
            "Missing-column owner feedback",
            test_missing_column_reports_actual_owner,
        ),
        (
            "Empty SQL rejection",
            test_empty_sql_rejection,
        ),
        (
            "Non-string SQL rejection",
            test_non_string_sql_rejection,
        ),
        (
            "Multiple statements rejection",
            test_multiple_statements_rejection,
        ),
        (
            "INSERT rejection",
            test_insert_rejection,
        ),
        (
            "UPDATE rejection",
            test_update_rejection,
        ),
        (
            "DELETE rejection",
            test_delete_rejection,
        ),
        (
            "DROP rejection",
            test_drop_rejection,
        ),
        (
            "CREATE rejection",
            test_create_rejection,
        ),
        (
            "Markdown SELECT execution",
            test_markdown_select_execution,
        ),
    ]

    passed_tests = 0

    for test_number, (
        test_name,
        test_function,
    ) in enumerate(
        tests,
        start=1,
    ):
        passed = run_test(
            test_number=test_number,
            test_name=test_name,
            test_function=test_function,
            execution_tool=execution_tool,
        )

        if passed:
            passed_tests += 1

    total_tests = len(tests)
    failed_tests = (
        total_tests - passed_tests
    )

    print("\n" + "=" * 80)
    print(
        "FINAL AGENT 3B TEST SUMMARY"
    )
    print("=" * 80)
    print(f"Total tests : {total_tests}")
    print(f"Passed      : {passed_tests}")
    print(f"Failed      : {failed_tests}")

    if failed_tests == 0:
        print(
            "\nRESULT: ALL AGENT 3B TESTS PASSED"
        )
        print(
            "SQL validation and execution are ready "
            "for LangGraph integration."
        )

        raise SystemExit(0)

    print(
        "\nRESULT: SOME AGENT 3B TESTS FAILED"
    )
    print(
        "Resolve the failed cases before adding "
        "the SQL retry loop to LangGraph."
    )

    raise SystemExit(1)


def test_sql_execution_outcome_has_completed_stage():
    database_manager = DatabaseManager(max_rows=100)
    sql_validator = SQLValidator(dialect="sqlite")
    schema_policy_validator = SchemaPolicyValidator(
        database_manager=database_manager,
        sql_validator=sql_validator,
        maximum_result_rows=100,
    )
    execution_tool = SQLExecutionTool(
        database_manager=database_manager,
        sql_validator=sql_validator,
        schema_policy_validator=schema_policy_validator,
        role=UserRole.ANALYST,
    )

    outcome = execution_tool.validate_and_execute(
        'SELECT "CategoryID" '
        'FROM "Categories" '
        "LIMIT 5;"
    )

    assert outcome.success
    assert outcome.stage == "completed"


if __name__ == "__main__":
    main()