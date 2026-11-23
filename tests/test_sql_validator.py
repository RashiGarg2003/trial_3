"""
Standalone tests for SQLValidator.

Run from the project root:

    python -m tests.test_sql_validator
"""

import sys
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from tools.sql_validator import (
    SQLValidationError,
    SQLValidationResult,
    SQLValidator,
)


def print_test_header(
    test_number: int,
    test_name: str,
) -> None:
    """
    Print a formatted test heading.
    """

    print("\n" + "=" * 80)
    print(f"TEST {test_number}: {test_name}")
    print("=" * 80)


def print_validation_result(
    result: SQLValidationResult,
) -> None:
    """
    Print the SQL validation result.
    """

    print(f"Valid              : {result.is_valid}")
    print(f"Statement type     : {result.statement_type}")
    print(f"Referenced tables  : {result.referenced_tables}")
    print(f"Referenced columns : {result.referenced_columns}")
    print(f"Errors             : {result.errors}")

    if result.normalized_sql:
        print("\nNormalized SQL:")
        print(result.normalized_sql)


def require_valid(
    result: SQLValidationResult,
) -> None:
    """
    Assert that a result is valid.
    """

    if not result.is_valid:
        raise AssertionError(
            "Expected valid SQL, but validation failed:\n"
            + "\n".join(result.errors)
        )


def require_invalid(
    result: SQLValidationResult,
) -> None:
    """
    Assert that a result is invalid.
    """

    if result.is_valid:
        raise AssertionError(
            "Expected SQL validation to fail, "
            "but the query was accepted."
        )

    if not result.errors:
        raise AssertionError(
            "Invalid SQL did not return validation errors."
        )


def test_basic_select(
    validator: SQLValidator,
) -> bool:
    """
    Confirm that a basic SELECT query is accepted.
    """

    sql = """
        SELECT
            ProductName,
            UnitPrice
        FROM Products
        LIMIT 5;
    """

    result = validator.validate(sql)

    print_validation_result(result)
    require_valid(result)

    if "Products" not in result.referenced_tables:
        raise AssertionError(
            "The Products table was not detected."
        )

    return True


def test_join_query(
    validator: SQLValidator,
) -> bool:
    """
    Confirm that a read-only JOIN query is accepted.
    """

    sql = """
        SELECT
            p.ProductName,
            c.CategoryName
        FROM Products AS p
        JOIN Categories AS c
            ON p.CategoryID = c.CategoryID;
    """

    result = validator.validate(sql)

    print_validation_result(result)
    require_valid(result)

    expected_tables = {
        "Products",
        "Categories",
    }

    actual_tables = set(
        result.referenced_tables
    )

    if not expected_tables.issubset(actual_tables):
        raise AssertionError(
            "Expected joined tables were not detected.\n"
            f"Expected: {sorted(expected_tables)}\n"
            f"Received: {result.referenced_tables}"
        )

    return True


def test_aggregate_query(
    validator: SQLValidator,
) -> bool:
    """
    Confirm that aggregation and grouping are accepted.
    """

    sql = """
        SELECT
            CategoryID,
            COUNT(*) AS product_count,
            AVG(UnitPrice) AS average_price
        FROM Products
        GROUP BY CategoryID
        ORDER BY average_price DESC;
    """

    result = validator.validate(sql)

    print_validation_result(result)
    require_valid(result)

    if result.statement_type != "Select":
        raise AssertionError(
            "Expected statement type Select, "
            f"received {result.statement_type}."
        )

    return True


def test_common_table_expression(
    validator: SQLValidator,
) -> bool:
    """
    Confirm that a read-only common table expression is accepted.
    """

    sql = """
        WITH product_totals AS (
            SELECT
                ProductID,
                SUM(Quantity) AS total_quantity
            FROM "Order Details"
            GROUP BY ProductID
        )
        SELECT
            ProductID,
            total_quantity
        FROM product_totals
        ORDER BY total_quantity DESC
        LIMIT 5;
    """

    result = validator.validate(sql)

    print_validation_result(result)
    require_valid(result)

    if "Order Details" not in result.referenced_tables:
        raise AssertionError(
            "The physical Order Details table "
            "was not detected."
        )

    if "product_totals" in result.referenced_tables:
        raise AssertionError(
            "The CTE alias was incorrectly treated "
            "as a physical database table."
        )

    return True


def test_union_query(
    validator: SQLValidator,
) -> bool:
    """
    Confirm that a read-only UNION query is accepted.
    """

    sql = """
        SELECT Country
        FROM Customers

        UNION

        SELECT Country
        FROM Suppliers;
    """

    result = validator.validate(sql)

    print_validation_result(result)
    require_valid(result)

    expected_tables = {
        "Customers",
        "Suppliers",
    }

    if not expected_tables.issubset(
        set(result.referenced_tables)
    ):
        raise AssertionError(
            "UNION table references were not "
            "extracted correctly."
        )

    return True


def test_markdown_sql_cleanup(
    validator: SQLValidator,
) -> bool:
    """
    Confirm that SQL Markdown fences are removed.
    """

    sql = """```sql
SELECT ProductName
FROM Products
LIMIT 3;
```"""

    cleaned_sql = validator.clean_sql(sql)

    if "```" in cleaned_sql:
        raise AssertionError(
            "Markdown code fences were not removed."
        )

    result = validator.validate(sql)

    print(f"Cleaned SQL:\n{cleaned_sql}\n")
    print_validation_result(result)

    require_valid(result)

    return True


def test_sqlite_markdown_cleanup(
    validator: SQLValidator,
) -> bool:
    """
    Confirm that SQLite Markdown fences are removed.
    """

    sql = """```sqlite
SELECT CustomerID
FROM Customers
LIMIT 2;
```"""

    result = validator.validate(sql)

    print_validation_result(result)
    require_valid(result)

    return True


def test_empty_sql_rejection(
    validator: SQLValidator,
) -> bool:
    """
    Confirm that empty SQL is rejected.
    """

    result = validator.validate("   ")

    print_validation_result(result)
    require_invalid(result)

    return True


def test_invalid_syntax_rejection(
    validator: SQLValidator,
) -> bool:
    """
    Confirm that clearly malformed SQL is rejected.
    """

    sql = """
        SELECT (
        FROM Products;
    """

    result = validator.validate(sql)

    print_validation_result(result)
    require_invalid(result)

    return True


def test_multiple_statements_rejection(
    validator: SQLValidator,
) -> bool:
    """
    Confirm that multiple SQL statements are rejected.
    """

    sql = """
        SELECT * FROM Products;
        SELECT * FROM Categories;
    """

    result = validator.validate(sql)

    print_validation_result(result)
    require_invalid(result)

    if not any(
        "Exactly one SQL statement" in error
        for error in result.errors
    ):
        raise AssertionError(
            "The result did not explain that only "
            "one statement is allowed."
        )

    return True


def test_insert_rejection(
    validator: SQLValidator,
) -> bool:
    """
    Confirm that INSERT statements are rejected.
    """

    sql = """
        INSERT INTO Products (
            ProductName
        )
        VALUES (
            'Forbidden Product'
        );
    """

    result = validator.validate(sql)

    print_validation_result(result)
    require_invalid(result)

    return True


def test_update_rejection(
    validator: SQLValidator,
) -> bool:
    """
    Confirm that UPDATE statements are rejected.
    """

    sql = """
        UPDATE Products
        SET UnitPrice = 0;
    """

    result = validator.validate(sql)

    print_validation_result(result)
    require_invalid(result)

    return True


def test_delete_rejection(
    validator: SQLValidator,
) -> bool:
    """
    Confirm that DELETE statements are rejected.
    """

    sql = """
        DELETE FROM Products;
    """

    result = validator.validate(sql)

    print_validation_result(result)
    require_invalid(result)

    return True


def test_drop_rejection(
    validator: SQLValidator,
) -> bool:
    """
    Confirm that DROP statements are rejected.
    """

    sql = """
        DROP TABLE Products;
    """

    result = validator.validate(sql)

    print_validation_result(result)
    require_invalid(result)

    return True


def test_create_rejection(
    validator: SQLValidator,
) -> bool:
    """
    Confirm that CREATE statements are rejected.
    """

    sql = """
        CREATE TABLE forbidden_table (
            id INTEGER PRIMARY KEY
        );
    """

    result = validator.validate(sql)

    print_validation_result(result)
    require_invalid(result)

    return True


def test_alter_rejection(
    validator: SQLValidator,
) -> bool:
    """
    Confirm that ALTER statements are rejected.
    """

    sql = """
        ALTER TABLE Products
        ADD COLUMN forbidden_column TEXT;
    """

    result = validator.validate(sql)

    print_validation_result(result)
    require_invalid(result)

    return True


def test_select_with_write_statement_rejection(
    validator: SQLValidator,
) -> bool:
    """
    Confirm that a SELECT followed by a write statement is rejected.
    """

    sql = """
        SELECT * FROM Products;
        DROP TABLE Products;
    """

    result = validator.validate(sql)

    print_validation_result(result)
    require_invalid(result)

    return True


def test_validate_or_raise_success(
    validator: SQLValidator,
) -> bool:
    """
    Confirm that validate_or_raise returns valid results.
    """

    result = validator.validate_or_raise(
        """
        SELECT
            ProductID,
            ProductName
        FROM Products
        LIMIT 5;
        """
    )

    print_validation_result(result)
    require_valid(result)

    return True


def test_validate_or_raise_failure(
    validator: SQLValidator,
) -> bool:
    """
    Confirm that validate_or_raise raises for unsafe SQL.
    """

    try:
        validator.validate_or_raise(
            "DELETE FROM Products;"
        )

    except SQLValidationError as exc:
        print(f"Expected validation error:\n{exc}")
        return True

    raise AssertionError(
        "validate_or_raise accepted unsafe SQL."
    )


def test_non_string_input_rejection(
    validator: SQLValidator,
) -> bool:
    """
    Confirm that non-string input is rejected safely.
    """

    result = validator.validate(
        123  # type: ignore[arg-type]
    )

    print_validation_result(result)
    require_invalid(result)

    return True


def run_test(
    test_number: int,
    test_name: str,
    test_function: Callable[
        [SQLValidator],
        bool,
    ],
    validator: SQLValidator,
) -> bool:
    """
    Execute one SQLValidator test.
    """

    print_test_header(
        test_number=test_number,
        test_name=test_name,
    )

    try:
        test_function(validator)

    except Exception as exc:
        print("\n[FAILED]")
        print(f"{type(exc).__name__}: {exc}")
        return False

    print("\n[PASSED]")
    return True


def main() -> None:
    """
    Run the complete SQLValidator test suite.
    """

    print("\nSQL VALIDATOR TEST SUITE")
    print("Dialect: SQLite")

    try:
        validator = SQLValidator(
            dialect="sqlite"
        )

    except Exception as exc:
        print("\nCould not initialize SQLValidator.")
        print(f"{type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc

    tests: list[
        tuple[
            str,
            Callable[[SQLValidator], bool],
        ]
    ] = [
        (
            "Basic SELECT query",
            test_basic_select,
        ),
        (
            "JOIN query",
            test_join_query,
        ),
        (
            "Aggregate query",
            test_aggregate_query,
        ),
        (
            "Common table expression",
            test_common_table_expression,
        ),
        (
            "UNION query",
            test_union_query,
        ),
        (
            "SQL Markdown cleanup",
            test_markdown_sql_cleanup,
        ),
        (
            "SQLite Markdown cleanup",
            test_sqlite_markdown_cleanup,
        ),
        (
            "Empty SQL rejection",
            test_empty_sql_rejection,
        ),
        (
            "Invalid SQL syntax rejection",
            test_invalid_syntax_rejection,
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
            "ALTER rejection",
            test_alter_rejection,
        ),
        (
            "SELECT plus write rejection",
            test_select_with_write_statement_rejection,
        ),
        (
            "validate_or_raise success",
            test_validate_or_raise_success,
        ),
        (
            "validate_or_raise failure",
            test_validate_or_raise_failure,
        ),
        (
            "Non-string input rejection",
            test_non_string_input_rejection,
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
            validator=validator,
        )

        if passed:
            passed_tests += 1

    total_tests = len(tests)
    failed_tests = total_tests - passed_tests

    print("\n" + "=" * 80)
    print("FINAL SQL VALIDATOR TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests : {total_tests}")
    print(f"Passed      : {passed_tests}")
    print(f"Failed      : {failed_tests}")

    if failed_tests == 0:
        print(
            "\nRESULT: ALL SQL VALIDATOR TESTS PASSED"
        )
        print(
            "The SQL safety layer is ready for "
            "SQL Agent integration."
        )
        raise SystemExit(0)

    print(
        "\nRESULT: SOME SQL VALIDATOR TESTS FAILED"
    )
    print(
        "Resolve the failed validation cases before "
        "connecting SQLCoder."
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()