"""
Standalone tests for the Northwind DatabaseManager.

Run from the project root:

    python -m tests.test_db_manager
"""

import sys
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from database.db_manager import DatabaseManager, QueryResult


def print_test_header(
    test_number: int,
    test_name: str,
) -> None:
    """
    Print a formatted test header.
    """

    print("\n" + "=" * 80)
    print(f"TEST {test_number}: {test_name}")
    print("=" * 80)


def test_database_connection(
    manager: DatabaseManager,
) -> bool:
    """
    Confirm that the database can be opened and queried.
    """

    result = manager.test_connection()

    if result is not True:
        raise AssertionError(
            "Database connection test did not return True."
        )

    print("Database connection successful.")
    return True


def test_database_integrity(
    manager: DatabaseManager,
) -> bool:
    """
    Confirm that SQLite reports a valid database file.
    """

    result = manager.run_integrity_check()

    if result.lower() != "ok":
        raise AssertionError(
            f"Expected integrity result 'ok', received '{result}'."
        )

    print(f"Integrity check result: {result}")
    return True


def test_table_discovery(
    manager: DatabaseManager,
) -> bool:
    """
    Confirm that tables can be discovered dynamically.

    This test does not enforce a specific Northwind table list because
    different Northwind SQLite distributions can contain different
    table names and additional tables.
    """

    tables = manager.get_tables()

    if not tables:
        raise AssertionError(
            "No database tables were discovered."
        )

    print(f"Discovered {len(tables)} tables:")

    for table_name in tables:
        print(f"- {table_name}")

    return True


def test_view_discovery(
    manager: DatabaseManager,
) -> bool:
    """
    Confirm that database views can be retrieved.

    An empty view list is valid because some SQLite databases do not
    define views.
    """

    views = manager.get_views()

    print(f"Discovered {len(views)} views:")

    if views:
        for view_name in views:
            print(f"- {view_name}")
    else:
        print("- No views found")

    return True


def test_column_metadata(
    manager: DatabaseManager,
) -> bool:
    """
    Confirm that column metadata can be retrieved for a real table.

    The first table is selected dynamically rather than hardcoding a
    Northwind table name.
    """

    tables = manager.get_tables()

    if not tables:
        raise AssertionError(
            "Column metadata cannot be tested because no tables exist."
        )

    selected_table = tables[0]
    columns = manager.get_columns(selected_table)

    if not columns:
        raise AssertionError(
            f"No columns were found for table '{selected_table}'."
        )

    print(f"Selected table: {selected_table}")
    print(f"Column count : {len(columns)}")

    for column in columns:
        print(
            f"- {column['name']} | "
            f"type={column['data_type']} | "
            f"nullable={column['nullable']} | "
            f"primary_key={column['primary_key']}"
        )

    required_keys = {
        "name",
        "data_type",
        "nullable",
        "default_value",
        "primary_key",
    }

    for column in columns:
        missing_keys = required_keys.difference(
            column.keys()
        )

        if missing_keys:
            raise AssertionError(
                "Column metadata is missing fields: "
                f"{sorted(missing_keys)}"
            )

    return True


def test_foreign_key_metadata(
    manager: DatabaseManager,
) -> bool:
    """
    Confirm that foreign-key metadata can be retrieved.

    The test checks every table until one containing a foreign key is
    found. A database without declared foreign keys is still handled
    correctly.
    """

    tables = manager.get_tables()
    table_with_foreign_key = None
    foreign_keys = []

    for table_name in tables:
        discovered_keys = manager.get_foreign_keys(
            table_name
        )

        if discovered_keys:
            table_with_foreign_key = table_name
            foreign_keys = discovered_keys
            break

    if table_with_foreign_key is None:
        print(
            "No declared foreign keys were found. "
            "The metadata operation completed successfully."
        )
        return True

    print(
        f"Foreign keys found in table: "
        f"{table_with_foreign_key}"
    )

    for foreign_key in foreign_keys:
        print(
            f"- {foreign_key['source_column']} -> "
            f"{foreign_key['target_table']}."
            f"{foreign_key['target_column']}"
        )

    return True


def test_real_query_execution(
    manager: DatabaseManager,
) -> bool:
    """
    Execute a real query against a dynamically selected table.
    """

    tables = manager.get_tables()

    if not tables:
        raise AssertionError(
            "Query execution cannot be tested because no tables exist."
        )

    selected_table = tables[0]
    quoted_table = (
        '"' + selected_table.replace('"', '""') + '"'
    )

    query = f"""
        SELECT *
        FROM {quoted_table}
        LIMIT 5;
    """

    result = manager.execute_read_query(query)

    if not isinstance(result, QueryResult):
        raise AssertionError(
            "execute_read_query did not return QueryResult."
        )

    if result.row_count > 5:
        raise AssertionError(
            "The query returned more than five rows."
        )

    if not result.columns:
        raise AssertionError(
            "The query result does not contain column names."
        )

    print(f"Selected table : {selected_table}")
    print(f"Columns        : {result.columns}")
    print(f"Rows returned  : {result.row_count}")
    print(f"Truncated      : {result.truncated}")
    print(
        f"Execution time : "
        f"{result.execution_time_ms} ms"
    )

    for row in result.rows:
        print(row)

    return True


def test_parameterized_query(
    manager: DatabaseManager,
) -> bool:
    """
    Confirm that parameterized SQL queries work correctly.
    """

    result = manager.execute_read_query(
        """
        SELECT
            ? AS test_text,
            ? AS test_number;
        """,
        parameters=(
            "Northwind",
            100,
        ),
    )

    if result.row_count != 1:
        raise AssertionError(
            "The parameterized query should return one row."
        )

    first_row = result.rows[0]

    if first_row["test_text"] != "Northwind":
        raise AssertionError(
            "The text parameter was not returned correctly."
        )

    if first_row["test_number"] != 100:
        raise AssertionError(
            "The numeric parameter was not returned correctly."
        )

    print(f"Parameterized result: {first_row}")
    return True


def test_maximum_row_limit(
    manager: DatabaseManager,
) -> bool:
    """
    Confirm that DatabaseManager limits returned rows.

    A temporary manager with max_rows=3 is used. The query itself
    produces more than three rows without depending on a business table.
    """

    limited_manager = DatabaseManager(
        database_path=manager.database_path,
        max_rows=3,
    )

    result = limited_manager.execute_read_query(
        """
        WITH generated_numbers(value) AS (
            SELECT 1
            UNION ALL
            SELECT value + 1
            FROM generated_numbers
            WHERE value < 10
        )
        SELECT value
        FROM generated_numbers;
        """
    )

    if result.row_count != 3:
        raise AssertionError(
            "Expected DatabaseManager to return exactly "
            f"3 rows, but received {result.row_count}."
        )

    if result.truncated is not True:
        raise AssertionError(
            "Expected truncated=True when the result exceeded max_rows."
        )

    print(f"Returned rows: {result.rows}")
    print(f"Row count    : {result.row_count}")
    print(f"Truncated    : {result.truncated}")

    return True


def test_empty_query_rejection(
    manager: DatabaseManager,
) -> bool:
    """
    Confirm that an empty SQL query is rejected.
    """

    try:
        manager.execute_read_query("   ")

    except ValueError as exc:
        print(f"Expected validation error: {exc}")
        return True

    raise AssertionError(
        "DatabaseManager accepted an empty SQL query."
    )


def test_missing_table_handling(
    manager: DatabaseManager,
) -> bool:
    """
    Confirm that querying a missing table produces an error.
    """

    try:
        manager.execute_read_query(
            """
            SELECT *
            FROM table_that_does_not_exist;
            """
        )

    except RuntimeError as exc:
        print(f"Expected database error: {exc}")
        return True

    raise AssertionError(
        "Querying a missing table did not produce an error."
    )


def test_missing_table_metadata(
    manager: DatabaseManager,
) -> bool:
    """
    Confirm that metadata requests reject missing tables.
    """

    try:
        manager.get_columns(
            "table_that_does_not_exist"
        )

    except ValueError as exc:
        print(f"Expected metadata error: {exc}")
        return True

    raise AssertionError(
        "get_columns accepted a missing table."
    )


def test_write_protection(
    manager: DatabaseManager,
) -> bool:
    """
    Confirm that database modification is blocked.

    The database is opened with SQLite read-only URI mode and
    PRAGMA query_only, so a CREATE TABLE statement must fail.
    """

    try:
        manager.execute_read_query(
            """
            CREATE TABLE forbidden_test_table (
                id INTEGER PRIMARY KEY
            );
            """
        )

    except (RuntimeError, ValueError) as exc:
        print(f"Expected write-protection error: {exc}")
        return True

    raise AssertionError(
        "A database modification query was not blocked."
    )


def run_test(
    test_number: int,
    test_name: str,
    test_function: Callable[
        [DatabaseManager],
        bool,
    ],
    manager: DatabaseManager,
) -> bool:
    """
    Execute one test and print its pass/fail status.
    """

    print_test_header(
        test_number=test_number,
        test_name=test_name,
    )

    try:
        test_function(manager)

    except Exception as exc:
        print("\n[FAILED]")
        print(f"{type(exc).__name__}: {exc}")
        return False

    print("\n[PASSED]")
    return True


def main() -> None:
    """
    Run all standalone DatabaseManager tests.
    """

    print("\nDATABASE MANAGER TEST SUITE")

    try:
        manager = DatabaseManager()

    except Exception as exc:
        print("\nCould not initialize DatabaseManager.")
        print(f"{type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc

    print(f"Database: {manager.database_path}")

    tests: list[
        tuple[
            str,
            Callable[[DatabaseManager], bool],
        ]
    ] = [
        (
            "Database connection",
            test_database_connection,
        ),
        (
            "Database integrity",
            test_database_integrity,
        ),
        (
            "Table discovery",
            test_table_discovery,
        ),
        (
            "View discovery",
            test_view_discovery,
        ),
        (
            "Column metadata",
            test_column_metadata,
        ),
        (
            "Foreign-key metadata",
            test_foreign_key_metadata,
        ),
        (
            "Real query execution",
            test_real_query_execution,
        ),
        (
            "Parameterized query",
            test_parameterized_query,
        ),
        (
            "Maximum row limit",
            test_maximum_row_limit,
        ),
        (
            "Empty query rejection",
            test_empty_query_rejection,
        ),
        (
            "Missing table query handling",
            test_missing_table_handling,
        ),
        (
            "Missing table metadata handling",
            test_missing_table_metadata,
        ),
        (
            "Database write protection",
            test_write_protection,
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
            manager=manager,
        )

        if passed:
            passed_tests += 1

    total_tests = len(tests)
    failed_tests = total_tests - passed_tests

    print("\n" + "=" * 80)
    print("FINAL DATABASE MANAGER TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests : {total_tests}")
    print(f"Passed      : {passed_tests}")
    print(f"Failed      : {failed_tests}")

    if failed_tests == 0:
        print("\nRESULT: ALL DATABASE MANAGER TESTS PASSED")
        print(
            "The database foundation is ready for "
            "schema extraction."
        )
        raise SystemExit(0)

    print("\nRESULT: SOME DATABASE MANAGER TESTS FAILED")
    print(
        "Resolve the failed tests before creating "
        "the Schema Manager."
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()