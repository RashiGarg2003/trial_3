"""
Standalone tests for the SchemaManager.

Run from the project root:

    python -m tests.test_schema_manager
"""

import sys
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from database.db_manager import DatabaseManager
from database.schema_manager import (
    ColumnSchema,
    DatabaseSchema,
    SchemaManager,
    TableSchema,
)


def print_test_header(
    test_number: int,
    test_name: str,
) -> None:
    """
    Print a formatted test title.
    """

    print("\n" + "=" * 80)
    print(f"TEST {test_number}: {test_name}")
    print("=" * 80)


def test_schema_manager_initialization(
    schema_manager: SchemaManager,
) -> bool:
    """
    Confirm that SchemaManager initializes correctly.
    """

    if not isinstance(
        schema_manager.database_manager,
        DatabaseManager,
    ):
        raise AssertionError(
            "SchemaManager does not contain a valid "
            "DatabaseManager instance."
        )

    print("SchemaManager initialized successfully.")
    return True


def test_complete_database_schema(
    schema_manager: SchemaManager,
) -> bool:
    """
    Confirm that the full database schema can be extracted.
    """

    schema = schema_manager.get_database_schema(
        include_views=False
    )

    if not isinstance(schema, DatabaseSchema):
        raise AssertionError(
            "Expected a DatabaseSchema object."
        )

    if not schema.tables:
        raise AssertionError(
            "The extracted database schema contains no tables."
        )

    if schema.database_type != "SQLite":
        raise AssertionError(
            "Expected database type to be SQLite, "
            f"but received '{schema.database_type}'."
        )

    print(f"Database name : {schema.database_name}")
    print(f"Database type : {schema.database_type}")
    print(f"Table count   : {len(schema.tables)}")
    print(f"View count    : {len(schema.views)}")

    for table in schema.tables:
        print(
            f"- {table.name}: "
            f"{len(table.columns)} columns, "
            f"{len(table.foreign_keys)} foreign keys"
        )

    return True


def test_single_table_schema(
    schema_manager: SchemaManager,
) -> bool:
    """
    Confirm that metadata for one discovered table can be extracted.

    The first available table is selected dynamically.
    """

    available_tables = (
        schema_manager.database_manager.get_tables()
    )

    if not available_tables:
        raise AssertionError(
            "No tables are available for testing."
        )

    selected_table = available_tables[0]

    table_schema = schema_manager.get_table_schema(
        object_name=selected_table,
        object_type="table",
    )

    if not isinstance(table_schema, TableSchema):
        raise AssertionError(
            "Expected a TableSchema object."
        )

    if table_schema.name != selected_table:
        raise AssertionError(
            "The returned table name does not match "
            "the selected table."
        )

    if table_schema.object_type != "table":
        raise AssertionError(
            "Expected object_type to be 'table'."
        )

    if not table_schema.columns:
        raise AssertionError(
            f"Table '{selected_table}' has no extracted columns."
        )

    print(f"Selected table : {table_schema.name}")
    print(f"Object type    : {table_schema.object_type}")
    print(f"Column count   : {len(table_schema.columns)}")
    print(
        f"Foreign keys  : "
        f"{len(table_schema.foreign_keys)}"
    )

    for column in table_schema.columns:
        print(
            f"- {column.name} | "
            f"type={column.data_type} | "
            f"nullable={column.nullable} | "
            f"primary_key={column.primary_key}"
        )

    return True


def test_column_schema_types(
    schema_manager: SchemaManager,
) -> bool:
    """
    Confirm that extracted columns use ColumnSchema objects.
    """

    available_tables = (
        schema_manager.database_manager.get_tables()
    )

    if not available_tables:
        raise AssertionError(
            "No tables are available for column testing."
        )

    table_schema = schema_manager.get_table_schema(
        object_name=available_tables[0],
        object_type="table",
    )

    for column in table_schema.columns:
        if not isinstance(column, ColumnSchema):
            raise AssertionError(
                "A column was not converted to ColumnSchema."
            )

        if not column.name.strip():
            raise AssertionError(
                "An extracted column has an empty name."
            )

        if not column.data_type.strip():
            raise AssertionError(
                f"Column '{column.name}' has an empty data type."
            )

    print(
        f"Validated {len(table_schema.columns)} "
        "ColumnSchema objects."
    )

    return True


def test_selected_table_schema(
    schema_manager: SchemaManager,
) -> bool:
    """
    Confirm that schema extraction can be restricted to selected tables.
    """

    available_tables = (
        schema_manager.database_manager.get_tables()
    )

    if not available_tables:
        raise AssertionError(
            "No tables are available for selection testing."
        )

    selection_size = min(
        2,
        len(available_tables),
    )

    selected_tables = available_tables[:selection_size]

    schema = schema_manager.get_database_schema(
        include_views=False,
        selected_tables=selected_tables,
    )

    returned_names = [
        table.name
        for table in schema.tables
    ]

    if returned_names != selected_tables:
        raise AssertionError(
            "Selected table extraction returned unexpected tables.\n"
            f"Expected: {selected_tables}\n"
            f"Received: {returned_names}"
        )

    print(f"Selected tables: {selected_tables}")
    print(f"Returned tables: {returned_names}")

    return True


def test_duplicate_table_selection(
    schema_manager: SchemaManager,
) -> bool:
    """
    Confirm that duplicate selected table names are removed.
    """

    available_tables = (
        schema_manager.database_manager.get_tables()
    )

    if not available_tables:
        raise AssertionError(
            "No tables are available for duplicate testing."
        )

    selected_table = available_tables[0]

    schema = schema_manager.get_database_schema(
        selected_tables=[
            selected_table,
            selected_table,
        ]
    )

    returned_names = [
        table.name
        for table in schema.tables
    ]

    if returned_names != [selected_table]:
        raise AssertionError(
            "Duplicate table selection was not removed.\n"
            f"Received: {returned_names}"
        )

    print(f"Returned tables: {returned_names}")
    return True


def test_schema_dictionary_conversion(
    schema_manager: SchemaManager,
) -> bool:
    """
    Confirm that structured schema objects can be serialized.
    """

    available_tables = (
        schema_manager.database_manager.get_tables()
    )

    selected_table = available_tables[0]

    schema = schema_manager.get_database_schema(
        selected_tables=[selected_table]
    )

    schema_dictionary = (
        schema_manager.schema_to_dict(schema)
    )

    required_keys = {
        "database_name",
        "database_type",
        "tables",
        "views",
    }

    missing_keys = required_keys.difference(
        schema_dictionary.keys()
    )

    if missing_keys:
        raise AssertionError(
            "Serialized schema is missing keys: "
            f"{sorted(missing_keys)}"
        )

    if not schema_dictionary["tables"]:
        raise AssertionError(
            "Serialized schema contains no table data."
        )

    first_table = schema_dictionary["tables"][0]

    if "columns" not in first_table:
        raise AssertionError(
            "Serialized table contains no columns field."
        )

    print("Schema dictionary keys:")
    print(list(schema_dictionary.keys()))
    print(
        f"Serialized table: "
        f"{first_table['name']}"
    )

    return True


def test_table_schema_formatting(
    schema_manager: SchemaManager,
) -> bool:
    """
    Confirm that one table can be formatted as readable text.
    """

    available_tables = (
        schema_manager.database_manager.get_tables()
    )

    selected_table = available_tables[0]

    table_schema = schema_manager.get_table_schema(
        object_name=selected_table,
        object_type="table",
    )

    formatted_schema = (
        schema_manager.format_table_schema(
            table_schema
        )
    )

    if selected_table not in formatted_schema:
        raise AssertionError(
            "Formatted schema does not contain "
            "the selected table name."
        )

    if "COLUMNS:" not in formatted_schema:
        raise AssertionError(
            "Formatted schema does not contain "
            "the COLUMNS section."
        )

    for column in table_schema.columns:
        if column.name not in formatted_schema:
            raise AssertionError(
                "Formatted schema is missing column: "
                f"{column.name}"
            )

    print(formatted_schema)
    return True


def test_complete_schema_formatting(
    schema_manager: SchemaManager,
) -> bool:
    """
    Confirm that the complete database schema can be formatted.
    """

    available_tables = (
        schema_manager.database_manager.get_tables()
    )

    selected_tables = available_tables[:2]

    schema = schema_manager.get_database_schema(
        selected_tables=selected_tables
    )

    formatted_schema = (
        schema_manager.format_database_schema(
            schema
        )
    )

    required_text = [
        "DATABASE:",
        "DATABASE TYPE:",
        "TABLE COUNT:",
        "VIEW COUNT:",
    ]

    for text in required_text:
        if text not in formatted_schema:
            raise AssertionError(
                f"Formatted schema is missing '{text}'."
            )

    for table_name in selected_tables:
        if table_name not in formatted_schema:
            raise AssertionError(
                "Formatted schema is missing table: "
                f"{table_name}"
            )

    print(formatted_schema)
    return True


def test_prompt_ready_schema(
    schema_manager: SchemaManager,
) -> bool:
    """
    Confirm that a prompt-ready schema can be generated.

    This text will later be supplied to SQLCoder.
    """

    available_tables = (
        schema_manager.database_manager.get_tables()
    )

    selected_tables = available_tables[:2]

    prompt = schema_manager.get_schema_prompt(
        include_views=False,
        selected_tables=selected_tables,
    )

    if not prompt.strip():
        raise AssertionError(
            "The schema prompt is empty."
        )

    required_instructions = [
        "SQLite",
        "Do not invent tables",
        "Do not invent",
    ]

    for instruction in required_instructions:
        if instruction not in prompt:
            raise AssertionError(
                "Schema prompt is missing required text: "
                f"{instruction}"
            )

    for table_name in selected_tables:
        if table_name not in prompt:
            raise AssertionError(
                "Schema prompt is missing selected table: "
                f"{table_name}"
            )

    print(prompt)
    return True


def test_view_schema_extraction(
    schema_manager: SchemaManager,
) -> bool:
    """
    Confirm that views are included only when requested.

    An empty list is valid when the database has no views.
    """

    available_views = (
        schema_manager.database_manager.get_views()
    )

    schema_without_views = (
        schema_manager.get_database_schema(
            include_views=False
        )
    )

    if schema_without_views.views:
        raise AssertionError(
            "Views were included when include_views=False."
        )

    schema_with_views = (
        schema_manager.get_database_schema(
            include_views=True
        )
    )

    returned_view_names = [
        view.name
        for view in schema_with_views.views
    ]

    if returned_view_names != available_views:
        raise AssertionError(
            "Extracted view names do not match database metadata.\n"
            f"Expected: {available_views}\n"
            f"Received: {returned_view_names}"
        )

    print(f"Available views: {len(available_views)}")

    for view_name in returned_view_names:
        print(f"- {view_name}")

    return True


def test_missing_table_rejection(
    schema_manager: SchemaManager,
) -> bool:
    """
    Confirm that requesting a missing table raises an error.
    """

    try:
        schema_manager.get_table_schema(
            object_name="table_that_does_not_exist",
            object_type="table",
        )

    except ValueError as exc:
        print(f"Expected validation error: {exc}")
        return True

    raise AssertionError(
        "SchemaManager accepted a missing table."
    )


def test_missing_selected_table_rejection(
    schema_manager: SchemaManager,
) -> bool:
    """
    Confirm that selected_tables rejects unknown tables.
    """

    try:
        schema_manager.get_database_schema(
            selected_tables=[
                "table_that_does_not_exist"
            ]
        )

    except ValueError as exc:
        print(f"Expected validation error: {exc}")
        return True

    raise AssertionError(
        "SchemaManager accepted an unknown selected table."
    )


def test_empty_object_name_rejection(
    schema_manager: SchemaManager,
) -> bool:
    """
    Confirm that an empty object name is rejected.
    """

    try:
        schema_manager.get_table_schema(
            object_name="   ",
            object_type="table",
        )

    except ValueError as exc:
        print(f"Expected validation error: {exc}")
        return True

    raise AssertionError(
        "SchemaManager accepted an empty object name."
    )


def test_invalid_object_type_rejection(
    schema_manager: SchemaManager,
) -> bool:
    """
    Confirm that unsupported object types are rejected.
    """

    available_tables = (
        schema_manager.database_manager.get_tables()
    )

    selected_table = available_tables[0]

    try:
        schema_manager.get_table_schema(
            object_name=selected_table,
            object_type="procedure",
        )

    except ValueError as exc:
        print(f"Expected validation error: {exc}")
        return True

    raise AssertionError(
        "SchemaManager accepted an invalid object type."
    )


def test_invalid_schema_conversion(
    schema_manager: SchemaManager,
) -> bool:
    """
    Confirm that schema_to_dict rejects unsupported objects.
    """

    try:
        schema_manager.schema_to_dict(
            {"invalid": "schema"}
        )

    except TypeError as exc:
        print(f"Expected type error: {exc}")
        return True

    raise AssertionError(
        "schema_to_dict accepted an invalid object."
    )


def run_test(
    test_number: int,
    test_name: str,
    test_function: Callable[
        [SchemaManager],
        bool,
    ],
    schema_manager: SchemaManager,
) -> bool:
    """
    Execute one test and print pass or fail.
    """

    print_test_header(
        test_number=test_number,
        test_name=test_name,
    )

    try:
        test_function(schema_manager)

    except Exception as exc:
        print("\n[FAILED]")
        print(f"{type(exc).__name__}: {exc}")
        return False

    print("\n[PASSED]")
    return True


def main() -> None:
    """
    Run all standalone SchemaManager tests.
    """

    print("\nSCHEMA MANAGER TEST SUITE")

    try:
        database_manager = DatabaseManager()

        schema_manager = SchemaManager(
            database_manager=database_manager
        )

    except Exception as exc:
        print("\nCould not initialize SchemaManager.")
        print(f"{type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc

    print(
        f"Database: "
        f"{database_manager.database_path}"
    )

    tests: list[
        tuple[
            str,
            Callable[[SchemaManager], bool],
        ]
    ] = [
        (
            "Schema Manager initialization",
            test_schema_manager_initialization,
        ),
        (
            "Complete database schema extraction",
            test_complete_database_schema,
        ),
        (
            "Single table schema extraction",
            test_single_table_schema,
        ),
        (
            "Column schema object validation",
            test_column_schema_types,
        ),
        (
            "Selected-table schema extraction",
            test_selected_table_schema,
        ),
        (
            "Duplicate table selection",
            test_duplicate_table_selection,
        ),
        (
            "Schema dictionary conversion",
            test_schema_dictionary_conversion,
        ),
        (
            "Table schema formatting",
            test_table_schema_formatting,
        ),
        (
            "Complete schema formatting",
            test_complete_schema_formatting,
        ),
        (
            "Prompt-ready schema generation",
            test_prompt_ready_schema,
        ),
        (
            "View schema extraction",
            test_view_schema_extraction,
        ),
        (
            "Missing table rejection",
            test_missing_table_rejection,
        ),
        (
            "Missing selected table rejection",
            test_missing_selected_table_rejection,
        ),
        (
            "Empty object name rejection",
            test_empty_object_name_rejection,
        ),
        (
            "Invalid object type rejection",
            test_invalid_object_type_rejection,
        ),
        (
            "Invalid schema conversion rejection",
            test_invalid_schema_conversion,
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
            schema_manager=schema_manager,
        )

        if passed:
            passed_tests += 1

    total_tests = len(tests)
    failed_tests = total_tests - passed_tests

    print("\n" + "=" * 80)
    print("FINAL SCHEMA MANAGER TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests : {total_tests}")
    print(f"Passed      : {passed_tests}")
    print(f"Failed      : {failed_tests}")

    if failed_tests == 0:
        print(
            "\nRESULT: ALL SCHEMA MANAGER TESTS PASSED"
        )
        print(
            "The real database schema is ready "
            "for use by the agents."
        )
        raise SystemExit(0)

    print(
        "\nRESULT: SOME SCHEMA MANAGER TESTS FAILED"
    )
    print(
        "Resolve the failed tests before "
        "building the agents."
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()