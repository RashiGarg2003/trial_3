"""
Northwind database inspection script.

This file performs a simple read-only inspection of northwind.db.

It verifies:

1. The database file exists.
2. SQLite can open the database.
3. The database contains tables.
4. Each table contains readable records.

Run from the project root:

    python -m database.inspect_database
"""

import sqlite3
from pathlib import Path


DATABASE_PATH = Path(__file__).resolve().parent / "northwind.db"


def quote_identifier(identifier: str) -> str:
    """
    Safely quote a SQLite identifier.

    Database table names are read from SQLite metadata rather than
    supplied by an application user.
    """

    escaped_identifier = identifier.replace('"', '""')
    return f'"{escaped_identifier}"'


def verify_database_file() -> None:
    """
    Verify that the Northwind database file exists and is not empty.
    """

    if not DATABASE_PATH.exists():
        raise FileNotFoundError(
            "Northwind database was not found.\n"
            f"Expected path: {DATABASE_PATH}"
        )

    if not DATABASE_PATH.is_file():
        raise ValueError(
            f"The database path is not a file: {DATABASE_PATH}"
        )

    file_size = DATABASE_PATH.stat().st_size

    if file_size == 0:
        raise ValueError(
            "The northwind.db file is empty. "
            "Download the populated SQLite database again."
        )


def get_connection() -> sqlite3.Connection:
    """
    Open Northwind in read-only mode.

    Using read-only mode prevents this inspection script from changing
    tables or data.
    """

    database_uri = DATABASE_PATH.resolve().as_uri() + "?mode=ro"

    connection = sqlite3.connect(
        database_uri,
        uri=True,
    )

    connection.row_factory = sqlite3.Row

    return connection


def get_table_names(
    connection: sqlite3.Connection,
) -> list[str]:
    """
    Return all user-defined table names.
    """

    query = """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name NOT LIKE 'sqlite_%'
        ORDER BY name;
    """

    cursor = connection.execute(query)

    return [
        row["name"]
        for row in cursor.fetchall()
    ]


def get_row_count(
    connection: sqlite3.Connection,
    table_name: str,
) -> int:
    """
    Return the row count for one table.
    """

    quoted_table = quote_identifier(table_name)

    query = f"SELECT COUNT(*) AS row_count FROM {quoted_table};"

    cursor = connection.execute(query)
    row = cursor.fetchone()

    return int(row["row_count"])


def run_integrity_check(
    connection: sqlite3.Connection,
) -> str:
    """
    Run SQLite's built-in database integrity check.
    """

    cursor = connection.execute("PRAGMA integrity_check;")
    row = cursor.fetchone()

    return str(row[0])


def inspect_database() -> None:
    """
    Inspect and print basic Northwind database information.
    """

    verify_database_file()

    file_size_mb = DATABASE_PATH.stat().st_size / (
        1024 * 1024
    )

    print("=" * 72)
    print("NORTHWIND DATABASE INSPECTION")
    print("=" * 72)
    print(f"Database path : {DATABASE_PATH}")
    print(f"File size     : {file_size_mb:.2f} MB")

    with get_connection() as connection:
        integrity_result = run_integrity_check(connection)

        print(f"Integrity     : {integrity_result}")

        if integrity_result.lower() != "ok":
            raise RuntimeError(
                "SQLite integrity checkk failed: "
                f"{integrity_result}"
            )

        table_names = get_table_names(connection)

        if not table_names:
            raise RuntimeError(
                "The database opened successfully, "
                "but it contains no user tables."
            )

        print(f"Table count   : {len(table_names)}")
        print("AVAILABLE TABLES")
        print("-" * 72)

        for table_number, table_name in enumerate(
            table_names,
            start=1,
        ):
            row_count = get_row_count(
                connection=connection,
                table_name=table_name,
            )

            print(
                f"{table_number:>2}. "
                f"{table_name:<30} "
                f"{row_count:>8} rows"
            )

        print("\n" + "=" * 72)
        print("RESULT: NORTHWIND DATABASE IS READY")
        print("=" * 72)


def main() -> None:
    """
    Run the database inspection with readable error handling.
    """

    try:
        inspect_database()

    except Exception as exc:
        print("\n" + "=" * 72)
        print("RESULT: DATABASE INSPECTION FAILED")
        print("=" * 72)
        print(exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
