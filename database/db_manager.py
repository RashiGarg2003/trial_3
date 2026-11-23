"""
Read-only SQLite database manager for the Northwind database.

Responsibilities:

1. Verify that the database file exists.
2. Open SQLite connections in read-only mode.
3. Return available tables and views.
4. Return table columns.
5. Execute read-only SQL queries.
6. Return query results in a structured format.
7. Prevent database modification.

This component does not use an LLM and does not generate SQL.

Run this file directly for a basic smoke test:

    python -m database.db_manager
"""

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


DEFAULT_DATABASE_PATH = (
    Path(__file__).resolve().parent / "northwind.db"
)


@dataclass
class QueryResult:
    """
    Structured result returned after executing a SQL query.
    """

    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    execution_time_ms: float
    truncated: bool
    query: str


class DatabaseManager:
    """
    Provides controlled read-only access to a SQLite database.

    The manager is independent of Northwind business rules.
    It can work with any SQLite database supplied through database_path.
    """

    def __init__(
        self,
        database_path: Optional[Path | str] = None,
        max_rows: int = 1000,
        timeout_seconds: float = 15.0,
    ) -> None:
        """
        Initialize the database manager.

        Args:
            database_path:
                Path to the SQLite database file. If omitted,
                database/northwind.db is used.

            max_rows:
                Maximum number of rows returned by one query.

            timeout_seconds:
                Maximum time SQLite waits when the database is busy.
        """

        if database_path is None:
            self.database_path = DEFAULT_DATABASE_PATH
        else:
            self.database_path = Path(database_path).resolve()

        if max_rows < 1:
            raise ValueError(
                "max_rows must be greater than zero."
            )

        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be greater than zero."
            )

        self.max_rows = max_rows
        self.timeout_seconds = timeout_seconds

        self._validate_database_file()

    def _validate_database_file(self) -> None:
        """
        Ensure that the configured database file exists and is not empty.
        """

        if not self.database_path.exists():
            raise FileNotFoundError(
                "Database file was not found.\n"
                f"Expected path: {self.database_path}"
            )

        if not self.database_path.is_file():
            raise ValueError(
                "The configured database path is not a file:\n"
                f"{self.database_path}"
            )

        if self.database_path.stat().st_size == 0:
            raise ValueError(
                "The configured database file is empty:\n"
                f"{self.database_path}"
            )

    def get_connection(self) -> sqlite3.Connection:
        """
        Create and return a read-only SQLite connection.

        The connection uses:

        - URI read-only mode
        - Dictionary-like rows
        - Foreign-key enforcement
        - SQLite query-only protection
        """

        database_uri = (
            self.database_path.as_uri() + "?mode=ro"
        )

        connection = sqlite3.connect(
            database_uri,
            uri=True,
            timeout=self.timeout_seconds,
        )

        connection.row_factory = sqlite3.Row

        connection.execute(
            "PRAGMA foreign_keys = ON;"
        )

        connection.execute(
            "PRAGMA query_only = ON;"
        )

        return connection

    def test_connection(self) -> bool:
        """
        Test whether SQLite can open and query the database.

        Returns:
            True when the connection succeeds.

        Raises:
            ConnectionError when the connection fails.
        """

        try:
            with self.get_connection() as connection:
                cursor = connection.execute(
                    "SELECT 1 AS connection_test;"
                )

                result = cursor.fetchone()

                return (
                    result is not None
                    and result["connection_test"] == 1
                )

        except sqlite3.Error as exc:
            raise ConnectionError(
                "Could not connect to the SQLite database. "
                f"Original error: {exc}"
            ) from exc

    def run_integrity_check(self) -> str:
        """
        Run SQLite's built-in integrity check.
        """

        try:
            with self.get_connection() as connection:
                cursor = connection.execute(
                    "PRAGMA integrity_check;"
                )

                row = cursor.fetchone()

                if row is None:
                    raise RuntimeError(
                        "SQLite returned no integrity-check result."
                    )

                return str(row[0])

        except sqlite3.Error as exc:
            raise RuntimeError(
                "Database integrity check failed. "
                f"Original error: {exc}"
            ) from exc

    def get_tables(self) -> list[str]:
        """
        Return all user-defined table names.

        SQLite internal tables are excluded.
        """

        query = """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name;
        """

        try:
            with self.get_connection() as connection:
                cursor = connection.execute(query)

                return [
                    str(row["name"])
                    for row in cursor.fetchall()
                ]

        except sqlite3.Error as exc:
            raise RuntimeError(
                "Could not retrieve database tables. "
                f"Original error: {exc}"
            ) from exc

    def get_views(self) -> list[str]:
        """
        Return all database view names.
        """

        query = """
        SELECT name
        FROM sqlite_master
        WHERE type = 'view'
        ORDER BY name;
        """

        try:
            with self.get_connection() as connection:
                cursor = connection.execute(query)
                return [
                str(row["name"])
                for row in cursor.fetchall()
                ]

        except sqlite3.Error as exc:
            raise RuntimeError(
            "Could not retrieve database views. "
            f"Original error: {exc}"
        ) from exc


    def _quote_identifier(
        self,
        identifier: str,
    ) -> str:
        """
        Safely quote a SQLite table, view, or column identifier.

        This method escapes embedded double quotes and surrounds
        the identifier with double quotes.
        """

        if not isinstance(identifier, str):
            raise TypeError(
                "The SQLite identifier must be a string."
            )

        cleaned_identifier = identifier.strip()

        if not cleaned_identifier:
            raise ValueError(
                "The SQLite identifier cannot be empty."
            )

        escaped_identifier = cleaned_identifier.replace(
            '"',
            '""',
        )

        return f'"{escaped_identifier}"'

    def _object_exists(
        self,
        object_name: str,
        allowed_types: tuple[str, ...],
    ) -> bool:
        """
        Check whether a table or view exists.

        Args:
            object_name:
                Table or view name.

            allowed_types:
                SQLite object types to check, such as
                ("table",) or ("table", "view").
        """

        placeholders = ", ".join(
            "?"
            for _ in allowed_types
        )

        query = f"""
            SELECT 1
            FROM sqlite_master
            WHERE name = ?
              AND type IN ({placeholders})
            LIMIT 1;
        """

        parameters = (
            object_name,
            *allowed_types,
        )

        with self.get_connection() as connection:
            cursor = connection.execute(
                query,
                parameters,
            )

            return cursor.fetchone() is not None

    def get_columns(
        self,
        table_name: str,
    ) -> list[dict[str, Any]]:
        """
        Return column metadata for a table or view.

        Each result contains:

        - name
        - data_type
        - nullable
        - default_value
        - primary_key
        """

        cleaned_table_name = table_name.strip()

        if not cleaned_table_name:
            raise ValueError(
                "table_name cannot be empty."
            )

        object_exists = self._object_exists(
            object_name=cleaned_table_name,
            allowed_types=("table", "view"),
        )

        if not object_exists:
            raise ValueError(
                "The requested table or view does not exist: "
                f"{cleaned_table_name}"
            )

        quoted_table_name = self._quote_identifier(
            cleaned_table_name
        )

        query = (
            f"PRAGMA table_info({quoted_table_name});"
        )

        try:
            with self.get_connection() as connection:
                cursor = connection.execute(query)
                column_rows = cursor.fetchall()

                return [
                    {
                        "name": row["name"],
                        "data_type": row["type"],
                        "nullable": not bool(
                            row["notnull"]
                        ),
                        "default_value": row["dflt_value"],
                        "primary_key": bool(row["pk"]),
                    }
                    for row in column_rows
                ]

        except sqlite3.Error as exc:
            raise RuntimeError(
                "Could not retrieve column information for "
                f"'{cleaned_table_name}'. "
                f"Original error: {exc}"
            ) from exc

    def get_foreign_keys(
        self,
        table_name: str,
    ) -> list[dict[str, Any]]:
        """
        Return foreign-key relationships for one table.
        """

        cleaned_table_name = table_name.strip()

        if not cleaned_table_name:
            raise ValueError(
                "table_name cannot be empty."
            )

        table_exists = self._object_exists(
            object_name=cleaned_table_name,
            allowed_types=("table",),
        )

        if not table_exists:
            raise ValueError(
                "The requested table does not exist: "
                f"{cleaned_table_name}"
            )

        quoted_table_name = self._quote_identifier(
            cleaned_table_name
        )

        query = (
            f"PRAGMA foreign_key_list({quoted_table_name});"
        )

        try:
            with self.get_connection() as connection:
                cursor = connection.execute(query)
                foreign_key_rows = cursor.fetchall()

                return [
                    {
                        "source_column": row["from"],
                        "target_table": row["table"],
                        "target_column": row["to"],
                        "on_update": row["on_update"],
                        "on_delete": row["on_delete"],
                    }
                    for row in foreign_key_rows
                ]

        except sqlite3.Error as exc:
            raise RuntimeError(
                "Could not retrieve foreign keys for "
                f"'{cleaned_table_name}'. "
                f"Original error: {exc}"
            ) from exc

    def execute_read_query(
        self,
        query: str,
        parameters: Optional[
            tuple[Any, ...] | dict[str, Any]
        ] = None,
    ) -> QueryResult:
        """
        Execute a read-only SQL query.

        The database itself is opened in read-only and query-only mode.
        SQLite also rejects multiple statements passed to execute().

        Args:
            query:
                SQL query to execute.

            parameters:
                Optional positional or named SQL parameters.

        Returns:
            QueryResult containing columns, rows, row count,
            execution time, and truncation status.
        """

        cleaned_query = query.strip()

        if not cleaned_query:
            raise ValueError(
                "The SQL query cannot be empty."
            )

        if parameters is None:
            parameters = ()

        start_time = time.perf_counter()

        try:
            with self.get_connection() as connection:
                cursor = connection.execute(
                    cleaned_query,
                    parameters,
                )

                if cursor.description is None:
                    raise ValueError(
                        "The query did not return a result set. "
                        "Only read-only analytical queries are allowed."
                    )

                columns = [
                    column[0]
                    for column in cursor.description
                ]

                fetched_rows = cursor.fetchmany(
                    self.max_rows + 1
                )

                truncated = (
                    len(fetched_rows) > self.max_rows
                )

                if truncated:
                    fetched_rows = fetched_rows[
                        :self.max_rows
                    ]

                rows = [
                    dict(row)
                    for row in fetched_rows
                ]

        except sqlite3.Error as exc:
            raise RuntimeError(
                "SQL query execution failed.\n"
                f"Query: {cleaned_query}\n"
                f"SQLite error: {exc}"
            ) from exc

        execution_time_ms = (
            time.perf_counter() - start_time
        ) * 1000

        return QueryResult(
            columns=columns,
            rows=rows,
            row_count=len(rows),
            execution_time_ms=round(
                execution_time_ms,
                3,
            ),
            truncated=truncated,
            query=cleaned_query,
        )


    def main() -> None:
        """
        Run a simple smoke test for the database manager.

        A dedicated test file will be created next.
        """

        print("=" * 72)
        print("DATABASE MANAGER SMOKE TEST")
        print("=" * 72)

        try:
            manager = DatabaseManager()

            connection_status = manager.test_connection()
            integrity_status = manager.run_integrity_check()
            tables = manager.get_tables()
            views = manager.get_views()

            print(
                f"Database path : {manager.database_path}"
            )
            print(
                f"Connection    : "
                f"{'successful' if connection_status else 'failed'}"
            )
            print(f"Integrity     : {integrity_status}")
            print(f"Tables        : {len(tables)}")
            print(f"Views         : {len(views)}")

            print("\nAvailable tables:")

            for table_name in tables:
                print(f"- {table_name}")

            sample_result = manager.execute_read_query(
                """
                SELECT *
                FROM Customers
                LIMIT 5;
                """
            )

            print("\nSample query:")
            print(sample_result.query)

            print(
                f"\nReturned rows : "
                f"{sample_result.row_count}"
            )
            print(
                f"Execution time: "
                f"{sample_result.execution_time_ms} ms"
            )

            for row in sample_result.rows:
                print(row)

            print("\n" + "=" * 72)
            print("RESULT: DATABASE MANAGER IS WORKING")
            print("=" * 72)

        except Exception as exc:
            print("\n" + "=" * 72)
            print("RESULT: DATABASE MANAGER FAILED")
            print("=" * 72)
            print(exc)
            raise SystemExit(1) from exc


    if __name__ == "__main__":
        main()