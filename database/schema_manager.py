"""
Dynamic schema manager for SQLite databases.

Responsibilities:

1. Discover tables and views using DatabaseManager.
2. Extract columns, primary keys, and foreign keys.
3. Build structured schema objects.
4. Create a readable schema description for SQL generation.
5. Avoid hardcoded Northwind table or column names.

Run from the project root:

    python -m database.schema_manager
"""

from dataclasses import asdict, dataclass
from typing import Any, Optional

from database.db_manager import DatabaseManager


@dataclass
class ColumnSchema:
    """
    Metadata for one database column.
    """

    name: str
    data_type: str
    nullable: bool
    default_value: Any
    primary_key: bool


@dataclass
class ForeignKeySchema:
    """
    Metadata for one foreign-key relationship.
    """

    source_column: str
    target_table: str
    target_column: str
    on_update: str
    on_delete: str


@dataclass
class TableSchema:
    """
    Complete metadata for one table or view.
    """

    name: str
    object_type: str
    columns: list[ColumnSchema]
    foreign_keys: list[ForeignKeySchema]


@dataclass
class DatabaseSchema:
    """
    Complete database schema.
    """

    database_name: str
    database_type: str
    tables: list[TableSchema]
    views: list[TableSchema]


class SchemaManager:
    """
    Extracts and formats database schema information.

    This class does not contain hardcoded Northwind tables, columns,
    joins, metrics, or business logic. All metadata is discovered from
    the connected SQLite database.
    """

    def __init__(
        self,
        database_manager: DatabaseManager,
    ) -> None:
        """
        Initialize the Schema Manager.

        Args:
            database_manager:
                Configured DatabaseManager instance.
        """

        if not isinstance(
            database_manager,
            DatabaseManager,
        ):
            raise TypeError(
                "database_manager must be an instance "
                "of DatabaseManager."
            )

        self.database_manager = database_manager

    def _build_column_schema(
        self,
        column_metadata: dict[str, Any],
    ) -> ColumnSchema:
        """
        Convert raw column metadata into ColumnSchema.
        """

        return ColumnSchema(
            name=str(column_metadata["name"]),
            data_type=str(
                column_metadata["data_type"] or "UNKNOWN"
            ),
            nullable=bool(
                column_metadata["nullable"]
            ),
            default_value=column_metadata[
                "default_value"
            ],
            primary_key=bool(
                column_metadata["primary_key"]
            ),
        )

    def _build_foreign_key_schema(
        self,
        foreign_key_metadata: dict[str, Any],
    ) -> ForeignKeySchema:
        """
        Convert raw foreign-key metadata into ForeignKeySchema.
        """

        return ForeignKeySchema(
            source_column=str(
                foreign_key_metadata["source_column"]
            ),
            target_table=str(
                foreign_key_metadata["target_table"]
            ),
            target_column=str(
                foreign_key_metadata["target_column"]
            ),
            on_update=str(
                foreign_key_metadata["on_update"]
                or "NO ACTION"
            ),
            on_delete=str(
                foreign_key_metadata["on_delete"]
                or "NO ACTION"
            ),
        )

    def get_table_schema(
        self,
        object_name: str,
        object_type: str = "table",
    ) -> TableSchema:
        """
        Return complete metadata for one table or view.

        Args:
            object_name:
                Existing table or view name.

            object_type:
                Either "table" or "view".
        """

        cleaned_name = object_name.strip()
        cleaned_type = object_type.strip().lower()

        if not cleaned_name:
            raise ValueError(
                "object_name cannot be empty."
            )

        if cleaned_type not in {"table", "view"}:
            raise ValueError(
                "object_type must be either "
                "'table' or 'view'."
            )

        if cleaned_type == "table":
            available_objects = (
                self.database_manager.get_tables()
            )
        else:
            available_objects = (
                self.database_manager.get_views()
            )

        if cleaned_name not in available_objects:
            raise ValueError(
                f"The {cleaned_type} "
                f"'{cleaned_name}' does not exist."
            )

        raw_columns = (
            self.database_manager.get_columns(
                cleaned_name
            )
        )

        columns = [
            self._build_column_schema(column)
            for column in raw_columns
        ]

        foreign_keys: list[ForeignKeySchema] = []

        if cleaned_type == "table":
            raw_foreign_keys = (
                self.database_manager.get_foreign_keys(
                    cleaned_name
                )
            )

            foreign_keys = [
                self._build_foreign_key_schema(
                    foreign_key
                )
                for foreign_key in raw_foreign_keys
            ]

        return TableSchema(
            name=cleaned_name,
            object_type=cleaned_type,
            columns=columns,
            foreign_keys=foreign_keys,
        )

    def get_database_schema(
        self,
        include_views: bool = False,
        selected_tables: Optional[
            list[str]
        ] = None,
    ) -> DatabaseSchema:
        """
        Return structured metadata for the complete database.

        Args:
            include_views:
                Whether database views should also be included.

            selected_tables:
                Optional table names to include. If omitted,
                every discovered table is included.

        Returns:
            DatabaseSchema containing structured table and view
            information.
        """

        available_tables = (
            self.database_manager.get_tables()
        )

        if selected_tables is None:
            table_names = available_tables
        else:
            table_names = []

            for table_name in selected_tables:
                cleaned_name = table_name.strip()

                if not cleaned_name:
                    raise ValueError(
                        "selected_tables cannot contain "
                        "empty table names."
                    )

                if cleaned_name not in available_tables:
                    raise ValueError(
                        "Selected table does not exist: "
                        f"{cleaned_name}"
                    )

                if cleaned_name not in table_names:
                    table_names.append(cleaned_name)

        table_schemas = [
            self.get_table_schema(
                object_name=table_name,
                object_type="table",
            )
            for table_name in table_names
        ]

        view_schemas: list[TableSchema] = []

        if include_views:
            view_names = (
                self.database_manager.get_views()
            )

            view_schemas = [
                self.get_table_schema(
                    object_name=view_name,
                    object_type="view",
                )
                for view_name in view_names
            ]

        return DatabaseSchema(
            database_name=(
                self.database_manager.database_path.name
            ),
            database_type="SQLite",
            tables=table_schemas,
            views=view_schemas,
        )

    def schema_to_dict(
        self,
        schema: DatabaseSchema,
    ) -> dict[str, Any]:
        """
        Convert DatabaseSchema into a serializable dictionary.
        """

        if not isinstance(schema, DatabaseSchema):
            raise TypeError(
                "schema must be a DatabaseSchema instance."
            )

        return asdict(schema)

    def format_table_schema(
        self,
        table_schema: TableSchema,
    ) -> str:
        """
        Format one table or view as readable text.

        The output is suitable for logs, debugging, or model prompts.
        """

        lines = [
            (
                f"{table_schema.object_type.upper()}: "
                f"{table_schema.name}"
            ),
            "COLUMNS:",
        ]

        for column in table_schema.columns:
            attributes: list[str] = []

            if column.primary_key:
                attributes.append("PRIMARY KEY")

            if not column.nullable:
                attributes.append("NOT NULL")

            if column.default_value is not None:
                attributes.append(
                    f"DEFAULT {column.default_value}"
                )

            attribute_text = ""

            if attributes:
                attribute_text = (
                    " [" + ", ".join(attributes) + "]"
                )

            lines.append(
                f"- {column.name} "
                f"{column.data_type}"
                f"{attribute_text}"
            )

        if table_schema.foreign_keys:
            lines.append("FOREIGN KEYS:")

            for foreign_key in (
                table_schema.foreign_keys
            ):
                lines.append(
                    f"- {foreign_key.source_column} "
                    f"REFERENCES "
                    f"{foreign_key.target_table}"
                    f"({foreign_key.target_column})"
                )

        return "\n".join(lines)

    def format_database_schema(
        self,
        schema: DatabaseSchema,
    ) -> str:
        """
        Format the complete database schema as readable text.

        This output can later be included in the SQLCoder prompt.
        """

        if not isinstance(schema, DatabaseSchema):
            raise TypeError(
                "schema must be a DatabaseSchema instance."
            )

        sections = [
            f"DATABASE: {schema.database_name}",
            f"DATABASE TYPE: {schema.database_type}",
            f"TABLE COUNT: {len(schema.tables)}",
            f"VIEW COUNT: {len(schema.views)}",
        ]

        for table_schema in schema.tables:
            sections.append(
                self.format_table_schema(
                    table_schema
                )
            )

        for view_schema in schema.views:
            sections.append(
                self.format_table_schema(
                    view_schema
                )
            )

        return "\n\n".join(sections)

    def get_schema_prompt(
        self,
        include_views: bool = False,
        selected_tables: Optional[
            list[str]
        ] = None,
    ) -> str:
        """
        Return a prompt-ready schema description.

        SQLCoder will later receive this output along with the
        analytical request. No SQL is generated in this method.
        """

        schema = self.get_database_schema(
            include_views=include_views,
            selected_tables=selected_tables,
        )

        formatted_schema = (
            self.format_database_schema(schema)
        )

        return (
            "Use only the following SQLite database "
            "schema when generating SQL.\n"
            "Do not invent tables, columns, or relationships.\n\n"
            f"{formatted_schema}"
        )


def main() -> None:
    """
    Run a standalone Schema Manager smoke test.
    """

    print("=" * 80)
    print("SCHEMA MANAGER SMOKE TEST")
    print("=" * 80)

    try:
        database_manager = DatabaseManager()
        schema_manager = SchemaManager(
            database_manager=database_manager
        )

        schema = (
            schema_manager.get_database_schema(
                include_views=False
            )
        )

        print(
            f"Database name : "
            f"{schema.database_name}"
        )
        print(
            f"Database type : "
            f"{schema.database_type}"
        )
        print(
            f"Table count   : "
            f"{len(schema.tables)}"
        )
        print(
            f"View count    : "
            f"{len(schema.views)}"
        )

        print("\n" + "-" * 80)
        print("FORMATTED DATABASE SCHEMA")
        print("-" * 80)

        formatted_schema = (
            schema_manager.format_database_schema(
                schema
            )
        )

        print(formatted_schema)

        print("\n" + "=" * 80)
        print("RESULT: SCHEMA MANAGER IS WORKING")
        print("=" * 80)

    except Exception as exc:
        print("\n" + "=" * 80)
        print("RESULT: SCHEMA MANAGER FAILED")
        print("=" * 80)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()