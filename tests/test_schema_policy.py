from database.db_manager import DatabaseManager
from security.models import UserRole
from security.schema_policy import SchemaPolicyValidator
from tools.sql_validator import SQLValidator


def test_safe_single_column_query_allowed_for_analyst():
    database_manager = DatabaseManager()
    validator = SchemaPolicyValidator(
        database_manager=database_manager,
        sql_validator=SQLValidator(dialect="sqlite"),
        maximum_result_rows=1000,
    )

    table_name = database_manager.get_tables()[0]
    column_name = str(
        database_manager.get_columns(table_name)[0]["name"]
    )

    result = validator.validate(
        sql=(
            f'SELECT "{column_name}" '
            f'FROM "{table_name}" '
            "LIMIT 5"
        ),
        role=UserRole.ANALYST,
    )

    assert result.allowed
    assert result.role == UserRole.ANALYST
