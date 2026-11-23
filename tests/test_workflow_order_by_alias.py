from database.db_manager import DatabaseManager
from security.schema_policy import SchemaPolicyValidator
from tools.sql_validator import SQLValidator


def test_join_query_with_output_alias_in_order_by_is_allowed():
    database_manager = DatabaseManager(max_rows=100)
    sql_validator = SQLValidator(dialect="sqlite")
    validator = SchemaPolicyValidator(
        database_manager=database_manager,
        sql_validator=sql_validator,
        maximum_result_rows=100,
    )

    sql = """
        SELECT
            p.ProductName,
            SUM(od.UnitPrice * od.Quantity) AS revenue
        FROM "Order Details" AS od
        JOIN Products AS p
          ON od.ProductID = p.ProductID
        GROUP BY p.ProductName
        ORDER BY revenue DESC
        LIMIT 5
    """

    result = validator.validate(sql)

    assert result.allowed
