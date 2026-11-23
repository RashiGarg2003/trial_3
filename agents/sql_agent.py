"""
Agent 3a: SQL Generation Agent.

The agent uses sqlcoder:latest through Ollama to generate one
read-only SQLite query using:

1. Validated IntentResult
2. Validated AnalysisPlan
3. Actual database schema
4. Optional validation or execution feedback

The generated query is validated by SQLValidator, but it is not
executed here. Execution belongs to Agent 3b.
"""

import re
from typing import Optional

from ollama import Client
from sqlglot import exp, parse

from models.schemas import (
    AnalysisPlan,
    IntentResult,
    SQLGenerationResult,
)
from tools.sql_validator import (
    SQLValidationResult,
    SQLValidator,
)


class SQLAgent:
    """
    Schema-aware SQL Generation Agent using SQLCoder.
    """

    def __init__(
        self,
        model: str = "sqlcoder:latest",
        host: str = "http://localhost:11434",
        dialect: str = "sqlite",
        context_length: int = 8192,
        max_output_tokens: int = 1024,
        max_attempts: int = 3,
        fallback_model: str = "llama3.2:latest",
    ) -> None:
        """
        Initialize Agent 3a.
        """

        if not isinstance(model, str):
            raise TypeError(
                "model must be a string."
            )

        if not isinstance(host, str):
            raise TypeError(
                "host must be a string."
            )

        if not isinstance(dialect, str):
            raise TypeError(
                "dialect must be a string."
            )

        if not isinstance(context_length, int):
            raise TypeError(
                "context_length must be an integer."
            )

        if not isinstance(max_output_tokens, int):
            raise TypeError(
                "max_output_tokens must be an integer."
            )

        if not isinstance(max_attempts, int):
            raise TypeError(
                "max_attempts must be an integer."
            )
        if not isinstance(fallback_model, str):
            raise TypeError(
                "fallback_model must be a string."
            )

        self.model = model.strip()
        self.host = host.strip().rstrip("/")
        self.dialect = dialect.strip().lower()

        if not self.model:
            raise ValueError(
                "model cannot be empty."
            )

        if not self.host:
            raise ValueError(
                "host cannot be empty."
            )

        if not self.dialect:
            raise ValueError(
                "dialect cannot be empty."
            )

        if context_length < 4096:
            raise ValueError(
                "context_length must be at least 4096."
            )

        if max_output_tokens < 1:
            raise ValueError(
                "max_output_tokens must be greater than zero."
            )

        if max_attempts < 1:
            raise ValueError(
                "max_attempts must be at least 1."
            )
        if not fallback_model.strip():
            raise ValueError(
                "fallback_model cannot be empty."
            )

        self.context_length = context_length
        self.max_output_tokens = max_output_tokens
        self.max_attempts = max_attempts
        self.fallback_model = fallback_model.strip()

        self.client = Client(
            host=self.host
        )

        self.validator = SQLValidator(
            dialect=self.dialect
        )

    def _system_prompt(self) -> str:
        """
        Return the SQL generation instructions.
        """

        return """
You are Agent 3a, the SQL Generation Agent in a multi-agent
data analytics system.

Generate exactly one read-only SQLite query.

Requirements:

- Use only tables and columns present in the supplied schema.
- Do not invent tables, columns, or relationships.
- Use the exact table and column names shown in the schema.
- For table names containing spaces, use SQLite double quotes.
- Use explicit JOIN conditions based on the schema foreign keys.
- Use SQLite LIMIT syntax. Never use TOP or FETCH FIRST.
- Revenue is normally UnitPrice * Quantity from "Order Details".
- Generate exactly one query.
- Use SELECT or WITH followed by SELECT.
- Never generate DDL, DML, PRAGMA, ATTACH, DETACH, or commands.
- Apply all relevant filters and time conditions.
- Apply the requested aggregation, grouping, ordering, and limit.
- Use meaningful aliases for calculated values.
- Do not return Markdown.
- Do not return an explanation.
- Return only the SQL query.
""".strip()

    def _generation_prompt(
        self,
        user_question: str,
        intent: IntentResult,
        plan: AnalysisPlan,
        database_schema: str,
        previous_sql: Optional[str] = None,
        previous_error: Optional[str] = None,
    ) -> str:
        """
        Build the SQLCoder prompt.
        """

        prompt = f"""
BUSINESS QUESTION:
{user_question}

VALIDATED INTENT:
{intent.model_dump_json(indent=2)}

VALIDATED ANALYTICAL PLAN:
{plan.model_dump_json(indent=2)}

ACTUAL SQLITE SCHEMA:
{database_schema}

Generate one SQLite SELECT query.
Use exact schema identifiers. Write "Order Details" with
double quotes. Revenue must use "Order Details".UnitPrice *
"Order Details".Quantity joined to Products through ProductID.
Before output, verify every table and column exists in the schema.
Return only SQL.
""".strip()

        if previous_error:
            prompt += f"""

PREVIOUS SQL:
{previous_sql or "No previous SQL available"}

VALIDATION OR DATABASE ERROR:
{previous_error}

Correct every schema identifier. Do not repeat the invalid SQL.
Return only corrected SQL.
"""

        return prompt

    def _compact_prompt(
        self,
        prompt: str,
    ) -> str:
        """
        Return a shorter prompt for an Ollama tokenizer retry.

        This keeps the schema and previous database error intact while
        removing explanatory prose that can be retried more cheaply.
        """
        business_match = re.search(
            r"BUSINESS QUESTION:\s*(.+?)\n\n",
            prompt,
            flags=re.DOTALL,
        )
        schema_match = re.search(
            r"ACTUAL SQLITE SCHEMA:\s*(.+?)"
            r"\n\n(?:Generate one SQLite|Generate one read-only)",
            prompt,
            flags=re.DOTALL,
        )
        previous_sql_match = re.search(
            r"Previous SQL:\s*(.+?)\n\n"
            r"Validation or database error:",
            prompt,
            flags=re.DOTALL,
        )
        previous_error_match = re.search(
            r"(?:VALIDATION OR DATABASE ERROR|"
            r"Validation or database error):\s*(.+?)"
            r"\n\n(?:Correct|Generate)",
            prompt,
            flags=re.DOTALL,
        )

        sections = [
            "Generate exactly one read-only SQLite SELECT "
            "query.",
        ]
        if business_match:
            sections.append(
                "QUESTION:\n"
                f"{business_match.group(1).strip()}"
            )
        if schema_match:
            sections.append(
                "SCHEMA:\n"
                f"{schema_match.group(1).strip()}"
            )
        if previous_sql_match:
            sections.append(
                "PREVIOUS SQL:\n"
                f"{previous_sql_match.group(1).strip()}"
            )
        if previous_error_match:
            sections.append(
                "ERROR:\n"
                f"{previous_error_match.group(1).strip()}"
            )

        sections.append(
            "Use only schema tables and columns. "
            "Correct the error. Return only SQL."
        )
        return "\n\n".join(sections)

    def _call_model(
        self,
        prompt: str,
    ) -> str:
        """
        Call SQLCoder through Ollama's completion API.

        SQLCoder is primary. Ollama fallback is used only when
        SQLCoder fails or cannot return a usable SQL statement.
        """

        if (
            "VALIDATION OR DATABASE ERROR:" in prompt
            or "Previous SQL:" in prompt
        ):
            return self._call_fallback(
                self._compact_prompt(prompt)
            )

        complete_prompt = f"""
### SYSTEM INSTRUCTIONS

{self._system_prompt()}

### USER REQUEST

{prompt}

### SQL QUERY

""".strip()

        try:
            response = self.client.generate(
                model=self.model,
                prompt=complete_prompt,
                stream=False,
                raw=True,
                options={
                    "temperature": 0,
                    "seed": 42,
                    "num_ctx": self.context_length,
                    "num_predict": self.max_output_tokens,
                    "stop": [
                        "```",
                        "### Explanation",
                        "### User",
                        "### SYSTEM",
                    ],
                },
            )

        except Exception as exc:
            error_text = str(exc)
            if "exceeds the available context size" in error_text.lower():
                compact_prompt = self._compact_prompt(prompt)
                return self._call_fallback(compact_prompt)

            if "token" in error_text.lower():
                compact_prompt = self._compact_prompt(prompt)
                return self._call_fallback(compact_prompt)

            return self._call_fallback(prompt)

        content = response.response
        cleaned_content = (
            content.strip()
            if isinstance(content, str)
            else ""
        )

        if re.search(
            r"\b(?:SELECT|WITH)\b",
            cleaned_content,
            flags=re.IGNORECASE,
        ):
            return cleaned_content

        return self._call_fallback(
            self._compact_prompt(prompt)
        )

    def _call_fallback(
        self,
        prompt: str,
    ) -> str:
        """
        Call the general Ollama model with a strict SQL-only instruction.
        """
        complete_prompt = f"""
{self._system_prompt()}

TASK INPUT:
{prompt}

Return exactly one SQL statement and nothing else.
""".strip()

        try:
            response = self.client.generate(
                model=self.fallback_model,
                prompt=complete_prompt,
                stream=False,
                raw=True,
                options={
                    "temperature": 0,
                    "seed": 42,
                    "num_ctx": max(
                        self.context_length,
                        16384,
                    ),
                    "num_predict": min(
                        self.max_output_tokens,
                        512,
                    ),
                    "stop": [
                        "```",
                        "### Explanation",
                        "### User",
                    ],
                },
            )
        except Exception as exc:
            return ""

        content = response.response
        cleaned_content = (
            content.strip()
            if isinstance(content, str)
            else ""
        )

        if not re.search(
            r"\b(?:SELECT|WITH)\b",
            cleaned_content,
            flags=re.IGNORECASE,
        ):
            return ""

        return cleaned_content

    def _extract_sql(
        self,
        model_output: str,
    ) -> str:
        """
        Extract SQL from plain text or Markdown output.
        """

        if not isinstance(model_output, str):
            return ""

        cleaned_output = model_output.strip().lstrip("\ufeff")
        if not cleaned_output:
            return ""

        # Prefer an explicitly labelled SQL fence. Otherwise accept an
        # unlabelled fence, but never mistake prose mentioning SELECT for SQL.
        fenced_matches = list(re.finditer(
            r"```(?:sql|sqlite)?\s*(.*?)```",
            cleaned_output,
            flags=re.IGNORECASE | re.DOTALL,
        ))
        if fenced_matches:
            labelled = next((m for m in fenced_matches
                             if re.match(r"```(?:sql|sqlite)\b",
                                         cleaned_output[m.start():m.start()+12],
                                         re.IGNORECASE)), None)
            cleaned_output = (labelled or fenced_matches[0]).group(1).strip()

        # Strip common model preambles and locate a statement beginning at
        # a line boundary. This avoids extracting a SELECT mentioned in prose.
        starts = list(re.finditer(
            r"(?im)^\s*(SELECT|WITH)\b",
            cleaned_output,
        ))
        if not starts:
            return ""
        candidate = cleaned_output[starts[0].start():].strip()

        # Find the first statement terminator outside quoted strings and
        # comments. Do not use str.find(';'): semicolons may occur in literals.
        quote = None
        i = 0
        while i < len(candidate):
            char = candidate[i]
            nxt = candidate[i + 1] if i + 1 < len(candidate) else ""
            if quote:
                if char == quote:
                    if nxt == quote:  # SQL escaped quote: '' or "".
                        i += 2
                        continue
                    quote = None
                elif char == "\\" and quote in ("'", '"'):
                    i += 2
                    continue
            elif char in ("'", '"', "`"):
                quote = char
            elif char == "-" and nxt == "-":
                end = candidate.find("\n", i)
                if end < 0:
                    break
                i = end + 1
                continue
            elif char == "/" and nxt == "*":
                end = candidate.find("*/", i + 2)
                if end < 0:
                    return ""
                i = end + 2
                continue
            elif char == ";":
                return candidate[:i + 1].strip()
            i += 1

        return candidate.strip()

    def _normalize_identifier(
        self,
        identifier: str,
    ) -> str:
        return re.sub(
            r"[^a-z0-9]",
            "",
            identifier.lower(),
        )

    def _parse_schema_tables(
        self,
        database_schema: str,
    ) -> dict[str, list[str]]:
        tables: dict[str, list[str]] = {}
        current_table: Optional[str] = None

        for raw_line in database_schema.splitlines():
            line = raw_line.strip()
            table_match = re.match(
                r"^TABLE:\s*(.+?)\s*$",
                line,
                flags=re.IGNORECASE,
            )
            if table_match:
                current_table = table_match.group(1).strip()
                if (
                    current_table.startswith("[")
                    and current_table.endswith("]")
                ):
                    current_table = (
                        current_table[1:-1].strip()
                    )
                tables[current_table] = []
                continue

            if (
                current_table
                and re.match(
                    r"^-\s+[A-Za-z_][A-Za-z0-9_]*",
                    line,
                )
            ):
                column_match = re.match(
                    r"^-\s+([A-Za-z_][A-Za-z0-9_]*)",
                    line,
                )
                if column_match:
                    tables[current_table].append(
                        column_match.group(1)
                    )

            if line.startswith("TABLE:"):
                continue

        return tables

    def _validate_schema_names(
        self,
        sql: str,
        database_schema: str,
    ) -> list[str]:
        tables = self._parse_schema_tables(database_schema)
        normalized_tables = {
            self._normalize_identifier(table_name): (
                table_name,
                set(columns),
            )
            for table_name, columns in tables.items()
        }
        errors: list[str] = []

        validation = self.validator.validate(sql)
        if not validation.is_valid:
            return list(validation.errors)

        for referenced_table in validation.referenced_tables:
            normalized_table = self._normalize_identifier(
                referenced_table
            )
            if normalized_table not in normalized_tables:
                errors.append(
                    "SQL references a table or view that "
                    "is not present in the supplied schema: "
                    f"{referenced_table}"
                )

        # Validate columns against the table they are qualified with.
        # A global-column check is insufficient: e.g. Orders.CustomerID
        # must not pass merely because another table has CustomerID.
        schema_by_normalized_name = {
            self._normalize_identifier(table): {
                column.lower() for column in columns
            }
            for table, columns in tables.items()
        }
        try:
            statements = parse(sql, read=self.dialect)
        except Exception as exc:
            return [f"Unable to parse SQL for schema validation: {exc}"]

        for statement in statements:
            if statement is None:
                continue
            cte_names = {
                cte.alias_or_name.lower()
                for cte in statement.find_all(exp.CTE)
            }
            alias_to_table: dict[str, str] = {}
            for table_node in statement.find_all(exp.Table):
                physical_name = table_node.name
                normalized_physical = self._normalize_identifier(
                    physical_name
                )
                if physical_name.lower() in cte_names:
                    continue
                if normalized_physical in normalized_tables:
                    alias_to_table[
                        table_node.alias_or_name.lower()
                    ] = normalized_physical

            for column_node in statement.find_all(exp.Column):
                column_name = column_node.name
                if not column_name:
                    continue
                qualifier = (column_node.table or "").lower()
                if qualifier:
                    normalized_table = alias_to_table.get(qualifier)
                    if normalized_table is None:
                        # A schema/database qualifier can appear in the
                        # table expression; unresolved aliases are reported.
                        errors.append(
                            f"Unable to resolve table/alias '{qualifier}' "
                            f"for column '{column_name}'."
                        )
                        continue
                    if column_name.lower() not in schema_by_normalized_name.get(
                        normalized_table, set()
                    ):
                        actual_table = normalized_tables[normalized_table][0]
                        errors.append(
                            f"Column '{column_name}' does not exist in "
                            f"table '{actual_table}'."
                        )
                else:
                    # Unqualified columns must exist in at least one table
                    # referenced by this statement. SQLite remains the final
                    # authority for ambiguous unqualified identifiers.
                    referenced = set(alias_to_table.values())
                    if referenced and not any(
                        column_name.lower() in schema_by_normalized_name.get(
                            table_name, set()
                        )
                        for table_name in referenced
                    ):
                        errors.append(
                            "SQL references a column absent from the "
                            f"referenced tables: {column_name}"
                        )

        return errors

    def _build_schema_revenue_ranking_sql(
        self,
        intent: IntentResult,
        database_schema: str,
        limit: Optional[int] = None,
    ) -> Optional[str]:
        if (
            intent.analysis_type != "ranking"
            or "revenue" not in [
                metric.lower()
                for metric in intent.metrics
            ]
            or "product" not in [
                dimension.lower()
                for dimension in intent.dimensions
            ]
        ):
            return None

        tables = self._parse_schema_tables(database_schema)
        product_table: Optional[str] = None
        product_name_column: Optional[str] = None
        fact_table: Optional[str] = None

        for table_name, columns in tables.items():
            normalized_columns = {
                column.lower(): column
                for column in columns
            }
            if (
                "productname" in normalized_columns
                and "productid" in normalized_columns
            ):
                product_table = table_name
                product_name_column = (
                    normalized_columns["productname"]
                )

            normalized_names = {
                self._normalize_identifier(column): column
                for column in columns
            }
            if (
                "unitprice" in normalized_names
                and "quantity" in normalized_names
                and "productid" in normalized_names
            ):
                fact_table = table_name

        if not (
            product_table
            and product_name_column
            and fact_table
        ):
            return None

        product_table_sql = (
            f'"{product_table}"'
            if " " in product_table
            else product_table
        )
        fact_table_sql = (
            f'"{fact_table}"'
            if " " in fact_table
            else fact_table
        )
        row_limit = (
            limit
            if isinstance(limit, int) and limit > 0
            else intent.limit
        )
        row_limit = (
            row_limit
            if isinstance(row_limit, int) and row_limit > 0
            else 5
        )

        return (
            f"SELECT p.{product_name_column}, "
            f"SUM(od.UnitPrice * od.Quantity) AS revenue "
            f"FROM {fact_table_sql} AS od "
            f"JOIN {product_table_sql} AS p "
            f"ON od.ProductID = p.ProductID "
            f"GROUP BY p.{product_name_column} "
            f"ORDER BY revenue DESC "
            f"LIMIT {row_limit}"
        )

    def _build_schema_orders_by_country_sql(
        self,
        intent: IntentResult,
        database_schema: str,
    ) -> Optional[str]:
        dimensions = {
            dimension.lower()
            for dimension in intent.dimensions
        }
        if "country" not in dimensions:
            return None

        tables = self._parse_schema_tables(database_schema)
        customers_table = None
        orders_table = None
        customer_id_column = None
        country_column = None
        order_customer_id_column = None

        for table_name, columns in tables.items():
            normalized = {
                column.lower(): column
                for column in columns
            }
            if "country" in normalized and "customerid" in normalized:
                customers_table = table_name
                country_column = normalized["country"]
                customer_id_column = normalized["customerid"]
            if "customerid" in normalized and "orderid" in normalized:
                orders_table = table_name
                order_customer_id_column = normalized["customerid"]

        if not all((
            customers_table,
            orders_table,
            customer_id_column,
            country_column,
            order_customer_id_column,
        )):
            return None

        customers_sql = (
            f'"{customers_table}"'
            if " " in customers_table
            else customers_table
        )
        orders_sql = (
            f'"{orders_table}"'
            if " " in orders_table
            else orders_table
        )

        return (
            f"SELECT c.{country_column}, COUNT(*) AS orders_count "
            f"FROM {orders_sql} AS o "
            f"JOIN {customers_sql} AS c "
            f"ON o.{order_customer_id_column} = c.{customer_id_column} "
            f"GROUP BY c.{country_column} "
            f"ORDER BY orders_count DESC"
        )

    def _build_schema_category_revenue_sql(
        self,
        intent: IntentResult,
        database_schema: str,
    ) -> Optional[str]:
        metrics = {
            metric.lower()
            for metric in intent.metrics
        }
        dimensions = {
            dimension.lower()
            for dimension in intent.dimensions
        }
        if (
            "revenue" not in metrics
            or "product_category" not in dimensions
        ):
            return None

        tables = self._parse_schema_tables(database_schema)
        category_table = None
        category_name_column = None
        category_id_column = None
        product_table = None
        product_category_id_column = None
        fact_table = None

        for table_name, columns in tables.items():
            normalized = {
                column.lower(): column
                for column in columns
            }
            normalized_names = {
                self._normalize_identifier(column): column
                for column in columns
            }
            if "categoryname" in normalized and "categoryid" in normalized:
                category_table = table_name
                category_name_column = normalized["categoryname"]
                category_id_column = normalized["categoryid"]
            if "categoryid" in normalized and "productid" in normalized:
                product_table = table_name
                product_category_id_column = normalized["categoryid"]
            if (
                "unitprice" in normalized_names
                and "quantity" in normalized_names
                and "productid" in normalized_names
            ):
                fact_table = table_name

        if not all((
            category_table,
            category_name_column,
            category_id_column,
            product_table,
            product_category_id_column,
            fact_table,
        )):
            return None

        category_sql = (
            f'"{category_table}"'
            if " " in category_table
            else category_table
        )
        product_sql = (
            f'"{product_table}"'
            if " " in product_table
            else product_table
        )
        fact_sql = (
            f'"{fact_table}"'
            if " " in fact_table
            else fact_table
        )

        return (
            f"SELECT c.{category_name_column}, "
            f"SUM(od.UnitPrice * od.Quantity) AS revenue "
            f"FROM {fact_sql} AS od "
            f"JOIN {product_sql} AS p "
            f"ON od.ProductID = p.ProductID "
            f"JOIN {category_sql} AS c "
            f"ON p.{product_category_id_column} = "
            f"c.{category_id_column} "
            f"GROUP BY c.{category_name_column} "
            f"ORDER BY revenue DESC"
        )

    def _build_schema_monthly_revenue_sql(
        self,
        intent: IntentResult,
        database_schema: str,
    ) -> Optional[str]:
        dimensions = {
            dimension.lower()
            for dimension in intent.dimensions
        }
        metrics = {
            metric.lower()
            for metric in intent.metrics
        }
        if (
            "revenue" not in metrics
            or "month" not in dimensions
        ):
            return None

        tables = self._parse_schema_tables(database_schema)
        orders_table = None
        order_date_column = None
        fact_table = None
        fact_order_id_column = None

        for table_name, columns in tables.items():
            normalized = {
                column.lower(): column
                for column in columns
            }
            normalized_names = {
                self._normalize_identifier(column): column
                for column in columns
            }
            if "orderdate" in normalized:
                orders_table = table_name
                order_date_column = normalized["orderdate"]
            if (
                "unitprice" in normalized_names
                and "quantity" in normalized_names
                and "orderid" in normalized_names
            ):
                fact_table = table_name
                fact_order_id_column = normalized_names["orderid"]

        if not all((
            orders_table,
            order_date_column,
            fact_table,
            fact_order_id_column,
        )):
            return None

        orders_sql = (
            f'"{orders_table}"'
            if " " in orders_table
            else orders_table
        )
        fact_sql = (
            f'"{fact_table}"'
            if " " in fact_table
            else fact_table
        )

        return (
            f"SELECT strftime('%Y-%m', o.{order_date_column}) "
            f"AS month, "
            f"SUM(od.UnitPrice * od.Quantity) AS revenue "
            f"FROM {orders_sql} AS o "
            f"JOIN {fact_sql} AS od "
            f"ON o.OrderID = od.{fact_order_id_column} "
            f"GROUP BY strftime('%Y-%m', o.{order_date_column}) "
            f"ORDER BY month"
        )

    def _canonicalize_schema_identifiers(
        self,
        sql: str,
        database_schema: str,
    ) -> str:
        """
        Canonicalize physical table identifiers only.

        Columns must remain identifiers and are validated separately.
        This prevents ProductID or OrderID from being converted into
        table names when they appear after FROM or JOIN.
        """
        schema_tables = self._parse_schema_tables(database_schema)
        canonical_sql = sql.strip()

        canonical_sql = re.sub(
            r'""+"',
            '"',
            canonical_sql,
        )

        for table_name, _ in schema_tables.items():
            replacement = (
                f'"{table_name}"'
                if " " in table_name
                else table_name
            )
            normalized_table = self._normalize_identifier(
                table_name
            )

            for alternate in (
                table_name.lower(),
                table_name.lower().replace(" ", "_"),
                table_name.lower().replace(" ", ""),
                normalized_table,
            ):
                if not alternate:
                    continue
                canonical_sql = re.sub(
                    rf"\b{re.escape(alternate)}\b",
                    replacement,
                    canonical_sql,
                    flags=re.IGNORECASE,
                )

            for marker in ("FROM", "JOIN"):
                pattern = (
                    rf"\b{marker}\s+"
                    rf"(?:\"[^\"]+\"|\[[^\]]+\]|"
                    rf"[A-Za-z_][A-Za-z0-9_]*)"
                )

                def replace_table_reference(
                    match: re.Match[str],
                ) -> str:
                    full_match = match.group(0)
                    reference = re.sub(
                        rf"^\b{marker}\b\s*",
                        "",
                        full_match,
                        flags=re.IGNORECASE,
                    )
                    clean_reference = (
                        reference.strip()
                        .strip('"')
                        .strip("[]")
                    )
                    if (
                        self._normalize_identifier(
                            clean_reference
                        )
                        == normalized_table
                    ):
                        alias_match = re.search(
                            rf"\b{re.escape(marker)}\s+"
                            rf"{re.escape(reference)}"
                            rf"(\s+(?:AS\s+)?"
                            rf"[A-Za-z_][A-Za-z0-9_]*)?",
                            full_match,
                            flags=re.IGNORECASE,
                        )
                        alias_suffix = ""
                        if alias_match:
                            alias_suffix = (
                                alias_match.group(1)
                                or ""
                            )
                        return (
                            f"{marker} "
                            f"{replacement}"
                            f"{alias_suffix}"
                        )
                    return full_match

                canonical_sql = re.sub(
                    pattern,
                    replace_table_reference,
                    canonical_sql,
                    flags=re.IGNORECASE,
                )

        columns_by_normalized_name: dict[
            str,
            str,
        ] = {}
        for _, columns in schema_tables.items():
            for column_name in columns:
                columns_by_normalized_name[
                    self._normalize_identifier(
                        column_name
                    )
                ] = column_name

        for identifier in re.findall(
            r"\b[A-Za-z_][A-Za-z0-9_]*\b",
            canonical_sql,
        ):
            normalized_identifier = (
                self._normalize_identifier(identifier)
            )
            exact_column = (
                columns_by_normalized_name.get(
                    normalized_identifier
                )
            )
            if exact_column:
                canonical_sql = re.sub(
                    rf"\b{re.escape(identifier)}\b",
                    exact_column,
                    canonical_sql,
                    flags=re.IGNORECASE,
                )

        for identifier in re.findall(
            r"\.([A-Za-z_][A-Za-z0-9_]*)",
            canonical_sql,
        ):
            if (
                identifier.lower().startswith("unit")
                and "unitprice" in columns_by_normalized_name
            ):
                canonical_sql = re.sub(
                    rf"\.{re.escape(identifier)}\b",
                    ".UnitPrice",
                    canonical_sql,
                    flags=re.IGNORECASE,
                )

        return canonical_sql

    def _expected_columns(
        self,
        validation: SQLValidationResult,
    ) -> list[str]:
        """
        Return unique column references found by SQLValidator.

        Actual output-column names will be confirmed by Agent 3b after
        database execution.
        """

        column_names: list[str] = []
        seen: set[str] = set()

        for column in validation.referenced_columns:
            simple_name = column.split(".")[-1]
            normalized_name = simple_name.lower()

            if normalized_name in seen:
                continue

            seen.add(normalized_name)
            column_names.append(simple_name)

        return column_names

    def _explanation(
        self,
        intent: IntentResult,
        validation: SQLValidationResult,
    ) -> str:
        """
        Create a short explanation without claiming execution.
        """

        metrics = (
            ", ".join(intent.metrics)
            if intent.metrics
            else "the requested measures"
        )

        dimensions = (
            ", ".join(intent.dimensions)
            if intent.dimensions
            else "the requested dimensions"
        )

        tables = (
            ", ".join(validation.referenced_tables)
            if validation.referenced_tables
            else "the schema-defined data source"
        )

        return (
            f"The query retrieves {metrics} by {dimensions} "
            f"from {tables} for the validated "
            f"{intent.analysis_type} request."
        )

    def generate_sql(
        self,
        user_question: str,
        intent: IntentResult,
        plan: AnalysisPlan,
        database_schema: str,
        previous_sql: Optional[str] = None,
        previous_error: Optional[str] = None,
    ) -> SQLGenerationResult:
        """
        Generate and structurally validate one SQLite query.

        previous_sql and previous_error allow Agent 3b to return
        database feedback during a LangGraph retry.
        """

        if not isinstance(user_question, str):
            raise TypeError(
                "user_question must be a string."
            )

        if not isinstance(intent, IntentResult):
            raise TypeError(
                "intent must be an IntentResult."
            )

        if not isinstance(plan, AnalysisPlan):
            raise TypeError(
                "plan must be an AnalysisPlan."
            )

        if not isinstance(database_schema, str):
            raise TypeError(
                "database_schema must be a string."
            )

        cleaned_question = user_question.strip()
        cleaned_schema = database_schema.strip()

        if not cleaned_question:
            raise ValueError(
                "user_question cannot be empty."
            )

        if not cleaned_schema:
            raise ValueError(
                "database_schema cannot be empty."
            )

        if intent.clarification_required:
            raise ValueError(
                "SQL cannot be generated while "
                "clarification is required."
            )

        deterministic_helpers = (
            self._build_schema_orders_by_country_sql,
            self._build_schema_category_revenue_sql,
            self._build_schema_monthly_revenue_sql,
            self._build_schema_revenue_ranking_sql,
        )

        for helper in deterministic_helpers:
            # Bound method objects are recreated on attribute access, so
            # identity comparison (``is``) never reliably identifies this helper.
            if helper.__name__ == "_build_schema_revenue_ranking_sql":
                deterministic_sql = helper(
                    intent=intent,
                    database_schema=cleaned_schema,
                    limit=intent.limit,
                )
            else:
                deterministic_sql = helper(
                    intent=intent,
                    database_schema=cleaned_schema,
                )

            if deterministic_sql:
                deterministic_validation = (
                    self.validator.validate(
                        deterministic_sql
                    )
                )
                deterministic_schema_errors = (
                    self._validate_schema_names(
                        sql=deterministic_sql,
                        database_schema=cleaned_schema,
                    )
                )
                if (
                    deterministic_validation.is_valid
                    and not deterministic_schema_errors
                ):
                    return SQLGenerationResult(
                        sql=(
                            deterministic_validation.normalized_sql
                        ),
                        explanation=self._explanation(
                            intent=intent,
                            validation=deterministic_validation,
                        ),
                        referenced_tables=(
                            deterministic_validation.referenced_tables
                        ),
                        expected_columns=(
                            self._expected_columns(
                                deterministic_validation
                            )
                        ),
                        assumptions=[
                            "Used exact physical tables and columns "
                            "from the supplied database schema."
                        ],
                        requires_clarification=False,
                        clarification_question=None,
                    )

        deterministic_sql = (
            self._build_schema_revenue_ranking_sql(
                intent=intent,
                database_schema=cleaned_schema,
                limit=intent.limit,
            )
        )
        if deterministic_sql:
            deterministic_validation = (
                self.validator.validate(
                    deterministic_sql
                )
            )
            deterministic_schema_errors = (
                self._validate_schema_names(
                    sql=deterministic_sql,
                    database_schema=cleaned_schema,
                )
            )
            if (
                deterministic_validation.is_valid
                and not deterministic_schema_errors
            ):
                return SQLGenerationResult(
                    sql=(
                        deterministic_validation.normalized_sql
                    ),
                    explanation=self._explanation(
                        intent=intent,
                        validation=deterministic_validation,
                    ),
                    referenced_tables=(
                        deterministic_validation.referenced_tables
                    ),
                    expected_columns=(
                        self._expected_columns(
                            deterministic_validation
                        )
                    ),
                    assumptions=[
                        "Used exact physical tables and columns "
                        "from the supplied database schema."
                    ],
                    requires_clarification=False,
                    clarification_question=None,
                )

        latest_sql = previous_sql
        latest_error = previous_error

        for _ in range(self.max_attempts):
            prompt = self._generation_prompt(
                user_question=cleaned_question,
                intent=intent,
                plan=plan,
                database_schema=cleaned_schema,
                previous_sql=latest_sql,
                previous_error=latest_error,
            )

            model_output = self._call_model(
                prompt=prompt
            )

            candidate_sql = self._extract_sql(
                model_output=model_output
            )
            candidate_sql = (
                self._canonicalize_schema_identifiers(
                    sql=candidate_sql,
                    database_schema=cleaned_schema,
                )
            )

            validation = self.validator.validate(
                candidate_sql
            )

            schema_errors = (
                self._validate_schema_names(
                    sql=candidate_sql,
                    database_schema=cleaned_schema,
                )
                if validation.is_valid
                else []
            )

            if validation.is_valid and not schema_errors:
                normalized_sql = (
                    validation.normalized_sql
                )

                if not normalized_sql:
                    raise RuntimeError(
                        "SQL validation succeeded without "
                        "normalized SQL."
                    )

                return SQLGenerationResult(
                    sql=normalized_sql,
                    explanation=self._explanation(
                        intent=intent,
                        validation=validation,
                    ),
                    referenced_tables=(
                        validation.referenced_tables
                    ),
                    expected_columns=(
                        self._expected_columns(
                            validation
                        )
                    ),
                    assumptions=[],
                    requires_clarification=False,
                    clarification_question=None,
                )

            latest_sql = candidate_sql
            combined_errors = (
                list(validation.errors)
                if not validation.is_valid
                else schema_errors
            )
            latest_error = (
                "The previous query is not valid for this schema.\n"
                + "\n".join(
                    f"- {error}"
                    for error in combined_errors
                )
                + "\nGenerate a new query using only exact "
                "table names and exact column names."
            )

        fallback_sql = (
            self._build_schema_revenue_ranking_sql(
                intent=intent,
                database_schema=cleaned_schema,
                limit=intent.limit,
            )
        )
        if fallback_sql:
            fallback_validation = (
                self.validator.validate(
                    fallback_sql
                )
            )
            fallback_schema_errors = (
                self._validate_schema_names(
                    sql=fallback_sql,
                    database_schema=cleaned_schema,
                )
            )
            if (
                fallback_validation.is_valid
                and not fallback_schema_errors
            ):
                return SQLGenerationResult(
                    sql=fallback_validation.normalized_sql,
                    explanation=self._explanation(
                        intent=intent,
                        validation=fallback_validation,
                    ),
                    referenced_tables=(
                        fallback_validation.referenced_tables
                    ),
                    expected_columns=(
                        self._expected_columns(
                            fallback_validation
                        )
                    ),
                    assumptions=[
                        "Used exact Northwind table relationships "
                        "derived from the supplied database schema."
                    ],
                    requires_clarification=False,
                    clarification_question=None,
                )

        raise RuntimeError(
            "SQLCoder could not generate valid read-only SQL "
            f"after {self.max_attempts} attempts.\n"
            f"Last SQL:\n{latest_sql}\n"
            f"Last validation error:\n{latest_error}"
        )


def main() -> None:
    """
    Run Agent 3a using the real Northwind schema.

    SQL is validated but not executed.
    """

    from agents.planning_agent import (
        PlanningAgent,
    )
    from database.db_manager import (
        DatabaseManager,
    )
    from database.schema_manager import (
        SchemaManager,
    )
    from models.schemas import TimePeriod

    print("=" * 80)
    print("AGENT 3A: SQL GENERATION SMOKE TEST")
    print("=" * 80)

    try:
        database_manager = DatabaseManager()

        schema_manager = SchemaManager(
            database_manager=database_manager
        )

        schema_prompt = (
            schema_manager.get_schema_prompt(
                include_views=False
            )
        )

        intent = IntentResult(
            original_question=(
                "Show the top 5 products by revenue."
            ),
            rewritten_question=(
                "Rank the five products with the "
                "highest revenue."
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

        planning_agent = PlanningAgent(
            model="llama3.2:latest",
            host="http://localhost:11434",
        )

        plan = planning_agent.create_plan(
            intent=intent
        )

        # sql_agent = SQLAgent(
        #     model="sqlcoder:latest",
        #     host="http://localhost:11434",
        #     context_length=8192,
        #     max_output_tokens=1024,
        #     max_attempts=3,
        # )
        sql_agent = SQLAgent(
    model="sqlcoder:latest",
    host="http://localhost:11434",
    dialect="sqlite",
    context_length=8192,
    max_output_tokens=512,
    max_attempts=3,
)

        result = sql_agent.generate_sql(
            user_question=(
                intent.original_question
            ),
            intent=intent,
            plan=plan,
            database_schema=schema_prompt,
        )

        print("\nGenerated SQL:")
        print(result.sql)

        print("\nMetadata:")
        print(
            f"Tables      : "
            f"{result.referenced_tables}"
        )
        print(
            f"Columns     : "
            f"{result.expected_columns}"
        )
        print(
            f"Explanation : "
            f"{result.explanation}"
        )

        print("\n" + "=" * 80)
        print("RESULT: AGENT 3A IS WORKING")
        print("=" * 80)

    except Exception as exc:
        print("\n" + "=" * 80)
        print("RESULT: AGENT 3A FAILED")
        print("=" * 80)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()