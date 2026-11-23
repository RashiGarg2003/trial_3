"""
Schema and data-access policy validator.

This validator runs after Agent 3a generates SQL and before Agent 3b
executes it.

Security responsibilities:

1. Parse generated SQLite SQL using SQLGlot.
2. Validate referenced tables against the actual database schema.
3. Validate referenced columns.
4. Classify sensitive columns using central security policy.
5. Enforce role-based table and column access.
6. Prevent row-level personal or confidential-data extraction.
7. Require aggregation where configured.
8. Prevent restricted and secret fields from being returned.
9. Enforce result-row limits.

This component does not:

- Generate SQL.
- Execute SQL.
- Call an LLM.
- Modify the database.
- Replace SQLValidator.
- Grant database write access.

The existing SQLValidator and read-only SQLite connection remain
mandatory security layers.
"""

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from sqlglot import exp, parse_one
from sqlglot.errors import ParseError

from database.db_manager import DatabaseManager
from security.models import (
    ColumnPolicy,
    DataClassification,
    RiskLevel,
    SchemaPolicyFinding,
    SchemaPolicyResult,
    SQLAccessType,
    SQLColumnReference,
    TablePolicy,
    UserRole,
)
from tools.sql_validator import SQLValidator


# ---------------------------------------------------------------------
# Internal SQL metadata
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ParsedSQLMetadata:
    """
    SQL metadata extracted from the SQLGlot syntax tree.
    """

    referenced_tables: list[str]
    referenced_columns: list[
        SQLColumnReference
    ]
    has_aggregation: bool
    has_group_by: bool
    result_limit: Optional[int]


# ---------------------------------------------------------------------
# Schema policy validator
# ---------------------------------------------------------------------

class SchemaPolicyValidator:
    """
    Validate SQL against schema, privacy, and role policies.

    Explicit table and column policies may be supplied. Columns without
    an explicit policy receive a generic policy based on their names
    and the configured default classification.
    """

    DEFAULT_ALLOWED_ROLES = {
        UserRole.VIEWER,
        UserRole.ANALYST,
        UserRole.ADMIN,
    }

    # These are generic security classifications, not Northwind
    # business rules.
    SECRET_COLUMN_PATTERN = re.compile(
        r"(?:^|_)("
        r"password|passwd|pwd|secret|api_?key|"
        r"access_?token|refresh_?token|jwt|"
        r"private_?key|connection_?string"
        r")(?:_|$)",
        flags=re.IGNORECASE,
    )

    RESTRICTED_COLUMN_PATTERN = re.compile(
        r"(?:^|_)("
        r"aadhaar|aadhar|pan_?number|passport|"
        r"credit_?card|debit_?card|card_?number|"
        r"cvv|bank_?account|account_?number|"
        r"upi_?id|salary|medical|health_?record"
        r")(?:_|$)",
        flags=re.IGNORECASE,
    )

    PERSONAL_COLUMN_PATTERN = re.compile(
        r"(?:^|_)("
        r"email|e_?mail|phone|mobile|address|"
        r"postal_?code|zip_?code|date_?of_?birth|"
        r"dob|personal_?id|employee_?id|"
        r"customer_?name|contact_?name"
        r")(?:_|$)",
        flags=re.IGNORECASE,
    )

    AGGREGATE_EXPRESSION_TYPES = (
        exp.Sum,
        exp.Avg,
        exp.Count,
        exp.Min,
        exp.Max,
    )

    def __init__(
        self,
        database_manager: DatabaseManager,
        sql_validator: Optional[
            SQLValidator
        ] = None,
        table_policies: Optional[
            list[TablePolicy]
        ] = None,
        default_classification: (
            DataClassification
        ) = DataClassification.INTERNAL,
        maximum_result_rows: int = 1000,
        allow_unqualified_columns: bool = True,
    ) -> None:
        """
        Initialize the schema-policy validator.

        Args:
            database_manager:
                Existing read-only database manager.

            sql_validator:
                Existing AST-based SQL safety validator.

            table_policies:
                Optional explicit table and column policies.

            default_classification:
                Classification assigned to ordinary unlisted columns.

            maximum_result_rows:
                Maximum explicit SQL LIMIT permitted.

            allow_unqualified_columns:
                Whether unqualified columns may be resolved when the
                query references exactly one table.
        """

        if not isinstance(
            database_manager,
            DatabaseManager,
        ):
            raise TypeError(
                "database_manager must be a "
                "DatabaseManager instance."
            )

        if (
            sql_validator is not None
            and not isinstance(
                sql_validator,
                SQLValidator,
            )
        ):
            raise TypeError(
                "sql_validator must be a "
                "SQLValidator instance."
            )

        if not isinstance(
            maximum_result_rows,
            int,
        ):
            raise TypeError(
                "maximum_result_rows must be an integer."
            )

        if maximum_result_rows < 1:
            raise ValueError(
                "maximum_result_rows must be greater than zero."
            )

        if not isinstance(
            allow_unqualified_columns,
            bool,
        ):
            raise TypeError(
                "allow_unqualified_columns must be a boolean."
            )

        self.database_manager = database_manager

        self.sql_validator = (
            sql_validator
            if sql_validator is not None
            else SQLValidator(
                dialect="sqlite"
            )
        )

        self.default_classification = (
            DataClassification(
                default_classification
            )
        )

        self.maximum_result_rows = (
            maximum_result_rows
        )

        self.allow_unqualified_columns = (
            allow_unqualified_columns
        )

        self.table_policies = (
            table_policies or []
        )

        self._table_policy_map = {
            self._normalize_identifier(
                policy.table_name
            ): policy
            for policy in self.table_policies
        }

    # -----------------------------------------------------------------
    # Normalization
    # -----------------------------------------------------------------

    def _normalize_identifier(
        self,
        value: str,
    ) -> str:
        """
        Normalize an SQLite identifier for policy comparison.
        """

        return " ".join(
            value.strip().lower().split()
        )

    def _normalize_role(
        self,
        role: UserRole | str,
    ) -> str:
        """
        Normalize a role value.
        """
        if isinstance(role, UserRole):
            return role.value.strip().lower()

        cleaned_role = str(role).strip().lower()
        prefix = "userrole."

        if cleaned_role.startswith(prefix):
            cleaned_role = cleaned_role[len(prefix):]

        return cleaned_role

    def _normalize_roles(
        self,
        roles: Iterable[
            UserRole | str
        ],
    ) -> set[str]:
        """
        Normalize a role collection.
        """

        return {
            self._normalize_role(role)
            for role in roles
        }

    # -----------------------------------------------------------------
    # Actual schema discovery
    # -----------------------------------------------------------------

    def _schema_tables(
        self,
    ) -> dict[str, str]:
        """
        Return normalized actual table and view names.
        """

        object_names = [
            *self.database_manager.get_tables(),
            *self.database_manager.get_views(),
        ]

        return {
            self._normalize_identifier(
                object_name
            ): object_name
            for object_name in object_names
        }

    def _schema_columns(
        self,
        table_name: str,
    ) -> dict[str, str]:
        """
        Return normalized column names for one real table.
        """

        columns = (
            self.database_manager.get_columns(
                table_name
            )
        )

        column_map: dict[str, str] = {}

        for column in columns:
            column_name = column.get(
                "name"
            )

            if not isinstance(
                column_name,
                str,
            ):
                continue

            column_map[
                self._normalize_identifier(
                    column_name
                )
            ] = column_name

        return column_map

    # -----------------------------------------------------------------
    # SQL parsing
    # -----------------------------------------------------------------

    def _extract_limit(
        self,
        expression: exp.Expression,
    ) -> Optional[int]:
        """
        Extract a literal SQL LIMIT value.
        """

        limit_expression = expression.args.get(
            "limit"
        )

        if limit_expression is None:
            return None

        limit_value = limit_expression.expression

        if isinstance(
            limit_value,
            exp.Literal,
        ) and limit_value.is_int:
            return int(
                limit_value.this
            )

        return None

    def _alias_map(
        self,
        expression: exp.Expression,
    ) -> dict[str, str]:
        """
        Map table aliases to physical table names.
        """

        aliases: dict[str, str] = {}

        for table in expression.find_all(
            exp.Table
        ):
            table_name = table.name

            if not table_name:
                continue

            alias = table.alias_or_name

            aliases[
                self._normalize_identifier(
                    alias
                )
            ] = table_name

            aliases[
                self._normalize_identifier(
                    table_name
                )
            ] = table_name

        return aliases

    def _has_ancestor(
        self,
        expression: exp.Expression,
        ancestor_types: tuple[
            type[exp.Expression],
            ...,
        ],
    ) -> bool:
        """
        Check whether a syntax node belongs to a specific SQL clause.
        """

        parent = expression.parent

        while parent is not None:
            if isinstance(
                parent,
                ancestor_types,
            ):
                return True

            parent = parent.parent

        return False

    def _column_access_types(
        self,
        column: exp.Column,
    ) -> set[SQLAccessType]:
        """
        Determine how one column is used by the query.
        """

        access_types: set[
            SQLAccessType
        ] = set()

        if self._has_ancestor(
            column,
            (exp.Where, exp.Having),
        ):
            access_types.add(
                SQLAccessType.FILTER
            )

        if self._has_ancestor(
            column,
            (exp.Group,),
        ):
            access_types.add(
                SQLAccessType.GROUP_BY
            )

        if self._has_ancestor(
            column,
            (exp.Order,),
        ):
            access_types.add(
                SQLAccessType.ORDER_BY
            )

        if self._has_ancestor(
            column,
            (exp.Join,),
        ):
            access_types.add(
                SQLAccessType.JOIN
            )

        if self._has_ancestor(
            column,
            self.AGGREGATE_EXPRESSION_TYPES,
        ):
            access_types.add(
                SQLAccessType.AGGREGATE
            )

        if not access_types:
            access_types.add(
                SQLAccessType.SELECT
            )

        return access_types

    def _is_aggregated(
        self,
        column: exp.Column,
    ) -> bool:
        """
        Return True when a column is used inside an aggregate.
        """

        return self._has_ancestor(
            column,
            self.AGGREGATE_EXPRESSION_TYPES,
        )

    def _output_alias(
        self,
        column: exp.Column,
    ) -> Optional[str]:
        """
        Return an output alias when the column belongs to one.
        """

        parent = column.parent

        while parent is not None:
            if isinstance(parent, exp.Alias):
                return parent.alias

            if isinstance(
                parent,
                (
                    exp.Select,
                    exp.Where,
                    exp.Group,
                    exp.Order,
                    exp.Join,
                ),
            ):
                break

            parent = parent.parent

        return None

    def _parse_sql(
        self,
        sql: str,
    ) -> ParsedSQLMetadata:
        """
        Parse SQL and extract policy-relevant metadata.
        """

        try:
            expression = parse_one(
                sql,
                read="sqlite",
            )

        except ParseError as exc:
            raise ValueError(
                "SQL could not be parsed for "
                f"schema policy validation: {exc}"
            ) from exc

        alias_map = self._alias_map(
            expression
        )

        referenced_tables: list[str] = []
        seen_tables: set[str] = set()

        for table in expression.find_all(
            exp.Table
        ):
            table_name = table.name

            if not table_name:
                continue

            normalized_name = (
                self._normalize_identifier(
                    table_name
                )
            )

            if normalized_name in seen_tables:
                continue

            seen_tables.add(
                normalized_name
            )

            referenced_tables.append(
                table_name
            )

        column_references: list[
            SQLColumnReference
        ] = []

        seen_references: set[
            tuple[
                Optional[str],
                str,
                tuple[str, ...],
            ]
        ] = set()

        for column in expression.find_all(
            exp.Column
        ):
            column_name = column.name

            if not column_name:
                continue

            qualifier = column.table

            table_name: Optional[str] = None

            if qualifier:
                table_name = alias_map.get(
                    self._normalize_identifier(
                        qualifier
                    ),
                    qualifier,
                )

            elif (
                self.allow_unqualified_columns
                and len(
                    referenced_tables
                )
                == 1
            ):
                table_name = (
                    referenced_tables[0]
                )

            access_types = (
                self._column_access_types(
                    column
                )
            )

            normalized_access_types = tuple(
                sorted(
                    str(access_type)
                    for access_type
                    in access_types
                )
            )

            reference_key = (
                (
                    self._normalize_identifier(
                        table_name
                    )
                    if table_name
                    else None
                ),
                self._normalize_identifier(
                    column_name
                ),
                normalized_access_types,
            )

            if reference_key in seen_references:
                continue

            seen_references.add(
                reference_key
            )

            column_references.append(
                SQLColumnReference(
                    table_name=table_name,
                    column_name=column_name,
                    access_types=access_types,
                    is_aggregated=(
                        self._is_aggregated(
                            column
                        )
                    ),
                    output_alias=(
                        self._output_alias(
                            column
                        )
                    ),
                )
            )

        has_aggregation = any(
            isinstance(
                node,
                self.AGGREGATE_EXPRESSION_TYPES,
            )
            for node in expression.walk()
        )

        has_group_by = (
            expression.args.get(
                "group"
            )
            is not None
        )

        return ParsedSQLMetadata(
            referenced_tables=(
                referenced_tables
            ),
            referenced_columns=(
                column_references
            ),
            has_aggregation=(
                has_aggregation
            ),
            has_group_by=(
                has_group_by
            ),
            result_limit=(
                self._extract_limit(
                    expression
                )
            ),
        )

    # -----------------------------------------------------------------
    # Policy creation and lookup
    # -----------------------------------------------------------------

    def _classification_for_column(
        self,
        column_name: str,
    ) -> DataClassification:
        """
        Classify an unlisted column using central semantic rules.
        """

        normalized_name = re.sub(
            r"[^a-z0-9]+",
            "_",
            column_name.lower(),
        ).strip("_")

        if self.SECRET_COLUMN_PATTERN.search(
            normalized_name
        ):
            return DataClassification.SECRET

        if self.RESTRICTED_COLUMN_PATTERN.search(
            normalized_name
        ):
            return (
                DataClassification.RESTRICTED
            )

        if self.PERSONAL_COLUMN_PATTERN.search(
            normalized_name
        ):
            return DataClassification.PERSONAL

        return self.default_classification

    def _default_column_policy(
        self,
        table_name: str,
        column_name: str,
    ) -> ColumnPolicy:
        """
        Create the generic policy for an unlisted column.
        """

        classification = (
            self._classification_for_column(
                column_name
            )
        )

        if classification in {
            DataClassification.SECRET,
            DataClassification.RESTRICTED,
        }:
            return ColumnPolicy(
                table_name=table_name,
                column_name=column_name,
                classification=classification,
                allowed_roles=set(),
                allow_selection=False,
                allow_filtering=False,
                allow_grouping=False,
                allow_ordering=False,
                allow_joining=False,
                allow_aggregation=False,
                require_aggregation=False,
                redact_output=True,
            )

        if (
            classification
            == DataClassification.PERSONAL
        ):
            return ColumnPolicy(
                table_name=table_name,
                column_name=column_name,
                classification=classification,
                allowed_roles={
                    UserRole.ANALYST,
                    UserRole.ADMIN,
                },
                allow_selection=False,
                allow_filtering=True,
                allow_grouping=False,
                allow_ordering=False,
                allow_joining=True,
                allow_aggregation=True,
                require_aggregation=False,
                redact_output=True,
            )

        return ColumnPolicy(
            table_name=table_name,
            column_name=column_name,
            classification=classification,
            allowed_roles=set(
                self.DEFAULT_ALLOWED_ROLES
            ),
            allow_selection=True,
            allow_filtering=True,
            allow_grouping=True,
            allow_ordering=True,
            allow_joining=True,
            allow_aggregation=True,
            require_aggregation=False,
            redact_output=False,
        )

    def _table_policy(
        self,
        table_name: str,
    ) -> TablePolicy:
        """
        Return explicit or generated table policy.
        """

        normalized_table = (
            self._normalize_identifier(
                table_name
            )
        )

        explicit_policy = (
            self._table_policy_map.get(
                normalized_table
            )
        )

        if explicit_policy is not None:
            return explicit_policy

        return TablePolicy(
            table_name=table_name,
            allowed_roles=set(
                self.DEFAULT_ALLOWED_ROLES
            ),
            allow_access=True,
            allow_row_level_output=True,
            require_aggregation=False,
            maximum_rows=(
                self.maximum_result_rows
            ),
            columns=[],
        )

    def _column_policy(
        self,
        table_name: str,
        column_name: str,
    ) -> ColumnPolicy:
        """
        Return explicit or generated column policy.
        """

        table_policy = self._table_policy(
            table_name
        )

        normalized_column = (
            self._normalize_identifier(
                column_name
            )
        )

        for policy in table_policy.columns:
            if (
                self._normalize_identifier(
                    policy.column_name
                )
                == normalized_column
            ):
                return policy

        return self._default_column_policy(
            table_name=table_name,
            column_name=column_name,
        )

    # -----------------------------------------------------------------
    # Finding helpers
    # -----------------------------------------------------------------

    def _finding(
        self,
        *,
        violation_type: str,
        message: str,
        severity: RiskLevel,
        table_name: Optional[str] = None,
        column_name: Optional[str] = None,
        classification: Optional[
            DataClassification
        ] = None,
    ) -> SchemaPolicyFinding:
        """
        Create one policy finding.
        """

        return SchemaPolicyFinding(
            table_name=table_name,
            column_name=column_name,
            classification=classification,
            violation_type=violation_type,
            severity=severity,
            message=message,
        )

    def _is_role_allowed(
        self,
        role: UserRole,
        allowed_roles: Iterable[
            UserRole | str
        ],
    ) -> bool:
        """
        Check role membership.
        """

        return (
            self._normalize_role(role)
            in self._normalize_roles(
                allowed_roles
            )
        )

    def _validate_column_usage(
        self,
        reference: SQLColumnReference,
        policy: ColumnPolicy,
        role: UserRole,
    ) -> list[SchemaPolicyFinding]:
        """
        Validate one SQL column against its policy.
        """

        findings: list[
            SchemaPolicyFinding
        ] = []

        classification = (
            DataClassification(
                policy.classification
            )
        )

        if not self._is_role_allowed(
            role,
            policy.allowed_roles,
        ):
            findings.append(
                self._finding(
                    table_name=(
                        reference.table_name
                    ),
                    column_name=(
                        reference.column_name
                    ),
                    classification=classification,
                    violation_type=(
                        "role_not_allowed"
                    ),
                    severity=RiskLevel.HIGH,
                    message=(
                        "The current role is not "
                        "authorized to access a "
                        "referenced column."
                    ),
                )
            )

            return findings

        usage_checks = {
            SQLAccessType.SELECT: (
                policy.allow_selection,
                "selection_not_allowed",
            ),
            SQLAccessType.FILTER: (
                policy.allow_filtering,
                "filtering_not_allowed",
            ),
            SQLAccessType.GROUP_BY: (
                policy.allow_grouping,
                "grouping_not_allowed",
            ),
            SQLAccessType.ORDER_BY: (
                policy.allow_ordering,
                "ordering_not_allowed",
            ),
            SQLAccessType.JOIN: (
                policy.allow_joining,
                "joining_not_allowed",
            ),
            SQLAccessType.AGGREGATE: (
                policy.allow_aggregation,
                "selection_not_allowed",
            ),
        }

        for access_type in (
            reference.access_types
        ):
            normalized_access = (
                SQLAccessType(access_type)
            )

            allowed, violation = (
                usage_checks[
                    normalized_access
                ]
            )

            if allowed:
                continue

            findings.append(
                self._finding(
                    table_name=(
                        reference.table_name
                    ),
                    column_name=(
                        reference.column_name
                    ),
                    classification=classification,
                    violation_type=violation,
                    severity=(
                        RiskLevel.CRITICAL
                        if classification
                        in {
                            DataClassification.SECRET,
                            DataClassification.RESTRICTED,
                        }
                        else RiskLevel.HIGH
                    ),
                    message=(
                        "The generated query uses "
                        "a protected column in an "
                        "unauthorized SQL operation."
                    ),
                )
            )

        if (
            policy.require_aggregation
            and not reference.is_aggregated
        ):
            findings.append(
                self._finding(
                    table_name=(
                        reference.table_name
                    ),
                    column_name=(
                        reference.column_name
                    ),
                    classification=classification,
                    violation_type=(
                        "aggregation_required"
                    ),
                    severity=RiskLevel.HIGH,
                    message=(
                        "The protected column may only "
                        "be used in an approved aggregate."
                    ),
                )
            )

        return findings

    # -----------------------------------------------------------------
    # Public validation
    # -----------------------------------------------------------------

    def validate(
        self,
        sql: str,
        role: UserRole | str = (
            UserRole.ANALYST
        ),
    ) -> SchemaPolicyResult:
        """
        Validate generated SQL against schema and privacy policy.

        Args:
            sql:
                SQL produced by Agent 3a.

            role:
                Current authenticated user role.

        Returns:
            SchemaPolicyResult.
        """

        if not isinstance(sql, str):
            raise TypeError(
                "sql must be a string."
            )

        cleaned_sql = sql.strip()

        if not cleaned_sql:
            raise ValueError(
                "sql cannot be empty."
            )

        normalized_role = UserRole(role)

        structural_result = (
            self.sql_validator.validate(
                cleaned_sql
            )
        )

        if not structural_result.is_valid:
            findings = [
                self._finding(
                    violation_type="other",
                    severity=RiskLevel.CRITICAL,
                    message=(
                        "The generated SQL failed "
                        "structural safety validation."
                    ),
                )
            ]

            return SchemaPolicyResult(
                allowed=False,
                role=normalized_role,
                referenced_tables=[],
                referenced_columns=[],
                findings=findings,
                redacted_output_columns=[],
                requires_aggregation=False,
                decision="block",
                user_message=(
                    "The query was blocked because it "
                    "failed SQL safety validation."
                ),
            )

        metadata = self._parse_sql(
            cleaned_sql
        )

        actual_tables = self._schema_tables()

        findings: list[
            SchemaPolicyFinding
        ] = []

        redacted_columns: list[str] = []

        requires_aggregation = False

        # Validate tables.
        for table_name in (
            metadata.referenced_tables
        ):
            normalized_table = (
                self._normalize_identifier(
                    table_name
                )
            )

            if normalized_table not in actual_tables:
                findings.append(
                    self._finding(
                        table_name=table_name,
                        violation_type=(
                            "unknown_table"
                        ),
                        severity=(
                            RiskLevel.CRITICAL
                        ),
                        message=(
                            "The query references a table "
                            "that is not present in the "
                            "approved database schema."
                        ),
                    )
                )

                continue

            physical_table = actual_tables.get(
                normalized_table
            )

            if physical_table is None:
                continue

            table_policy = self._table_policy(
                physical_table
            )

            if not table_policy.allow_access:
                findings.append(
                    self._finding(
                        table_name=physical_table,
                        violation_type=(
                            "table_not_allowed"
                        ),
                        severity=(
                            RiskLevel.CRITICAL
                        ),
                        message=(
                            "Access to a referenced table "
                            "is prohibited by policy."
                        ),
                    )
                )

            elif not self._is_role_allowed(
                normalized_role,
                table_policy.allowed_roles,
            ):
                findings.append(
                    self._finding(
                        table_name=physical_table,
                        violation_type=(
                            "role_not_allowed"
                        ),
                        severity=RiskLevel.HIGH,
                        message=(
                            "The current role is not "
                            "authorized to access a "
                            "referenced table."
                        ),
                    )
                )

            if table_policy.require_aggregation:
                requires_aggregation = True

                if not metadata.has_aggregation:
                    findings.append(
                        self._finding(
                            table_name=(
                                physical_table
                            ),
                            violation_type=(
                                "aggregation_required"
                            ),
                            severity=(
                                RiskLevel.HIGH
                            ),
                            message=(
                                "The referenced table "
                                "requires aggregated access."
                            ),
                        )
                    )

            if (
                table_policy.maximum_rows
                is not None
                and metadata.result_limit
                is not None
                and metadata.result_limit
                > table_policy.maximum_rows
            ):
                findings.append(
                    self._finding(
                        table_name=physical_table,
                        violation_type=(
                            "result_limit_exceeded"
                        ),
                        severity=RiskLevel.HIGH,
                        message=(
                            "The requested row limit "
                            "exceeds table policy."
                        ),
                    )
                )

        # Validate columns.
        for reference in (
            metadata.referenced_columns
        ):
            table_name = reference.table_name

            if table_name is None:
                reference_access_types = {
                    SQLAccessType(value)
                    for value in reference.access_types
                }
                if (
                    SQLAccessType.ORDER_BY
                    in reference_access_types
                ):
                    continue

                findings.append(
                    self._finding(
                        column_name=(
                            reference.column_name
                        ),
                        violation_type=(
                            "unknown_column"
                        ),
                        severity=RiskLevel.HIGH,
                        message=(
                            "The column could not be "
                            "resolved to an approved table."
                        ),
                    )
                )

                continue

            normalized_table = (
                self._normalize_identifier(
                    table_name
                )
            )

            physical_table = (
                actual_tables.get(
                    normalized_table
                )
            )

            if physical_table is None:
                continue

            actual_columns = (
                self._schema_columns(
                    physical_table
                )
            )

            normalized_column = (
                self._normalize_identifier(
                    reference.column_name
                )
            )

            physical_column = (
                actual_columns.get(
                    normalized_column
                )
            )

            # SQL aliases used in ORDER BY may not be physical
            # database columns. SQLite resolves those during EXPLAIN.
            if physical_column is None:
                if (
                    SQLAccessType.ORDER_BY
                    in {
                        SQLAccessType(value)
                        for value
                        in reference.access_types
                    }
                ):
                    continue

                findings.append(
                    self._finding(
                        table_name=physical_table,
                        column_name=(
                            reference.column_name
                        ),
                        violation_type=(
                            "unknown_column"
                        ),
                        severity=(
                            RiskLevel.CRITICAL
                        ),
                        message=(
                            "The query references a column "
                            "that is not present in the "
                            "approved database schema."
                        ),
                    )
                )

                continue

            policy = self._column_policy(
                table_name=physical_table,
                column_name=physical_column,
            )

            classification = (
                DataClassification(
                    policy.classification
                )
            )

            if policy.require_aggregation:
                requires_aggregation = True

            if policy.redact_output:
                output_name = (
                    reference.output_alias
                    or physical_column
                )

                if (
                    output_name
                    not in redacted_columns
                ):
                    redacted_columns.append(
                        output_name
                    )

            findings.extend(
                self._validate_column_usage(
                    reference=(
                        reference.model_copy(
                            update={
                                "table_name": (
                                    physical_table
                                ),
                                "column_name": (
                                    physical_column
                                ),
                            }
                        )
                    ),
                    policy=policy,
                    role=normalized_role,
                )
            )

            if (
                classification
                in {
                    DataClassification.SECRET,
                    DataClassification.RESTRICTED,
                }
            ):
                findings.append(
                    self._finding(
                        table_name=physical_table,
                        column_name=physical_column,
                        classification=classification,
                        violation_type=(
                            "column_not_allowed"
                        ),
                        severity=(
                            RiskLevel.CRITICAL
                        ),
                        message=(
                            "The query references a "
                            "restricted or secret column."
                        ),
                    )
                )

        if (
            metadata.result_limit
            is not None
            and metadata.result_limit
            > self.maximum_result_rows
        ):
            findings.append(
                self._finding(
                    violation_type=(
                        "result_limit_exceeded"
                    ),
                    severity=RiskLevel.HIGH,
                    message=(
                        "The requested LIMIT exceeds "
                        "the application row policy."
                    ),
                )
            )

        blocked = bool(findings)

        return SchemaPolicyResult(
            allowed=not blocked,
            role=normalized_role,
            referenced_tables=list(
                metadata.referenced_tables
            ),
            referenced_columns=list(
                metadata.referenced_columns
            ),
            findings=findings,
            redacted_output_columns=(
                redacted_columns
            ),
            requires_aggregation=(
                requires_aggregation
            ),
            decision=(
                "block"
                if blocked
                else "allow"
            ),
            user_message=(
                "The query was blocked by schema "
                "and privacy policy."
                if blocked
                else (
                    "The query passed schema, role, "
                    "and privacy-policy validation."
                )
            ),
        )


# ---------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------

def main() -> None:
    """
    Run the validator against the actual SQLite database.

    The test discovers a real table and column dynamically.
    """

    print("=" * 80)
    print("SCHEMA POLICY VALIDATOR SMOKE TEST")
    print("=" * 80)

    try:
        database_manager = DatabaseManager()

        validator = SchemaPolicyValidator(
            database_manager=(
                database_manager
            ),
            sql_validator=SQLValidator(
                dialect="sqlite"
            ),
            maximum_result_rows=1000,
        )

        tables = (
            database_manager.get_tables()
        )

        if not tables:
            raise RuntimeError(
                "No database tables were found."
            )

        table_name = tables[0]

        columns = (
            database_manager.get_columns(
                table_name
            )
        )

        if not columns:
            raise RuntimeError(
                "The selected table contains "
                "no columns."
            )

        column_name = str(
            columns[0]["name"]
        )

        quoted_table = (
            '"'
            + table_name.replace(
                '"',
                '""',
            )
            + '"'
        )

        quoted_column = (
            '"'
            + column_name.replace(
                '"',
                '""',
            )
            + '"'
        )

        safe_sql = f"""
            SELECT {quoted_column}
            FROM {quoted_table}
            LIMIT 5
        """

        unsafe_sql = """
            DELETE FROM Products
        """

        safe_result = validator.validate(
            sql=safe_sql,
            role=UserRole.ANALYST,
        )

        unsafe_result = validator.validate(
            sql=unsafe_sql,
            role=UserRole.ANALYST,
        )

        print("\nSafe query:")
        print(safe_sql.strip())

        print(
            f"Allowed : "
            f"{safe_result.allowed}"
        )

        print(
            f"Message : "
            f"{safe_result.user_message}"
        )

        if safe_result.findings:
            print("Findings:")

            for finding in (
                safe_result.findings
            ):
                print(
                    f"- {finding.violation_type}: "
                    f"{finding.message}"
                )

        print("\nUnsafe query:")
        print(unsafe_sql.strip())

        print(
            f"Allowed : "
            f"{unsafe_result.allowed}"
        )

        print(
            f"Message : "
            f"{unsafe_result.user_message}"
        )

        if not safe_result.allowed:
            raise RuntimeError(
                "The safe query was incorrectly blocked."
            )

        if unsafe_result.allowed:
            raise RuntimeError(
                "The unsafe query was incorrectly allowed."
            )

        print("\n" + "=" * 80)
        print(
            "RESULT: SCHEMA POLICY VALIDATOR "
            "IS WORKING"
        )
        print("=" * 80)

    except Exception as exc:
        print("\n" + "=" * 80)
        print(
            "RESULT: SCHEMA POLICY VALIDATOR "
            "FAILED"
        )
        print("=" * 80)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()