"""
Deterministic output validator for the Multi-Agent Analytics System.

This validator runs after Agent 5 creates FinalReport.

It verifies:

1. Final SQL remains structurally safe and SELECT-only.
2. Query-result shape is internally consistent.
3. Chart columns exist in the actual query result.
4. Chart measure columns are numeric.
5. Numerical claims are supported by returned data.
6. Ranking output remains consistent with returned rows.
7. Sensitive information does not appear in the output.
8. Secret values do not appear in the output.
9. System-prompt or internal-instruction leakage is absent.
10. Unsupported causal language is not used.
11. Truncated results are disclosed in report limitations.

This component does not:

- Call an LLM.
- Generate or execute SQL.
- Modify the report.
- Access the database.
- Replace SQLValidator, SchemaPolicyValidator, or OutputGuard.

The result of this validator is later consumed by the LLM evaluator
and LangGraph routing.
"""

import json
import math
import re
from typing import Any, Iterable, Optional

from models.schemas import (
    AnalysisResult,
    FinalReport,
    QueryExecutionResult,
)
from security.models import (
    DeterministicValidationResult,
    EvaluationDecision,
    NumericClaimValidation,
    PIIFinding,
)
from security.pii_detector import PIIDetector
from tools.sql_validator import SQLValidator


class DeterministicOutputValidator:
    """
    Validate the final analytical output using deterministic rules.

    The validator treats privacy, secrets, prompt leakage, and unsafe
    SQL as blocking conditions.

    Unsupported numerical claims and chart incompatibility normally
    require report regeneration.
    """

    NUMBER_PATTERN = re.compile(
        r"""
        (?<![\w.])
        (?P<currency>[$₹€£])?
        (?P<sign>[-+]?)
        (?P<number>
            (?:
                \d{1,3}(?:,\d{3})+
                |
                \d+
            )
            (?:\.\d+)?
        )
        (?P<percent>\s*%)?
        (?![\w.])
        """,
        flags=re.VERBOSE,
    )

    CAUSAL_LANGUAGE_PATTERN = re.compile(
        r"\b(?:"
        r"caused|causes|causing|"
        r"proved|proves|proven|"
        r"resulted\s+in|results\s+in|"
        r"definitely\s+due\s+to|"
        r"directly\s+led\s+to|"
        r"solely\s+because|"
        r"is\s+the\s+reason\s+for|"
        r"was\s+responsible\s+for"
        r")\b",
        flags=re.IGNORECASE,
    )

    CAUSAL_CAUTION_PATTERN = re.compile(
        r"\b(?:"
        r"may|might|could|possible|possibly|"
        r"suggests?|indicates?|associated|association|"
        r"contributed|contributor|correlated|correlation|"
        r"cannot\s+confirm|cannot\s+establish|"
        r"does\s+not\s+establish|not\s+proven|"
        r"requires?\s+further\s+analysis"
        r")\b",
        flags=re.IGNORECASE,
    )

    PROMPT_LEAKAGE_PATTERNS = [
        re.compile(
            r"\b(?:system|developer)\s+prompt\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\bhidden\s+instructions?\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\binternal\s+instructions?\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\byou\s+are\s+agent\s+\d+\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\breturn\s+only\s+json\s+matching\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\bvalidated\s+analytical\s+plan:\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\bactual\s+sqlite\s+schema:\b",
            flags=re.IGNORECASE,
        ),
    ]

    RANKING_TERMS = {
        "top",
        "bottom",
        "highest",
        "lowest",
        "ranking",
        "ranked",
        "leading",
        "largest",
        "smallest",
    }

    def __init__(
        self,
        sql_validator: Optional[
            SQLValidator
        ] = None,
        pii_detector: Optional[
            PIIDetector
        ] = None,
        numeric_tolerance: float = 0.01,
        percentage_tolerance: float = 0.10,
        maximum_claims_to_validate: int = 100,
    ) -> None:
        """
        Initialize the deterministic validator.

        Args:
            sql_validator:
                Existing SQLValidator configured for SQLite.

            pii_detector:
                Existing deterministic PII and secret detector.

            numeric_tolerance:
                Absolute tolerance for ordinary numeric claims.

            percentage_tolerance:
                Absolute tolerance in percentage points.

            maximum_claims_to_validate:
                Maximum numerical claims inspected in one report.
        """

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

        if (
            pii_detector is not None
            and not isinstance(
                pii_detector,
                PIIDetector,
            )
        ):
            raise TypeError(
                "pii_detector must be a "
                "PIIDetector instance."
            )

        if not isinstance(
            numeric_tolerance,
            (int, float),
        ):
            raise TypeError(
                "numeric_tolerance must be numeric."
            )

        if numeric_tolerance < 0:
            raise ValueError(
                "numeric_tolerance cannot be negative."
            )

        if not isinstance(
            percentage_tolerance,
            (int, float),
        ):
            raise TypeError(
                "percentage_tolerance must be numeric."
            )

        if percentage_tolerance < 0:
            raise ValueError(
                "percentage_tolerance cannot be negative."
            )

        if not isinstance(
            maximum_claims_to_validate,
            int,
        ):
            raise TypeError(
                "maximum_claims_to_validate must "
                "be an integer."
            )

        if maximum_claims_to_validate < 1:
            raise ValueError(
                "maximum_claims_to_validate must "
                "be greater than zero."
            )

        self.sql_validator = (
            sql_validator
            if sql_validator is not None
            else SQLValidator(
                dialect="sqlite"
            )
        )

        self.pii_detector = (
            pii_detector
            if pii_detector is not None
            else PIIDetector()
        )

        self.numeric_tolerance = float(
            numeric_tolerance
        )

        self.percentage_tolerance = float(
            percentage_tolerance
        )

        self.maximum_claims_to_validate = (
            maximum_claims_to_validate
        )

    # -----------------------------------------------------------------
    # Input validation
    # -----------------------------------------------------------------

    def _validate_inputs(
        self,
        user_question: str,
        generated_sql: str,
        query_result: QueryExecutionResult,
        analysis_result: AnalysisResult,
        final_report: FinalReport,
    ) -> None:
        """
        Validate public method inputs.
        """

        if not isinstance(
            user_question,
            str,
        ):
            raise TypeError(
                "user_question must be a string."
            )

        if not user_question.strip():
            raise ValueError(
                "user_question cannot be empty."
            )

        if not isinstance(
            generated_sql,
            str,
        ):
            raise TypeError(
                "generated_sql must be a string."
            )

        if not generated_sql.strip():
            raise ValueError(
                "generated_sql cannot be empty."
            )

        if not isinstance(
            query_result,
            QueryExecutionResult,
        ):
            raise TypeError(
                "query_result must be a "
                "QueryExecutionResult instance."
            )

        if not isinstance(
            analysis_result,
            AnalysisResult,
        ):
            raise TypeError(
                "analysis_result must be an "
                "AnalysisResult instance."
            )

        if not isinstance(
            final_report,
            FinalReport,
        ):
            raise TypeError(
                "final_report must be a "
                "FinalReport instance."
            )

    # -----------------------------------------------------------------
    # Report text extraction
    # -----------------------------------------------------------------

    def _object_to_text(
        self,
        value: Any,
    ) -> str:
        """
        Convert a structured report value into searchable text.
        """

        if value is None:
            return ""

        if isinstance(value, str):
            return value

        if isinstance(
            value,
            (int, float, bool),
        ):
            return str(value)

        if isinstance(value, dict):
            return " ".join(
                self._object_to_text(
                    item
                )
                for item in value.values()
            )

        if isinstance(
            value,
            (list, tuple, set),
        ):
            return " ".join(
                self._object_to_text(
                    item
                )
                for item in value
            )

        if hasattr(
            value,
            "model_dump",
        ):
            return self._object_to_text(
                value.model_dump(
                    mode="json"
                )
            )

        return str(value)

    def _report_sections(
        self,
        final_report: FinalReport,
    ) -> list[tuple[str, str]]:
        """
        Return named report sections for claim validation.
        """

        sections: list[
            tuple[str, str]
        ] = [
            (
                "business_summary",
                self._object_to_text(
                    final_report.business_summary
                ),
            ),
            (
                "key_findings",
                self._object_to_text(
                    final_report.key_findings
                ),
            ),
            (
                "root_cause_indicators",
                self._object_to_text(
                    final_report
                    .root_cause_indicators
                ),
            ),
            (
                "recommendations",
                self._object_to_text(
                    final_report.recommendations
                ),
            ),
            (
                "limitations",
                self._object_to_text(
                    final_report.limitations
                ),
            ),
        ]

        if final_report.chart is not None:
            sections.append(
                (
                    "chart",
                    self._object_to_text(
                        final_report.chart
                    ),
                )
            )

        return sections

    def _complete_report_text(
        self,
        analysis_result: AnalysisResult,
        final_report: FinalReport,
    ) -> str:
        """
        Return all analysis and reporting text in one string.
        """

        return " ".join(
            [
                self._object_to_text(
                    analysis_result
                ),
                self._object_to_text(
                    final_report
                ),
            ]
        ).strip()

    # -----------------------------------------------------------------
    # SQL validation
    # -----------------------------------------------------------------

    def _validate_sql(
        self,
        generated_sql: str,
        query_result: QueryExecutionResult,
    ) -> tuple[bool, list[str]]:
        """
        Validate SQL safety and executed-query consistency.
        """

        issues: list[str] = []

        validation_result = (
            self.sql_validator.validate(
                generated_sql
            )
        )

        if not validation_result.is_valid:
            issues.append(
                "The final SQL failed independent "
                "structural safety validation."
            )

        executed_query = (
            query_result.query.strip()
        )

        if not executed_query:
            issues.append(
                "QueryExecutionResult does not contain "
                "the executed SQL."
            )

        elif validation_result.is_valid:
            executed_validation = (
                self.sql_validator.validate(
                    executed_query
                )
            )

            if not executed_validation.is_valid:
                issues.append(
                    "The SQL stored in QueryExecutionResult "
                    "failed independent safety validation."
                )

        return (
            len(issues) == 0,
            issues,
        )

    # -----------------------------------------------------------------
    # Query-result validation
    # -----------------------------------------------------------------

    def _validate_result_shape(
        self,
        query_result: QueryExecutionResult,
    ) -> tuple[bool, list[str]]:
        """
        Validate result columns, rows, and row count.
        """

        issues: list[str] = []

        if (
            query_result.row_count
            != len(query_result.rows)
        ):
            issues.append(
                "Query result row_count does not match "
                "the number of returned rows."
            )

        if not query_result.columns:
            issues.append(
                "Query result contains no declared columns."
            )

        available_columns = set(
            query_result.columns
        )

        for row_number, row in enumerate(
            query_result.rows,
            start=1,
        ):
            if not isinstance(row, dict):
                issues.append(
                    f"Query result row {row_number} "
                    "is not a dictionary."
                )

                continue

            unexpected_columns = (
                set(row.keys())
                - available_columns
            )

            if unexpected_columns:
                issues.append(
                    f"Query result row {row_number} "
                    "contains undeclared columns."
                )

        if query_result.execution_time_ms < 0:
            issues.append(
                "Query execution time cannot be negative."
            )

        return (
            len(issues) == 0,
            issues,
        )

    # -----------------------------------------------------------------
    # Chart validation
    # -----------------------------------------------------------------

    def _series_is_numeric(
        self,
        query_result: QueryExecutionResult,
        column_name: str,
    ) -> bool:
        """
        Check whether all non-null values in one column are numeric.
        """

        non_null_values = [
            row.get(column_name)
            for row in query_result.rows
            if row.get(column_name) is not None
        ]

        if not non_null_values:
            return False

        for value in non_null_values:
            if isinstance(
                value,
                bool,
            ):
                return False

            if isinstance(
                value,
                (int, float),
            ):
                continue

            if isinstance(value, str):
                try:
                    float(
                        value.replace(
                            ",",
                            "",
                        )
                    )

                except ValueError:
                    return False

                continue

            return False

        return True

    def _validate_chart(
        self,
        query_result: QueryExecutionResult,
        final_report: FinalReport,
    ) -> tuple[bool, list[str]]:
        """
        Validate chart fields against actual query-result columns.
        """

        issues: list[str] = []

        chart = final_report.chart

        if chart is None:
            return True, issues

        chart_type = str(
            chart.chart_type
        )

        if chart_type == "none":
            return True, issues

        available_columns = set(
            query_result.columns
        )

        referenced_columns: list[str] = []

        if chart.x_column:
            referenced_columns.append(
                chart.x_column
            )

        referenced_columns.extend(
            chart.y_columns
        )

        if chart.category_column:
            referenced_columns.append(
                chart.category_column
            )

        invalid_columns = [
            column
            for column in referenced_columns
            if column not in available_columns
        ]

        if invalid_columns:
            issues.append(
                "The chart references columns that are "
                "not present in the query result: "
                f"{invalid_columns}."
            )

        graphical_types = {
            "line",
            "bar",
            "horizontal_bar",
            "area",
            "scatter",
            "pie",
        }

        if chart_type in graphical_types:
            if not chart.x_column:
                issues.append(
                    f"Chart type '{chart_type}' requires "
                    "an x_column."
                )

            if not chart.y_columns:
                issues.append(
                    f"Chart type '{chart_type}' requires "
                    "at least one y-column."
                )

        if not invalid_columns:
            for y_column in chart.y_columns:
                if not self._series_is_numeric(
                    query_result=query_result,
                    column_name=y_column,
                ):
                    issues.append(
                        f"Chart measure column '{y_column}' "
                        "is not numeric."
                    )

        if (
            chart_type == "scatter"
            and len(chart.y_columns) != 1
        ):
            issues.append(
                "A scatter chart requires exactly "
                "one y-column."
            )

        if (
            chart_type == "pie"
            and len(chart.y_columns) != 1
        ):
            issues.append(
                "A pie chart requires exactly "
                "one numeric measure column."
            )

        return (
            len(issues) == 0,
            issues,
        )

    # -----------------------------------------------------------------
    # Evidence collection
    # -----------------------------------------------------------------

    def _coerce_number(
        self,
        value: Any,
    ) -> Optional[float]:
        """
        Convert a database value into a finite number.
        """

        if value is None:
            return None

        if isinstance(value, bool):
            return None

        if isinstance(
            value,
            (int, float),
        ):
            numeric_value = float(value)

        elif isinstance(value, str):
            try:
                numeric_value = float(
                    value.replace(
                        ",",
                        "",
                    ).strip()
                )

            except ValueError:
                return None

        else:
            return None

        if not math.isfinite(
            numeric_value
        ):
            return None

        return numeric_value

    def _numeric_evidence(
        self,
        query_result: QueryExecutionResult,
    ) -> list[tuple[str, float, str]]:
        """
        Build deterministic numeric evidence.

        Returns:
            Tuples containing metric name, value, and safe evidence.
        """

        evidence: list[
            tuple[str, float, str]
        ] = []

        # Row-count evidence.
        evidence.append(
            (
                "row_count",
                float(
                    query_result.row_count
                ),
                (
                    "The query returned "
                    f"{query_result.row_count} rows."
                ),
            )
        )

        numeric_columns: dict[
            str,
            list[float]
        ] = {}

        for row_number, row in enumerate(
            query_result.rows,
            start=1,
        ):
            for column_name in (
                query_result.columns
            ):
                numeric_value = (
                    self._coerce_number(
                        row.get(column_name)
                    )
                )

                if numeric_value is None:
                    continue

                numeric_columns.setdefault(
                    column_name,
                    [],
                ).append(
                    numeric_value
                )

                evidence.append(
                    (
                        column_name,
                        numeric_value,
                        (
                            f"Returned row {row_number} "
                            f"contains {column_name}."
                        ),
                    )
                )

        for column_name, values in (
            numeric_columns.items()
        ):
            if not values:
                continue

            calculated_values = {
                f"{column_name}.minimum": min(
                    values
                ),
                f"{column_name}.maximum": max(
                    values
                ),
                f"{column_name}.sum": sum(
                    values
                ),
                f"{column_name}.mean": (
                    sum(values)
                    / len(values)
                ),
                f"{column_name}.count": float(
                    len(values)
                ),
            }

            for metric_name, metric_value in (
                calculated_values.items()
            ):
                evidence.append(
                    (
                        metric_name,
                        float(metric_value),
                        (
                            "Deterministic calculation "
                            f"for {metric_name}."
                        ),
                    )
                )

            total = sum(values)

            if total != 0:
                for value in values:
                    percentage = (
                        value
                        / total
                        * 100.0
                    )

                    evidence.append(
                        (
                            (
                                f"{column_name}."
                                "share_percentage"
                            ),
                            percentage,
                            (
                                "Deterministic share "
                                f"calculation for {column_name}."
                            ),
                        )
                    )

        return evidence

    def _is_close(
        self,
        claimed_value: float,
        verified_value: float,
        tolerance: float,
    ) -> bool:
        """
        Compare values with absolute and relative tolerance.
        """

        if math.isclose(
            claimed_value,
            verified_value,
            rel_tol=1e-6,
            abs_tol=tolerance,
        ):
            return True

        # Permit ordinary display rounding.
        rounded_claim = round(
            claimed_value,
            2,
        )

        rounded_verified = round(
            verified_value,
            2,
        )

        return (
            abs(
                rounded_claim
                - rounded_verified
            )
            <= tolerance
        )

    # -----------------------------------------------------------------
    # Numerical claim extraction and validation
    # -----------------------------------------------------------------

    def _split_claim_sentences(
        self,
        text: str,
    ) -> list[str]:
        """
        Split report text into short claim-like segments.
        """

        segments = re.split(
            r"(?<=[.!?])\s+|\n+|(?<=;)\s+",
            text,
        )

        return [
            segment.strip()
            for segment in segments
            if segment.strip()
        ]

    def _looks_structural_number(
        self,
        claim: str,
        claimed_value: float,
    ) -> bool:
        """
        Exclude common structural numbers from strict metric checks.

        Examples:

        - "top 5 products"
        - "priority 1"
        - "step 2"

        These are request or formatting parameters, not calculated
        business claims.
        """

        structural_patterns = [
            re.compile(
                r"\btop\s+\d+\b",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"\bbottom\s+\d+\b",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"\bstep\s+\d+\b",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"\bpriority\s+\d+\b",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"\bfirst\s+\d+\b",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"\blast\s+\d+\b",
                flags=re.IGNORECASE,
            ),
        ]

        return any(
            pattern.search(claim)
            for pattern in structural_patterns
        )

    def _validate_numeric_claims(
        self,
        query_result: QueryExecutionResult,
        final_report: FinalReport,
    ) -> tuple[
        bool,
        list[
            NumericClaimValidation
        ],
        list[str],
        list[str],
        list[str],
    ]:
        """
        Validate report numbers against returned or derived evidence.

        Returns:
            valid
            numeric claim records
            verified claims
            unsupported claims
            issues
        """

        evidence = self._numeric_evidence(
            query_result
        )

        numeric_claims: list[
            NumericClaimValidation
        ] = []

        verified_claims: list[str] = []

        unsupported_claims: list[str] = []

        issues: list[str] = []

        claims_processed = 0

        for section_name, section_text in (
            self._report_sections(
                final_report
            )
        ):
            for claim in (
                self._split_claim_sentences(
                    section_text
                )
            ):
                for match in (
                    self.NUMBER_PATTERN.finditer(
                        claim
                    )
                ):
                    if (
                        claims_processed
                        >= self
                        .maximum_claims_to_validate
                    ):
                        issues.append(
                            "The report contains more "
                            "numerical claims than the "
                            "configured validation limit."
                        )

                        return (
                            False,
                            numeric_claims,
                            verified_claims,
                            unsupported_claims,
                            issues,
                        )

                    raw_number = (
                        match.group(
                            "number"
                        )
                    )

                    sign = match.group(
                        "sign"
                    ) or ""

                    is_percentage = bool(
                        match.group(
                            "percent"
                        )
                    )

                    claimed_value = float(
                        (
                            sign
                            + raw_number.replace(
                                ",",
                                "",
                            )
                        )
                    )

                    if self._looks_structural_number(
                        claim=claim,
                        claimed_value=(
                            claimed_value
                        ),
                    ):
                        continue

                    claims_processed += 1

                    tolerance = (
                        self.percentage_tolerance
                        if is_percentage
                        else self.numeric_tolerance
                    )

                    matching_evidence = [
                        (
                            metric_name,
                            verified_value,
                            evidence_text,
                        )
                        for (
                            metric_name,
                            verified_value,
                            evidence_text,
                        ) in evidence
                        if self._is_close(
                            claimed_value=(
                                claimed_value
                            ),
                            verified_value=(
                                verified_value
                            ),
                            tolerance=tolerance,
                        )
                    ]

                    if matching_evidence:
                        metric_name = (
                            matching_evidence[0][0]
                        )

                        verified_value = (
                            matching_evidence[0][1]
                        )

                        evidence_texts = [
                            item[2]
                            for item
                            in matching_evidence[:5]
                        ]

                        numeric_claims.append(
                            NumericClaimValidation(
                                claim=claim,
                                metric_name=(
                                    metric_name
                                ),
                                claimed_value=(
                                    claimed_value
                                ),
                                verified_value=(
                                    verified_value
                                ),
                                tolerance=tolerance,
                                supported=True,
                                evidence=(
                                    evidence_texts
                                ),
                                issue=None,
                            )
                        )

                        verified_claims.append(
                            claim
                        )

                    else:
                        issue = (
                            "The numerical value in this "
                            "claim could not be verified "
                            "from returned rows or supported "
                            "deterministic calculations."
                        )

                        numeric_claims.append(
                            NumericClaimValidation(
                                claim=claim,
                                metric_name=None,
                                claimed_value=(
                                    claimed_value
                                ),
                                verified_value=None,
                                tolerance=tolerance,
                                supported=False,
                                evidence=[],
                                issue=issue,
                            )
                        )

                        unsupported_claims.append(
                            claim
                        )

        if unsupported_claims:
            issues.append(
                "One or more numerical claims are not "
                "supported by the query result."
            )

        return (
            len(unsupported_claims) == 0,
            numeric_claims,
            list(
                dict.fromkeys(
                    verified_claims
                )
            ),
            list(
                dict.fromkeys(
                    unsupported_claims
                )
            ),
            issues,
        )

    # -----------------------------------------------------------------
    # Ranking validation
    # -----------------------------------------------------------------

    def _question_requests_ranking(
        self,
        user_question: str,
    ) -> bool:
        """
        Determine whether the question requests ranking.
        """

        normalized_words = {
            word.lower()
            for word in re.findall(
                r"[A-Za-z]+",
                user_question,
            )
        }

        return bool(
            normalized_words
            & self.RANKING_TERMS
        )

    def _first_numeric_column(
        self,
        query_result: QueryExecutionResult,
    ) -> Optional[str]:
        """
        Return the first result column containing numeric data.
        """

        for column_name in (
            query_result.columns
        ):
            if self._series_is_numeric(
                query_result=query_result,
                column_name=column_name,
            ):
                return column_name

        return None

    def _validate_ranking(
        self,
        user_question: str,
        generated_sql: str,
        query_result: QueryExecutionResult,
    ) -> tuple[bool, list[str]]:
        """
        Validate ranking order when the request clearly asks for it.
        """

        issues: list[str] = []

        if not self._question_requests_ranking(
            user_question
        ):
            return True, issues

        if query_result.row_count < 2:
            return True, issues

        numeric_column = (
            self._first_numeric_column(
                query_result
            )
        )

        if numeric_column is None:
            issues.append(
                "A ranking was requested, but the result "
                "contains no numeric ranking measure."
            )

            return False, issues

        values = [
            self._coerce_number(
                row.get(
                    numeric_column
                )
            )
            for row in query_result.rows
        ]

        if any(
            value is None
            for value in values
        ):
            issues.append(
                "The ranking measure contains "
                "nonnumeric values."
            )

            return False, issues

        numeric_values = [
            float(value)
            for value in values
            if value is not None
        ]

        question_lower = (
            user_question.lower()
        )

        sql_upper = generated_sql.upper()

        descending_requested = any(
            term in question_lower
            for term in (
                "top",
                "highest",
                "largest",
                "leading",
            )
        )

        ascending_requested = any(
            term in question_lower
            for term in (
                "bottom",
                "lowest",
                "smallest",
            )
        )

        if (
            descending_requested
            or " DESC" in sql_upper
        ):
            expected_values = sorted(
                numeric_values,
                reverse=True,
            )

        elif (
            ascending_requested
            or " ASC" in sql_upper
        ):
            expected_values = sorted(
                numeric_values
            )

        else:
            issues.append(
                "A ranking was requested, but the "
                "query order cannot be determined."
            )

            return False, issues

        if numeric_values != expected_values:
            issues.append(
                "Returned rows are not ordered "
                "consistently with the requested ranking."
            )

        return (
            len(issues) == 0,
            issues,
        )

    # -----------------------------------------------------------------
    # Privacy, secret, and prompt-leakage validation
    # -----------------------------------------------------------------

    def _scan_sensitive_output(
        self,
        output_text: str,
    ) -> tuple[
        bool,
        bool,
        list[PIIFinding],
        list[str],
    ]:
        """
        Scan final output for PII and secrets.

        Returns:
            pii_safe
            secret_safe
            privacy findings
            issues
        """

        issues: list[str] = []

        if not output_text.strip():
            return (
                True,
                True,
                [],
                issues,
            )

        pii_result = (
            self.pii_detector.detect(
                output_text
            )
        )

        findings = list(
            pii_result.findings
        )

        secret_types = {
            "private_key",
            "jwt_token",
            "database_connection_string",
            "password",
            "api_key",
            "authorization_bearer_token",
        }

        secret_findings = [
            finding
            for finding in findings
            if finding.entity_type
            in secret_types
        ]

        non_secret_findings = [
            finding
            for finding in findings
            if finding.entity_type
            not in secret_types
        ]

        pii_safe = (
            len(non_secret_findings)
            == 0
        )

        secret_safe = (
            len(secret_findings)
            == 0
        )

        if not pii_safe:
            issues.append(
                "The generated output contains "
                "personal, confidential, or "
                "restricted information."
            )

        if not secret_safe:
            issues.append(
                "The generated output contains "
                "secret-like information."
            )

        return (
            pii_safe,
            secret_safe,
            findings,
            issues,
        )

    def _validate_prompt_leakage(
        self,
        output_text: str,
    ) -> tuple[bool, list[str]]:
        """
        Detect obvious internal prompt leakage.
        """

        issues: list[str] = []

        matched_categories = []

        for pattern in (
            self.PROMPT_LEAKAGE_PATTERNS
        ):
            if pattern.search(
                output_text
            ):
                matched_categories.append(
                    pattern.pattern
                )

        if matched_categories:
            issues.append(
                "The generated output appears to "
                "contain internal prompt or instruction "
                "language."
            )

        return (
            len(issues) == 0,
            issues,
        )

    # -----------------------------------------------------------------
    # Causal-language validation
    # -----------------------------------------------------------------

    def _validate_causal_language(
        self,
        final_report: FinalReport,
    ) -> tuple[bool, list[str]]:
        """
        Flag unsupported definitive causal wording.

        This analytics pipeline uses observational SQL results and does
        not establish causality.
        """

        issues: list[str] = []

        relevant_text = " ".join(
            [
                self._object_to_text(
                    final_report.business_summary
                ),
                self._object_to_text(
                    final_report
                    .root_cause_indicators
                ),
                self._object_to_text(
                    final_report.recommendations
                ),
            ]
        )

        for sentence in (
            self._split_claim_sentences(
                relevant_text
            )
        ):
            definitive_match = (
                self.CAUSAL_LANGUAGE_PATTERN.search(
                    sentence
                )
            )

            if definitive_match is None:
                continue

            if self.CAUSAL_CAUTION_PATTERN.search(
                sentence
            ):
                continue

            issues.append(
                "The report contains definitive causal "
                "language that is not supported by "
                "observational query results."
            )

            break

        return (
            len(issues) == 0,
            issues,
        )

    # -----------------------------------------------------------------
    # Truncation validation
    # -----------------------------------------------------------------

    def _validate_truncation_disclosure(
        self,
        query_result: QueryExecutionResult,
        final_report: FinalReport,
    ) -> tuple[bool, list[str]]:
        """
        Ensure truncated results are disclosed.
        """

        issues: list[str] = []

        if not query_result.truncated:
            return True, issues

        limitation_text = " ".join(
            final_report.limitations
        ).lower()

        truncation_terms = (
            "truncated",
            "row limit",
            "returned rows",
            "partial result",
            "limited result",
        )

        if not any(
            term in limitation_text
            for term in truncation_terms
        ):
            issues.append(
                "The query result was truncated, but "
                "the report limitations do not disclose it."
            )

        return (
            len(issues) == 0,
            issues,
        )

    # -----------------------------------------------------------------
    # Decision policy
    # -----------------------------------------------------------------

    def _decision(
        self,
        *,
        schema_valid: bool,
        sql_safe: bool,
        numerical_claims_valid: bool,
        ranking_valid: bool,
        chart_valid: bool,
        pii_safe: bool,
        secret_safe: bool,
        prompt_leakage_safe: bool,
        causal_language_safe: bool,
        truncation_disclosed: bool,
    ) -> EvaluationDecision:
        """
        Derive the deterministic workflow decision.

        Blocking:
            Unsafe SQL, PII, secrets, or prompt leakage.

        Regeneration:
            Unsupported numbers, invalid chart, ranking mismatch,
            unsupported causal language, or missing truncation warning.

        Approval:
            All checks pass.
        """

        if not all(
            [
                schema_valid,
                sql_safe,
                pii_safe,
                secret_safe,
                prompt_leakage_safe,
            ]
        ):
            return EvaluationDecision.BLOCK

        if not all(
            [
                numerical_claims_valid,
                ranking_valid,
                chart_valid,
                causal_language_safe,
                truncation_disclosed,
            ]
        ):
            return EvaluationDecision.REGENERATE

        return EvaluationDecision.APPROVE

    # -----------------------------------------------------------------
    # Public validation method
    # -----------------------------------------------------------------

    def validate(
        self,
        *,
        user_question: str,
        generated_sql: str,
        query_result: QueryExecutionResult,
        analysis_result: AnalysisResult,
        final_report: FinalReport,
    ) -> DeterministicValidationResult:
        """
        Validate the final analytical output.

        Returns:
            DeterministicValidationResult.
        """

        self._validate_inputs(
            user_question=user_question,
            generated_sql=generated_sql,
            query_result=query_result,
            analysis_result=analysis_result,
            final_report=final_report,
        )

        issues: list[str] = []

        sql_safe, sql_issues = (
            self._validate_sql(
                generated_sql=generated_sql,
                query_result=query_result,
            )
        )

        issues.extend(
            sql_issues
        )

        schema_valid, shape_issues = (
            self._validate_result_shape(
                query_result
            )
        )

        issues.extend(
            shape_issues
        )

        chart_valid, chart_issues = (
            self._validate_chart(
                query_result=query_result,
                final_report=final_report,
            )
        )

        issues.extend(
            chart_issues
        )

        (
            numerical_claims_valid,
            numeric_claims,
            verified_claims,
            unsupported_claims,
            numeric_issues,
        ) = self._validate_numeric_claims(
            query_result=query_result,
            final_report=final_report,
        )

        issues.extend(
            numeric_issues
        )

        ranking_valid, ranking_issues = (
            self._validate_ranking(
                user_question=user_question,
                generated_sql=generated_sql,
                query_result=query_result,
            )
        )

        issues.extend(
            ranking_issues
        )

        output_text = (
            self._complete_report_text(
                analysis_result=(
                    analysis_result
                ),
                final_report=final_report,
            )
        )

        (
            pii_safe,
            secret_safe,
            privacy_findings,
            privacy_issues,
        ) = self._scan_sensitive_output(
            output_text
        )

        issues.extend(
            privacy_issues
        )

        (
            prompt_leakage_safe,
            prompt_issues,
        ) = self._validate_prompt_leakage(
            output_text
        )

        issues.extend(
            prompt_issues
        )

        (
            causal_language_safe,
            causal_issues,
        ) = self._validate_causal_language(
            final_report
        )

        issues.extend(
            causal_issues
        )

        (
            truncation_disclosed,
            truncation_issues,
        ) = self._validate_truncation_disclosure(
            query_result=query_result,
            final_report=final_report,
        )

        issues.extend(
            truncation_issues
        )

        # DeterministicValidationResult does not currently have a
        # separate truncation field. Missing truncation disclosure is
        # therefore reflected in causal_language_safe so approval
        # cannot occur without complete limitations.
        effective_causal_safety = (
            causal_language_safe
            and truncation_disclosed
        )

        decision = self._decision(
            schema_valid=schema_valid,
            sql_safe=sql_safe,
            numerical_claims_valid=(
                numerical_claims_valid
            ),
            ranking_valid=ranking_valid,
            chart_valid=chart_valid,
            pii_safe=pii_safe,
            secret_safe=secret_safe,
            prompt_leakage_safe=(
                prompt_leakage_safe
            ),
            causal_language_safe=(
                effective_causal_safety
            ),
            truncation_disclosed=(
                truncation_disclosed
            ),
        )

        all_checks = [
            schema_valid,
            sql_safe,
            numerical_claims_valid,
            ranking_valid,
            chart_valid,
            pii_safe,
            secret_safe,
            prompt_leakage_safe,
            effective_causal_safety,
        ]

        valid = all(
            all_checks
        )

        return DeterministicValidationResult(
            valid=valid,
            schema_valid=schema_valid,
            sql_safe=sql_safe,
            numerical_claims_valid=(
                numerical_claims_valid
            ),
            ranking_valid=ranking_valid,
            chart_valid=chart_valid,
            pii_safe=pii_safe,
            secret_safe=secret_safe,
            prompt_leakage_safe=(
                prompt_leakage_safe
            ),
            causal_language_safe=(
                effective_causal_safety
            ),
            numeric_claims=numeric_claims,
            verified_claims=(
                verified_claims
            ),
            unsupported_claims=(
                unsupported_claims
            ),
            privacy_findings=(
                privacy_findings
            ),
            issues=list(
                dict.fromkeys(
                    issues
                )
            ),
            decision=decision,
        )


# ---------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------

def main():
    """Run the deterministic validator smoke test using the complete LangGraph output.

    Ollama is used by the workflow, but not by this validator.
    """

    from workflow.graph import (
        run_workflow,
    )

    print("=" * 80)
    print(
        "DETERMINISTIC OUTPUT VALIDATOR "
        "SMOKE TEST"
    )
    print("=" * 80)

    question = (
        "Show the top 5 products by revenue."
    )

    try:
        state = run_workflow(
            user_question=question,
            max_sql_retries=3,
            recursion_limit=45,
        )

        if (
            state.get(
                "workflow_status"
            )
            != "completed"
        ):
            raise RuntimeError(
                "The complete workflow did not "
                "finish successfully. "
                f"Status: "
                f"{state.get('workflow_status')}. "
                f"Error: "
                f"{state.get('error_message')}."
            )

        generated_sql = state.get(
            "generated_sql"
        )

        query_result = state.get(
            "query_result"
        )

        analysis_result = state.get(
            "analysis_result"
        )

        final_report = state.get(
            "final_report"
        )

        if not isinstance(
            generated_sql,
            str,
        ):
            raise RuntimeError(
                "Generated SQL is missing."
            )

        if not isinstance(
            query_result,
            QueryExecutionResult,
        ):
            raise RuntimeError(
                "QueryExecutionResult is missing."
            )

        if not isinstance(
            analysis_result,
            AnalysisResult,
        ):
            raise RuntimeError(
                "AnalysisResult is missing."
            )

        if not isinstance(
            final_report,
            FinalReport,
        ):
            raise RuntimeError(
                "FinalReport is missing."
            )

        validator = (
            DeterministicOutputValidator()
        )

        result = validator.validate(
            user_question=question,
            generated_sql=generated_sql,
            query_result=query_result,
            analysis_result=analysis_result,
            final_report=final_report,
        )

        print(
            f"\nValid                   : "
            f"{result.valid}"
        )

        print(
            f"Decision                : "
            f"{result.decision}"
        )

        print(
            f"Schema valid            : "
            f"{result.schema_valid}"
        )

        print(
            f"SQL safe                : "
            f"{result.sql_safe}"
        )

        print(
            f"Numerical claims valid  : "
            f"{result.numerical_claims_valid}"
        )

        print(
            f"Ranking valid           : "
            f"{result.ranking_valid}"
        )

        print(
            f"Chart valid             : "
            f"{result.chart_valid}"
        )

        print(
            f"PII safe                : "
            f"{result.pii_safe}"
        )

        print(
            f"Secret safe             : "
            f"{result.secret_safe}"
        )

        print(
            f"Prompt leakage safe     : "
            f"{result.prompt_leakage_safe}"
        )

        print(
            f"Causal language safe    : "
            f"{result.causal_language_safe}"
        )

        print(
            f"Verified claims         : "
            f"{len(result.verified_claims)}"
        )

        print(
            f"Unsupported claims      : "
            f"{len(result.unsupported_claims)}"
        )

        if result.issues:
            print("\nIssues:")

            for issue in result.issues:
                print(f"- {issue}")

        if result.unsupported_claims:
            print("\nUnsupported claims:")

            for claim in (
                result.unsupported_claims
            ):
                print(f"- {claim}")

        print("\n" + "=" * 80)

        if str(
            result.decision
        ) == "approve":
            print(
                "RESULT: DETERMINISTIC OUTPUT "
                "VALIDATION PASSED"
            )

        else:
            print(
                "RESULT: DETERMINISTIC OUTPUT "
                "VALIDATION REQUIRES ACTION"
            )

        print("=" * 80)

    except Exception as exc:
        print("\n" + "=" * 80)
        print(
            "RESULT: DETERMINISTIC OUTPUT "
            "VALIDATOR FAILED"
        )
        print("=" * 80)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()