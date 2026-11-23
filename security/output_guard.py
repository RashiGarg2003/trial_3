"""
Final output guard for the Multi-Agent Analytics System.

This guard runs after:

1. Agent 5 Reporting
2. Deterministic output validation
3. LLM output evaluation

Responsibilities:

- Block output containing secrets or restricted identifiers.
- Redact permitted personal information.
- Detect internal prompt leakage.
- Validate query-result columns against authorized output columns.
- Remove unauthorized technical fields.
- Produce a sanitized payload for FastAPI.
- Preserve deterministic and LLM security decisions.

The output guard does not:

- Generate SQL.
- Execute SQL.
- Call an LLM.
- Recalculate analytical values.
- Override deterministic security failures.
"""

import copy
import re
from typing import Any, Optional

from models.schemas import (
    FinalReport,
    QueryExecutionResult,
)
from security.models import (
    DeterministicValidationResult,
    OutputEvaluation,
    OutputGuardResult,
    PIIFinding,
    RedactionRecord,
    SecurityDecision,
    UserRole,
)
from security.pii_detector import PIIDetector
def enum_value(
    value: Any,
) -> Any:
    """
    Return the primitive value stored by an Enum.
    """

    if value is None:
        return None

    primitive_value = getattr(
        value,
        "value",
        value,
    )

    if isinstance(
        primitive_value,
        str,
    ):
        return primitive_value.strip().lower()

    return primitive_value

class OutputGuard:
    """
    Final deterministic protection layer before API serialization.

    Blocking conditions:

    - Deterministic validation returned block.
    - LLM evaluation returned block.
    - Secret information is detected.
    - Restricted identifiers are detected.
    - Prompt or internal-instruction leakage is detected.
    - Unauthorized query-result columns are present.

    Redaction conditions:

    - Permitted personal information such as email or phone data is
      detected in otherwise authorized output.
    """

    SECRET_ENTITY_TYPES = {
        "private_key",
        "jwt_token",
        "database_connection_string",
        "password",
        "api_key",
        "authorization_bearer_token",
    }

    RESTRICTED_ENTITY_TYPES = {
        "aadhaar_number",
        "pan_number",
        "payment_card_number",
        "bank_account_number",
        "upi_identifier",
        "indian_passport_number",
    }

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
            r"\bactual\s+sqlite\s+schema\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\bvalidated\s+analytical\s+plan\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\bignore\s+(?:all\s+)?previous\s+instructions\b",
            flags=re.IGNORECASE,
        ),
    ]

    ROLE_TECHNICAL_FIELDS = {
        UserRole.VIEWER: set(),
        UserRole.ANALYST: {
            "generated_sql",
            "referenced_tables",
            "sql_retry_count",
            "max_sql_retries",
            "intent",
            "plan",
        },
        UserRole.ADMIN: {
            "generated_sql",
            "referenced_tables",
            "sql_retry_count",
            "max_sql_retries",
            "intent",
            "plan",
            "execution_trace",
        },
    }

    def __init__(
        self,
        pii_detector: Optional[
            PIIDetector
        ] = None,
        maximum_output_rows: int = 1000,
    ) -> None:
        """
        Initialize the output guard.

        Args:
            pii_detector:
                Existing deterministic privacy detector.

            maximum_output_rows:
                Maximum query-result rows permitted in the sanitized
                API payload.
        """

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
            maximum_output_rows,
            int,
        ):
            raise TypeError(
                "maximum_output_rows must be an integer."
            )

        if maximum_output_rows < 1:
            raise ValueError(
                "maximum_output_rows must be greater than zero."
            )

        self.pii_detector = (
            pii_detector
            if pii_detector is not None
            else PIIDetector()
        )

        self.maximum_output_rows = (
            maximum_output_rows
        )

    # -----------------------------------------------------------------
    # Input validation
    # -----------------------------------------------------------------

    def _validate_inputs(
        self,
        final_report: FinalReport,
        query_result: QueryExecutionResult,
        deterministic_result: (
            DeterministicValidationResult
        ),
        output_evaluation: OutputEvaluation,
        role: UserRole,
    ) -> None:
        """
        Validate objects supplied to the final guard.
        """

        if not isinstance(
            final_report,
            FinalReport,
        ):
            raise TypeError(
                "final_report must be a FinalReport instance."
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
            deterministic_result,
            DeterministicValidationResult,
        ):
            raise TypeError(
                "deterministic_result must be a "
                "DeterministicValidationResult instance."
            )

        if not isinstance(
            output_evaluation,
            OutputEvaluation,
        ):
            raise TypeError(
                "output_evaluation must be an "
                "OutputEvaluation instance."
            )

        if not isinstance(
            role,
            UserRole,
        ):
            raise TypeError(
                "role must be a UserRole instance."
            )

    # -----------------------------------------------------------------
    # Prompt-leakage detection
    # -----------------------------------------------------------------

    def _contains_prompt_leakage(
        self,
        text: str,
    ) -> bool:
        """
        Detect internal prompt or instruction leakage.
        """

        return any(
            pattern.search(text)
            for pattern
            in self.PROMPT_LEAKAGE_PATTERNS
        )

    # -----------------------------------------------------------------
    # PII scanning and redaction
    # -----------------------------------------------------------------

    def _scan_text(
        self,
        text: str,
        location: str,
    ) -> tuple[
        str,
        list[PIIFinding],
        list[RedactionRecord],
        bool,
        bool,
    ]:
        """
        Scan and sanitize one string.

        Returns:
            sanitized text
            PII findings
            redaction records
            secret detected
            restricted identifier detected
        """

        if not text.strip():
            return (
                text,
                [],
                [],
                False,
                False,
            )

        result = self.pii_detector.detect(
            text
        )

        findings = list(
            result.findings
        )

        secret_detected = any(
            finding.entity_type
            in self.SECRET_ENTITY_TYPES
            for finding in findings
        )

        restricted_detected = any(
            finding.entity_type
            in self.RESTRICTED_ENTITY_TYPES
            for finding in findings
        )

        redactions = [
            RedactionRecord(
                entity_type=(
                    finding.entity_type
                ),
                replacement=(
                    finding.redacted_value
                ),
                location=location,
                reason=(
                    "Sensitive information was "
                    "removed from the final output."
                ),
            )
            for finding in findings
            if finding.entity_type
            not in self.SECRET_ENTITY_TYPES
            and finding.entity_type
            not in self.RESTRICTED_ENTITY_TYPES
        ]

        return (
            result.redacted_text,
            findings,
            redactions,
            secret_detected,
            restricted_detected,
        )

    def _sanitize_value(
        self,
        value: Any,
        location: str,
    ) -> tuple[
        Any,
        list[PIIFinding],
        list[RedactionRecord],
        bool,
        bool,
        bool,
    ]:
        """
        Recursively sanitize a structured output value.

        Returns:
            sanitized value
            findings
            redactions
            secret detected
            restricted detected
            prompt leakage detected
        """

        if value is None:
            return (
                None,
                [],
                [],
                False,
                False,
                False,
            )

        if isinstance(value, str):
            (
                sanitized_text,
                findings,
                redactions,
                secret_detected,
                restricted_detected,
            ) = self._scan_text(
                text=value,
                location=location,
            )

            prompt_leakage = (
                self._contains_prompt_leakage(
                    value
                )
            )

            return (
                sanitized_text,
                findings,
                redactions,
                secret_detected,
                restricted_detected,
                prompt_leakage,
            )

        if isinstance(
            value,
            (int, float, bool),
        ):
            return (
                value,
                [],
                [],
                False,
                False,
                False,
            )

        if isinstance(value, list):
            sanitized_list: list[Any] = []
            all_findings: list[
                PIIFinding
            ] = []
            all_redactions: list[
                RedactionRecord
            ] = []

            secret_detected = False
            restricted_detected = False
            prompt_leakage = False

            for index, item in enumerate(
                value
            ):
                (
                    sanitized_item,
                    findings,
                    redactions,
                    item_secret,
                    item_restricted,
                    item_prompt_leakage,
                ) = self._sanitize_value(
                    value=item,
                    location=(
                        f"{location}[{index}]"
                    ),
                )

                sanitized_list.append(
                    sanitized_item
                )

                all_findings.extend(
                    findings
                )

                all_redactions.extend(
                    redactions
                )

                secret_detected = (
                    secret_detected
                    or item_secret
                )

                restricted_detected = (
                    restricted_detected
                    or item_restricted
                )

                prompt_leakage = (
                    prompt_leakage
                    or item_prompt_leakage
                )

            return (
                sanitized_list,
                all_findings,
                all_redactions,
                secret_detected,
                restricted_detected,
                prompt_leakage,
            )

        if isinstance(value, dict):
            sanitized_dictionary: dict[
                str,
                Any
            ] = {}

            all_findings: list[
                PIIFinding
            ] = []

            all_redactions: list[
                RedactionRecord
            ] = []

            secret_detected = False
            restricted_detected = False
            prompt_leakage = False

            for key, item in value.items():
                safe_key = str(key)

                (
                    sanitized_item,
                    findings,
                    redactions,
                    item_secret,
                    item_restricted,
                    item_prompt_leakage,
                ) = self._sanitize_value(
                    value=item,
                    location=(
                        f"{location}.{safe_key}"
                    ),
                )

                sanitized_dictionary[
                    safe_key
                ] = sanitized_item

                all_findings.extend(
                    findings
                )

                all_redactions.extend(
                    redactions
                )

                secret_detected = (
                    secret_detected
                    or item_secret
                )

                restricted_detected = (
                    restricted_detected
                    or item_restricted
                )

                prompt_leakage = (
                    prompt_leakage
                    or item_prompt_leakage
                )

            return (
                sanitized_dictionary,
                all_findings,
                all_redactions,
                secret_detected,
                restricted_detected,
                prompt_leakage,
            )

        if hasattr(
            value,
            "model_dump",
        ):
            return self._sanitize_value(
                value=value.model_dump(
                    mode="json"
                ),
                location=location,
            )

        return self._sanitize_value(
            value=str(value),
            location=location,
        )

    # -----------------------------------------------------------------
    # Query-result policy
    # -----------------------------------------------------------------

    def _validate_query_result(
        self,
        query_result: QueryExecutionResult,
        allowed_output_columns: Optional[
            set[str]
        ],
    ) -> tuple[
        bool,
        list[str],
    ]:
        """
        Validate query-result shape and output-column authorization.
        """

        issues: list[str] = []

        if (
            query_result.row_count
            != len(query_result.rows)
        ):
            issues.append(
                "Query-result row_count does not "
                "match the included rows."
            )

        if (
            query_result.row_count
            > self.maximum_output_rows
        ):
            issues.append(
                "Query result exceeds the maximum "
                "number of rows permitted in an "
                "API response."
            )

        if allowed_output_columns is not None:
            unauthorized_columns = (
                set(query_result.columns)
                - allowed_output_columns
            )

            if unauthorized_columns:
                issues.append(
                    "The output includes columns that "
                    "are not authorized for this response."
                )

        declared_columns = set(
            query_result.columns
        )

        for row in query_result.rows:
            unexpected_columns = (
                set(row.keys())
                - declared_columns
            )

            if unexpected_columns:
                issues.append(
                    "A query-result row contains "
                    "undeclared output columns."
                )

                break

        return (
            len(issues) == 0,
            issues,
        )

    # -----------------------------------------------------------------
    # Technical-detail filtering
    # -----------------------------------------------------------------

    def filter_technical_details(
        self,
        technical_details: Optional[
            dict[str, Any]
        ],
        role: UserRole,
    ) -> Optional[dict[str, Any]]:
        """
        Restrict technical details according to user role.

        Viewer:
            No technical details.

        Analyst:
            SQL, intent, plan, referenced tables, and retry metadata.

        Admin:
            Analyst fields plus execution trace.

        No role receives secrets, raw prompts, full schema, or the
        original unredacted user question.
        """

        if technical_details is None:
            return None

        if not isinstance(
            technical_details,
            dict,
        ):
            raise TypeError(
                "technical_details must be a dictionary."
            )

        allowed_fields = (
            self.ROLE_TECHNICAL_FIELDS[
                role
            ]
        )

        if not allowed_fields:
            return None

        always_blocked_fields = {
            "original_user_question",
            "database_schema",
            "system_prompt",
            "developer_prompt",
            "environment",
            "environment_variables",
            "password",
            "api_key",
            "access_token",
            "authorization",
        }

        filtered_details = {
            key: copy.deepcopy(value)
            for key, value
            in technical_details.items()
            if key in allowed_fields
            and key
            not in always_blocked_fields
        }

        (
            sanitized_details,
            _,
            _,
            secret_detected,
            restricted_detected,
            prompt_leakage,
        ) = self._sanitize_value(
            value=filtered_details,
            location="technical_details",
        )

        if (
            secret_detected
            or restricted_detected
            or prompt_leakage
        ):
            return None

        if not isinstance(
            sanitized_details,
            dict,
        ):
            return None

        return sanitized_details

    # -----------------------------------------------------------------
    # Decision precedence
    # -----------------------------------------------------------------

    def _authoritative_decision(
        self,
        deterministic_result: (
            DeterministicValidationResult
        ),
        output_evaluation: OutputEvaluation,
    ) -> Optional[
        SecurityDecision
    ]:
        """
        Preserve previous blocking and review decisions.
        """

        deterministic_decision = enum_value(
            deterministic_result.decision
        )

        evaluator_decision = enum_value(
            output_evaluation.decision
        )

        if deterministic_decision == "block":
            return SecurityDecision.BLOCK

        if evaluator_decision == "block":
            return SecurityDecision.BLOCK

        if deterministic_decision == "review":
            return SecurityDecision.REVIEW

        if evaluator_decision == "review":
            return SecurityDecision.REVIEW

        if deterministic_decision == "regenerate":
            return SecurityDecision.REVIEW

        if evaluator_decision == "regenerate":
            return SecurityDecision.REVIEW

        return None

    # -----------------------------------------------------------------
    # Public output guard
    # -----------------------------------------------------------------

    def guard(
        self,
        *,
        final_report: FinalReport,
        query_result: QueryExecutionResult,
        deterministic_result: (
            DeterministicValidationResult
        ),
        output_evaluation: OutputEvaluation,
        role: UserRole | str = (
            UserRole.VIEWER
        ),
        allowed_output_columns: Optional[
            set[str]
        ] = None,
        technical_details: Optional[
            dict[str, Any]
        ] = None,
    ) -> OutputGuardResult:
        """
        Guard and sanitize the final workflow payload.

        Args:
            final_report:
                Agent 5 output.

            query_result:
                Data returned by Agent 3b.

            deterministic_result:
                Objective output validation.

            output_evaluation:
                LLM evaluation result.

            role:
                Authenticated user role.

            allowed_output_columns:
                Optional set of columns approved by schema policy.

            technical_details:
                Optional FastAPI technical metadata.

        Returns:
            OutputGuardResult.
        """

        normalized_role = UserRole(
            role
        )

        self._validate_inputs(
            final_report=final_report,
            query_result=query_result,
            deterministic_result=(
                deterministic_result
            ),
            output_evaluation=(
                output_evaluation
            ),
            role=normalized_role,
        )

        authoritative_decision = (
            self._authoritative_decision(
                deterministic_result=(
                    deterministic_result
                ),
                output_evaluation=(
                    output_evaluation
                ),
            )
        )

        query_valid, query_issues = (
            self._validate_query_result(
                query_result=query_result,
                allowed_output_columns=(
                    allowed_output_columns
                ),
            )
        )

        report_payload = (
            final_report.model_dump(
                mode="json"
            )
        )

        query_payload = {
            "columns": list(
                query_result.columns
            ),
            "rows": copy.deepcopy(
                query_result.rows
            ),
            "row_count": (
                query_result.row_count
            ),
            "execution_time_ms": (
                query_result
                .execution_time_ms
            ),
            "truncated": (
                query_result.truncated
            ),
        }

        filtered_technical_details = (
            self.filter_technical_details(
                technical_details=(
                    technical_details
                ),
                role=normalized_role,
            )
        )

        raw_payload = {
            "final_report": (
                report_payload
            ),
            "query_result": (
                query_payload
            ),
            "technical_details": (
                filtered_technical_details
            ),
        }

        (
            sanitized_payload,
            privacy_findings,
            redactions,
            secret_detected,
            restricted_detected,
            prompt_leakage,
        ) = self._sanitize_value(
            value=raw_payload,
            location="api_response",
        )

        issues = list(
            query_issues
        )

        if secret_detected:
            issues.append(
                "Secret-like information was "
                "detected in the final output."
            )

        if restricted_detected:
            issues.append(
                "Restricted personal information "
                "was detected in the final output."
            )

        if prompt_leakage:
            issues.append(
                "Internal prompt or instruction "
                "language was detected in the "
                "final output."
            )

        contains_unauthorized_fields = (
            not query_valid
        )

        if authoritative_decision is not None:
            decision = (
                authoritative_decision
            )

        elif (
            secret_detected
            or restricted_detected
            or prompt_leakage
            or contains_unauthorized_fields
        ):
            decision = SecurityDecision.BLOCK

        elif redactions:
            decision = (
                SecurityDecision
                .ALLOW_REDACTED
            )

        else:
            decision = SecurityDecision.ALLOW

        pii_result = (
            self.pii_detector.detect(
                self._payload_text(
                    raw_payload
                )
            )
        )

        if decision == SecurityDecision.BLOCK:
            output_payload = None

            user_message = (
                "The generated response was blocked "
                "because it contains restricted, "
                "unauthorized, or unsafe information."
            )

        elif decision == SecurityDecision.REVIEW:
            output_payload = (
                sanitized_payload
                if isinstance(
                    sanitized_payload,
                    dict,
                )
                else None
            )

            user_message = (
                "The generated response requires "
                "security or quality review before "
                "it can be released."
            )

        elif (
            decision
            == SecurityDecision
            .ALLOW_REDACTED
        ):
            output_payload = (
                sanitized_payload
                if isinstance(
                    sanitized_payload,
                    dict,
                )
                else None
            )

            user_message = (
                "The response passed final security "
                "screening after personal information "
                "was redacted."
            )

        else:
            output_payload = (
                sanitized_payload
                if isinstance(
                    sanitized_payload,
                    dict,
                )
                else None
            )

            user_message = (
                "The response passed final output "
                "security screening."
            )

        return OutputGuardResult(
            decision=decision,
            sanitized_payload=(
                output_payload
            ),
            redactions=redactions,
            pii_result=pii_result,
            contains_secret_data=(
                secret_detected
            ),
            contains_prompt_leakage=(
                prompt_leakage
            ),
            contains_unauthorized_fields=(
                contains_unauthorized_fields
            ),
            issues=list(
                dict.fromkeys(
                    issues
                )
            ),
            user_message=user_message,
        )

    def _payload_text(
        self,
        payload: dict[str, Any],
    ) -> str:
        """
        Convert output payload into text for one final privacy scan.
        """

        values: list[str] = []

        def collect(
            value: Any,
        ) -> None:
            if value is None:
                return

            if isinstance(value, str):
                values.append(value)
                return

            if isinstance(value, dict):
                for child_value in (
                    value.values()
                ):
                    collect(child_value)
                return

            if isinstance(
                value,
                (list, tuple, set),
            ):
                for child_value in value:
                    collect(child_value)
                return

            if isinstance(
                value,
                (int, float, bool),
            ):
                values.append(
                    str(value)
                )

        collect(payload)

        combined_text = " ".join(
            values
        ).strip()

        if not combined_text:
            return (
                "No sensitive output content."
            )

        return combined_text


# ---------------------------------------------------------------------
# Standalone smoke test
# ---------------------------------------------------------------------

def main() -> None:
    """
    Run an Ollama-independent smoke test of the final output guard.

    This test validates the guard itself with a known-safe payload. It
    does not depend on workflow or evaluator availability.
    """
    from models.schemas import (
        AnalysisResult,
        FinalReport,
        QueryExecutionResult,
    )
    from security.models import (
        DeterministicValidationResult,
        OutputEvaluation,
        SecurityDecision,
        UserRole,
    )

    print("=" * 80)
    print("FINAL OUTPUT GUARD SMOKE TEST")
    print("=" * 80)

    question = (
        "Show the top 5 products by revenue."
    )

    try:
        generated_sql = (
            "SELECT product, SUM(revenue) AS revenue "
            "FROM sales "
            "GROUP BY product "
            "ORDER BY revenue DESC "
            "LIMIT 5"
        )

        query_result = QueryExecutionResult(
            columns=[
                "product",
                "revenue",
            ],
            rows=[
                {
                    "product": "Product A",
                    "revenue": 100,
                }
            ],
            row_count=1,
            execution_time_ms=1.0,
            truncated=False,
            query=generated_sql,
        )

        analysis_result = AnalysisResult(
            summary=(
                "Product A has the highest revenue "
                "in the smoke-test result."
            ),
        )

        final_report = FinalReport(
            business_summary=(
                "Product A leads the revenue ranking "
                "with 100 in the smoke-test result."
            ),
            key_findings=[
                "Product A has revenue of 100."
            ],
            root_cause_indicators=[],
            recommendations=[],
            chart=None,
            limitations=[],
        )

        deterministic_result = (
            DeterministicValidationResult(
                valid=True,
                schema_valid=True,
                sql_safe=True,
                numerical_claims_valid=True,
                ranking_valid=True,
                chart_valid=True,
                pii_safe=True,
                secret_safe=True,
                prompt_leakage_safe=True,
                causal_language_safe=True,
                numeric_claims=[],
                verified_claims=[
                    "Product A has revenue of 100."
                ],
                unsupported_claims=[],
                privacy_findings=[],
                issues=[],
                decision="approve",
            )
        )

        evaluation = OutputEvaluation(
            answers_user_question=True,
            grounded_in_query_result=True,
            numerical_claims_supported=True,
            findings_supported=True,
            recommendations_supported=True,
            root_cause_language_cautious=True,
            chart_matches_result=True,
            contains_sensitive_data=False,
            contains_prompt_leakage=False,
            contains_secret_data=False,
            relevance_score=1.0,
            groundedness_score=1.0,
            accuracy_score=1.0,
            safety_score=1.0,
            overall_score=1.0,
            issues=[],
            decision="approve",
            explanation=(
                "The known-safe smoke-test output "
                "is grounded and contains no unsafe content."
            ),
        )

        guard = OutputGuard(
            maximum_output_rows=1000
        )

        result = guard.guard(
            final_report=final_report,
            query_result=query_result,
            deterministic_result=(
                deterministic_result
            ),
            output_evaluation=evaluation,
            role=UserRole.ANALYST,
            allowed_output_columns=set(
                query_result.columns
            ),
            technical_details={
                "generated_sql": (
                    generated_sql
                ),
                "referenced_tables": (
                    ["sales"]
                ),
                "sql_retry_count": (
                    0
                ),
                "max_sql_retries": (
                    3
                ),
                "execution_trace": (
                    []
                ),
            },
        )

        print(
            f"\nDeterministic decision : "
            f"{deterministic_result.decision}"
        )

        print(
            f"LLM evaluation decision: "
            f"{evaluation.decision}"
        )

        print(
            f"Output guard decision  : "
            f"{result.decision}"
        )

        print(
            f"Redactions             : "
            f"{len(result.redactions)}"
        )

        print(
            f"Contains secrets       : "
            f"{result.contains_secret_data}"
        )

        print(
            f"Prompt leakage         : "
            f"{result.contains_prompt_leakage}"
        )

        print(
            f"Unauthorized fields    : "
            f"{result.contains_unauthorized_fields}"
        )

        print(
            f"Payload available      : "
            f"{result.sanitized_payload is not None}"
        )

        print(
            f"User message           : "
            f"{result.user_message}"
        )

        if result.issues:
            print("\nIssues:")

            for issue in result.issues:
                print(f"- {issue}")

        print("\n" + "=" * 80)

        if str(result.decision) in {
            "allow",
            "allow_redacted",
        }:
            print(
                "RESULT: FINAL OUTPUT GUARD PASSED"
            )

        else:
            print(
                "RESULT: FINAL OUTPUT GUARD "
                "REQUIRES ACTION"
            )

        print("=" * 80)

    except Exception as exc:
        print("\n" + "=" * 80)
        print(
            "RESULT: FINAL OUTPUT GUARD FAILED"
        )
        print("=" * 80)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()