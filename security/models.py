"""
Shared security, privacy, authorization, and evaluation models.

These Pydantic models define structured contracts for:

1. PII and confidential-data detection
2. Prompt-injection assessment
3. Input-query guardrails
4. Database schema-access policy
5. SQL-policy validation
6. Deterministic output validation
7. LLM-based output evaluation
8. Final output redaction
9. Tool authorization
10. Security audit events

This file contains no detection logic, LLM calls, database access,
tool execution, or business-specific outputs.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


# ---------------------------------------------------------------------
# Shared base model
# ---------------------------------------------------------------------

class SecurityModel(BaseModel):
    """
    Base model for security and evaluation contracts.

    Unknown fields are rejected so security decisions cannot silently
    contain unsupported attributes.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
        use_enum_values=True,
    )


# ---------------------------------------------------------------------
# Shared enums
# ---------------------------------------------------------------------

class DataClassification(str, Enum):
    """
    Sensitivity classification for input, database, and output data.
    """

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    PERSONAL = "personal"
    RESTRICTED = "restricted"
    SECRET = "secret"


class UserRole(str, Enum):
    """
    Supported application roles.

    None of these roles receives database write permission.
    """

    VIEWER = "viewer"
    ANALYST = "analyst"
    ADMIN = "admin"


class SecurityDecision(str, Enum):
    """
    General security decision.
    """

    ALLOW = "allow"
    ALLOW_REDACTED = "allow_redacted"
    CLARIFY = "clarify"
    BLOCK = "block"
    REVIEW = "review"


class EvaluationDecision(str, Enum):
    """
    Decision returned by output validators.
    """

    APPROVE = "approve"
    REGENERATE = "regenerate"
    BLOCK = "block"
    REVIEW = "review"


class ToolDecision(str, Enum):
    """
    Tool authorization result.
    """

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class RiskLevel(str, Enum):
    """
    General risk severity.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PIIAction(str, Enum):
    """
    Action for one detected sensitive entity.
    """

    ALLOW = "allow"
    REDACT = "redact"
    BLOCK = "block"
    REVIEW = "review"


class SQLAccessType(str, Enum):
    """
    How a column is used by generated SQL.
    """

    SELECT = "select"
    FILTER = "filter"
    GROUP_BY = "group_by"
    ORDER_BY = "order_by"
    JOIN = "join"
    AGGREGATE = "aggregate"


# ---------------------------------------------------------------------
# PII and confidential-data models
# ---------------------------------------------------------------------

class PIIFinding(SecurityModel):
    """
    One personal, confidential, restricted, or secret entity detected
    in user input or generated output.

    raw_value is intentionally not stored to reduce accidental
    disclosure through logs or API responses.
    """

    entity_type: str = Field(
        min_length=1,
        description=(
            "Detected entity category, such as email_address, "
            "phone_number, aadhaar_number, pan_number, password, "
            "api_key, access_token, or bank_account."
        ),
    )

    classification: DataClassification = Field(
        description="Sensitivity classification.",
    )

    action: PIIAction = Field(
        description="Required treatment for this entity.",
    )

    start_index: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Start position in the inspected text, when available."
        ),
    )

    end_index: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "End position in the inspected text, when available."
        ),
    )

    redacted_value: str = Field(
        min_length=1,
        description=(
            "Safe replacement such as [EMAIL_REDACTED]."
        ),
    )

    confidence: float = Field(
        default=1.0,
        ge=0,
        le=1,
        description="Detector confidence.",
    )

    reason: str = Field(
        min_length=1,
        description=(
            "Safe explanation without reproducing the sensitive value."
        ),
    )

    @model_validator(mode="after")
    def validate_location(
        self,
    ) -> "PIIFinding":
        """
        Validate optional text offsets.
        """

        if (
            self.start_index is None
            and self.end_index is not None
        ):
            raise ValueError(
                "start_index is required when end_index is provided."
            )

        if (
            self.start_index is not None
            and self.end_index is None
        ):
            raise ValueError(
                "end_index is required when start_index is provided."
            )

        if (
            self.start_index is not None
            and self.end_index is not None
            and self.end_index <= self.start_index
        ):
            raise ValueError(
                "end_index must be greater than start_index."
            )

        return self


class PIIDetectionResult(SecurityModel):
    """
    Result of scanning input or output for sensitive information.
    """

    contains_sensitive_data: bool = Field(
        description=(
            "Whether personal, confidential, restricted, or secret "
            "data was detected."
        ),
    )

    highest_classification: Optional[
        DataClassification
    ] = Field(
        default=None,
        description=(
            "Highest sensitivity level among detected entities."
        ),
    )

    findings: list[PIIFinding] = Field(
        default_factory=list,
        description="Detected sensitive entities.",
    )

    redacted_text: str = Field(
        description=(
            "Text after applying required safe replacements."
        ),
    )

    decision: SecurityDecision = Field(
        description="Overall treatment decision.",
    )

    user_message: str = Field(
        min_length=1,
        description=(
            "Safe explanation suitable for the user."
        ),
    )

    @model_validator(mode="after")
    def validate_detection_state(
        self,
    ) -> "PIIDetectionResult":
        """
        Ensure sensitive-data fields agree.
        """

        if self.contains_sensitive_data and not self.findings:
            raise ValueError(
                "findings are required when sensitive data is detected."
            )

        if (
            not self.contains_sensitive_data
            and self.findings
        ):
            raise ValueError(
                "findings must be empty when no sensitive data is "
                "detected."
            )

        if (
            self.contains_sensitive_data
            and self.highest_classification is None
        ):
            raise ValueError(
                "highest_classification is required when sensitive "
                "data is detected."
            )

        if (
            not self.contains_sensitive_data
            and self.highest_classification is not None
        ):
            raise ValueError(
                "highest_classification must be None when no sensitive "
                "data is detected."
            )

        return self


# ---------------------------------------------------------------------
# Prompt-injection assessment
# ---------------------------------------------------------------------

class PromptRiskAssessment(SecurityModel):
    """
    Structured semantic assessment of prompt-injection and unsafe
    instruction risk.
    """

    prompt_injection_detected: bool = False
    instruction_override_attempt: bool = False
    system_prompt_extraction_attempt: bool = False
    unsafe_database_action: bool = False
    sql_validation_bypass_attempt: bool = False
    sensitive_data_request: bool = False
    secret_extraction_attempt: bool = False
    filesystem_access_attempt: bool = False
    environment_access_attempt: bool = False
    unapproved_tool_request: bool = False
    privilege_escalation_attempt: bool = False
    arbitrary_code_execution_attempt: bool = False

    risk_level: RiskLevel = Field(
        description="Highest detected risk level.",
    )

    decision: Literal[
        "allow",
        "block",
        "review",
    ] = Field(
        description="Prompt-risk decision.",
    )

    reason: str = Field(
        min_length=1,
        description=(
            "Safe explanation without revealing internal prompts."
        ),
    )

    confidence: float = Field(
        ge=0,
        le=1,
        description="Classification confidence.",
    )


# ---------------------------------------------------------------------
# Input guard models
# ---------------------------------------------------------------------

class UnsafeOperationFinding(SecurityModel):
    """
    One unsafe action detected in the user's requested intent.

    matched_text must contain only a safe operation label or a short
    sanitized fragment. It must not contain a secret.
    """

    operation: str = Field(
        min_length=1,
        description=(
            "Unsafe operation category, such as delete, update, "
            "insert, drop, write_file, run_shell, or reveal_secret."
        ),
    )

    category: Literal[
        "database_modification",
        "validation_bypass",
        "prompt_injection",
        "secret_extraction",
        "personal_data_extraction",
        "filesystem_access",
        "environment_access",
        "arbitrary_code_execution",
        "unapproved_tool",
        "privilege_escalation",
        "other",
    ]

    matched_text: Optional[str] = Field(
        default=None,
        description=(
            "Short sanitized fragment that triggered detection."
        ),
    )

    severity: RiskLevel

    reason: str = Field(
        min_length=1,
        description="Safe reason for the finding.",
    )


class InputGuardResult(SecurityModel):
    """
    Final decision produced before a question reaches Agent 1.
    """

    decision: SecurityDecision

    original_question_hash: str = Field(
        min_length=1,
        description=(
            "One-way hash of the original input for audit correlation."
        ),
    )

    sanitized_question: Optional[str] = Field(
        default=None,
        description=(
            "Question after permitted redaction. None when blocked."
        ),
    )

    risk_level: RiskLevel

    risk_categories: list[str] = Field(
        default_factory=list,
        description="Detected security-risk categories.",
    )

    unsafe_operations: list[
        UnsafeOperationFinding
    ] = Field(
        default_factory=list,
    )

    pii_result: PIIDetectionResult

    prompt_risk: Optional[
        PromptRiskAssessment
    ] = None

    user_message: str = Field(
        min_length=1,
        description="Safe message returned to the user.",
    )

    requires_human_review: bool = False

    @model_validator(mode="after")
    def validate_guard_result(
        self,
    ) -> "InputGuardResult":
        """
        Validate relationships between guard fields.
        """

        if self.decision == "block":
            if self.sanitized_question is not None:
                raise ValueError(
                    "sanitized_question must be None for blocked input."
                )

        elif not self.sanitized_question:
            raise ValueError(
                "sanitized_question is required for non-blocked input."
            )

        if (
            self.decision == "review"
            and not self.requires_human_review
        ):
            raise ValueError(
                "requires_human_review must be True when decision is "
                "review."
            )

        if (
            self.requires_human_review
            and self.decision != "review"
        ):
            raise ValueError(
                "decision must be review when human review is required."
            )

        return self


# ---------------------------------------------------------------------
# Database schema and column policy
# ---------------------------------------------------------------------

class ColumnPolicy(SecurityModel):
    """
    Access policy for one database column.
    """

    table_name: str = Field(
        min_length=1,
    )

    column_name: str = Field(
        min_length=1,
    )

    classification: DataClassification

    allowed_roles: set[UserRole] = Field(
        default_factory=set,
    )

    allow_selection: bool = True
    allow_filtering: bool = True
    allow_grouping: bool = True
    allow_ordering: bool = True
    allow_joining: bool = True
    allow_aggregation: bool = True

    require_aggregation: bool = False
    redact_output: bool = False

    maximum_distinct_values: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Optional maximum number of distinct values that may be "
            "returned for this field."
        ),
    )


class TablePolicy(SecurityModel):
    """
    Access policy for one table or view.
    """

    table_name: str = Field(
        min_length=1,
    )

    allowed_roles: set[UserRole] = Field(
        default_factory=set,
    )

    allow_access: bool = True
    allow_row_level_output: bool = True
    require_aggregation: bool = False

    maximum_rows: Optional[int] = Field(
        default=None,
        ge=1,
    )

    columns: list[ColumnPolicy] = Field(
        default_factory=list,
    )


class SQLColumnReference(SecurityModel):
    """
    One table-and-column reference found in generated SQL.
    """

    table_name: Optional[str] = None

    column_name: str = Field(
        min_length=1,
    )

    access_types: set[SQLAccessType] = Field(
        default_factory=set,
    )

    is_aggregated: bool = False
    output_alias: Optional[str] = None


class SchemaPolicyFinding(SecurityModel):
    """
    One schema-authorization violation.
    """

    table_name: Optional[str] = None
    column_name: Optional[str] = None

    classification: Optional[
        DataClassification
    ] = None

    violation_type: Literal[
        "table_not_allowed",
        "column_not_allowed",
        "role_not_allowed",
        "row_level_access_not_allowed",
        "aggregation_required",
        "filtering_not_allowed",
        "grouping_not_allowed",
        "ordering_not_allowed",
        "joining_not_allowed",
        "selection_not_allowed",
        "result_limit_exceeded",
        "unknown_table",
        "unknown_column",
        "other",
    ]

    severity: RiskLevel

    message: str = Field(
        min_length=1,
        description=(
            "Safe policy explanation without exposing confidential "
            "schema information."
        ),
    )


class SchemaPolicyResult(SecurityModel):
    """
    Final schema-access decision for generated SQL.
    """

    allowed: bool
    role: UserRole

    referenced_tables: list[str] = Field(
        default_factory=list,
    )

    referenced_columns: list[
        SQLColumnReference
    ] = Field(
        default_factory=list,
    )

    findings: list[
        SchemaPolicyFinding
    ] = Field(
        default_factory=list,
    )

    redacted_output_columns: list[str] = Field(
        default_factory=list,
    )

    requires_aggregation: bool = False

    decision: Literal[
        "allow",
        "block",
        "review",
    ]

    user_message: str = Field(
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_policy_decision(
        self,
    ) -> "SchemaPolicyResult":
        """
        Ensure allowed and decision agree.
        """

        if self.allowed and self.decision != "allow":
            raise ValueError(
                "allowed=True requires decision='allow'."
            )

        if (
            not self.allowed
            and self.decision == "allow"
        ):
            raise ValueError(
                "decision='allow' requires allowed=True."
            )

        if (
            self.decision == "block"
            and not self.findings
        ):
            raise ValueError(
                "Blocked schema access requires at least one finding."
            )

        return self


# ---------------------------------------------------------------------
# Deterministic output-validation models
# ---------------------------------------------------------------------

class NumericClaimValidation(SecurityModel):
    """
    Validation result for one numerical claim in the generated report.
    """

    claim: str = Field(
        min_length=1,
    )

    metric_name: Optional[str] = None

    claimed_value: Optional[float] = None
    verified_value: Optional[float] = None

    tolerance: float = Field(
        default=0.01,
        ge=0,
    )

    supported: bool

    evidence: list[str] = Field(
        default_factory=list,
    )

    issue: Optional[str] = None


class DeterministicValidationResult(SecurityModel):
    """
    Objective validation result for Agent 4 and Agent 5 outputs.
    """

    valid: bool

    schema_valid: bool
    sql_safe: bool
    numerical_claims_valid: bool
    ranking_valid: bool
    chart_valid: bool
    pii_safe: bool
    secret_safe: bool
    prompt_leakage_safe: bool
    causal_language_safe: bool

    numeric_claims: list[
        NumericClaimValidation
    ] = Field(
        default_factory=list,
    )

    verified_claims: list[str] = Field(
        default_factory=list,
    )

    unsupported_claims: list[str] = Field(
        default_factory=list,
    )

    privacy_findings: list[
        PIIFinding
    ] = Field(
        default_factory=list,
    )

    issues: list[str] = Field(
        default_factory=list,
    )

    decision: EvaluationDecision

    @model_validator(mode="after")
    def validate_deterministic_result(
        self,
    ) -> "DeterministicValidationResult":
        """
        Ensure approval is consistent with validation fields.
        """

        safety_fields = [
            self.schema_valid,
            self.sql_safe,
            self.numerical_claims_valid,
            self.ranking_valid,
            self.chart_valid,
            self.pii_safe,
            self.secret_safe,
            self.prompt_leakage_safe,
            self.causal_language_safe,
        ]

        if (
            self.decision == "approve"
            and not all(safety_fields)
        ):
            raise ValueError(
                "decision='approve' requires all validation checks "
                "to pass."
            )

        if self.valid != all(safety_fields):
            raise ValueError(
                "valid must reflect all deterministic validation "
                "checks."
            )

        return self


# ---------------------------------------------------------------------
# LLM output-evaluation models
# ---------------------------------------------------------------------

class OutputEvaluation(SecurityModel):
    """
    Structured LLM evaluation of the final analytical response.

    This evaluation complements deterministic checks. It does not
    override SQL, privacy, or authorization enforcement.
    """

    answers_user_question: bool
    grounded_in_query_result: bool
    numerical_claims_supported: bool
    findings_supported: bool
    recommendations_supported: bool
    root_cause_language_cautious: bool
    chart_matches_result: bool

    contains_sensitive_data: bool
    contains_prompt_leakage: bool
    contains_secret_data: bool

    relevance_score: float = Field(
        ge=0,
        le=1,
    )

    groundedness_score: float = Field(
        ge=0,
        le=1,
    )

    accuracy_score: float = Field(
        ge=0,
        le=1,
    )

    safety_score: float = Field(
        ge=0,
        le=1,
    )

    overall_score: float = Field(
        ge=0,
        le=1,
    )

    issues: list[str] = Field(
        default_factory=list,
    )

    decision: EvaluationDecision

    explanation: str = Field(
        min_length=1,
        description=(
            "Concise explanation grounded in supplied evidence."
        ),
    )

    @model_validator(mode="after")
    def validate_evaluation_decision(
        self,
    ) -> "OutputEvaluation":
        """
        Prevent approval when critical safety checks fail.
        """

        if self.decision == "approve":
            if self.contains_sensitive_data:
                raise ValueError(
                    "Sensitive output cannot be approved."
                )

            if self.contains_prompt_leakage:
                raise ValueError(
                    "Prompt leakage cannot be approved."
                )

            if self.contains_secret_data:
                raise ValueError(
                    "Secret data cannot be approved."
                )

            if not self.grounded_in_query_result:
                raise ValueError(
                    "Ungrounded output cannot be approved."
                )

            if not self.numerical_claims_supported:
                raise ValueError(
                    "Unsupported numerical claims cannot be approved."
                )

        return self


# ---------------------------------------------------------------------
# Predefined benchmark models
# ---------------------------------------------------------------------

class SQLBenchmarkRequirements(SecurityModel):
    """
    Expected structural SQL properties for one predefined example.
    """

    read_only: bool = True
    requires_group_by: bool = False
    requires_order_by: bool = False

    sort_direction: Optional[
        Literal[
            "ascending",
            "descending",
        ]
    ] = None

    maximum_rows: Optional[int] = Field(
        default=None,
        ge=1,
    )

    required_tables: list[str] = Field(
        default_factory=list,
    )

    forbidden_tables: list[str] = Field(
        default_factory=list,
    )

    required_operations: list[str] = Field(
        default_factory=list,
    )


class BenchmarkExample(SecurityModel):
    """
    One predefined workflow benchmark.

    Benchmarks compare expected properties rather than exact LLM text.
    """

    test_id: str = Field(
        min_length=1,
    )

    question: str = Field(
        min_length=3,
    )

    expected_final_status: Literal[
        "completed",
        "clarification_required",
        "security_blocked",
        "security_review_required",
        "output_blocked",
        "output_review_required",
        "failed",
    ]

    expected_analysis_types: list[str] = Field(
        default_factory=list,
    )

    required_metrics: list[str] = Field(
        default_factory=list,
    )

    required_dimensions: list[str] = Field(
        default_factory=list,
    )

    sql_requirements: Optional[
        SQLBenchmarkRequirements
    ] = None

    expected_chart_types: list[str] = Field(
        default_factory=list,
    )

    required_report_sections: list[str] = Field(
        default_factory=list,
    )

    forbidden_data_types: list[str] = Field(
        default_factory=list,
    )

    expected_security_decision: SecurityDecision = (
        SecurityDecision.ALLOW
    )


class BenchmarkResult(SecurityModel):
    """
    Result of running one predefined benchmark.
    """

    test_id: str = Field(
        min_length=1,
    )

    passed: bool

    final_status: str = Field(
        min_length=1,
    )

    checks_passed: list[str] = Field(
        default_factory=list,
    )

    checks_failed: list[str] = Field(
        default_factory=list,
    )

    execution_time_ms: float = Field(
        ge=0,
    )

    security_decision: Optional[
        SecurityDecision
    ] = None

    deterministic_evaluation: Optional[
        DeterministicValidationResult
    ] = None

    llm_evaluation: Optional[
        OutputEvaluation
    ] = None


# ---------------------------------------------------------------------
# Output redaction and final guard models
# ---------------------------------------------------------------------

class RedactionRecord(SecurityModel):
    """
    Safe record of one output redaction.

    original_value is intentionally not stored.
    """

    entity_type: str = Field(
        min_length=1,
    )

    replacement: str = Field(
        min_length=1,
    )

    location: Optional[str] = None

    reason: str = Field(
        min_length=1,
    )


class OutputGuardResult(SecurityModel):
    """
    Final output-security decision before API serialization.
    """

    decision: SecurityDecision

    sanitized_payload: Optional[
        dict[str, Any]
    ] = None

    redactions: list[
        RedactionRecord
    ] = Field(
        default_factory=list,
    )

    pii_result: PIIDetectionResult

    contains_secret_data: bool = False
    contains_prompt_leakage: bool = False
    contains_unauthorized_fields: bool = False

    issues: list[str] = Field(
        default_factory=list,
    )

    user_message: str = Field(
        min_length=1,
    )

    @model_validator(mode="after")
    def validate_output_guard(
        self,
    ) -> "OutputGuardResult":
        """
        Validate final payload availability.
        """

        if self.decision == "block":
            if self.sanitized_payload is not None:
                raise ValueError(
                    "Blocked output must not contain sanitized_payload."
                )

        elif self.sanitized_payload is None:
            raise ValueError(
                "Non-blocked output requires sanitized_payload."
            )

        return self


# ---------------------------------------------------------------------
# Tool authorization models
# ---------------------------------------------------------------------

class ToolPermission(SecurityModel):
    """
    Policy for one approved application tool.
    """

    tool_name: str = Field(
        min_length=1,
    )

    description: str = Field(
        min_length=1,
    )

    allowed_roles: set[UserRole] = Field(
        default_factory=set,
    )

    read_only: bool = True

    requires_approval: bool = False

    maximum_calls_per_request: int = Field(
        default=1,
        ge=1,
        le=20,
    )

    allowed_argument_names: set[str] = Field(
        default_factory=set,
    )

    blocked_argument_names: set[str] = Field(
        default_factory=set,
    )


class ToolAuthorizationRequest(SecurityModel):
    """
    Authorization request before invoking one tool.
    """

    tool_name: str = Field(
        min_length=1,
    )

    role: UserRole

    arguments: dict[str, Any] = Field(
        default_factory=dict,
    )

    current_call_count: int = Field(
        default=0,
        ge=0,
    )

    request_id: str = Field(
        min_length=1,
    )


class ToolAuthorizationResult(SecurityModel):
    """
    Tool permission decision.
    """

    decision: ToolDecision

    tool_name: str = Field(
        min_length=1,
    )

    role: UserRole

    allowed_arguments: dict[str, Any] = Field(
        default_factory=dict,
    )

    removed_arguments: list[str] = Field(
        default_factory=list,
    )

    reason: str = Field(
        min_length=1,
    )


# ---------------------------------------------------------------------
# Secure audit-event model
# ---------------------------------------------------------------------

class SecurityAuditEvent(SecurityModel):
    """
    Sanitized event for security and workflow audit logging.

    Raw questions, PII, passwords, tokens, prompts, schemas, and result
    rows must not be stored in this event.
    """

    event_id: str = Field(
        min_length=1,
    )

    request_id: str = Field(
        min_length=1,
    )

    timestamp_utc: datetime = Field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ),
    )

    event_type: Literal[
        "input_guard",
        "pii_detection",
        "prompt_injection",
        "authorization",
        "sql_policy",
        "tool_authorization",
        "deterministic_evaluation",
        "llm_evaluation",
        "output_guard",
        "workflow_completion",
        "workflow_failure",
    ]

    user_id_hash: Optional[str] = None
    question_hash: Optional[str] = None

    role: Optional[UserRole] = None

    decision: str = Field(
        min_length=1,
    )

    risk_level: Optional[
        RiskLevel
    ] = None

    workflow_status: Optional[str] = None
    current_node: Optional[str] = None

    sql_retry_count: int = Field(
        default=0,
        ge=0,
    )

    row_count: Optional[int] = Field(
        default=None,
        ge=0,
    )

    execution_time_ms: Optional[float] = Field(
        default=None,
        ge=0,
    )

    categories: list[str] = Field(
        default_factory=list,
    )

    safe_message: str = Field(
        min_length=1,
    )