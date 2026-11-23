"""
Pydantic request and response schemas for the FastAPI backend.

These models define the public API contract between:

1. Streamlit frontend
2. FastAPI backend
3. LangGraph workflow

Supported workflow outcomes:

- completed
- clarification_required
- security_blocked
- security_review_required
- output_blocked
- output_review_required
- failed

Security design:

- Raw PII findings are not returned.
- Original unredacted questions are not required in blocked responses.
- Internal prompts, full schemas, secrets, and guard patterns are not
  included in public API contracts.
- Output-quality metadata contains decisions and scores only.
- Unknown fields are rejected.

This file does not:

- Start Ollama
- Run LangGraph
- Generate SQL
- Execute SQL
- Access the database
- Contain workflow business logic
"""

from typing import (
    Any,
    Literal,
    Optional,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


# ---------------------------------------------------------------------
# Shared API literals
# ---------------------------------------------------------------------

APIStatus = Literal[
    "completed",
    "clarification_required",
    "security_blocked",
    "security_review_required",
    "output_blocked",
    "output_review_required",
    "failed",
]


SecurityDecisionValue = Literal[
    "pending",
    "allow",
    "allow_redacted",
    "clarify",
    "block",
    "review",
]


RiskLevelValue = Literal[
    "low",
    "medium",
    "high",
    "critical",
]


EvaluationDecisionValue = Literal[
    "approve",
    "regenerate",
    "block",
    "review",
]


# ---------------------------------------------------------------------
# Shared base schema
# ---------------------------------------------------------------------

class APISchema(BaseModel):
    """
    Base model for all API request and response schemas.

    Unknown fields are rejected so clients cannot silently send or
    receive unsupported properties.
    """

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


# ---------------------------------------------------------------------
# Analytics request
# ---------------------------------------------------------------------

class AnalyticsQueryRequest(APISchema):
    """
    Request body for the analytics workflow endpoint.

    Example:

        {
            "question": "Show the top 5 products by revenue.",
            "max_sql_retries": 3,
            "include_technical_details": true
        }
    """

    question: str = Field(
        min_length=3,
        max_length=2000,
        description=(
            "Natural-language business question submitted "
            "by the user."
        ),
        examples=[
            "Show the top 5 products by revenue."
        ],
    )

    max_sql_retries: int = Field(
        default=3,
        ge=0,
        le=5,
        description=(
            "Maximum number of SQL correction attempts allowed "
            "during one workflow execution."
        ),
    )

    include_technical_details: bool = Field(
        default=False,
        description=(
            "Whether the client requests generated SQL, plan, intent, "
            "and trace information. Backend authorization and output "
            "security remain authoritative."
        ),
    )

    @model_validator(mode="after")
    def validate_question(
        self,
    ) -> "AnalyticsQueryRequest":
        """
        Reject punctuation-only or whitespace-only questions.
        """

        meaningful_characters = [
            character
            for character in self.question
            if character.isalnum()
        ]

        if not meaningful_characters:
            raise ValueError(
                "question must contain meaningful "
                "letters or numbers."
            )

        return self


# ---------------------------------------------------------------------
# Error and security response details
# ---------------------------------------------------------------------

class APIErrorDetail(APISchema):
    """
    Structured technical error information.

    This model is used only when status='failed'.
    """

    error_type: str = Field(
        min_length=1,
        description=(
            "Safe application or workflow error category."
        ),
    )

    message: str = Field(
        min_length=1,
        description=(
            "Safe error message suitable for an API client."
        ),
    )

    retryable: bool = Field(
        default=False,
        description=(
            "Whether the same request may reasonably be retried."
        ),
    )

    stage: Optional[str] = Field(
        default=None,
        description=(
            "Safe workflow stage where the failure occurred."
        ),
    )


class SecurityResponse(APISchema):
    """
    Safe summary of a security decision.

    This response deliberately excludes:

    - Raw matched text
    - Sensitive values
    - PII offsets
    - Detection regular expressions
    - Internal prompts
    - Full model reasoning
    """

    decision: SecurityDecisionValue = Field(
        description=(
            "Final input or output security decision."
        ),
    )

    risk_level: Optional[
        RiskLevelValue
    ] = Field(
        default=None,
        description=(
            "Highest safe risk classification available."
        ),
    )

    risk_categories: list[str] = Field(
        default_factory=list,
        description=(
            "Sanitized high-level risk categories, such as "
            "database_modification or sensitive_data."
        ),
    )

    message: str = Field(
        min_length=1,
        description=(
            "Safe user-facing explanation of the security outcome."
        ),
    )

    question_was_redacted: bool = Field(
        default=False,
        description=(
            "Whether personal information was removed before "
            "downstream processing."
        ),
    )

    requires_review: bool = Field(
        default=False,
        description=(
            "Whether controlled human or administrative review "
            "is required."
        ),
    )

    @model_validator(mode="after")
    def validate_security_decision(
        self,
    ) -> "SecurityResponse":
        """
        Ensure decision and review fields agree.
        """

        if (
            self.decision == "review"
            and not self.requires_review
        ):
            raise ValueError(
                "requires_review must be True when "
                "decision is review."
            )

        if (
            self.requires_review
            and self.decision != "review"
        ):
            raise ValueError(
                "decision must be review when "
                "requires_review is True."
            )

        return self


class OutputQualityResponse(APISchema):
    """
    Safe summary of output-validation and evaluation results.

    Detailed unsupported claims, PII matches, prompts, and internal
    evaluator instructions are not exposed through this model.
    """

    deterministic_decision: Optional[
        EvaluationDecisionValue
    ] = Field(
        default=None,
        description=(
            "Decision from deterministic output validation."
        ),
    )

    llm_evaluation_decision: Optional[
        EvaluationDecisionValue
    ] = Field(
        default=None,
        description=(
            "Decision from the structured LLM evaluator."
        ),
    )

    output_guard_decision: Optional[
        SecurityDecisionValue
    ] = Field(
        default=None,
        description=(
            "Decision from final output security screening."
        ),
    )

    deterministic_valid: Optional[bool] = Field(
        default=None,
        description=(
            "Whether all deterministic output checks passed."
        ),
    )

    numerical_claims_valid: Optional[bool] = Field(
        default=None,
        description=(
            "Whether numerical claims were supported by query evidence."
        ),
    )

    ranking_valid: Optional[bool] = Field(
        default=None,
        description=(
            "Whether ranking claims matched returned row ordering."
        ),
    )

    chart_valid: Optional[bool] = Field(
        default=None,
        description=(
            "Whether the chart specification matched actual results."
        ),
    )

    pii_safe: Optional[bool] = Field(
        default=None,
        description=(
            "Whether deterministic privacy checks passed."
        ),
    )

    secret_safe: Optional[bool] = Field(
        default=None,
        description=(
            "Whether secret scanning passed."
        ),
    )

    prompt_leakage_safe: Optional[bool] = Field(
        default=None,
        description=(
            "Whether prompt-leakage scanning passed."
        ),
    )

    relevance_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )

    groundedness_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )

    accuracy_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )

    safety_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )

    overall_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )

    redaction_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of permitted output redactions."
        ),
    )

    sanitized_payload_available: bool = Field(
        default=False,
        description=(
            "Whether the final output guard produced a safe payload."
        ),
    )

    message: Optional[str] = Field(
        default=None,
        description=(
            "Safe summary of the output-quality outcome."
        ),
    )


# ---------------------------------------------------------------------
# Intent and planning details
# ---------------------------------------------------------------------

class IntentSummaryResponse(APISchema):
    """
    Safe summary of Agent 1 output.
    """

    analysis_type: str = Field(
        min_length=1,
        description=(
            "Primary analytical request type."
        ),
    )

    metrics: list[str] = Field(
        default_factory=list,
        description=(
            "Business metrics detected in the question."
        ),
    )

    dimensions: list[str] = Field(
        default_factory=list,
        description=(
            "Grouping or comparison dimensions."
        ),
    )

    filters: list[
        dict[str, Any]
    ] = Field(
        default_factory=list,
        description=(
            "Structured and sanitized filters extracted "
            "from the request."
        ),
    )

    sort_order: Optional[str] = Field(
        default=None,
        description=(
            "Requested sorting direction."
        ),
    )

    limit: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Requested maximum result count."
        ),
    )


class PlanStepResponse(APISchema):
    """
    Safe summary of one Planning Agent step.
    """

    step_id: int = Field(
        ge=1,
        description=(
            "Sequential analytical step identifier."
        ),
    )

    title: str = Field(
        min_length=1,
        description="Short step title.",
    )

    description: str = Field(
        min_length=1,
        description="Step purpose.",
    )

    step_type: str = Field(
        min_length=1,
        description=(
            "Analytical operation type."
        ),
    )

    depends_on: list[int] = Field(
        default_factory=list,
        description=(
            "Earlier plan steps required by this step."
        ),
    )

    requires_sql: bool = Field(
        description=(
            "Whether the step requires database access."
        ),
    )

    status: str = Field(
        min_length=1,
        description=(
            "Current plan-step status."
        ),
    )


class PlanSummaryResponse(APISchema):
    """
    Safe summary of Agent 2 output.
    """

    objective: str = Field(
        min_length=1,
        description=(
            "Overall analytical objective."
        ),
    )

    strategy: str = Field(
        min_length=1,
        description=(
            "Analytical execution strategy."
        ),
    )

    steps: list[
        PlanStepResponse
    ] = Field(
        default_factory=list,
        description=(
            "Ordered analytical plan steps."
        ),
    )


# ---------------------------------------------------------------------
# Business-report response details
# ---------------------------------------------------------------------

class RecommendationResponse(APISchema):
    """
    One business recommendation returned by Agent 5.
    """

    action: str = Field(
        min_length=1,
        description=(
            "Recommended business action."
        ),
    )

    rationale: str = Field(
        min_length=1,
        description=(
            "Evidence-based reason for the action."
        ),
    )

    priority: Literal[
        "low",
        "medium",
        "high",
    ] = Field(
        description=(
            "Recommendation priority."
        ),
    )


class ChartResponse(APISchema):
    """
    Validated chart specification returned to Streamlit.
    """

    chart_type: Literal[
        "line",
        "bar",
        "horizontal_bar",
        "area",
        "scatter",
        "pie",
        "table",
        "none",
    ] = Field(
        description=(
            "Requested visualization type."
        ),
    )

    title: str = Field(
        min_length=1,
        description=(
            "Business-friendly chart title."
        ),
    )

    x_column: Optional[str] = Field(
        default=None,
        description=(
            "Query-result column used on the X-axis."
        ),
    )

    y_columns: list[str] = Field(
        default_factory=list,
        description=(
            "Numeric query-result columns used on the Y-axis."
        ),
    )

    category_column: Optional[str] = Field(
        default=None,
        description=(
            "Optional grouping or color column."
        ),
    )

    rationale: str = Field(
        min_length=1,
        description=(
            "Reason for choosing this chart."
        ),
    )


class QueryResultResponse(APISchema):
    """
    Sanitized query-result data returned to the frontend.
    """

    columns: list[str] = Field(
        default_factory=list,
        description=(
            "Authorized columns returned by SQLite."
        ),
    )

    rows: list[
        dict[str, Any]
    ] = Field(
        default_factory=list,
        description=(
            "Sanitized rows returned by the approved read-only query."
        ),
    )

    row_count: int = Field(
        ge=0,
        description=(
            "Number of rows included in this response."
        ),
    )

    execution_time_ms: float = Field(
        ge=0,
        description=(
            "SQLite execution time in milliseconds."
        ),
    )

    truncated: bool = Field(
        description=(
            "Whether database output exceeded the configured row limit."
        ),
    )

    @model_validator(mode="after")
    def validate_result_shape(
        self,
    ) -> "QueryResultResponse":
        """
        Ensure row_count and declared columns agree with included rows.
        """

        if self.row_count != len(
            self.rows
        ):
            raise ValueError(
                "row_count must equal the number "
                "of rows included in the response."
            )

        available_columns = set(
            self.columns
        )

        for row_number, row in enumerate(
            self.rows,
            start=1,
        ):
            unexpected_columns = (
                set(row.keys())
                - available_columns
            )

            if unexpected_columns:
                raise ValueError(
                    f"Result row {row_number} contains "
                    "columns not declared in columns: "
                    f"{sorted(unexpected_columns)}"
                )

        return self


# ---------------------------------------------------------------------
# Technical-details response
# ---------------------------------------------------------------------

class TraceEventResponse(APISchema):
    """
    Sanitized LangGraph execution-trace event.
    """

    node: str = Field(
        min_length=1,
        description=(
            "Workflow node name."
        ),
    )

    status: str = Field(
        min_length=1,
        description=(
            "Node execution status."
        ),
    )

    message: str = Field(
        min_length=1,
        max_length=500,
        description=(
            "Sanitized execution message."
        ),
    )


class TechnicalDetailsResponse(APISchema):
    """
    Optional, role-filtered technical information.

    Authorization and OutputGuard decide whether these fields may be
    returned. Viewer responses should normally receive None.
    """

    generated_sql: Optional[str] = Field(
        default=None,
        description=(
            "Final validated and executed read-only SQL."
        ),
    )

    referenced_tables: list[str] = Field(
        default_factory=list,
        description=(
            "Approved tables referenced by the generated SQL."
        ),
    )

    sql_retry_count: int = Field(
        default=0,
        ge=0,
        description=(
            "Number of SQL retry attempts used."
        ),
    )

    max_sql_retries: int = Field(
        default=0,
        ge=0,
        description=(
            "Configured SQL retry limit."
        ),
    )

    intent: Optional[
        IntentSummaryResponse
    ] = Field(
        default=None,
        description=(
            "Safe summary of Agent 1 output."
        ),
    )

    plan: Optional[
        PlanSummaryResponse
    ] = Field(
        default=None,
        description=(
            "Safe summary of Agent 2 output."
        ),
    )

    execution_trace: list[
        TraceEventResponse
    ] = Field(
        default_factory=list,
        description=(
            "Sanitized LangGraph execution trace."
        ),
    )


# ---------------------------------------------------------------------
# Main analytics response
# ---------------------------------------------------------------------

class AnalyticsQueryResponse(APISchema):
    """
    Public response returned by the analytics endpoint.

    This single contract supports successful, clarification, controlled
    security, output-review, and technical-failure outcomes.
    """

    status: APIStatus = Field(
        description=(
            "Final controlled workflow status."
        ),
    )

    question: str = Field(
        min_length=1,
        description=(
            "Safe question suitable for display. For blocked input, "
            "this should be a generic or sanitized value rather than "
            "an unredacted sensitive request."
        ),
    )

    question_was_redacted: bool = Field(
        default=False,
        description=(
            "Whether the submitted question was redacted before "
            "analytical processing."
        ),
    )

    security: Optional[
        SecurityResponse
    ] = Field(
        default=None,
        description=(
            "Safe input or output security-decision metadata."
        ),
    )

    output_quality: Optional[
        OutputQualityResponse
    ] = Field(
        default=None,
        description=(
            "Safe output-validation and evaluation summary."
        ),
    )

    clarification_required: bool = Field(
        default=False,
        description=(
            "Whether the user must provide more information."
        ),
    )

    clarification_question: Optional[str] = Field(
        default=None,
        description=(
            "Focused follow-up question generated by Agent 1."
        ),
    )

    business_summary: Optional[str] = Field(
        default=None,
        description=(
            "Final sanitized business-friendly answer."
        ),
    )

    key_findings: list[str] = Field(
        default_factory=list,
        description=(
            "Sanitized findings from the analysis."
        ),
    )

    root_cause_indicators: list[str] = Field(
        default_factory=list,
        description=(
            "Evidence-supported possible root-cause indicators."
        ),
    )

    recommendations: list[
        RecommendationResponse
    ] = Field(
        default_factory=list,
        description=(
            "Evidence-based business recommendations."
        ),
    )

    chart: Optional[
        ChartResponse
    ] = Field(
        default=None,
        description=(
            "Validated chart specification."
        ),
    )

    query_result: Optional[
        QueryResultResponse
    ] = Field(
        default=None,
        description=(
            "Sanitized data returned by Agent 3b."
        ),
    )

    limitations: list[str] = Field(
        default_factory=list,
        description=(
            "Analysis and reporting limitations."
        ),
    )

    technical_details: Optional[
        TechnicalDetailsResponse
    ] = Field(
        default=None,
        description=(
            "Optional role-filtered technical workflow details."
        ),
    )

    error: Optional[
        APIErrorDetail
    ] = Field(
        default=None,
        description=(
            "Structured technical error. Used only for status failed."
        ),
    )

    @model_validator(mode="after")
    def validate_response_state(
        self,
    ) -> "AnalyticsQueryResponse":
        """
        Ensure response content agrees with the final workflow status.
        """

        controlled_security_statuses = {
            "security_blocked",
            "security_review_required",
            "output_blocked",
            "output_review_required",
        }

        # -------------------------------------------------------------
        # Successful response
        # -------------------------------------------------------------

        if self.status == "completed":
            if self.clarification_required:
                raise ValueError(
                    "A completed response cannot "
                    "require clarification."
                )

            if not self.business_summary:
                raise ValueError(
                    "business_summary is required "
                    "when status is completed."
                )

            if self.query_result is None:
                raise ValueError(
                    "query_result is required when "
                    "status is completed."
                )

            if self.error is not None:
                raise ValueError(
                    "A completed response cannot "
                    "contain an error."
                )

            if (
                self.security is not None
                and self.security.decision
                in {
                    "block",
                    "review",
                }
            ):
                raise ValueError(
                    "A completed response cannot contain "
                    "a blocking or review security decision."
                )

            if (
                self.output_quality is not None
                and not (
                    self.output_quality
                    .sanitized_payload_available
                )
            ):
                raise ValueError(
                    "A completed response requires "
                    "a sanitized output payload."
                )

        # -------------------------------------------------------------
        # Clarification response
        # -------------------------------------------------------------

        elif (
            self.status
            == "clarification_required"
        ):
            if not self.clarification_required:
                raise ValueError(
                    "clarification_required must be "
                    "True for a clarification response."
                )

            if not self.clarification_question:
                raise ValueError(
                    "clarification_question is required "
                    "for a clarification response."
                )

            if self.error is not None:
                raise ValueError(
                    "A clarification response cannot "
                    "contain a technical error."
                )

            if self.business_summary is not None:
                raise ValueError(
                    "A clarification response cannot "
                    "contain a final business summary."
                )

            if self.query_result is not None:
                raise ValueError(
                    "A clarification response cannot "
                    "contain query results."
                )

        # -------------------------------------------------------------
        # Controlled security and quality responses
        # -------------------------------------------------------------

        elif self.status in (
            controlled_security_statuses
        ):
            if self.security is None:
                raise ValueError(
                    "security metadata is required "
                    "for blocked or review responses."
                )

            if self.error is not None:
                raise ValueError(
                    "Controlled security outcomes cannot "
                    "contain technical error details."
                )

            if self.clarification_required:
                raise ValueError(
                    "Controlled security outcomes cannot "
                    "require analytical clarification."
                )

            if self.clarification_question is not None:
                raise ValueError(
                    "Controlled security outcomes cannot "
                    "contain a clarification question."
                )

            if self.business_summary is not None:
                raise ValueError(
                    "Blocked or review-required responses "
                    "cannot contain a released business summary."
                )

            if self.key_findings:
                raise ValueError(
                    "Blocked or review-required responses "
                    "cannot contain released findings."
                )

            if self.root_cause_indicators:
                raise ValueError(
                    "Blocked or review-required responses "
                    "cannot contain root-cause indicators."
                )

            if self.recommendations:
                raise ValueError(
                    "Blocked or review-required responses "
                    "cannot contain recommendations."
                )

            if self.chart is not None:
                raise ValueError(
                    "Blocked or review-required responses "
                    "cannot contain a chart."
                )

            if self.query_result is not None:
                raise ValueError(
                    "Blocked or review-required responses "
                    "cannot contain query-result rows."
                )

            if self.technical_details is not None:
                raise ValueError(
                    "Blocked or review-required responses "
                    "cannot expose technical details."
                )

            if (
                self.status
                in {
                    "security_blocked",
                    "output_blocked",
                }
                and self.security.decision
                != "block"
            ):
                raise ValueError(
                    "Blocked statuses require "
                    "security.decision='block'."
                )

            if (
                self.status
                in {
                    "security_review_required",
                    "output_review_required",
                }
                and self.security.decision
                != "review"
            ):
                raise ValueError(
                    "Review statuses require "
                    "security.decision='review'."
                )

            if (
                self.status
                in {
                    "security_review_required",
                    "output_review_required",
                }
                and not (
                    self.security
                    .requires_review
                )
            ):
                raise ValueError(
                    "Review statuses require "
                    "security.requires_review=True."
                )

        # -------------------------------------------------------------
        # Technical failure response
        # -------------------------------------------------------------

        elif self.status == "failed":
            if self.error is None:
                raise ValueError(
                    "error is required when "
                    "status is failed."
                )

            if self.business_summary is not None:
                raise ValueError(
                    "A failed response cannot contain "
                    "a released business summary."
                )

            if self.query_result is not None:
                raise ValueError(
                    "A failed response cannot expose "
                    "query-result rows."
                )

        return self


# ---------------------------------------------------------------------
# Health and readiness responses
# ---------------------------------------------------------------------

class HealthResponse(APISchema):
    """
    Basic FastAPI process-health response.
    """

    status: Literal[
        "healthy",
        "unhealthy",
    ] = Field(
        description=(
            "FastAPI process health."
        ),
    )

    service: str = Field(
        default=(
            "multi-agent-analytics-api"
        ),
        description=(
            "API service identifier."
        ),
    )

    version: str = Field(
        default="1.0.0",
        description=(
            "API contract version."
        ),
    )


class ReadinessComponent(APISchema):
    """
    Readiness status for one backend dependency.
    """

    component: str = Field(
        min_length=1,
        description=(
            "Dependency name."
        ),
    )

    ready: bool = Field(
        description=(
            "Whether the dependency is ready."
        ),
    )

    message: str = Field(
        min_length=1,
        description=(
            "Safe readiness description."
        ),
    )


class ReadinessResponse(APISchema):
    """
    Readiness response for FastAPI dependencies.
    """

    status: Literal[
        "ready",
        "not_ready",
    ] = Field(
        description=(
            "Overall backend readiness."
        ),
    )

    components: list[
        ReadinessComponent
    ] = Field(
        default_factory=list,
        description=(
            "Ollama, model, database, and "
            "LangGraph readiness checks."
        ),
    )


class ModelStatusResponse(APISchema):
    """
    Status of one required Ollama model.
    """

    name: str = Field(
        min_length=1,
        description=(
            "Configured model name."
        ),
    )

    installed: bool = Field(
        description=(
            "Whether the model is installed locally."
        ),
    )


class ModelsResponse(APISchema):
    """
    Configured Ollama-model status.
    """

    ollama_available: bool = Field(
        description=(
            "Whether the Ollama API is responding."
        ),
    )

    models: list[
        ModelStatusResponse
    ] = Field(
        default_factory=list,
        description=(
            "Required model installation statuses."
        ),
    )