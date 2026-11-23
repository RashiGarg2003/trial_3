"""
FastAPI backend for the Multi-Agent Data Analytics System.

Endpoints:

GET  /health
GET  /ready
GET  /api/v1/models
POST /api/v1/analytics/query

The analytics endpoint executes the complete LangGraph workflow:

1. Intent Understanding Agent
2. Planning Agent
3. SQL Generation Agent
4. SQL Validation and Execution
5. Analysis Agent
6. Reporting Agent

The backend returns structured Pydantic responses for Streamlit.

Current version:
    Local development API.

Future security additions:
    - Authentication
    - RBAC
    - Prompt-injection protection
    - Rate limiting
    - Secure audit logging
    - MCP tool permissions
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    status,
)
from fastapi.exceptions import (
    RequestValidationError,
)
from fastapi.middleware.cors import (
    CORSMiddleware,
)
from fastapi.responses import JSONResponse

from api.dependencies import (
    AppServices,
    ServiceInitializationError,
    check_database_readiness,
    check_graph_readiness,
    check_ollama_readiness,
    clear_dependency_caches,
    get_services,
    get_settings,
)
from api.schemas import (
    AnalyticsQueryRequest,
    AnalyticsQueryResponse,
    APIErrorDetail,
    ChartResponse,
    HealthResponse,
    IntentSummaryResponse,
    ModelStatusResponse,
    ModelsResponse,
    OutputQualityResponse,
    PlanStepResponse,
    PlanSummaryResponse,
    QueryResultResponse,
    ReadinessComponent,
    ReadinessResponse,
    RecommendationResponse,
    SecurityResponse,
    TechnicalDetailsResponse,
    TraceEventResponse,
)
from models.schemas import (
    AnalysisPlan,
    FinalReport,
    IntentResult,
    QueryExecutionResult,
    SQLGenerationResult,
)

from security.models import (
    DeterministicValidationResult,
    InputGuardResult,
    OutputEvaluation,
    OutputGuardResult,
    PromptRiskAssessment,
)
from workflow.state import (
    GraphState,
    create_initial_state,
)


# Serializes local model execution in the MVP.
#
# Ollama can queue requests internally, but running multiple complete
# workflows simultaneously may consume substantial CPU, RAM, or VRAM.
WORKFLOW_SEMAPHORE = asyncio.Semaphore(1)


# ---------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------

@asynccontextmanager
async def lifespan(
    application: FastAPI,
) -> AsyncIterator[None]:
    """
    Initialize backend services when FastAPI starts.

    Startup validates:

    - Ollama availability
    - Required local models
    - Read-only SQLite database
    - Database integrity
    - Compiled LangGraph
    - Chart tool

    The initialized service container is stored in application.state.
    """

    try:
        services = await asyncio.to_thread(
            get_services
        )

        application.state.services = services
        application.state.startup_error = None

    except Exception as exc:
        application.state.services = None
        application.state.startup_error = str(exc)

    yield

    clear_dependency_caches()


settings = get_settings()


app = FastAPI(
    title=settings.application_name,
    version=settings.application_version,
    description=(
        "FastAPI backend for the local Multi-Agent Data "
        "Analytics System powered by Ollama, LangGraph, "
        "SQLite, Pydantic, and SQLGlot."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------
# CORS configuration
# ---------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_credentials=False,
    allow_methods=[
        "GET",
        "POST",
        "OPTIONS",
    ],
    allow_headers=[
        "Content-Type",
        "Accept",
    ],
)


# ---------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------

@app.exception_handler(
    RequestValidationError
)
async def request_validation_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    Return a consistent response for invalid API request bodies.
    """

    safe_errors: list[dict[str, Any]] = []

    for error in exc.errors():
        safe_error = dict(error)
        safe_error.pop("ctx", None)
        safe_error.pop("url", None)
        safe_error["loc"] = [
            str(location)
            for location in safe_error.get(
                "loc",
                (),
            )
        ]
        safe_errors.append(safe_error)

    return JSONResponse(
        status_code=(
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ),
        content={
            "detail": "Request validation failed.",
            "errors": safe_errors,
        },
    )


@app.exception_handler(
    ServiceInitializationError
)
async def service_initialization_handler(
    request: Request,
    exc: ServiceInitializationError,
) -> JSONResponse:
    """
    Return a safe response when backend services are unavailable.
    """

    return JSONResponse(
        status_code=(
            status.HTTP_503_SERVICE_UNAVAILABLE
        ),
        content={
            "detail": (
                "Backend services are not ready."
            ),
            "error_type": (
                type(exc).__name__
            ),
        },
    )


# ---------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------

def get_application_services(
    request: Request,
) -> AppServices:
    """
    Return services initialized during FastAPI startup.

    Raises:
        HTTPException:
            When startup initialization failed.
    """

    services = getattr(
        request.app.state,
        "services",
        None,
    )

    startup_error = getattr(
        request.app.state,
        "startup_error",
        None,
    )

    if not isinstance(
        services,
        AppServices,
    ):
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Backend services are unavailable. "
                f"Startup error: "
                f"{startup_error or 'unknown error'}"
            ),
        )

    return services

def build_safe_display_question(
    state: GraphState,
    request_body: AnalyticsQueryRequest,
) -> str:
    """
    Return a safe question suitable for the API response.

    The original unredacted question is never returned after security
    screening. The sanitized GraphState question is preferred.

    For blocked sensitive input, a generic label is returned.
    """

    workflow_status = state.get(
        "workflow_status"
    )

    safe_question = state.get(
        "user_question"
    )

    question_was_redacted = state.get(
        "question_was_redacted",
        False,
    )

    if (
        workflow_status
        in {
            "security_blocked",
            "security_review_required",
        }
        and question_was_redacted
    ):
        return (
            "Request screened by security policy."
        )

    if (
        isinstance(safe_question, str)
        and safe_question.strip()
    ):
        return safe_question.strip()

    # This fallback is safe for technical failures that occur before
    # security screening. Do not use it for known sensitive input.
    if workflow_status == "failed":
        return request_body.question

    return (
        "Analytics request."
    )

def build_security_response(
    state: GraphState,
    *,
    decision_override: str | None = None,
    message_override: str | None = None,
    requires_review_override: bool | None = None,
) -> SecurityResponse:
    """
    Build safe public security metadata.

    Raw PII findings, matched text, prompts, and guard reasoning are
    deliberately excluded.
    """

    input_guard_result = state.get(
        "input_guard_result"
    )

    prompt_risk = state.get(
        "prompt_risk_assessment"
    )

    risk_categories = list(
        state.get(
            "security_risk_categories",
            [],
        )
    )

    risk_level: str | None = None

    if isinstance(
        input_guard_result,
        InputGuardResult,
    ):
        risk_level = enum_value(
            input_guard_result.risk_level
        )

    if isinstance(
        prompt_risk,
        PromptRiskAssessment,
    ):
        prompt_risk_level = enum_value(
            prompt_risk.risk_level
        )

        risk_priority = {
            "low": 0,
            "medium": 1,
            "high": 2,
            "critical": 3,
        }

        if (
            risk_level is None
            or risk_priority.get(
                prompt_risk_level,
                0,
            )
            > risk_priority.get(
                risk_level,
                0,
            )
        ):
            risk_level = prompt_risk_level

    decision = enum_value(
        decision_override
        or state.get(
            "security_decision"
        )
        or "pending"
    )

    message = (
        message_override
        or state.get(
            "security_message"
        )
        or state.get(
            "output_validation_message"
        )
        or (
            "The request was processed according "
            "to application security policy."
        )
    )

    requires_review = (
        requires_review_override
        if requires_review_override is not None
        else bool(
            state.get(
                "security_review_required",
                False,
            )
        )
    )

    return SecurityResponse(
        decision=decision,
        risk_level=risk_level,
        risk_categories=risk_categories,
        message=message,
        question_was_redacted=bool(
            state.get(
                "question_was_redacted",
                False,
            )
        ),
        requires_review=requires_review,
    )


def build_output_quality_response(
    state: GraphState,
) -> OutputQualityResponse | None:
    """
    Build a safe summary of output validation.

    Enum fields are converted to their primitive string values before
    API-schema validation.
    """

    deterministic_result = state.get(
        "deterministic_validation_result"
    )

    output_evaluation = state.get(
        "output_evaluation"
    )

    output_guard_result = state.get(
        "output_guard_result"
    )

    if not any(
        [
            isinstance(
                deterministic_result,
                DeterministicValidationResult,
            ),
            isinstance(
                output_evaluation,
                OutputEvaluation,
            ),
            isinstance(
                output_guard_result,
                OutputGuardResult,
            ),
        ]
    ):
        return None

    deterministic_decision = None

    if isinstance(
        deterministic_result,
        DeterministicValidationResult,
    ):
        deterministic_decision = enum_value(
            deterministic_result.decision
        )

    llm_evaluation_decision = None

    if isinstance(
        output_evaluation,
        OutputEvaluation,
    ):
        llm_evaluation_decision = enum_value(
            output_evaluation.decision
        )

    output_guard_decision = None

    if isinstance(
        output_guard_result,
        OutputGuardResult,
    ):
        output_guard_decision = enum_value(
            output_guard_result.decision
        )

    return OutputQualityResponse(
        deterministic_decision=(
            deterministic_decision
        ),
        llm_evaluation_decision=(
            llm_evaluation_decision
        ),
        output_guard_decision=(
            output_guard_decision
        ),
        deterministic_valid=(
            deterministic_result.valid
            if isinstance(
                deterministic_result,
                DeterministicValidationResult,
            )
            else None
        ),
        numerical_claims_valid=(
            deterministic_result
            .numerical_claims_valid
            if isinstance(
                deterministic_result,
                DeterministicValidationResult,
            )
            else None
        ),
        ranking_valid=(
            deterministic_result
            .ranking_valid
            if isinstance(
                deterministic_result,
                DeterministicValidationResult,
            )
            else None
        ),
        chart_valid=(
            deterministic_result
            .chart_valid
            if isinstance(
                deterministic_result,
                DeterministicValidationResult,
            )
            else None
        ),
        pii_safe=(
            deterministic_result.pii_safe
            if isinstance(
                deterministic_result,
                DeterministicValidationResult,
            )
            else None
        ),
        secret_safe=(
            deterministic_result.secret_safe
            if isinstance(
                deterministic_result,
                DeterministicValidationResult,
            )
            else None
        ),
        prompt_leakage_safe=(
            deterministic_result
            .prompt_leakage_safe
            if isinstance(
                deterministic_result,
                DeterministicValidationResult,
            )
            else None
        ),
        relevance_score=(
            output_evaluation.relevance_score
            if isinstance(
                output_evaluation,
                OutputEvaluation,
            )
            else None
        ),
        groundedness_score=(
            output_evaluation
            .groundedness_score
            if isinstance(
                output_evaluation,
                OutputEvaluation,
            )
            else None
        ),
        accuracy_score=(
            output_evaluation.accuracy_score
            if isinstance(
                output_evaluation,
                OutputEvaluation,
            )
            else None
        ),
        safety_score=(
            output_evaluation.safety_score
            if isinstance(
                output_evaluation,
                OutputEvaluation,
            )
            else None
        ),
        overall_score=(
            output_evaluation.overall_score
            if isinstance(
                output_evaluation,
                OutputEvaluation,
            )
            else None
        ),
        redaction_count=(
            len(
                output_guard_result.redactions
            )
            if isinstance(
                output_guard_result,
                OutputGuardResult,
            )
            else 0
        ),
        sanitized_payload_available=(
            isinstance(
                output_guard_result,
                OutputGuardResult,
            )
            and output_guard_result
            .sanitized_payload
            is not None
        ),
        message=state.get(
            "output_validation_message"
        ),
    )


# ---------------------------------------------------------------------
# GraphState conversion helpers
# ---------------------------------------------------------------------
def enum_value(
    value: Any,
) -> Any:
    """
    Return the primitive value stored by an Enum.

    Examples:

        EvaluationDecision.APPROVE -> "approve"
        SecurityDecision.ALLOW -> "allow"
        "approve" -> "approve"
        None -> None
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

def build_intent_summary(
    intent: IntentResult | None,
) -> IntentSummaryResponse | None:
    """
    Convert IntentResult into the public intent summary.
    """

    if intent is None:
        return None

    filters = [
        filter_condition.model_dump(
            mode="json"
        )
        for filter_condition in intent.filters
    ]

    return IntentSummaryResponse(
        analysis_type=(
            intent.analysis_type
        ),
        metrics=list(
            intent.metrics
        ),
        dimensions=list(
            intent.dimensions
        ),
        filters=filters,
        sort_order=intent.sort_order,
        limit=intent.limit,
    )


def build_plan_summary(
    plan: AnalysisPlan | None,
) -> PlanSummaryResponse | None:
    """
    Convert AnalysisPlan into a safe API summary.
    """

    if plan is None:
        return None

    steps = [
        PlanStepResponse(
            step_id=step.step_id,
            title=step.title,
            description=step.description,
            step_type=step.step_type,
            depends_on=list(
                step.depends_on
            ),
            requires_sql=(
                step.requires_sql
            ),
            status=step.status,
        )
        for step in plan.steps
    ]

    return PlanSummaryResponse(
        objective=plan.objective,
        strategy=plan.strategy,
        steps=steps,
    )


def build_query_result_response(
    query_result: (
        QueryExecutionResult | None
    ),
) -> QueryResultResponse | None:
    """
    Convert QueryExecutionResult into the public API result.
    """

    if query_result is None:
        return None

    return QueryResultResponse(
        columns=list(
            query_result.columns
        ),
        rows=list(
            query_result.rows
        ),
        row_count=(
            query_result.row_count
        ),
        execution_time_ms=(
            query_result.execution_time_ms
        ),
        truncated=(
            query_result.truncated
        ),
    )


def build_chart_response(
    final_report: FinalReport | None,
) -> ChartResponse | None:
    """
    Convert FinalReport chart configuration into an API schema.
    """

    if (
        final_report is None
        or final_report.chart is None
    ):
        return None

    chart = final_report.chart

    return ChartResponse(
        chart_type=chart.chart_type,
        title=chart.title,
        x_column=chart.x_column,
        y_columns=list(
            chart.y_columns
        ),
        category_column=(
            chart.category_column
        ),
        rationale=chart.rationale,
    )


def build_recommendations(
    final_report: FinalReport | None,
) -> list[RecommendationResponse]:
    """
    Convert report recommendations into API schemas.
    """

    if final_report is None:
        return []

    return [
        RecommendationResponse(
            action=recommendation.action,
            rationale=(
                recommendation.rationale
            ),
            priority=(
                recommendation.priority
            ),
        )
        for recommendation
        in final_report.recommendations
    ]


def build_trace_response(
    state: GraphState,
) -> list[TraceEventResponse]:
    """
    Convert the sanitized LangGraph trace into API schemas.

    Unexpected or malformed trace events are excluded.
    """

    trace_events: list[
        TraceEventResponse
    ] = []

    for event in state.get(
        "execution_trace",
        [],
    ):
        if not isinstance(event, dict):
            continue

        node = event.get("node")
        event_status = event.get(
            "status"
        )
        message = event.get("message")

        if not all(
            isinstance(value, str)
            and bool(value.strip())
            for value in (
                node,
                event_status,
                message,
            )
        ):
            continue

        trace_events.append(
            TraceEventResponse(
                node=node,
                status=event_status,
                message=message,
            )
        )

    return trace_events


def build_technical_details(
    state: GraphState,
    include_technical_details: bool,
    services: AppServices,
) -> TechnicalDetailsResponse | None:
    """
    Build optional technical details.

    Technical details are returned only when:

    - The client requested them.
    - Backend configuration allows them.

    RBAC will later replace this development-only check.
    """

    if not include_technical_details:
        return None

    if not (
        services
        .settings
        .expose_technical_details
    ):
        return None

    intent = state.get("intent")
    plan = state.get("plan")
    sql_result = state.get(
        "sql_result"
    )

    referenced_tables: list[str] = []

    if isinstance(
        sql_result,
        SQLGenerationResult,
    ):
        referenced_tables = list(
            sql_result.referenced_tables
        )

    return TechnicalDetailsResponse(
        generated_sql=state.get(
            "generated_sql"
        ),
        referenced_tables=(
            referenced_tables
        ),
        sql_retry_count=state.get(
            "sql_retry_count",
            0,
        ),
        max_sql_retries=state.get(
            "max_sql_retries",
            0,
        ),
        intent=build_intent_summary(
            intent
            if isinstance(
                intent,
                IntentResult,
            )
            else None
        ),
        plan=build_plan_summary(
            plan
            if isinstance(
                plan,
                AnalysisPlan,
            )
            else None
        ),
        execution_trace=(
            build_trace_response(
                state
            )
        ),
    )

def build_report_from_sanitized_payload(
    state: GraphState,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any] | None,
]:
    """
    Retrieve report, query-result, and technical-detail dictionaries
    approved by OutputGuard.

    Raw FinalReport and QueryExecutionResult must not be used for a
    completed API response after OutputGuard has run.
    """

    output_guard_result = state.get(
        "output_guard_result"
    )

    if not isinstance(
        output_guard_result,
        OutputGuardResult,
    ):
        raise RuntimeError(
            "Completed workflow does not contain "
            "OutputGuardResult."
        )

    if str(output_guard_result.decision) not in {
        "allow",
        "allow_redacted",
    }:
        raise RuntimeError(
            "Completed workflow contains a "
            "non-releasable output decision."
        )

    payload = output_guard_result.sanitized_payload

    if not isinstance(payload, dict):
        raise RuntimeError(
            "OutputGuard did not return a "
            "sanitized payload."
        )

    report_payload = payload.get("final_report")
    query_payload = payload.get("query_result")
    technical_payload = payload.get("technical_details")

    if not isinstance(report_payload, dict):
        raise RuntimeError(
            "Sanitized payload does not contain "
            "a final_report dictionary."
        )

    if not isinstance(query_payload, dict):
        raise RuntimeError(
            "Sanitized payload does not contain "
            "a query_result dictionary."
        )

    if (
        technical_payload is not None
        and not isinstance(technical_payload, dict)
    ):
        raise RuntimeError(
            "Sanitized technical details must "
            "be a dictionary or None."
        )

    return (
        report_payload,
        query_payload,
        technical_payload,
    )


def build_sanitized_query_result_response(
    query_payload: dict[str, Any],
) -> QueryResultResponse:
    """
    Validate sanitized query-result data against the public API model.
    """

    return QueryResultResponse(
        columns=list(
            query_payload.get(
                "columns",
                [],
            )
        ),
        rows=list(
            query_payload.get(
                "rows",
                [],
            )
        ),
        row_count=int(
            query_payload.get(
                "row_count",
                0,
            )
        ),
        execution_time_ms=float(
            query_payload.get(
                "execution_time_ms",
                0.0,
            )
        ),
        truncated=bool(
            query_payload.get(
                "truncated",
                False,
            )
        ),
    )

def build_sanitized_technical_details(
    technical_payload: dict[str, Any] | None,
    include_technical_details: bool,
) -> TechnicalDetailsResponse | None:
    """
    Convert technical details already filtered by OutputGuard.

    Only explicitly approved API fields are copied. Full internal
    IntentResult, AnalysisPlan, and PlanStep dictionaries are not
    passed directly to API response models because the public schemas
    intentionally expose fewer fields.
    """

    if not include_technical_details:
        return None

    if technical_payload is None:
        return None

    if not isinstance(
        technical_payload,
        dict,
    ):
        raise TypeError(
            "technical_payload must be a dictionary."
        )

    # -------------------------------------------------------------
    # Safe IntentResult conversion
    # -------------------------------------------------------------

    intent_response = None

    intent_payload = technical_payload.get(
        "intent"
    )

    if isinstance(
        intent_payload,
        dict,
    ):
        raw_filters = intent_payload.get(
            "filters",
            [],
        )

        safe_filters = [
            filter_item
            for filter_item in raw_filters
            if isinstance(
                filter_item,
                dict,
            )
        ]

        raw_limit = intent_payload.get(
            "limit"
        )

        safe_limit = (
            raw_limit
            if isinstance(
                raw_limit,
                int,
            )
            and raw_limit >= 1
            else None
        )

        analysis_type = intent_payload.get(
            "analysis_type"
        )

        if (
            isinstance(
                analysis_type,
                str,
            )
            and analysis_type.strip()
        ):
            intent_response = (
                IntentSummaryResponse(
                    analysis_type=(
                        analysis_type
                    ),
                    metrics=[
                        str(metric)
                        for metric
                        in intent_payload.get(
                            "metrics",
                            [],
                        )
                        if str(metric).strip()
                    ],
                    dimensions=[
                        str(dimension)
                        for dimension
                        in intent_payload.get(
                            "dimensions",
                            [],
                        )
                        if str(
                            dimension
                        ).strip()
                    ],
                    filters=safe_filters,
                    sort_order=(
                        intent_payload.get(
                            "sort_order"
                        )
                    ),
                    limit=safe_limit,
                )
            )

    # -------------------------------------------------------------
    # Safe AnalysisPlan conversion
    # -------------------------------------------------------------

    plan_response = None

    plan_payload = technical_payload.get(
        "plan"
    )

    if isinstance(
        plan_payload,
        dict,
    ):
        safe_steps: list[
            PlanStepResponse
        ] = []

        raw_steps = plan_payload.get(
            "steps",
            [],
        )

        for step_payload in raw_steps:
            if not isinstance(
                step_payload,
                dict,
            ):
                continue

            step_id = step_payload.get(
                "step_id"
            )

            title = step_payload.get(
                "title"
            )

            description = step_payload.get(
                "description"
            )

            step_type = step_payload.get(
                "step_type"
            )

            requires_sql = step_payload.get(
                "requires_sql"
            )

            step_status = step_payload.get(
                "status",
                "pending",
            )

            if not isinstance(
                step_id,
                int,
            ):
                continue

            if not isinstance(
                title,
                str,
            ) or not title.strip():
                continue

            if not isinstance(
                description,
                str,
            ) or not description.strip():
                continue

            if not isinstance(
                step_type,
                str,
            ) or not step_type.strip():
                continue

            if not isinstance(
                requires_sql,
                bool,
            ):
                continue

            if not isinstance(
                step_status,
                str,
            ) or not step_status.strip():
                step_status = "pending"

            raw_dependencies = (
                step_payload.get(
                    "depends_on",
                    [],
                )
            )

            dependencies = [
                dependency
                for dependency
                in raw_dependencies
                if isinstance(
                    dependency,
                    int,
                )
                and dependency >= 1
            ]

            safe_steps.append(
                PlanStepResponse(
                    step_id=step_id,
                    title=title,
                    description=(
                        description
                    ),
                    step_type=step_type,
                    depends_on=dependencies,
                    requires_sql=(
                        requires_sql
                    ),
                    status=step_status,
                )
            )

        objective = plan_payload.get(
            "objective"
        )

        strategy = plan_payload.get(
            "strategy"
        )

        if (
            isinstance(
                objective,
                str,
            )
            and objective.strip()
            and isinstance(
                strategy,
                str,
            )
            and strategy.strip()
        ):
            plan_response = (
                PlanSummaryResponse(
                    objective=objective,
                    strategy=strategy,
                    steps=safe_steps,
                )
            )

    # -------------------------------------------------------------
    # Safe execution-trace conversion
    # -------------------------------------------------------------

    trace_response: list[
        TraceEventResponse
    ] = []

    trace_payload = technical_payload.get(
        "execution_trace",
        [],
    )

    if isinstance(
        trace_payload,
        list,
    ):
        for event in trace_payload:
            if not isinstance(
                event,
                dict,
            ):
                continue

            node = event.get(
                "node"
            )

            event_status = event.get(
                "status"
            )

            message = event.get(
                "message"
            )

            if not all(
                isinstance(value, str)
                and bool(value.strip())
                for value in (
                    node,
                    event_status,
                    message,
                )
            ):
                continue

            trace_response.append(
                TraceEventResponse(
                    node=node,
                    status=event_status,
                    message=(
                        message[:500]
                    ),
                )
            )

    # -------------------------------------------------------------
    # Final role-filtered technical response
    # -------------------------------------------------------------

    generated_sql = technical_payload.get(
        "generated_sql"
    )

    if not isinstance(
        generated_sql,
        str,
    ):
        generated_sql = None

    referenced_tables = [
        str(table_name)
        for table_name
        in technical_payload.get(
            "referenced_tables",
            [],
        )
        if str(table_name).strip()
    ]

    sql_retry_count = technical_payload.get(
        "sql_retry_count",
        0,
    )

    max_sql_retries = technical_payload.get(
        "max_sql_retries",
        0,
    )

    if not isinstance(
        sql_retry_count,
        int,
    ):
        sql_retry_count = 0

    if not isinstance(
        max_sql_retries,
        int,
    ):
        max_sql_retries = 0

    return TechnicalDetailsResponse(
        generated_sql=generated_sql,
        referenced_tables=(
            referenced_tables
        ),
        sql_retry_count=max(
            sql_retry_count,
            0,
        ),
        max_sql_retries=max(
            max_sql_retries,
            0,
        ),
        intent=intent_response,
        plan=plan_response,
        execution_trace=(
            trace_response
        ),
    )

def build_completed_response(
    state: GraphState,
    request_body: AnalyticsQueryRequest,
    services: AppServices,
) -> AnalyticsQueryResponse:
    """
    Build a successful response exclusively from OutputGuard-approved
    content.
    """

    (
        report_payload,
        query_payload,
        technical_payload,
    ) = build_report_from_sanitized_payload(
        state
    )

    business_summary = (
        report_payload.get(
            "business_summary"
        )
    )

    if not isinstance(
        business_summary,
        str,
    ) or not business_summary.strip():
        raise RuntimeError(
            "Sanitized report does not contain "
            "a valid business summary."
        )

    chart_payload = report_payload.get(
        "chart"
    )

    chart_response = None

    if isinstance(
        chart_payload,
        dict,
    ):
        chart_response = (
            ChartResponse.model_validate(
                chart_payload
            )
        )

    recommendation_payloads = (
        report_payload.get(
            "recommendations",
            [],
        )
    )

    recommendations = [
        RecommendationResponse
        .model_validate(
            recommendation
        )
        for recommendation
        in recommendation_payloads
        if isinstance(
            recommendation,
            dict,
        )
    ]

    output_guard_result = state.get(
        "output_guard_result"
    )

    guard_decision = (
        enum_value(
            output_guard_result.decision
        )
        if isinstance(
            output_guard_result,
            OutputGuardResult,
        )
        else "allow"
    )

    return AnalyticsQueryResponse(
        status="completed",
        question=(
            build_safe_display_question(
                state=state,
                request_body=request_body,
            )
        ),
        question_was_redacted=bool(
            state.get(
                "question_was_redacted",
                False,
            )
        ),
        security=build_security_response(
            state,
            decision_override=(
                guard_decision
            ),
            message_override=state.get(
                "output_validation_message"
            ),
            requires_review_override=False,
        ),
        output_quality=(
            build_output_quality_response(
                state
            )
        ),
        clarification_required=False,
        clarification_question=None,
        business_summary=(
            business_summary.strip()
        ),
        key_findings=list(
            report_payload.get(
                "key_findings",
                [],
            )
        ),
        root_cause_indicators=list(
            report_payload.get(
                "root_cause_indicators",
                [],
            )
        ),
        recommendations=recommendations,
        chart=chart_response,
        query_result=(
            build_sanitized_query_result_response(
                query_payload
            )
        ),
        limitations=list(
            report_payload.get(
                "limitations",
                [],
            )
        ),
        technical_details=(
            build_sanitized_technical_details(
                technical_payload=(
                    technical_payload
                ),
                include_technical_details=(
                    request_body
                    .include_technical_details
                ),
            )
        ),
        error=None,
    )

def build_clarification_response(
    state: GraphState,
    request_body: AnalyticsQueryRequest,
    services: AppServices,
) -> AnalyticsQueryResponse:
    """
    Build a response when Agent 1 requires clarification.
    """

    clarification_question = (
        state.get(
            "clarification_question"
        )
    )

    if not isinstance(
        clarification_question,
        str,
    ) or not clarification_question.strip():
        clarification_question = (
            "Could you provide more details "
            "about the analysis you want?"
        )

    return AnalyticsQueryResponse(
        status="clarification_required",
        question=build_safe_display_question(
            state=state,
            request_body=request_body,
        ),
        question_was_redacted=bool(
            state.get(
                "question_was_redacted",
                False,
            )
        ),
        security=build_security_response(
            state,
            requires_review_override=False,
        ),
        output_quality=None,
        clarification_required=True,
        clarification_question=(
            clarification_question
        ),
        business_summary=None,
        key_findings=[],
        root_cause_indicators=[],
        recommendations=[],
        chart=None,
        query_result=None,
        limitations=[],
        technical_details=None,
        error=None,
    )

def build_controlled_security_response(
    state: GraphState,
    request_body: AnalyticsQueryRequest,
) -> AnalyticsQueryResponse:
    """
    Build a safe blocked or review-required response.

    No report, rows, chart, SQL, plan, trace, or raw detection findings
    are released.
    """

    workflow_status = state.get(
        "workflow_status"
    )

    supported_statuses = {
        "security_blocked",
        "security_review_required",
        "output_blocked",
        "output_review_required",
    }

    if workflow_status not in supported_statuses:
        raise ValueError(
            "Unsupported controlled security status: "
            f"{workflow_status!r}."
        )

    is_review = workflow_status in {
        "security_review_required",
        "output_review_required",
    }

    decision = (
        "review"
        if is_review
        else "block"
    )

    if workflow_status.startswith(
        "output_"
    ):
        message = (
            state.get(
                "output_validation_message"
            )
            or (
                "The generated response requires review."
                if is_review
                else (
                    "The generated response was blocked "
                    "by output security controls."
                )
            )
        )

    else:
        message = (
            state.get(
                "security_message"
            )
            or (
                "The request requires security review."
                if is_review
                else (
                    "The request was blocked by "
                    "input security policy."
                )
            )
        )

    return AnalyticsQueryResponse(
        status=workflow_status,
        question=(
            build_safe_display_question(
                state=state,
                request_body=request_body,
            )
        ),
        question_was_redacted=bool(
            state.get(
                "question_was_redacted",
                False,
            )
        ),
        security=build_security_response(
            state,
            decision_override=decision,
            message_override=message,
            requires_review_override=(
                is_review
            ),
        ),
        output_quality=(
            build_output_quality_response(
                state
            )
        ),
        clarification_required=False,
        clarification_question=None,
        business_summary=None,
        key_findings=[],
        root_cause_indicators=[],
        recommendations=[],
        chart=None,
        query_result=None,
        limitations=[],
        technical_details=None,
        error=None,
    )


def build_failed_response(
    state: GraphState,
    request_body: AnalyticsQueryRequest,
    services: AppServices,
) -> AnalyticsQueryResponse:
    """
    Build a structured workflow-failure response.
    """

    error_type = state.get(
        "error_type"
    )

    error_message = state.get(
        "error_message"
    )

    current_node = state.get(
        "current_node"
    )

    safe_error_type = (
        error_type
        if isinstance(error_type, str)
        and error_type.strip()
        else "WorkflowError"
    )

    safe_error_message = (
        error_message
        if isinstance(error_message, str)
        and error_message.strip()
        else (
            "The analytics workflow could "
            "not complete the request."
        )
    )

    retry_count = state.get(
        "sql_retry_count",
        0,
    )

    max_retries = state.get(
        "max_sql_retries",
        0,
    )

    retryable = (
        isinstance(retry_count, int)
        and isinstance(max_retries, int)
        and retry_count < max_retries
    )

    return AnalyticsQueryResponse(
        status="failed",
        question=build_safe_display_question(
            state=state,
            request_body=request_body,
        ),
        question_was_redacted=bool(
            state.get(
                "question_was_redacted",
                False,
            )
        ),
        security=None,
        output_quality=(
            build_output_quality_response(
                state
            )
        ),
        clarification_required=False,
        clarification_question=None,
        business_summary=None,
        key_findings=[],
        root_cause_indicators=[],
        recommendations=[],
        chart=None,
        query_result=None,
        limitations=[],
        technical_details=None,
        error=APIErrorDetail(
            error_type=safe_error_type,
            message=safe_error_message,
            retryable=retryable,
            stage=(
                current_node
                if isinstance(
                    current_node,
                    str,
                )
                else None
            ),
        ),
    )


def convert_state_to_response(
    state: GraphState,
    request_body: AnalyticsQueryRequest,
    services: AppServices,
) -> AnalyticsQueryResponse:
    """
    Convert final LangGraph state into a public API response.
    """

    workflow_status = state.get(
        "workflow_status"
    )

    if workflow_status == "completed":
        return build_completed_response(
            state=state,
            request_body=request_body,
            services=services,
        )

    if (
        workflow_status
        == "clarification_required"
    ):
        return build_clarification_response(
            state=state,
            request_body=request_body,
            services=services,
        )

    if workflow_status in {
        "security_blocked",
        "security_review_required",
        "output_blocked",
        "output_review_required",
    }:
        return (
            build_controlled_security_response(
                state=state,
                request_body=request_body,
            )
        )

    return build_failed_response(
        state=state,
        request_body=request_body,
        services=services,
    )

# ---------------------------------------------------------------------
# Workflow execution
# ---------------------------------------------------------------------

def invoke_graph_sync(
    services: AppServices,
    request_body: AnalyticsQueryRequest,
) -> GraphState:
    """
    Invoke LangGraph synchronously.

    This function runs inside asyncio.to_thread() so FastAPI's event
    loop remains available for health and readiness requests.
    """

    maximum_allowed_retries = (
        services
        .settings
        .maximum_sql_retries
    )

    requested_retries = (
        request_body.max_sql_retries
    )

    effective_retries = min(
        requested_retries,
        maximum_allowed_retries,
    )

    initial_state = create_initial_state(
        user_question=(
            request_body.question
        ),
        max_sql_retries=(
            effective_retries
        ),
    )

    final_state = services.graph.invoke(
        initial_state,
        config={
            "recursion_limit": (
                services
                .settings
                .langgraph_recursion_limit
            ),
        },
    )

    return final_state


async def execute_analytics_workflow(
    services: AppServices,
    request_body: AnalyticsQueryRequest,
) -> GraphState:
    """
    Execute one complete workflow without blocking FastAPI's event loop.

    A process-local semaphore permits one full local-model workflow at
    a time for the MVP. This avoids exhausting local RAM or VRAM when
    several requests arrive simultaneously.
    """

    async with WORKFLOW_SEMAPHORE:
        return await asyncio.to_thread(
            invoke_graph_sync,
            services,
            request_body,
        )


# ---------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Check FastAPI process health",
)
async def health() -> HealthResponse:
    """
    Confirm that the FastAPI process is responding.

    This endpoint does not call Ollama, SQLite, or LangGraph.
    """

    return HealthResponse(
        status="healthy",
        service=(
            "multi-agent-analytics-api"
        ),
        version=(
            settings.application_version
        ),
    )


@app.get(
    "/ready",
    response_model=ReadinessResponse,
    tags=["System"],
    summary="Check backend dependency readiness",
)
async def ready(
    services: AppServices = Depends(
        get_application_services
    ),
) -> ReadinessResponse:
    """
    Check Ollama, database, and LangGraph readiness.
    """

    ollama_ready, ollama_message = (
        await asyncio.to_thread(
            check_ollama_readiness,
            services,
        )
    )

    database_ready, database_message = (
        await asyncio.to_thread(
            check_database_readiness,
            services,
        )
    )

    graph_ready, graph_message = (
        check_graph_readiness(
            services
        )
    )

    components = [
        ReadinessComponent(
            component="ollama",
            ready=ollama_ready,
            message=ollama_message,
        ),
        ReadinessComponent(
            component="database",
            ready=database_ready,
            message=database_message,
        ),
        ReadinessComponent(
            component="langgraph",
            ready=graph_ready,
            message=graph_message,
        ),
    ]

    overall_ready = all(
        component.ready
        for component in components
    )

    return ReadinessResponse(
        status=(
            "ready"
            if overall_ready
            else "not_ready"
        ),
        components=components,
    )


@app.get(
    "/api/v1/models",
    response_model=ModelsResponse,
    tags=["Models"],
    summary="Show required Ollama model status",
)
async def models(
    services: AppServices = Depends(
        get_application_services
    ),
) -> ModelsResponse:
    """
    Return installation status for required local models.
    """

    ollama_available = (
        await asyncio.to_thread(
            services
            .ollama_manager
            .is_server_running
        )
    )

    installed_models: set[str] = set()

    if ollama_available:
        try:
            model_list = (
                await asyncio.to_thread(
                    services
                    .ollama_manager
                    .get_installed_models
                )
            )

            installed_models = set(
                model_list
            )

        except Exception:
            installed_models = set()

    model_statuses = [
        ModelStatusResponse(
            name=model_name,
            installed=(
                model_name
                in installed_models
            ),
        )
        for model_name
        in services.settings.required_models
    ]

    return ModelsResponse(
        ollama_available=(
            ollama_available
        ),
        models=model_statuses,
    )


@app.post(
    "/api/v1/analytics/query",
    response_model=AnalyticsQueryResponse,
    tags=["Analytics"],
    summary="Run the complete analytics workflow",
)
async def analytics_query(
    request_body: AnalyticsQueryRequest,
    services: AppServices = Depends(
        get_application_services
    ),
) -> AnalyticsQueryResponse:
    """
    The endpoint returns HTTP 200 for controlled workflow outcomes:

- completed
- clarification_required
- security_blocked
- security_review_required
- output_blocked
- output_review_required
- failed

Invalid request bodies return HTTP 422.
Unavailable backend services return HTTP 503.
Unexpected backend exceptions return HTTP 500.
    """

    try:
        final_state = (
            await execute_analytics_workflow(
                services=services,
                request_body=request_body,
            )
        )

        return convert_state_to_response(
            state=final_state,
            request_body=request_body,
            services=services,
        )

    except HTTPException:
        raise

    except Exception as exc:
        import traceback

        print("\n" + "=" * 80)
        print("ANALYTICS API ERROR")
        print("=" * 80)
        traceback.print_exc()
        print("=" * 80 + "\n")

        error_detail = {
            "error_type": type(exc).__name__,
            "message": str(exc),
        }

        # Include detailed information for Pydantic validation errors
        if hasattr(exc, "errors"):
            try:
                error_detail["validation_errors"] = exc.errors()
            except Exception:
                pass

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_detail,
        ) from exc


@app.get(
    "/",
    tags=["System"],
    summary="API root",
)
async def root() -> dict[str, str]:
    """
    Return useful local API links.
    """

    return {
        "service": (
            settings.application_name
        ),
        "version": (
            settings.application_version
        ),
        "health": "/health",
        "readiness": "/ready",
        "models": "/api/v1/models",
        "analytics": (
            "/api/v1/analytics/query"
        ),
        "documentation": "/docs",
    }


# ---------------------------------------------------------------------
# Local development entry point
# ---------------------------------------------------------------------

def main() -> None:
    """
    Run the API directly with:

        python -m api.main

    The recommended development command remains:

        python -m uvicorn api.main:app --reload --port 8000
    """

    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()