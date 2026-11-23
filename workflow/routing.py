"""
Conditional routing functions for the LangGraph workflow.

Routing functions inspect GraphState and return the name of the next
workflow route.

These functions do not:

- Call an LLM.
- Execute database queries.
- Modify GraphState.
- Contain database-specific business logic.
- Generate user-facing reports.

Current workflow routes:

Input Guard:
    - prompt_guard
    - security_block
    - security_review
    - failure

Prompt-Injection Guard:
    - intent
    - security_block
    - security_review
    - failure

Intent Agent:
    - clarification
    - planning
    - failure

Planning Agent:
    - schema
    - failure

Schema Manager:
    - sql_generation
    - failure

SQL Generation:
    - sql_validation_execution
    - failure

SQL Validation and Execution:
    - retry_sql
    - analysis
    - failure

Analysis Agent:
    - reporting
    - failure

Reporting Agent:
    - deterministic_validation
    - failure

Deterministic Output Validation:
    - llm_evaluation
    - output_block
    - output_review
    - failure

LLM Output Evaluation:
    - output_guard
    - output_block
    - output_review
    - failure

Final Output Guard:
    - completed
    - output_block
    - output_review
    - failure
"""

from typing import Literal

from security.models import (
    DeterministicValidationResult,
    OutputEvaluation,
    OutputGuardResult,
)

from workflow.state import GraphState
def enum_value(
    value: object,
) -> object:
    """
    Return a plain Enum value for routing comparisons.
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


# ---------------------------------------------------------------------
# Route types
# ---------------------------------------------------------------------

InputGuardRoute = Literal[
    "prompt_guard",
    "security_block",
    "security_review",
    "failure",
]


PromptGuardRoute = Literal[
    "intent",
    "security_block",
    "security_review",
    "failure",
]


IntentRoute = Literal[
    "clarification",
    "planning",
    "failure",
]


PlanningRoute = Literal[
    "schema",
    "failure",
]


SchemaRoute = Literal[
    "sql_generation",
    "failure",
]


SQLGenerationRoute = Literal[
    "sql_validation_execution",
    "failure",
]


SQLExecutionRoute = Literal[
    "retry_sql",
    "analysis",
    "failure",
]


AnalysisRoute = Literal[
    "reporting",
    "failure",
]


ReportingRoute = Literal[
    "deterministic_validation",
    "failure",
]


DeterministicValidationRoute = Literal[
    "llm_evaluation",
    "output_block",
    "output_review",
    "failure",
]


LLMOutputEvaluationRoute = Literal[
    "output_guard",
    "output_block",
    "output_review",
    "failure",
]


OutputGuardRoute = Literal[
    "completed",
    "output_block",
    "output_review",
    "failure",
]


# ---------------------------------------------------------------------
# Shared failure check
# ---------------------------------------------------------------------

def has_workflow_failed(
    state: GraphState,
) -> bool:
    """
    Return True when a technical workflow failure exists.

    Controlled security and output decisions are not treated as
    technical failures.

    These controlled states include:

    - security_blocked
    - security_review_required
    - output_blocked
    - output_review_required
    """

    workflow_status = state.get(
        "workflow_status"
    )

    error_message = state.get(
        "error_message"
    )

    if workflow_status == "failed":
        return True

    return (
        isinstance(
            error_message,
            str,
        )
        and bool(
            error_message.strip()
        )
    )


# ---------------------------------------------------------------------
# Input-security routing
# ---------------------------------------------------------------------

def route_after_input_guard(
    state: GraphState,
) -> InputGuardRoute:
    """
    Decide what happens after deterministic input screening.

    Routes:

    - prompt_guard:
      Safe or safely redacted input continues to semantic screening.

    - security_block:
      A prohibited operation or restricted value was detected.

    - security_review:
      The request requires human security review.

    - failure:
      Input-guard execution failed technically or returned an
      inconsistent state.
    """

    if has_workflow_failed(state):
        return "failure"

    guard_result = state.get(
        "input_guard_result"
    )

    if guard_result is None:
        return "failure"

    decision = state.get(
        "security_decision"
    )

    workflow_status = state.get(
        "workflow_status"
    )

    if (
        decision == "block"
        or workflow_status
        == "security_blocked"
    ):
        return "security_block"

    if (
        decision == "review"
        or state.get(
            "security_review_required",
            False,
        )
        or workflow_status
        == "security_review_required"
    ):
        return "security_review"

    if decision not in {
        "allow",
        "allow_redacted",
    }:
        return "failure"

    sanitized_question = state.get(
        "user_question"
    )

    if not isinstance(
        sanitized_question,
        str,
    ):
        return "failure"

    if not sanitized_question.strip():
        return "failure"

    return "prompt_guard"


def route_after_prompt_guard(
    state: GraphState,
) -> PromptGuardRoute:
    """
    Decide what happens after semantic prompt-risk evaluation.

    Routes:

    - intent:
      Both security layers approved the request.

    - security_block:
      Semantic screening found prohibited or malicious intent.

    - security_review:
      Semantic screening found an ambiguous protected request.

    - failure:
      Prompt-risk evaluation failed technically or returned an
      inconsistent state.
    """

    if has_workflow_failed(state):
        return "failure"

    assessment = state.get(
        "prompt_risk_assessment"
    )

    if assessment is None:
        return "failure"

    decision = state.get(
        "security_decision"
    )

    workflow_status = state.get(
        "workflow_status"
    )

    if (
        decision == "block"
        or workflow_status
        == "security_blocked"
    ):
        return "security_block"

    if (
        decision == "review"
        or state.get(
            "security_review_required",
            False,
        )
        or workflow_status
        == "security_review_required"
    ):
        return "security_review"

    if decision not in {
        "allow",
        "allow_redacted",
    }:
        return "failure"

    sanitized_question = state.get(
        "user_question"
    )

    if not isinstance(
        sanitized_question,
        str,
    ):
        return "failure"

    if not sanitized_question.strip():
        return "failure"

    return "intent"


# ---------------------------------------------------------------------
# Agent 1 routing
# ---------------------------------------------------------------------

from typing import Literal


def route_after_intent(
    state: dict,
) -> Literal["clarification", "planning"]:
    """
    Route an underspecified request to clarification.
    Route a sufficiently specific request to planning.
    """

    clarification_required = state.get(
        "clarification_required",
        False,
    )

    # Normalize unexpected model/state values.
    if isinstance(clarification_required, list):
        clarification_required = any(
            bool(value)
            for value in clarification_required
        )

    elif isinstance(clarification_required, str):
        clarification_required = (
            clarification_required.strip().lower()
            in {"true", "yes", "1"}
        )

    else:
        clarification_required = bool(
            clarification_required
        )

    if clarification_required:
        return "clarification"

    return "planning"

# ---------------------------------------------------------------------
# Agent 2 routing
# ---------------------------------------------------------------------

def route_after_planning(
    state: GraphState,
) -> PlanningRoute:
    """
    Route a valid analytical plan to schema preparation.
    """

    if has_workflow_failed(state):
        return "failure"

    plan = state.get(
        "plan"
    )

    if plan is None:
        return "failure"

    if not plan.steps:
        return "failure"

    return "schema"


# ---------------------------------------------------------------------
# Schema routing
# ---------------------------------------------------------------------

def route_after_schema(
    state: GraphState,
) -> SchemaRoute:
    """
    Route a valid non-empty database schema to Agent 3a.
    """

    if has_workflow_failed(state):
        return "failure"

    database_schema = state.get(
        "database_schema"
    )

    if not isinstance(
        database_schema,
        str,
    ):
        return "failure"

    if not database_schema.strip():
        return "failure"

    return "sql_generation"


# ---------------------------------------------------------------------
# Agent 3a routing
# ---------------------------------------------------------------------

def route_after_sql_generation(
    state: GraphState,
) -> SQLGenerationRoute:
    """
    Route generated SQL to independent validation and execution.
    """

    if has_workflow_failed(state):
        return "failure"

    sql_result = state.get(
        "sql_result"
    )

    generated_sql = state.get(
        "generated_sql"
    )

    if sql_result is None:
        return "failure"

    if not isinstance(
        generated_sql,
        str,
    ):
        return "failure"

    if not generated_sql.strip():
        return "failure"

    if sql_result.requires_clarification:
        return "failure"

    return "sql_validation_execution"


# ---------------------------------------------------------------------
# SQL retry control
# ---------------------------------------------------------------------

def can_retry_sql(
    state: GraphState,
) -> bool:
    """
    Determine whether another SQL-generation attempt is allowed.

    Retry is allowed only when:

    - Validation or execution produced an error.
    - The retry count is lower than the configured maximum.
    """

    retry_count = state.get(
        "sql_retry_count",
        0,
    )

    max_retries = state.get(
        "max_sql_retries",
        0,
    )

    if not isinstance(
        retry_count,
        int,
    ):
        return False

    if not isinstance(
        max_retries,
        int,
    ):
        return False

    validation_errors = state.get(
        "sql_validation_errors",
        [],
    )

    execution_error = state.get(
        "sql_execution_error"
    )

    has_validation_errors = bool(
        validation_errors
    )

    has_execution_error = (
        isinstance(
            execution_error,
            str,
        )
        and bool(
            execution_error.strip()
        )
    )

    return (
        (
            has_validation_errors
            or has_execution_error
        )
        and retry_count < max_retries
    )


def route_after_sql_execution(
    state: GraphState,
) -> SQLExecutionRoute:
    """
    Decide what happens after Agent 3b.

    Routes:

    - analysis:
      SQL passed validation and executed successfully.

    - retry_sql:
      SQL failed and retry attempts remain.

    - failure:
      The error is non-retryable or retries are exhausted.
    """

    query_result = state.get(
        "query_result"
    )

    validation_errors = state.get(
        "sql_validation_errors",
        [],
    )

    execution_error = state.get(
        "sql_execution_error"
    )

    has_sql_errors = (
        bool(validation_errors)
        or (
            isinstance(
                execution_error,
                str,
            )
            and bool(
                execution_error.strip()
            )
        )
    )

    if (
        query_result is not None
        and not has_sql_errors
    ):
        return "analysis"

    if can_retry_sql(state):
        return "retry_sql"

    return "failure"


# ---------------------------------------------------------------------
# Agent 4 routing
# ---------------------------------------------------------------------

def route_after_analysis(
    state: GraphState,
) -> AnalysisRoute:
    """
    Route a validated AnalysisResult to Agent 5.
    """

    if has_workflow_failed(state):
        return "failure"

    analysis_result = state.get(
        "analysis_result"
    )

    if analysis_result is None:
        return "failure"

    return "reporting"


# ---------------------------------------------------------------------
# Agent 5 routing
# ---------------------------------------------------------------------

def route_after_reporting(
    state: GraphState,
) -> ReportingRoute:
    """
    Route Agent 5 output to deterministic validation.

    Agent 5 no longer completes the workflow directly. Every report
    must pass deterministic validation, LLM evaluation, and the final
    output guard.
    """

    if has_workflow_failed(state):
        return "failure"

    final_report = state.get(
        "final_report"
    )

    if final_report is None:
        return "failure"

    workflow_status = state.get(
        "workflow_status"
    )

    if (
        workflow_status
        != "validating_output"
    ):
        return "failure"

    return "deterministic_validation"


# ---------------------------------------------------------------------
# Deterministic output-validation routing
# ---------------------------------------------------------------------

def route_after_deterministic_validation(
    state: GraphState,
) -> DeterministicValidationRoute:
    """
    Route after objective output validation.

    Routes:

    - llm_evaluation:
      All deterministic checks passed.

    - output_block:
      Unsafe SQL, PII, secrets, unauthorized output, or prompt
      leakage was detected.

    - output_review:
      The report requires regeneration or manual quality review.

    - failure:
      Validation failed technically or returned inconsistent state.
    """

    if has_workflow_failed(state):
        return "failure"

    validation_result = state.get(
        "deterministic_validation_result"
    )

    if not isinstance(
        validation_result,
        DeterministicValidationResult,
    ):
        return "failure"

    decision = enum_value(
    validation_result.decision
)

    workflow_status = state.get(
        "workflow_status"
    )

    if (
        decision == "block"
        or workflow_status
        == "output_blocked"
    ):
        return "output_block"

    if decision in {
        "regenerate",
        "review",
    }:
        return "output_review"

    if (
        workflow_status
        == "output_review_required"
    ):
        return "output_review"

    if decision != "approve":
        return "failure"

    if (
        workflow_status
        != "evaluating_output"
    ):
        return "failure"

    return "llm_evaluation"


# ---------------------------------------------------------------------
# LLM output-evaluation routing
# ---------------------------------------------------------------------

def route_after_llm_output_evaluation(
    state: GraphState,
) -> LLMOutputEvaluationRoute:
    """
    Route after LLM-based output evaluation.

    The LLM evaluator cannot override a deterministic security block.
    """

    if has_workflow_failed(state):
        return "failure"

    deterministic_result = state.get(
        "deterministic_validation_result"
    )

    if not isinstance(
        deterministic_result,
        DeterministicValidationResult,
    ):
        return "failure"

    deterministic_decision = enum_value(
    deterministic_result.decision
)

    if deterministic_decision == "block":
        return "output_block"

    evaluation = state.get(
        "output_evaluation"
    )

    if not isinstance(
        evaluation,
        OutputEvaluation,
    ):
        return "failure"

    decision = enum_value(
    evaluation.decision
)

    workflow_status = state.get(
        "workflow_status"
    )

    if (
        decision == "block"
        or workflow_status
        == "output_blocked"
    ):
        return "output_block"

    if decision in {
        "regenerate",
        "review",
    }:
        return "output_review"

    if (
        workflow_status
        == "output_review_required"
    ):
        return "output_review"

    if decision != "approve":
        return "failure"

    if (
        workflow_status
        != "guarding_output"
    ):
        return "failure"

    return "output_guard"


# ---------------------------------------------------------------------
# Final output-guard routing
# ---------------------------------------------------------------------

def route_after_output_guard(
    state: GraphState,
) -> OutputGuardRoute:
    """
    Route the final sanitized output.

    Routes:

    - completed:
      Output was allowed or safely redacted.

    - output_block:
      Restricted information, secrets, unauthorized fields, or prompt
      leakage was detected.

    - output_review:
      The final response requires security or quality review.

    - failure:
      OutputGuard returned missing or inconsistent state.
    """

    if has_workflow_failed(state):
        return "failure"

    guard_result = state.get(
        "output_guard_result"
    )

    if not isinstance(
        guard_result,
        OutputGuardResult,
    ):
        return "failure"

    decision = enum_value(
    guard_result.decision
)

    workflow_status = state.get(
        "workflow_status"
    )

    if (
        decision == "block"
        or workflow_status
        == "output_blocked"
    ):
        return "output_block"

    if (
        decision == "review"
        or workflow_status
        == "output_review_required"
    ):
        return "output_review"

    if decision not in {
        "allow",
        "allow_redacted",
    }:
        return "failure"

    if (
        guard_result.sanitized_payload
        is None
    ):
        return "failure"

    if workflow_status != "completed":
        return "failure"

    return "completed"


# ---------------------------------------------------------------------
# Standalone routing smoke test
# ---------------------------------------------------------------------

def main() -> None:
    """
    Run routing smoke tests without calling Ollama or SQLite.
    """

    from models.schemas import (
        IntentResult,
        TimePeriod,
    )

    from security.input_guard import (
        InputGuard,
    )

    from security.models import (
        PIIDetectionResult,
        PromptRiskAssessment,
    )

    from workflow.state import (
        create_initial_state,
    )

    print("=" * 80)
    print("LANGGRAPH ROUTING SMOKE TEST")
    print("=" * 80)

    passed_tests = 0
    total_tests = 14

    try:
        safe_guard = InputGuard()

        # -------------------------------------------------------------
        # Test 1: Safe input-guard route
        # -------------------------------------------------------------

        safe_input_state = (
            create_initial_state(
                user_question=(
                    "Show revenue by product "
                    "category."
                )
            )
        )

        safe_guard_result = (
            safe_guard.evaluate(
                safe_input_state[
                    "original_user_question"
                ]
            )
        )

        safe_input_state[
            "input_guard_result"
        ] = safe_guard_result

        safe_input_state[
            "security_decision"
        ] = str(
            safe_guard_result.decision
        )

        safe_input_state[
            "user_question"
        ] = (
            safe_guard_result
            .sanitized_question
            or safe_input_state[
                "user_question"
            ]
        )

        safe_input_state[
            "workflow_status"
        ] = "evaluating_prompt_risk"

        safe_input_route = (
            route_after_input_guard(
                safe_input_state
            )
        )

        print(
            "Safe input route       : "
            f"{safe_input_route}"
        )

        if (
            safe_input_route
            == "prompt_guard"
        ):
            passed_tests += 1

        # -------------------------------------------------------------
        # Test 2: Blocked input-guard route
        # -------------------------------------------------------------

        blocked_input_state = (
            create_initial_state(
                user_question=(
                    "Delete all product records."
                )
            )
        )

        blocked_guard_result = (
            safe_guard.evaluate(
                blocked_input_state[
                    "original_user_question"
                ]
            )
        )

        blocked_input_state[
            "input_guard_result"
        ] = blocked_guard_result

        blocked_input_state[
            "security_decision"
        ] = str(
            blocked_guard_result.decision
        )

        blocked_input_state[
            "security_message"
        ] = (
            blocked_guard_result
            .user_message
        )

        blocked_input_state[
            "workflow_status"
        ] = "security_blocked"

        blocked_input_route = (
            route_after_input_guard(
                blocked_input_state
            )
        )

        print(
            "Blocked input route    : "
            f"{blocked_input_route}"
        )

        if (
            blocked_input_route
            == "security_block"
        ):
            passed_tests += 1

        # -------------------------------------------------------------
        # Test 3: Safe semantic-prompt route
        # -------------------------------------------------------------

        safe_prompt_state = (
            create_initial_state(
                user_question=(
                    "Show revenue by product "
                    "category."
                )
            )
        )

        safe_prompt_state[
            "input_guard_result"
        ] = safe_guard_result

        safe_prompt_state[
            "prompt_risk_assessment"
        ] = PromptRiskAssessment(
            prompt_injection_detected=False,
            instruction_override_attempt=False,
            system_prompt_extraction_attempt=False,
            unsafe_database_action=False,
            sql_validation_bypass_attempt=False,
            sensitive_data_request=False,
            secret_extraction_attempt=False,
            filesystem_access_attempt=False,
            environment_access_attempt=False,
            unapproved_tool_request=False,
            privilege_escalation_attempt=False,
            arbitrary_code_execution_attempt=False,
            risk_level="low",
            decision="allow",
            reason=(
                "The request is a normal "
                "read-only analytics question."
            ),
            confidence=0.98,
        )

        safe_prompt_state[
            "security_decision"
        ] = "allow"

        safe_prompt_state[
            "workflow_status"
        ] = "understanding_intent"

        safe_prompt_route = (
            route_after_prompt_guard(
                safe_prompt_state
            )
        )

        print(
            "Safe prompt route      : "
            f"{safe_prompt_route}"
        )

        if safe_prompt_route == "intent":
            passed_tests += 1

        # -------------------------------------------------------------
        # Test 4: Semantic-review route
        # -------------------------------------------------------------

        review_prompt_state = (
            create_initial_state(
                user_question=(
                    "Provide detailed customer "
                    "information."
                )
            )
        )

        review_prompt_state[
            "input_guard_result"
        ] = safe_guard.evaluate(
            review_prompt_state[
                "original_user_question"
            ]
        )

        review_prompt_state[
            "prompt_risk_assessment"
        ] = PromptRiskAssessment(
            prompt_injection_detected=False,
            instruction_override_attempt=False,
            system_prompt_extraction_attempt=False,
            unsafe_database_action=False,
            sql_validation_bypass_attempt=False,
            sensitive_data_request=True,
            secret_extraction_attempt=False,
            filesystem_access_attempt=False,
            environment_access_attempt=False,
            unapproved_tool_request=False,
            privilege_escalation_attempt=False,
            arbitrary_code_execution_attempt=False,
            risk_level="high",
            decision="review",
            reason=(
                "The requested information may "
                "contain protected customer data."
            ),
            confidence=0.72,
        )

        review_prompt_state[
            "security_decision"
        ] = "review"

        review_prompt_state[
            "security_review_required"
        ] = True

        review_prompt_state[
            "workflow_status"
        ] = "security_review_required"

        review_prompt_route = (
            route_after_prompt_guard(
                review_prompt_state
            )
        )

        print(
            "Prompt review route    : "
            f"{review_prompt_route}"
        )

        if (
            review_prompt_route
            == "security_review"
        ):
            passed_tests += 1

        # -------------------------------------------------------------
        # Test 5: Clarification route
        # -------------------------------------------------------------

        clarification_state = (
            create_initial_state(
                user_question=(
                    "Show the results."
                )
            )
        )

        clarification_intent = (
            IntentResult(
                original_question=(
                    "Show the results."
                ),
                rewritten_question=(
                    "Determine which metric and "
                    "grouping the user wants."
                ),
                analysis_type=(
                    "general_analysis"
                ),
                metrics=[],
                dimensions=[],
                filters=[],
                time_period=TimePeriod(),
                sort_order=None,
                limit=None,
                clarification_required=True,
                clarification_question=(
                    "Which metric and grouping "
                    "would you like to analyze?"
                ),
            )
        )

        clarification_state[
            "intent"
        ] = clarification_intent

        clarification_state[
            "clarification_required"
        ] = True

        clarification_state[
            "clarification_question"
        ] = (
            clarification_intent
            .clarification_question
        )

        clarification_state[
            "workflow_status"
        ] = "clarification_required"

        clarification_route = (
            route_after_intent(
                clarification_state
            )
        )

        print(
            "Clarification route    : "
            f"{clarification_route}"
        )

        if (
            clarification_route
            == "clarification"
        ):
            passed_tests += 1

        # -------------------------------------------------------------
        # Test 6: Technical failure route
        # -------------------------------------------------------------

        failure_state = (
            create_initial_state(
                user_question="Show sales."
            )
        )

        failure_state[
            "workflow_status"
        ] = "failed"

        failure_state[
            "error_message"
        ] = "Controlled test failure."

        failure_route = (
            route_after_intent(
                failure_state
            )
        )

        print(
            "Failure route          : "
            f"{failure_route}"
        )

        if failure_route == "failure":
            passed_tests += 1

        # -------------------------------------------------------------
        # Test 7: Schema route
        # -------------------------------------------------------------

        schema_state = (
            create_initial_state(
                user_question="Show sales."
            )
        )

        schema_state[
            "database_schema"
        ] = (
            "DATABASE: northwind.db\n"
            "DATABASE TYPE: SQLite"
        )

        schema_route = (
            route_after_schema(
                schema_state
            )
        )

        print(
            "Schema route           : "
            f"{schema_route}"
        )

        if (
            schema_route
            == "sql_generation"
        ):
            passed_tests += 1

        # -------------------------------------------------------------
        # Test 8: SQL retry route
        # -------------------------------------------------------------

        retry_state = create_initial_state(
            user_question="Show sales.",
            max_sql_retries=3,
        )

        retry_state[
            "sql_retry_count"
        ] = 1

        retry_state[
            "sql_execution_error"
        ] = "Unknown column."

        retry_route = (
            route_after_sql_execution(
                retry_state
            )
        )

        print(
            "SQL retry route        : "
            f"{retry_route}"
        )

        if retry_route == "retry_sql":
            passed_tests += 1

        # -------------------------------------------------------------
        # Test 9: Retry-exhausted route
        # -------------------------------------------------------------

        exhausted_state = (
            create_initial_state(
                user_question="Show sales.",
                max_sql_retries=3,
            )
        )

        exhausted_state[
            "sql_retry_count"
        ] = 3

        exhausted_state[
            "sql_execution_error"
        ] = "Unknown column."

        exhausted_route = (
            route_after_sql_execution(
                exhausted_state
            )
        )

        print(
            "Retry exhausted        : "
            f"{exhausted_route}"
        )

        if exhausted_route == "failure":
            passed_tests += 1

        # -------------------------------------------------------------
        # Test 10: Missing-result route
        # -------------------------------------------------------------

        missing_result_state = (
            create_initial_state(
                user_question="Show sales."
            )
        )

        missing_result_route = (
            route_after_sql_execution(
                missing_result_state
            )
        )

        print(
            "Missing result route   : "
            f"{missing_result_route}"
        )

        if (
            missing_result_route
            == "failure"
        ):
            passed_tests += 1

        # -------------------------------------------------------------
        # Test 11: Reporting route
        # -------------------------------------------------------------

        reporting_state = (
            create_initial_state(
                user_question=(
                    "Show the top 5 products "
                    "by revenue."
                )
            )
        )

        reporting_state[
            "final_report"
        ] = object()  # type: ignore[assignment]

        reporting_state[
            "workflow_status"
        ] = "validating_output"

        reporting_route = (
            route_after_reporting(
                reporting_state
            )
        )

        print(
            "Reporting route        : "
            f"{reporting_route}"
        )

        if (
            reporting_route
            == "deterministic_validation"
        ):
            passed_tests += 1

        # -------------------------------------------------------------
        # Test 12: Deterministic-validation approval route
        # -------------------------------------------------------------

        deterministic_state = (
            create_initial_state(
                user_question=(
                    "Show the top 5 products "
                    "by revenue."
                )
            )
        )

        deterministic_state[
            "deterministic_validation_result"
        ] = DeterministicValidationResult(
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
            verified_claims=[],
            unsupported_claims=[],
            privacy_findings=[],
            issues=[],
            decision="approve",
        )

        deterministic_state[
            "workflow_status"
        ] = "evaluating_output"

        deterministic_route = (
            route_after_deterministic_validation(
                deterministic_state
            )
        )

        print(
            "Deterministic route    : "
            f"{deterministic_route}"
        )

        if (
            deterministic_route
            == "llm_evaluation"
        ):
            passed_tests += 1

        # -------------------------------------------------------------
        # Test 13: LLM-evaluation approval route
        # -------------------------------------------------------------

        llm_state = dict(
            deterministic_state
        )

        llm_state[
            "output_evaluation"
        ] = OutputEvaluation(
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
            relevance_score=0.95,
            groundedness_score=0.95,
            accuracy_score=0.95,
            safety_score=1.0,
            overall_score=0.95,
            issues=[],
            decision="approve",
            explanation=(
                "The report is grounded, "
                "accurate, relevant, and safe."
            ),
        )

        llm_state[
            "workflow_status"
        ] = "guarding_output"

        llm_route = (
            route_after_llm_output_evaluation(
                llm_state
            )
        )

        print(
            "LLM evaluation route  : "
            f"{llm_route}"
        )

        if llm_route == "output_guard":
            passed_tests += 1

        # -------------------------------------------------------------
        # Test 14: Output-guard completion route
        # -------------------------------------------------------------

        output_guard_state = dict(
            llm_state
        )

        safe_pii_result = (
            PIIDetectionResult(
                contains_sensitive_data=False,
                highest_classification=None,
                findings=[],
                redacted_text=(
                    "Safe output content."
                ),
                decision="allow",
                user_message=(
                    "No sensitive information "
                    "was detected."
                ),
            )
        )

        output_guard_state[
            "output_guard_result"
        ] = OutputGuardResult(
            decision="allow",
            sanitized_payload={
                "final_report": {},
                "query_result": {},
                "technical_details": None,
            },
            redactions=[],
            pii_result=safe_pii_result,
            contains_secret_data=False,
            contains_prompt_leakage=False,
            contains_unauthorized_fields=False,
            issues=[],
            user_message=(
                "The response passed final "
                "output security screening."
            ),
        )

        output_guard_state[
            "workflow_status"
        ] = "completed"

        output_route = (
            route_after_output_guard(
                output_guard_state
            )
        )

        print(
            "Output guard route     : "
            f"{output_route}"
        )

        if output_route == "completed":
            passed_tests += 1

        # -------------------------------------------------------------
        # Final summary
        # -------------------------------------------------------------

        failed_tests = (
            total_tests - passed_tests
        )

        print("\n" + "-" * 80)
        print("ROUTING TEST SUMMARY")
        print("-" * 80)

        print(
            f"Total tests : {total_tests}"
        )

        print(
            f"Passed      : {passed_tests}"
        )

        print(
            f"Failed      : {failed_tests}"
        )

        if failed_tests:
            raise RuntimeError(
                "One or more routing smoke "
                "tests failed."
            )

        print("\n" + "=" * 80)
        print(
            "RESULT: LANGGRAPH ROUTING "
            "IS WORKING"
        )
        print("=" * 80)

    except Exception as exc:
        print("\n" + "=" * 80)
        print(
            "RESULT: LANGGRAPH ROUTING FAILED"
        )
        print("=" * 80)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()