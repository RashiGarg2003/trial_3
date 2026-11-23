"""
Shared LangGraph state for the Multi-Agent Data Analytics System.

GraphState carries validated data between workflow nodes.

The state does not execute agents or contain routing logic. It defines
only the information that can move through the LangGraph workflow.

Security design:

1. The original user question is retained only in runtime state.
2. The sanitized user question is passed to Agent 1.
3. Input-guard and prompt-risk decisions are stored as structured
   Pydantic models.
4. Blocked or review-required requests stop before database access.
5. Sensitive values must never be written to the execution trace.
"""

from typing import (
    Any,
    Literal,
    Optional,
    TypedDict,
)

from models.schemas import (
    AnalysisPlan,
    AnalysisResult,
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


WorkflowStatus = Literal[
    "initialized",

    # Input security
    "screening_input",
    "evaluating_prompt_risk",
    "security_blocked",
    "security_review_required",

    # Agent 1
    "understanding_intent",
    "clarification_required",

    # Agent 2
    "planning",

    # Agent 3
    "generating_sql",
    "validating_sql",
    "executing_sql",
    "retrying_sql",

    # Agent 4
    "analyzing",

    # Agent 5
    "reporting",

    # Output validation and security
    "validating_output",
    "evaluating_output",
    "guarding_output",
    "regenerating_report",
    "output_blocked",
    "output_review_required",

    # Final states
    "completed",
    "failed",
]


SecurityWorkflowDecision = Literal[
    "pending",
    "allow",
    "allow_redacted",
    "clarify",
    "block",
    "review",
]


class GraphState(TypedDict, total=False):
    """
    Shared state passed between LangGraph nodes.

    Fields are optional because every workflow node gradually adds its
    validated output.

    Security-sensitive fields such as original_user_question should
    remain inside runtime state and must not be written to logs,
    execution traces, or unrestricted API responses.
    """

    # -----------------------------------------------------------------
    # Original and sanitized request
    # -----------------------------------------------------------------

    original_user_question: str

    user_question: str

    question_was_redacted: bool

    # -----------------------------------------------------------------
    # Current workflow lifecycle
    # -----------------------------------------------------------------

    workflow_status: WorkflowStatus

    current_node: str

    # -----------------------------------------------------------------
    # Input security
    # -----------------------------------------------------------------

    input_guard_result: Optional[
        InputGuardResult
    ]

    prompt_risk_assessment: Optional[
        PromptRiskAssessment
    ]

    security_decision: SecurityWorkflowDecision

    security_message: Optional[str]

    security_review_required: bool

    security_risk_categories: list[str]

    # -----------------------------------------------------------------
    # Agent 1 output
    # -----------------------------------------------------------------

    intent: Optional[
        IntentResult
    ]

    # -----------------------------------------------------------------
    # Clarification route
    # -----------------------------------------------------------------

    clarification_required: bool

    clarification_question: Optional[str]

    # -----------------------------------------------------------------
    # Agent 2 output
    # -----------------------------------------------------------------

    plan: Optional[
        AnalysisPlan
    ]

    # -----------------------------------------------------------------
    # Schema context passed to Agent 3a
    # -----------------------------------------------------------------

    database_schema: Optional[str]

    # -----------------------------------------------------------------
    # Agent 3a output
    # -----------------------------------------------------------------

    sql_result: Optional[
        SQLGenerationResult
    ]

    generated_sql: Optional[str]

    # -----------------------------------------------------------------
    # Agent 3b validation and execution
    # -----------------------------------------------------------------

    sql_validation_errors: list[str]

    sql_execution_error: Optional[str]

    query_result: Optional[
        QueryExecutionResult
    ]

    # -----------------------------------------------------------------
    # SQL retry control
    # -----------------------------------------------------------------

    sql_retry_count: int

    max_sql_retries: int

    # -----------------------------------------------------------------
    # Agent 4 output
    # -----------------------------------------------------------------

    analysis_result: Optional[
        AnalysisResult
    ]

    # -----------------------------------------------------------------
    # Agent 5 output
    # -----------------------------------------------------------------

    final_report: Optional[
        FinalReport
    ]

    # -----------------------------------------------------------------
    # Final output validation and protection
    # -----------------------------------------------------------------

    deterministic_validation_result: Optional[
        DeterministicValidationResult
    ]

    output_evaluation: Optional[
        OutputEvaluation
    ]

    output_guard_result: Optional[
        OutputGuardResult
    ]

    # Sanitized payload approved by the final OutputGuard.
    # FastAPI should eventually serialize this payload instead of
    # rebuilding an unrestricted response from raw state values.
    sanitized_output_payload: Optional[
        dict[str, Any]
    ]

    # Report regeneration is bounded to prevent an infinite
    # Agent 5 and evaluator loop.
    report_regeneration_count: int

    max_report_regenerations: int

    output_validation_message: Optional[str]

    # -----------------------------------------------------------------
    # General workflow errors
    # -----------------------------------------------------------------

    error_message: Optional[str]

    error_type: Optional[str]

    # -----------------------------------------------------------------
    # Safe execution trace
    # -----------------------------------------------------------------

    execution_trace: list[
        dict[str, Any]
    ]


def create_initial_state(
    user_question: str,
    max_sql_retries: int = 3,
    max_report_regenerations: int = 1,
) -> GraphState:
    """
    Create a clean initial state for one workflow execution.

    Args:
        user_question:
            Natural-language analytical question.

        max_sql_retries:
            Maximum SQL correction attempts.

        max_report_regenerations:
            Maximum number of times Agent 5 may regenerate a report after
            deterministic or LLM evaluation rejects its quality.

    Returns:
        Initialized GraphState.

    Security note:
        The original question is stored in runtime state so the input
        guard can inspect it. After guardrail processing, Agent 1 must
        use user_question, which will contain the sanitized version.
    """

    if not isinstance(
        user_question,
        str,
    ):
        raise TypeError(
            "user_question must be a string."
        )

    cleaned_question = (
        user_question.strip()
    )

    if not cleaned_question:
        raise ValueError(
            "user_question cannot be empty."
        )

    if len(cleaned_question) > 2000:
        raise ValueError(
            "user_question cannot exceed "
            "2000 characters."
        )

    meaningful_characters = [
        character
        for character in cleaned_question
        if character.isalnum()
    ]

    if not meaningful_characters:
        raise ValueError(
            "user_question must contain meaningful "
            "letters or numbers."
        )

    if not isinstance(
        max_sql_retries,
        int,
    ):
        raise TypeError(
            "max_sql_retries must be an integer."
        )

    if max_sql_retries < 0:
        raise ValueError(
            "max_sql_retries cannot be negative."
        )

    if max_sql_retries > 5:
        raise ValueError(
            "max_sql_retries cannot exceed 5."
        )

    if not isinstance(
        max_report_regenerations,
        int,
    ):
        raise TypeError(
            "max_report_regenerations must be "
            "an integer."
        )

    if max_report_regenerations < 0:
        raise ValueError(
            "max_report_regenerations cannot "
            "be negative."
        )

    if max_report_regenerations > 2:
        raise ValueError(
            "max_report_regenerations cannot "
            "exceed 2."
        )

    return GraphState(
        # Original and current question initially contain the same
        # value. The input guard may replace user_question with a
        # redacted version.
        original_user_question=(
            cleaned_question
        ),
        user_question=cleaned_question,
        question_was_redacted=False,

        # Workflow lifecycle
        workflow_status="initialized",
        current_node="start",

        # Security state
        input_guard_result=None,
        prompt_risk_assessment=None,
        security_decision="pending",
        security_message=None,
        security_review_required=False,
        security_risk_categories=[],

        # Agent 1
        intent=None,

        # Clarification
        clarification_required=False,
        clarification_question=None,

        # Agent 2
        plan=None,

        # Schema
        database_schema=None,

        # Agent 3a
        sql_result=None,
        generated_sql=None,

        # Agent 3b
        sql_validation_errors=[],
        sql_execution_error=None,
        query_result=None,

        # Retry control
        sql_retry_count=0,
        max_sql_retries=(
            max_sql_retries
        ),

        # Agent 4
        analysis_result=None,

        # Agent 5
        final_report=None,

        # Final output validation and protection
        deterministic_validation_result=None,
        output_evaluation=None,
        output_guard_result=None,
        sanitized_output_payload=None,

        # Bounded report-regeneration control
        report_regeneration_count=0,
        max_report_regenerations=(
            max_report_regenerations
        ),

        output_validation_message=None,

        # General errors
        error_message=None,
        error_type=None,
        # Safe trace
        execution_trace=[
            {
                "node": "start",
                "status": "initialized",
                "message": (
                    "Workflow state created."
                ),
            }
        ],
    )


def add_trace_event(
    state: GraphState,
    node: str,
    status: str,
    message: str,
) -> list[dict[str, Any]]:
    """
    Return a new execution trace containing one additional event.

    Sensitive values, prompts, SQL result rows, secrets, personal
    identifiers, database schemas, access tokens, and raw questions
    must not be written to this trace.
    """

    if not isinstance(node, str):
        raise TypeError(
            "node must be a string."
        )

    if not isinstance(status, str):
        raise TypeError(
            "status must be a string."
        )

    if not isinstance(message, str):
        raise TypeError(
            "message must be a string."
        )

    cleaned_node = node.strip()
    cleaned_status = status.strip()
    cleaned_message = message.strip()

    if not cleaned_node:
        raise ValueError(
            "node cannot be empty."
        )

    if not cleaned_status:
        raise ValueError(
            "status cannot be empty."
        )

    if not cleaned_message:
        raise ValueError(
            "message cannot be empty."
        )

    # Avoid accidentally storing unlimited model or exception output
    # inside the execution trace.
    maximum_message_length = 500

    if (
        len(cleaned_message)
        > maximum_message_length
    ):
        cleaned_message = (
            cleaned_message[
                :maximum_message_length
            ]
            + "..."
        )

    existing_trace = list(
        state.get(
            "execution_trace",
            [],
        )
    )

    existing_trace.append(
        {
            "node": cleaned_node,
            "status": cleaned_status,
            "message": cleaned_message,
        }
    )

    return existing_trace


def main() -> None:
    """
    Run a standalone GraphState smoke test.
    """

    print("=" * 80)
    print("LANGGRAPH STATE SMOKE TEST")
    print("=" * 80)

    try:
        state = create_initial_state(
            user_question=(
                "Show the top 5 products "
                "by revenue."
            ),
            max_sql_retries=3,
            max_report_regenerations=1,
        )

        state["execution_trace"] = (
            add_trace_event(
                state=state,
                node="state_test",
                status="completed",
                message=(
                    "GraphState validation succeeded."
                ),
            )
        )

        print(
            f"Original question      : "
            f"{state['original_user_question']}"
        )

        print(
            f"Current question       : "
            f"{state['user_question']}"
        )

        print(
            f"Question redacted      : "
            f"{state['question_was_redacted']}"
        )

        print(
            f"Security decision      : "
            f"{state['security_decision']}"
        )

        print(
            f"Security review        : "
            f"{state['security_review_required']}"
        )

        print(
            f"Workflow status        : "
            f"{state['workflow_status']}"
        )

        print(
            f"Current node           : "
            f"{state['current_node']}"
        )

        print(
            f"SQL retries            : "
            f"{state['sql_retry_count']}/"
            f"{state['max_sql_retries']}"
        )

        print(
            f"Report regenerations   : "
            f"{state['report_regeneration_count']}/"
            f"{state['max_report_regenerations']}"
        )

        print(
            f"Deterministic result   : "
            f"{state['deterministic_validation_result']}"
        )

        print(
            f"LLM evaluation         : "
            f"{state['output_evaluation']}"
        )

        print(
            f"Output guard result    : "
            f"{state['output_guard_result']}"
        )

        print(
            f"Trace events           : "
            f"{len(state['execution_trace'])}"
        )

        if (
            state["original_user_question"]
            != state["user_question"]
        ):
            raise RuntimeError(
                "The initial original and current "
                "questions should match."
            )

        if (
            state["security_decision"]
            != "pending"
        ):
            raise RuntimeError(
                "Initial security decision must "
                "be pending."
            )

        if state["input_guard_result"] is not None:
            raise RuntimeError(
                "Initial input guard result must "
                "be None."
            )

        if (
            state["prompt_risk_assessment"]
            is not None
        ):
            raise RuntimeError(
                "Initial prompt-risk assessment "
                "must be None."
            )

        print("\nExecution trace:")

        if (
            state[
                "deterministic_validation_result"
            ]
            is not None
        ):
            raise RuntimeError(
                "Initial deterministic validation "
                "result must be None."
            )

        if state["output_evaluation"] is not None:
            raise RuntimeError(
                "Initial output evaluation must be None."
            )

        if state["output_guard_result"] is not None:
            raise RuntimeError(
                "Initial output guard result must be None."
            )

        if (
            state["report_regeneration_count"]
            != 0
        ):
            raise RuntimeError(
                "Initial report-regeneration count "
                "must be zero."
            )

        if (
            state["max_report_regenerations"]
            != 1
        ):
            raise RuntimeError(
                "Unexpected report-regeneration limit."
            )

        for event in state[
            "execution_trace"
        ]:
            print(
                f"- {event['node']} | "
                f"{event['status']} | "
                f"{event['message']}"
            )

        print("\n" + "=" * 80)
        print(
            "RESULT: LANGGRAPH STATE IS WORKING"
        )
        print("=" * 80)

    except Exception as exc:
        print("\n" + "=" * 80)
        print(
            "RESULT: LANGGRAPH STATE FAILED"
        )
        print("=" * 80)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()