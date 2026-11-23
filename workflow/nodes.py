"""
LangGraph nodes for the Multi-Agent Data Analytics System.

Integrated workflow nodes:

1. Input and prompt-injection security guards
2. Agent 1: Intent Understanding and clarification handling
3. Agent 2: Planning
4. Database schema preparation
5. Agent 3a: SQL Generation
6. Agent 3b: SQL Validation and read-only Execution
7. Agent 4: Analysis
8. Agent 5: Reporting
9. Deterministic output validation, LLM evaluation, and output guard
10. Controlled workflow failure handling

The compiled workflow currently executes one SQL retrieval/analysis pass
per request, with a bounded SQL correction loop. It does not yet execute
each SQL-requiring plan step as an independent analysis cycle.

Each node:

- Reads values from GraphState.
- Calls an existing project component.
- Returns a partial GraphState update.
- Adds a safe execution-trace event.
- Handles errors without exposing secrets or database rows.
"""

from typing import Any, Optional
from agents.analysis_agent import AnalysisAgent
from agents.intent_agent import IntentAgent
from agents.planning_agent import PlanningAgent
from agents.reporting_agent import ReportingAgent
from agents.sql_agent import SQLAgent
from security.models import UserRole
from security.schema_policy import (
    SchemaPolicyValidator,
)

from database.db_manager import DatabaseManager
from database.schema_manager import SchemaManager


from security.output_guard import OutputGuard

from models.schemas import (
    AnalysisPlan,
    IntentResult,
)

from tools.sql_execution_tool import (
    SQLExecutionOutcome,
    SQLExecutionTool,
)
from tools.sql_validator import SQLValidator
from security.input_guard import InputGuard
from security.prompt_injection_guard import (
    PromptInjectionGuard,
)
from security.models import (
    DeterministicValidationResult,
    InputGuardResult,
    OutputEvaluation,
    OutputGuardResult,
    PromptRiskAssessment,
    UserRole,
)


from evaluation.deterministic_validator import (
    DeterministicOutputValidator,
)
from evaluation.llm_evaluator import (
    LLMOutputEvaluator,
)

from workflow.state import (
    GraphState,
    add_trace_event,
)

def enum_value(
    value: Any,
) -> Any:
    """
    Return the primitive value stored in an Enum.

    Examples:

        EvaluationDecision.APPROVE -> "approve"
        SecurityDecision.ALLOW -> "allow"
        "approve" -> "approve"
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


class WorkflowNodes:
    """
    Collection of all LangGraph workflow nodes.

    Existing agents and tools are injected into this class so that
    LangGraph does not create duplicate Ollama clients or database
    connections for every node.
    """

    def __init__(
    self,
    input_guard: InputGuard,
    prompt_injection_guard: PromptInjectionGuard,
    intent_agent: IntentAgent,
    planning_agent: PlanningAgent,
    sql_agent: SQLAgent,
    analysis_agent: AnalysisAgent,
    reporting_agent: ReportingAgent,
    deterministic_validator: DeterministicOutputValidator,
    llm_output_evaluator: LLMOutputEvaluator,
    output_guard: OutputGuard,
    database_manager: DatabaseManager,
    schema_manager: SchemaManager,
    sql_execution_tool: SQLExecutionTool,
) -> None:

        """
        Initialize workflow nodes using existing project components.
        """
        if not isinstance(
            input_guard,
            InputGuard,
        ):
            raise TypeError(
                "input_guard must be an "
                "InputGuard instance."
            )

        if not isinstance(
            prompt_injection_guard,
            PromptInjectionGuard,
        ):
            raise TypeError(
                "prompt_injection_guard must be a "
                "PromptInjectionGuard instance."
            )


        if not isinstance(
            intent_agent,
            IntentAgent,
        ):
            raise TypeError(
                "intent_agent must be an IntentAgent instance."
            )

        if not isinstance(
            planning_agent,
            PlanningAgent,
        ):
            raise TypeError(
                "planning_agent must be a PlanningAgent instance."
            )

        if not isinstance(
            sql_agent,
            SQLAgent,
        ):
            raise TypeError(
                "sql_agent must be a SQLAgent instance."
            )

        if not isinstance(
            analysis_agent,
            AnalysisAgent,
                ):
                    raise TypeError(
                "analysis_agent must be an "
                "AnalysisAgent instance."
            )

        if not isinstance(
            reporting_agent,
            ReportingAgent,
                ):
                    raise TypeError(
                "reporting_agent must be a "
                "ReportingAgent instance."
            )   

        if not isinstance(
            deterministic_validator,
            DeterministicOutputValidator,
        ):
            raise TypeError(
                "deterministic_validator must be a "
                "DeterministicOutputValidator instance."
            )

        if not isinstance(
            llm_output_evaluator,
            LLMOutputEvaluator,
        ):
            raise TypeError(
                "llm_output_evaluator must be an "
                "LLMOutputEvaluator instance."
            )

        if not isinstance(
            output_guard,
            OutputGuard,
        ):
            raise TypeError(
                "output_guard must be an "
                "OutputGuard instance."
            )

        if not isinstance(
            database_manager,
            DatabaseManager,
        ):
            raise TypeError(
                "database_manager must be a "
                "DatabaseManager instance."
            )

        if not isinstance(
            schema_manager,
            SchemaManager,
        ):
            raise TypeError(
                "schema_manager must be a "
                "SchemaManager instance."
            )

        if not isinstance(
            sql_execution_tool,
            SQLExecutionTool,
        ):
            raise TypeError(
                "sql_execution_tool must be a "
                "SQLExecutionTool instance."
            )

        self.input_guard = input_guard
        self.prompt_injection_guard = (
            prompt_injection_guard
        )

        self.intent_agent = intent_agent
        self.planning_agent = planning_agent
        self.sql_agent = sql_agent
        self.analysis_agent = analysis_agent
        self.reporting_agent = reporting_agent

        self.deterministic_validator = (
            deterministic_validator
        )

        self.llm_output_evaluator = (
            llm_output_evaluator
        )

        self.output_guard = output_guard

        self.database_manager = database_manager
        self.schema_manager = schema_manager
        self.sql_execution_tool = sql_execution_tool

    def _get_original_question(
        self,
        state: GraphState,
    ) -> str:
        """
        Return the original unsanitized question.

        This value is used only by the deterministic input guard. It must
        not be written to the execution trace or unrestricted API output.
        """

        question = state.get(
            "original_user_question"
        )

        if not isinstance(
            question,
            str,
        ):
            raise ValueError(
                "GraphState does not contain a valid "
                "original_user_question."
            )

        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError(
                "GraphState original_user_question "
                "cannot be empty."
            )

        return cleaned_question

    def _get_required_question(
        self,
        state: GraphState,
    ) -> str:
        """
        Return the validated user question from GraphState.
        """

        question = state.get(
            "user_question"
        )

        if not isinstance(
            question,
            str,
        ):
            raise ValueError(
                "GraphState does not contain a valid "
                "user_question."
            )

        cleaned_question = question.strip()

        if not cleaned_question:
            raise ValueError(
                "GraphState user_question cannot be empty."
            )

        return cleaned_question

    def _get_required_intent(
        self,
        state: GraphState,
    ) -> IntentResult:
        """
        Return the validated IntentResult from GraphState.
        """

        intent = state.get("intent")

        if not isinstance(
            intent,
            IntentResult,
        ):
            raise ValueError(
                "GraphState does not contain a valid "
                "IntentResult."
            )

        return intent

    def _get_required_plan(
        self,
        state: GraphState,
    ) -> AnalysisPlan:
        """
        Return the validated AnalysisPlan from GraphState.
        """

        plan = state.get("plan")

        if not isinstance(
            plan,
            AnalysisPlan,
        ):
            raise ValueError(
                "GraphState does not contain a valid "
                "AnalysisPlan."
            )

        if not plan.steps:
            raise ValueError(
                "The AnalysisPlan contains no steps."
            )

        return plan

    def _get_required_schema(
        self,
        state: GraphState,
    ) -> str:
        """
        Return the prompt-ready database schema.
        """

        database_schema = state.get(
            "database_schema"
        )

        if not isinstance(
            database_schema,
            str,
        ):
            raise ValueError(
                "GraphState does not contain a valid "
                "database schema."
            )

        cleaned_schema = (
            database_schema.strip()
        )

        if not cleaned_schema:
            raise ValueError(
                "GraphState database schema is empty."
            )

        return cleaned_schema

    def _error_update(
        self,
        state: GraphState,
        node_name: str,
        error: Exception,
    ) -> dict[str, Any]:
        """
        Build a controlled workflow failure update.

        Prompts, secrets, complete schema text, and database result rows
        are not added to the execution trace.
        """

        error_type = type(error).__name__

        trace = add_trace_event(
            state=state,
            node=node_name,
            status="failed",
            message=(
                f"{node_name} failed with "
                f"{error_type}."
            ),
        )

        return {
            "workflow_status": "failed",
            "current_node": node_name,
            "error_type": error_type,
            "error_message": str(error),
            "execution_trace": trace,
        }

    def input_guard_node(
            self,
            state: GraphState,
        ) -> dict[str, Any]:
            """
            Execute deterministic input-security screening.

            Reads:
                original_user_question

            Writes:
                input_guard_result
                user_question
                question_was_redacted
                security_decision
                security_message
                security_review_required
                security_risk_categories
                workflow_status
                current_node
                execution_trace

            Clear destructive, privacy, secret-extraction, filesystem, and
            other prohibited requests are stopped before Agent 1.
            """

            node_name = "input_guard"

            try:
                original_question = (
                    self._get_original_question(
                        state
                    )
                )

                guard_result = (
                    self.input_guard.evaluate(
                        original_question
                    )
                )

                if not isinstance(
                    guard_result,
                    InputGuardResult,
                ):
                    raise TypeError(
                        "InputGuard did not return "
                        "InputGuardResult."
                    )

                decision = str(
                    guard_result.decision
                )

                if decision == "block":
                    workflow_status = (
                        "security_blocked"
                    )

                    safe_question = (
                        state.get(
                            "user_question",
                            original_question,
                        )
                    )

                    trace_status = "blocked"

                    trace_message = (
                        "The request was blocked by "
                        "deterministic input-security controls."
                    )

                elif decision == "review":
                    workflow_status = (
                        "security_review_required"
                    )

                    safe_question = (
                        guard_result.sanitized_question
                        or state.get(
                            "user_question",
                            original_question,
                        )
                    )

                    trace_status = (
                        "review_required"
                    )

                    trace_message = (
                        "The request requires security review "
                        "before analytical processing."
                    )

                else:
                    safe_question = (
                        guard_result.sanitized_question
                    )

                    if not isinstance(
                        safe_question,
                        str,
                    ) or not safe_question.strip():
                        raise RuntimeError(
                            "InputGuard approved the request "
                            "without a sanitized question."
                        )

                    safe_question = (
                        safe_question.strip()
                    )

                    workflow_status = (
                        "evaluating_prompt_risk"
                    )

                    trace_status = "completed"

                    if decision == "allow_redacted":
                        trace_message = (
                            "Input security screening passed "
                            "after sensitive values were redacted."
                        )

                    else:
                        trace_message = (
                            "Input security screening passed."
                        )

                question_was_redacted = (
                    isinstance(safe_question, str)
                    and safe_question
                    != original_question
                )

                trace = add_trace_event(
                    state=state,
                    node=node_name,
                    status=trace_status,
                    message=trace_message,
                )

                return {
                    "input_guard_result": (
                        guard_result
                    ),
                    "user_question": safe_question,
                    "question_was_redacted": (
                        question_was_redacted
                    ),
                    "security_decision": decision,
                    "security_message": (
                        guard_result.user_message
                    ),
                    "security_review_required": (
                        decision == "review"
                    ),
                    "security_risk_categories": list(
                        guard_result.risk_categories
                    ),
                    "workflow_status": (
                        workflow_status
                    ),
                    "current_node": node_name,
                    "error_type": None,
                    "error_message": None,
                    "execution_trace": trace,
                }

            except Exception as exc:
                return self._error_update(
                    state=state,
                    node_name=node_name,
                    error=exc,
                )

    def prompt_injection_guard_node(
            self,
            state: GraphState,
        ) -> dict[str, Any]:
            """
            Execute semantic prompt-risk classification.

            Only the sanitized question is sent to the local LLM.

            Reads:
                user_question
                input_guard_result

            Writes:
                prompt_risk_assessment
                security_decision
                security_message
                security_review_required
                workflow_status
                current_node
                execution_trace
            """

            node_name = "prompt_injection_guard"

            try:
                question = self._get_required_question(
                    state
                )

                input_guard_result = state.get(
                    "input_guard_result"
                )

                if not isinstance(
                    input_guard_result,
                    InputGuardResult,
                ):
                    raise ValueError(
                        "Prompt-risk evaluation requires "
                        "InputGuardResult."
                    )

                deterministic_decision = str(
                    input_guard_result.decision
                )

                if deterministic_decision in {
                    "block",
                    "review",
                }:
                    raise ValueError(
                        "Prompt-risk evaluation cannot run "
                        "after a blocking or review decision."
                    )

                assessment = (
                    self.prompt_injection_guard.assess(
                        question=question,
                        deterministic_result=(
                            input_guard_result
                        ),
                    )
                )

                if not isinstance(
                    assessment,
                    PromptRiskAssessment,
                ):
                    raise TypeError(
                        "PromptInjectionGuard did not return "
                        "PromptRiskAssessment."
                    )

                semantic_decision = str(
                    assessment.decision
                )

                if semantic_decision == "block":
                    final_decision = "block"

                    workflow_status = (
                        "security_blocked"
                    )

                    security_message = (
                        "The request was blocked because "
                        "semantic security screening detected "
                        "an unsafe instruction."
                    )

                    trace_status = "blocked"

                    trace_message = (
                        "The request was blocked by semantic "
                        "prompt-risk screening."
                    )

                elif semantic_decision == "review":
                    final_decision = "review"

                    workflow_status = (
                        "security_review_required"
                    )

                    security_message = (
                        "The request requires security review "
                        "before analytical processing."
                    )

                    trace_status = (
                        "review_required"
                    )

                    trace_message = (
                        "Semantic prompt-risk screening "
                        "requires human review."
                    )

                else:
                    # Preserve the fact that deterministic PII screening
                    # redacted the question.
                    if (
                        deterministic_decision
                        == "allow_redacted"
                    ):
                        final_decision = (
                            "allow_redacted"
                        )

                    else:
                        final_decision = "allow"

                    workflow_status = (
                        "understanding_intent"
                    )

                    security_message = (
                        "The request passed input and "
                        "prompt-risk screening."
                    )

                    trace_status = "completed"

                    trace_message = (
                        "Semantic prompt-risk screening passed."
                    )

                trace = add_trace_event(
                    state=state,
                    node=node_name,
                    status=trace_status,
                    message=trace_message,
                )

                return {
                    "prompt_risk_assessment": (
                        assessment
                    ),
                    "security_decision": (
                        final_decision
                    ),
                    "security_message": (
                        security_message
                    ),
                    "security_review_required": (
                        final_decision == "review"
                    ),
                    "workflow_status": (
                        workflow_status
                    ),
                    "current_node": node_name,
                    "error_type": None,
                    "error_message": None,
                    "execution_trace": trace,
                }

            except Exception as exc:
                return self._error_update(
                    state=state,
                    node_name=node_name,
                    error=exc,
                )

    def intent_node(
        self,
        state: GraphState,
    ) -> dict[str, Any]:
        """
        Execute Agent 1: Intent Understanding.

        Reads:
            user_question

        Writes:
            intent
            clarification_required
            clarification_question
            workflow_status
            current_node
            execution_trace

        A request proceeds to planning only when the structured intent
        contains sufficient analytical context. Otherwise, the workflow
        asks the user for clarification before SQL generation.
        """

        node_name = "intent_agent"

        try:
            question = (
                self._get_required_question(
                    state
                )
            )

            intent = (
                self.intent_agent.understand(
                    question
                )
            )

            if intent is None:
                raise RuntimeError(
                    "IntentAgent returned no IntentResult."
                )

            # ---------------------------------------------------------
            # Deterministic clarification safeguard
            # ---------------------------------------------------------
            #
            # Do not depend only on the LLM's clarification decision.
            # Verify whether the structured intent contains enough context
            # for planning and SQL generation.
            #
            # This is generalizable because it evaluates the Pydantic
            # output fields rather than matching fixed user phrases.

            has_metric = bool(
                [
                    metric
                    for metric in intent.metrics
                    if str(metric).strip()
                ]
            )

            has_dimension = bool(
                [
                    dimension
                    for dimension in intent.dimensions
                    if str(dimension).strip()
                ]
            )

            has_filter = bool(
                intent.filters
            )

            has_actionable_intent = any(
                [
                    has_metric,
                    has_dimension,
                    has_filter,
                ]
            )

            # ---------------------------------------------------------
            # Force clarification for an underspecified request
            # ---------------------------------------------------------

            if (
                not intent.clarification_required
                and not has_actionable_intent
            ):
                intent = intent.model_copy(
                    update={
                        "clarification_required": True,
                        "clarification_question": (
                            "What metric would you like to analyze, "
                            "and how should the results be grouped, "
                            "filtered, or compared?"
                        ),
                    }
                )

            # Ensure that clarification always contains a useful question.
            if intent.clarification_required:
                clarification_question = (
                    intent.clarification_question
                )

                if (
                    not isinstance(
                        clarification_question,
                        str,
                    )
                    or not clarification_question.strip()
                ):
                    clarification_question = (
                        "What metric would you like to analyze, "
                        "and how should the results be grouped, "
                        "filtered, or compared?"
                    )

                    intent = intent.model_copy(
                        update={
                            "clarification_question": (
                                clarification_question
                            ),
                        }
                    )

                next_status = (
                    "clarification_required"
                )

                trace_status = (
                    "clarification_required"
                )

                trace_message = (
                    "Agent 1 requires additional "
                    "analytical context."
                )

            else:
                next_status = "planning"

                trace_status = "completed"

                trace_message = (
                    "Intent understood and validated."
                )

            trace = add_trace_event(
                state=state,
                node=node_name,
                status=trace_status,
                message=trace_message,
            )

            return {
                "intent": intent,
                "clarification_required": (
                    intent.clarification_required
                ),
                "clarification_question": (
                    intent.clarification_question
                ),
                "workflow_status": (
                    next_status
                ),
                "current_node": node_name,
                "error_type": None,
                "error_message": None,
                "execution_trace": trace,
            }

        except Exception as exc:
            error_update = self._error_update(
                state=state,
                node_name=node_name,
                error=exc,
            )
            return error_update
    
    def clarification_node(
            self,
            state: GraphState,
        ) -> dict[str, Any]:
            """
            End the current execution with a focused follow-up question.
            """

            node_name = "clarification"

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
                    "What metric would you like to analyze, "
                    "and how should the results be grouped "
                    "or filtered?"
                )

            trace = add_trace_event(
                state=state,
                node=node_name,
                status="waiting_for_user",
                message=(
                    "Workflow paused because additional "
                    "analytical context is required."
                ),
            )

            return {
                "clarification_required": True,
                "clarification_question": (
                    clarification_question
                ),
                "workflow_status": (
                    "clarification_required"
                ),
                "current_node": node_name,
                "error_type": None,
                "error_message": None,
                "execution_trace": trace,
            }

    def planning_node(
        self,
        state: GraphState,
    ) -> dict[str, Any]:
        """
        Execute Agent 2: Planning.

        Reads:
            intent

        Writes:
            plan
            workflow_status
            execution_trace
        """

        node_name = "planning_agent"

        try:
            intent = self._get_required_intent(
                state
            )

            if intent.clarification_required:
                raise ValueError(
                    "Planning cannot begin while "
                    "clarification is required."
                )

            plan = (
                self.planning_agent.create_plan(
                    intent=intent
                )
            )

            trace = add_trace_event(
                state=state,
                node=node_name,
                status="completed",
                message=(
                    "Analytical plan created and "
                    "validated."
                ),
            )

            return {
                "plan": plan,
                "workflow_status": (
                    "generating_sql"
                ),
                "current_node": node_name,
                "error_type": None,
                "error_message": None,
                "execution_trace": trace,
            }

        except Exception as exc:
            return self._error_update(
                state=state,
                node_name=node_name,
                error=exc,
            )

    def schema_node(
        self,
        state: GraphState,
    ) -> dict[str, Any]:
        """
        Prepare the actual SQLite schema for Agent 3a.

        The database schema is extracted dynamically. No table or
        column names are hardcoded.
        """

        node_name = "schema_manager"

        try:
            connection_status = (
                self.database_manager
                .test_connection()
            )

            if not connection_status:
                raise ConnectionError(
                    "Database connection test failed."
                )

            integrity_status = (
                self.database_manager
                .run_integrity_check()
            )

            if integrity_status.lower() != "ok":
                raise RuntimeError(
                    "Database integrity check failed: "
                    f"{integrity_status}"
                )

            database_schema = (
                self.schema_manager
                .get_schema_prompt(
                    include_views=False
                )
            )

            cleaned_schema = (
                database_schema.strip()
            )

            if not cleaned_schema:
                raise RuntimeError(
                    "SchemaManager returned an "
                    "empty database schema."
                )

            trace = add_trace_event(
                state=state,
                node=node_name,
                status="completed",
                message=(
                    "Database verified and schema "
                    "prepared for SQL generation."
                ),
            )

            return {
                "database_schema": cleaned_schema,
                "workflow_status": (
                    "generating_sql"
                ),
                "current_node": node_name,
                "error_type": None,
                "error_message": None,
                "execution_trace": trace,
            }

        except Exception as exc:
            return self._error_update(
                state=state,
                node_name=node_name,
                error=exc,
            )

    def _build_sql_retry_feedback(
        self,
        state: GraphState,
    ) -> Optional[str]:
        """
        Combine Agent 3b validation and database errors.

        This feedback is returned to Agent 3a during a retry.
        """

        feedback: list[str] = []

        validation_errors = state.get(
            "sql_validation_errors",
            [],
        )

        if isinstance(
            validation_errors,
            list,
        ):
            for error in validation_errors:
                if (
                    isinstance(error, str)
                    and error.strip()
                ):
                    feedback.append(
                        error.strip()
                    )

        execution_error = state.get(
            "sql_execution_error"
        )

        if (
            isinstance(execution_error, str)
            and execution_error.strip()
        ):
            feedback.append(
                execution_error.strip()
            )

        if not feedback:
            return None

        return "\n".join(
            f"- {error}"
            for error in feedback
        )

    def sql_generation_node(
        self,
        state: GraphState,
    ) -> dict[str, Any]:
        """
        Execute Agent 3a: SQL Generation.

        On the first attempt, Agent 3a receives:

        - Question
        - Intent
        - Plan
        - Actual schema

        During retry, Agent 3a additionally receives:

        - Previous SQL
        - Agent 3b validation error
        - Actual SQLite compilation or execution error
        """

        node_name = "sql_generation_agent"

        try:
            question = (
                self._get_required_question(
                    state
                )
            )

            intent = self._get_required_intent(
                state
            )

            if intent.clarification_required:
                raise ValueError(
                    "SQL generation cannot begin while "
                    "clarification is required."
                )

            plan = self._get_required_plan(
                state
            )

            database_schema = (
                self._get_required_schema(
                    state
                )
            )

            retry_count = state.get(
                "sql_retry_count",
                0,
            )

            if not isinstance(
                retry_count,
                int,
            ):
                raise ValueError(
                    "sql_retry_count must be "
                    "an integer."
                )

            previous_sql: Optional[str] = None
            previous_error: Optional[str] = None

            if retry_count > 0:
                stored_sql = state.get(
                    "generated_sql"
                )

                if (
                    isinstance(stored_sql, str)
                    and stored_sql.strip()
                ):
                    previous_sql = (
                        stored_sql.strip()
                    )

                previous_error = (
                    self._build_sql_retry_feedback(
                        state
                    )
                )

                if previous_error is None:
                    raise ValueError(
                        "SQL retry was requested but "
                        "no Agent 3b error feedback "
                        "is available."
                    )

            sql_result = (
                self.sql_agent.generate_sql(
                    user_question=question,
                    intent=intent,
                    plan=plan,
                    database_schema=database_schema,
                    previous_sql=previous_sql,
                    previous_error=previous_error,
                )
            )

            generated_sql = (
                sql_result.sql.strip()
            )

            if not generated_sql:
                raise RuntimeError(
                    "Agent 3a returned empty SQL."
                )

            if retry_count > 0:
                trace_message = (
                    "Agent 3a generated corrected SQL "
                    f"for retry {retry_count}."
                )
            else:
                trace_message = (
                    "Agent 3a generated one read-only "
                    "SQLite query."
                )

            trace = add_trace_event(
                state=state,
                node=node_name,
                status="completed",
                message=trace_message,
            )

            return {
                "sql_result": sql_result,
                "generated_sql": generated_sql,

                # The previous error has been consumed by Agent 3a.
                # Agent 3b will populate these fields again if the
                # corrected query still fails.
                "sql_validation_errors": [],
                "sql_execution_error": None,
                "query_result": None,

                "workflow_status": (
                    "validating_sql"
                ),
                "current_node": node_name,
                "error_type": None,
                "error_message": None,
                "execution_trace": trace,
            }

        except Exception as exc:
            return self._error_update(
                state=state,
                node_name=node_name,
                error=exc,
            )

    def _sql_retry_update(
        self,
        state: GraphState,
        outcome: SQLExecutionOutcome,
    ) -> dict[str, Any]:
        """
        Return a retry update after Agent 3b rejects SQL.
        """

        node_name = (
            "sql_validation_execution"
        )

        retry_count = state.get(
            "sql_retry_count",
            0,
        )

        if not isinstance(
            retry_count,
            int,
        ):
            retry_count = 0

        next_retry_count = (
            retry_count + 1
        )

        trace = add_trace_event(
            state=state,
            node=node_name,
            status="retry_required",
            message=(
                "Agent 3b rejected SQL during "
                f"'{outcome.stage}'. "
                f"Retry count is {next_retry_count}."
            ),
        )

        return {
            "query_result": None,
            "sql_validation_errors": list(
                outcome.validation_errors
            ),
            "sql_execution_error": (
                outcome.execution_error
            ),
            "sql_retry_count": (
                next_retry_count
            ),
            "workflow_status": (
                "retrying_sql"
            ),
            "current_node": node_name,
            "error_type": None,
            "error_message": None,
            "execution_trace": trace,
        }

    def sql_validation_execution_node(
        self,
        state: GraphState,
    ) -> dict[str, Any]:
        """
        Execute Agent 3b: SQL Validation and Execution.

        Validation sequence:

        1. SQL AST safety validation
        2. Approved table and view validation
        3. SQLite EXPLAIN QUERY PLAN
        4. Read-only database execution
        5. QueryExecutionResult validation
        """

        node_name = (
            "sql_validation_execution"
        )

        try:
            generated_sql = state.get(
                "generated_sql"
            )

            if not isinstance(
                generated_sql,
                str,
            ):
                raise ValueError(
                    "GraphState does not contain "
                    "generated SQL."
                )

            cleaned_sql = (
                generated_sql.strip()
            )

            if not cleaned_sql:
                raise ValueError(
                    "GraphState generated SQL is empty."
                )

            outcome = (
                self.sql_execution_tool
                .validate_and_execute(
                    cleaned_sql
                )
            )

            if not outcome.success:
                if outcome.retryable:
                    return self._sql_retry_update(
                        state=state,
                        outcome=outcome,
                    )

                raise RuntimeError(
                    "Agent 3b rejected the SQL with "
                    "a non-retryable error."
                )

            if outcome.query_result is None:
                raise RuntimeError(
                    "Agent 3b reported success but "
                    "returned no query result."
                )

            trace = add_trace_event(
                state=state,
                node=node_name,
                status="completed",
                message=(
                    "SQL passed independent validation "
                    "and executed successfully."
                ),
            )

            return {
                "query_result": (
                    outcome.query_result
                ),
                "sql_validation_errors": [],
                "sql_execution_error": None,
                "workflow_status": "analyzing",
                "current_node": node_name,
                "error_type": None,
                "error_message": None,
                "execution_trace": trace,
            }

        except Exception as exc:
            return self._error_update(
                state=state,
                node_name=node_name,
                error=exc,
            )

    def analysis_node(
        self,
        state: GraphState,
    ) -> dict[str, Any]:
        """
        Execute Agent 4: Analysis.

        Agent 4 receives the actual query result returned by Agent 3b.

        Reads:
            user_question
            intent
            plan
            query_result

        Writes:
            analysis_result
            workflow_status
            current_node
            execution_trace
        """

        node_name = "analysis_agent"

        try:
            question = self._get_required_question(
                state
            )

            intent = self._get_required_intent(
                state
            )

            if intent.clarification_required:
                raise ValueError(
                    "Analysis cannot begin while "
                    "clarification is required."
                )

            plan = self._get_required_plan(
                state
            )

            query_result = state.get(
                "query_result"
            )

            if query_result is None:
                raise ValueError(
                    "GraphState does not contain "
                    "QueryExecutionResult."
                )

            analysis_result = (
                self.analysis_agent.analyze(
                    user_question=question,
                    intent=intent,
                    plan=plan,
                    query_result=query_result,
                )
            )

            trace = add_trace_event(
                state=state,
                node=node_name,
                status="completed",
                message=(
                    "Query results were analyzed using "
                    "deterministic statistics and "
                    "evidence-grounded interpretation."
                ),
            )

            return {
                "analysis_result": analysis_result,
                "workflow_status": "reporting",
                "current_node": node_name,
                "error_type": None,
                "error_message": None,
                "execution_trace": trace,
            }

        except Exception as exc:
            return self._error_update(
                state=state,
                node_name=node_name,
                error=exc,
            )

    def reporting_node(
        self,
        state: GraphState,
    ) -> dict[str, Any]:
        """
        Execute Agent 5: Reporting.

        Agent 5 converts the validated analysis and query result into
        a business-friendly final report.

        Reads:
            user_question
            intent
            analysis_result
            query_result

        Writes:
            final_report
            workflow_status
            current_node
            execution_trace
        """

        node_name = "reporting_agent"

        try:
            question = self._get_required_question(
                state
            )

            intent = self._get_required_intent(
                state
            )

            if intent.clarification_required:
                raise ValueError(
                    "Reporting cannot begin while "
                    "clarification is required."
                )

            analysis_result = state.get(
                "analysis_result"
            )

            if analysis_result is None:
                raise ValueError(
                    "GraphState does not contain "
                    "AnalysisResult."
                )

            query_result = state.get(
                "query_result"
            )

            if query_result is None:
                raise ValueError(
                    "GraphState does not contain "
                    "QueryExecutionResult."
                )

            final_report = (
                self.reporting_agent.create_report(
                    user_question=question,
                    intent=intent,
                    analysis_result=analysis_result,
                    query_result=query_result,
                )
            )

            trace = add_trace_event(
                state=state,
                node=node_name,
                status="completed",
                message=(
                    "Business report, chart specification, "
                    "root-cause indicators, and recommendations "
                    "were generated."
                ),
            )

            return {
                "final_report": final_report,

                # Clear previous validation results whenever a new report
                # is generated.
                "deterministic_validation_result": None,
                "output_evaluation": None,
                "output_guard_result": None,
                "sanitized_output_payload": None,
                "output_validation_message": None,

                "workflow_status": "validating_output",
                "current_node": node_name,
                "error_type": None,
                "error_message": None,
                "execution_trace": trace,
            }

        except Exception as exc:
            return self._error_update(
                state=state,
                node_name=node_name,
                error=exc,
            )

    def _build_output_technical_details(
        self,
        state: GraphState,
    ) -> dict[str, Any]:
        """
        Build technical metadata for OutputGuard filtering.

        This dictionary must not include:

        - Original user question
        - Complete database schema
        - Prompts
        - Environment variables
        - Credentials
        - Query-result rows
        """

        sql_result = state.get(
            "sql_result"
        )

        referenced_tables: list[str] = []

        if sql_result is not None:
            referenced_tables = list(
                sql_result.referenced_tables
            )

        intent = state.get("intent")
        plan = state.get("plan")

        return {
            "generated_sql": state.get(
                "generated_sql"
            ),
            "referenced_tables": (
                referenced_tables
            ),
            "sql_retry_count": state.get(
                "sql_retry_count",
                0,
            ),
            "max_sql_retries": state.get(
                "max_sql_retries",
                0,
            ),
            "intent": (
                intent.model_dump(
                    mode="json"
                )
                if intent is not None
                else None
            ),
            "plan": (
                plan.model_dump(
                    mode="json"
                )
                if plan is not None
                else None
            ),
            "execution_trace": list(
                state.get(
                    "execution_trace",
                    [],
                )
            ),
        }

    def deterministic_validation_node(
        self,
        state: GraphState,
    ) -> dict[str, Any]:
        """
        Validate Agent 5 output using objective checks.

        Reads:
            user_question
            generated_sql
            query_result
            analysis_result
            final_report

        Writes:
            deterministic_validation_result
            workflow_status
            output_validation_message
            execution_trace
        """

        node_name = (
            "deterministic_output_validation"
        )

        try:
            question = self._get_required_question(
                state
            )

            generated_sql = state.get(
                "generated_sql"
            )

            if not isinstance(
                generated_sql,
                str,
            ) or not generated_sql.strip():
                raise ValueError(
                    "Deterministic validation requires "
                    "generated SQL."
                )

            query_result = state.get(
                "query_result"
            )

            if query_result is None:
                raise ValueError(
                    "Deterministic validation requires "
                    "QueryExecutionResult."
                )

            analysis_result = state.get(
                "analysis_result"
            )

            if analysis_result is None:
                raise ValueError(
                    "Deterministic validation requires "
                    "AnalysisResult."
                )

            final_report = state.get(
                "final_report"
            )

            if final_report is None:
                raise ValueError(
                    "Deterministic validation requires "
                    "FinalReport."
                )

            validation_result = (
                self.deterministic_validator.validate(
                    user_question=question,
                    generated_sql=generated_sql,
                    query_result=query_result,
                    analysis_result=analysis_result,
                    final_report=final_report,
                )
            )

            if not isinstance(
                validation_result,
                DeterministicValidationResult,
            ):
                raise TypeError(
                    "DeterministicOutputValidator did not "
                    "return DeterministicValidationResult."
                )

            decision = enum_value(
                validation_result.decision
            )

            if decision == "approve":
                workflow_status = (
                    "evaluating_output"
                )

                trace_status = "completed"

                trace_message = (
                    "The report passed deterministic "
                    "output validation."
                )

                validation_message = (
                    "Deterministic output validation passed."
                )

            elif decision == "block":
                workflow_status = (
                    "output_blocked"
                )

                trace_status = "blocked"

                trace_message = (
                    "The report was blocked by deterministic "
                    "output validation."
                )

                validation_message = (
                    "The generated report contains an unsafe "
                    "or unauthorized output."
                )

            else:
                # Regeneration is not connected yet. Until the bounded
                # retry route is added, regeneration or review decisions
                # stop safely for review.
                workflow_status = (
                    "output_review_required"
                )

                trace_status = (
                    "review_required"
                )

                trace_message = (
                    "The report requires correction or "
                    "quality review."
                )

                validation_message = (
                    "The generated report did not satisfy "
                    "all deterministic quality checks."
                )

            trace = add_trace_event(
                state=state,
                node=node_name,
                status=trace_status,
                message=trace_message,
            )

            return {
                "deterministic_validation_result": (
                    validation_result
                ),
                "output_validation_message": (
                    validation_message
                ),
                "workflow_status": (
                    workflow_status
                ),
                "current_node": node_name,
                "error_type": None,
                "error_message": None,
                "execution_trace": trace,
            }

        except Exception as exc:
            return self._error_update(
                state=state,
                node_name=node_name,
                error=exc,
            )

    def llm_output_evaluation_node(
            self,
            state: GraphState,
        ) -> dict[str, Any]:
            """
            Evaluate report relevance, groundedness, accuracy, and safety.

            This node runs only after deterministic validation approves the
            report.
            """

            node_name = "llm_output_evaluation"

            try:
                question = self._get_required_question(
                    state
                )

                generated_sql = state.get(
                    "generated_sql"
                )

                if not isinstance(
                    generated_sql,
                    str,
                ) or not generated_sql.strip():
                    raise ValueError(
                        "LLM output evaluation requires "
                        "generated SQL."
                    )

                query_result = state.get(
                    "query_result"
                )

                if query_result is None:
                    raise ValueError(
                        "LLM output evaluation requires "
                        "QueryExecutionResult."
                    )

                analysis_result = state.get(
                    "analysis_result"
                )

                if analysis_result is None:
                    raise ValueError(
                        "LLM output evaluation requires "
                        "AnalysisResult."
                    )

                final_report = state.get(
                    "final_report"
                )

                if final_report is None:
                    raise ValueError(
                        "LLM output evaluation requires "
                        "FinalReport."
                    )

                deterministic_result = state.get(
                    "deterministic_validation_result"
                )

                if not isinstance(
                    deterministic_result,
                    DeterministicValidationResult,
                ):
                    raise ValueError(
                        "LLM output evaluation requires "
                        "DeterministicValidationResult."
                    )

                evaluation = (
                    self.llm_output_evaluator.evaluate(
                        user_question=question,
                        generated_sql=generated_sql,
                        query_result=query_result,
                        analysis_result=analysis_result,
                        final_report=final_report,
                        deterministic_result=(
                            deterministic_result
                        ),
                    )
                )

                if not isinstance(
                    evaluation,
                    OutputEvaluation,
                ):
                    raise TypeError(
                        "LLMOutputEvaluator did not return "
                        "OutputEvaluation."
                    )

                decision = enum_value(
                    evaluation.decision
                )

                if decision == "approve":
                    workflow_status = (
                        "guarding_output"
                    )

                    trace_status = "completed"

                    trace_message = (
                        "The report passed semantic output "
                        "quality evaluation."
                    )

                    validation_message = (
                        "LLM output evaluation approved "
                        "the report."
                    )

                elif decision == "block":
                    workflow_status = (
                        "output_blocked"
                    )

                    trace_status = "blocked"

                    trace_message = (
                        "The report was blocked by semantic "
                        "output evaluation."
                    )

                    validation_message = (
                        "The generated report contains an unsafe "
                        "or ungrounded output."
                    )

                else:
                    workflow_status = (
                        "output_review_required"
                    )

                    trace_status = (
                        "review_required"
                    )

                    trace_message = (
                        "The report requires correction or "
                        "quality review after semantic evaluation."
                    )

                    validation_message = (
                        "The generated report did not satisfy "
                        "all semantic quality thresholds."
                    )

                trace = add_trace_event(
                    state=state,
                    node=node_name,
                    status=trace_status,
                    message=trace_message,
                )

                return {
                    "output_evaluation": evaluation,
                    "output_validation_message": (
                        validation_message
                    ),
                    "workflow_status": (
                        workflow_status
                    ),
                    "current_node": node_name,
                    "error_type": None,
                    "error_message": None,
                    "execution_trace": trace,
                }

            except Exception as exc:
                return self._error_update(
                    state=state,
                    node_name=node_name,
                    error=exc,
                )

    def output_guard_node(
        self,
        state: GraphState,
    ) -> dict[str, Any]:
        """
        Apply final privacy, secret, authorization, and redaction checks.

        The current workflow uses the analyst role until authentication
        and role-based identity are integrated.
        """

        node_name = "output_guard"

        try:
            final_report = state.get(
                "final_report"
            )

            if final_report is None:
                raise ValueError(
                    "OutputGuard requires FinalReport."
                )

            query_result = state.get(
                "query_result"
            )

            if query_result is None:
                raise ValueError(
                    "OutputGuard requires "
                    "QueryExecutionResult."
                )

            deterministic_result = state.get(
                "deterministic_validation_result"
            )

            if not isinstance(
                deterministic_result,
                DeterministicValidationResult,
            ):
                raise ValueError(
                    "OutputGuard requires "
                    "DeterministicValidationResult."
                )

            output_evaluation = state.get(
                "output_evaluation"
            )

            if not isinstance(
                output_evaluation,
                OutputEvaluation,
            ):
                raise ValueError(
                    "OutputGuard requires "
                    "OutputEvaluation."
                )

            # Agent 3b already returned only validated output columns.
            # Schema-policy-specific redaction can later narrow this set.
            allowed_output_columns = set(
                query_result.columns
            )

            guard_result = (
                self.output_guard.guard(
                    final_report=final_report,
                    query_result=query_result,
                    deterministic_result=(
                        deterministic_result
                    ),
                    output_evaluation=(
                        output_evaluation
                    ),
                    role=UserRole.ANALYST,
                    allowed_output_columns=(
                        allowed_output_columns
                    ),
                    technical_details=(
                        self._build_output_technical_details(
                            state
                        )
                    ),
                )
            )

            if not isinstance(
                guard_result,
                OutputGuardResult,
            ):
                raise TypeError(
                    "OutputGuard did not return "
                    "OutputGuardResult."
                )

            decision = enum_value(
                guard_result.decision
            )

            if decision in {
                "allow",
                "allow_redacted",
            }:
                if (
                    guard_result
                    .sanitized_payload
                    is None
                ):
                    raise RuntimeError(
                        "OutputGuard approved the output "
                        "without a sanitized payload."
                    )

                workflow_status = "completed"

                trace_status = "completed"

                trace_message = (
                    "The final response passed output "
                    "security screening."
                    if decision == "allow"
                    else (
                        "The final response passed output "
                        "security screening after redaction."
                    )
                )

            elif decision == "block":
                workflow_status = (
                    "output_blocked"
                )

                trace_status = "blocked"

                trace_message = (
                    "The final response was blocked by "
                    "output security controls."
                )

            else:
                workflow_status = (
                    "output_review_required"
                )

                trace_status = (
                    "review_required"
                )

                trace_message = (
                    "The final response requires security "
                    "or quality review."
                )

            trace = add_trace_event(
                state=state,
                node=node_name,
                status=trace_status,
                message=trace_message,
            )

            return {
                "output_guard_result": (
                    guard_result
                ),
                "sanitized_output_payload": (
                    guard_result
                    .sanitized_payload
                ),
                "output_validation_message": (
                    guard_result.user_message
                ),
                "workflow_status": (
                    workflow_status
                ),
                "current_node": node_name,
                "error_type": None,
                "error_message": None,
                "execution_trace": trace,
            }

        except Exception as exc:
            return self._error_update(
                state=state,
                node_name=node_name,
                error=exc,
            )

    def security_block_node(
            self,
            state: GraphState,
        ) -> dict[str, Any]:
            """
            End a prohibited request safely.

            This is a controlled security outcome, not an application error.
            Agent 1 and all database-related nodes must not run afterward.
            """

            node_name = "security_block"

            security_message = state.get(
                "security_message"
            )

            if not isinstance(
                security_message,
                str,
            ) or not security_message.strip():
                security_message = (
                    "The request cannot be processed "
                    "because it violates application "
                    "security policy."
                )

            trace = add_trace_event(
                state=state,
                node=node_name,
                status="blocked",
                message=(
                    "Workflow stopped before Agent 1 "
                    "because of a security-policy decision."
                ),
            )

            return {
                "security_decision": "block",
                "security_message": (
                    security_message
                ),
                "security_review_required": False,
                "workflow_status": (
                    "security_blocked"
                ),
                "current_node": node_name,

                # Security blocking is deliberately separate from a
                # technical workflow failure.
                "error_type": None,
                "error_message": None,

                "execution_trace": trace,
            }

    def security_review_node(
            self,
            state: GraphState,
        ) -> dict[str, Any]:
            """
            Pause an ambiguous or protected request for review.

            The current MVP stops safely rather than sending the request to
            Agent 1 or the database.
            """

            node_name = "security_review"

            security_message = state.get(
                "security_message"
            )

            if not isinstance(
                security_message,
                str,
            ) or not security_message.strip():
                security_message = (
                    "The request requires security review "
                    "before it can be processed."
                )

            trace = add_trace_event(
                state=state,
                node=node_name,
                status="review_required",
                message=(
                    "Workflow stopped before Agent 1 "
                    "because security review is required."
                ),
            )

            return {
                "security_decision": "review",
                "security_message": (
                    security_message
                ),
                "security_review_required": True,
                "workflow_status": (
                    "security_review_required"
                ),
                "current_node": node_name,
                "error_type": None,
                "error_message": None,
                "execution_trace": trace,
            }
            
    def failure_node(
        self,
        state: GraphState,
    ) -> dict[str, Any]:
        """
        End the workflow through a controlled failure route.
        """

        node_name = "failure"

        error_type = state.get(
            "error_type"
        )

        error_message = state.get(
            "error_message"
        )

        sql_retry_count = state.get(
            "sql_retry_count",
            0,
        )

        max_sql_retries = state.get(
            "max_sql_retries",
            0,
        )

        if (
            not error_message
            and isinstance(
                sql_retry_count,
                int,
            )
            and isinstance(
                max_sql_retries,
                int,
            )
            and sql_retry_count
            >= max_sql_retries
        ):
            error_type = (
                "SQLRetryLimitExceeded"
            )

            error_message = (
                "SQL validation or execution failed "
                "after the maximum number of retries."
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
            else "The workflow could not be completed."
        )

        trace = add_trace_event(
            state=state,
            node=node_name,
            status="failed",
            message=(
                "Workflow ended through the "
                "controlled failure route."
            ),
        )

        return {
            "workflow_status": "failed",
            "current_node": node_name,
            "error_type": safe_error_type,
            "error_message": safe_error_message,
            "execution_trace": trace,
        }


def build_workflow_nodes() -> WorkflowNodes:
    """
    Build all current workflow nodes.

    Ollama model allocation:

    - llama3.2:latest:
      Intent Agent and Planning Agent

    - sqlcoder:latest:
      SQL Generation Agent

    Database access remains read-only.
    """

    database_manager = DatabaseManager(
        max_rows=1000,
    )

    schema_manager = SchemaManager(
        database_manager=database_manager
    )

    input_guard = InputGuard()

    prompt_injection_guard = (
        PromptInjectionGuard(
            model="llama3.2:latest",
            host="http://localhost:11434",
            context_length=4096,
            max_output_tokens=700,
            maximum_attempts=2,
            blocking_confidence_threshold=0.80,
            review_confidence_threshold=0.60,
        )
    )


    intent_agent = IntentAgent(
        model="llama3.2:latest",
        host="http://localhost:11434",
    )

    planning_agent = PlanningAgent(
        model="llama3.2:latest",
        host="http://localhost:11434",
    )

    sql_agent = SQLAgent(
        model="sqlcoder:latest",
        host="http://localhost:11434",
        dialect="sqlite",
        context_length=8192,
        max_output_tokens=1024,
        max_attempts=3,
    )

    analysis_agent = AnalysisAgent(
    model="llama3.2:latest",
    host="http://localhost:11434",
    context_length=8192,
    max_output_tokens=1500,
    maximum_sample_rows=50,
)

    reporting_agent = ReportingAgent(
    model="llama3.2:latest",
    host="http://localhost:11434",
    context_length=8192,
    max_output_tokens=1800,
    maximum_sample_rows=30,
)

    deterministic_validator = (
        DeterministicOutputValidator(
            numeric_tolerance=0.01,
            percentage_tolerance=0.10,
            maximum_claims_to_validate=100,
        )
    )

    llm_output_evaluator = (
        LLMOutputEvaluator(
            model="llama3.2:latest",
            host="http://localhost:11434",
            context_length=8192,
            max_output_tokens=1200,
            maximum_attempts=2,
            maximum_sample_rows=30,
            approval_threshold=0.90,
            regeneration_threshold=0.80,
        )
    )

    output_guard = OutputGuard(
        maximum_output_rows=1000
    )
    sql_validator = SQLValidator(
        dialect="sqlite"
    )

    schema_policy_validator = (
        SchemaPolicyValidator(
            database_manager=database_manager,
            sql_validator=sql_validator,
            maximum_result_rows=1000,
        )
    )

    sql_execution_tool = SQLExecutionTool(
        database_manager=database_manager,
        sql_validator=sql_validator,
        schema_policy_validator=(
            schema_policy_validator
        ),
        role=UserRole.ANALYST,
        explain_before_execution=True,
    )

    return WorkflowNodes(
    input_guard=input_guard,
    prompt_injection_guard=(
        prompt_injection_guard
    ),
    intent_agent=intent_agent,
    planning_agent=planning_agent,
    sql_agent=sql_agent,
    analysis_agent=analysis_agent,
    reporting_agent=reporting_agent,
    deterministic_validator=(
        deterministic_validator
    ),
    llm_output_evaluator=(
        llm_output_evaluator
    ),
    output_guard=output_guard,
    database_manager=database_manager,
    schema_manager=schema_manager,
    sql_execution_tool=sql_execution_tool,
)

def main() -> None:
    """
    Run all currently implemented nodes sequentially.

    This smoke test does not yet compile the retry loop in LangGraph.
    """

    from workflow.state import (
        create_initial_state,
    )

    print("=" * 80)
    print("WORKFLOW NODES SMOKE TEST")
    print("=" * 80)

    question = (
        "Show the top 5 products by revenue."
    )

    try:
        nodes = build_workflow_nodes()

        state = create_initial_state(
            user_question=question,
            max_sql_retries=3,
        )

        print(f"Question: {question}")

        node_sequence = [
            (
                "Input Guard",
                nodes.input_guard_node,
            ),
            (
                "Prompt Injection Guard",
                nodes.prompt_injection_guard_node,
            ),
            (
                "Agent 1",
                nodes.intent_node,
            ),
            (
                "Agent 2",
                nodes.planning_node,
            ),
            (
                "Schema Manager",
                nodes.schema_node,
            ),
            (
                "Agent 3a",
                nodes.sql_generation_node,
            ),
            (
                "Agent 3b",
                nodes.sql_validation_execution_node,
            ),
            (
                "Agent 4",
                nodes.analysis_node,
            ),
            (
                "Agent 5",
                nodes.reporting_node,
            ),

            (
                "Deterministic Output Validation",
                nodes.deterministic_validation_node,
            ),
            (
                "LLM Output Evaluation",
                nodes.llm_output_evaluation_node,
            ),
            (
                "Final Output Guard",
                nodes.output_guard_node,
            ),
        ]

        for display_name, node_function in (
            node_sequence
        ):
            if state.get(
                "workflow_status"
            ) == "failed":
                break

            if (
                display_name != "Agent 1"
                and state.get(
                    "clarification_required",
                    False,
                )
            ):
                break

            print(
                f"\nRunning {display_name}..."
            )

            update = node_function(state)
            state.update(update)

            if state.get(
                "workflow_status"
            ) in {
                "output_blocked",
                "output_review_required",
            }:
                break

            if (
                state.get("workflow_status")
                == "security_blocked"
            ):
                block_update = (
                    nodes.security_block_node(
                        state
                    )
                )

                state.update(block_update)
                break

            if (
                state.get("workflow_status")
                == "security_review_required"
            ):
                review_update = (
                    nodes.security_review_node(
                        state
                    )
                )
                state.update(review_update)
                break

            if (
                state.get("workflow_status")
                == "retrying_sql"
            ):
                break

            print(
                f"Status: "
                f"{state.get('workflow_status')}"
            )

        if state.get(
            "clarification_required",
            False,
        ):
            print(
                "\nClarification question:"
            )
            print(
                state.get(
                    "clarification_question"
                )
            )

        elif state.get(
            "workflow_status"
        ) == "security_blocked":
            print("\n" + "=" * 80)
            print(
                "RESULT: REQUEST BLOCKED BY SECURITY POLICY"
            )
            print("=" * 80)
            print(
                state.get(
                    "security_message",
                    (
                        "The request was blocked "
                        "by security policy."
                    ),
                )
            )

        elif state.get(
            "workflow_status"
        ) == "security_review_required":
            print("\n" + "=" * 80)
            print(
                "RESULT: SECURITY REVIEW REQUIRED"
            )
            print("=" * 80)
            print(
                state.get(
                    "security_message",
                    (
                        "The request requires security review "
                        "before analytical processing."
                    ),
                )
            )

        elif state.get(
            "workflow_status"
        ) == "retrying_sql":
            print(
                "\nAgent 3b requested SQL retry."
            )
            print(
                f"Retry count: "
                f"{state.get('sql_retry_count')}"
            )
            print(
                f"Validation errors: "
                f"{state.get('sql_validation_errors')}"
            )
            print(
                f"Execution error: "
                f"{state.get('sql_execution_error')}"
            )

        elif state.get(
            "workflow_status"
        ) == "failed":
            raise RuntimeError(
                state.get(
                    "error_message",
                    "Workflow nodes failed.",
                )
            )

        else:
            query_result = state.get(
                "query_result"
            )

            if query_result is None:
                raise RuntimeError(
                    "Agent 3b returned no query result."
                )

            print("\nGenerated SQL:")
            print(
                state.get(
                    "generated_sql"
                )
            )

            print("\nQuery result:")
            print(
                f"Columns       : "
                f"{query_result.columns}"
            )
            print(
                f"Rows returned : "
                f"{query_result.row_count}"
            )
            print(
                f"Truncated     : "
                f"{query_result.truncated}"
            )

            for row in query_result.rows:
                print(row)

            analysis_result = state.get(
    "analysis_result"
)

            if analysis_result is not None:
                print("\nAnalysis result:")
                print(
        analysis_result.model_dump_json(
            indent=2
        )
    )

            final_report = state.get(
                "final_report"
            )

            if final_report is not None:
                print("\nFinal report:")
                print(
                    final_report.model_dump_json(
                        indent=2
                    )
                )

        deterministic_result = state.get(
            "deterministic_validation_result"
        )

        if deterministic_result is not None:
            print("\nDeterministic validation:")
            print(
                deterministic_result.model_dump_json(
                    indent=2
                )
            )

        output_evaluation = state.get(
            "output_evaluation"
        )

        if output_evaluation is not None:
            print("\nLLM output evaluation:")
            print(
                output_evaluation.model_dump_json(
                    indent=2
                )
            )

        output_guard_result = state.get(
            "output_guard_result"
        )

        if output_guard_result is not None:
            print("\nOutput guard:")
            print(
                f"Decision : "
                f"{output_guard_result.decision}"
            )
            print(
                f"Message  : "
                f"{output_guard_result.user_message}"
            )
            print(
                f"Redactions: "
                f"{len(output_guard_result.redactions)}"
            )

        print(
            f"\nFinal status: "
            f"{state.get('workflow_status')}"
        )

        print("\nExecution trace:")

        for event in state.get(
            "execution_trace",
            [],
        ):
            print(
                f"- {event['node']} | "
                f"{event['status']} | "
                f"{event['message']}"
            )

        print("\n" + "=" * 80)
        print("RESULT: WORKFLOW NODES ARE WORKING")
        print("=" * 80)

    except Exception as exc:
        print("\n" + "=" * 80)
        print("RESULT: WORKFLOW NODES FAILED")
        print("=" * 80)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()