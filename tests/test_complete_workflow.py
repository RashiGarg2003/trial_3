"""
End-to-end tests for the complete Multi-Agent Data Analytics workflow.

The tests execute the compiled LangGraph workflow through:

1. Agent 1: Intent Understanding
2. Agent 2: Planning
3. Agent 3a: SQL Generation
4. Agent 3b: SQL Validation and Execution
5. Agent 4: Analysis
6. Agent 5: Reporting

The tests verify:

- Successful workflow completion
- Structured Pydantic outputs
- Read-only SQL
- Query execution
- Evidence-grounded analysis
- Final business reporting
- Valid chart columns
- SQL retry limits
- Clarification routing
- Safe execution tracing

Run from the project root:

    python -m tests.test_complete_workflow

Important:
    Ollama must be running and the following models must be installed:

    - llama3.2:latest
    - sqlcoder:latest
"""

import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from models.schemas import (
    AnalysisPlan,
    AnalysisResult,
    FinalReport,
    IntentResult,
    QueryExecutionResult,
    SQLGenerationResult,
)
from tools.sql_validator import SQLValidator
from workflow.graph import run_workflow
from workflow.state import GraphState


SUCCESS_TEST_CASES: list[dict[str, Any]] = [
    {
        "name": "Product revenue ranking",
        "question": (
            "Show the top 5 products by revenue."
        ),
        "expected_analysis_types": [
            "ranking",
        ],
        "expected_maximum_rows": 5,
    },
    {
        "name": "Orders by country",
        "question": (
            "Show the number of orders by country."
        ),
        "expected_analysis_types": [
            "aggregation",
            "ranking",
        ],
        "expected_maximum_rows": None,
    },
    {
        "name": "Revenue by product category",
        "question": (
            "Compare revenue across product categories."
        ),
        "expected_analysis_types": [
            "comparison",
            "aggregation",
        ],
        "expected_maximum_rows": None,
    },
    {
        "name": "Monthly revenue trend",
        "question": (
            "Show the monthly revenue trend."
        ),
        "expected_analysis_types": [
            "trend_analysis",
        ],
        "expected_maximum_rows": None,
    },
]


CLARIFICATION_TEST_CASE = {
    "name": "Incomplete analytical request",
    "question": "Analyze the results.",
}


REQUIRED_SUCCESS_TRACE_NODES = [
    "start",
    "intent_agent",
    "planning_agent",
    "schema_manager",
    "sql_generation_agent",
    "sql_validation_execution",
    "analysis_agent",
    "reporting_agent",
]


SENSITIVE_TRACE_TERMS = [
    "password=",
    "secret=",
    "api_key=",
    "token=",
    "authorization:",
    "bearer ",
    "actual sqlite schema:",
    "sample_rows",
]


def print_test_header(
    test_number: int,
    test_name: str,
) -> None:
    """
    Print a readable test heading.
    """

    print("\n" + "=" * 80)
    print(
        f"TEST {test_number}: {test_name}"
    )
    print("=" * 80)


def print_workflow_summary(
    state: GraphState,
    elapsed_seconds: float,
) -> None:
    """
    Print a concise workflow summary without exposing the full schema.
    """

    print(
        f"Workflow status       : "
        f"{state.get('workflow_status')}"
    )

    print(
        f"Current node          : "
        f"{state.get('current_node')}"
    )

    print(
        f"Clarification required: "
        f"{state.get('clarification_required')}"
    )

    print(
        f"SQL retry count       : "
        f"{state.get('sql_retry_count', 0)}/"
        f"{state.get('max_sql_retries', 0)}"
    )

    print(
        f"Execution time        : "
        f"{elapsed_seconds:.2f} seconds"
    )

    intent = state.get("intent")

    if intent is not None:
        print(
            f"Analysis type         : "
            f"{intent.analysis_type}"
        )

        print(
            f"Metrics               : "
            f"{intent.metrics}"
        )

        print(
            f"Dimensions            : "
            f"{intent.dimensions}"
        )

    generated_sql = state.get(
        "generated_sql"
    )

    if generated_sql:
        print("\nGenerated SQL:")
        print(generated_sql)

    query_result = state.get(
        "query_result"
    )

    if query_result is not None:
        print("\nQuery result:")
        print(
            f"Columns               : "
            f"{query_result.columns}"
        )

        print(
            f"Rows returned         : "
            f"{query_result.row_count}"
        )

        print(
            f"Truncated             : "
            f"{query_result.truncated}"
        )

    final_report = state.get(
        "final_report"
    )

    if final_report is not None:
        print("\nBusiness summary:")
        print(
            final_report.business_summary
        )

        if final_report.chart is not None:
            print("\nChart:")
            print(
                f"Type                  : "
                f"{final_report.chart.chart_type}"
            )

            print(
                f"X column              : "
                f"{final_report.chart.x_column}"
            )

            print(
                f"Y columns             : "
                f"{final_report.chart.y_columns}"
            )

    error_message = state.get(
        "error_message"
    )

    if error_message:
        print("\nWorkflow error:")
        print(error_message)


def validate_pydantic_outputs(
    state: GraphState,
) -> list[str]:
    """
    Confirm that every successful workflow output uses the expected
    shared Pydantic model.
    """

    errors: list[str] = []

    expected_outputs = [
        (
            "intent",
            IntentResult,
        ),
        (
            "plan",
            AnalysisPlan,
        ),
        (
            "sql_result",
            SQLGenerationResult,
        ),
        (
            "query_result",
            QueryExecutionResult,
        ),
        (
            "analysis_result",
            AnalysisResult,
        ),
        (
            "final_report",
            FinalReport,
        ),
    ]

    for field_name, expected_type in expected_outputs:
        value = state.get(field_name)

        if not isinstance(
            value,
            expected_type,
        ):
            errors.append(
                f"GraphState field '{field_name}' "
                f"should contain {expected_type.__name__}, "
                f"but received "
                f"{type(value).__name__}."
            )

    return errors


def validate_workflow_status(
    state: GraphState,
) -> list[str]:
    """
    Validate the final successful workflow status.
    """

    errors: list[str] = []

    workflow_status = state.get(
        "workflow_status"
    )

    if workflow_status != "completed":
        errors.append(
            "Successful workflow should finish with "
            "workflow_status='completed', but received "
            f"{workflow_status!r}."
        )

    current_node = state.get(
        "current_node"
    )

    if current_node != "reporting_agent":
        errors.append(
            "Successful workflow should finish at "
            "'reporting_agent', but received "
            f"{current_node!r}."
        )

    if state.get(
        "clarification_required",
        False,
    ):
        errors.append(
            "A complete business question should not "
            "require clarification."
        )

    if state.get("error_message"):
        errors.append(
            "Successful workflow contains an "
            "unexpected error_message."
        )

    if state.get("error_type"):
        errors.append(
            "Successful workflow contains an "
            "unexpected error_type."
        )

    return errors


def validate_analysis_type(
    state: GraphState,
    expected_analysis_types: list[str],
) -> list[str]:
    """
    Validate Agent 1's selected analytical category.
    """

    errors: list[str] = []

    intent = state.get("intent")

    if not isinstance(
        intent,
        IntentResult,
    ):
        return [
            "Cannot validate analysis type because "
            "IntentResult is missing."
        ]

    if (
        intent.analysis_type
        not in expected_analysis_types
    ):
        errors.append(
            "Expected analysis type to be one of "
            f"{expected_analysis_types}, but received "
            f"{intent.analysis_type!r}."
        )

    return errors


def validate_sql_safety(
    state: GraphState,
) -> list[str]:
    """
    Independently validate the final generated SQL.
    """

    errors: list[str] = []

    generated_sql = state.get(
        "generated_sql"
    )

    if not isinstance(
        generated_sql,
        str,
    ):
        return [
            "GraphState does not contain generated SQL."
        ]

    if not generated_sql.strip():
        return [
            "Generated SQL is empty."
        ]

    validator = SQLValidator(
        dialect="sqlite"
    )

    validation_result = validator.validate(
        generated_sql
    )

    if not validation_result.is_valid:
        errors.append(
            "Final generated SQL failed independent "
            "SQLValidator validation: "
            + "; ".join(
                validation_result.errors
            )
        )

    if len(
        validation_result.referenced_tables
    ) == 0:
        errors.append(
            "Generated SQL does not reference "
            "a database table or view."
        )

    return errors


def validate_query_result(
    state: GraphState,
    expected_maximum_rows: int | None,
) -> list[str]:
    """
    Validate the executed query result.
    """

    errors: list[str] = []

    query_result = state.get(
        "query_result"
    )

    if not isinstance(
        query_result,
        QueryExecutionResult,
    ):
        return [
            "QueryExecutionResult is missing."
        ]

    if not query_result.columns:
        errors.append(
            "Query result contains no columns."
        )

    if (
        query_result.row_count
        != len(query_result.rows)
    ):
        errors.append(
            "Query result row count does not match the "
            "number of returned rows."
        )
        

    if query_result.row_count < 1:
        errors.append(
            "The integration-test query returned no rows."
        )

    if (
        expected_maximum_rows is not None
        and query_result.row_count
        > expected_maximum_rows
    ):
        errors.append(
            "The query result exceeded the requested "
            f"maximum of {expected_maximum_rows} rows. "
            f"Received {query_result.row_count}."
        )

    if not query_result.query.strip():
        errors.append(
            "QueryExecutionResult does not contain "
            "the executed SQL."
        )

    return errors


def validate_analysis_result(
    state: GraphState,
) -> list[str]:
    """
    Validate Agent 4's evidence-grounded result.
    """

    errors: list[str] = []

    analysis_result = state.get(
        "analysis_result"
    )

    if not isinstance(
        analysis_result,
        AnalysisResult,
    ):
        return [
            "AnalysisResult is missing."
        ]

    if not analysis_result.summary.strip():
        errors.append(
            "AnalysisResult summary is empty."
        )

    for finding in analysis_result.findings:
        if not finding.title.strip():
            errors.append(
                "An analysis finding contains "
                "an empty title."
            )

        if not finding.description.strip():
            errors.append(
                "An analysis finding contains "
                "an empty description."
            )

    return errors


def validate_final_report(
    state: GraphState,
) -> list[str]:
    """
    Validate Agent 5's final business report.
    """

    errors: list[str] = []

    final_report = state.get(
        "final_report"
    )

    if not isinstance(
        final_report,
        FinalReport,
    ):
        return [
            "FinalReport is missing."
        ]

    if not final_report.business_summary.strip():
        errors.append(
            "FinalReport business_summary is empty."
        )

    for recommendation in (
        final_report.recommendations
    ):
        if not recommendation.action.strip():
            errors.append(
                "A recommendation contains "
                "an empty action."
            )

        if not recommendation.rationale.strip():
            errors.append(
                "A recommendation contains "
                "an empty rationale."
            )

    return errors


def validate_chart_columns(
    state: GraphState,
) -> list[str]:
    """
    Confirm that Agent 5's chart specification uses only columns
    returned by Agent 3b.
    """

    errors: list[str] = []

    final_report = state.get(
        "final_report"
    )

    query_result = state.get(
        "query_result"
    )

    if not isinstance(
        final_report,
        FinalReport,
    ):
        return [
            "Cannot validate chart because "
            "FinalReport is missing."
        ]

    if not isinstance(
        query_result,
        QueryExecutionResult,
    ):
        return [
            "Cannot validate chart because "
            "QueryExecutionResult is missing."
        ]

    chart = final_report.chart

    if chart is None:
        return errors

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
        errors.append(
            "Chart specification references columns "
            "not returned by the query: "
            f"{invalid_columns}. "
            f"Available columns: "
            f"{query_result.columns}."
        )

    return errors


def validate_retry_limit(
    state: GraphState,
) -> list[str]:
    """
    Confirm that SQL retries did not exceed the configured limit.
    """

    errors: list[str] = []

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
        errors.append(
            "sql_retry_count is not an integer."
        )

        return errors

    if not isinstance(
        max_retries,
        int,
    ):
        errors.append(
            "max_sql_retries is not an integer."
        )

        return errors

    if retry_count > max_retries:
        errors.append(
            "SQL retry count exceeded the configured "
            f"maximum: {retry_count}/{max_retries}."
        )

    return errors


def validate_execution_trace(
    state: GraphState,
) -> list[str]:
    """
    Confirm that all expected workflow nodes appear in the trace.
    """

    errors: list[str] = []

    execution_trace = state.get(
        "execution_trace",
        [],
    )

    if not isinstance(
        execution_trace,
        list,
    ):
        return [
            "execution_trace is not a list."
        ]

    traced_nodes = [
        event.get("node")
        for event in execution_trace
        if isinstance(event, dict)
    ]

    for required_node in (
        REQUIRED_SUCCESS_TRACE_NODES
    ):
        if required_node not in traced_nodes:
            errors.append(
                "Execution trace is missing node: "
                f"{required_node}."
            )

    return errors


def validate_safe_trace(
    state: GraphState,
) -> list[str]:
    """
    Confirm that execution traces do not contain obvious sensitive
    values or full prompt sections.
    """

    errors: list[str] = []

    execution_trace = state.get(
        "execution_trace",
        [],
    )

    trace_text = " ".join(
        str(event)
        for event in execution_trace
    ).lower()

    for sensitive_term in SENSITIVE_TRACE_TERMS:
        if sensitive_term.lower() in trace_text:
            errors.append(
                "Execution trace contains a disallowed "
                f"sensitive term: {sensitive_term!r}."
            )

    return errors


def validate_success_state(
    state: GraphState,
    test_case: dict[str, Any],
) -> list[str]:
    """
    Run all validations for one successful workflow.
    """

    errors: list[str] = []

    errors.extend(
        validate_workflow_status(state)
    )

    errors.extend(
        validate_pydantic_outputs(state)
    )

    errors.extend(
        validate_analysis_type(
            state=state,
            expected_analysis_types=(
                test_case[
                    "expected_analysis_types"
                ]
            ),
        )
    )

    errors.extend(
        validate_sql_safety(state)
    )

    errors.extend(
        validate_query_result(
            state=state,
            expected_maximum_rows=(
                test_case[
                    "expected_maximum_rows"
                ]
            ),
        )
    )

    errors.extend(
        validate_analysis_result(state)
    )

    errors.extend(
        validate_final_report(state)
    )

    errors.extend(
        validate_chart_columns(state)
    )

    errors.extend(
        validate_retry_limit(state)
    )

    errors.extend(
        validate_execution_trace(state)
    )

    errors.extend(
        validate_safe_trace(state)
    )

    return errors


def run_success_test(
    test_number: int,
    test_case: dict[str, Any],
) -> bool:
    """
    Run one complete successful workflow test.
    """

    print_test_header(
        test_number=test_number,
        test_name=test_case["name"],
    )

    question = test_case["question"]

    print(f"Question: {question}")
    print("\nRunning complete LangGraph workflow...")

    start_time = time.perf_counter()

    try:
        state = run_workflow(
            user_question=question,
            max_sql_retries=3,
            recursion_limit=35,
        )

    except Exception as exc:
        print("\n[FAILED]")
        print(
            f"Workflow raised "
            f"{type(exc).__name__}: {exc}"
        )

        return False

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    print_workflow_summary(
        state=state,
        elapsed_seconds=elapsed_seconds,
    )

    errors = validate_success_state(
        state=state,
        test_case=test_case,
    )

    if errors:
        print("\n[FAILED]")

        for error_number, error in enumerate(
            errors,
            start=1,
        ):
            print(
                f"{error_number}. {error}"
            )

        return False

    print("\n[PASSED]")
    return True


def run_clarification_test(
    test_number: int,
) -> bool:
    """
    Confirm that an incomplete question follows the clarification route
    without generating or executing SQL.
    """

    print_test_header(
        test_number=test_number,
        test_name=(
            CLARIFICATION_TEST_CASE["name"]
        ),
    )

    question = (
        CLARIFICATION_TEST_CASE[
            "question"
        ]
    )

    print(f"Question: {question}")
    print("\nRunning clarification workflow...")

    start_time = time.perf_counter()

    try:
        state = run_workflow(
            user_question=question,
            max_sql_retries=3,
            recursion_limit=15,
        )

    except Exception as exc:
        print("\n[FAILED]")
        print(
            f"Workflow raised "
            f"{type(exc).__name__}: {exc}"
        )

        return False

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    print_workflow_summary(
        state=state,
        elapsed_seconds=elapsed_seconds,
    )

    errors: list[str] = []

    if not state.get(
        "clarification_required",
        False,
    ):
        errors.append(
            "Incomplete question did not trigger "
            "clarification."
        )

    if (
        state.get("workflow_status")
        != "clarification_required"
    ):
        errors.append(
            "Clarification workflow should finish with "
            "workflow_status='clarification_required', "
            f"but received "
            f"{state.get('workflow_status')!r}."
        )

    clarification_question = (
        state.get(
            "clarification_question"
        )
    )

    if not isinstance(
        clarification_question,
        str,
    ):
        errors.append(
            "Clarification question is missing."
        )

    elif not clarification_question.strip():
        errors.append(
            "Clarification question is empty."
        )

    fields_that_must_be_empty = [
        "plan",
        "generated_sql",
        "query_result",
        "analysis_result",
        "final_report",
    ]

    for field_name in fields_that_must_be_empty:
        if state.get(field_name) is not None:
            errors.append(
                f"Clarification workflow unexpectedly "
                f"populated '{field_name}'."
            )

    trace_nodes = [
        event.get("node")
        for event in state.get(
            "execution_trace",
            [],
        )
        if isinstance(event, dict)
    ]

    if "intent_agent" not in trace_nodes:
        errors.append(
            "Clarification trace is missing "
            "the Intent Agent."
        )

    if "clarification" not in trace_nodes:
        errors.append(
            "Clarification trace is missing "
            "the clarification node."
        )

    forbidden_nodes = [
        "planning_agent",
        "sql_generation_agent",
        "sql_validation_execution",
        "analysis_agent",
        "reporting_agent",
    ]

    for forbidden_node in forbidden_nodes:
        if forbidden_node in trace_nodes:
            errors.append(
                "Clarification workflow should not "
                f"execute '{forbidden_node}'."
            )

    errors.extend(
        validate_safe_trace(state)
    )

    if errors:
        print("\n[FAILED]")

        for error_number, error in enumerate(
            errors,
            start=1,
        ):
            print(
                f"{error_number}. {error}"
            )

        return False

    print("\n[PASSED]")
    return True


def main() -> None:
    """
    Run all complete workflow tests.
    """

    print("\n" + "=" * 80)
    print(
        "COMPLETE MULTI-AGENT WORKFLOW "
        "TEST SUITE"
    )
    print("=" * 80)

    print(
        "\nModels:"
        "\n- llama3.2:latest"
        "\n- sqlcoder:latest"
    )

    print(
        "\nEach successful test calls multiple "
        "local agents and may take several minutes."
    )

    passed_tests = 0
    current_test_number = 1

    for test_case in SUCCESS_TEST_CASES:
        passed = run_success_test(
            test_number=current_test_number,
            test_case=test_case,
        )

        if passed:
            passed_tests += 1

        current_test_number += 1

    clarification_passed = (
        run_clarification_test(
            test_number=current_test_number
        )
    )

    if clarification_passed:
        passed_tests += 1

    total_tests = (
        len(SUCCESS_TEST_CASES) + 1
    )

    failed_tests = (
        total_tests - passed_tests
    )

    print("\n" + "=" * 80)
    print(
        "FINAL COMPLETE WORKFLOW TEST SUMMARY"
    )
    print("=" * 80)

    print(
        f"Total tests : {total_tests}"
    )

    print(
        f"Passed      : {passed_tests}"
    )

    print(
        f"Failed      : {failed_tests}"
    )

    if failed_tests == 0:
        print(
            "\nRESULT: ALL COMPLETE WORKFLOW "
            "TESTS PASSED"
        )

        print(
            "The five-agent LangGraph backend is "
            "ready for chart rendering, security, "
            "and Streamlit integration."
        )

        raise SystemExit(0)

    print(
        "\nRESULT: SOME COMPLETE WORKFLOW "
        "TESTS FAILED"
    )

    print(
        "Review each failed workflow before "
        "building the Streamlit interface."
    )

    raise SystemExit(1)


if __name__ == "__main__":
    main()