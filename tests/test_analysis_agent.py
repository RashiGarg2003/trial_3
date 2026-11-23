"""
Tests for Agent 4: Analysis Agent.

The tests verify:

1. Query results are converted into a DataFrame.
2. Numeric columns are detected dynamically.
3. Deterministic statistics are calculated correctly.
4. Empty query results are handled without calling Ollama.
5. Real query results from Agent 3 are interpreted by llama3.2.
6. Agent 4 returns a validated AnalysisResult.

Run from the project root:

    python -m tests.test_analysis_agent
"""

import sys
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agents.analysis_agent import AnalysisAgent
from models.schemas import (
    AnalysisPlan,
    AnalysisResult,
    IntentResult,
    PlanStep,
    QueryExecutionResult,
    TimePeriod,
)


def build_test_intent() -> IntentResult:
    """
    Build validated analytical intent for testing.
    """

    return IntentResult(
        original_question=(
            "Show the top 5 products by revenue."
        ),
        rewritten_question=(
            "Rank the five products with the highest revenue."
        ),
        analysis_type="ranking",
        metrics=["revenue"],
        dimensions=["product"],
        filters=[],
        time_period=TimePeriod(),
        sort_order="descending",
        limit=5,
        clarification_required=False,
        clarification_question=None,
    )


def build_test_plan() -> AnalysisPlan:
    """
    Build a valid analytical plan.
    """

    return AnalysisPlan(
        objective=(
            "Identify the five products with the highest revenue."
        ),
        strategy=(
            "Retrieve product revenue, rank the products, "
            "and analyze the returned values."
        ),
        steps=[
            PlanStep(
                step_id=1,
                title="Retrieve product revenue",
                description=(
                    "Retrieve product-level revenue values."
                ),
                step_type="data_retrieval",
                depends_on=[],
                required_metrics=["revenue"],
                required_dimensions=["product"],
                required_data=[
                    "product",
                    "revenue",
                ],
                expected_output=(
                    "Product-level revenue records."
                ),
                requires_sql=True,
                status="pending",
            ),
            PlanStep(
                step_id=2,
                title="Rank products",
                description=(
                    "Rank products according to revenue."
                ),
                step_type="ranking",
                depends_on=[1],
                required_metrics=["revenue"],
                required_dimensions=["product"],
                required_data=[],
                expected_output=(
                    "Products ranked by revenue."
                ),
                requires_sql=True,
                status="pending",
            ),
        ],
        final_output_requirements=[
            "Product",
            "Revenue",
            "Ranking",
        ],
    )


def build_test_query_result() -> QueryExecutionResult:
    """
    Build deterministic query-result data for testing.

    This is a test fixture, not an application response.
    """

    return QueryExecutionResult(
        columns=[
            "product",
            "revenue",
        ],
        rows=[
            {
                "product": "Product A",
                "revenue": 50000.0,
            },
            {
                "product": "Product B",
                "revenue": 42000.0,
            },
            {
                "product": "Product C",
                "revenue": 35000.0,
            },
            {
                "product": "Product D",
                "revenue": 28000.0,
            },
            {
                "product": "Product E",
                "revenue": 20000.0,
            },
        ],
        row_count=5,
        execution_time_ms=2.5,
        truncated=False,
        query=(
            "SELECT product, revenue "
            "FROM product_revenue "
            "ORDER BY revenue DESC "
            "LIMIT 5;"
        ),
    )


def build_empty_query_result() -> QueryExecutionResult:
    """
    Build a valid empty query result.
    """

    return QueryExecutionResult(
        columns=[
            "product",
            "revenue",
        ],
        rows=[],
        row_count=0,
        execution_time_ms=1.0,
        truncated=False,
        query=(
            "SELECT product, revenue "
            "FROM product_revenue "
            "WHERE 1 = 0;"
        ),
    )


def print_test_header(
    test_number: int,
    test_name: str,
) -> None:
    """
    Print a formatted test heading.
    """

    print("\n" + "=" * 80)
    print(f"TEST {test_number}: {test_name}")
    print("=" * 80)


def test_dataframe_creation(
    agent: AnalysisAgent,
) -> bool:
    """
    Confirm that query rows become a DataFrame.
    """

    query_result = build_test_query_result()

    dataframe = agent._records_to_dataframe(
        query_result
    )

    if len(dataframe) != 5:
        raise AssertionError(
            f"Expected 5 DataFrame rows, received {len(dataframe)}."
        )

    if list(dataframe.columns) != [
        "product",
        "revenue",
    ]:
        raise AssertionError(
            "Unexpected DataFrame columns: "
            f"{list(dataframe.columns)}"
        )

    print(dataframe)
    return True


def test_numeric_detection(
    agent: AnalysisAgent,
) -> bool:
    """
    Confirm that numeric columns are detected dynamically.
    """

    dataframe = agent._records_to_dataframe(
        build_test_query_result()
    )

    numeric_columns = (
        agent._detect_numeric_columns(
            dataframe
        )
    )

    if "revenue" not in numeric_columns:
        raise AssertionError(
            "The revenue column was not detected as numeric."
        )

    if "product" in numeric_columns:
        raise AssertionError(
            "The product column was incorrectly detected as numeric."
        )

    print(f"Numeric columns: {numeric_columns}")
    return True


def test_deterministic_profile(
    agent: AnalysisAgent,
) -> bool:
    """
    Confirm that Pandas calculates profile values correctly.
    """

    profile = agent._build_data_profile(
        build_test_query_result()
    )

    revenue_profile = profile[
        "numeric_profile"
    ]["revenue"]

    expected_sum = 175000.0
    expected_mean = 35000.0
    expected_minimum = 20000.0
    expected_maximum = 50000.0

    if revenue_profile["sum"] != expected_sum:
        raise AssertionError(
            "Unexpected revenue sum. "
            f"Expected {expected_sum}, "
            f"received {revenue_profile['sum']}."
        )

    if revenue_profile["mean"] != expected_mean:
        raise AssertionError(
            "Unexpected revenue mean. "
            f"Expected {expected_mean}, "
            f"received {revenue_profile['mean']}."
        )

    if revenue_profile["minimum"] != expected_minimum:
        raise AssertionError(
            "Unexpected minimum revenue."
        )

    if revenue_profile["maximum"] != expected_maximum:
        raise AssertionError(
            "Unexpected maximum revenue."
        )

    if profile["row_count"] != 5:
        raise AssertionError(
            "Profile row count is incorrect."
        )

    print("Revenue profile:")
    print(revenue_profile)

    return True


def test_empty_result_handling(
    agent: AnalysisAgent,
) -> bool:
    """
    Confirm that an empty result is handled without model inference.
    """

    result = agent.analyze(
        user_question=(
            "Show the top 5 products by revenue."
        ),
        intent=build_test_intent(),
        plan=build_test_plan(),
        query_result=(
            build_empty_query_result()
        ),
    )

    if not isinstance(
        result,
        AnalysisResult,
    ):
        raise AssertionError(
            "Empty-result handling did not return AnalysisResult."
        )

    if result.findings:
        raise AssertionError(
            "An empty result should not contain findings."
        )

    if result.trends:
        raise AssertionError(
            "An empty result should not contain trends."
        )

    if result.possible_drivers:
        raise AssertionError(
            "An empty result should not contain possible drivers."
        )

    if not result.limitations:
        raise AssertionError(
            "An empty result should contain a limitation."
        )

    print(result.model_dump_json(indent=2))
    return True


def test_model_analysis(
    agent: AnalysisAgent,
) -> bool:
    """
    Confirm that llama3.2 returns a validated AnalysisResult.
    """

    result = agent.analyze(
        user_question=(
            "Show the top 5 products by revenue."
        ),
        intent=build_test_intent(),
        plan=build_test_plan(),
        query_result=(
            build_test_query_result()
        ),
    )

    if not isinstance(
        result,
        AnalysisResult,
    ):
        raise AssertionError(
            "AnalysisAgent did not return AnalysisResult."
        )

    if not result.summary.strip():
        raise AssertionError(
            "Analysis summary is empty."
        )

    if not result.findings:
        raise AssertionError(
            "Analysis contains no findings."
        )

    for finding in result.findings:
        if not finding.title.strip():
            raise AssertionError(
                "A finding has no title."
            )

        if not finding.description.strip():
            raise AssertionError(
                "A finding has no description."
            )

    print(result.model_dump_json(indent=2))
    return True


def test_invalid_input_rejection(
    agent: AnalysisAgent,
) -> bool:
    """
    Confirm that invalid query-result input is rejected.
    """

    try:
        agent.analyze(
            user_question="Show revenue.",
            intent=build_test_intent(),
            plan=build_test_plan(),
            query_result={
                "rows": []
            },
        )  # type: ignore[arg-type]

    except TypeError as exc:
        print(f"Expected type error: {exc}")
        return True

    raise AssertionError(
        "AnalysisAgent accepted an invalid query result."
    )


def run_test(
    test_number: int,
    test_name: str,
    test_function: Callable[
        [AnalysisAgent],
        bool,
    ],
    agent: AnalysisAgent,
) -> bool:
    """
    Execute one Agent 4 test.
    """

    print_test_header(
        test_number=test_number,
        test_name=test_name,
    )

    try:
        test_function(agent)

    except Exception as exc:
        print("\n[FAILED]")
        print(
            f"{type(exc).__name__}: {exc}"
        )
        return False

    print("\n[PASSED]")
    return True


def main() -> None:
    """
    Run all Agent 4 tests.
    """

    print("\nAGENT 4 TEST SUITE")
    print("Component: Analysis Agent")
    print("Model: llama3.2:latest")

    agent = AnalysisAgent(
        model="llama3.2:latest",
        host="http://localhost:11434",
        context_length=8192,
        max_output_tokens=1500,
        maximum_sample_rows=50,
    )

    tests: list[
        tuple[
            str,
            Callable[
                [AnalysisAgent],
                bool,
            ],
        ]
    ] = [
        (
            "DataFrame creation",
            test_dataframe_creation,
        ),
        (
            "Numeric-column detection",
            test_numeric_detection,
        ),
        (
            "Deterministic data profile",
            test_deterministic_profile,
        ),
        (
            "Empty-result handling",
            test_empty_result_handling,
        ),
        (
            "Model-based analysis",
            test_model_analysis,
        ),
        (
            "Invalid-input rejection",
            test_invalid_input_rejection,
        ),
    ]

    passed_tests = 0

    for test_number, (
        test_name,
        test_function,
    ) in enumerate(
        tests,
        start=1,
    ):
        passed = run_test(
            test_number=test_number,
            test_name=test_name,
            test_function=test_function,
            agent=agent,
        )

        if passed:
            passed_tests += 1

    total_tests = len(tests)
    failed_tests = (
        total_tests - passed_tests
    )

    print("\n" + "=" * 80)
    print("FINAL AGENT 4 TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests : {total_tests}")
    print(f"Passed      : {passed_tests}")
    print(f"Failed      : {failed_tests}")

    if failed_tests == 0:
        print(
            "\nRESULT: ALL AGENT 4 TESTS PASSED"
        )
        print(
            "The Analysis Agent is ready for "
            "LangGraph integration."
        )
        raise SystemExit(0)

    print(
        "\nRESULT: SOME AGENT 4 TESTS FAILED"
    )
    print(
        "Resolve the failed cases before adding "
        "Agent 4 to LangGraph."
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()