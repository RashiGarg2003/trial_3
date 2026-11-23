"""
Standalone tests for the shared Pydantic schemas.

Run from the project root:

    python -m tests.test_schemas
"""

import sys
from pathlib import Path
from typing import Callable

from pydantic import ValidationError


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from models.schemas import (
    AnalysisFinding,
    AnalysisPlan,
    AnalysisResult,
    ChartSpecification,
    FilterCondition,
    FinalReport,
    IntentResult,
    PlanStep,
    QueryExecutionResult,
    Recommendation,
    SQLGenerationRequest,
    SQLGenerationResult,
    TimePeriod,
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


def build_valid_intent() -> IntentResult:
    """
    Build a valid reusable IntentResult.
    """

    return IntentResult(
        original_question=(
            "Show the top 5 products by revenue in 2025."
        ),
        rewritten_question=(
            "Rank the five highest-revenue products for 2025."
        ),
        analysis_type="ranking",
        metrics=["revenue"],
        dimensions=["product"],
        filters=[],
        time_period=TimePeriod(
            start_date="2025-01-01",
            end_date="2025-12-31",
            granularity="year",
        ),
        sort_order="descending",
        limit=5,
        clarification_required=False,
        clarification_question=None,
    )


def build_valid_plan() -> AnalysisPlan:
    """
    Build a valid reusable AnalysisPlan.
    """

    return AnalysisPlan(
        objective=(
            "Identify the five products with the highest "
            "revenue during 2025."
        ),
        strategy=(
            "Retrieve product-level revenue for the requested "
            "period, aggregate it, rank it, and return five results."
        ),
        steps=[
            PlanStep(
                step_id=1,
                title="Retrieve relevant sales data",
                description=(
                    "Retrieve the data required to calculate "
                    "product-level revenue for the requested period."
                ),
                step_type="data_retrieval",
                depends_on=[],
                required_metrics=["revenue"],
                required_dimensions=["product"],
                required_data=[
                    "product information",
                    "order transaction information",
                    "quantity",
                    "unit price",
                    "discount",
                    "order date",
                ],
                expected_output=(
                    "Transaction-level data required for "
                    "revenue calculation."
                ),
                requires_sql=True,
                status="pending",
            ),
            PlanStep(
                step_id=2,
                title="Aggregate revenue by product",
                description=(
                    "Calculate total revenue for every product."
                ),
                step_type="aggregation",
                depends_on=[1],
                required_metrics=["revenue"],
                required_dimensions=["product"],
                required_data=[],
                expected_output=(
                    "One aggregated revenue value for each product."
                ),
                requires_sql=True,
                status="pending",
            ),
            PlanStep(
                step_id=3,
                title="Rank products",
                description=(
                    "Sort products by revenue and retain "
                    "the requested number of results."
                ),
                step_type="ranking",
                depends_on=[2],
                required_metrics=["revenue"],
                required_dimensions=["product"],
                required_data=[],
                expected_output=(
                    "Five products ranked by total revenue."
                ),
                requires_sql=True,
                status="pending",
            ),
        ],
        final_output_requirements=[
            "Product name",
            "Total revenue",
            "Descending revenue ranking",
        ],
    )


def test_valid_time_period() -> bool:
    """
    Confirm that a valid time-period model can be created.
    """

    time_period = TimePeriod(
        start_date="2025-01-01",
        end_date="2025-12-31",
        current_period="2025",
        comparison_period=None,
        granularity="month",
    )

    if time_period.start_date != "2025-01-01":
        raise AssertionError(
            "The start date was not stored correctly."
        )

    if time_period.granularity != "month":
        raise AssertionError(
            "The time granularity was not stored correctly."
        )

    print(time_period.model_dump())
    return True


def test_valid_filter_condition() -> bool:
    """
    Confirm that a generic filter can be created.
    """

    filter_condition = FilterCondition(
        field="region",
        operator="equals",
        value="North",
    )

    if filter_condition.field != "region":
        raise AssertionError(
            "The filter field was not stored correctly."
        )

    if filter_condition.value != "North":
        raise AssertionError(
            "The filter value was not stored correctly."
        )

    print(filter_condition.model_dump())
    return True


def test_valid_intent_result() -> bool:
    """
    Confirm that a complete IntentResult can be created.
    """

    intent = build_valid_intent()

    if intent.analysis_type != "ranking":
        raise AssertionError(
            "The analysis type was not stored correctly."
        )

    if intent.limit != 5:
        raise AssertionError(
            "The result limit was not stored correctly."
        )

    print(intent.model_dump_json(indent=2))
    return True


def test_intent_json_schema() -> bool:
    """
    Confirm that Pydantic can generate a JSON schema.

    This schema will later be passed to Ollama for structured output.
    """

    json_schema = IntentResult.model_json_schema()

    if json_schema.get("type") != "object":
        raise AssertionError(
            "IntentResult did not produce an object JSON schema."
        )

    properties = json_schema.get("properties", {})

    required_properties = {
        "original_question",
        "rewritten_question",
        "analysis_type",
        "metrics",
        "dimensions",
        "filters",
        "time_period",
        "clarification_required",
    }

    missing_properties = required_properties.difference(
        properties.keys()
    )

    if missing_properties:
        raise AssertionError(
            "IntentResult JSON schema is missing properties: "
            f"{sorted(missing_properties)}"
        )

    print(
        "IntentResult JSON schema generated successfully."
    )
    print(
        f"Property count: {len(properties)}"
    )

    return True


def test_clarification_consistency() -> bool:
    """
    Confirm that clarification fields are validated.
    """

    try:
        IntentResult(
            original_question="Analyze the data.",
            rewritten_question="Analyze the data.",
            analysis_type="general_analysis",
            metrics=[],
            dimensions=[],
            filters=[],
            time_period=TimePeriod(),
            clarification_required=True,
            clarification_question=None,
        )

    except ValidationError as exc:
        print("Expected validation error:")
        print(exc)
        return True

    raise AssertionError(
        "IntentResult accepted clarification_required=True "
        "without a clarification question."
    )


def test_unexpected_intent_field_rejection() -> bool:
    """
    Confirm that unknown fields are rejected.
    """

    valid_data = build_valid_intent().model_dump()

    valid_data["invented_field"] = "unexpected"

    try:
        IntentResult.model_validate(valid_data)

    except ValidationError as exc:
        print("Expected extra-field validation error:")
        print(exc)
        return True

    raise AssertionError(
        "IntentResult accepted an unexpected field."
    )


def test_valid_analysis_plan() -> bool:
    """
    Confirm that a valid multi-step plan can be created.
    """

    plan = build_valid_plan()

    if len(plan.steps) != 3:
        raise AssertionError(
            "The expected three plan steps were not created."
        )

    if plan.steps[1].depends_on != [1]:
        raise AssertionError(
            "Plan dependencies were not stored correctly."
        )

    print(plan.model_dump_json(indent=2))
    return True


def test_duplicate_step_id_rejection() -> bool:
    """
    Confirm that duplicate plan step IDs are rejected.
    """

    try:
        AnalysisPlan(
            objective="Test duplicate step IDs.",
            strategy="Create two steps with the same ID.",
            steps=[
                PlanStep(
                    step_id=1,
                    title="First step",
                    description="First test step.",
                    step_type="data_retrieval",
                    expected_output="First output.",
                    requires_sql=True,
                ),
                PlanStep(
                    step_id=1,
                    title="Second step",
                    description="Second test step.",
                    step_type="aggregation",
                    expected_output="Second output.",
                    requires_sql=True,
                ),
            ],
        )

    except ValidationError as exc:
        print("Expected duplicate-step validation error:")
        print(exc)
        return True

    raise AssertionError(
        "AnalysisPlan accepted duplicate step IDs."
    )


def test_future_dependency_rejection() -> bool:
    """
    Confirm that a plan step cannot depend on a future step.
    """

    try:
        AnalysisPlan(
            objective="Test future dependencies.",
            strategy="Create an invalid plan dependency.",
            steps=[
                PlanStep(
                    step_id=1,
                    title="First step",
                    description="Invalid first step.",
                    step_type="data_retrieval",
                    depends_on=[2],
                    expected_output="First output.",
                    requires_sql=True,
                ),
                PlanStep(
                    step_id=2,
                    title="Second step",
                    description="Second step.",
                    step_type="aggregation",
                    depends_on=[],
                    expected_output="Second output.",
                    requires_sql=True,
                ),
            ],
        )

    except ValidationError as exc:
        print("Expected dependency validation error:")
        print(exc)
        return True

    raise AssertionError(
        "AnalysisPlan accepted a future dependency."
    )


def test_unknown_dependency_rejection() -> bool:
    """
    Confirm that unknown dependency IDs are rejected.
    """

    try:
        AnalysisPlan(
            objective="Test unknown dependencies.",
            strategy="Reference a step that does not exist.",
            steps=[
                PlanStep(
                    step_id=1,
                    title="Only step",
                    description="Step with an unknown dependency.",
                    step_type="data_retrieval",
                    depends_on=[99],
                    expected_output="Test output.",
                    requires_sql=True,
                ),
            ],
        )

    except ValidationError as exc:
        print("Expected unknown-dependency error:")
        print(exc)
        return True

    raise AssertionError(
        "accepted an unknown dependency."
    )


def test_valid_sql_generation_request() -> bool:
    """
    Confirm that a SQLGenerationRequest can combine intent,
    planning, and actual database schema context.
    """

    request = SQLGenerationRequest(
        user_question=(
            "Show the top 5 products by revenue in 2025."
        ),
        intent=build_valid_intent(),
        plan=build_valid_plan(),
        database_type="SQLite",
        database_schema=(
            "DATABASE: northwind.db\n"
            "DATABASE TYPE: SQLite\n"
            "TABLE: Example\n"
            "COLUMNS:\n"
            "- id INTEGER"
        ),
        previous_sql=None,
        previous_error=None,
    )

    if request.database_type != "SQLite":
        raise AssertionError(
            "The database type was not stored correctly."
        )

    if request.intent.analysis_type != "ranking":
        raise AssertionError(
            "The nested IntentResult was not stored correctly."
        )

    print(
        "SQLGenerationRequest created successfully."
    )
    print(
        f"Plan steps: {len(request.plan.steps)}"
    )

    return True


def test_valid_sql_generation_result() -> bool:
    """
    Confirm that a structured SQL result can be created.
    """

    result = SQLGenerationResult(
        sql=(
            "SELECT product_name, SUM(revenue) AS total_revenue "
            "FROM sales "
            "GROUP BY product_name "
            "ORDER BY total_revenue DESC "
            "LIMIT 5;"
        ),
        explanation=(
            "Aggregate revenue by product and return "
            "the five highest values."
        ),
        referenced_tables=["sales"],
        expected_columns=[
            "product_name",
            "total_revenue",
        ],
        assumptions=[],
        requires_clarification=False,
        clarification_question=None,
    )

    if not result.sql.strip().upper().startswith("SELECT"):
        raise AssertionError(
            "The generated SQL was not stored correctly."
        )

    print(result.model_dump_json(indent=2))
    return True


def test_sql_clarification_consistency() -> bool:
    """
    Confirm that SQL clarification fields are validated.
    """

    try:
        SQLGenerationResult(
            sql="SELECT 1;",
            explanation=(
                "A reliable query cannot yet be generated."
            ),
            referenced_tables=[],
            expected_columns=[],
            assumptions=[],
            requires_clarification=True,
            clarification_question=None,
        )

    except ValidationError as exc:
        print("Expected SQL clarification error:")
        print(exc)
        return True

    raise AssertionError(
        "SQLGenerationResult accepted an inconsistent "
        "clarification state."
    )


def test_valid_query_execution_result() -> bool:
    """
    Confirm that database results can be represented serializably.
    """

    result = QueryExecutionResult(
        columns=["product", "revenue"],
        rows=[
            {
                "product": "Example Product",
                "revenue": 1250.50,
            }
        ],
        row_count=1,
        execution_time_ms=2.75,
        truncated=False,
        query=(
            "SELECT product, revenue FROM example;"
        ),
    )

    if result.row_count != len(result.rows):
        raise AssertionError(
            "The row count does not match the included rows."
        )

    print(result.model_dump_json(indent=2))
    return True


def test_negative_row_count_rejection() -> bool:
    """
    Confirm that negative row counts are rejected.
    """

    try:
        QueryExecutionResult(
            columns=[],
            rows=[],
            row_count=-1,
            execution_time_ms=1.0,
            truncated=False,
            query="SELECT 1;",
        )

    except ValidationError as exc:
        print("Expected row-count validation error:")
        print(exc)
        return True

    raise AssertionError(
        "QueryExecutionResult accepted a negative row count."
    )


def test_valid_analysis_result() -> bool:
    """
    Confirm that structured analytical findings can be created.
    """

    result = AnalysisResult(
        summary=(
            "Revenue is concentrated among a small number "
            "of products."
        ),
        findings=[
            AnalysisFinding(
                title="Highest-revenue product",
                description=(
                    "The leading product generated the highest "
                    "revenue in the result set."
                ),
                evidence=[
                    "The product appears first in descending order."
                ],
                importance="high",
            )
        ],
        trends=[],
        anomalies=[],
        possible_drivers=[
            "Higher order volume",
            "Higher unit price",
        ],
        limitations=[
            "The result reflects only the requested period."
        ],
    )

    if not result.findings:
        raise AssertionError(
            "The analysis result contains no findings."
        )

    print(result.model_dump_json(indent=2))
    return True


def test_valid_chart_specification() -> bool:
    """
    Confirm that a generic chart specification can be created.
    """

    chart = ChartSpecification(
        chart_type="bar",
        title="Revenue by Product",
        x_column="product",
        y_columns=["revenue"],
        category_column=None,
        rationale=(
            "A bar chart clearly compares revenue "
            "across products."
        ),
    )

    if chart.chart_type != "bar":
        raise AssertionError(
            "The chart type was not stored correctly."
        )

    print(chart.model_dump_json(indent=2))
    return True


def test_valid_final_report() -> bool:
    """
    Confirm that a complete final report can be created.
    """

    report = FinalReport(
        business_summary=(
            "The requested products were ranked by total revenue."
        ),
        key_findings=[
            "The first product has the highest revenue.",
            "Revenue differs across the returned products.",
        ],
        root_cause_indicators=[],
        recommendations=[
            Recommendation(
                action=(
                    "Review the leading products for repeatable "
                    "sales patterns."
                ),
                rationale=(
                    "The revenue ranking identifies where the "
                    "largest contribution occurred."
                ),
                priority="medium",
            )
        ],
        chart=ChartSpecification(
            chart_type="bar",
            title="Top Products by Revenue",
            x_column="product",
            y_columns=["total_revenue"],
            category_column=None,
            rationale=(
                "A bar chart supports direct comparison "
                "between ranked products."
            ),
        ),
        limitations=[
            "Recommendations are limited to available query results."
        ],
    )

    if not report.business_summary:
        raise AssertionError(
            "The final report has no business summary."
        )

    if report.chart is None:
        raise AssertionError(
            "The final report chart was not stored."
        )

    print(report.model_dump_json(indent=2))
    return True


def run_test(
    test_number: int,
    test_name: str,
    test_function: Callable[[], bool],
) -> bool:
    """
    Execute one test and print pass or fail.
    """

    print_test_header(
        test_number=test_number,
        test_name=test_name,
    )

    try:
        test_function()

    except Exception as exc:
        print("\n[FAILED]")
        print(f"{type(exc).__name__}: {exc}")
        return False

    print("\n[PASSED]")
    return True


def main() -> None:
    """
    Run all shared-schema tests.
    """

    print("\nSHARED PYDANTIC SCHEMA TEST SUITE")

    tests: list[
        tuple[
            str,
            Callable[[], bool],
        ]
    ] = [
        (
            "Valid time period",
            test_valid_time_period,
        ),
        (
            "Valid filter condition",
            test_valid_filter_condition,
        ),
        (
            "Valid intent result",
            test_valid_intent_result,
        ),
        (
            "Intent JSON schema generation",
            test_intent_json_schema,
        ),
        (
            "Intent clarification consistency",
            test_clarification_consistency,
        ),
        (
            "Unexpected intent field rejection",
            test_unexpected_intent_field_rejection,
        ),
        (
            "Valid analysis plan",
            test_valid_analysis_plan,
        ),
        (
            "Duplicate plan step ID rejection",
            test_duplicate_step_id_rejection,
        ),
        (
            "Future plan dependency rejection",
            test_future_dependency_rejection,
        ),
        (
            "Unknown plan dependency rejection",
            test_unknown_dependency_rejection,
        ),
        (
            "Valid SQL generation request",
            test_valid_sql_generation_request,
        ),
        (
            "Valid SQL generation result",
            test_valid_sql_generation_result,
        ),
        (
            "SQL clarification consistency",
            test_sql_clarification_consistency,
        ),
        (
            "Valid query execution result",
            test_valid_query_execution_result,
        ),
        (
            "Negative row count rejection",
            test_negative_row_count_rejection,
        ),
        (
            "Valid analysis result",
            test_valid_analysis_result,
        ),
        (
            "Valid chart specification",
            test_valid_chart_specification,
        ),
        (
            "Valid final report",
            test_valid_final_report,
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
        )

        if passed:
            passed_tests += 1

    total_tests = len(tests)
    failed_tests = total_tests - passed_tests

    print("\n" + "=" * 80)
    print("FINAL SHARED SCHEMA TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests : {total_tests}")
    print(f"Passed      : {passed_tests}")
    print(f"Failed      : {failed_tests}")

    if failed_tests == 0:
        print("\nRESULT: ALL SHARED SCHEMA TESTS PASSED")
        print(
            "The structured contracts are ready "
            "for agent integration."
        )
        raise SystemExit(0)

    print("\nRESULT: SOME SHARED SCHEMA TESTS FAILED")
    print(
        "Resolve the failed tests before updating "
        "the Intent Agent."
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()