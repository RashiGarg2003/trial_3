"""
Standalone tests for ChartTool.

The tests verify:

1. Vertical bar-chart rendering.
2. Horizontal bar-chart rendering.
3. Line-chart rendering.
4. Area-chart rendering.
5. Scatter-chart rendering.
6. Pie-chart rendering.
7. Table rendering.
8. chart_type='none'.
9. Missing-column fallback.
10. Nonnumeric Y-column fallback.
11. Empty-result handling.
12. Maximum chart-row enforcement.
13. Invalid input rejection.

Run from the project root:

    python -m tests.test_chart_tool

These tests do not call Ollama, LangGraph, or the database.
"""

import sys
from pathlib import Path
from typing import Callable

import plotly.graph_objects as go


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from models.schemas import (
    ChartSpecification,
    QueryExecutionResult,
)
from tools.chart_tool import (
    ChartRenderResult,
    ChartTool,
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


def create_category_result() -> QueryExecutionResult:
    """
    Create category-and-value test data.
    """

    return QueryExecutionResult(
        columns=[
            "Category",
            "Revenue",
        ],
        rows=[
            {
                "Category": "Beverages",
                "Revenue": 1200.50,
            },
            {
                "Category": "Seafood",
                "Revenue": 980.25,
            },
            {
                "Category": "Confections",
                "Revenue": 875.00,
            },
            {
                "Category": "Produce",
                "Revenue": 640.75,
            },
        ],
        row_count=4,
        execution_time_ms=1.25,
        truncated=False,
        query=(
            "SELECT Category, Revenue "
            "FROM category_revenue;"
        ),
    )


def create_time_series_result() -> QueryExecutionResult:
    """
    Create monthly time-series test data.
    """

    return QueryExecutionResult(
        columns=[
            "Month",
            "Revenue",
        ],
        rows=[
            {
                "Month": "2025-01",
                "Revenue": 1000.0,
            },
            {
                "Month": "2025-02",
                "Revenue": 1250.0,
            },
            {
                "Month": "2025-03",
                "Revenue": 1180.0,
            },
            {
                "Month": "2025-04",
                "Revenue": 1400.0,
            },
        ],
        row_count=4,
        execution_time_ms=1.10,
        truncated=False,
        query=(
            "SELECT Month, Revenue "
            "FROM monthly_revenue;"
        ),
    )


def create_scatter_result() -> QueryExecutionResult:
    """
    Create two-measure test data for a scatter chart.
    """

    return QueryExecutionResult(
        columns=[
            "OrderCount",
            "Revenue",
            "Region",
        ],
        rows=[
            {
                "OrderCount": 10,
                "Revenue": 1200.0,
                "Region": "North",
            },
            {
                "OrderCount": 18,
                "Revenue": 2100.0,
                "Region": "South",
            },
            {
                "OrderCount": 13,
                "Revenue": 1650.0,
                "Region": "East",
            },
        ],
        row_count=3,
        execution_time_ms=1.30,
        truncated=False,
        query=(
            "SELECT OrderCount, Revenue, Region "
            "FROM regional_performance;"
        ),
    )


def create_empty_result() -> QueryExecutionResult:
    """
    Create a valid query result containing no rows.
    """

    return QueryExecutionResult(
        columns=[
            "Category",
            "Revenue",
        ],
        rows=[],
        row_count=0,
        execution_time_ms=0.75,
        truncated=False,
        query=(
            "SELECT Category, Revenue "
            "FROM category_revenue "
            "WHERE 1 = 0;"
        ),
    )


def create_large_result(
    row_count: int = 20,
) -> QueryExecutionResult:
    """
    Create a result larger than the configured chart-row limit.
    """

    rows = [
        {
            "Category": f"Category {index}",
            "Revenue": float(
                index * 100
            ),
        }
        for index in range(
            1,
            row_count + 1,
        )
    ]

    return QueryExecutionResult(
        columns=[
            "Category",
            "Revenue",
        ],
        rows=rows,
        row_count=len(rows),
        execution_time_ms=2.5,
        truncated=False,
        query=(
            "SELECT Category, Revenue "
            "FROM category_revenue;"
        ),
    )


def create_text_value_result() -> QueryExecutionResult:
    """
    Create data containing a nonnumeric measure column.
    """

    return QueryExecutionResult(
        columns=[
            "Category",
            "Performance",
        ],
        rows=[
            {
                "Category": "A",
                "Performance": "High",
            },
            {
                "Category": "B",
                "Performance": "Low",
            },
        ],
        row_count=2,
        execution_time_ms=1.0,
        truncated=False,
        query=(
            "SELECT Category, Performance "
            "FROM performance_labels;"
        ),
    )


def require_figure(
    result: ChartRenderResult,
) -> None:
    """
    Confirm that a valid Plotly figure was returned.
    """

    if result.figure is None:
        raise AssertionError(
            "Expected a Plotly figure, "
            "but figure is None."
        )

    if not isinstance(
        result.figure,
        go.Figure,
    ):
        raise AssertionError(
            "The rendered object is not "
            "a Plotly go.Figure."
        )


def require_requested_chart(
    result: ChartRenderResult,
    expected_chart_type: str,
) -> None:
    """
    Confirm that the requested chart type was rendered.
    """

    require_figure(result)

    if (
        result.rendered_chart_type
        != expected_chart_type
    ):
        raise AssertionError(
            f"Expected rendered chart type "
            f"'{expected_chart_type}', but received "
            f"'{result.rendered_chart_type}'."
        )

    if result.used_fallback:
        raise AssertionError(
            "A valid chart unexpectedly used "
            "the table fallback."
        )


def require_table_fallback(
    result: ChartRenderResult,
) -> None:
    """
    Confirm that an invalid chart safely became a table.
    """

    require_figure(result)

    if result.rendered_chart_type != "table":
        raise AssertionError(
            "Expected table fallback, but received "
            f"'{result.rendered_chart_type}'."
        )

    if not result.used_fallback:
        raise AssertionError(
            "Table fallback should set "
            "used_fallback=True."
        )

    if not result.warnings:
        raise AssertionError(
            "Table fallback should contain "
            "at least one warning."
        )


def print_result(
    result: ChartRenderResult,
) -> None:
    """
    Print concise rendering metadata.
    """

    print(
        f"Rendered chart type : "
        f"{result.rendered_chart_type}"
    )

    print(
        f"Used fallback       : "
        f"{result.used_fallback}"
    )

    print(
        f"Figure created      : "
        f"{result.figure is not None}"
    )

    print(
        f"Warnings            : "
        f"{result.warnings}"
    )

    if result.figure is not None:
        print(
            f"Plotly traces       : "
            f"{len(result.figure.data)}"
        )


def test_bar_chart(
    chart_tool: ChartTool,
) -> bool:
    """
    Test vertical bar-chart rendering.
    """

    specification = ChartSpecification(
        chart_type="bar",
        title="Revenue by Category",
        x_column="Category",
        y_columns=[
            "Revenue",
        ],
        category_column=None,
        rationale=(
            "A bar chart compares revenue "
            "across categories."
        ),
    )

    result = chart_tool.render(
        chart_specification=specification,
        query_result=create_category_result(),
    )

    print_result(result)

    require_requested_chart(
        result,
        "bar",
    )

    return True


def test_horizontal_bar_chart(
    chart_tool: ChartTool,
) -> bool:
    """
    Test horizontal bar-chart rendering.
    """

    specification = ChartSpecification(
        chart_type="horizontal_bar",
        title="Revenue Ranking",
        x_column="Category",
        y_columns=[
            "Revenue",
        ],
        category_column=None,
        rationale=(
            "A horizontal bar chart supports "
            "category ranking."
        ),
    )

    result = chart_tool.render(
        chart_specification=specification,
        query_result=create_category_result(),
    )

    print_result(result)

    require_requested_chart(
        result,
        "horizontal_bar",
    )

    return True


def test_line_chart(
    chart_tool: ChartTool,
) -> bool:
    """
    Test line-chart rendering.
    """

    specification = ChartSpecification(
        chart_type="line",
        title="Monthly Revenue Trend",
        x_column="Month",
        y_columns=[
            "Revenue",
        ],
        category_column=None,
        rationale=(
            "A line chart displays changes "
            "over ordered months."
        ),
    )

    result = chart_tool.render(
        chart_specification=specification,
        query_result=create_time_series_result(),
    )

    print_result(result)

    require_requested_chart(
        result,
        "line",
    )

    return True


def test_area_chart(
    chart_tool: ChartTool,
) -> bool:
    """
    Test area-chart rendering.
    """

    specification = ChartSpecification(
        chart_type="area",
        title="Monthly Revenue",
        x_column="Month",
        y_columns=[
            "Revenue",
        ],
        category_column=None,
        rationale=(
            "An area chart shows the magnitude "
            "of revenue over time."
        ),
    )

    result = chart_tool.render(
        chart_specification=specification,
        query_result=create_time_series_result(),
    )

    print_result(result)

    require_requested_chart(
        result,
        "area",
    )

    return True


def test_scatter_chart(
    chart_tool: ChartTool,
) -> bool:
    """
    Test scatter-chart rendering.
    """

    specification = ChartSpecification(
        chart_type="scatter",
        title="Orders and Revenue",
        x_column="OrderCount",
        y_columns=[
            "Revenue",
        ],
        category_column="Region",
        rationale=(
            "A scatter chart shows the relationship "
            "between order count and revenue."
        ),
    )

    result = chart_tool.render(
        chart_specification=specification,
        query_result=create_scatter_result(),
    )

    print_result(result)

    require_requested_chart(
        result,
        "scatter",
    )

    return True


def test_pie_chart(
    chart_tool: ChartTool,
) -> bool:
    """
    Test pie-chart rendering.
    """

    specification = ChartSpecification(
        chart_type="pie",
        title="Revenue Contribution",
        x_column="Category",
        y_columns=[
            "Revenue",
        ],
        category_column=None,
        rationale=(
            "A pie chart shows category "
            "contribution to the total."
        ),
    )

    result = chart_tool.render(
        chart_specification=specification,
        query_result=create_category_result(),
    )

    print_result(result)

    require_requested_chart(
        result,
        "pie",
    )

    return True


def test_table_chart(
    chart_tool: ChartTool,
) -> bool:
    """
    Test explicit table rendering.
    """

    specification = ChartSpecification(
        chart_type="table",
        title="Category Results",
        x_column=None,
        y_columns=[],
        category_column=None,
        rationale=(
            "A table displays the exact result values."
        ),
    )

    result = chart_tool.render(
        chart_specification=specification,
        query_result=create_category_result(),
    )

    print_result(result)
    require_figure(result)

    if result.rendered_chart_type != "table":
        raise AssertionError(
            "Explicit table request did not "
            "render a table."
        )

    if result.used_fallback:
        raise AssertionError(
            "An explicit table request should not "
            "be marked as a fallback."
        )

    return True


def test_none_chart(
    chart_tool: ChartTool,
) -> bool:
    """
    Test chart_type='none'.
    """

    specification = ChartSpecification(
        chart_type="none",
        title="No Visualization Required",
        x_column=None,
        y_columns=[],
        category_column=None,
        rationale=(
            "The analysis does not require "
            "a visualization."
        ),
    )

    result = chart_tool.render(
        chart_specification=specification,
        query_result=create_category_result(),
    )

    print_result(result)

    if result.figure is not None:
        raise AssertionError(
            "chart_type='none' should not "
            "return a figure."
        )

    if result.rendered_chart_type != "none":
        raise AssertionError(
            "Expected rendered chart type 'none'."
        )

    if result.used_fallback:
        raise AssertionError(
            "chart_type='none' is intentional and "
            "should not be a fallback."
        )

    return True


def test_missing_column_fallback(
    chart_tool: ChartTool,
) -> bool:
    """
    Test fallback when Agent 5 references unavailable columns.
    """

    specification = ChartSpecification(
        chart_type="bar",
        title="Invalid Column Test",
        x_column="MissingCategory",
        y_columns=[
            "MissingRevenue",
        ],
        category_column=None,
        rationale=(
            "This specification intentionally "
            "uses missing columns."
        ),
    )

    result = chart_tool.render(
        chart_specification=specification,
        query_result=create_category_result(),
    )

    print_result(result)
    require_table_fallback(result)

    warning_text = " ".join(
        result.warnings
    ).lower()

    if "unavailable columns" not in warning_text:
        raise AssertionError(
            "Missing-column fallback did not "
            "explain the missing columns."
        )

    return True


def test_nonnumeric_y_fallback(
    chart_tool: ChartTool,
) -> bool:
    """
    Test fallback for a nonnumeric Y-axis.
    """

    specification = ChartSpecification(
        chart_type="bar",
        title="Nonnumeric Measure",
        x_column="Category",
        y_columns=[
            "Performance",
        ],
        category_column=None,
        rationale=(
            "This specification intentionally uses "
            "a nonnumeric Y-axis."
        ),
    )

    result = chart_tool.render(
        chart_specification=specification,
        query_result=create_text_value_result(),
    )

    print_result(result)
    require_table_fallback(result)

    warning_text = " ".join(
        result.warnings
    ).lower()

    if "not numeric" not in warning_text:
        raise AssertionError(
            "Nonnumeric fallback did not explain "
            "the incompatible Y-axis."
        )

    return True


def test_empty_result_handling(
    chart_tool: ChartTool,
) -> bool:
    """
    Test chart behavior when SQL returns no rows.
    """

    specification = ChartSpecification(
        chart_type="bar",
        title="Empty Result",
        x_column="Category",
        y_columns=[
            "Revenue",
        ],
        category_column=None,
        rationale=(
            "The requested chart has no rows."
        ),
    )

    result = chart_tool.render(
        chart_specification=specification,
        query_result=create_empty_result(),
    )

    print_result(result)

    if result.figure is not None:
        raise AssertionError(
            "Empty results should not produce "
            "a Plotly figure."
        )

    if result.rendered_chart_type != "none":
        raise AssertionError(
            "Empty results should render "
            "chart type 'none'."
        )

    if not result.used_fallback:
        raise AssertionError(
            "Empty result handling should set "
            "used_fallback=True."
        )

    if not result.warnings:
        raise AssertionError(
            "Empty result handling should "
            "contain a warning."
        )

    return True


def test_row_limit_warning(
    chart_tool: ChartTool,
) -> bool:
    """
    Test chart-row limitation.
    """

    limited_tool = ChartTool(
        maximum_chart_rows=5
    )

    specification = ChartSpecification(
        chart_type="bar",
        title="Limited Category Results",
        x_column="Category",
        y_columns=[
            "Revenue",
        ],
        category_column=None,
        rationale=(
            "A bar chart displays category values."
        ),
    )

    result = limited_tool.render(
        chart_specification=specification,
        query_result=create_large_result(
            row_count=20
        ),
    )

    print_result(result)

    require_requested_chart(
        result,
        "bar",
    )

    if not result.warnings:
        raise AssertionError(
            "Expected a warning when chart rows "
            "exceed maximum_chart_rows."
        )

    warning_text = " ".join(
        result.warnings
    ).lower()

    if "first 5" not in warning_text:
        raise AssertionError(
            "The row-limit warning does not mention "
            "the configured limit."
        )

    return True


def test_invalid_chart_input(
    chart_tool: ChartTool,
) -> bool:
    """
    Test rejection of an invalid chart object.
    """

    try:
        chart_tool.render(
            chart_specification={
                "chart_type": "bar"
            },  # type: ignore[arg-type]
            query_result=create_category_result(),
        )

    except TypeError as exc:
        print(
            f"Expected type error: {exc}"
        )

        return True

    raise AssertionError(
        "ChartTool accepted an invalid "
        "chart_specification object."
    )


def test_invalid_query_result_input(
    chart_tool: ChartTool,
) -> bool:
    """
    Test rejection of an invalid query-result object.
    """

    specification = ChartSpecification(
        chart_type="table",
        title="Invalid Query Result",
        x_column=None,
        y_columns=[],
        category_column=None,
        rationale=(
            "Input validation test."
        ),
    )

    try:
        chart_tool.render(
            chart_specification=specification,
            query_result={
                "rows": []
            },  # type: ignore[arg-type]
        )

    except TypeError as exc:
        print(
            f"Expected type error: {exc}"
        )

        return True

    raise AssertionError(
        "ChartTool accepted an invalid "
        "query_result object."
    )


def run_test(
    test_number: int,
    test_name: str,
    test_function: Callable[
        [ChartTool],
        bool,
    ],
    chart_tool: ChartTool,
) -> bool:
    """
    Run one ChartTool test.
    """

    print_test_header(
        test_number=test_number,
        test_name=test_name,
    )

    try:
        test_function(
            chart_tool
        )

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
    Run all ChartTool tests.
    """

    print("\n" + "=" * 80)
    print("CHART TOOL TEST SUITE")
    print("=" * 80)

    try:
        chart_tool = ChartTool(
            maximum_chart_rows=100
        )

    except Exception as exc:
        print(
            "\nCould not initialize ChartTool."
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise SystemExit(1) from exc

    tests: list[
        tuple[
            str,
            Callable[[ChartTool], bool],
        ]
    ] = [
        (
            "Vertical bar chart",
            test_bar_chart,
        ),
        (
            "Horizontal bar chart",
            test_horizontal_bar_chart,
        ),
        (
            "Line chart",
            test_line_chart,
        ),
        (
            "Area chart",
            test_area_chart,
        ),
        (
            "Scatter chart",
            test_scatter_chart,
        ),
        (
            "Pie chart",
            test_pie_chart,
        ),
        (
            "Explicit table chart",
            test_table_chart,
        ),
        (
            "No chart requested",
            test_none_chart,
        ),
        (
            "Missing-column table fallback",
            test_missing_column_fallback,
        ),
        (
            "Nonnumeric Y-axis fallback",
            test_nonnumeric_y_fallback,
        ),
        (
            "Empty-result handling",
            test_empty_result_handling,
        ),
        (
            "Maximum chart-row limit",
            test_row_limit_warning,
        ),
        (
            "Invalid chart input rejection",
            test_invalid_chart_input,
        ),
        (
            "Invalid query-result input rejection",
            test_invalid_query_result_input,
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
            chart_tool=chart_tool,
        )

        if passed:
            passed_tests += 1

    total_tests = len(tests)

    failed_tests = (
        total_tests - passed_tests
    )

    print("\n" + "=" * 80)
    print("FINAL CHART TOOL TEST SUMMARY")
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
            "\nRESULT: ALL CHART TOOL TESTS PASSED"
        )

        print(
            "Chart rendering is ready for "
            "Streamlit integration."
        )

        raise SystemExit(0)

    print(
        "\nRESULT: SOME CHART TOOL TESTS FAILED"
    )

    print(
        "Resolve the failed chart cases before "
        "adding visualization to Streamlit."
    )

    raise SystemExit(1)


if __name__ == "__main__":
    main()
    