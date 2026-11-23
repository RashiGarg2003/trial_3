"""
Chart rendering tool for the Multi-Agent Data Analytics System.

This tool converts Agent 5's validated ChartSpecification and the
actual QueryExecutionResult into a Plotly figure.

Responsibilities:

1. Convert query rows into a Pandas DataFrame.
2. Validate that chart columns exist in the actual result.
3. Validate numeric columns where required.
4. Render supported Plotly charts.
5. Fall back safely to a table when a graphical chart is incompatible.
6. Return structured rendering metadata.

Supported chart types:

- line
- bar
- horizontal_bar
- area
- scatter
- pie
- table
- none

This component does not:

- Call an LLM.
- Generate SQL.
- Execute SQL.
- Modify database records.
- Invent chart columns.
- Recalculate business findings.
"""

from dataclasses import dataclass
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from models.schemas import (
    ChartSpecification,
    QueryExecutionResult,
)


@dataclass
class ChartRenderResult:
    """
    Structured chart rendering result.

    Attributes:
        figure:
            Plotly figure or None when no visualization is requested.

        rendered_chart_type:
            Chart type that was actually rendered.

        used_fallback:
            True when the requested graphical chart was replaced by
            a table.

        warnings:
            Non-fatal chart validation or rendering messages.
    """

    figure: Optional[go.Figure]
    rendered_chart_type: str
    used_fallback: bool
    warnings: list[str]


class ChartTool:
    """
    Validates and renders Plotly charts from actual query results.
    """

    GRAPHICAL_CHART_TYPES = {
        "line",
        "bar",
        "horizontal_bar",
        "area",
        "scatter",
        "pie",
    }

    def __init__(
        self,
        maximum_chart_rows: int = 100,
    ) -> None:
        """
        Initialize the chart tool.

        Args:
            maximum_chart_rows:
                Maximum number of returned rows used in a graphical
                chart. A larger result is limited for readability.

                Table rendering can still display the same validated
                subset.
        """

        if not isinstance(
            maximum_chart_rows,
            int,
        ):
            raise TypeError(
                "maximum_chart_rows must be an integer."
            )

        if maximum_chart_rows < 1:
            raise ValueError(
                "maximum_chart_rows must be greater than zero."
            )

        self.maximum_chart_rows = (
            maximum_chart_rows
        )

    def _build_dataframe(
        self,
        query_result: QueryExecutionResult,
    ) -> pd.DataFrame:
        """
        Convert QueryExecutionResult into a DataFrame.

        The DataFrame columns preserve the exact database result names.
        """

        dataframe = pd.DataFrame(
            query_result.rows,
            columns=query_result.columns,
        )

        return dataframe

    def _validate_inputs(
        self,
        chart_specification: ChartSpecification,
        query_result: QueryExecutionResult,
    ) -> None:
        """
        Validate input object types.
        """

        if not isinstance(
            chart_specification,
            ChartSpecification,
        ):
            raise TypeError(
                "chart_specification must be a "
                "ChartSpecification instance."
            )

        if not isinstance(
            query_result,
            QueryExecutionResult,
        ):
            raise TypeError(
                "query_result must be a "
                "QueryExecutionResult instance."
            )

    def _referenced_columns(
        self,
        chart_specification: ChartSpecification,
    ) -> list[str]:
        """
        Return all columns referenced by the chart specification.
        """

        columns: list[str] = []

        if chart_specification.x_column:
            columns.append(
                chart_specification.x_column
            )

        columns.extend(
            chart_specification.y_columns
        )

        if chart_specification.category_column:
            columns.append(
                chart_specification.category_column
            )

        unique_columns: list[str] = []
        seen_columns: set[str] = set()

        for column in columns:
            normalized_column = (
                column.strip().lower()
            )

            if normalized_column in seen_columns:
                continue

            seen_columns.add(
                normalized_column
            )

            unique_columns.append(column)

        return unique_columns

    def _missing_columns(
        self,
        chart_specification: ChartSpecification,
        dataframe: pd.DataFrame,
    ) -> list[str]:
        """
        Return specification columns missing from the DataFrame.
        """

        available_columns = set(
            str(column)
            for column in dataframe.columns
        )

        return [
            column
            for column in self._referenced_columns(
                chart_specification
            )
            if column not in available_columns
        ]

    def _prepare_numeric_column(
        self,
        dataframe: pd.DataFrame,
        column: str,
    ) -> bool:
        """
        Convert one result column to numeric when possible.

        Returns:
            True when all non-null values are numeric or can be
            converted to numeric.

            False when the column contains incompatible values.
        """

        if column not in dataframe.columns:
            return False

        original_series = dataframe[
            column
        ]

        non_null_series = (
            original_series.dropna()
        )

        if non_null_series.empty:
            return False

        numeric_series = pd.to_numeric(
            original_series,
            errors="coerce",
        )

        original_non_null_count = int(
            original_series.notna().sum()
        )

        converted_non_null_count = int(
            numeric_series.notna().sum()
        )

        if (
            original_non_null_count
            != converted_non_null_count
        ):
            return False

        dataframe[column] = numeric_series

        return True

    def _validate_chart_structure(
        self,
        chart_specification: ChartSpecification,
        dataframe: pd.DataFrame,
    ) -> list[str]:
        """
        Validate structural chart requirements.

        Returns:
            A list of compatibility warnings. Any warning causes a
            safe table fallback.
        """

        warnings: list[str] = []

        chart_type = (
            chart_specification.chart_type
        )

        if chart_type in {
            "table",
            "none",
        }:
            return warnings

        if not chart_specification.x_column:
            warnings.append(
                f"Chart type '{chart_type}' requires "
                "an x_column."
            )

        if not chart_specification.y_columns:
            warnings.append(
                f"Chart type '{chart_type}' requires "
                "at least one y-column."
            )

        missing_columns = self._missing_columns(
            chart_specification=(
                chart_specification
            ),
            dataframe=dataframe,
        )

        if missing_columns:
            warnings.append(
                "Chart specification references unavailable "
                f"columns: {missing_columns}."
            )

            return warnings

        for y_column in (
            chart_specification.y_columns
        ):
            is_numeric = (
                self._prepare_numeric_column(
                    dataframe=dataframe,
                    column=y_column,
                )
            )

            if not is_numeric:
                warnings.append(
                    f"Y-axis column '{y_column}' "
                    "is not numeric."
                )

        if (
            chart_type == "scatter"
            and len(
                chart_specification.y_columns
            )
            != 1
        ):
            warnings.append(
                "A scatter chart requires exactly "
                "one y-column."
            )

        if (
            chart_type == "pie"
            and len(
                chart_specification.y_columns
            )
            != 1
        ):
            warnings.append(
                "A pie chart requires exactly "
                "one numeric value column."
            )

        return warnings

    def _limit_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Limit displayed rows for readability.
        """

        if len(dataframe) <= self.maximum_chart_rows:
            return dataframe, []

        limited_dataframe = dataframe.head(
            self.maximum_chart_rows
        ).copy()

        warnings = [
            (
                "The chart displays only the first "
                f"{self.maximum_chart_rows} of "
                f"{len(dataframe)} returned rows."
            )
        ]

        return limited_dataframe, warnings

    def _render_table(
        self,
        dataframe: pd.DataFrame,
        title: str,
    ) -> go.Figure:
        """
        Render query results as a Plotly table.
        """

        header_values = [
            str(column).replace("_", " ").strip().title()
            for column in dataframe.columns
        ]

        def display_value(value):
            if value is None or pd.isna(value):
                return "—"
            if isinstance(value, float):
                return f"{value:,.2f}" if value % 1 else f"{value:,.0f}"
            if isinstance(value, int) and not isinstance(value, bool):
                return f"{value:,}"
            return str(value)

        cell_values = [
            [display_value(value) for value in dataframe[column].tolist()]
            for column in dataframe.columns
        ]

        figure = go.Figure(
            data=[
                go.Table(
                    header={
                        "values": header_values,
                        "align": "left",
                    },
                    cells={
                        "values": cell_values,
                        "align": "left",
                    },
                )
            ]
        )

        figure.update_layout(
            title=title,
        )

        return figure

    def _render_line(
        self,
        dataframe: pd.DataFrame,
        chart: ChartSpecification,
    ) -> go.Figure:
        """
        Render a line chart.
        """

        return px.line(
            dataframe,
            x=chart.x_column,
            y=chart.y_columns,
            color=chart.category_column,
            markers=True,
            title=chart.title,
        )

    def _render_bar(
        self,
        dataframe: pd.DataFrame,
        chart: ChartSpecification,
    ) -> go.Figure:
        """
        Render a vertical bar chart.
        """

        return px.bar(
            dataframe,
            x=chart.x_column,
            y=chart.y_columns,
            color=chart.category_column,
            title=chart.title,
            barmode="group",
        )

    def _render_horizontal_bar(
        self,
        dataframe: pd.DataFrame,
        chart: ChartSpecification,
    ) -> go.Figure:
        """
        Render a horizontal bar chart.

        The first numeric Y column becomes the horizontal value axis.
        The configured X column becomes the category axis.
        """

        value_column = (
            chart.y_columns[0]
        )

        return px.bar(
            dataframe,
            x=value_column,
            y=chart.x_column,
            color=chart.category_column,
            title=chart.title,
            orientation="h",
        )

    def _render_area(
        self,
        dataframe: pd.DataFrame,
        chart: ChartSpecification,
    ) -> go.Figure:
        """
        Render an area chart.
        """

        return px.area(
            dataframe,
            x=chart.x_column,
            y=chart.y_columns,
            color=chart.category_column,
            title=chart.title,
        )

    def _render_scatter(
        self,
        dataframe: pd.DataFrame,
        chart: ChartSpecification,
    ) -> go.Figure:
        """
        Render a scatter chart.
        """

        return px.scatter(
            dataframe,
            x=chart.x_column,
            y=chart.y_columns[0],
            color=chart.category_column,
            title=chart.title,
        )

    def _render_pie(
        self,
        dataframe: pd.DataFrame,
        chart: ChartSpecification,
    ) -> go.Figure:
        """
        Render a pie chart.
        """

        return px.pie(
            dataframe,
            names=chart.x_column,
            values=chart.y_columns[0],
            color=chart.category_column,
            title=chart.title,
        )

    def _render_graphical_chart(
        self,
        dataframe: pd.DataFrame,
        chart: ChartSpecification,
    ) -> go.Figure:
        """
        Dispatch a validated graphical chart to its renderer.
        """

        renderers = {
            "line": self._render_line,
            "bar": self._render_bar,
            "horizontal_bar": (
                self._render_horizontal_bar
            ),
            "area": self._render_area,
            "scatter": self._render_scatter,
            "pie": self._render_pie,
        }

        renderer = renderers.get(
            chart.chart_type
        )

        if renderer is None:
            raise ValueError(
                "Unsupported graphical chart type: "
                f"{chart.chart_type}"
            )

        return renderer(
            dataframe,
            chart,
        )

    def render(
        self,
        chart_specification: ChartSpecification,
        query_result: QueryExecutionResult,
    ) -> ChartRenderResult:
        """
        Validate and render the requested chart.

        Returns:
            ChartRenderResult containing the Plotly figure and
            rendering metadata.

        Behavior:

        - chart_type='none':
          Returns figure=None.

        - chart_type='table':
          Returns a Plotly table.

        - Compatible graphical specification:
          Returns the requested Plotly chart.

        - Invalid or incompatible graphical specification:
          Returns a safe Plotly table with warnings.
        """

        self._validate_inputs(
            chart_specification=(
                chart_specification
            ),
            query_result=query_result,
        )

        dataframe = self._build_dataframe(
            query_result
        )

        if (
            chart_specification.chart_type
            == "none"
        ):
            return ChartRenderResult(
                figure=None,
                rendered_chart_type="none",
                used_fallback=False,
                warnings=[],
            )

        if dataframe.empty:
            return ChartRenderResult(
                figure=None,
                rendered_chart_type="none",
                used_fallback=True,
                warnings=[
                    (
                        "No chart was rendered because "
                        "the query returned no rows."
                    )
                ],
            )

        display_dataframe, row_warnings = (
            self._limit_dataframe(
                dataframe
            )
        )

        if (
            chart_specification.chart_type
            == "table"
        ):
            figure = self._render_table(
                dataframe=display_dataframe,
                title=chart_specification.title,
            )

            return ChartRenderResult(
                figure=figure,
                rendered_chart_type="table",
                used_fallback=False,
                warnings=row_warnings,
            )

        compatibility_warnings = (
            self._validate_chart_structure(
                chart_specification=(
                    chart_specification
                ),
                dataframe=display_dataframe,
            )
        )

        all_warnings = [
            *row_warnings,
            *compatibility_warnings,
        ]

        if compatibility_warnings:
            figure = self._render_table(
                dataframe=display_dataframe,
                title=(
                    f"{chart_specification.title} "
                    "(Table View)"
                ),
            )

            return ChartRenderResult(
                figure=figure,
                rendered_chart_type="table",
                used_fallback=True,
                warnings=all_warnings,
            )

        try:
            figure = self._render_graphical_chart(
                dataframe=display_dataframe,
                chart=chart_specification,
            )

        except Exception as exc:
            all_warnings.append(
                "The requested chart could not be rendered. "
                f"A table was used instead. Error: {exc}"
            )

            figure = self._render_table(
                dataframe=display_dataframe,
                title=(
                    f"{chart_specification.title} "
                    "(Table View)"
                ),
            )

            return ChartRenderResult(
                figure=figure,
                rendered_chart_type="table",
                used_fallback=True,
                warnings=all_warnings,
            )

        return ChartRenderResult(
            figure=figure,
            rendered_chart_type=(
                chart_specification.chart_type
            ),
            used_fallback=False,
            warnings=all_warnings,
        )


def main() -> None:
    """
    Run a standalone chart-tool smoke test.

    This test first executes the complete LangGraph workflow and then
    renders Agent 5's ChartSpecification using the actual query result.
    """

    from workflow.graph import run_workflow

    print("=" * 80)
    print("CHART TOOL SMOKE TEST")
    print("=" * 80)

    question = (
        "Show the top 5 products by revenue."
    )

    try:
        state = run_workflow(
            user_question=question,
            max_sql_retries=3,
            recursion_limit=35,
        )

        if (
            state.get("workflow_status")
            != "completed"
        ):
            raise RuntimeError(
                "The workflow did not complete. "
                f"Status: "
                f"{state.get('workflow_status')}. "
                f"Error: "
                f"{state.get('error_message')}"
            )

        final_report = state.get(
            "final_report"
        )

        query_result = state.get(
            "query_result"
        )

        if final_report is None:
            raise RuntimeError(
                "FinalReport is missing."
            )

        if query_result is None:
            raise RuntimeError(
                "QueryExecutionResult is missing."
            )

        if final_report.chart is None:
            raise RuntimeError(
                "FinalReport contains no "
                "ChartSpecification."
            )

        chart_tool = ChartTool(
            maximum_chart_rows=100
        )

        render_result = chart_tool.render(
            chart_specification=(
                final_report.chart
            ),
            query_result=query_result,
        )

        print(
            f"Requested chart type : "
            f"{final_report.chart.chart_type}"
        )

        print(
            f"Rendered chart type  : "
            f"{render_result.rendered_chart_type}"
        )

        print(
            f"Used table fallback  : "
            f"{render_result.used_fallback}"
        )

        print(
            f"Warnings             : "
            f"{render_result.warnings}"
        )

        print(
            f"Figure created       : "
            f"{render_result.figure is not None}"
        )

        if (
            final_report.chart.chart_type
            != "none"
            and render_result.figure is None
        ):
            raise RuntimeError(
                "The chart tool did not produce "
                "a Plotly figure."
            )

        print("\n" + "=" * 80)
        print("RESULT: CHART TOOL IS WORKING")
        print("=" * 80)

    except Exception as exc:
        print("\n" + "=" * 80)
        print("RESULT: CHART TOOL FAILED")
        print("=" * 80)
        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()