"""
Agent 5: Reporting Agent.

Agent 5 converts validated analytical results into a business-friendly
final report.

Inputs:

1. Original business question
2. Validated IntentResult
3. Validated AnalysisResult from Agent 4
4. Actual QueryExecutionResult from Agent 3b

Output:

1. Business summary
2. Key findings
3. Root-cause indicators
4. Actionable recommendations
5. Chart specification
6. Analysis limitations

The Reporting Agent does not:

- Generate SQL
- Execute SQL
- Modify database records
- Recalculate business measures
- Invent analytical values
- Present possible causes as proven causes
"""

import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from ollama import Client
from pydantic import ValidationError

from models.schemas import (
    AnalysisResult,
    ChartSpecification,
    FinalReport,
    IntentResult,
    QueryExecutionResult,
)


class ReportingAgent:
    """
    Generates the final validated business report.

    The local LLM creates business-friendly language and recommendations
    using only the validated evidence produced by earlier agents.
    """

    def __init__(
        self,
        model: str = "llama3.2:latest",
        host: str = "http://localhost:11434",
        context_length: int = 8192,
        max_output_tokens: int = 1800,
        maximum_sample_rows: int = 30,
    ) -> None:
        """
        Initialize Agent 5.

        Args:
            model:
                Ollama model used for report generation.

            host:
                Local Ollama API address.

            context_length:
                Context window for the reporting request.

            max_output_tokens:
                Maximum number of generated output tokens.

            maximum_sample_rows:
                Maximum query-result rows included in the prompt.
        """

        if not isinstance(model, str):
            raise TypeError(
                "model must be a string."
            )

        if not isinstance(host, str):
            raise TypeError(
                "host must be a string."
            )

        if not isinstance(context_length, int):
            raise TypeError(
                "context_length must be an integer."
            )

        if not isinstance(max_output_tokens, int):
            raise TypeError(
                "max_output_tokens must be an integer."
            )

        if not isinstance(maximum_sample_rows, int):
            raise TypeError(
                "maximum_sample_rows must be an integer."
            )

        cleaned_model = model.strip()
        cleaned_host = host.strip()

        if not cleaned_model:
            raise ValueError(
                "model cannot be empty."
            )

        if not cleaned_host:
            raise ValueError(
                "host cannot be empty."
            )

        if context_length < 4096:
            raise ValueError(
                "context_length must be at least 4096."
            )

        if max_output_tokens < 1:
            raise ValueError(
                "max_output_tokens must be greater than zero."
            )

        if maximum_sample_rows < 1:
            raise ValueError(
                "maximum_sample_rows must be greater than zero."
            )

        self.model = cleaned_model
        self.host = cleaned_host.rstrip("/")
        self.context_length = context_length
        self.max_output_tokens = max_output_tokens
        self.maximum_sample_rows = maximum_sample_rows

        self.client = Client(
            host=self.host
        )

    def _system_prompt(self) -> str:
        """
        Return Agent 5 reporting instructions.
        """

        return """
You are Agent 5, the Reporting Agent in a multi-agent business data
analytics system.

You receive validated analytical evidence produced from real database
results.

Create a concise, business-friendly FinalReport.

The report must contain:

1. Business summary
2. Key findings
3. Root-cause indicators
4. Actionable recommendations
5. One chart specification when visualization is useful
6. Limitations

Evidence rules:

- Use only the supplied AnalysisResult and query-result evidence.
- Do not invent values, percentages, dates, categories, or causes.
- Do not perform new calculations.
- Preserve exact evidence when mentioning numbers.
- Do not present possible drivers as proven causes.
- Root-cause indicators must be supported by the analysis evidence.
- If evidence cannot establish root cause, say so clearly.
- Recommendations must connect directly to a finding or possible driver.
- Avoid vague recommendations such as "improve performance".
- Recommendations must state a specific business action.
- Do not recommend irreversible actions based on limited evidence.
- Include relevant limitations from the Analysis Agent.
- Mention truncation when the query result is truncated.

Chart rules:

- Use only columns that exist in QUERY RESULT METADATA.
- Choose a chart only when the returned structure supports it.
- Use "line" for ordered time-series data.
- Use "bar" or "horizontal_bar" for category comparisons and rankings.
- Use "scatter" for relationships between numeric measures.
- Use "pie" only for a small number of parts representing a whole.
- Use "table" when the result is unsuitable for a graphical chart.
- Use "none" when no visualization is useful.
- Do not invent x_column, y_columns, or category_column names.
- Chart title and rationale must be business-friendly.

Return only JSON matching the supplied FinalReport schema.
""".strip()

    def _safe_value(
        self,
        value: Any,
    ) -> Any:
        """
        Convert database values into JSON-safe values.
        """

        if value is None:
            return None

        if isinstance(
            value,
            (datetime, date),
        ):
            return value.isoformat()

        if isinstance(value, Decimal):
            return float(value)

        if hasattr(value, "item"):
            try:
                value = value.item()

            except Exception:
                pass

        if isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
                list,
                dict,
            ),
        ):
            return value

        return str(value)

    def _prepare_query_context(
        self,
        query_result: QueryExecutionResult,
    ) -> dict[str, Any]:
        """
        Prepare limited query-result context for report generation.

        The complete result stays in GraphState. Only a limited sample
        is sent to the LLM.
        """

        sample_rows = []

        for row in query_result.rows[
            :self.maximum_sample_rows
        ]:
            safe_row = {
                str(column): self._safe_value(
                    value
                )
                for column, value in row.items()
            }

            sample_rows.append(safe_row)

        return {
            "columns": list(
                query_result.columns
            ),
            "row_count": (
                query_result.row_count
            ),
            "truncated": (
                query_result.truncated
            ),
            "execution_time_ms": (
                query_result.execution_time_ms
            ),
            "sample_row_count": len(
                sample_rows
            ),
            "sample_rows": sample_rows,
        }

    def _build_user_prompt(
        self,
        user_question: str,
        intent: IntentResult,
        analysis_result: AnalysisResult,
        query_result: QueryExecutionResult,
    ) -> str:
        """
        Build an evidence-grounded reporting prompt.
        """

        query_context = (
            self._prepare_query_context(
                query_result
            )
        )

        query_context_json = json.dumps(
            query_context,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

        return f"""
Create the final business report for the following analytical request.

BUSINESS QUESTION:

{user_question}

VALIDATED INTENT:

{intent.model_dump_json(indent=2)}

VALIDATED ANALYSIS RESULT:

{analysis_result.model_dump_json(indent=2)}

QUERY RESULT METADATA AND SAMPLE:

{query_context_json}

Create a FinalReport.

Requirements:

1. The business summary must directly answer the business question.
2. Key findings must come from AnalysisResult.
3. Root-cause indicators must remain cautious and evidence-based.
4. Recommendations must be linked to findings or possible drivers.
5. Recommend only one chart specification.
6. Chart columns must exactly match available query-result columns.
7. Preserve all important analysis limitations.
8. Do not generate SQL.
9. Do not invent values.
10. Return only JSON matching the FinalReport schema.
""".strip()

    def _call_model(
        self,
        prompt: str,
    ) -> str:
        """
        Call llama3.2 through the local Ollama API.
        """

        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                format=(
                    FinalReport
                    .model_json_schema()
                ),
                stream=False,
                options={
                    "temperature": 0,
                    "seed": 42,
                    "num_ctx": self.context_length,
                    "num_predict": (
                        self.max_output_tokens
                    ),
                },
            )

        except Exception as exc:
            error_text = str(exc)

            if (
                "exceeds the available context size"
                in error_text.lower()
            ):
                raise RuntimeError(
                    "The Reporting Agent prompt exceeded "
                    "the configured context window.\n"
                    f"Context length: "
                    f"{self.context_length}\n"
                    "Reduce maximum_sample_rows or increase "
                    f"context_length.\nOriginal error: {exc}"
                ) from exc

            raise ConnectionError(
                "Could not call the Reporting Agent "
                "through Ollama.\n"
                f"Host: {self.host}\n"
                f"Model: {self.model}\n"
                f"Original error: {exc}"
            ) from exc

        content = response.message.content

        if content is None:
            raise RuntimeError(
                "The Reporting Agent returned no content."
            )

        cleaned_content = content.strip()

        if not cleaned_content:
            raise RuntimeError(
                "The Reporting Agent returned an "
                "empty response."
            )

        return cleaned_content

    def _validate_model_output(
        self,
        model_output: str,
    ) -> FinalReport:
        """
        Validate Agent 5 output using FinalReport.
        """

        try:
            return FinalReport.model_validate_json(
                model_output
            )

        except ValidationError as exc:
            raise RuntimeError(
                "The Reporting Agent returned output "
                "that did not match FinalReport.\n\n"
                f"Raw output:\n{model_output}\n\n"
                f"Validation error:\n{exc}"
            ) from exc

    def _validate_chart_columns(
        self,
        report: FinalReport,
        query_result: QueryExecutionResult,
    ) -> FinalReport:
        """
        Ensure chart columns exist in the actual query result.

        If the model invents a chart column, the chart is replaced with
        a safe table specification. The report content is preserved.
        """

        chart = report.chart

        if chart is None:
            return report

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

        if not invalid_columns:
            return report

        report.chart = ChartSpecification(
            chart_type="table",
            title="Query Results",
            x_column=None,
            y_columns=[],
            category_column=None,
            rationale=(
                "A table is used because the originally "
                "recommended chart referenced columns that "
                "are not present in the query result."
            ),
        )

        chart_limitation = (
            "The initial chart recommendation referenced unavailable "
            "result columns, so the output was converted to a table."
        )

        if chart_limitation not in report.limitations:
            report.limitations.append(
                chart_limitation
            )

        return report

    def _add_missing_limitations(
        self,
        report: FinalReport,
        analysis_result: AnalysisResult,
        query_result: QueryExecutionResult,
    ) -> FinalReport:
        """
        Preserve Analysis Agent limitations and truncation warnings.
        """

        existing_limitations = {
            limitation.strip().lower()
            for limitation in report.limitations
            if limitation.strip()
        }

        for limitation in analysis_result.limitations:
            normalized_limitation = (
                limitation.strip().lower()
            )

            if (
                normalized_limitation
                and normalized_limitation
                not in existing_limitations
            ):
                report.limitations.append(
                    limitation
                )

                existing_limitations.add(
                    normalized_limitation
                )

        if query_result.truncated:
            truncation_message = (
                "The database result was truncated at the configured "
                "row limit, so the report reflects only returned rows."
            )

            if (
                truncation_message.lower()
                not in existing_limitations
            ):
                report.limitations.append(
                    truncation_message
                )

        return report

    def _empty_result_report(
        self,
        analysis_result: AnalysisResult,
    ) -> FinalReport:
        """
        Return a deterministic report when no database rows exist.
        """

        limitations = list(
            analysis_result.limitations
        )

        empty_limitation = (
            "No records were available for charting or "
            "business recommendations."
        )

        if empty_limitation not in limitations:
            limitations.append(
                empty_limitation
            )

        return FinalReport(
            business_summary=(
                analysis_result.summary
            ),
            key_findings=[],
            root_cause_indicators=[],
            recommendations=[],
            chart=ChartSpecification(
                chart_type="none",
                title="No Data Available",
                x_column=None,
                y_columns=[],
                category_column=None,
                rationale=(
                    "No chart can be created because "
                    "the query returned no records."
                ),
            ),
            limitations=limitations,
        )

    def create_report(
        self,
        user_question: str,
        intent: IntentResult,
        analysis_result: AnalysisResult,
        query_result: QueryExecutionResult,
    ) -> FinalReport:
        """
        Create a validated business report.

        Returns:
            FinalReport.
        """

        if not isinstance(user_question, str):
            raise TypeError(
                "user_question must be a string."
            )

        if not isinstance(intent, IntentResult):
            raise TypeError(
                "intent must be an IntentResult."
            )

        if not isinstance(
            analysis_result,
            AnalysisResult,
        ):
            raise TypeError(
                "analysis_result must be an "
                "AnalysisResult."
            )

        if not isinstance(
            query_result,
            QueryExecutionResult,
        ):
            raise TypeError(
                "query_result must be a "
                "QueryExecutionResult."
            )

        cleaned_question = (
            user_question.strip()
        )

        if not cleaned_question:
            raise ValueError(
                "user_question cannot be empty."
            )

        if intent.clarification_required:
            raise ValueError(
                "Reporting cannot begin while "
                "clarification is required."
            )

        if query_result.row_count == 0:
            return self._empty_result_report(
                analysis_result
            )

        prompt = self._build_user_prompt(
            user_question=cleaned_question,
            intent=intent,
            analysis_result=analysis_result,
            query_result=query_result,
        )

        model_output = self._call_model(
            prompt
        )

        report = self._validate_model_output(
            model_output
        )

        report = self._validate_chart_columns(
            report=report,
            query_result=query_result,
        )

        report = self._add_missing_limitations(
            report=report,
            analysis_result=analysis_result,
            query_result=query_result,
        )

        return report


def main() -> None:
    """
    Run Agent 5 using the current LangGraph output through Agent 4.

    The current compiled graph must already include Agent 4.
    """

    from workflow.graph import run_workflow

    print("=" * 80)
    print("AGENT 5: REPORTING SMOKE TEST")
    print("=" * 80)

    question = (
        "Show the top 5 products by revenue."
    )

    try:
        state = run_workflow(
            user_question=question,
            max_sql_retries=3,
            recursion_limit=30,
        )

        if state.get(
            "workflow_status"
        ) == "failed":
            raise RuntimeError(
                state.get(
                    "error_message",
                    "Workflow through Agent 4 failed.",
                )
            )

        intent = state.get("intent")

        analysis_result = state.get(
            "analysis_result"
        )

        query_result = state.get(
            "query_result"
        )

        if intent is None:
            raise RuntimeError(
                "IntentResult is missing."
            )

        if analysis_result is None:
            raise RuntimeError(
                "AnalysisResult is missing."
            )

        if query_result is None:
            raise RuntimeError(
                "QueryExecutionResult is missing."
            )

        agent = ReportingAgent(
            model="llama3.2:latest",
            host="http://localhost:11434",
            context_length=8192,
            max_output_tokens=1800,
            maximum_sample_rows=30,
        )

        report = agent.create_report(
            user_question=question,
            intent=intent,
            analysis_result=analysis_result,
            query_result=query_result,
        )

        print("\nFinalReport:")
        print(
            report.model_dump_json(
                indent=2
            )
        )

        print("\n" + "=" * 80)
        print("RESULT: AGENT 5 IS WORKING")
        print("=" * 80)

    except Exception as exc:
        print("\n" + "=" * 80)
        print("RESULT: AGENT 5 FAILED")
        print("=" * 80)
        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()