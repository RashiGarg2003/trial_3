"""
Agent 4: Analysis Agent.

Agent 4 analyzes the actual data returned by Agent 3b.

Responsibilities:

1. Validate the query result.
2. Build a Pandas DataFrame.
3. Calculate deterministic descriptive statistics.
4. Identify numeric, categorical, and time-like columns.
5. Prepare evidence grounded in the returned data.
6. Ask llama3.2:latest to interpret the calculated evidence.
7. Return a validated AnalysisResult.

The LLM does not perform the primary calculations. Python and Pandas
calculate the values, while the LLM explains their business meaning.

This agent does not:

- Generate SQL.
- Execute SQL.
- Modify database records.
- Invent unavailable values.
- Claim causation without supporting evidence.
- Produce the final report or chart.
"""

import json
import math
from datetime import date, datetime
from decimal import Decimal
from typing import Any

import pandas as pd
from ollama import Client
from pydantic import ValidationError

from models.schemas import (
    AnalysisPlan,
    AnalysisResult,
    IntentResult,
    QueryExecutionResult,
)


class AnalysisAgent:
    """
    Evidence-grounded Analysis Agent.

    Pandas calculates deterministic statistics. The local LLM converts
    those calculations into structured business findings.
    """

    def __init__(
        self,
        model: str = "llama3.2:latest",
        host: str = "http://localhost:11434",
        context_length: int = 8192,
        max_output_tokens: int = 1500,
        maximum_sample_rows: int = 50,
    ) -> None:
        """
        Initialize Agent 4.

        Args:
            model:
                Ollama model used for interpretation.

            host:
                Local Ollama API address.

            context_length:
                Context window used for the analysis request.

            max_output_tokens:
                Maximum response length.

            maximum_sample_rows:
                Maximum number of query-result rows included in the
                model prompt. Calculations still use all returned rows.
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
        Return the analysis instructions.
        """

        return """
You are Agent 4, the Analysis Agent in a multi-agent business data
analytics system.

You receive:

1. A validated user intent.
2. A validated analytical plan.
3. The executed SQL query.
4. Deterministic statistics calculated from the real query result.
5. A limited sample of the real query rows.

Your responsibility is to interpret only the supplied evidence and
return an AnalysisResult.

Rules:

- Use only the supplied query result and calculated profile.
- Do not invent values, categories, dates, percentages, or causes.
- Do not perform unsupported calculations mentally.
- Use exact values from the evidence where useful.
- Distinguish observations from possible explanations.
- Treat possible_drivers as evidence-supported contributors, not
  proven causes.
- Do not claim causation from correlation or ranking alone.
- For root-cause analysis, identify only indicators supported by the
  available dimensions and measures.
- If the data cannot establish a cause, state that limitation.
- Mention empty, truncated, or limited results when relevant.
- Keep findings concise and business-friendly.
- Every major finding should contain evidence.
- Add limitations when the query result is incomplete or insufficient.
- Do not generate SQL.
- Do not recommend charts.
- Do not produce final business recommendations.
- Return only JSON matching the supplied AnalysisResult schema.
""".strip()

    def _safe_value(
        self,
        value: Any,
    ) -> Any:
        """
        Convert values into JSON-safe Python objects.
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

        if isinstance(value, float):
            if math.isnan(value):
                return None

            if math.isinf(value):
                return None

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

    def _records_to_dataframe(
        self,
        query_result: QueryExecutionResult,
    ) -> pd.DataFrame:
        """
        Convert the validated query result into a DataFrame.
        """

        dataframe = pd.DataFrame(
            query_result.rows,
            columns=query_result.columns,
        )

        return dataframe

    def _detect_numeric_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> list[str]:
        """
        Detect numeric columns.

        Object columns are tested using numeric conversion because
        SQLite drivers may return some numbers as strings.
        """

        numeric_columns: list[str] = []

        for column in dataframe.columns:
            series = dataframe[column]

            if pd.api.types.is_numeric_dtype(
                series
            ):
                numeric_columns.append(
                    str(column)
                )

                continue

            non_null_series = series.dropna()

            if non_null_series.empty:
                continue

            converted_series = pd.to_numeric(
                non_null_series,
                errors="coerce",
            )

            conversion_ratio = (
                converted_series.notna().sum()
                / len(non_null_series)
            )

            if conversion_ratio == 1.0:
                numeric_columns.append(
                    str(column)
                )

        return numeric_columns

    def _detect_datetime_columns(
        self,
        dataframe: pd.DataFrame,
        numeric_columns: list[str],
    ) -> list[str]:
        """
        Detect columns whose non-null values consistently parse as dates.
        """

        datetime_columns: list[str] = []

        for column in dataframe.columns:
            column_name = str(column)

            if column_name in numeric_columns:
                continue

            series = dataframe[column].dropna()

            if series.empty:
                continue

            converted_series = pd.to_datetime(
                series,
                errors="coerce",
            )

            conversion_ratio = (
                converted_series.notna().sum()
                / len(series)
            )

            if conversion_ratio == 1.0:
                datetime_columns.append(
                    column_name
                )

        return datetime_columns

    def _numeric_profile(
        self,
        dataframe: pd.DataFrame,
        numeric_columns: list[str],
    ) -> dict[str, Any]:
        """
        Calculate deterministic statistics for numeric columns.
        """

        profile: dict[str, Any] = {}

        for column in numeric_columns:
            numeric_series = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            ).dropna()

            if numeric_series.empty:
                continue

            column_profile: dict[str, Any] = {
                "count": int(
                    numeric_series.count()
                ),
                "sum": self._safe_value(
                    numeric_series.sum()
                ),
                "mean": self._safe_value(
                    numeric_series.mean()
                ),
                "median": self._safe_value(
                    numeric_series.median()
                ),
                "minimum": self._safe_value(
                    numeric_series.min()
                ),
                "maximum": self._safe_value(
                    numeric_series.max()
                ),
            }

            if len(numeric_series) > 1:
                column_profile[
                    "standard_deviation"
                ] = self._safe_value(
                    numeric_series.std()
                )

            else:
                column_profile[
                    "standard_deviation"
                ] = None

            profile[column] = column_profile

        return profile

    def _categorical_profile(
        self,
        dataframe: pd.DataFrame,
        excluded_columns: set[str],
    ) -> dict[str, Any]:
        """
        Calculate value-frequency information for categorical columns.

        Only the five most frequent values are included in the prompt.
        """

        profile: dict[str, Any] = {}

        for column in dataframe.columns:
            column_name = str(column)

            if column_name in excluded_columns:
                continue

            series = dataframe[column]

            non_null_count = int(
                series.notna().sum()
            )

            unique_count = int(
                series.nunique(
                    dropna=True
                )
            )

            value_counts = (
                series.dropna()
                .astype(str)
                .value_counts()
                .head(5)
            )

            top_values = [
                {
                    "value": str(value),
                    "count": int(count),
                }
                for value, count
                in value_counts.items()
            ]

            profile[column_name] = {
                "non_null_count": non_null_count,
                "unique_count": unique_count,
                "top_values": top_values,
            }

        return profile

    def _datetime_profile(
        self,
        dataframe: pd.DataFrame,
        datetime_columns: list[str],
    ) -> dict[str, Any]:
        """
        Calculate date ranges for detected datetime columns.
        """

        profile: dict[str, Any] = {}

        for column in datetime_columns:
            converted_series = pd.to_datetime(
                dataframe[column],
                errors="coerce",
            ).dropna()

            if converted_series.empty:
                continue

            profile[column] = {
                "count": int(
                    converted_series.count()
                ),
                "earliest": (
                    converted_series.min()
                    .isoformat()
                ),
                "latest": (
                    converted_series.max()
                    .isoformat()
                ),
            }

        return profile

    def _ranked_rows(
        self,
        dataframe: pd.DataFrame,
        numeric_columns: list[str],
    ) -> dict[str, Any]:
        """
        Extract top and bottom rows for numeric result columns.

        Ranking is calculated independently for every numeric column.
        """

        rankings: dict[str, Any] = {}

        for column in numeric_columns:
            ranked_frame = dataframe.copy()

            ranked_frame[column] = pd.to_numeric(
                ranked_frame[column],
                errors="coerce",
            )

            ranked_frame = ranked_frame.dropna(
                subset=[column]
            )

            if ranked_frame.empty:
                continue

            top_rows = (
                ranked_frame.nlargest(
                    min(
                        5,
                        len(ranked_frame),
                    ),
                    column,
                )
                .to_dict(
                    orient="records"
                )
            )

            bottom_rows = (
                ranked_frame.nsmallest(
                    min(
                        5,
                        len(ranked_frame),
                    ),
                    column,
                )
                .to_dict(
                    orient="records"
                )
            )

            rankings[column] = {
                "top_rows": [
                    {
                        key: self._safe_value(value)
                        for key, value
                        in row.items()
                    }
                    for row in top_rows
                ],
                "bottom_rows": [
                    {
                        key: self._safe_value(value)
                        for key, value
                        in row.items()
                    }
                    for row in bottom_rows
                ],
            }

        return rankings

    def _build_data_profile(
        self,
        query_result: QueryExecutionResult,
    ) -> dict[str, Any]:
        """
        Build deterministic evidence from all returned rows.
        """

        dataframe = self._records_to_dataframe(
            query_result
        )

        numeric_columns = (
            self._detect_numeric_columns(
                dataframe
            )
        )

        datetime_columns = (
            self._detect_datetime_columns(
                dataframe=dataframe,
                numeric_columns=numeric_columns,
            )
        )

        categorical_columns = [
            str(column)
            for column in dataframe.columns
            if str(column)
            not in set(
                numeric_columns
                + datetime_columns
            )
        ]

        missing_values = {
            str(column): int(
                dataframe[column]
                .isna()
                .sum()
            )
            for column in dataframe.columns
        }

        sample_frame = dataframe.head(
            self.maximum_sample_rows
        )

        sample_rows = [
            {
                str(key): self._safe_value(
                    value
                )
                for key, value
                in row.items()
            }
            for row in sample_frame.to_dict(
                orient="records"
            )
        ]

        return {
            "row_count": query_result.row_count,
            "column_count": len(
                query_result.columns
            ),
            "columns": list(
                query_result.columns
            ),
            "truncated": query_result.truncated,
            "execution_time_ms": (
                query_result.execution_time_ms
            ),
            "numeric_columns": numeric_columns,
            "categorical_columns": (
                categorical_columns
            ),
            "datetime_columns": datetime_columns,
            "missing_values": missing_values,
            "numeric_profile": (
                self._numeric_profile(
                    dataframe=dataframe,
                    numeric_columns=numeric_columns,
                )
            ),
            "categorical_profile": (
                self._categorical_profile(
                    dataframe=dataframe,
                    excluded_columns=set(
                        numeric_columns
                        + datetime_columns
                    ),
                )
            ),
            "datetime_profile": (
                self._datetime_profile(
                    dataframe=dataframe,
                    datetime_columns=datetime_columns,
                )
            ),
            "rankings": self._ranked_rows(
                dataframe=dataframe,
                numeric_columns=numeric_columns,
            ),
            "sample_rows": sample_rows,
            "sample_row_count": len(
                sample_rows
            ),
        }

    def _build_user_prompt(
        self,
        user_question: str,
        intent: IntentResult,
        plan: AnalysisPlan,
        query_result: QueryExecutionResult,
        data_profile: dict[str, Any],
    ) -> str:
        """
        Build the evidence-grounded analysis prompt.
        """

        profile_json = json.dumps(
            data_profile,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

        return f"""
Analyze the real query result for the following business question.

BUSINESS QUESTION:

{user_question}

VALIDATED INTENT:

{intent.model_dump_json(indent=2)}

ANALYTICAL PLAN:

{plan.model_dump_json(indent=2)}

EXECUTED SQL:

{query_result.query}

DETERMINISTIC DATA PROFILE:

{profile_json}

Create an AnalysisResult using only this evidence.

Requirements:

1. The summary must directly answer what the returned data shows.
2. Important findings must contain supporting values or observations.
3. Trends must be listed only when ordered time data supports them.
4. Anomalies must be listed only when the evidence supports them.
5. possible_drivers must be phrased as possible contributors.
6. Do not claim a proven root cause unless the result directly tests it.
7. Add limitations when the result is truncated, empty, aggregated,
   lacks comparison data, or cannot establish causation.
8. Do not provide final recommendations.
9. Return only JSON matching the AnalysisResult schema.
""".strip()

    def _call_model(
        self,
        prompt: str,
    ) -> str:
        """
        Call llama3.2 through Ollama.
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
                    AnalysisResult
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
                    "The Analysis Agent prompt exceeded "
                    "the configured context window.\n"
                    f"Context length: "
                    f"{self.context_length}\n"
                    "Reduce maximum_sample_rows or increase "
                    f"context_length.\nOriginal error: {exc}"
                ) from exc

            raise ConnectionError(
                "Could not call the Analysis Agent "
                "through Ollama.\n"
                f"Host: {self.host}\n"
                f"Model: {self.model}\n"
                f"Original error: {exc}"
            ) from exc

        content = response.message.content

        if content is None:
            raise RuntimeError(
                "The Analysis Agent returned no content."
            )

        cleaned_content = content.strip()

        if not cleaned_content:
            raise RuntimeError(
                "The Analysis Agent returned an "
                "empty response."
            )

        return cleaned_content

    def _validate_model_output(
        self,
        model_output: str,
    ) -> AnalysisResult:
        """
        Validate LLM output against AnalysisResult.
        """

        try:
            return (
                AnalysisResult
                .model_validate_json(
                    model_output
                )
            )

        except ValidationError as exc:
            raise RuntimeError(
                "The Analysis Agent returned output "
                "that did not match AnalysisResult.\n\n"
                f"Raw output:\n{model_output}\n\n"
                f"Validation error:\n{exc}"
            ) from exc

    def _empty_result_analysis(
        self,
        query_result: QueryExecutionResult,
    ) -> AnalysisResult:
        """
        Return a deterministic result when SQL returns no rows.

        Calling the LLM is unnecessary because no analytical evidence
        exists.
        """

        limitations = [
            (
                "The executed query returned no rows, "
                "so trends, anomalies, drivers, and "
                "root-cause indicators cannot be evaluated."
            )
        ]

        if query_result.truncated:
            limitations.append(
                "The query result was marked as truncated."
            )

        return AnalysisResult(
            summary=(
                "The query executed successfully but "
                "returned no records for the requested "
                "analysis."
            ),
            findings=[],
            trends=[],
            anomalies=[],
            possible_drivers=[],
            limitations=limitations,
        )

    def analyze(
        self,
        user_question: str,
        intent: IntentResult,
        plan: AnalysisPlan,
        query_result: QueryExecutionResult,
    ) -> AnalysisResult:
        """
        Analyze actual data returned by Agent 3b.

        Returns:
            Validated AnalysisResult.
        """

        if not isinstance(user_question, str):
            raise TypeError(
                "user_question must be a string."
            )

        if not isinstance(intent, IntentResult):
            raise TypeError(
                "intent must be an IntentResult."
            )

        if not isinstance(plan, AnalysisPlan):
            raise TypeError(
                "plan must be an AnalysisPlan."
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
                "Analysis cannot begin while "
                "clarification is required."
            )

        if query_result.row_count == 0:
            return self._empty_result_analysis(
                query_result
            )

        data_profile = self._build_data_profile(
            query_result
        )

        prompt = self._build_user_prompt(
            user_question=cleaned_question,
            intent=intent,
            plan=plan,
            query_result=query_result,
            data_profile=data_profile,
        )

        model_output = self._call_model(
            prompt
        )

        result = self._validate_model_output(
            model_output
        )

        if query_result.truncated:
            truncation_message = (
                "The database result exceeded the "
                "configured maximum-row limit, so the "
                "analysis uses only the returned rows."
            )

            if (
                truncation_message
                not in result.limitations
            ):
                result.limitations.append(
                    truncation_message
                )

        return result


def main() -> None:
    """
    Run Agent 4 using the current LangGraph output through Agent 3.

    This smoke test uses actual Northwind query results.
    """

    from workflow.graph import run_workflow

    print("=" * 80)
    print("AGENT 4: ANALYSIS SMOKE TEST")
    print("=" * 80)

    question = (
        "Show the top 5 products by revenue."
    )

    try:
        state = run_workflow(
            user_question=question,
            max_sql_retries=3,
            recursion_limit=25,
        )

        if state.get(
            "workflow_status"
        ) == "failed":
            raise RuntimeError(
                state.get(
                    "error_message",
                    "Workflow through Agent 3 failed.",
                )
            )

        intent = state.get("intent")
        plan = state.get("plan")
        query_result = state.get(
            "query_result"
        )

        if intent is None:
            raise RuntimeError(
                "IntentResult is missing."
            )

        if plan is None:
            raise RuntimeError(
                "AnalysisPlan is missing."
            )

        if query_result is None:
            raise RuntimeError(
                "QueryExecutionResult is missing."
            )

        agent = AnalysisAgent(
            model="llama3.2:latest",
            host="http://localhost:11434",
            context_length=8192,
            max_output_tokens=1500,
            maximum_sample_rows=50,
        )

        result = agent.analyze(
            user_question=question,
            intent=intent,
            plan=plan,
            query_result=query_result,
        )

        print("\nAnalysisResult:")
        print(
            result.model_dump_json(
                indent=2
            )
        )

        print("\n" + "=" * 80)
        print("RESULT: AGENT 4 IS WORKING")
        print("=" * 80)

    except Exception as exc:
        print("\n" + "=" * 80)
        print("RESULT: AGENT 4 FAILED")
        print("=" * 80)
        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()