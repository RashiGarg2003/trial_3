"""
Intent Understanding Agent.

This agent converts a natural-language business question into a
validated IntentResult.

Model:
    llama3.2:latest running locally through Ollama

Responsibilities:
    1. Understand the user's analytical question.
    2. Extract metrics, dimensions, filters, and time requirements.
    3. Identify the requested analysis type.
    4. Determine whether clarification is necessary.
    5. Return schema-validated structured output.

This agent does not:
    - Inspect the database schema.
    - Generate SQL.
    - Execute SQL.
    - Calculate analytical results.
    - Apply hardcoded business rules.
"""

from ollama import Client
from pydantic import ValidationError

from models.schemas import (
    IntentResult,
)


class IntentAgent:
    """
    Model-driven Intent Understanding Agent.

    The agent uses Ollama structured output and validates the response
    using the shared IntentResult Pydantic model.
    """

    def __init__(
        self,
        model: str = "llama3.2:latest",
        host: str = "http://localhost:11434",
    ) -> None:
        """
        Initialize the Intent Agent.

        Args:
            model:
                Name of the Ollama model used for intent extraction.

            host:
                Address of the local Ollama API.
        """

        if not model.strip():
            raise ValueError(
                "The Ollama model name cannot be empty."
            )

        if not host.strip():
            raise ValueError(
                "The Ollama host cannot be empty."
            )

        self.model = model.strip()
        self.host = host.rstrip("/")
        self.client = Client(host=self.host)

    def _build_system_prompt(self) -> str:
        """
        Build the model instructions for intent extraction.

        The instructions define semantic reasoning and clarification rules
        without hardcoding database tables, columns, schemas, or business
        mappings.
        """

        return """
        You are the Intent Understanding Agent in a multi-agent business
        data analytics system.

        Your only responsibility is to understand the user's analytical
        question and return structured analytical intent.

        Determine the intent from the meaning of the complete question.

        Extract:

        1. The original question.
        2. A clearer analytical version of the question.
        3. The primary analysis type.
        4. Requested business metrics.
        5. Requested dimensions or grouping concepts.
        6. Explicit filter conditions.
        7. Time-period requirements.
        8. Sorting requirements, when relevant.
        9. Result limits, when explicitly requested.
        10. Whether clarification is required.
        11. One focused clarification question when required.

        ANALYSIS-TYPE DEFINITIONS

        aggregation:
        Calculate totals, counts, averages, minimums, maximums, or other
        summarized measures.

        comparison:
        Compare two or more periods, categories, groups, or segments.

        trend_analysis:
        Analyze how a measure changes across time.

        ranking:
        Order entities using a metric and identify leading or trailing
        results.

        root_cause_analysis:
        Investigate possible reasons or contributors behind an observed
        outcome. Do not claim that causation is proven.

        anomaly_detection:
        Identify unusual observations, spikes, drops, or outliers.

        distribution_analysis:
        Analyze how values are distributed across a population, category,
        or numeric range.

        general_analysis:
        Use only when the request is meaningful and actionable but does not
        fit a more specific analysis type.

        Do not use general_analysis merely to avoid asking for clarification.

        ACTIONABILITY POLICY

        A request is actionable only when enough information is available to
        define a meaningful database analysis.

        A request normally contains actionable analytical context when at
        least one of the following is clearly identifiable:

        - A business metric, such as a total, count, average, revenue,
        quantity, price, cost, order count, or another requested measure.
        - A dimension or entity to analyze, group, compare, list, or rank.
        - An explicit filter condition.
        - A meaningful time period.
        - A comparison target.
        - A ranking objective with a measurable target.
        - A specific analytical calculation.
        - A clearly defined anomaly, distribution, or root-cause target.

        A result limit or sort direction alone is not sufficient if the user
        has not identified what should be analyzed.

        CLARIFICATION POLICY

        Set clarification_required to true when the request does not provide
        enough information to define a meaningful analytical operation.

        Requests such as the following require clarification:

        - Analyze the data.
        - Show performance.
        - Give me insights.
        - Compare the results.
        - Show the best ones.
        - Why did it decrease?
        - Find the problem.
        - Tell me what happened.

        These requests are vague because the requested metric, entity,
        dimension, comparison, filter, or time period is unknown.

        For an underspecified request:

        - Set clarification_required to true.
        - Keep metrics empty when no metric is known.
        - Keep dimensions empty when no dimension is known.
        - Do not invent filters.
        - Do not invent dates or time periods.
        - Do not assume that the user means revenue, sales, orders, products,
        customers, profit, cost, or any other business concept.
        - Do not generate a speculative rewritten question.
        - Ask exactly one focused clarification question.
        - Ask for the minimum information required to continue.
        - Do not proceed using general_analysis as a fallback.

        A suitable generic clarification question is:

        "What metric would you like to analyze, and how should the results be
        grouped, filtered, or compared?"

        When some context is available, make the clarification question more
        specific.

        Example:

        User question:
        "Show the top five."

        Correct clarification question:
        "Which entity should be ranked, and which metric should be used to
        identify the top five?"

        Example:

        User question:
        "Why did revenue decrease?"

        Correct clarification question:
        "Which time period, region, product, or category should be examined
        for the revenue decrease?"

        Do not ask for clarification when the request is already actionable.

        Examples that normally do not require clarification:

        - Show total revenue by product category.
        - Show the top five products by revenue.
        - Count orders by country.
        - Show the monthly order trend.
        - List discontinued products.
        - Show orders placed in 1997.
        - Compare revenue between 1996 and 1997.

        GENERAL EXTRACTION RULES

        Use semantic understanding of the complete question.

        Do not generate SQL.

        Do not answer the business question.

        Do not calculate results.

        Do not invent database tables or columns.

        Do not assume a specific database schema.

        Do not invent metrics, dimensions, filters, dates, or business
        measures.

        Preserve explicit filter values from the question.

        Express each filter using:

        - field
        - operator
        - value

        Use lowercase, concise, business-level names for metrics and
        dimensions.

        Keep metrics and dimensions separate:

        - A metric is a value being calculated or measured.
        - A dimension is an entity or category used to group, compare, list,
        or rank results.

        Extract a limit only when the user explicitly requests a specific
        number.

        Extract a sort order only when ordering is relevant.

        Determine sorting direction from the ranking objective:

        - Leading, largest, greatest, highest, top, or best-performing values
        require descending order.
        - Trailing, smallest, least, lowest, bottom, or worst-performing values
        require ascending order.

        Confirm that the selected sorting direction would return the records
        requested by the user.

        Use ISO date format when exact dates can be represented reliably.

        Otherwise, preserve the user's business-period expression without
        inventing exact dates.

        ROOT-CAUSE REQUESTS

        A root-cause request must identify an observed outcome to investigate.

        For example:

        - "Why did revenue decrease in 1997?" is actionable.
        - "Why did it decrease?" requires clarification.

        Represent root-cause requests as investigations of possible drivers.
        Do not state or imply that the available data will prove causality.

        OUTPUT CONSISTENCY RULES

        When clarification_required is false:

        - clarification_question must be null.
        - The structured intent must contain sufficient analytical context.
        - Do not return empty metrics, empty dimensions, empty filters, no time
        period, no ranking target, and no other analytical information at
        the same time.

        When clarification_required is true:

        - clarification_question must contain exactly one focused question.
        - Do not populate missing fields with assumptions.
        - The rewritten question must describe the unresolved analytical need
        without pretending that the missing information is known.

        Before returning the result, verify:

        1. Is the requested analytical outcome clear?
        2. Is at least one meaningful metric, dimension, filter, time period,
        comparison, ranking target, or calculation identifiable?
        3. Would a planning agent know what analysis to plan without inventing
        business requirements?
        4. Would a SQL-generation agent know what kind of result is expected
        without guessing the user's objective?

        If the answer to questions 1, 3, or 4 is no, set
        clarification_required to true.

        Return only JSON matching the supplied schema.
        """.strip()

    def _build_user_prompt(
        self,
        user_question: str,
    ) -> str:
        """
        Build the user message sent to the model.
        """

        return f"""
Analyze the following business question and return its structured
analytical intent.

Business question:
{user_question}

Return only the structured result matching the supplied JSON schema.
Do not include Markdown, commentary, SQL, or an analytical answer.
""".strip()

    def _call_model(
        self,
        user_question: str,
    ) -> str:
        """
        Call the local Ollama model and return its raw response.

        Args:
            user_question:
                Cleaned business question.

        Returns:
            Raw JSON text produced by the model.
        """

        try:
            response = self.client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._build_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": self._build_user_prompt(
                            user_question
                        ),
                    },
                ],
                format=IntentResult.model_json_schema(),
                stream=False,
                options={
                    "temperature": 0,
                    "seed": 42,
                },
            )

        except Exception as exc:
            raise ConnectionError(
                "Could not communicate with Ollama.\n"
                f"Host: {self.host}\n"
                f"Model: {self.model}\n"
                "Make sure Ollama is running and the model is "
                f"installed.\nOriginal error: {exc}"
            ) from exc

        raw_content = response.message.content

        if raw_content is None:
            raise RuntimeError(
                "The Intent Agent returned no content."
            )

        cleaned_content = raw_content.strip()

        if not cleaned_content:
            raise RuntimeError(
                "The Intent Agent returned an empty response."
            )

        return cleaned_content

    def _validate_response(
        self,
        raw_content: str,
    ) -> IntentResult:
        """
        Validate the model response using IntentResult.

        Invalid output is rejected instead of being silently modified.
        """

        try:
            return IntentResult.model_validate_json(
                raw_content
            )

        except ValidationError as exc:
            raise RuntimeError(
                "The Intent Agent returned output that did not match "
                "the required IntentResult schema.\n\n"
                f"Raw model output:\n{raw_content}\n\n"
                f"Validation error:\n{exc}"
            ) from exc

    def _enforce_clarification_policy(
        self,
        result: IntentResult,
    ) -> IntentResult:
        """
        Enforce a general, schema-driven clarification policy.

        A request cannot proceed to planning and SQL generation unless
        IntentResult contains enough information to define an analytical
        operation.

        This does not use hardcoded business keywords. It validates the
        structured intent produced by the model.
        """

        if not isinstance(
            result,
            IntentResult,
        ):
            raise TypeError(
                "result must be an IntentResult instance."
            )

        metrics = [
            str(metric).strip()
            for metric in result.metrics
            if str(metric).strip()
        ]

        dimensions = [
            str(dimension).strip()
            for dimension in result.dimensions
            if str(dimension).strip()
        ]

        has_metric = bool(metrics)
        has_dimension = bool(dimensions)
        has_meaningful_filter = any(
            isinstance(filter_condition, object)
            and bool(str(filter_condition.field).strip())
            and bool(str(filter_condition.operator).strip())
            and filter_condition.value not in (None, "", [], {})
            for filter_condition in result.filters
        )
        has_actionable_context = (
            has_metric
            or has_dimension
            or has_meaningful_filter
        )

        if has_actionable_context:
            return result.model_copy(
                update={
                    "clarification_required": False,
                    "clarification_question": None,
                    "metrics": metrics,
                    "dimensions": dimensions,
                }
            )

        clarification_question = (
            result.clarification_question
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
                "and how should the results be grouped or filtered?"
            )

        return result.model_copy(
            update={
                "clarification_required": True,
                "clarification_question": (
                    clarification_question
                ),
                "metrics": metrics,
                "dimensions": dimensions,
            }
        )

    def understand(
        self,
        user_question: str,
    ) -> IntentResult:
        """
        Convert a natural-language analytical question into a
        validated IntentResult.

        Args:
            user_question:
                Business question entered by the user.

        Returns:
            Validated IntentResult.

        Raises:
            TypeError:
                If user_question is not a string.

            ValueError:
                If user_question is empty.

            ConnectionError:
                If Ollama cannot be reached.

            RuntimeError:
                If the model returns empty or invalid structured output.
        """

        if not isinstance(user_question, str):
            raise TypeError(
                "user_question must be a string."
            )

        cleaned_question = user_question.strip()

        if not cleaned_question:
            raise ValueError(
                "The user question cannot be empty."
            )

        raw_content = self._call_model(
            user_question=cleaned_question
        )

        result = self._validate_response(
            raw_content=raw_content
        )

        return self._enforce_clarification_policy(
            result=result
        )


def main() -> None:
    """
    Run a simple standalone Intent Agent smoke test.

    A separate test file performs broader validation.
    """

    print("=" * 80)
    print("INTENT AGENT SMOKE TEST")
    print("=" * 80)

    question = (
        "Show the top 5 products by revenue in 2025."
    )

    print(f"Model    : llama3.2:latest")
    print(f"Question : {question}")
    print("\nCalling local Ollama model...")

    try:
        agent = IntentAgent(
            model="llama3.2:latest",
            host="http://localhost:11434",
        )

        result = agent.understand(question)

        print("\nStructured Intent:")
        print(result.model_dump_json(indent=2))

        print("\n" + "=" * 80)
        print("RESULT: INTENT AGENT IS WORKING")
        print("=" * 80)

    except Exception as exc:
        print("\n" + "=" * 80)
        print("RESULT: INTENT AGENT FAILED")
        print("=" * 80)
        print(f"{type(exc).__name__}: {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()