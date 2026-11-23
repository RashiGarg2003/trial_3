"""
Planning Agent for the Multi-Agent Data Analytics System.

The Planning Agent converts a validated IntentResult into an ordered
and executable AnalysisPlan.

Model:
    llama3.2:latest running locally through Ollama

Approach:
    1. The model creates the analytical plan.
    2. Pydantic validates the structure.
    3. Generic normalization fills missing intent context.
    4. Generic dependency normalization connects sequential steps.
    5. Final consistency validation confirms plan completeness.

The normalization is schema-driven. It does not contain hardcoded
metrics, dimensions, database tables, columns, or business rules.
"""

from ollama import Client
from pydantic import ValidationError

from models.schemas import (
    AnalysisPlan,
    IntentResult,
    PlanStep,
)


class PlanningAgent:
    """
    Creates an analytical execution plan from validated intent.

    The Planning Agent does not inspect database tables and does not
    generate or execute SQL.
    """

    def __init__(
        self,
        model: str = "llama3.2:latest",
        host: str = "http://localhost:11434",
    ) -> None:
        """
        Initialize the Planning Agent.

        Args:
            model:
                Local Ollama model used for planning.

            host:
                Ollama API address.
        """

        if not isinstance(model, str):
            raise TypeError(
                "The Ollama model name must be a string."
            )

        if not isinstance(host, str):
            raise TypeError(
                "The Ollama host must be a string."
            )

        cleaned_model = model.strip()
        cleaned_host = host.strip()

        if not cleaned_model:
            raise ValueError(
                "The Ollama model name cannot be empty."
            )

        if not cleaned_host:
            raise ValueError(
                "The Ollama host cannot be empty."
            )

        self.model = cleaned_model
        self.host = cleaned_host.rstrip("/")
        self.client = Client(host=self.host)

    def _build_system_prompt(self) -> str:
        """
        Build the Planning Agent system instructions.
        """

        return """
You are the Planning Agent in a multi-agent business data analytics
system.

You receive a validated analytical intent.

Create a concise and ordered analytical execution plan that later
agents can follow.

Your plan must contain:

1. One clear analytical objective.
2. One clear strategy.
3. Ordered analytical steps.
4. A step type for every step.
5. Metrics relevant to each step.
6. Dimensions relevant to each step.
7. Conceptual data requirements.
8. The expected output of every step.
9. Whether each step requires SQL.
10. Final output requirements.

Planning boundaries:

- Use only the supplied validated intent.
- Do not inspect or assume a database schema.
- Do not invent physical table names.
- Do not invent physical column names.
- Do not generate SQL.
- Do not execute SQL.
- Do not calculate results.
- Do not claim that findings have already been discovered.
- Do not provide final recommendations.
- Do not include programming or user-interface implementation details.
- Keep every step atomic and understandable.
- Use sequential positive step IDs beginning with 1.
- Set every new step status to "pending".
- Return only JSON matching the supplied schema.

Use required_metrics for business measures used by a step.

Use required_dimensions for grouping, ranking, comparison, or time
dimensions used by a step.

Use required_data for conceptual information needed by a step. These
must not be assumed physical database table or column names.

Use requires_sql=true when a step needs database retrieval, filtering,
joining, grouping, aggregation, comparison, or ranking.

Use requires_sql=false when a step interprets retrieved results,
prepares visualization guidance, or prepares reporting requirements.
""".strip()

    def _build_user_prompt(
        self,
        intent: IntentResult,
    ) -> str:
        """
        Build the model prompt from validated intent.
        """

        intent_json = intent.model_dump_json(
            indent=2
        )

        return f"""
Create an analytical execution plan for the following validated intent.

Validated intent:

{intent_json}

Create only the steps necessary to answer this request.

Use the supplied metrics and dimensions in the relevant plan steps.

Return the complete AnalysisPlan as JSON matching the supplied schema.
Do not generate SQL.
Do not invent database tables or columns.
""".strip()

    def _call_model(
        self,
        intent: IntentResult,
    ) -> str:
        """
        Call the local Ollama model and return raw JSON.
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
                            intent=intent
                        ),
                    },
                ],
                format=AnalysisPlan.model_json_schema(),
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
                f"Original error: {exc}"
            ) from exc

        raw_content = response.message.content

        if raw_content is None:
            raise RuntimeError(
                "The Planning Agent returned no content."
            )

        cleaned_content = raw_content.strip()

        if not cleaned_content:
            raise RuntimeError(
                "The Planning Agent returned an empty response."
            )

        return cleaned_content

    def _validate_model_response(
        self,
        raw_content: str,
    ) -> AnalysisPlan:
        """
        Validate raw model output against AnalysisPlan.
        """

        try:
            return AnalysisPlan.model_validate_json(
                raw_content
            )

        except ValidationError as exc:
            raise RuntimeError(
                "The Planning Agent returned output that did not "
                "match the required AnalysisPlan schema.\n\n"
                f"Raw model output:\n{raw_content}\n\n"
                f"Validation error:\n{exc}"
            ) from exc

    def _merge_unique_values(
        self,
        existing_values: list[str],
        inherited_values: list[str],
    ) -> list[str]:
        """
        Merge two text lists without duplicate normalized values.

        Values are inherited dynamically from IntentResult.
        """

        merged_values: list[str] = []
        normalized_values: set[str] = set()

        for values in [
            existing_values,
            inherited_values,
        ]:
            for value in values:
                if not isinstance(value, str):
                    continue

                cleaned_value = value.strip()

                if not cleaned_value:
                    continue

                normalized_value = cleaned_value.lower()

                if normalized_value in normalized_values:
                    continue

                normalized_values.add(normalized_value)
                merged_values.append(cleaned_value)

        return merged_values

    def _step_uses_analytical_context(
        self,
        step: PlanStep,
    ) -> bool:
        """
        Determine whether a step uses analytical intent context.

        This decision is based on step properties,not business
        keywords, metric names, or database entities.
        """

        if step.requires_sql:
            return True

        analytical_step_types = {
            "data_retrieval",
            "aggregation",
            "comparison",
            "trend_analysis",
            "ranking",
            "root_cause_analysis",
            "anomaly_detection",
            "statistical_analysis",
            "general_analysis",
        }

        return step.step_type in analytical_step_types

    def _normalize_steps(
        self,
        intent: IntentResult,
        plan: AnalysisPlan,
    ) -> list[PlanStep]:
        """
        Normalize all model-generated plan steps.

        Generic normalization rules:

        1. Rebuild sequential step IDs.
        2. Set every step status to pending.
        3. Preserve and inherit metrics from validated intent.
        4. Preserve and inherit dimensions from validated intent.
        5. Repair missing or invalid dependencies.
        6. Ensure the primary validated analysis type appears in
           the plan.

        No metric, dimension, table, column, or business term is
        hardcoded.
        """

        normalized_steps: list[PlanStep] = []

        for position, original_step in enumerate(
            plan.steps,
            start=1,
        ):
            step_data = original_step.model_dump()

            step_data["step_id"] = position
            step_data["status"] = "pending"

            step_data["required_metrics"] = (
                self._merge_unique_values(
                    existing_values=(
                        original_step.required_metrics
                    ),
                    inherited_values=intent.metrics,
                )
            )

            step_data["required_dimensions"] = (
                self._merge_unique_values(
                    existing_values=(
                        original_step.required_dimensions
                    ),
                    inherited_values=intent.dimensions,
                )
            )

            if position == 1:
                step_data["depends_on"] = []

            else:
                valid_dependencies = sorted(
                    {
                        dependency
                        for dependency
                        in original_step.depends_on
                        if 1 <= dependency < position
                    }
                )

                if valid_dependencies:
                    step_data["depends_on"] = (
                        valid_dependencies
                    )
                else:
                    step_data["depends_on"] = [
                        position - 1
                    ]

            normalized_step = PlanStep.model_validate(
                step_data
            )

            normalized_steps.append(
                normalized_step
            )

        normalized_steps = (
            self._ensure_primary_analysis_step(
                intent=intent,
                steps=normalized_steps,
            )
        )

        return normalized_steps

    def _ensure_primary_analysis_step(
        self,
        intent: IntentResult,
        steps: list[PlanStep],
    ) -> list[PlanStep]:
        """
        Ensure that the plan contains the primary analysis type
        identified by the Intent Agent.

        Local models sometimes describe the correct analytical action
        but assign a generic step type such as data_retrieval or
        aggregation.

        This method uses intent.analysis_type as the validated source
        of truth. It does not inspect titles using keywords and does
        not contain fixed business logic.
        """

        if not steps:
            return steps

        existing_step_types = {
            step.step_type
            for step in steps
        }

        primary_analysis_type = (
            intent.analysis_type
        )

        if primary_analysis_type in existing_step_types:
            return steps

        eligible_step_indexes = [
            index
            for index, step in enumerate(steps)
            if step.requires_sql
        ]

        if eligible_step_indexes:
            target_index = eligible_step_indexes[0]
        else:
            target_index = len(steps) - 1

        target_step = steps[target_index]

        updated_step_data = (
            target_step.model_dump()
        )

        updated_step_data["step_type"] = (
            primary_analysis_type
        )

        updated_step = PlanStep.model_validate(
            updated_step_data
        )

        updated_steps = list(steps)
        updated_steps[target_index] = updated_step

        return updated_steps
    
    def _normalize_plan(
        self,
        intent: IntentResult,
        plan: AnalysisPlan,
    ) -> AnalysisPlan:
        """
        Create a normalized AnalysisPlan.

        Model-generated objective, strategy, step descriptions, step
        types, expected outputs, and final requirements are preserved.
        """

        normalized_steps = self._normalize_steps(
            intent=intent,
            plan=plan,
        )

        normalized_plan = AnalysisPlan(
            objective=plan.objective,
            strategy=plan.strategy,
            steps=normalized_steps,
            final_output_requirements=(
                plan.final_output_requirements
            ),
        )

        return normalized_plan

    def _find_consistency_errors(
        self,
        intent: IntentResult,
        plan: AnalysisPlan,
    ) -> list:
        """
        Validate consistency between intent and normalized plan.

        This validation reads values dynamically from IntentResult.
        It contains no fixed metric or dimension names.
        """

        errors: list[str] = []

        plan_metrics = {
            metric.strip().lower()
            for step in plan.steps
            for metric in step.required_metrics
            if metric.strip()
        }

        plan_dimensions = {
            dimension.strip().lower()
            for step in plan.steps
            for dimension in step.required_dimensions
            if dimension.strip()
        }

        expected_metrics = {
            metric.strip().lower()
            for metric in intent.metrics
            if metric.strip()
        }

        expected_dimensions = {
            dimension.strip().lower()
            for dimension in intent.dimensions
            if dimension.strip()
        }

        missing_metrics = (
            expected_metrics - plan_metrics
        )

        missing_dimensions = (
            expected_dimensions - plan_dimensions
        )

        if missing_metrics:
            errors.append(
                "The plan is missing validated intent metrics: "
                f"{sorted(missing_metrics)}"
            )

        if missing_dimensions:
            errors.append(
                "The plan is missing validated intent dimensions: "
                f"{sorted(missing_dimensions)}"
            )

        expected_step_ids = list(
            range(
                1,
                len(plan.steps) + 1,
            )
        )

        actual_step_ids = [
            step.step_id
            for step in plan.steps
        ]

        if actual_step_ids != expected_step_ids:
            errors.append(
                "Plan step IDs are not sequential. "
                f"Expected {expected_step_ids}, "
                f"received {actual_step_ids}."
            )

        for step in plan.steps:
            if step.status != "pending":
                errors.append(
                    f"Step {step.step_id} does not have "
                    "pending status."
                )

            for dependency in step.depends_on:
                if dependency >= step.step_id:
                    errors.append(
                        f"Step {step.step_id} has invalid "
                        f"dependency {dependency}."
                    )

        return errors

    def create_plan(
        self,
        intent: IntentResult,
    ) -> AnalysisPlan:
        """
        Create a complete validated analytical plan.

        Args:
            intent:
                Validated Intent Agent output.

        Returns:
            Normalized and validated AnalysisPlan.
        """

        if not isinstance(intent, IntentResult):
            raise TypeError(
                "intent must be an IntentResult instance."
            )

        if intent.clarification_required:
            raise ValueError(
                "An analytical plan cannot be created while "
                "the intent requires clarification. "
                f"Clarification question: "
                f"{intent.clarification_question}"
            )

        raw_content = self._call_model(
            intent=intent
        )

        model_plan = self._validate_model_response(
            raw_content=raw_content
        )

        normalized_plan = self._normalize_plan(
            intent=intent,
            plan=model_plan,
        )

        consistency_errors = (
            self._find_consistency_errors(
                intent=intent,
                plan=normalized_plan,
            )
        )

        if consistency_errors:
            formatted_errors = "\n".join(
                f"- {error}"
                for error in consistency_errors
            )

            raise RuntimeError(
                "The Planning Agent produced an inconsistent "
                "analytical plan after normalization.\n"
                f"{formatted_errors}"
            )

        return normalized_plan


def build_smoke_test_intent() -> IntentResult:
    """
    Build sample validated intent for the smoke test.
    """

    return IntentResult(
        original_question=(
            "Show the top 5 products by revenue in 2025."
        ),
        rewritten_question=(
            "Rank the five products with the highest "
            "revenue during 2025."
        ),
        analysis_type="ranking",
        metrics=[
            "revenue",
        ],
        dimensions=[
            "product",
        ],
        filters=[],
        time_period={
            "start_date": "2025-01-01",
            "end_date": "2025-12-31",
            "current_period": None,
            "comparison_period": None,
            "granularity": "year",
        },
        sort_order="descending",
        limit=5,
        clarification_required=False,
        clarification_question=None,
    )


def main() -> None:
    """
    Run a standalone Planning Agent smoke test.
    """

    print("=" * 80)
    print("PLANNING AGENT SMOKE TEST")
    print("=" * 80)

    try:
        intent = build_smoke_test_intent()

        agent = PlanningAgent(
            model="llama3.2:latest",
            host="http://localhost:11434",
        )

        print(f"Model    : {agent.model}")
        print(
            f"Question : "
            f"{intent.original_question}"
        )
        print("\nCalling local Ollama model...")

        plan = agent.create_plan(
            intent=intent
        )

        print("\nNormalized AnalysisPlan:")
        print(
            plan.model_dump_json(
                indent=2
            )
        )

        print("\nPlan steps:")

        for step in plan.steps:
            dependencies = (
                ", ".join(
                    str(dependency)
                    for dependency in step.depends_on
                )
                if step.depends_on
                else "None"
            )

            print(
                f"- Step {step.step_id}: "
                f"{step.title}"
            )
            print(
                f"  Type         : "
                f"{step.step_type}"
            )
            print(
                f"  Metrics      : "
                f"{step.required_metrics}"
            )
            print(
                f"  Dimensions   : "
                f"{step.required_dimensions}"
            )
            print(
                f"  Depends on   : "
                f"{dependencies}"
            )
            print(
                f"  Requires SQL : "
                f"{step.requires_sql}"
            )
            print(
                f"  Status       : "
                f"{step.status}"
            )

        print("\n" + "=" * 80)
        print("RESULT: PLANNING AGENT IS WORKING")
        print("=" * 80)

    except Exception as exc:
        print("\n" + "=" * 80)
        print("RESULT: PLANNING AGENT FAILED")
        print("=" * 80)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()