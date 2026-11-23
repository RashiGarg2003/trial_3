"""
Standalone tests for the Planning Agent.

The tests verify that the Planning Agent:

1. Accepts a validated IntentResult.
2. Returns a validated AnalysisPlan.
3. Creates unique and ordered steps.
4. Creates valid step dependencies.
5. Preserves the intended analysis objective.
6. Does not generate SQL.
7. Does not mark new steps as completed.
8. Rejects invalid input.
9. Rejects intents that require clarification.

Run from the project root:

    python -m tests.test_planning_agent
"""

import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agents.planning_agent import PlanningAgent
from models.schemas import (
    AnalysisPlan,
    FilterCondition,
    IntentResult,
    TimePeriod,
)


TEST_CASES: list[dict[str, Any]] = [
    {
        "name": "Ranking plan",
        "intent": IntentResult(
            original_question=(
                "Show the top 5 products by revenue in 2025."
            ),
            rewritten_question=(
                "Rank the five products with the highest "
                "revenue during 2025."
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
        ),
        "required_step_types": [
            "data_retrieval",
            "ranking",
        ],
        "required_metrics": [
            "revenue",
        ],
        "required_dimensions": [
            "product",
        ],
    },
    {
        "name": "Trend analysis plan",
        "intent": IntentResult(
            original_question=(
                "Show the monthly sales trend for 2025."
            ),
            rewritten_question=(
                "Analyze monthly sales changes during 2025."
            ),
            analysis_type="trend_analysis",
            metrics=["sales"],
            dimensions=["month"],
            filters=[],
            time_period=TimePeriod(
                start_date="2025-01-01",
                end_date="2025-12-31",
                granularity="month",
            ),
            sort_order=None,
            limit=None,
            clarification_required=False,
            clarification_question=None,
        ),
        "required_step_types": [
            "data_retrieval",
            "trend_analysis",
        ],
        "required_metrics": [
            "sales",
        ],
        "required_dimensions": [
            "month",
        ],
    },
    {
        "name": "Comparison plan",
        "intent": IntentResult(
            original_question=(
                "Compare total revenue between the North "
                "and South regions."
            ),
            rewritten_question=(
                "Compare total revenue for the North and "
                "South regions."
            ),
            analysis_type="comparison",
            metrics=["revenue"],
            dimensions=["region"],
            filters=[
                FilterCondition(
                    field="region",
                    operator="in",
                    value=["North", "South"],
                )
            ],
            time_period=TimePeriod(),
            sort_order=None,
            limit=None,
            clarification_required=False,
            clarification_question=None,
        ),
        "required_step_types": [
            "data_retrieval",
            "comparison",
        ],
        "required_metrics": [
            "revenue",
        ],
        "required_dimensions": [
            "region",
        ],
    },
    {
        "name": "Root-cause analysis plan",
        "intent": IntentResult(
            original_question=(
                "Why did sales decrease in the North region "
                "during the last quarter?"
            ),
            rewritten_question=(
                "Investigate possible contributors to the sales "
                "decrease in the North region during the last quarter."
            ),
            analysis_type="root_cause_analysis",
            metrics=["sales"],
            dimensions=["region"],
            filters=[
                FilterCondition(
                    field="region",
                    operator="equals",
                    value="North",
                )
            ],
            time_period=TimePeriod(
                current_period="last quarter",
                granularity="quarter",
            ),
            sort_order=None,
            limit=None,
            clarification_required=False,
            clarification_question=None,
        ),
        "required_step_types": [
            "data_retrieval",
            "root_cause_analysis",
        ],
        "required_metrics": [
            "sales",
        ],
        "required_dimensions": [],
    },
    {
        "name": "Anomaly detection plan",
        "intent": IntentResult(
            original_question=(
                "Identify unusual spikes or drops in monthly "
                "order revenue."
            ),
            rewritten_question=(
                "Detect unusual monthly changes in order revenue."
            ),
            analysis_type="anomaly_detection",
            metrics=["revenue"],
            dimensions=["month"],
            filters=[],
            time_period=TimePeriod(
                granularity="month",
            ),
            sort_order=None,
            limit=None,
            clarification_required=False,
            clarification_question=None,
        ),
        "required_step_types": [
            "data_retrieval",
            "anomaly_detection",
        ],
        "required_metrics": [
            "revenue",
        ],
        "required_dimensions": [
            "month",
        ],
    },
]


def normalize_text(value: str) -> str:
    """
    Normalize text for semantic comparisons.
    """

    return " ".join(
        value.lower()
        .replace("_", " ")
        .replace("-", " ")
        .strip()
        .split()
    )


def concept_is_present(
    expected_concept: str,
    actual_values: list[str],
) -> bool:
    """
    Check whether a required concept appears in a list.

    Additional model-generated concepts are allowed.
    """

    expected = normalize_text(expected_concept)
    expected_words = set(expected.split())

    for actual_value in actual_values:
        actual = normalize_text(actual_value)
        actual_words = set(actual.split())

        if actual == expected:
            return True

        if expected in actual:
            return True

        if actual in expected:
            return True

        if (
            expected_words
            and expected_words.issubset(actual_words)
        ):
            return True

    return False


def collect_plan_metrics(
    plan: AnalysisPlan,
) -> list[str]:
    """
    Collect metrics from every plan step.
    """

    metrics: list[str] = []

    for step in plan.steps:
        metrics.extend(step.required_metrics)

    return metrics


def collect_plan_dimensions(
    plan: AnalysisPlan,
) -> list[str]:
    """
    Collect dimensions from every plan step.
    """

    dimensions: list[str] = []

    for step in plan.steps:
        dimensions.extend(step.required_dimensions)

    return dimensions


def contains_sql(plan: AnalysisPlan) -> bool:
    """
    Detect whether the Planning Agent generated SQL text.

    The Planning Agent may indicate that a step requires SQL, but it
    must not generate the SQL statement itself.
    """

    serialized_plan = json.dumps(
        plan.model_dump(mode="json"),
        ensure_ascii=False,
    ).lower()

    sql_patterns = [
        "select ",
        "insert into",
        "update ",
        "delete from",
        "drop table",
        "create table",
        "alter table",
        "group by",
        "order by",
    ]

    return any(
        pattern in serialized_plan
        for pattern in sql_patterns
    )


def validate_step_ids(
    plan: AnalysisPlan,
) -> list[str]:
    """
    Validate uniqueness and ordering of plan step IDs.
    """

    errors: list[str] = []

    step_ids = [
        step.step_id
        for step in plan.steps
    ]

    expected_step_ids = list(
        range(
            1,
            len(plan.steps) + 1,
        )
    )

    if step_ids != expected_step_ids:
        errors.append(
            "Step IDs should be sequential and start from 1. "
            f"Expected {expected_step_ids}, received {step_ids}."
        )

    if len(step_ids) != len(set(step_ids)):
        errors.append(
            "Plan step IDs must be unique."
        )

    return errors


def validate_dependencies(
    plan: AnalysisPlan,
) -> list[str]:
    """
    Confirm that every dependency references an earlier step.
    """

    errors: list[str] = []

    existing_step_ids = {
        step.step_id
        for step in plan.steps
    }

    for step in plan.steps:
        for dependency in step.depends_on:
            if dependency not in existing_step_ids:
                errors.append(
                    f"Step {step.step_id} references unknown "
                    f"dependency {dependency}."
                )

            if dependency >= step.step_id:
                errors.append(
                    f"Step {step.step_id} depends on step "
                    f"{dependency}, which is not an earlier step."
                )

    return errors

def validate_step_statuses(
    plan: AnalysisPlan,
) -> list[str]:
    """
    Confirm that newly generated plan steps are pending.
    """

    errors: list[str] = []

    for step in plan.steps:
        if step.status != "pending":
            errors.append(
                f"Step {step.step_id} should have status "
                f"'pending', but received '{step.status}'."
            )

    return errors


def validate_required_step_types(
    plan: AnalysisPlan,
    required_step_types: list[str],
) -> list[str]:
    """
    Confirm that the plan includes necessary analytical operations.
    """

    errors: list[str] = []

    actual_step_types = [
        step.step_type
        for step in plan.steps
    ]

    for required_step_type in required_step_types:
        if required_step_type not in actual_step_types:
            errors.append(
                f"Plan should contain step type "
                f"'{required_step_type}', but received "
                f"{actual_step_types}."
            )

    return errors


def validate_required_concepts(
    field_name: str,
    expected_values: list[str],
    actual_values: list[str],
) -> list:
    """
    Validate metrics and dimensions required by the intent.
    """

    errors: list[str] = []

    for expected_value in expected_values:
        if not concept_is_present(
            expected_concept=expected_value,
            actual_values=actual_values,
        ):
            errors.append(
                f"Plan {field_name} should contain "
                f"'{expected_value}', but received "
                f"{actual_values}."
            )

    return errors


def validate_plan(
    plan: AnalysisPlan,
    test_case: dict[str, Any],
) -> list:
    """
    Validate one Planning Agent result.
    """

    errors: list[str] = []

    if not isinstance(plan, AnalysisPlan):
        return [
            "The Planning Agent did not return an AnalysisPlan."
        ]

    if not plan.objective.strip():
        errors.append(
            "The plan objective cannot be empty."
        )

    if not plan.strategy.strip():
        errors.append(
            "The plan strategy cannot be empty."
        )

    if not plan.steps:
        errors.append(
            "The analytical plan must contain at least one step."
        )

        return errors

    errors.extend(
        validate_step_ids(plan)
    )

    errors.extend(
        validate_dependencies(plan)
    )

    errors.extend(
        validate_step_statuses(plan)
    )

    errors.extend(
        validate_required_step_types(
            plan=plan,
            required_step_types=test_case[
                "required_step_types"
            ],
        )
    )

    errors.extend(
        validate_required_concepts(
            field_name="metrics",
            expected_values=test_case[
                "required_metrics"
            ],
            actual_values=collect_plan_metrics(plan),
        )
    )

    errors.extend(
        validate_required_concepts(
            field_name="dimensions",
            expected_values=test_case[
                "required_dimensions"
            ],
            actual_values=collect_plan_dimensions(plan),
        )
    )

    if not any(
        step.requires_sql
        for step in plan.steps
    ):
        errors.append(
            "The plan should contain at least one step "
            "that requires database retrieval."
        )

    if contains_sql(plan):
        errors.append(
            "The Planning Agent generated SQL text. "
            "SQL generation belongs to the SQL Agent."
        )

    return errors


def print_plan(
    plan: AnalysisPlan,
) -> None:
    """
    Print a plan in readable JSON and summary formats.
    """

    print("\nStructured AnalysisPlan:")
    print(
        plan.model_dump_json(
            indent=2
        )
    )

    print("\nPlan steps:")

    for step in plan.steps:
        dependencies = (
            ", ".join(
                str(value)
                for value in step.depends_on
            )
            if step.depends_on
            else "None"
        )

        print(
            f"- Step {step.step_id}: {step.title}"
        )
        print(
            f"  Type         : {step.step_type}"
        )
        print(
            f"  Depends on   : {dependencies}"
        )
        print(
            f"  Requires SQL : {step.requires_sql}"
        )
        print(
            f"  Status       : {step.status}"
        )


def run_model_test(
    agent: PlanningAgent,
    test_case: dict[str, Any],
    test_number: int,
) -> bool:
    """
    Run one model-based Planning Agent test.
    """

    print("\n" + "=" * 80)
    print(
        f"TEST {test_number}: "
        f"{test_case['name']}"
    )
    print("=" * 80)

    intent: IntentResult = test_case["intent"]

    print(
        f"Question      : "
        f"{intent.original_question}"
    )
    print(
        f"Analysis type : "
        f"{intent.analysis_type}"
    )
    print("\nCalling llama3.2:latest...")

    try:
        plan = agent.create_plan(
            intent=intent
        )

    except Exception as exc:
        print("\n[FAILED]")
        print(
            f"{type(exc).__name__}: {exc}"
        )
        return False

    print_plan(plan)

    errors = validate_plan(
        plan=plan,
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


def test_invalid_input(
    agent: PlanningAgent,
) -> bool:
    """
    Confirm that non-IntentResult input is rejected.
    """

    try:
        agent.create_plan(
            intent={
                "analysis_type": "ranking"
            }
        )  # type: ignore[arg-type]

    except TypeError as exc:
        print(
            f"Expected type error: {exc}"
        )
        return True

    raise AssertionError(
        "PlanningAgent accepted invalid input."
    )


def test_clarification_required(
    agent: PlanningAgent,
) -> bool:
    """
    Confirm that an unresolved intent is not planned.
    """

    unresolved_intent = IntentResult(
        original_question="Show the results.",
        rewritten_question=(
            "Determine which results the user wants."
        ),
        analysis_type="general_analysis",
        metrics=[],
        dimensions=[],
        filters=[],
        time_period=TimePeriod(),
        sort_order=None,
        limit=None,
        clarification_required=True,
        clarification_question=(
            "Which metric and grouping should be analyzed?"
        ),
    )

    try:
        agent.create_plan(
            intent=unresolved_intent
        )

    except ValueError as exc:
        print(
            f"Expected clarification error: {exc}"
        )
        return True

    raise AssertionError(
        "PlanningAgent created a plan for an "
        "intent that requires clarification."
    )


def run_local_test(
    agent: PlanningAgent,
    test_number: int,
    test_name: str,
    test_function: Any,
) -> bool:
    """
    Run one local validation test.
    """

    print("\n" + "=" * 80)
    print(
        f"TEST {test_number}: {test_name}"
    )
    print("=" * 80)

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
    Run the full Planning Agent test suite.
    """

    print("\nPLANNING AGENT TEST SUITE")
    print("Model : llama3.2:latest")
    print("Host  : http://localhost:11434")

    try:
        agent = PlanningAgent(
            model="llama3.2:latest",
            host="http://localhost:11434",
        )

    except Exception as exc:
        print("\nCould not initialize PlanningAgent.")
        print(
            f"{type(exc).__name__}: {exc}"
        )
        raise SystemExit(1) from exc

    passed_tests = 0
    current_test_number = 1

    for test_case in TEST_CASES:
        passed = run_model_test(
            agent=agent,
            test_case=test_case,
            test_number=current_test_number,
        )

        if passed:
            passed_tests += 1

        current_test_number += 1

    local_tests = [
        (
            "Invalid input rejection",
            test_invalid_input,
        ),
        (
            "Clarification-required rejection",
            test_clarification_required,
        ),
    ]

    for test_name, test_function in local_tests:
        passed = run_local_test(
            agent=agent,
            test_number=current_test_number,
            test_name=test_name,
            test_function=test_function,
        )

        if passed:
            passed_tests += 1

        current_test_number += 1

    total_tests = (
        len(TEST_CASES)
        + len(local_tests)
    )

    failed_tests = total_tests - passed_tests

    print("\n" + "=" * 80)
    print("FINAL PLANNING AGENT TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests : {total_tests}")
    print(f"Passed      : {passed_tests}")
    print(f"Failed      : {failed_tests}")

    if failed_tests == 0:
        print(
            "\nRESULT: ALL PLANNING AGENT TESTS PASSED"
        )
        print(
            "The Planning Agent is ready for "
            "SQL Agent integration."
        )
        raise SystemExit(0)

    print(
        "\nRESULT: SOME PLANNING AGENT TESTS FAILED"
    )
    print(
        "Review the failed plan and validation details "
        "before building the SQL Agent."
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()