"""
Standalone tests for the Intent Understanding Agent.

These tests verify:

1. Ollama connectivity.
2. Structured output generation.
3. Shared IntentResult schema compatibility.
4. Semantic extraction across different analytical questions.
5. Input validation.
6. No SQL generation inside the Intent Agent.

Run from the project root:

    python -m tests.test_intent_agent
"""

import json
import sys
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from agents.intent_agent import IntentAgent
from models.schemas import IntentResult


TEST_CASES: list[dict[str, Any]] = [
    {
        "name": "Ranking request",
        "question": (
            "Show the top 5 products by revenue in 2025."
        ),
        "allowed_analysis_types": [
            "ranking",
        ],
        "required_metrics": [
            "revenue",
        ],
        "required_dimensions": [
            "product",
        ],
        "expected_limit": 5,
        "validate_limit": True,
        "expected_sort_order": "descending",
        "validate_sort_order": True,
    },
    {
        "name": "Trend analysis request",
        "question": (
            "Show the monthly sales trend for 2025."
        ),
        "allowed_analysis_types": [
            "trend_analysis",
        ],
        "required_metrics": [
            "sales",
        ],
        "required_dimensions": [
            "month",
        ],
        "validate_limit": False,
        "validate_sort_order": False,
    },
    {
        "name": "Comparison request",
        "question": (
            "Compare total revenue between the North "
            "and South regions."
        ),
        "allowed_analysis_types": [
            "comparison",
        ],
        "required_metrics": [
            "revenue",
        ],
        "required_dimensions": [
            "region",
        ],
        "validate_limit": False,
        "validate_sort_order": False,
    },
    {
        "name": "Root-cause analysis request",
        "question": (
            "Why did sales decrease in the North region "
            "during the last quarter?"
        ),
        "allowed_analysis_types": [
            "root_cause_analysis",
        ],
        "required_metrics": [
            "sales",
        ],
        "required_dimensions": [],
        "validate_limit": False,
        "validate_sort_order": False,
    },
    {
        "name": "Anomaly detection request",
        "question": (
            "Identify unusual spikes or drops in monthly "
            "order revenue."
        ),
        "allowed_analysis_types": [
            "anomaly_detection",
        ],
        "required_metrics": [
            "revenue",
        ],
        "required_dimensions": [
            "month",
        ],
        "validate_limit": False,
        "validate_sort_order": False,
    },
    {
        "name": "Aggregation request",
        "question": (
            "What is the total number of orders by country?"
        ),
        "allowed_analysis_types": [
            "aggregation",
        ],
        "required_metrics": [
            "order count",
        ],
        "required_dimensions": [
            "country",
        ],
        "validate_limit": False,
        "validate_sort_order": False,
    },
]


def normalize_text(value: str) -> str:
    """
    Normalize model terminology for semantic comparison.
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
    Check whether an expected concept appears in model output.

    Small wording variations are accepted, for example:

        revenue
        total revenue

        order count
        number of orders
    """

    expected = normalize_text(expected_concept)
    expected_words = set(expected.split())

    semantic_aliases: dict[str, set[str]] = {
        "order count": {
            "order count",
            "number of orders",
            "orders count",
            "total orders",
            "count of orders",
        },
    }

    accepted_values = semantic_aliases.get(
        expected,
        {expected},
    )

    for actual_value in actual_values:
        actual = normalize_text(actual_value)
        actual_words = set(actual.split())

        if actual in accepted_values:
            return True

        if expected == actual:
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


def validate_required_concepts(
    field_name: str,
    expected_values: list[str],
    actual_values: list[str],
) -> list[str]:
    """
    Validate required metrics or dimensions.

    Additional valid values are allowed.
    """

    errors: list[str] = []

    for expected_value in expected_values:
        if not concept_is_present(
            expected_concept=expected_value,
            actual_values=actual_values,
        ):
            errors.append(
                f"{field_name} should contain "
                f"'{expected_value}', but received "
                f"{actual_values}"
            )

    return errors


def contains_sql(result: IntentResult) -> bool:
    """
    Check whether the Intent Agent incorrectly generated SQL.
    """

    serialized_result = json.dumps(
        result.model_dump(mode="json"),
        ensure_ascii=False,
    ).lower()

    sql_patterns = [
        "select ",
        "insert ",
        "update ",
        "delete ",
        "drop table",
        "create table",
        "alter table",
    ]

    return any(
        pattern in serialized_result
        for pattern in sql_patterns
    )


def validate_agent_result(
    result: IntentResult,
    test_case: dict[str, Any],
) -> list[str]:
    """
    Validate one structured Intent Agent response.
    """

    errors: list[str] = []

    if not isinstance(result, IntentResult):
        errors.append(
            "The agent did not return an IntentResult."
        )

        return errors

    allowed_analysis_types = test_case[
        "allowed_analysis_types"
    ]

    if result.analysis_type not in allowed_analysis_types:
        errors.append(
            "analysis_type should be one of "
            f"{allowed_analysis_types}, but received "
            f"'{result.analysis_type}'"
        )

    errors.extend(
        validate_required_concepts(
            field_name="metrics",
            expected_values=test_case[
                "required_metrics"
            ],
            actual_values=result.metrics,
        )
    )

    errors.extend(
        validate_required_concepts(
            field_name="dimensions",
            expected_values=test_case[
                "required_dimensions"
            ],
            actual_values=result.dimensions,
        )
    )

    if test_case.get("validate_limit", False):
        expected_limit = test_case["expected_limit"]

        if result.limit != expected_limit:
            errors.append(
                f"limit should be {expected_limit!r}, "
                f"but received {result.limit!r}"
            )

    if test_case.get(
        "validate_sort_order",
        False,
    ):
        expected_sort_order = test_case[
            "expected_sort_order"
        ]

        if result.sort_order != expected_sort_order:
            errors.append(
                "sort_order should be "
                f"{expected_sort_order!r}, "
                f"but received {result.sort_order!r}"
            )
            

    if result.clarification_required:
        errors.append(
            "The complete test question should not "
            "require clarification."
        )

    if contains_sql(result):
        errors.append(
            "The Intent Agent generated SQL even though "
            "SQL generation is outside its responsibility."
        )

    return errors


def print_result(result: IntentResult) -> None:
    """
    Print the structured result in readable JSON.
    """

    print(
        result.model_dump_json(
            indent=2,
        )
    )


def run_semantic_test(
    agent: IntentAgent,
    test_case: dict[str, Any],
    test_number: int,
) -> bool:
    """
    Run one semantic Intent Agent test.
    """

    print("\n" + "=" * 80)
    print(
        f"TEST {test_number}: "
        f"{test_case['name']}"
    )
    print("=" * 80)
    print(
        f"Question: {test_case['question']}"
    )
    print("\nCalling llama3.2:latest...")

    try:
        result = agent.understand(
            test_case["question"]
        )

    except Exception as exc:
        print("\n[FAILED]")
        print(
            f"{type(exc).__name__}: {exc}"
        )
        return False

    print("\nStructured output:")
    print_result(result)

    errors = validate_agent_result(
        result=result,
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


def test_empty_question(
    agent: IntentAgent,
) -> bool:
    """
    Confirm that empty questions are rejected locally.
    """

    try:
        agent.understand("   ")

    except ValueError as exc:
        print(
            f"Expected validation error: {exc}"
        )
        return True

    raise AssertionError(
        "The Intent Agent accepted an empty question."
    )


def test_non_string_question(
    agent: IntentAgent,
) -> bool:
    """
    Confirm that non-string input is rejected locally.
    """

    try:
        agent.understand(123)  # type: ignore[arg-type]

    except TypeError as exc:
        print(
            f"Expected type error: {exc}"
        )
        return True

    raise AssertionError(
        "The Intent Agent accepted non-string input."
    )


def run_local_validation_test(
    agent: IntentAgent,
    test_number: int,
    test_name: str,
    test_function: Callable[
        [IntentAgent],
        bool,
    ],
) -> bool:
    """
    Run one local input-validation test.
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
    Run the full Intent Agent test suite.
    """

    print("\nINTENT AGENT TEST SUITE")
    print("Model : llama3.2:latest")
    print("Host  : http://localhost:11434")

    try:
        agent = IntentAgent(
            model="llama3.2:latest",
            host="http://localhost:11434",
        )

    except Exception as exc:
        print("\nCould not initialize IntentAgent.")
        print(
            f"{type(exc).__name__}: {exc}"
        )
        raise SystemExit(1) from exc

    passed_tests = 0
    current_test_number = 1

    for test_case in TEST_CASES:
        passed = run_semantic_test(
            agent=agent,
            test_case=test_case,
            test_number=current_test_number,
        )

        if passed:
            passed_tests += 1

        current_test_number += 1

    local_tests: list[
        tuple[
            str,
            Callable[[IntentAgent], bool],
        ]
    ] = [
        (
            "Empty question validation",
            test_empty_question,
        ),
        (
            "Non-string question validation",
            test_non_string_question,
        ),
    ]

    for test_name, test_function in local_tests:
        passed = run_local_validation_test(
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
    print("FINAL INTENT AGENT TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests : {total_tests}")
    print(f"Passed      : {passed_tests}")
    print(f"Failed      : {failed_tests}")

    if failed_tests == 0:
        print(
            "\nRESULT: ALL INTENT AGENT TESTS PASSED"
        )
        print(
            "The Intent Agent is ready for "
            "Planning Agent integration."
        )
        raise SystemExit(0)

    print(
        "\nRESULT: SOME INTENT AGENT TESTS FAILED"
    )
    print(
        "Review the model output and validation errors "
        "before creating the Planning Agent."
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()