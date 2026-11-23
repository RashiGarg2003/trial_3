"""
Benchmark runner for the Multi-Agent Data Analytics System.

The runner validates the complete workflow against predefined examples.

It compares properties rather than exact LLM wording:

- Final workflow status
- Input-security decision
- Intent type
- Required metrics
- Required dimensions
- SQL read-only safety
- SQL structure
- Required tables
- Forbidden tables
- LIMIT
- Sort direction
- Chart type
- Required report sections
- Forbidden sensitive-data categories
- Output-validation decisions
- Workflow execution time

The runner does not:

- Compare exact generated summaries.
- Require exact SQL text.
- Modify the database.
- Bypass security controls.
- Treat expected blocked outcomes as failures.
"""

import json
import re
import time
from pathlib import Path
from typing import Any, Optional

from sqlglot import exp, parse_one
from sqlglot.errors import ParseError

from security.models import (
    BenchmarkExample,
    BenchmarkResult,
)
from security.pii_detector import PIIDetector
from tools.sql_validator import SQLValidator
from workflow.graph import run_workflow
from workflow.state import GraphState


EXAMPLES_PATH = (
    Path(__file__).resolve().parent
    / "examples.json"
)

RESULTS_PATH = (
    Path(__file__).resolve().parent
    / "benchmark_results.json"
)


def enum_value(
    value: Any,
) -> Any:
    """
    Return the primitive value stored by an Enum.
    """

    if value is None:
        return None

    primitive_value = getattr(
        value,
        "value",
        value,
    )

    if isinstance(
        primitive_value,
        str,
    ):
        return (
            primitive_value
            .strip()
            .lower()
        )

    return primitive_value


def normalize_text(
    value: Any,
) -> str:
    """
    Normalize a value for flexible semantic comparison.
    """

    if value is None:
        return ""

    return re.sub(
        r"[^a-z0-9]+",
        " ",
        str(value).lower(),
    ).strip()


def load_examples(
    path: Path = EXAMPLES_PATH,
) -> list[BenchmarkExample]:
    """
    Load and validate predefined benchmark examples.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark examples were not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError(
            "examples.json must contain a JSON array."
        )

    examples = [
        BenchmarkExample.model_validate(
            item
        )
        for item in payload
    ]

    if not examples:
        raise ValueError(
            "examples.json does not contain any examples."
        )

    test_ids = [
        example.test_id
        for example in examples
    ]

    if (
        len(test_ids)
        != len(set(test_ids))
    ):
        raise ValueError(
            "Every benchmark test_id must be unique."
        )

    return examples


def add_check(
    condition: bool,
    check_name: str,
    passed: list[str],
    failed: list[str],
    detail: Optional[str] = None,
) -> None:
    """
    Add one benchmark assertion result.
    """

    if condition:
        passed.append(
            check_name
        )

    else:
        failure_message = check_name

        if detail:
            failure_message += (
                f": {detail}"
            )

        failed.append(
            failure_message
        )


def extract_sql_metadata(
    sql: str,
) -> dict[str, Any]:
    """
    Extract structural SQL properties using SQLGlot.
    """

    try:
        expression = parse_one(
            sql,
            read="sqlite",
        )

    except ParseError as exc:
        return {
            "parse_valid": False,
            "parse_error": str(exc),
            "tables": [],
            "has_group_by": False,
            "has_order_by": False,
            "sort_direction": None,
            "limit": None,
            "operations": [],
        }

    tables = list(
        dict.fromkeys(
            table.name
            for table
            in expression.find_all(
                exp.Table
            )
            if table.name
        )
    )

    has_group_by = (
        expression.args.get(
            "group"
        )
        is not None
    )

    order_expression = (
        expression.args.get(
            "order"
        )
    )

    has_order_by = (
        order_expression
        is not None
    )

    sort_direction = None

    if order_expression is not None:
        ordered_items = list(
            order_expression.expressions
        )

        if ordered_items:
            first_order = (
                ordered_items[0]
            )

            if bool(
                first_order.args.get(
                    "desc"
                )
            ):
                sort_direction = (
                    "descending"
                )

            else:
                sort_direction = (
                    "ascending"
                )

    limit_value = None

    limit_expression = (
        expression.args.get(
            "limit"
        )
    )

    if limit_expression is not None:
        literal = (
            limit_expression.expression
        )

        if (
            isinstance(
                literal,
                exp.Literal,
            )
            and literal.is_int
        ):
            limit_value = int(
                literal.this
            )

    operations: list[str] = []

    operation_types = {
        "sum": exp.Sum,
        "count": exp.Count,
        "average": exp.Avg,
        "minimum": exp.Min,
        "maximum": exp.Max,
    }

    for operation_name, expression_type in (
        operation_types.items()
    ):
        if any(
            True
            for _ in expression.find_all(
                expression_type
            )
        ):
            operations.append(
                operation_name
            )

    return {
        "parse_valid": True,
        "parse_error": None,
        "tables": tables,
        "has_group_by": (
            has_group_by
        ),
        "has_order_by": (
            has_order_by
        ),
        "sort_direction": (
            sort_direction
        ),
        "limit": limit_value,
        "operations": operations,
    }


def contains_expected_value(
    actual_values: list[str],
    expected_value: str,
) -> bool:
    """
    Compare semantic labels without requiring exact formatting.

    Examples:

        actual "order_count" matches expected "order count"
        actual "trend_analysis" matches expected "trend"
    """

    normalized_expected = (
        normalize_text(
            expected_value
        )
    )

    for actual_value in actual_values:
        normalized_actual = (
            normalize_text(
                actual_value
            )
        )

        if (
            normalized_expected
            == normalized_actual
        ):
            return True

        if (
            normalized_expected
            in normalized_actual
            or normalized_actual
            in normalized_expected
        ):
            return True

    return False


def get_output_text(
    state: GraphState,
) -> str:
    """
    Collect released or generated output text for privacy checks.

    Raw original input is deliberately excluded.
    """

    values: list[str] = []

    final_report = state.get(
        "final_report"
    )

    if final_report:
        values.append(
            final_report.model_dump_json()
        )

    sanitized_payload = state.get(
        "sanitized_output_payload"
    )

    if isinstance(
        sanitized_payload,
        dict,
    ):
        values.append(
            json.dumps(
                sanitized_payload,
                ensure_ascii=False,
                default=str,
            )
        )

    return " ".join(values)


def check_expected_status(
    example: BenchmarkExample,
    state: GraphState,
    passed: list[str],
    failed: list[str],
) -> None:
    """
    Validate the final workflow status.
    """

    actual_status = state.get(
        "workflow_status"
    )

    add_check(
        condition=(
            actual_status
            == example.expected_final_status
        ),
        check_name="final_status",
        passed=passed,
        failed=failed,
        detail=(
            f"expected "
            f"{example.expected_final_status}, "
            f"received {actual_status}"
        ),
    )


def check_security_decision(
    example: BenchmarkExample,
    state: GraphState,
    passed: list[str],
    failed: list[str],
) -> None:
    """
    Validate the input-security decision.
    """

    expected_decision = enum_value(
        example.expected_security_decision
    )

    actual_decision = enum_value(
        state.get(
            "security_decision"
        )
    )

    add_check(
        condition=(
            actual_decision
            == expected_decision
        ),
        check_name="security_decision",
        passed=passed,
        failed=failed,
        detail=(
            f"expected {expected_decision}, "
            f"received {actual_decision}"
        ),
    )

    if (
        expected_decision
        == "allow_redacted"
    ):
        add_check(
            condition=bool(
                state.get(
                    "question_was_redacted",
                    False,
                )
            ),
            check_name=(
                "question_was_redacted"
            ),
            passed=passed,
            failed=failed,
            detail=(
                "Expected sensitive input "
                "to be redacted."
            ),
        )


def check_intent(
    example: BenchmarkExample,
    state: GraphState,
    passed: list[str],
    failed: list[str],
) -> None:
    """
    Validate structured intent properties.
    """

    if (
        not example.expected_analysis_types
        and not example.required_metrics
        and not example.required_dimensions
    ):
        return

    intent = state.get(
        "intent"
    )

    if intent is None:
        add_check(
            condition=False,
            check_name="intent_available",
            passed=passed,
            failed=failed,
            detail=(
                "IntentResult is missing."
            ),
        )

        return

    add_check(
        condition=True,
        check_name="intent_available",
        passed=passed,
        failed=failed,
    )

    if example.expected_analysis_types:
        actual_type = normalize_text(
            intent.analysis_type
        )

        expected_types = [
            normalize_text(
                expected_type
            )
            for expected_type
            in example
            .expected_analysis_types
        ]

        analysis_type_matches = any(
            (
                actual_type
                == expected_type
                or expected_type
                in actual_type
                or actual_type
                in expected_type
            )
            for expected_type
            in expected_types
        )

        add_check(
            condition=(
                analysis_type_matches
            ),
            check_name="analysis_type",
            passed=passed,
            failed=failed,
            detail=(
                f"expected one of "
                f"{example.expected_analysis_types}, "
                f"received {intent.analysis_type}"
            ),
        )

    actual_metrics = [
        str(metric)
        for metric in intent.metrics
    ]

    for required_metric in (
        example.required_metrics
    ):
        add_check(
            condition=(
                contains_expected_value(
                    actual_values=(
                        actual_metrics
                    ),
                    expected_value=(
                        required_metric
                    ),
                )
            ),
            check_name=(
                "required_metric:"
                f"{required_metric}"
            ),
            passed=passed,
            failed=failed,
            detail=(
                f"Actual metrics: "
                f"{actual_metrics}"
            ),
        )

    actual_dimensions = [
        str(dimension)
        for dimension
        in intent.dimensions
    ]

    for required_dimension in (
        example.required_dimensions
    ):
        add_check(
            condition=(
                contains_expected_value(
                    actual_values=(
                        actual_dimensions
                    ),
                    expected_value=(
                        required_dimension
                    ),
                )
            ),
            check_name=(
                "required_dimension:"
                f"{required_dimension}"
            ),
            passed=passed,
            failed=failed,
            detail=(
                f"Actual dimensions: "
                f"{actual_dimensions}"
            ),
        )


def check_sql(
    example: BenchmarkExample,
    state: GraphState,
    passed: list[str],
    failed: list[str],
) -> None:
    """
    Validate generated SQL properties.
    """

    requirements = (
        example.sql_requirements
    )

    if requirements is None:
        generated_sql = state.get(
            "generated_sql"
        )

        if (
            example.expected_final_status
            in {
                "security_blocked",
                "security_review_required",
                "clarification_required",
            }
        ):
            add_check(
                condition=(
                    generated_sql is None
                ),
                check_name=(
                    "sql_not_generated"
                ),
                passed=passed,
                failed=failed,
                detail=(
                    "SQL should not be generated "
                    "for this controlled outcome."
                ),
            )

        return

    generated_sql = state.get(
        "generated_sql"
    )

    if not isinstance(
        generated_sql,
        str,
    ) or not generated_sql.strip():
        add_check(
            condition=False,
            check_name="sql_available",
            passed=passed,
            failed=failed,
            detail=(
                "Generated SQL is missing."
            ),
        )

        return

    add_check(
        condition=True,
        check_name="sql_available",
        passed=passed,
        failed=failed,
    )

    validator = SQLValidator(
        dialect="sqlite"
    )

    validation_result = (
        validator.validate(
            generated_sql
        )
    )

    if requirements.read_only:
        add_check(
            condition=(
                validation_result.is_valid
            ),
            check_name="sql_read_only",
            passed=passed,
            failed=failed,
            detail=(
                "; ".join(
                    validation_result.errors
                )
                if validation_result.errors
                else None
            ),
        )

    metadata = extract_sql_metadata(
        generated_sql
    )

    add_check(
        condition=bool(
            metadata["parse_valid"]
        ),
        check_name="sql_parse_valid",
        passed=passed,
        failed=failed,
        detail=(
            metadata.get(
                "parse_error"
            )
        ),
    )

    if not metadata["parse_valid"]:
        return

    if requirements.requires_group_by:
        add_check(
            condition=bool(
                metadata[
                    "has_group_by"
                ]
            ),
            check_name=(
                "sql_requires_group_by"
            ),
            passed=passed,
            failed=failed,
        )

    if requirements.requires_order_by:
        add_check(
            condition=bool(
                metadata[
                    "has_order_by"
                ]
            ),
            check_name=(
                "sql_requires_order_by"
            ),
            passed=passed,
            failed=failed,
        )

    if (
        requirements.sort_direction
        is not None
    ):
        add_check(
            condition=(
                metadata[
                    "sort_direction"
                ]
                == requirements
                .sort_direction
            ),
            check_name=(
                "sql_sort_direction"
            ),
            passed=passed,
            failed=failed,
            detail=(
                f"expected "
                f"{requirements.sort_direction}, "
                f"received "
                f"{metadata['sort_direction']}"
            ),
        )

    if (
        requirements.maximum_rows
        is not None
        and metadata["limit"]
        is not None
    ):
        add_check(
            condition=(
                metadata["limit"]
                <= requirements.maximum_rows
            ),
            check_name=(
                "sql_result_limit"
            ),
            passed=passed,
            failed=failed,
            detail=(
                f"LIMIT {metadata['limit']} "
                f"exceeds "
                f"{requirements.maximum_rows}"
            ),
        )

    actual_tables = [
        normalize_text(table)
        for table
        in metadata["tables"]
    ]

    for required_table in (
        requirements.required_tables
    ):
        normalized_required = (
            normalize_text(
                required_table
            )
        )

        add_check(
            condition=(
                normalized_required
                in actual_tables
            ),
            check_name=(
                "required_table:"
                f"{required_table}"
            ),
            passed=passed,
            failed=failed,
            detail=(
                f"Actual tables: "
                f"{metadata['tables']}"
            ),
        )

    for forbidden_table in (
        requirements.forbidden_tables
    ):
        normalized_forbidden = (
            normalize_text(
                forbidden_table
            )
        )

        add_check(
            condition=(
                normalized_forbidden
                not in actual_tables
            ),
            check_name=(
                "forbidden_table:"
                f"{forbidden_table}"
            ),
            passed=passed,
            failed=failed,
        )

    actual_operations = set(
        metadata["operations"]
    )

    for required_operation in (
        requirements.required_operations
    ):
        normalized_operation = (
            normalize_text(
                required_operation
            )
        )

        add_check(
            condition=(
                normalized_operation
                in actual_operations
            ),
            check_name=(
                "sql_operation:"
                f"{required_operation}"
            ),
            passed=passed,
            failed=failed,
            detail=(
                f"Actual operations: "
                f"{sorted(actual_operations)}"
            ),
        )


def check_report(
    example: BenchmarkExample,
    state: GraphState,
    passed: list[str],
    failed: list[str],
) -> None:
    """
    Validate chart and report properties.
    """

    if (
        not example.expected_chart_types
        and not example
        .required_report_sections
    ):
        return

    final_report = state.get(
        "final_report"
    )

    if final_report is None:
        add_check(
            condition=False,
            check_name="final_report_available",
            passed=passed,
            failed=failed,
            detail=(
                "FinalReport is missing."
            ),
        )

        return

    add_check(
        condition=True,
        check_name="final_report_available",
        passed=passed,
        failed=failed,
    )

    report_payload = (
        final_report.model_dump(
            mode="json"
        )
    )

    for required_section in (
        example.required_report_sections
    ):
        section_value = (
            report_payload.get(
                required_section
            )
        )

        has_content = bool(
            section_value
        )

        add_check(
            condition=has_content,
            check_name=(
                "report_section:"
                f"{required_section}"
            ),
            passed=passed,
            failed=failed,
        )

    if example.expected_chart_types:
        chart = final_report.chart

        actual_chart_type = (
            enum_value(
                chart.chart_type
            )
            if chart is not None
            else "none"
        )

        add_check(
            condition=(
                actual_chart_type
                in example
                .expected_chart_types
            ),
            check_name="chart_type",
            passed=passed,
            failed=failed,
            detail=(
                f"expected one of "
                f"{example.expected_chart_types}, "
                f"received {actual_chart_type}"
            ),
        )


def check_privacy(
    example: BenchmarkExample,
    state: GraphState,
    passed: list[str],
    failed: list[str],
) -> None:
    """
    Verify forbidden sensitive categories are not released.
    """

    if not example.forbidden_data_types:
        return

    output_text = get_output_text(
        state
    )

    if not output_text.strip():
        # Blocked requests intentionally have no released output.
        add_check(
            condition=True,
            check_name=(
                "no_sensitive_output_released"
            ),
            passed=passed,
            failed=failed,
        )

        return

    detector = PIIDetector()

    detection_result = (
        detector.detect(
            output_text
        )
    )

    detected_types = {
        finding.entity_type
        for finding
        in detection_result.findings
    }

    for forbidden_type in (
        example.forbidden_data_types
    ):
        add_check(
            condition=(
                forbidden_type
                not in detected_types
            ),
            check_name=(
                "forbidden_data:"
                f"{forbidden_type}"
            ),
            passed=passed,
            failed=failed,
            detail=(
                f"Detected types: "
                f"{sorted(detected_types)}"
            ),
        )


def check_output_validation(
    example: BenchmarkExample,
    state: GraphState,
    passed: list[str],
    failed: list[str],
) -> None:
    """
    Validate the output-quality gates for completed workflows.
    """

    if (
        example.expected_final_status
        != "completed"
    ):
        return

    deterministic_result = state.get(
        "deterministic_validation_result"
    )

    output_evaluation = state.get(
        "output_evaluation"
    )

    output_guard_result = state.get(
        "output_guard_result"
    )

    add_check(
        condition=(
            deterministic_result
            is not None
        ),
        check_name=(
            "deterministic_result_available"
        ),
        passed=passed,
        failed=failed,
    )

    if deterministic_result is not None:
        add_check(
            condition=(
                enum_value(
                    deterministic_result
                    .decision
                )
                == "approve"
            ),
            check_name=(
                "deterministic_approved"
            ),
            passed=passed,
            failed=failed,
            detail=(
                f"Decision: "
                f"{enum_value(deterministic_result.decision)}"
            ),
        )

    add_check(
        condition=(
            output_evaluation
            is not None
        ),
        check_name=(
            "output_evaluation_available"
        ),
        passed=passed,
        failed=failed,
    )

    if output_evaluation is not None:
        add_check(
            condition=(
                enum_value(
                    output_evaluation
                    .decision
                )
                == "approve"
            ),
            check_name=(
                "llm_evaluation_approved"
            ),
            passed=passed,
            failed=failed,
            detail=(
                f"Decision: "
                f"{enum_value(output_evaluation.decision)}"
            ),
        )

    add_check(
        condition=(
            output_guard_result
            is not None
        ),
        check_name=(
            "output_guard_available"
        ),
        passed=passed,
        failed=failed,
    )

    if output_guard_result is not None:
        guard_decision = enum_value(
            output_guard_result.decision
        )

        add_check(
            condition=(
                guard_decision
                in {
                    "allow",
                    "allow_redacted",
                }
            ),
            check_name=(
                "output_guard_approved"
            ),
            passed=passed,
            failed=failed,
            detail=(
                f"Decision: "
                f"{guard_decision}"
            ),
        )

        add_check(
            condition=(
                output_guard_result
                .sanitized_payload
                is not None
            ),
            check_name=(
                "sanitized_payload_available"
            ),
            passed=passed,
            failed=failed,
        )


def run_example(
    example: BenchmarkExample,
) -> BenchmarkResult:
    """
    Execute and evaluate one benchmark example.
    """

    checks_passed: list[str] = []
    checks_failed: list[str] = []

    started_at = time.perf_counter()

    state: GraphState

    try:
        state = run_workflow(
            user_question=(
                example.question
            ),
            max_sql_retries=3,
            max_report_regenerations=1,
            recursion_limit=50,
        )

    except Exception as exc:
        elapsed_ms = (
            time.perf_counter()
            - started_at
        ) * 1000.0

        return BenchmarkResult(
            test_id=example.test_id,
            passed=False,
            final_status="exception",
            checks_passed=[],
            checks_failed=[
                (
                    f"workflow_exception: "
                    f"{type(exc).__name__}: "
                    f"{str(exc)[:300]}"
                )
            ],
            execution_time_ms=(
                elapsed_ms
            ),
            security_decision=None,
            deterministic_evaluation=None,
            llm_evaluation=None,
        )

    elapsed_ms = (
        time.perf_counter()
        - started_at
    ) * 1000.0

    check_expected_status(
        example=example,
        state=state,
        passed=checks_passed,
        failed=checks_failed,
    )

    check_security_decision(
        example=example,
        state=state,
        passed=checks_passed,
        failed=checks_failed,
    )

    check_intent(
        example=example,
        state=state,
        passed=checks_passed,
        failed=checks_failed,
    )

    check_sql(
        example=example,
        state=state,
        passed=checks_passed,
        failed=checks_failed,
    )

    check_report(
        example=example,
        state=state,
        passed=checks_passed,
        failed=checks_failed,
    )

    check_privacy(
        example=example,
        state=state,
        passed=checks_passed,
        failed=checks_failed,
    )

    check_output_validation(
        example=example,
        state=state,
        passed=checks_passed,
        failed=checks_failed,
    )

    return BenchmarkResult(
        test_id=example.test_id,
        passed=(
            len(checks_failed) == 0
        ),
        final_status=str(
            state.get(
                "workflow_status",
                "unknown",
            )
        ),
        checks_passed=(
            checks_passed
        ),
        checks_failed=(
            checks_failed
        ),
        execution_time_ms=(
            elapsed_ms
        ),
        security_decision=(
            state.get(
                "security_decision"
            )
        ),
        deterministic_evaluation=(
            state.get(
                "deterministic_validation_result"
            )
        ),
        llm_evaluation=(
            state.get(
                "output_evaluation"
            )
        ),
    )


def save_results(
    results: list[BenchmarkResult],
    path: Path = RESULTS_PATH,
) -> None:
    """
    Save benchmark results without raw questions or sensitive values.
    """

    payload = [
        result.model_dump(
            mode="json"
        )
        for result in results
    ]

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


def print_summary(
    results: list[BenchmarkResult],
) -> None:
    """
    Print the benchmark summary.
    """

    total = len(results)

    passed = sum(
        1
        for result in results
        if result.passed
    )

    failed = total - passed

    average_latency_ms = (
        sum(
            result.execution_time_ms
            for result in results
        )
        / total
        if total
        else 0.0
    )

    security_cases = [
        result
        for result in results
        if result.test_id.startswith(
            "block_"
        )
    ]

    security_passed = sum(
        1
        for result in security_cases
        if result.passed
    )

    security_pass_rate = (
        security_passed
        / len(security_cases)
        * 100.0
        if security_cases
        else 0.0
    )

    safe_cases = [
        result
        for result in results
        if result.test_id.startswith(
            "safe_"
        )
    ]

    safe_passed = sum(
        1
        for result in safe_cases
        if result.passed
    )

    safe_pass_rate = (
        safe_passed
        / len(safe_cases)
        * 100.0
        if safe_cases
        else 0.0
    )

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    print(
        f"Total cases              : {total}"
    )

    print(
        f"Passed                   : {passed}"
    )

    print(
        f"Failed                   : {failed}"
    )

    print(
        f"Overall pass rate        : "
        f"{(passed / total * 100.0) if total else 0.0:.2f}%"
    )

    print(
        f"Security pass rate       : "
        f"{security_pass_rate:.2f}%"
    )

    print(
        f"Safe-query pass rate     : "
        f"{safe_pass_rate:.2f}%"
    )

    print(
        f"Average latency          : "
        f"{average_latency_ms:.2f} ms"
    )

    if failed:
        print("\nFailed test IDs:")

        for result in results:
            if result.passed:
                continue

            print(
                f"\n- {result.test_id}"
            )

            print(
                f"  Final status: "
                f"{result.final_status}"
            )

            for check in (
                result.checks_failed
            ):
                print(
                    f"  • {check}"
                )

    print("\nResults file:")

    print(
        RESULTS_PATH
    )

    print("\n" + "=" * 80)

    if failed == 0:
        print(
            "RESULT: ALL BENCHMARKS PASSED"
        )

    else:
        print(
            "RESULT: BENCHMARKS REQUIRE REVIEW"
        )

    print("=" * 80)


def main() -> None:
    """
    Run all predefined benchmark examples.

    Ollama, required models, SQLite, and the complete LangGraph
    workflow must be available.
    """

    print("=" * 80)
    print(
        "MULTI-AGENT ANALYTICS "
        "BENCHMARK RUNNER"
    )
    print("=" * 80)

    examples = load_examples()

    print(
        f"Loaded {len(examples)} "
        "benchmark examples."
    )

    results: list[
        BenchmarkResult
    ] = []

    for index, example in enumerate(
        examples,
        start=1,
    ):
        print("\n" + "-" * 80)

        print(
            f"[{index}/{len(examples)}] "
            f"{example.test_id}"
        )

        print("-" * 80)

        result = run_example(
            example
        )

        results.append(
            result
        )

        print(
            f"Status       : "
            f"{result.final_status}"
        )

        print(
            f"Passed       : "
            f"{result.passed}"
        )

        print(
            f"Checks passed: "
            f"{len(result.checks_passed)}"
        )

        print(
            f"Checks failed: "
            f"{len(result.checks_failed)}"
        )

        print(
            f"Latency      : "
            f"{result.execution_time_ms:.2f} ms"
        )

        for failed_check in (
            result.checks_failed
        ):
            print(
                f"- {failed_check}"
            )

    save_results(
        results
    )

    print_summary(
        results
    )

    if not all(
        result.passed
        for result in results
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()