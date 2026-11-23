"""
Tests for the FastAPI backend.

These tests validate:

1. Health endpoint.
2. Readiness endpoint.
3. Ollama model-status endpoint.
4. Completed analytics response.
5. Clarification response.
6. Controlled workflow-failure response.
7. Optional technical details.
8. Request validation.
9. SQL retry-limit enforcement.
10. Unexpected backend-error handling.

The tests use dependency overrides and do not call:

- Ollama
- Local LLM models
- The real SQLite database
- The real LangGraph workflow

Run from the project root:

    python -m tests.test_api

Or with pytest:

    python -m pytest tests/test_api.py -v
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from fastapi.testclient import TestClient

from api.dependencies import (
    AppServices,
)
from api.main import (
    app,
    get_application_services,
)
from models.schemas import (
    AnalysisFinding,
    AnalysisPlan,
    AnalysisResult,
    ChartSpecification,
    FinalReport,
    IntentResult,
    PlanStep,
    QueryExecutionResult,
    Recommendation,
    SQLGenerationResult,
    TimePeriod,
)
from workflow.state import (
    GraphState,
    create_initial_state,
)


# ---------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------

class FakeGraph:
    """
    Controlled graph double used by FastAPI endpoint tests.

    The next_state value is returned whenever invoke() is called.
    """

    def __init__(
        self,
        next_state: GraphState,
    ) -> None:
        self.next_state = next_state
        self.last_input_state: GraphState | None = None
        self.last_config: dict[str, Any] | None = None

    def invoke(
        self,
        initial_state: GraphState,
        config: dict[str, Any] | None = None,
    ) -> GraphState:
        """
        Return the configured workflow state.
        """

        self.last_input_state = initial_state
        self.last_config = config

        return self.next_state


class FailingGraph:
    """
    Graph double that raises an unexpected backend exception.
    """

    def invoke(
        self,
        initial_state: GraphState,
        config: dict[str, Any] | None = None,
    ) -> GraphState:
        """
        Simulate an unexpected LangGraph failure.
        """

        raise RuntimeError(
            "Simulated unexpected graph failure."
        )


class FakeOllamaManager:
    """
    Ollama manager double for readiness and model tests.
    """

    def __init__(
        self,
        running: bool = True,
        installed_models: list[str] | None = None,
    ) -> None:
        self.running = running
        self.installed_models = (
            installed_models
            if installed_models is not None
            else [
                "llama3.2:latest",
                "sqlcoder:latest",
            ]
        )

    def is_server_running(self) -> bool:
        """
        Return the configured API status.
        """

        return self.running

    def get_installed_models(self) -> list[str]:
        """
        Return the configured model list.
        """

        return list(
            self.installed_models
        )


class FakeDatabaseManager:
    """
    Database manager double for readiness tests.
    """

    def __init__(
        self,
        connection_status: bool = True,
        integrity_result: str = "ok",
    ) -> None:
        self.connection_status = (
            connection_status
        )

        self.integrity_result = (
            integrity_result
        )

    def test_connection(self) -> bool:
        """
        Return configured connection status.
        """

        return self.connection_status

    def run_integrity_check(self) -> str:
        """
        Return configured integrity result.
        """

        return self.integrity_result


# ---------------------------------------------------------------------
# Structured state factories
# ---------------------------------------------------------------------

def build_intent() -> IntentResult:
    """
    Build a valid IntentResult for API tests.
    """

    return IntentResult(
        original_question=(
            "Show the top 5 products by revenue."
        ),
        rewritten_question=(
            "Rank the five products with the "
            "highest revenue."
        ),
        analysis_type="ranking",
        metrics=[
            "revenue",
        ],
        dimensions=[
            "product",
        ],
        filters=[],
        time_period=TimePeriod(),
        sort_order="descending",
        limit=5,
        clarification_required=False,
        clarification_question=None,
    )


def build_plan() -> AnalysisPlan:
    """
    Build a valid AnalysisPlan for API tests.
    """

    return AnalysisPlan(
        objective=(
            "Identify the five products with "
            "the highest revenue."
        ),
        strategy=(
            "Retrieve product revenue, aggregate it, "
            "rank the values, and retain five rows."
        ),
        steps=[
            PlanStep(
                step_id=1,
                title="Retrieve product revenue",
                description=(
                    "Retrieve and aggregate revenue "
                    "for each product."
                ),
                step_type="data_retrieval",
                depends_on=[],
                required_metrics=[
                    "revenue",
                ],
                required_dimensions=[
                    "product",
                ],
                required_data=[
                    "product information",
                    "order transaction information",
                ],
                expected_output=(
                    "Revenue aggregated by product."
                ),
                requires_sql=True,
                status="pending",
            ),
            PlanStep(
                step_id=2,
                title="Rank products",
                description=(
                    "Rank products by revenue and "
                    "retain five rows."
                ),
                step_type="ranking",
                depends_on=[
                    1,
                ],
                required_metrics=[
                    "revenue",
                ],
                required_dimensions=[
                    "product",
                ],
                required_data=[],
                expected_output=(
                    "Five products ranked by revenue."
                ),
                requires_sql=True,
                status="pending",
            ),
        ],
        final_output_requirements=[
            "ProductName",
            "TotalRevenue",
        ],
    )


def build_sql_result() -> SQLGenerationResult:
    """
    Build a validated SQLGenerationResult.
    """

    return SQLGenerationResult(
        sql=(
            'SELECT p.ProductName, '
            'SUM(od.UnitPrice * od.Quantity) '
            'AS TotalRevenue '
            'FROM Products AS p '
            'JOIN "Order Details" AS od '
            'ON p.ProductID = od.ProductID '
            'GROUP BY p.ProductID, p.ProductName '
            'ORDER BY TotalRevenue DESC '
            'LIMIT 5'
        ),
        explanation=(
            "The query calculates and ranks "
            "product revenue."
        ),
        referenced_tables=[
            "Products",
            "Order Details",
        ],
        expected_columns=[
            "ProductName",
            "TotalRevenue",
        ],
        assumptions=[],
        requires_clarification=False,
        clarification_question=None,
    )


def build_query_result() -> QueryExecutionResult:
    """
    Build a realistic query result.
    """

    sql_result = build_sql_result()

    rows = [
        {
            "ProductName": "Product A",
            "TotalRevenue": 50000.0,
        },
        {
            "ProductName": "Product B",
            "TotalRevenue": 42500.0,
        },
        {
            "ProductName": "Product C",
            "TotalRevenue": 39000.0,
        },
        {
            "ProductName": "Product D",
            "TotalRevenue": 34500.0,
        },
        {
            "ProductName": "Product E",
            "TotalRevenue": 31000.0,
        },
    ]

    return QueryExecutionResult(
        columns=[
            "ProductName",
            "TotalRevenue",
        ],
        rows=rows,
        row_count=len(rows),
        execution_time_ms=4.25,
        truncated=False,
        query=sql_result.sql,
    )


def build_analysis_result() -> AnalysisResult:
    """
    Build a valid Agent 4 output.
    """

    return AnalysisResult(
        summary=(
            "Product A generated the highest revenue "
            "among the five returned products."
        ),
        findings=[
            AnalysisFinding(
                title="Highest-revenue product",
                description=(
                    "Product A ranks first in the result."
                ),
                evidence=[
                    "Product A revenue is 50000.0.",
                ],
                importance="high",
            )
        ],
        trends=[],
        anomalies=[],
        possible_drivers=[
            (
                "Higher order volume or unit value may "
                "have contributed to Product A's ranking."
            )
        ],
        limitations=[
            (
                "The ranking alone does not establish "
                "the cause of revenue differences."
            )
        ],
    )


def build_final_report() -> FinalReport:
    """
    Build a valid Agent 5 output.
    """

    return FinalReport(
        business_summary=(
            "Product A leads the returned products "
            "with revenue of 50000.0."
        ),
        key_findings=[
            (
                "Product A generated the highest "
                "revenue in the returned ranking."
            ),
            (
                "The result contains five products "
                "ordered by revenue."
            ),
        ],
        root_cause_indicators=[
            (
                "The ranking suggests differences in "
                "sales volume or transaction value, but "
                "the query does not prove causation."
            )
        ],
        recommendations=[
            Recommendation(
                action=(
                    "Review Product A's transaction "
                    "volume and pricing pattern."
                ),
                rationale=(
                    "Product A leads the revenue ranking, "
                    "so its sales pattern may identify "
                    "repeatable commercial drivers."
                ),
                priority="medium",
            )
        ],
        chart=ChartSpecification(
            chart_type="bar",
            title="Top Products by Revenue",
            x_column="ProductName",
            y_columns=[
                "TotalRevenue",
            ],
            category_column=None,
            rationale=(
                "A bar chart clearly compares "
                "revenue across ranked products."
            ),
        ),
        limitations=[
            (
                "The result is limited to the five "
                "highest-revenue products."
            )
        ],
    )


def build_completed_state() -> GraphState:
    """
    Build a fully completed workflow state.
    """

    state = create_initial_state(
        user_question=(
            "Show the top 5 products by revenue."
        ),
        max_sql_retries=3,
    )

    sql_result = build_sql_result()

    state.update(
        {
            "workflow_status": "completed",
            "current_node": "reporting_agent",
            "intent": build_intent(),
            "plan": build_plan(),
            "database_schema": (
                "Sanitized test schema"
            ),
            "sql_result": sql_result,
            "generated_sql": sql_result.sql,
            "sql_validation_errors": [],
            "sql_execution_error": None,
            "query_result": (
                build_query_result()
            ),
            "analysis_result": (
                build_analysis_result()
            ),
            "final_report": (
                build_final_report()
            ),
            "error_type": None,
            "error_message": None,
            "execution_trace": [
                {
                    "node": "start",
                    "status": "initialized",
                    "message": (
                        "Workflow state created."
                    ),
                },
                {
                    "node": "intent_agent",
                    "status": "completed",
                    "message": (
                        "Intent understood and validated."
                    ),
                },
                {
                    "node": "planning_agent",
                    "status": "completed",
                    "message": (
                        "Analytical plan created."
                    ),
                },
                {
                    "node": "schema_manager",
                    "status": "completed",
                    "message": (
                        "Database schema prepared."
                    ),
                },
                {
                    "node": "sql_generation_agent",      "status": "completed",
                    "message": (
                        "Read-only SQL generated."
                    ),
                },
                {
                    "node": (
                        "sql_validation_execution"
                    ),
                    "status": "completed",
                    "message": (
                        "SQL executed successfully."
                    ),
                },
                {
                    "node": "analysis_agent",
                    "status": "completed",
                    "message": (
                        "Query result analyzed."
                    ),
                },
                {
                    "node": "reporting_agent",
                    "status": "completed",
                    "message": (
                        "Final report generated."
                    ),
                },
            ],
        }
    )

    return state


def build_clarification_state() -> GraphState:
    """
    Build a clarification-required state.
    """

    state = create_initial_state(
        user_question="Analyze the results.",
        max_sql_retries=3,
    )

    clarification_intent = IntentResult(
        original_question="Analyze the results.",
        rewritten_question=(
            "Determine the metric and grouping "
            "required for analysis."
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
            "Which metric and grouping would "
            "you like to analyze?"
        ),
    )

    state.update(
        {
            "workflow_status": (
                "clarification_required"
            ),
            "current_node": "clarification",
            "intent": clarification_intent,
            "clarification_required": True,
            "clarification_question": (
                clarification_intent
                .clarification_question
            ),
            "execution_trace": [
                {
                    "node": "start",
                    "status": "initialized",
                    "message": (
                        "Workflow state created."
                    ),
                },
                {
                    "node": "intent_agent",
                    "status": "completed",
                    "message": (
                        "Clarification is required."
                    ),
                },
                {
                    "node": "clarification",
                    "status": "waiting_for_user",
                    "message": (
                        "Workflow paused for clarification."
                    ),
                },
            ],
        }
    )

    return state


def build_failed_state() -> GraphState:
    """
    Build a controlled workflow-failure state.
    """

    state = create_initial_state(
        user_question=(
            "Show product revenue."
        ),
        max_sql_retries=3,
    )

    state.update(
        {
            "workflow_status": "failed",
            "current_node": (
                "sql_generation_agent"
            ),
            "error_type": "RuntimeError",
            "error_message": (
                "SQL generation failed after "
                "the configured attempts."
            ),
            "sql_retry_count": 3,
            "execution_trace": [
                {
                    "node": "start",
                    "status": "initialized",
                    "message": (
                        "Workflow state created."
                    ),
                },
                {
                    "node": (
                        "sql_generation_agent"
                    ),
                    "status": "failed",
                    "message": (
                        "SQL generation failed."
                    ),
                },
                {
                    "node": "failure",
                    "status": "failed",
                    "message": (
                        "Workflow ended safely."
                    ),
                },
            ],
        }
    )

    return state


def build_fake_services(
    state: GraphState,
    expose_technical_details: bool = True,
) -> AppServices:
    """
    Build a lightweight AppServices-compatible object.
    """

    settings = SimpleNamespace(
        application_name=(
            "Multi-Agent Analytics API"
        ),
        application_version="1.0.0",
        required_models=(
            "llama3.2:latest",
            "sqlcoder:latest",
        ),
        maximum_sql_retries=3,
        langgraph_recursion_limit=40,
        expose_technical_details=(
            expose_technical_details
        ),
    )

    return AppServices(
        settings=settings,
        ollama_manager=(
            FakeOllamaManager()
        ),  # type: ignore[arg-type]
        graph=FakeGraph(state),
        database_manager=(
            FakeDatabaseManager()
        ),  # type: ignore[arg-type]
        chart_tool=None,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------
# API tests
# ---------------------------------------------------------------------

class FastAPIBackendTests(
    unittest.TestCase
):
    """
    FastAPI endpoint test suite.
    """

    def setUp(self) -> None:
        """
        Reset dependency overrides before every test.
        """

        app.dependency_overrides.clear()

        self.completed_state = (
            build_completed_state()
        )

        self.services = (
            build_fake_services(
                self.completed_state
            )
        )

        app.dependency_overrides[
            get_application_services
        ] = lambda: self.services

        # TestClient is intentionally not used as a context manager.
        # This prevents the real lifespan initializer from starting
        # Ollama during isolated API unit tests.
        self.client = TestClient(
            app
        )

    def tearDown(self) -> None:
        """
        Clear API overrides after every test.
        """

        app.dependency_overrides.clear()

        self.client.close()

    def test_health_endpoint(
        self,
    ) -> None:
        """
        GET /health returns basic API health.
        """

        response = self.client.get(
            "/health"
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        payload = response.json()

        self.assertEqual(
            payload["status"],
            "healthy",
        )

        self.assertEqual(
            payload["service"],
            "multi-agent-analytics-api",
        )

        self.assertEqual(
            payload["version"],
            "1.0.0",
        )

    def test_ready_endpoint(
        self,
    ) -> None:
        """
        GET /ready reports all dependencies as ready.
        """

        response = self.client.get(
            "/ready"
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        payload = response.json()

        self.assertEqual(
            payload["status"],
            "ready",
        )

        components = {
            item["component"]: item
            for item in payload["components"]
        }

        self.assertTrue(
            components["ollama"]["ready"]
        )

        self.assertTrue(
            components["database"]["ready"]
        )

        self.assertTrue(
            components["langgraph"]["ready"]
        )

    def test_models_endpoint(
        self,
    ) -> None:
        """
        GET /api/v1/models returns required model status.
        """

        response = self.client.get(
            "/api/v1/models"
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        payload = response.json()

        self.assertTrue(
            payload["ollama_available"]
        )

        model_status = {
            item["name"]: item["installed"]
            for item in payload["models"]
        }

        self.assertTrue(
            model_status[
                "llama3.2:latest"
            ]
        )

        self.assertTrue(
            model_status[
                "sqlcoder:latest"
            ]
        )

    def test_completed_analytics_response_without_technical_details(
        self,
    ) -> None:
        """
        Completed workflow returns business output without optional
        technical information.
        """

        response = self.client.post(
            "/api/v1/analytics/query",
            json={
                "question": (
                    "Show the top 5 products "
                    "by revenue."
                ),
                "max_sql_retries": 3,
                "include_technical_details": False,
            },
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        payload = response.json()

        self.assertEqual(
            payload["status"],
            "completed",
        )

        self.assertFalse(
            payload["clarification_required"]
        )

        self.assertIsNotNone(
            payload["business_summary"]
        )

        self.assertEqual(
            payload["query_result"][
                "row_count"
            ],
            5,
        )

        self.assertEqual(
            payload["chart"]["chart_type"],
            "bar",
        )

        self.assertEqual(
            payload["chart"]["x_column"],
            "ProductName",
        )

        self.assertEqual(
            payload["chart"]["y_columns"],
            [
                "TotalRevenue",
            ],
        )

        self.assertIsNone(
            payload["technical_details"]
        )

        self.assertIsNone(
            payload["error"]
        )

    def test_completed_response_with_technical_details(
        self,
    ) -> None:
        """
        Technical details are returned when requested and enabled.
        """

        response = self.client.post(
            "/api/v1/analytics/query",
            json={
                "question": (
                    "Show the top 5 products "
                    "by revenue."
                ),
                "max_sql_retries": 3,
                "include_technical_details": True,
            },
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        payload = response.json()

        technical_details = payload[
            "technical_details"
        ]

        self.assertIsNotNone(
            technical_details
        )

        self.assertIn(
            "SELECT",
            technical_details[
                "generated_sql"
            ].upper(),
        )

        self.assertEqual(
            technical_details[
                "sql_retry_count"
            ],
            0,
        )

        self.assertIsNotNone(
            technical_details["intent"]
        )

        self.assertIsNotNone(
            technical_details["plan"]
        )

        self.assertGreaterEqual(
            len(
                technical_details[
                    "execution_trace"
                ]
            ),
            1,
        )

    def test_technical_details_disabled_by_backend(
        self,
    ) -> None:
        """
        Backend configuration overrides the client's request.
        """

        services = build_fake_services(
            self.completed_state,
            expose_technical_details=False,
        )

        app.dependency_overrides[
            get_application_services
        ] = lambda: services

        response = self.client.post(
            "/api/v1/analytics/query",
            json={
                "question": (
                    "Show the top 5 products "
                    "by revenue."
                ),
                "include_technical_details": True,
            },
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertIsNone(
            response.json()[
                "technical_details"
            ]
        )

    def test_clarification_response(
        self,
    ) -> None:
        """
        Clarification state is converted into an API response.
        """

        services = build_fake_services(
            build_clarification_state()
        )

        app.dependency_overrides[
            get_application_services
        ] = lambda: services

        response = self.client.post(
            "/api/v1/analytics/query",
            json={
                "question": (
                    "Analyze the results."
                ),
                "max_sql_retries": 3,
                "include_technical_details": False,
            },
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        payload = response.json()

        self.assertEqual(
            payload["status"],
            "clarification_required",
        )

        self.assertTrue(
            payload["clarification_required"]
        )

        self.assertIsNotNone(
            payload[
                "clarification_question"
            ]
        )

        self.assertIsNone(
            payload["business_summary"]
        )

        self.assertIsNone(
            payload["query_result"]
        )

        self.assertIsNone(
            payload["error"]
        )

    def test_controlled_failed_response(
        self,
    ) -> None:
        """
        Controlled graph failures return a structured response.
        """

        services = build_fake_services(
            build_failed_state()
        )

        app.dependency_overrides[
            get_application_services
        ] = lambda: services

        response = self.client.post(
            "/api/v1/analytics/query",
            json={
                "question": (
                    "Show product revenue."
                ),
                "max_sql_retries": 3,
                "include_technical_details": True,
            },
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        payload = response.json()

        self.assertEqual(
            payload["status"],
            "failed",
        )

        self.assertIsNotNone(
            payload["error"]
        )

        self.assertEqual(
            payload["error"]["error_type"],
            "RuntimeError",
        )

        self.assertEqual(
            payload["error"]["stage"],
            "sql_generation_agent",
        )

        self.assertFalse(
            payload["error"]["retryable"]
        )

    def test_question_too_short_returns_422(
        self,
    ) -> None:
        """
        Request validation rejects a question shorter than three
        characters.
        """

        response = self.client.post(
            "/api/v1/analytics/query",
            json={
                "question": "Hi",
            },
        )

        self.assertEqual(
            response.status_code,
            422,
        )

        payload = response.json()

        self.assertEqual(
            payload["detail"],
            "Request validation failed.",
        )

    def test_punctuation_only_question_returns_422(
        self,
    ) -> None:
        """
        Custom Pydantic validation rejects punctuation-only input.
        """

        response = self.client.post(
            "/api/v1/analytics/query",
            json={
                "question": "???",
            },
        )

        self.assertEqual(
            response.status_code,
            422,
        )

    def test_unknown_request_field_returns_422(
        self,
    ) -> None:
        """
        API schemas reject unsupported request fields.
        """

        response = self.client.post(
            "/api/v1/analytics/query",
            json={
                "question": (
                    "Show product revenue."
                ),
                "unsupported_field": True,
            },
        )

        self.assertEqual(
            response.status_code,
            422,
        )

    def test_retry_request_is_capped_by_backend(
        self,
    ) -> None:
        """
        Requested retries are limited by backend settings.
        """

        fake_graph = self.services.graph

        response = self.client.post(
            "/api/v1/analytics/query",
            json={
                "question": (
                    "Show the top 5 products "
                    "by revenue."
                ),
                "max_sql_retries": 5,
            },
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertIsInstance(
            fake_graph,
            FakeGraph,
        )

        self.assertIsNotNone(
            fake_graph.last_input_state
        )

        self.assertEqual(
            fake_graph.last_input_state[
                "max_sql_retries"
            ],
            3,
        )

    def test_recursion_limit_is_passed_to_graph(
        self,
    ) -> None:
        """
        Backend recursion protection is supplied to LangGraph.
        """

        fake_graph = self.services.graph

        response = self.client.post(
            "/api/v1/analytics/query",
            json={
                "question": (
                    "Show the top 5 products "
                    "by revenue."
                ),
            },
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertIsInstance(
            fake_graph,
            FakeGraph,
        )

        self.assertEqual(
            fake_graph.last_config,
            {
                "recursion_limit": 40,
            },
        )

    def test_unexpected_graph_failure_returns_500(
        self,
    ) -> None:
        """
        Unexpected backend exceptions return HTTP 500 without exposing
        the internal exception message.
        """

        settings = self.services.settings

        failing_services = AppServices(
            settings=settings,
            ollama_manager=(
                self.services.ollama_manager
            ),
            graph=FailingGraph(),
            database_manager=(
                self.services
                .database_manager
            ),
            chart_tool=None,  # type: ignore[arg-type]
        )

        app.dependency_overrides[
            get_application_services
        ] = lambda: failing_services

        response = self.client.post(
            "/api/v1/analytics/query",
            json={
                "question": (
                    "Show product revenue."
                ),
            },
        )

        self.assertEqual(
            response.status_code,
            500,
        )

        payload = response.json()

        self.assertEqual(
            payload["detail"]["message"],
            (
                "The analytics backend encountered "
                "an unexpected error."
            ),
        )

        self.assertNotIn(
            "Simulated unexpected graph failure",
            str(payload),
        )

    def test_api_root(
        self,
    ) -> None:
        """
        Root endpoint returns API paths.
        """

        response = self.client.get(
            "/"
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        payload = response.json()

        self.assertEqual(
            payload["health"],
            "/health",
        )

        self.assertEqual(
            payload["analytics"],
            (
                "/api/v1/analytics/query"
            ),
        )

        self.assertEqual(
            payload["documentation"],
            "/docs",
        )


# ---------------------------------------------------------------------
# Standalone execution
# ---------------------------------------------------------------------

def main() -> None:
    """
    Run the API tests using Python's built-in unittest runner.
    """

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(
        FastAPIBackendTests
    )

    runner = unittest.TextTestRunner(
        verbosity=2
    )

    result = runner.run(
        suite
    )

    total_tests = result.testsRun

    failed_tests = (
        len(result.failures)
        + len(result.errors)
    )

    passed_tests = (
        total_tests - failed_tests
    )

    print("\n" + "=" * 80)
    print("FINAL FASTAPI TEST SUMMARY")
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
            "\nRESULT: ALL FASTAPI TESTS PASSED"
        )

        print(
            "The backend API is ready for "
            "Streamlit integration."
        )

        raise SystemExit(0)

    print(
        "\nRESULT: SOME FASTAPI TESTS FAILED"
    )

    raise SystemExit(1)


if __name__ == "__main__":
    main()