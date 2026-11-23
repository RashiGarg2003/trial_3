"""
Shared FastAPI dependencies for the Multi-Agent Analytics backend.

This module initializes and reuses:

1. OllamaManager
2. Compiled LangGraph workflow
3. DatabaseManager
4. ChartTool
5. Backend configuration

The resources are created lazily and cached. Importing this file does
not immediately start Ollama or run the workflow.

FastAPI endpoints will receive these resources through dependency
injection.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from database.db_manager import DatabaseManager
from tools.chart_tool import ChartTool
from utils.ollama_manager import OllamaManager
from workflow.graph import create_graph


PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent


DEFAULT_OLLAMA_DIRECTORY = Path(
    r"C:\Users\Rashi.Garg\Ollama"
)

DEFAULT_DATABASE_PATH = (
    PROJECT_ROOT
    / "database"
    / "northwind.db"
)

DEFAULT_OLLAMA_HOST = (
    "http://localhost:11434"
)

DEFAULT_REQUIRED_MODELS = (
    "llama3.2:latest",
    "sqlcoder:latest",
)


@dataclass(frozen=True)
class APISettings:
    """
    Backend configuration loaded from environment variables.

    The dataclass is immutable so configuration cannot be changed
    accidentally while the API is running.
    """

    application_name: str
    application_version: str

    ollama_directory: Path
    ollama_host: str
    required_models: tuple[str, ...]

    database_path: Path

    maximum_result_rows: int
    maximum_chart_rows: int
    maximum_sql_retries: int
    langgraph_recursion_limit: int

    ollama_startup_timeout_seconds: float
    ollama_poll_interval_seconds: float

    expose_technical_details: bool


@dataclass
class AppServices:
    """
    Long-lived backend service container.

    Attributes:
        settings:
            Validated backend configuration.

        ollama_manager:
            Ollama server and model manager.

        graph:
            Compiled LangGraph application.

        database_manager:
            Read-only SQLite database manager used for readiness checks.

        chart_tool:
            Deterministic Plotly chart renderer.
    """

    settings: APISettings
    ollama_manager: OllamaManager
    graph: Any
    database_manager: DatabaseManager
    chart_tool: ChartTool


class ServiceInitializationError(RuntimeError):
    """
    Raised when required backend resources cannot be initialized.
    """


def _parse_boolean(
    value: str | None,
    default: bool,
) -> bool:
    """
    Parse a boolean environment variable.

    Accepted values:

    True:
        true, 1, yes, on

    False:
        false, 0, no, off
    """

    if value is None:
        return default

    normalized_value = (
        value.strip().lower()
    )

    true_values = {
        "true",
        "1",
        "yes",
        "on",
    }

    false_values = {
        "false",
        "0",
        "no",
        "off",
    }

    if normalized_value in true_values:
        return True

    if normalized_value in false_values:
        return False

    raise ValueError(
        "Boolean environment variable must contain "
        "one of: true, false, 1, 0, yes, no, on, off."
    )


def _parse_integer(
    variable_name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    """
    Parse and validate an integer environment variable.
    """

    raw_value = os.getenv(
        variable_name
    )

    if raw_value is None:
        return default

    try:
        parsed_value = int(
            raw_value.strip()
        )

    except ValueError as exc:
        raise ValueError(
            f"{variable_name} must be an integer."
        ) from exc

    if not (
        minimum
        <= parsed_value
        <= maximum
    ):
        raise ValueError(
            f"{variable_name} must be between "
            f"{minimum} and {maximum}."
        )

    return parsed_value


def _parse_float(
    variable_name: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    """
    Parse and validate a floating-point environment variable.
    """

    raw_value = os.getenv(
        variable_name
    )

    if raw_value is None:
        return default

    try:
        parsed_value = float(
            raw_value.strip()
        )

    except ValueError as exc:
        raise ValueError(
            f"{variable_name} must be numeric."
        ) from exc

    if not (
        minimum
        <= parsed_value
        <= maximum
    ):
        raise ValueError(
            f"{variable_name} must be between "
            f"{minimum} and {maximum}."
        )

    return parsed_value


def _parse_required_models(
    value: str | None,
) -> tuple[str, ...]:
    """
    Parse a comma-separated model list.

    Example environment value:

        llama3.2:latest,sqlcoder:latest
    """

    if value is None:
        return DEFAULT_REQUIRED_MODELS

    model_names = tuple(
        model_name.strip()
        for model_name in value.split(",")
        if model_name.strip()
    )

    if not model_names:
        raise ValueError(
            "REQUIRED_OLLAMA_MODELS cannot be empty."
        )

    return model_names


@lru_cache(maxsize=1)
def get_settings() -> APISettings:
    """
    Load and cache backend configuration.

    Environment variables override the local defaults.
    """

    ollama_directory = Path(
        os.getenv(
            "OLLAMA_DIRECTORY",
            str(
                DEFAULT_OLLAMA_DIRECTORY
            ),
        )
    ).expanduser().resolve()

    database_path = Path(
        os.getenv(
            "DATABASE_PATH",
            str(
                DEFAULT_DATABASE_PATH
            ),
        )
    ).expanduser().resolve()

    ollama_host = os.getenv(
        "OLLAMA_HOST",
        DEFAULT_OLLAMA_HOST,
    ).strip().rstrip("/")

    if not ollama_host:
        raise ValueError(
            "OLLAMA_HOST cannot be empty."
        )

    required_models = (
        _parse_required_models(
            os.getenv(
                "REQUIRED_OLLAMA_MODELS"
            )
        )
    )

    return APISettings(
        application_name=os.getenv(
            "API_APPLICATION_NAME",
            "Multi-Agent Analytics API",
        ).strip(),
        application_version=os.getenv(
            "API_APPLICATION_VERSION",
            "1.0.0",
        ).strip(),
        ollama_directory=(
            ollama_directory
        ),
        ollama_host=ollama_host,
        required_models=required_models,
        database_path=database_path,
        maximum_result_rows=(
            _parse_integer(
                variable_name=(
                    "MAXIMUM_RESULT_ROWS"
                ),
                default=1000,
                minimum=1,
                maximum=10000,
            )
        ),
        maximum_chart_rows=(
            _parse_integer(
                variable_name=(
                    "MAXIMUM_CHART_ROWS"
                ),
                default=100,
                minimum=1,
                maximum=1000,
            )
        ),
        maximum_sql_retries=(
            _parse_integer(
                variable_name=(
                    "MAXIMUM_SQL_RETRIES"
                ),
                default=3,
                minimum=0,
                maximum=5,
            )
        ),
        langgraph_recursion_limit=(
            _parse_integer(
                variable_name=(
                    "LANGGRAPH_RECURSION_LIMIT"
                ),
                default=50,
                minimum=10,
                maximum=100,
            )
        ),
        ollama_startup_timeout_seconds=(
            _parse_float(
                variable_name=(
                    "OLLAMA_STARTUP_TIMEOUT_SECONDS"
                ),
                default=90.0,
                minimum=5.0,
                maximum=300.0,
            )
        ),
        ollama_poll_interval_seconds=(
            _parse_float(
                variable_name=(
                    "OLLAMA_POLL_INTERVAL_SECONDS"
                ),
                default=1.0,
                minimum=0.1,
                maximum=10.0,
            )
        ),
        expose_technical_details=(
            _parse_boolean(
                value=os.getenv(
                    "EXPOSE_TECHNICAL_DETAILS"
                ),
                default=True,
            )
        ),
    )


def create_ollama_manager(
    settings: APISettings,
) -> OllamaManager:
    """
    Create the configured Ollama manager.

    This function does not start the server. Startup is performed in
    initialize_services().
    """

    return OllamaManager(
        ollama_directory=(
            settings.ollama_directory
        ),
        host=settings.ollama_host,
        startup_timeout_seconds=(
            settings
            .ollama_startup_timeout_seconds
        ),
        poll_interval_seconds=(
            settings
            .ollama_poll_interval_seconds
        ),
    )


def create_database_manager(
    settings: APISettings,
) -> DatabaseManager:
    """
    Create a read-only DatabaseManager for readiness checks.

    The LangGraph services also use the same configured database path.
    """

    return DatabaseManager(
        database_path=(
            settings.database_path
        ),
        max_rows=(
            settings.maximum_result_rows
        ),
    )


def initialize_services() -> AppServices:
    """
    Initialize all long-lived FastAPI resources.

    Processing order:

    1. Load settings.
    2. Check or start Ollama.
    3. Confirm required models.
    4. Confirm database connection and integrity.
    5. Compile the LangGraph workflow.
    6. Initialize ChartTool.

    Raises:
        ServiceInitializationError:
            If any required dependency cannot be initialized.
    """

    try:
        settings = get_settings()

        ollama_manager = (
            create_ollama_manager(
                settings
            )
        )

        ollama_manager.ensure_server_running()

        ollama_manager.validate_required_models(
            required_models=list(
                settings.required_models
            )
        )

        database_manager = (
            create_database_manager(
                settings
            )
        )

        connection_status = (
            database_manager
            .test_connection()
        )

        if not connection_status:
            raise ConnectionError(
                "Database connection test failed."
            )

        integrity_result = (
            database_manager
            .run_integrity_check()
        )

        if integrity_result.lower() != "ok":
            raise RuntimeError(
                "Database integrity check failed: "
                f"{integrity_result}"
            )

        graph = create_graph()

        chart_tool = ChartTool(
            maximum_chart_rows=(
                settings.maximum_chart_rows
            )
        )

        return AppServices(
            settings=settings,
            ollama_manager=ollama_manager,
            graph=graph,
            database_manager=database_manager,
            chart_tool=chart_tool,
        )

    except Exception as exc:
        raise ServiceInitializationError(
            "The FastAPI backend services could "
            "not be initialized. "
            f"Original error: {exc}"
        ) from exc


@lru_cache(maxsize=1)
def get_services() -> AppServices:
    """
    Return the cached backend service container.

    FastAPI endpoints use this function through Depends().
    The first call initializes all services. Later calls return the
    cached container.
    """

    return initialize_services()


def clear_dependency_caches() -> None:
    """
    Clear cached settings and services.

    This is intended for:

    - Automated tests
    - Development configuration changes
    - Controlled application shutdown

    It should not normally be called by API endpoints.
    """

    get_services.cache_clear()
    get_settings.cache_clear()


def check_ollama_readiness(
    services: AppServices,
) -> tuple[bool, str]:
    """
    Check Ollama API and required-model readiness.
    """

    try:
        if not (
            services
            .ollama_manager
            .is_server_running()
        ):
            return (
                False,
                "Ollama API is not responding.",
            )

        installed_models = set(
            services
            .ollama_manager
            .get_installed_models()
        )

        missing_models = [
            model_name
            for model_name
            in services.settings.required_models
            if model_name
            not in installed_models
        ]

        if missing_models:
            return (
                False,
                "Required Ollama models are missing: "
                + ", ".join(
                    missing_models
                ),
            )

        return (
            True,
            "Ollama and required models are available.",
        )

    except Exception as exc:
        return (
            False,
            "Ollama readiness check failed: "
            f"{type(exc).__name__}.",
        )


def check_database_readiness(
    services: AppServices,
) -> tuple[bool, str]:
    """
    Check read-only database readiness.
    """

    try:
        connection_status = (
            services
            .database_manager
            .test_connection()
        )

        if not connection_status:
            return (
                False,
                "Database connection test failed.",
            )

        integrity_result = (
            services
            .database_manager
            .run_integrity_check()
        )

        if integrity_result.lower() != "ok":
            return (
                False,
                "Database integrity check failed.",
            )

        return (
            True,
            "Read-only SQLite database is available.",
        )

    except Exception as exc:
        return (
            False,
            "Database readiness check failed: "
            f"{type(exc).__name__}.",
        )


def check_graph_readiness(
    services: AppServices,
) -> tuple[bool, str]:
    """
    Check that a compiled LangGraph object is available.
    """

    graph = services.graph

    if graph is None:
        return (
            False,
            "Compiled LangGraph is unavailable.",
        )

    if not hasattr(
        graph,
        "invoke",
    ):
        return (
            False,
            "Configured graph does not support invoke().",
        )

    return (
        True,
        "Compiled five-agent LangGraph is available.",
    )


def main() -> None:
    """
    Run a standalone dependency smoke test.
    """

    print("=" * 80)
    print("FASTAPI DEPENDENCIES SMOKE TEST")
    print("=" * 80)

    try:
        services = get_services()

        print(
            f"Application       : "
            f"{services.settings.application_name}"
        )

        print(
            f"Version           : "
            f"{services.settings.application_version}"
        )

        print(
            f"Ollama host       : "
            f"{services.settings.ollama_host}"
        )

        print(
            f"Ollama directory  : "
            f"{services.settings.ollama_directory}"
        )

        print(
            f"Database path     : "
            f"{services.settings.database_path}"
        )

        print(
            f"Required models   : "
            f"{list(services.settings.required_models)}"
        )

        print(
            f"Maximum rows      : "
            f"{services.settings.maximum_result_rows}"
        )

        ollama_ready, ollama_message = (
            check_ollama_readiness(
                services
            )
        )

        database_ready, database_message = (
            check_database_readiness(
                services
            )
        )

        graph_ready, graph_message = (
            check_graph_readiness(
                services
            )
        )

        print("\nReadiness checks:")

        print(
            f"- Ollama  : "
            f"{ollama_ready} | "
            f"{ollama_message}"
        )

        print(
            f"- Database: "
            f"{database_ready} | "
            f"{database_message}"
        )

        print(
            f"- Graph   : "
            f"{graph_ready} | "
            f"{graph_message}"
        )

        if not all(
            [
                ollama_ready,
                database_ready,
                graph_ready,
            ]
        ):
            raise RuntimeError(
                "One or more backend dependencies "
                "are not ready."
            )

        print("\n" + "=" * 80)
        print(
            "RESULT: FASTAPI DEPENDENCIES "
            "ARE WORKING"
        )
        print("=" * 80)

    except Exception as exc:
        print("\n" + "=" * 80)
        print(
            "RESULT: FASTAPI DEPENDENCIES FAILED"
        )
        print("=" * 80)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()