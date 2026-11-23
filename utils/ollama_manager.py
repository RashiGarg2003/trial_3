"""
Local Ollama server manager.

This utility:

1. Checks whether the Ollama API is already available.
2. Starts `ollama serve` only when required.
3. Waits until the API becomes ready.
4. Checks whether required models are installed.
5. Avoids starting duplicate Ollama server processes.

The future Streamlit application will call ensure_server_running()
during application startup.
"""

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen


DEFAULT_OLLAMA_DIRECTORY = Path(
    r"C:\Users\Rashi.Garg\Ollama"
)

DEFAULT_OLLAMA_HOST = "http://localhost:11434"

DEFAULT_REQUIRED_MODELS = [
    "llama3.2:latest",
    "sqlcoder:latest",
]


class OllamaManager:
    """
    Manages the locally installed Ollama server.

    The manager first checks the Ollama HTTP API. It starts a new
    process only when the API is unavailable.
    """

    def __init__(
        self,
        ollama_directory: Path | str = DEFAULT_OLLAMA_DIRECTORY,
        host: str = DEFAULT_OLLAMA_HOST,
        startup_timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        """
        Initialize the Ollama manager.

        Args:
            ollama_directory:
                Directory containing ollama.exe.

            host:
                Local Ollama API address.

            startup_timeout_seconds:
                Maximum time to wait for the server to start.

            poll_interval_seconds:
                Time between API readiness checks.
        """

        self.ollama_directory = Path(
            ollama_directory
        ).expanduser().resolve()

        self.host = host.strip().rstrip("/")

        if not self.host:
            raise ValueError(
                "The Ollama host cannot be empty."
            )

        if startup_timeout_seconds <= 0:
            raise ValueError(
                "startup_timeout_seconds must be greater than zero."
            )

        if poll_interval_seconds <= 0:
            raise ValueError(
                "poll_interval_seconds must be greater than zero."
            )

        self.startup_timeout_seconds = (
            startup_timeout_seconds
        )

        self.poll_interval_seconds = (
            poll_interval_seconds
        )

        self.process: Optional[subprocess.Popen] = None

    @property
    def tags_url(self) -> str:
        """
        Return the Ollama model-list API endpoint.
        """

        return f"{self.host}/api/tags"

    def find_executable(self) -> Path:
        """
        Find ollama.exe inside the configured installation directory.

        Returns:
            Resolved path to ollama.exe.

        Raises:
            FileNotFoundError:
                If the executable cannot be located.
        """

        direct_candidates = [
            self.ollama_directory / "ollama.exe",
            self.ollama_directory / "Ollama.exe",
        ]

        for candidate in direct_candidates:
            if candidate.exists() and candidate.is_file():
                return candidate

        if self.ollama_directory.exists():
            recursive_matches = list(
                self.ollama_directory.rglob(
                    "ollama.exe"
                )
            )

            if recursive_matches:
                return recursive_matches[0]

        raise FileNotFoundError(
            "Could not find ollama.exe.\n"
            f"Configured directory: {self.ollama_directory}\n"
            "Confirm the Ollama installation location."
        )

    def is_server_running(
        self,
        timeout_seconds: float = 2.0,
    ) -> bool:
        """
        Check whether the local Ollama API is responding.
        """

        request = Request(
            self.tags_url,
            method="GET",
        )

        try:
            with urlopen(
                request,
                timeout=timeout_seconds,
            ) as response:
                return response.status == 200

        except (
            URLError,
            TimeoutError,
            ConnectionError,
            OSError,
        ):
            return False

    def get_installed_models(self) -> list[str]:
        """
        Return model names reported by the Ollama API.

        Raises:
            ConnectionError:
                If the Ollama API cannot be reached.
        """

        request = Request(
            self.tags_url,
            method="GET",
        )

        try:
            with urlopen(
                request,
                timeout=5.0,
            ) as response:
                response_body = response.read().decode(
                    "utf-8"
                )

        except (
            URLError,
            TimeoutError,
            ConnectionError,
            OSError,
        ) as exc:
            raise ConnectionError(
                "Could not retrieve the installed Ollama models. "
                f"Host: {self.host}. "
                f"Original error: {exc}"
            ) from exc

        try:
            payload = json.loads(response_body)

        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "The Ollama API returned invalid JSON."
            ) from exc

        models = payload.get("models", [])

        return [
            model["name"]
            for model in models
            if isinstance(model, dict)
            and isinstance(model.get("name"), str)
        ]

    def get_missing_models(
        self,
        required_models: list[str],
    ) -> list[str]:
        """
        Return required model names that are not installed.
        """

        installed_models = set(
            self.get_installed_models()
        )

        return [
            model
            for model in required_models
            if model not in installed_models
        ]

    def _get_process_creation_flags(self) -> int:
        """
        Return Windows flags that prevent a visible server console.

        On non-Windows systems, no special flags are used.
        """

        if os.name == "nt":
            return subprocess.CREATE_NO_WINDOW

        return 0

    def start_server(self) -> subprocess.Popen:
        """
        Start `ollama serve` as a background process.

        If the API is already running, no new process is created.

        Returns:
            The newly created process, or the existing process
            previously created by this manager.
        """

        if self.is_server_running():
            if self.process is not None:
                return self.process

            raise RuntimeError(
                "The Ollama server is already running, but it was "
                "started outside this Python process. No additional "
                "server is required."
            )

        executable_path = self.find_executable()

        creation_flags = (
            self._get_process_creation_flags()
        )

        try:
            self.process = subprocess.Popen(
                [
                    str(executable_path),
                    "serve",
                ],
                cwd=str(executable_path.parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                creationflags=creation_flags,
            )

        except OSError as exc:
            raise RuntimeError(
                "Could not start the Ollama server.\n"
                f"Executable: {executable_path}\n"
                f"Original error: {exc}"
            ) from exc

        return self.process

    def wait_until_ready(self) -> None:
        """
        Wait until the Ollama API responds or the timeout expires.
        """

        deadline = (
            time.monotonic()
            + self.startup_timeout_seconds
        )

        while time.monotonic() < deadline:
            if self.process is not None:
                return_code = self.process.poll()

                if return_code is not None:
                    raise RuntimeError(
                        "The Ollama server process stopped before "
                        "the API became ready. "
                        f"Exit code: {return_code}"
                    )

            if self.is_server_running():
                return

            time.sleep(
                self.poll_interval_seconds
            )

        raise TimeoutError(
            "Ollama did not become ready within "
            f"{self.startup_timeout_seconds} seconds.\n"
            f"Expected API: {self.host}"
        )

    def ensure_server_running(self) -> bool:
        """
        Ensure that Ollama is available.

        Returns:
            False when the server was already running.
            True when this method started the server.
        """

        if self.is_server_running():
            return False

        self.start_server()
        self.wait_until_ready()

        return True

    def validate_required_models(
        self,
        required_models: Optional[
            list[str]
        ] = None,
    ) -> list[str]:
        """
        Confirm that required project models are installed.

        Returns:
            Installed model names.

        Raises:
            RuntimeError:
                If one or more required models are missing.
        """

        if required_models is None:
            required_models = (
                DEFAULT_REQUIRED_MODELS
            )

        missing_models = self.get_missing_models(
            required_models
        )

        if missing_models:
            pull_commands = "\n".join(
                f"ollama pull {model}"
                for model in missing_models
            )

            raise RuntimeError(
                "The following required Ollama models "
                f"are missing: {missing_models}\n\n"
                "Install them using:\n"
                f"{pull_commands}"
            )

        return self.get_installed_models()


def main() -> None:
    """
    Run a standalone Ollama Manager smoke test.
    """

    print("=" * 80)
    print("OLLAMA MANAGER SMOKE TEST")
    print("=" * 80)

    manager = OllamaManager(
        ollama_directory=(
            r"C:\Users\Rashi.Garg\Ollama"
        ),
        host="http://localhost:11434",
        startup_timeout_seconds=60,
    )

    try:
        print(
            f"Installation directory : "
            f"{manager.ollama_directory}"
        )

        executable_path = manager.find_executable()

        print(
            f"Ollama executable      : "
            f"{executable_path}"
        )

        was_already_running = (
            manager.is_server_running()
        )

        print(
            f"Initially running      : "
            f"{was_already_running}"
        )

        started_by_manager = (
            manager.ensure_server_running()
        )

        print(
            f"Started by Python      : "
            f"{started_by_manager}"
        )

        print(
            f"API available          : "
            f"{manager.is_server_running()}"
        )

        installed_models = (
            manager.validate_required_models()
        )

        print("\nInstalled models:")

        for model in installed_models:
            print(f"- {model}")

        print("\n" + "=" * 80)
        print("RESULT: OLLAMA MANAGER IS WORKING")
        print("=" * 80)

    except Exception as exc:
        print("\n" + "=" * 80)
        print("RESULT: OLLAMA MANAGER FAILED")
        print("=" * 80)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()