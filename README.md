# Multi-Agent Business Data Analytics System

A local, read-only SQLite analytics application orchestrated with LangGraph. It separates intent understanding, analytical planning, SQL generation, SQL validation/execution, analysis, reporting, and security/output validation.

## Current workflow

1. Input screening and prompt-injection checks.
2. Intent extraction and clarification routing.
3. Analytical plan creation.
4. Runtime schema discovery from the configured SQLite database.
5. SQLCoder query generation using the discovered schema.
6. Independent SQL safety/schema-policy validation and read-only execution.
7. Bounded SQL retry with validation/database feedback.
8. Evidence-based analysis and report generation.
9. Deterministic output validation, LLM evaluation, and output security guard.
10. Plotly chart rendering with a deterministic table fallback when chart inputs are incompatible.

The SQL generator rejects empty output, extracts SQL from fenced/plain model responses, and passes empty or invalid candidates through bounded retry/error handling. It sends the full discovered schema string to the SQL generation prompt; the compact-prompt fallback retains that schema rather than truncating it. Schema identifiers are checked before SQL execution.

## Requirements

- Python 3.10+
- Ollama running locally at `http://localhost:11434`
- Models pulled in Ollama: `sqlcoder:latest` and `llama3.2:latest`
- A SQLite database at `database/northwind.db` (or set `DATABASE_PATH`)

Install dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Database setup

The source archive supplied for this update did **not** contain a `.db` file. Place your Northwind SQLite database at `database/northwind.db`, or configure `DATABASE_PATH` in `.env`. The application intentionally does not fabricate a database. Confirm it is readable with:

```bash
python -m database.inspect_database
```

## Configuration

Copy `.env.example` to `.env` and update paths/secrets for your environment. Never commit `.env` or production secrets. `DATABASE_PATH` must point to a SQLite file. The database connection layer opens the database read-only.

## Run

Start Ollama and ensure the configured models are available:

```bash
ollama pull sqlcoder:latest
ollama pull llama3.2:latest
```

Start the API (terminal 1):

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Start Streamlit (terminal 2):

```bash
streamlit run app.py
```

## Tests

```bash
pytest -q
```

Tests that require Ollama or the Northwind database need those services/assets configured. Unit tests should be run first; integration tests should be run with the real database and local models.

## Important implementation boundary

The current compiled graph performs one SQL retrieval and one analysis/reporting pass for a user request. The planner can produce multiple plan steps, but this archive does not yet execute a separate SQL/result/chart cycle for every SQL-requiring plan step. This is an explicit remaining gap; do not interpret the existence of multiple plan steps as proof of a multi-query execution loop. Likewise, partial-success aggregation across independently executed analyses is not yet implemented. The existing SQL retry loop is bounded and distinct from a multi-analysis loop.

## Security

- Database reads are configured as read-only.
- SQL is parsed and checked before execution.
- Table/schema policy checks are applied.
- Input, prompt-injection, deterministic output, LLM output, and final output guards are wired into the workflow.
- Keep `.env`, database files, logs containing sensitive data, and local model credentials out of source control.
