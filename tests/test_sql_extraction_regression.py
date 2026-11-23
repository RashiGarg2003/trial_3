"""Regression tests for robust SQL extraction and schema-aware checks."""

from agents.sql_agent import SQLAgent


def _agent_without_ollama():
    # Extraction is pure and does not require an initialized Ollama client.
    return object.__new__(SQLAgent)


def test_extracts_sql_from_markdown_fence():
    agent = _agent_without_ollama()
    assert agent._extract_sql("```sql\nSELECT 1 AS value;\n```") == "SELECT 1 AS value;"


def test_extracts_sql_after_preamble():
    agent = _agent_without_ollama()
    output = "Here is the query you requested:\nSELECT 'a;b' AS value;\nThis is the result."
    assert agent._extract_sql(output) == "SELECT 'a;b' AS value;"


def test_empty_or_non_sql_output_is_empty():
    agent = _agent_without_ollama()
    assert agent._extract_sql("") == ""
    assert agent._extract_sql("I would use SELECT to retrieve rows.") == ""
    assert agent._extract_sql(None) == ""


def test_extracts_with_query():
    agent = _agent_without_ollama()
    sql = "WITH totals AS (SELECT 1 AS n) SELECT n FROM totals;"
    assert agent._extract_sql(sql) == sql
