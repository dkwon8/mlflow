"""Tests for in-memory error checking in summary.py."""

from unittest import mock

import pytest


def test_check_errors_against_files_finds_term():
    from mlflow.genai.improve.summary import _check_errors_against_files

    error_sigs = {"tool_call:model not found: 'gpt-TYPO'"}
    file_contents = [
        ("agent.py", "model = 'gpt-TYPO'\nprint('hello')"),
    ]

    resolved = _check_errors_against_files(error_sigs, file_contents)

    assert len(resolved) == 0


def test_check_errors_against_files_term_absent():
    from mlflow.genai.improve.summary import _check_errors_against_files

    error_sigs = {"tool_call:model not found: 'gpt-TYPO'"}
    file_contents = [
        ("agent.py", "model = 'gpt-5.4-mini'\nprint('hello')"),
    ]

    resolved = _check_errors_against_files(error_sigs, file_contents)

    assert error_sigs.issubset(resolved)


def test_check_errors_against_files_short_terms_skipped():
    """Terms < 4 chars are skipped. With no valid search terms, the sig is not resolved."""
    from mlflow.genai.improve.summary import _check_errors_against_files

    error_sigs = {"span:some generic error with no extractable terms"}
    file_contents = [
        ("config.py", "settings = {}"),
    ]

    resolved = _check_errors_against_files(error_sigs, file_contents)

    assert len(resolved) == 0


def test_compute_alerts_with_file_contents():
    from mlflow.genai.improve.summary import compute_alerts

    parsed_traces = [
        {
            "trace_id": "tr-001",
            "error_count": 1,
            "error_details": [
                {"span_name": "llm_call", "error_message": "model not found: 'gpt-TYPO'"}
            ],
            "user_query": "test",
            "start_ns": 0,
        },
    ]
    file_contents = [
        ("agent.py", "model = 'gpt-5.4-mini'"),
    ]

    alerts = compute_alerts(parsed_traces, file_contents=file_contents)

    assert len(alerts) == 0


def test_compute_alerts_without_files_shows_error():
    from mlflow.genai.improve.summary import compute_alerts

    parsed_traces = [
        {
            "trace_id": "tr-001",
            "error_count": 1,
            "error_details": [
                {"span_name": "llm_call", "error_message": "model not found: 'gpt-TYPO'"}
            ],
            "user_query": "test",
            "start_ns": 0,
        },
    ]

    alerts = compute_alerts(parsed_traces)

    assert len(alerts) == 1
    assert alerts[0]["trace_id"] == "tr-001"
