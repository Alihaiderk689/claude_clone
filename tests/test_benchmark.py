"""Unit tests for the pure/mockable parts of agent/benchmark.py.

The benchmark harness itself needs a real Ollama server (see CLAUDE.md --
`python -m agent.benchmark` is deliberately excluded from this suite). This
file covers what CAN be tested without one: _RecordingClient's Phase 9
metric capture (llm_calls, peak context size, time-to-first-token) against
a fake inner client, and the --compare report loader/differ against
hand-built JSON, consistent with the rest of the suite's mocked style.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from unittest import mock

import pytest

from agent.benchmark import TaskResult, _RecordingClient, _load_report, compare_reports


def make_updates(*dicts):
    for d in dicts:
        yield d


class FakeInnerClient:
    """Stands in for OllamaClient -- chat() just replays canned updates."""

    def __init__(self, updates_list):
        self._updates_list = updates_list
        self.calls = []

    def chat(self, messages, tools=None, cancel_event=None):
        self.calls.append(messages)
        return iter(self._updates_list)


class TestRecordingClientMetrics:
    def test_counts_llm_calls(self):
        inner = FakeInnerClient([{"content": "hi", "tool_calls": None}])
        client = _RecordingClient(inner)
        list(client.chat([{"role": "user", "content": "x"}]))
        list(client.chat([{"role": "user", "content": "y"}]))
        assert client.llm_calls == 2

    def test_tracks_peak_context_chars_across_calls(self):
        inner = FakeInnerClient([{"content": "hi", "tool_calls": None}])
        client = _RecordingClient(inner)
        list(client.chat([{"role": "user", "content": "a" * 100}]))
        assert client.peak_context_chars == 100
        list(client.chat([{"role": "user", "content": "a" * 50}]))
        assert client.peak_context_chars == 100  # smaller call doesn't lower the peak
        list(client.chat([{"role": "user", "content": "a" * 300}]))
        assert client.peak_context_chars == 300

    def test_sums_content_across_all_messages_not_just_the_last(self):
        inner = FakeInnerClient([{"content": "hi", "tool_calls": None}])
        client = _RecordingClient(inner)
        list(
            client.chat(
                [
                    {"role": "system", "content": "s" * 10},
                    {"role": "user", "content": "u" * 20},
                    {"role": "tool", "content": "t" * 30},
                ]
            )
        )
        assert client.peak_context_chars == 60

    def test_accumulates_prompt_and_completion_tokens(self):
        inner = FakeInnerClient(
            [
                {"content": "a", "tool_calls": None, "prompt_eval_count": 100, "eval_count": 10},
                {"content": "b", "tool_calls": None, "eval_count": 5},
            ]
        )
        client = _RecordingClient(inner)
        list(client.chat([{"role": "user", "content": "x"}]))
        assert client.prompt_tokens == 100
        assert client.completion_tokens == 15

    def test_records_time_to_first_token_only_once(self, monkeypatch):
        inner = FakeInnerClient([{"content": "a", "tool_calls": None}])
        client = _RecordingClient(inner)

        times = iter([100.0, 100.25, 200.0, 200.1])
        monkeypatch.setattr("agent.benchmark.time.monotonic", lambda: next(times))

        list(client.chat([{"role": "user", "content": "x"}]))
        assert client.time_to_first_token_seconds == pytest.approx(0.25)

        # A second call must NOT overwrite the first call's TTFT.
        list(client.chat([{"role": "user", "content": "y"}]))
        assert client.time_to_first_token_seconds == pytest.approx(0.25)

    def test_getattr_forwards_to_inner_client(self):
        inner = mock.Mock()
        inner.check_connection.return_value = True
        client = _RecordingClient(inner)
        assert client.check_connection() is True


class TestTaskResultDefaults:
    def test_phase9_fields_have_safe_defaults(self):
        r = TaskResult(
            name="x", success=True, duration_seconds=1.0, tool_calls=0, retries=0,
            prompt_tokens=None, completion_tokens=None,
        )
        assert r.llm_calls == 0
        assert r.peak_context_chars == 0
        assert r.estimated_peak_context_tokens == 0
        assert r.time_to_first_token_seconds is None
        assert r.peak_rss_bytes is None


class TestLoadReport:
    def test_loads_current_dict_shape(self, tmp_path):
        path = tmp_path / "report.json"
        path.write_text(json.dumps({"model": "qwen2.5-coder:3b", "host": "h", "timestamp": "t", "tasks": [{"name": "A"}]}))
        model, tasks = _load_report(path)
        assert model == "qwen2.5-coder:3b"
        assert tasks == [{"name": "A"}]

    def test_loads_pre_phase9_bare_array_shape(self, tmp_path):
        """A real baseline captured before Phase 9's JSON-shape change must
        still be usable for --compare."""
        path = tmp_path / "old_report.json"
        path.write_text(json.dumps([{"name": "A", "success": True}]))
        model, tasks = _load_report(path)
        assert model is None
        assert tasks == [{"name": "A", "success": True}]


class TestCompareReports:
    def test_compares_matching_tasks(self, tmp_path, capsys):
        before = tmp_path / "before.json"
        after = tmp_path / "after.json"
        before.write_text(json.dumps([{"name": "A", "success": False, "duration_seconds": 10.0, "tool_calls": 2, "retries": 0, "prompt_tokens": 100, "completion_tokens": 20}]))
        after.write_text(
            json.dumps(
                {
                    "model": "qwen2.5-coder:3b",
                    "tasks": [
                        {
                            "name": "A", "success": True, "duration_seconds": 8.0, "tool_calls": 2, "retries": 0,
                            "prompt_tokens": 90, "completion_tokens": 18, "llm_calls": 3,
                            "estimated_peak_context_tokens": 500, "time_to_first_token_seconds": 0.5,
                            "peak_rss_bytes": 1000,
                        }
                    ],
                }
            )
        )
        compare_reports(before, after)
        out = capsys.readouterr().out
        assert "## A" in out
        assert "success: False -> True" in out
        assert "Phase 9 metrics not available on both sides" in out

    def test_flags_new_and_missing_tasks(self, tmp_path, capsys):
        before = tmp_path / "before.json"
        after = tmp_path / "after.json"
        before.write_text(json.dumps([{"name": "Old task"}]))
        after.write_text(json.dumps({"tasks": [{"name": "New task"}]}))
        compare_reports(before, after)
        out = capsys.readouterr().out
        assert "## Old task" in out
        assert "present in before, missing from after" in out
        assert "## New task" in out
        assert "new task -- no baseline to compare against" in out
