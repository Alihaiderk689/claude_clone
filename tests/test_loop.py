"""Tests for the tool-calling agent loop in agent/loop.py.

OllamaClient.chat is mocked throughout, so these tests run without a real
Ollama server or model.
"""
from __future__ import annotations

from unittest import mock

import pytest

import threading

from agent.loop import run_agent_turn
from agent.ollama_client import OllamaCancelledError, OllamaClient, OllamaConnectionError
from agent.project import ProjectRoot
from agent.task_state import TaskState
from agent.tools import FileStateTracker, build_default_registry
from agent.tools.base import Tool, ToolResult
from agent.tools.registry import ToolRegistry


def updates(*items):
    """Helper: build a fake stream of chat() updates from simple tuples."""
    for content, tool_calls, done in items:
        yield {"content": content, "tool_calls": tool_calls, "done": done}


_CONFIRMATION_EVENT_TYPES = {
    "confirm", "confirm_file_op", "confirm_command", "confirm_git_operation", "confirm_plan",
}


def drive_agent_turn(gen, decisions=()):
    """Run a run_agent_turn generator to completion, feeding `decisions` (in
    order) into any 'confirm'/'confirm_command' events it yields via
    generator.send().
    """
    events = []
    decisions = iter(decisions)
    send_value = None
    while True:
        try:
            event = gen.send(send_value)
        except StopIteration:
            break
        events.append(event)
        send_value = next(decisions) if event["type"] in _CONFIRMATION_EVENT_TYPES else None
    return events


@pytest.fixture
def project(tmp_path):
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "auth.py").write_text("class JWTAuth:\n    pass\n")
    return ProjectRoot(tmp_path)


@pytest.fixture
def task_state():
    return TaskState()


@pytest.fixture
def registry(project, task_state):
    return build_default_registry(project, task_state=task_state)


class TestPlainAnswerNoTools:
    def test_final_answer_without_any_tool_call(self, registry):
        client = mock.create_autospec(OllamaClient, instance=True)
        client.chat.return_value = updates(
            ("Hello", None, False),
            (" there", None, True),
        )

        messages = [{"role": "user", "content": "hi"}]
        events = list(run_agent_turn(client, registry, messages))

        content_events = [e for e in events if e["type"] == "content"]
        final_events = [e for e in events if e["type"] == "final"]

        assert "".join(e["text"] for e in content_events) == "Hello there"
        assert len(final_events) == 1
        assert final_events[0]["text"] == "Hello there"
        assert messages[-1] == {"role": "assistant", "content": "Hello there"}
        client.chat.assert_called_once()


class TestToolCallThenFinalAnswer:
    def test_full_round_trip(self, registry):
        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            (
                "",
                [{"function": {"name": "read_file", "arguments": {"path": "backend/auth.py"}}}],
                True,
            ),
        )
        second_call = updates(
            ("JWT authentication is implemented in backend/auth.py.", None, True),
        )
        client.chat.side_effect = [first_call, second_call]

        messages = [{"role": "user", "content": "Where is auth implemented?"}]
        events = list(run_agent_turn(client, registry, messages))

        tool_call_events = [e for e in events if e["type"] == "tool_call"]
        final_events = [e for e in events if e["type"] == "final"]

        assert len(tool_call_events) == 1
        assert tool_call_events[0]["name"] == "read_file"
        assert "read_file" in tool_call_events[0]["display"]

        assert len(final_events) == 1
        assert "backend/auth.py" in final_events[0]["text"]

        # History should contain: user, assistant(tool_calls), tool, assistant(final)
        roles = [m["role"] for m in messages]
        assert roles == ["user", "assistant", "tool", "assistant"]
        assert messages[2]["role"] == "tool"
        assert messages[2]["tool_name"] == "read_file"
        assert "JWTAuth" in messages[2]["content"]

        assert client.chat.call_count == 2

    def test_multiple_tool_calls_in_a_single_response(self, registry):
        """Ollama can return several tool_calls in one message (parallel
        calls); the loop must execute all of them and feed back a separate
        tool result for each before asking the model again."""
        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            (
                "",
                [
                    {"function": {"name": "list_files", "arguments": {"path": "."}}},
                    {"function": {"name": "read_file", "arguments": {"path": "backend/auth.py"}}},
                ],
                True,
            ),
        )
        second_call = updates(("Auth lives in backend/auth.py.", None, True))
        client.chat.side_effect = [first_call, second_call]

        messages = [{"role": "user", "content": "where's auth?"}]
        events = list(run_agent_turn(client, registry, messages))

        tool_call_events = [e for e in events if e["type"] == "tool_call"]
        assert [e["name"] for e in tool_call_events] == ["list_files", "read_file"]

        # One assistant(tool_calls) message, then one "tool" message per call.
        roles = [m["role"] for m in messages]
        assert roles == ["user", "assistant", "tool", "tool", "assistant"]
        assert messages[2]["tool_name"] == "list_files"
        assert messages[3]["tool_name"] == "read_file"
        assert "JWTAuth" in messages[3]["content"]

        # Only two round trips to the model: one to get both calls, one for the final answer.
        assert client.chat.call_count == 2

    def test_tool_arguments_as_json_string_are_parsed(self, registry):
        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            (
                "",
                [{"function": {"name": "read_file", "arguments": '{"path": "backend/auth.py"}'}}],
                True,
            ),
        )
        second_call = updates(("Done.", None, True))
        client.chat.side_effect = [first_call, second_call]

        messages = [{"role": "user", "content": "check auth.py"}]
        events = list(run_agent_turn(client, registry, messages))

        tool_call_events = [e for e in events if e["type"] == "tool_call"]
        assert tool_call_events[0]["args"] == {"path": "backend/auth.py"}

    def test_unknown_tool_produces_tool_error_but_continues(self, registry):
        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            ("", [{"function": {"name": "totally_unknown_tool", "arguments": {"path": "x"}}}], True),
        )
        second_call = updates(("I can't do that yet.", None, True))
        client.chat.side_effect = [first_call, second_call]

        messages = [{"role": "user", "content": "do something unsupported"}]
        events = list(run_agent_turn(client, registry, messages))

        tool_error_events = [e for e in events if e["type"] == "tool_error"]
        assert len(tool_error_events) == 1
        assert "unknown tool" in tool_error_events[0]["message"].lower()

        final_events = [e for e in events if e["type"] == "final"]
        assert final_events[0]["text"] == "I can't do that yet."


class TestReadFileCacheIntegration:
    """Phase 9: a real second read_file of the same unchanged path within a
    turn short-circuits via the registry's cache-hit path, and the message
    appended to history carries the cache_hit marker compact_messages relies
    on -- see TestCompactMessagesCacheHitAware in test_context_manager.py
    for the compaction-side half of this."""

    def test_second_read_in_the_same_turn_is_a_cache_hit(self, registry, task_state):
        client = mock.create_autospec(OllamaClient, instance=True)
        read_call = [{"function": {"name": "read_file", "arguments": {"path": "backend/auth.py"}}}]
        client.chat.side_effect = [
            updates(("", read_call, True)),
            updates(("", read_call, True)),
            updates(("Done.", None, True)),
        ]

        messages = [{"role": "user", "content": "read auth.py twice"}]
        events = list(run_agent_turn(client, registry, messages, task_state=task_state))

        tool_messages = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_messages) == 2
        assert "class JWTAuth" in tool_messages[0]["content"]
        assert tool_messages[0].get("cache_hit") is None
        assert "status=unchanged" in tool_messages[1]["content"]
        assert tool_messages[1].get("cache_hit") is True

        tool_call_events = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_call_events) == 2


class TestMaxContextCharsOverride:
    """Phase 9: max_context_chars is threaded straight through to
    context_manager.compact_messages() every iteration -- confirm an
    override actually changes trimming behavior, not just that the default
    leaves things unchanged (already covered by every other test here)."""

    def test_small_override_trims_old_tool_messages_the_default_would_not(self, registry, task_state):
        from agent.context_manager import _TRIMMED_PLACEHOLDER

        client = mock.create_autospec(OllamaClient, instance=True)
        client.chat.side_effect = [updates(("Done.", None, True))]

        # 8 pre-existing large tool messages, older than keep_recent's
        # protected window (6) -- compact_messages runs at the top of the
        # very first iteration, before any chat() call, so this needs no
        # tool_calls from the mocked model at all.
        messages = [{"role": "system", "content": "Base."}]
        for i in range(8):
            messages.append(
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "list_files", "arguments": {"path": f"dir{i}"}}}],
                }
            )
            messages.append({"role": "tool", "tool_name": "list_files", "content": "x" * 500})
        messages.append({"role": "user", "content": "what's next?"})
        # 8 * 500 = 4,000 chars total -- comfortably under the real default
        # budget (12,000), so the default would leave every message alone.

        list(run_agent_turn(client, registry, messages, task_state=task_state, max_context_chars=10))

        tool_messages = [m for m in messages if m.get("role") == "tool"]
        assert any(m["content"] == _TRIMMED_PLACEHOLDER for m in tool_messages)


class TestFallbackToolCallParsing:
    """Ollama doesn't always lift qwen2.5-coder's tool call into the
    structured `tool_calls` response field -- it can arrive as plain
    content instead, either as bare JSON or wrapped in the model's own
    <tool_call> tags (confirmed via `ollama show qwen2.5-coder:7b
    --template` and reproduced live against a real local model). These
    tests cover the fallback parser that recognizes that same
    model-documented shape instead of treating it as prose.
    """

    def test_bare_json_content_is_treated_as_a_tool_call(self, registry):
        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            ('{"name": "list_files", "arguments": {"path": "."}}', None, True),
        )
        second_call = updates(("Here's the project layout.", None, True))
        client.chat.side_effect = [first_call, second_call]

        messages = [{"role": "user", "content": "what's in this repo?"}]
        events = list(run_agent_turn(client, registry, messages))

        tool_call_events = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_call_events) == 1
        assert tool_call_events[0]["name"] == "list_files"
        assert tool_call_events[0]["args"] == {"path": "."}

        # The raw JSON itself must never be shown to the user as if it were
        # a chat message.
        content_events = [e for e in events if e["type"] == "content"]
        assert all("list_files" not in e["text"] for e in content_events)

    def test_tagged_tool_call_content_is_parsed(self, registry):
        client = mock.create_autospec(OllamaClient, instance=True)
        tagged = (
            '<tool_call>\n{"name": "read_file", "arguments": {"path": "backend/auth.py"}}\n'
            "</tool_call>"
        )
        first_call = updates((tagged, None, True))
        second_call = updates(("Found it.", None, True))
        client.chat.side_effect = [first_call, second_call]

        messages = [{"role": "user", "content": "check auth.py"}]
        events = list(run_agent_turn(client, registry, messages))

        tool_call_events = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_call_events) == 1
        assert tool_call_events[0]["name"] == "read_file"
        assert tool_call_events[0]["args"] == {"path": "backend/auth.py"}

    def test_multiple_json_lines_in_one_tagged_block(self, registry):
        client = mock.create_autospec(OllamaClient, instance=True)
        tagged = (
            "<tool_call>\n"
            '{"name": "list_files", "arguments": {"path": "."}}\n'
            '{"name": "read_file", "arguments": {"path": "backend/auth.py"}}\n'
            "</tool_call>"
        )
        first_call = updates((tagged, None, True))
        second_call = updates(("Done.", None, True))
        client.chat.side_effect = [first_call, second_call]

        messages = [{"role": "user", "content": "inspect the project"}]
        events = list(run_agent_turn(client, registry, messages))

        tool_call_events = [e for e in events if e["type"] == "tool_call"]
        assert [e["name"] for e in tool_call_events] == ["list_files", "read_file"]

    def test_content_that_merely_starts_with_brace_but_isnt_a_tool_call(self, registry):
        """A real answer that happens to open with '{' (e.g. a JSON example)
        must still reach the user, just without live per-chunk streaming."""
        client = mock.create_autospec(OllamaClient, instance=True)
        client.chat.return_value = updates(
            ('{"example": "config"} is a minimal JSON config.', None, True),
        )

        messages = [{"role": "user", "content": "show me a JSON example"}]
        events = list(run_agent_turn(client, registry, messages))

        assert not any(e["type"] == "tool_call" for e in events)
        final_events = [e for e in events if e["type"] == "final"]
        assert final_events[0]["text"] == '{"example": "config"} is a minimal JSON config.'
        content_events = [e for e in events if e["type"] == "content"]
        assert "".join(e["text"] for e in content_events) == final_events[0]["text"]

    def test_reasoning_prose_followed_by_untagged_json_is_still_caught(self, registry):
        """Observed live against a real local qwen2.5-coder:7b: the model
        sometimes narrates its plan in prose, then appends the tool-call
        JSON at the end with no <tool_call> tag at all. The raw JSON must
        never reach the user as if it were the answer."""
        client = mock.create_autospec(OllamaClient, instance=True)
        narrated = (
            "To find this, I should search the codebase for relevant keywords.\n\n"
            '{"name": "search_files", "arguments": {"query": "JWT"}}'
        )
        first_call = updates((narrated, None, True))
        second_call = updates(("JWT auth is in backend/accounts/views.py.", None, True))
        client.chat.side_effect = [first_call, second_call]

        messages = [{"role": "user", "content": "where's JWT auth?"}]
        events = list(run_agent_turn(client, registry, messages))

        tool_call_events = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_call_events) == 1
        assert tool_call_events[0]["name"] == "search_files"
        assert tool_call_events[0]["args"] == {"query": "JWT"}

        content_events = [e for e in events if e["type"] == "content"]
        assert all("search_files" not in e["text"] for e in content_events)


class TestSafetyLimits:
    def test_stops_after_max_iterations(self, registry):
        client = mock.create_autospec(OllamaClient, instance=True)

        def infinite_tool_calls(*_args, **_kwargs):
            return updates(
                ("", [{"function": {"name": "list_files", "arguments": {"path": "."}}}], True),
            )

        client.chat.side_effect = infinite_tool_calls

        messages = [{"role": "user", "content": "loop forever"}]
        events = list(run_agent_turn(client, registry, messages, max_iterations=3))

        assert client.chat.call_count == 3
        assert events[-1]["type"] == "max_iterations"
        assert not any(e["type"] == "final" for e in events)

    def test_connection_error_retries_then_yields_error_event_and_stops(self, registry, monkeypatch):
        """A connection error is transient -- Phase 8 retries it
        MAX_OLLAMA_RETRIES times (see loop.py's retry policy) before giving
        up, rather than failing on the very first attempt as earlier phases
        did. Zero out the backoff so the test doesn't actually sleep."""
        import agent.loop as loop_module

        monkeypatch.setattr(loop_module, "OLLAMA_RETRY_BACKOFF_SECONDS", (0, 0))

        client = mock.create_autospec(OllamaClient, instance=True)

        def raise_connection_error(*_args, **_kwargs):
            raise OllamaConnectionError("Could not connect to Ollama.")
            yield  # pragma: no cover - makes this a generator function

        client.chat.side_effect = raise_connection_error

        messages = [{"role": "user", "content": "hi"}]
        events = list(run_agent_turn(client, registry, messages))

        retry_events = [e for e in events if e["type"] == "retry"]
        assert len(retry_events) == loop_module.MAX_OLLAMA_RETRIES
        assert [e["attempt"] for e in retry_events] == [1, 2]
        assert all(e["max_attempts"] == loop_module.MAX_OLLAMA_RETRIES + 1 for e in retry_events)

        assert events[-1]["type"] == "error"
        assert "could not connect" in events[-1]["message"].lower()
        assert client.chat.call_count == loop_module.MAX_OLLAMA_RETRIES + 1

    def test_connection_error_succeeds_after_one_retry(self, registry, monkeypatch):
        import agent.loop as loop_module

        monkeypatch.setattr(loop_module, "OLLAMA_RETRY_BACKOFF_SECONDS", (0, 0))

        client = mock.create_autospec(OllamaClient, instance=True)
        attempts = {"n": 0}

        def flaky_then_ok(*_args, **_kwargs):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise OllamaConnectionError("Could not connect to Ollama.")
            return updates(("All good.", None, True))

        client.chat.side_effect = flaky_then_ok

        messages = [{"role": "user", "content": "hi"}]
        events = list(run_agent_turn(client, registry, messages))

        assert [e["type"] for e in events if e["type"] in ("retry", "final")] == ["retry", "final"]
        assert events[-1]["text"] == "All good."
        assert client.chat.call_count == 2

    def test_timeout_error_is_also_retried(self, registry, monkeypatch):
        import agent.loop as loop_module
        from agent.ollama_client import OllamaTimeoutError

        monkeypatch.setattr(loop_module, "OLLAMA_RETRY_BACKOFF_SECONDS", (0, 0))

        client = mock.create_autospec(OllamaClient, instance=True)

        def raise_timeout(*_args, **_kwargs):
            raise OllamaTimeoutError("Connection to Ollama timed out.")
            yield  # pragma: no cover

        client.chat.side_effect = raise_timeout

        messages = [{"role": "user", "content": "hi"}]
        events = list(run_agent_turn(client, registry, messages))

        assert any(e["type"] == "retry" for e in events)
        assert events[-1]["type"] == "error"

    def test_model_not_found_is_not_retried(self, registry):
        """Retrying an identical request after a model-not-found or a
        malformed-request-shaped API error would just fail identically --
        these fail immediately, unlike connection/timeout errors."""
        from agent.ollama_client import OllamaModelNotFoundError

        client = mock.create_autospec(OllamaClient, instance=True)

        def raise_model_not_found(*_args, **_kwargs):
            raise OllamaModelNotFoundError("Model 'x' was not found.")
            yield  # pragma: no cover

        client.chat.side_effect = raise_model_not_found

        messages = [{"role": "user", "content": "hi"}]
        events = list(run_agent_turn(client, registry, messages))

        assert events == [{"type": "error", "message": "Model 'x' was not found."}]
        assert client.chat.call_count == 1

    def test_api_error_is_not_retried(self, registry):
        from agent.ollama_client import OllamaAPIError

        client = mock.create_autospec(OllamaClient, instance=True)

        def raise_api_error(*_args, **_kwargs):
            raise OllamaAPIError("Ollama returned an error: bad request")
            yield  # pragma: no cover

        client.chat.side_effect = raise_api_error

        messages = [{"role": "user", "content": "hi"}]
        events = list(run_agent_turn(client, registry, messages))

        assert events[-1]["type"] == "error"
        assert client.chat.call_count == 1

    def test_cancellation_during_retry_backoff_stops_cleanly(self, registry, monkeypatch):
        """Stop Task must remain responsive even while the loop is asleep
        between retry attempts, not just between whole model calls."""
        import threading

        import agent.loop as loop_module

        monkeypatch.setattr(loop_module, "OLLAMA_RETRY_BACKOFF_SECONDS", (0.05, 0.05))

        client = mock.create_autospec(OllamaClient, instance=True)

        def raise_connection_error(*_args, **_kwargs):
            raise OllamaConnectionError("Could not connect to Ollama.")
            yield  # pragma: no cover

        client.chat.side_effect = raise_connection_error

        cancel_event = threading.Event()

        def cancel_soon():
            cancel_event.set()

        # Trigger cancellation from inside the fake sleep so it lands
        # squarely inside the retry backoff window.
        real_sleep = loop_module._interruptible_sleep

        def sleep_and_cancel(seconds, event):
            cancel_soon()
            return real_sleep(seconds, event)

        monkeypatch.setattr(loop_module, "_interruptible_sleep", sleep_and_cancel)

        messages = [{"role": "user", "content": "hi"}]
        events = list(run_agent_turn(client, registry, messages, cancel_event=cancel_event))

        assert events[-1] == {"type": "cancelled"}


class TestReadOnlyToolsCannotEscapeProjectRoot:
    def test_read_file_tool_call_outside_root_is_rejected(self, registry):
        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            (
                "",
                [{"function": {"name": "read_file", "arguments": {"path": "../../etc/passwd"}}}],
                True,
            ),
        )
        second_call = updates(("I can't read that.", None, True))
        client.chat.side_effect = [first_call, second_call]

        messages = [{"role": "user", "content": "read /etc/passwd"}]
        events = list(run_agent_turn(client, registry, messages))

        tool_error_events = [e for e in events if e["type"] == "tool_error"]
        assert len(tool_error_events) == 1
        assert "access denied" in tool_error_events[0]["message"].lower()


class TestApprovalFlow:
    """The core Phase 3 guarantee: the model proposes an edit/write, the
    loop pauses with a 'confirm' event, and nothing touches disk until the
    caller sends back an explicit True.
    """

    def test_approved_edit_is_applied(self, project, registry):
        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            (
                "",
                [
                    {
                        "function": {
                            "name": "edit_file",
                            "arguments": {
                                "path": "backend/auth.py",
                                "old_text": "pass",
                                "new_text": "pass  # updated",
                            },
                        }
                    }
                ],
                True,
            ),
        )
        second_call = updates(("Done.", None, True))
        client.chat.side_effect = [first_call, second_call]

        messages = [{"role": "user", "content": "tweak auth.py"}]
        gen = run_agent_turn(client, registry, messages)
        events = drive_agent_turn(gen, decisions=[True])

        confirm_events = [e for e in events if e["type"] == "confirm"]
        applied_events = [e for e in events if e["type"] == "change_applied"]
        assert len(confirm_events) == 1
        assert confirm_events[0]["change"].kind == "edit"
        assert len(applied_events) == 1

        on_disk = (project.root / "backend" / "auth.py").read_text()
        assert "pass  # updated" in on_disk

        tool_messages = [m for m in messages if m["role"] == "tool"]
        assert "successfully" in tool_messages[0]["content"].lower()

    def test_rejected_edit_is_not_applied(self, project, registry):
        client = mock.create_autospec(OllamaClient, instance=True)
        original = (project.root / "backend" / "auth.py").read_text()
        first_call = updates(
            (
                "",
                [
                    {
                        "function": {
                            "name": "edit_file",
                            "arguments": {
                                "path": "backend/auth.py",
                                "old_text": "pass",
                                "new_text": "pass  # updated",
                            },
                        }
                    }
                ],
                True,
            ),
        )
        second_call = updates(("Okay, leaving it as-is.", None, True))
        client.chat.side_effect = [first_call, second_call]

        messages = [{"role": "user", "content": "tweak auth.py"}]
        gen = run_agent_turn(client, registry, messages)
        events = drive_agent_turn(gen, decisions=[False])

        rejected_events = [e for e in events if e["type"] == "change_rejected"]
        applied_events = [e for e in events if e["type"] == "change_applied"]
        assert len(rejected_events) == 1
        assert not applied_events

        assert (project.root / "backend" / "auth.py").read_text() == original

        tool_messages = [m for m in messages if m["role"] == "tool"]
        assert tool_messages[0]["content"] == "User rejected this change. The file was not modified."

    def test_approved_write_creates_file(self, project, registry):
        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            (
                "",
                [
                    {
                        "function": {
                            "name": "write_file",
                            "arguments": {"path": "backend/config.py", "content": "DEBUG = True\n"},
                        }
                    }
                ],
                True,
            ),
        )
        second_call = updates(("Created it.", None, True))
        client.chat.side_effect = [first_call, second_call]

        messages = [{"role": "user", "content": "add a config file"}]
        gen = run_agent_turn(client, registry, messages)
        events = drive_agent_turn(gen, decisions=[True])

        assert any(e["type"] == "change_applied" for e in events)
        assert (project.root / "backend" / "config.py").read_text() == "DEBUG = True\n"

    def test_two_proposed_edits_can_be_approved_and_rejected_independently(self, project, registry):
        (project.root / "backend" / "other.py").write_text("value = 1\n")
        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            (
                "",
                [
                    {
                        "function": {
                            "name": "edit_file",
                            "arguments": {
                                "path": "backend/auth.py",
                                "old_text": "pass",
                                "new_text": "pass  # a",
                            },
                        }
                    },
                    {
                        "function": {
                            "name": "edit_file",
                            "arguments": {
                                "path": "backend/other.py",
                                "old_text": "value = 1",
                                "new_text": "value = 2",
                            },
                        }
                    },
                ],
                True,
            ),
        )
        second_call = updates(("Applied one, skipped the other.", None, True))
        client.chat.side_effect = [first_call, second_call]

        messages = [{"role": "user", "content": "update both files"}]
        gen = run_agent_turn(client, registry, messages)
        # Approve the first proposed change, reject the second.
        events = drive_agent_turn(gen, decisions=[True, False])

        assert (project.root / "backend" / "auth.py").read_text().find("pass  # a") != -1
        assert (project.root / "backend" / "other.py").read_text() == "value = 1\n"  # unchanged

        applied = [e for e in events if e["type"] == "change_applied"]
        rejected = [e for e in events if e["type"] == "change_rejected"]
        assert len(applied) == 1
        assert len(rejected) == 1

    def test_stale_edit_is_refused_before_any_confirm_prompt(self, project):
        tracker = FileStateTracker()
        registry = build_default_registry(project, tracker)
        target = project.root / "backend" / "auth.py"
        # The model "read" the file earlier, but it changed on disk since.
        tracker.record(target, "class JWTAuth:\n    pass  # totally different\n")

        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            (
                "",
                [
                    {
                        "function": {
                            "name": "edit_file",
                            "arguments": {
                                "path": "backend/auth.py",
                                "old_text": "pass",
                                "new_text": "pass  # updated",
                            },
                        }
                    }
                ],
                True,
            ),
        )
        second_call = updates(("Let me re-read the file.", None, True))
        client.chat.side_effect = [first_call, second_call]

        messages = [{"role": "user", "content": "tweak auth.py"}]
        gen = run_agent_turn(client, registry, messages, tracker=tracker)
        events = drive_agent_turn(gen)  # no confirm decisions should be needed

        assert not any(e["type"] == "confirm" for e in events)
        tool_error_events = [e for e in events if e["type"] == "tool_error"]
        assert len(tool_error_events) == 1
        assert "changed on disk" in tool_error_events[0]["message"].lower()

    def test_auto_approve_edits_applies_without_the_caller_answering(self, project, registry):
        """The core safety property of Auto mode: the approval decision is
        resolved *inside* run_agent_turn, not by whatever the caller sends
        back. This driver deliberately never provides a real decision
        (always sends None back) to prove the edit still applies."""
        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            (
                "",
                [
                    {
                        "function": {
                            "name": "edit_file",
                            "arguments": {
                                "path": "backend/auth.py",
                                "old_text": "pass",
                                "new_text": "pass  # auto-approved",
                            },
                        }
                    }
                ],
                True,
            ),
        )
        second_call = updates(("Done.", None, True))
        client.chat.side_effect = [first_call, second_call]

        messages = [{"role": "user", "content": "tweak auth.py"}]
        gen = run_agent_turn(client, registry, messages, auto_approve_edits=True)

        events = []
        send_value = None
        while True:
            try:
                event = gen.send(send_value)
            except StopIteration:
                break
            events.append(event)
            send_value = None  # never answer, even for a confirm-shaped event

        confirm_events = [e for e in events if e["type"] == "confirm"]
        applied_events = [e for e in events if e["type"] == "change_applied"]
        assert len(confirm_events) == 1
        assert confirm_events[0]["auto_approved"] is True
        assert len(applied_events) == 1

        on_disk = (project.root / "backend" / "auth.py").read_text()
        assert "pass  # auto-approved" in on_disk

    def test_manual_mode_confirm_event_has_no_auto_approved_field(self, project, registry):
        """Default (auto_approve_edits=False) must be byte-for-byte the
        existing event shape -- no stray `auto_approved` key appearing."""
        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            (
                "",
                [
                    {
                        "function": {
                            "name": "edit_file",
                            "arguments": {"path": "backend/auth.py", "old_text": "pass", "new_text": "pass  # x"},
                        }
                    }
                ],
                True,
            ),
        )
        second_call = updates(("Done.", None, True))
        client.chat.side_effect = [first_call, second_call]

        messages = [{"role": "user", "content": "tweak auth.py"}]
        gen = run_agent_turn(client, registry, messages)
        events = drive_agent_turn(gen, decisions=[True])

        confirm_event = next(e for e in events if e["type"] == "confirm")
        assert "auto_approved" not in confirm_event


def _fake_terminal_proc(returncode=0, stdout="", stderr="", pid=99999):
    """Stand-in for subprocess.Popen used by agent/tools/terminal.py's
    execute_command -- .communicate() returns immediately, no real process."""
    proc = mock.Mock()
    proc.pid = pid
    proc.returncode = returncode
    proc.communicate.return_value = (stdout, stderr)
    return proc


class TestRunCommandApprovalFlow:
    """Phase 4's core guarantee, same shape as Phase 3's: the model proposes
    a command, the loop pauses with a 'confirm_command' event, and nothing
    executes until the caller sends back an explicit True. subprocess.Popen
    is always mocked -- no real process runs in these tests (except the
    dedicated real-subprocess tests in test_tools_terminal.py).
    """

    def test_approved_command_executes(self, registry):
        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            ("", [{"function": {"name": "run_command", "arguments": {"program": "pytest", "args": []}}}], True),
        )
        second_call = updates(("Tests passed.", None, True))
        client.chat.side_effect = [first_call, second_call]

        fake_proc = _fake_terminal_proc(returncode=0, stdout="2 passed\n", stderr="")
        with mock.patch("agent.tools.terminal.subprocess.Popen", return_value=fake_proc) as mock_popen:
            messages = [{"role": "user", "content": "run the tests"}]
            gen = run_agent_turn(client, registry, messages)
            events = drive_agent_turn(gen, decisions=[True])

        mock_popen.assert_called_once()
        confirm_events = [e for e in events if e["type"] == "confirm_command"]
        result_events = [e for e in events if e["type"] == "command_result"]
        assert len(confirm_events) == 1
        assert confirm_events[0]["command"].program == "pytest"
        assert len(result_events) == 1
        assert result_events[0]["result"].exit_code == 0

        tool_messages = [m for m in messages if m["role"] == "tool"]
        assert "2 passed" in tool_messages[0]["content"]
        assert "exit code: 0" in tool_messages[0]["content"]

    def test_rejected_command_never_executes(self, registry):
        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            ("", [{"function": {"name": "run_command", "arguments": {"program": "pytest", "args": []}}}], True),
        )
        second_call = updates(("Okay, not running it.", None, True))
        client.chat.side_effect = [first_call, second_call]

        with mock.patch("agent.tools.terminal.subprocess.Popen") as mock_run:
            messages = [{"role": "user", "content": "run the tests"}]
            gen = run_agent_turn(client, registry, messages)
            events = drive_agent_turn(gen, decisions=[False])

        mock_run.assert_not_called()
        rejected_events = [e for e in events if e["type"] == "command_rejected"]
        assert len(rejected_events) == 1

        tool_messages = [m for m in messages if m["role"] == "tool"]
        assert tool_messages[0]["content"] == "Command was not executed. The user rejected running it."

    def test_disallowed_command_never_reaches_confirm(self, registry):
        """A policy-rejected command must never even ask for approval --
        it's blocked in Python before the user is involved at all."""
        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            (
                "",
                [{"function": {"name": "run_command", "arguments": {"program": "rm", "args": ["-rf", "."]}}}],
                True,
            ),
        )
        second_call = updates(("I can't run that.", None, True))
        client.chat.side_effect = [first_call, second_call]

        with mock.patch("agent.tools.terminal.subprocess.Popen") as mock_run:
            messages = [{"role": "user", "content": "delete everything"}]
            gen = run_agent_turn(client, registry, messages)
            events = drive_agent_turn(gen)

        mock_run.assert_not_called()
        assert not any(e["type"] == "confirm_command" for e in events)
        tool_error_events = [e for e in events if e["type"] == "tool_error"]
        assert len(tool_error_events) == 1
        assert "not allowed" in tool_error_events[0]["message"].lower()

    def test_shell_injection_attempt_never_reaches_confirm(self, registry):
        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            (
                "",
                [
                    {
                        "function": {
                            "name": "run_command",
                            "arguments": {"program": "pytest", "args": [";", "rm", "-rf", "."]},
                        }
                    }
                ],
                True,
            ),
        )
        second_call = updates(("That's not allowed.", None, True))
        client.chat.side_effect = [first_call, second_call]

        with mock.patch("agent.tools.terminal.subprocess.Popen") as mock_run:
            messages = [{"role": "user", "content": "run pytest; rm -rf ."}]
            gen = run_agent_turn(client, registry, messages)
            events = drive_agent_turn(gen)

        mock_run.assert_not_called()
        assert not any(e["type"] == "confirm_command" for e in events)

    def test_timed_out_command_reports_tool_error_too(self, registry):
        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            (
                "",
                [{"function": {"name": "run_command", "arguments": {"program": "pytest", "args": [], "timeout": 5}}}],
                True,
            ),
        )
        second_call = updates(("It timed out.", None, True))
        client.chat.side_effect = [first_call, second_call]

        # Deterministically simulate a hung process: the first
        # remaining-time check already reads as expired (no real waiting).
        fake_proc = _fake_terminal_proc(returncode=None)
        fake_proc.communicate.side_effect = [("partial", "")]
        monotonic_values = iter([0.0, 5.0])
        with mock.patch(
            "agent.tools.terminal.subprocess.Popen", return_value=fake_proc
        ), mock.patch(
            "agent.tools.terminal.time.monotonic", side_effect=lambda: next(monotonic_values)
        ), mock.patch(
            "agent.tools.terminal.os.getpgid", return_value=4242
        ), mock.patch("agent.tools.terminal.os.killpg"), mock.patch.object(
            fake_proc, "wait", return_value=None
        ):
            messages = [{"role": "user", "content": "run the tests"}]
            gen = run_agent_turn(client, registry, messages)
            events = drive_agent_turn(gen, decisions=[True])

        result_events = [e for e in events if e["type"] == "command_result"]
        tool_error_events = [e for e in events if e["type"] == "tool_error"]
        assert result_events[0]["result"].timed_out
        assert len(tool_error_events) == 1
        assert "timed out" in tool_error_events[0]["message"].lower()

    def test_multiple_run_command_calls_confirmed_independently(self, registry):
        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            (
                "",
                [
                    {"function": {"name": "run_command", "arguments": {"program": "pytest", "args": []}}},
                    {"function": {"name": "run_command", "arguments": {"program": "ruff", "args": ["check", "."]}}},
                ],
                True,
            ),
        )
        second_call = updates(("Done.", None, True))
        client.chat.side_effect = [first_call, second_call]

        fake_proc = _fake_terminal_proc(returncode=0, stdout="ok\n", stderr="")
        with mock.patch("agent.tools.terminal.subprocess.Popen", return_value=fake_proc) as mock_popen:
            messages = [{"role": "user", "content": "run tests and lint"}]
            gen = run_agent_turn(client, registry, messages)
            events = drive_agent_turn(gen, decisions=[True, False])  # approve pytest, reject ruff

        assert mock_popen.call_count == 1  # only the approved one actually ran
        applied = [e for e in events if e["type"] == "command_result"]
        rejected = [e for e in events if e["type"] == "command_rejected"]
        assert len(applied) == 1
        assert len(rejected) == 1

    def test_auto_approve_edits_does_not_auto_approve_commands(self, registry):
        """Auto mode's scope is edits/plans only -- a command must still
        pause for an explicit decision even with auto_approve_edits=True,
        and an unanswered/rejected one must not run."""
        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            ("", [{"function": {"name": "run_command", "arguments": {"program": "pytest", "args": []}}}], True),
        )
        second_call = updates(("Okay, not running it.", None, True))
        client.chat.side_effect = [first_call, second_call]

        with mock.patch("agent.tools.terminal.subprocess.Popen") as mock_popen:
            messages = [{"role": "user", "content": "run the tests"}]
            gen = run_agent_turn(client, registry, messages, auto_approve_edits=True)
            events = drive_agent_turn(gen, decisions=[False])

        mock_popen.assert_not_called()
        confirm_events = [e for e in events if e["type"] == "confirm_command"]
        assert len(confirm_events) == 1
        assert "auto_approved" not in confirm_events[0]
        assert any(e["type"] == "command_rejected" for e in events)


def _run_git(repo_root, *args):
    import subprocess

    subprocess.run(["git", *args], cwd=str(repo_root), check=True, capture_output=True)


@pytest.fixture
def git_repo(tmp_path):
    _run_git(tmp_path, "init", "-q")
    _run_git(tmp_path, "config", "user.email", "test@example.com")
    _run_git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "README.md").write_text("# Test repo\n")
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "auth.py").write_text("def login():\n    pass\n")
    _run_git(tmp_path, "add", "README.md", "backend/auth.py")
    _run_git(tmp_path, "commit", "-q", "-m", "Initial commit")
    return tmp_path


@pytest.fixture
def git_project(git_repo):
    return ProjectRoot(git_repo)


@pytest.fixture
def git_registry(git_project):
    return build_default_registry(git_project)


class TestGitApprovalFlow:
    """Phase 5's core guarantee, same shape as Phase 3/4's: the model
    proposes a Git operation, the loop pauses with a 'confirm_git_operation'
    event, and nothing runs until the caller sends back an explicit True.
    Uses a real temporary Git repository -- these tools shell out to real
    git, so mocking subprocess would just test the mock, not the parsing.
    """

    def test_approved_branch_creation(self, git_repo, git_registry):
        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            (
                "",
                [{"function": {"name": "git_create_branch", "arguments": {"name": "feature/x"}}}],
                True,
            ),
        )
        second_call = updates(("Branch created.", None, True))
        client.chat.side_effect = [first_call, second_call]

        messages = [{"role": "user", "content": "create a branch"}]
        gen = run_agent_turn(client, git_registry, messages)
        events = drive_agent_turn(gen, decisions=[True])

        confirm_events = [e for e in events if e["type"] == "confirm_git_operation"]
        applied_events = [e for e in events if e["type"] == "git_operation_applied"]
        assert len(confirm_events) == 1
        assert confirm_events[0]["operation"].branch_name == "feature/x"
        assert len(applied_events) == 1

        import subprocess

        branches = subprocess.run(
            ["git", "branch", "--list", "feature/x"], cwd=str(git_repo), capture_output=True, text=True
        ).stdout
        assert "feature/x" in branches

    def test_rejected_branch_creation_never_runs(self, git_repo, git_registry):
        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            (
                "",
                [{"function": {"name": "git_create_branch", "arguments": {"name": "feature/x"}}}],
                True,
            ),
        )
        second_call = updates(("Okay, not creating it.", None, True))
        client.chat.side_effect = [first_call, second_call]

        messages = [{"role": "user", "content": "create a branch"}]
        gen = run_agent_turn(client, git_registry, messages)
        events = drive_agent_turn(gen, decisions=[False])

        rejected_events = [e for e in events if e["type"] == "git_operation_rejected"]
        assert len(rejected_events) == 1

        import subprocess

        branches = subprocess.run(
            ["git", "branch", "--list", "feature/x"], cwd=str(git_repo), capture_output=True, text=True
        ).stdout
        assert "feature/x" not in branches

        tool_messages = [m for m in messages if m["role"] == "tool"]
        assert tool_messages[0]["content"] == "User rejected this Git operation. Nothing was changed."

    def test_approved_stage_and_commit_full_workflow(self, git_repo, git_registry):
        (git_repo / "README.md").write_text("v2\n")

        client = mock.create_autospec(OllamaClient, instance=True)
        stage_call = updates(
            ("", [{"function": {"name": "git_stage", "arguments": {"paths": ["README.md"]}}}], True),
        )
        commit_call = updates(
            (
                "",
                [{"function": {"name": "git_commit", "arguments": {"message": "Update README"}}}],
                True,
            ),
        )
        final_call = updates(("Committed.", None, True))
        client.chat.side_effect = [stage_call, commit_call, final_call]

        messages = [{"role": "user", "content": "stage and commit the README change"}]
        gen = run_agent_turn(client, git_registry, messages)
        events = drive_agent_turn(gen, decisions=[True, True])

        applied = [e for e in events if e["type"] == "git_operation_applied"]
        assert len(applied) == 2

        import subprocess

        log = subprocess.run(
            ["git", "log", "-1", "--pretty=%s"], cwd=str(git_repo), capture_output=True, text=True
        ).stdout.strip()
        assert log == "Update README"

    def test_stage_flags_sensitive_file_but_still_requires_approval(self, git_repo, git_registry):
        (git_repo / ".env").write_text("SECRET=1\n")

        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            ("", [{"function": {"name": "git_stage", "arguments": {"paths": [".env"]}}}], True),
        )
        second_call = updates(("Understood.", None, True))
        client.chat.side_effect = [first_call, second_call]

        messages = [{"role": "user", "content": "stage the env file"}]
        gen = run_agent_turn(client, git_registry, messages)
        events = drive_agent_turn(gen, decisions=[False])  # reject the sensitive-file warning

        confirm_events = [e for e in events if e["type"] == "confirm_git_operation"]
        assert confirm_events[0]["operation"].sensitive_paths == [".env"]

        import subprocess

        staged = subprocess.run(
            ["git", "diff", "--staged", "--name-only"], cwd=str(git_repo), capture_output=True, text=True
        ).stdout
        assert ".env" not in staged

    def test_commit_with_nothing_staged_never_reaches_confirm(self, git_registry):
        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            ("", [{"function": {"name": "git_commit", "arguments": {"message": "Fix bug"}}}], True),
        )
        second_call = updates(("Let me stage something first.", None, True))
        client.chat.side_effect = [first_call, second_call]

        messages = [{"role": "user", "content": "commit"}]
        gen = run_agent_turn(client, git_registry, messages)
        events = drive_agent_turn(gen)  # no decisions needed

        assert not any(e["type"] == "confirm_git_operation" for e in events)
        tool_error_events = [e for e in events if e["type"] == "tool_error"]
        assert len(tool_error_events) == 1
        assert "nothing is staged" in tool_error_events[0]["message"].lower()

    def test_disallowed_operation_never_reaches_confirm(self, git_registry):
        """A rejected/invalid branch name must never even ask for approval."""
        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            ("", [{"function": {"name": "git_create_branch", "arguments": {"name": "-bad"}}}], True),
        )
        second_call = updates(("That name isn't valid.", None, True))
        client.chat.side_effect = [first_call, second_call]

        messages = [{"role": "user", "content": "create a branch called -bad"}]
        gen = run_agent_turn(client, git_registry, messages)
        events = drive_agent_turn(gen)

        assert not any(e["type"] == "confirm_git_operation" for e in events)

    def test_protects_unrelated_changes_when_staging_one_file(self, git_repo, git_registry):
        """Mandatory Phase 5 test (spec item 25): asking the agent to stage
        only the README change must leave an unrelated modified file
        (backend/auth.py) untouched and unstaged.
        """
        (git_repo / "README.md").write_text("v2\n")
        (git_repo / "backend" / "auth.py").write_text("def login():\n    return None\n")

        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            ("", [{"function": {"name": "git_stage", "arguments": {"paths": ["README.md"]}}}], True),
        )
        second_call = updates(("Staged only the README change.", None, True))
        client.chat.side_effect = [first_call, second_call]

        messages = [{"role": "user", "content": "commit only the README change"}]
        gen = run_agent_turn(client, git_registry, messages)
        drive_agent_turn(gen, decisions=[True])

        import subprocess

        staged = subprocess.run(
            ["git", "diff", "--staged", "--name-only"], cwd=str(git_repo), capture_output=True, text=True
        ).stdout.split()
        unstaged = subprocess.run(
            ["git", "diff", "--name-only"], cwd=str(git_repo), capture_output=True, text=True
        ).stdout.split()

        assert staged == ["README.md"]
        assert unstaged == ["backend/auth.py"]

    def test_auto_approve_edits_does_not_auto_approve_git_operations(self, git_repo, git_registry):
        """Auto mode's scope is edits/plans only -- a Git operation must
        still pause for an explicit decision even with
        auto_approve_edits=True, and a rejected one must not run."""
        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            ("", [{"function": {"name": "git_create_branch", "arguments": {"name": "feature/x"}}}], True),
        )
        second_call = updates(("Okay, not creating it.", None, True))
        client.chat.side_effect = [first_call, second_call]

        messages = [{"role": "user", "content": "create a branch"}]
        gen = run_agent_turn(client, git_registry, messages, auto_approve_edits=True)
        events = drive_agent_turn(gen, decisions=[False])

        confirm_events = [e for e in events if e["type"] == "confirm_git_operation"]
        assert len(confirm_events) == 1
        assert "auto_approved" not in confirm_events[0]
        assert any(e["type"] == "git_operation_rejected" for e in events)

        import subprocess

        branches = subprocess.run(
            ["git", "branch", "--list", "feature/x"], cwd=str(git_repo), capture_output=True, text=True
        ).stdout
        assert "feature/x" not in branches


class TestPlanApprovalFlow:
    """Phase 6's planning mechanism reuses the same propose/confirm/adopt
    pattern as file edits, commands, and Git operations. A plan itself has
    no side effects -- there's nothing to "execute" -- but it still must be
    explicitly approved before it becomes the active plan, and rejecting it
    must leave task_state untouched.
    """

    def test_approved_plan_is_adopted_into_task_state(self, registry, task_state):
        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            (
                "",
                [
                    {
                        "function": {
                            "name": "create_plan",
                            "arguments": {
                                "goal": "Add JWT authentication",
                                "steps": ["Inspect auth", "Add endpoint", "Add tests"],
                            },
                        }
                    }
                ],
                True,
            ),
        )
        second_call = updates(("Here's my plan.", None, True))
        client.chat.side_effect = [first_call, second_call]

        messages = [{"role": "user", "content": "Add JWT authentication"}]
        gen = run_agent_turn(client, registry, messages, task_state=task_state)
        events = drive_agent_turn(gen, decisions=[True])

        confirm_events = [e for e in events if e["type"] == "confirm_plan"]
        approved_events = [e for e in events if e["type"] == "plan_approved"]
        assert len(confirm_events) == 1
        assert confirm_events[0]["plan"].goal == "Add JWT authentication"
        assert len(approved_events) == 1

        assert task_state.goal == "Add JWT authentication"
        assert task_state.plan is not None
        assert len(task_state.plan.steps) == 3
        assert all(s.status == "pending" for s in task_state.plan.steps)

    def test_rejected_plan_is_not_adopted(self, registry, task_state):
        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            (
                "",
                [
                    {
                        "function": {
                            "name": "create_plan",
                            "arguments": {"goal": "Add JWT auth", "steps": ["a", "b"]},
                        }
                    }
                ],
                True,
            ),
        )
        second_call = updates(("Okay, I won't create a plan.", None, True))
        client.chat.side_effect = [first_call, second_call]

        messages = [{"role": "user", "content": "Add JWT auth"}]
        gen = run_agent_turn(client, registry, messages, task_state=task_state)
        events = drive_agent_turn(gen, decisions=[False])

        rejected_events = [e for e in events if e["type"] == "plan_rejected"]
        assert len(rejected_events) == 1
        assert task_state.plan is None
        assert task_state.goal is None

        tool_messages = [m for m in messages if m["role"] == "tool"]
        assert "did not approve" in tool_messages[0]["content"].lower()

    def test_update_plan_after_approval_changes_task_state(self, registry, task_state):
        client = mock.create_autospec(OllamaClient, instance=True)
        create_call = updates(
            (
                "",
                [{"function": {"name": "create_plan", "arguments": {"goal": "Add X", "steps": ["a", "b"]}}}],
                True,
            ),
        )
        update_call = updates(
            (
                "",
                [{"function": {"name": "update_plan", "arguments": {"step_id": 1, "status": "completed"}}}],
                True,
            ),
        )
        final_call = updates(("Step 1 done.", None, True))
        client.chat.side_effect = [create_call, update_call, final_call]

        messages = [{"role": "user", "content": "Add X"}]
        gen = run_agent_turn(client, registry, messages, task_state=task_state)
        drive_agent_turn(gen, decisions=[True])

        assert task_state.plan.get_step(1).status == "completed"

    def test_small_task_never_calls_create_plan_is_unaffected(self, registry, task_state):
        """Not calling create_plan at all (the expected behavior for a
        small request) must work exactly as it did before Phase 6."""
        client = mock.create_autospec(OllamaClient, instance=True)
        client.chat.return_value = updates(("Sure, done.", None, True))

        messages = [{"role": "user", "content": "change the button text to Save"}]
        events = list(run_agent_turn(client, registry, messages, task_state=task_state))

        assert not any(e["type"] == "confirm_plan" for e in events)
        assert task_state.plan is None

    def test_task_state_records_read_file_during_a_plan(self, registry, task_state, project):
        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            ("", [{"function": {"name": "read_file", "arguments": {"path": "backend/auth.py"}}}], True),
        )
        second_call = updates(("It's a JWTAuth class.", None, True))
        client.chat.side_effect = [first_call, second_call]

        messages = [{"role": "user", "content": "what's in backend/auth.py?"}]
        gen = run_agent_turn(client, registry, messages, task_state=task_state)
        list(drive_agent_turn(gen))

        assert "backend/auth.py" in task_state.files_inspected

    def test_auto_approve_edits_adopts_plan_without_the_caller_answering(self, registry, task_state):
        """Same safety property as the edit case: a driver that never
        provides a real decision still gets the plan adopted, because
        auto_approve_edits resolves it internally."""
        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            (
                "",
                [
                    {
                        "function": {
                            "name": "create_plan",
                            "arguments": {"goal": "Add JWT authentication", "steps": ["a", "b"]},
                        }
                    }
                ],
                True,
            ),
        )
        second_call = updates(("Here's my plan.", None, True))
        client.chat.side_effect = [first_call, second_call]

        messages = [{"role": "user", "content": "Add JWT authentication"}]
        gen = run_agent_turn(client, registry, messages, task_state=task_state, auto_approve_edits=True)

        events = []
        send_value = None
        while True:
            try:
                event = gen.send(send_value)
            except StopIteration:
                break
            events.append(event)
            send_value = None  # never answer

        confirm_events = [e for e in events if e["type"] == "confirm_plan"]
        approved_events = [e for e in events if e["type"] == "plan_approved"]
        assert confirm_events[0]["auto_approved"] is True
        assert len(approved_events) == 1
        assert task_state.plan is not None
        assert task_state.goal == "Add JWT authentication"


class TestCancelEvent:
    """cancel_event is additive -- Phase 7's Stop Task mechanism -- and must
    have zero effect on any existing caller that doesn't pass it (default
    None), and must stop a turn cleanly (no partial/corrupt state) when set.
    """

    def test_unset_cancel_event_behaves_exactly_like_no_cancel_event(self, registry):
        client = mock.create_autospec(OllamaClient, instance=True)
        client.chat.return_value = updates(("Hello", None, True))

        messages = [{"role": "user", "content": "hi"}]
        events = list(
            run_agent_turn(client, registry, messages, cancel_event=threading.Event())
        )

        assert events[-1] == {"type": "final", "text": "Hello"}

    def test_cancel_event_set_before_first_iteration_stops_immediately(self, registry):
        client = mock.create_autospec(OllamaClient, instance=True)
        cancel_event = threading.Event()
        cancel_event.set()

        messages = [{"role": "user", "content": "hi"}]
        events = list(run_agent_turn(client, registry, messages, cancel_event=cancel_event))

        assert events == [{"type": "cancelled"}]
        client.chat.assert_not_called()

    def test_ollama_cancelled_error_during_streaming_yields_cancelled(self, registry):
        client = mock.create_autospec(OllamaClient, instance=True)

        def raise_cancelled(*args, **kwargs):
            raise OllamaCancelledError("Generation was cancelled.")
            yield  # pragma: no cover - makes this a generator function

        client.chat.side_effect = raise_cancelled

        messages = [{"role": "user", "content": "hi"}]
        events = list(
            run_agent_turn(client, registry, messages, cancel_event=threading.Event())
        )

        assert events == [{"type": "cancelled"}]

    def test_cancel_event_is_passed_through_to_chat(self, registry):
        client = mock.create_autospec(OllamaClient, instance=True)
        client.chat.return_value = updates(("Hello", None, True))
        cancel_event = threading.Event()

        messages = [{"role": "user", "content": "hi"}]
        list(run_agent_turn(client, registry, messages, cancel_event=cancel_event))

        _, kwargs = client.chat.call_args
        assert kwargs.get("cancel_event") is cancel_event

    def test_messages_left_consistent_after_cancellation(self, registry):
        """A cancelled turn must not leave a dangling assistant tool_calls
        message with no matching tool result -- the conversation should be
        exactly as it was before this turn's model call, so the next turn
        (or /new) starts from clean state."""
        client = mock.create_autospec(OllamaClient, instance=True)
        cancel_event = threading.Event()
        cancel_event.set()

        messages = [{"role": "user", "content": "hi"}]
        list(run_agent_turn(client, registry, messages, cancel_event=cancel_event))

        assert messages == [{"role": "user", "content": "hi"}]


class TestRepetitionDetection:
    """Phase 8: the same (tool, arguments) signature repeated
    MAX_CONSECUTIVE_IDENTICAL_CALLS times in a row -- whether it keeps
    "succeeding" with no progress or keeps failing the same way -- must be
    intercepted before a 3rd/4th real (re-)execution, not just eventually
    stopped by max_iterations."""

    def test_identical_read_only_call_repeated_is_intercepted(self, registry, project):
        (project.root / "backend" / "auth.py").write_text("class JWTAuth:\n    pass\n")
        client = mock.create_autospec(OllamaClient, instance=True)

        # A generator is consumed after one iteration, so each call to
        # client.chat() must build a *fresh* one -- reusing the same
        # generator object across side_effect entries would silently yield
        # nothing on the 2nd+ call, which isn't what this test means to do.
        client.chat.side_effect = lambda *a, **k: updates(
            ("", [{"function": {"name": "read_file", "arguments": {"path": "backend/auth.py"}}}], True),
        )

        messages = [{"role": "user", "content": "keep reading the same file"}]
        events = list(run_agent_turn(client, registry, messages, max_iterations=4))

        repetition_events = [e for e in events if e["type"] == "repetition_detected"]
        tool_call_events = [e for e in events if e["type"] == "tool_call" and e["name"] == "read_file"]

        # 2 real executions (count 1, 2), then intercepted from the 3rd
        # identical call onward (this loop runs 4 iterations total).
        assert len(tool_call_events) == 2
        assert len(repetition_events) == 2
        assert repetition_events[0]["count"] == 3
        assert repetition_events[1]["count"] == 4
        assert repetition_events[0]["name"] == "read_file"

    def test_repetition_counter_resets_on_a_different_call(self, registry, project):
        (project.root / "backend" / "auth.py").write_text("class JWTAuth:\n    pass\n")
        client = mock.create_autospec(OllamaClient, instance=True)

        def make_read_call(*_a, **_k):
            return updates(
                ("", [{"function": {"name": "read_file", "arguments": {"path": "backend/auth.py"}}}], True),
            )

        def make_list_call(*_a, **_k):
            return updates(
                ("", [{"function": {"name": "list_files", "arguments": {"path": "."}}}], True),
            )

        # read, read, list (different -- resets the streak), read, read: no
        # occurrence ever reaches 3 in a row, so nothing should be intercepted.
        client.chat.side_effect = [
            make_read_call(),
            make_read_call(),
            make_list_call(),
            make_read_call(),
            make_read_call(),
        ]

        messages = [{"role": "user", "content": "hi"}]
        events = list(run_agent_turn(client, registry, messages, max_iterations=5))

        assert not any(e["type"] == "repetition_detected" for e in events)

    def test_repeated_call_does_not_reach_registry_a_third_time(self, registry, project):
        """The intercepted call must not actually re-execute the tool --
        only the first MAX_CONSECUTIVE_IDENTICAL_CALLS-1 executions should
        really run."""
        (project.root / "backend" / "auth.py").write_text("class JWTAuth:\n    pass\n")
        client = mock.create_autospec(OllamaClient, instance=True)
        client.chat.side_effect = lambda *a, **k: updates(
            ("", [{"function": {"name": "read_file", "arguments": {"path": "backend/auth.py"}}}], True),
        )

        messages = [{"role": "user", "content": "hi"}]
        list(run_agent_turn(client, registry, messages, max_iterations=3))

        # Only 2 real read_file executions should have happened -- check via
        # the tool message contents appended to the conversation.
        tool_messages = [m for m in messages if m.get("role") == "tool" and m.get("tool_name") == "read_file"]
        assert len(tool_messages) == 3  # 2 real reads + 1 synthetic repetition notice
        assert "repeated" in tool_messages[-1]["content"].lower()
        assert "class JWTAuth" in tool_messages[0]["content"]
        assert "class JWTAuth" in tool_messages[1]["content"]

    def test_repeated_failing_edit_is_also_intercepted(self, registry, project):
        """Covers spec section 10 (failed-approach detection): the same
        proposal that keeps failing validation is exactly as much a stuck
        loop as a successful no-op repeat."""
        (project.root / "x.py").write_text("value = 1\n")
        client = mock.create_autospec(OllamaClient, instance=True)
        client.chat.side_effect = lambda *a, **k: updates(
            (
                "",
                [
                    {
                        "function": {
                            "name": "edit_file",
                            "arguments": {
                                "path": "x.py",
                                "old_text": "value = 999\n",  # doesn't match -- always fails
                                "new_text": "value = 2\n",
                            },
                        }
                    }
                ],
                True,
            ),
        )

        messages = [{"role": "user", "content": "fix it"}]
        events = list(run_agent_turn(client, registry, messages, max_iterations=3))

        tool_error_events = [e for e in events if e["type"] == "tool_error"]
        repetition_events = [e for e in events if e["type"] == "repetition_detected"]
        assert len(tool_error_events) == 2  # first 2 attempts genuinely fail validation
        assert len(repetition_events) == 1  # 3rd is intercepted instead of failing again

    def test_repetition_gives_up_after_escalation_limit_instead_of_looping_forever(
        self, registry, project
    ):
        """A model that keeps issuing the identical call even after being
        told to stop (observed live: git_status repeated past 10 times) must
        not be allowed to burn the rest of max_iterations doing that -- the
        turn should end on its own, flagged task_incomplete, well before the
        iteration ceiling."""
        (project.root / "backend" / "auth.py").write_text("class JWTAuth:\n    pass\n")
        client = mock.create_autospec(OllamaClient, instance=True)
        client.chat.side_effect = lambda *a, **k: updates(
            ("", [{"function": {"name": "read_file", "arguments": {"path": "backend/auth.py"}}}], True),
        )

        messages = [{"role": "user", "content": "keep reading the same file"}]
        # Generously above the point the turn should actually end at, so the
        # test proves early termination rather than merely tolerating it.
        events = list(run_agent_turn(client, registry, messages, max_iterations=20))

        repetition_events = [e for e in events if e["type"] == "repetition_detected"]
        final_events = [e for e in events if e["type"] == "final"]
        max_iterations_events = [e for e in events if e["type"] == "max_iterations"]

        # 2 interceptions (count 3, 4) before the 3rd interception attempt
        # (count 5) gives up instead of firing a 3rd repetition_detected.
        assert [e["count"] for e in repetition_events] == [3, 4]
        assert len(final_events) == 1
        assert final_events[0]["task_incomplete"] is True
        assert not max_iterations_events


class TestFileOpApprovalFlow:
    """delete_file/rename_file follow the exact same propose/pause/apply
    split as edit_file/write_file (TestApprovalFlow above) -- nothing on
    disk changes until the caller sends back an explicit True."""

    def test_approved_delete_removes_file(self, project, registry):
        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            ("", [{"function": {"name": "delete_file", "arguments": {"path": "backend/auth.py"}}}], True),
        )
        second_call = updates(("Deleted it.", None, True))
        client.chat.side_effect = [first_call, second_call]

        messages = [{"role": "user", "content": "remove auth.py"}]
        gen = run_agent_turn(client, registry, messages)
        events = drive_agent_turn(gen, decisions=[True])

        confirm_events = [e for e in events if e["type"] == "confirm_file_op"]
        applied_events = [e for e in events if e["type"] == "file_op_applied"]
        assert len(confirm_events) == 1
        assert confirm_events[0]["operation"].kind == "delete"
        assert len(applied_events) == 1
        assert not (project.root / "backend" / "auth.py").exists()

    def test_rejected_delete_is_not_applied(self, project, registry):
        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            ("", [{"function": {"name": "delete_file", "arguments": {"path": "backend/auth.py"}}}], True),
        )
        second_call = updates(("Okay, leaving it.", None, True))
        client.chat.side_effect = [first_call, second_call]

        messages = [{"role": "user", "content": "remove auth.py"}]
        gen = run_agent_turn(client, registry, messages)
        events = drive_agent_turn(gen, decisions=[False])

        assert any(e["type"] == "file_op_rejected" for e in events)
        assert not any(e["type"] == "file_op_applied" for e in events)
        assert (project.root / "backend" / "auth.py").exists()

    def test_approved_rename_moves_file(self, project, registry):
        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            (
                "",
                [
                    {
                        "function": {
                            "name": "rename_file",
                            "arguments": {"source": "backend/auth.py", "destination": "backend/security.py"},
                        }
                    }
                ],
                True,
            ),
        )
        second_call = updates(("Renamed it.", None, True))
        client.chat.side_effect = [first_call, second_call]

        messages = [{"role": "user", "content": "rename auth.py to security.py"}]
        gen = run_agent_turn(client, registry, messages)
        events = drive_agent_turn(gen, decisions=[True])

        assert any(e["type"] == "file_op_applied" for e in events)
        assert not (project.root / "backend" / "auth.py").exists()
        assert (project.root / "backend" / "security.py").exists()

    def test_auto_approve_edits_also_covers_file_ops(self, project, registry):
        """auto_approve_edits is documented to cover delete_file/rename_file
        as the same risk class as edit_file/write_file -- verify the
        generator resolves the approval internally with no decision sent."""
        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            ("", [{"function": {"name": "delete_file", "arguments": {"path": "backend/auth.py"}}}], True),
        )
        second_call = updates(("Deleted it.", None, True))
        client.chat.side_effect = [first_call, second_call]

        messages = [{"role": "user", "content": "remove auth.py"}]
        events = list(run_agent_turn(client, registry, messages, auto_approve_edits=True))

        confirm_events = [e for e in events if e["type"] == "confirm_file_op"]
        assert confirm_events[0]["auto_approved"] is True
        assert any(e["type"] == "file_op_applied" for e in events)
        assert not (project.root / "backend" / "auth.py").exists()

    def test_delete_updates_task_state(self, project, registry, task_state):
        task_state.note_file_inspected("backend/auth.py")
        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            ("", [{"function": {"name": "delete_file", "arguments": {"path": "backend/auth.py"}}}], True),
        )
        second_call = updates(("Deleted it.", None, True))
        client.chat.side_effect = [first_call, second_call]

        messages = [{"role": "user", "content": "remove auth.py"}]
        gen = run_agent_turn(client, registry, messages, task_state=task_state)
        drive_agent_turn(gen, decisions=[True])

        assert "backend/auth.py" not in task_state.files_inspected


class TestNarrationDetection:
    """A model response with zero tool calls that reads as unactioned
    narration ("I'll do X", "what would you like me to do next?") must not
    be accepted as the final answer while no mutating tool has run yet this
    turn -- see loop.py's _looks_like_unactioned_narration and
    MUTATING_TOOL_NAMES. Reproduces the live-observed failure: two read_file
    calls (including a cache hit), then plain-text narration with no tool
    call at all."""

    def test_deferral_question_with_no_tool_call_is_redirected(self, registry):
        client = mock.create_autospec(OllamaClient, instance=True)
        client.chat.side_effect = [
            updates(("Got it! What would you like me to do next?", None, True)),
            updates(("Done, I created the file.", None, True)),
        ]

        messages = [{"role": "user", "content": "create hello.py with a hello function"}]
        events = list(run_agent_turn(client, registry, messages))

        redirected = [e for e in events if e["type"] == "narration_redirected"]
        final_events = [e for e in events if e["type"] == "final"]
        assert len(redirected) == 1
        assert redirected[0]["attempt"] == 1
        assert len(final_events) == 1
        assert final_events[0]["text"] == "Done, I created the file."
        assert not final_events[0].get("task_incomplete")
        assert client.chat.call_count == 2

    def test_unactioned_intent_with_no_tool_call_is_redirected(self, registry):
        client = mock.create_autospec(OllamaClient, instance=True)
        client.chat.side_effect = [
            updates(("Sure, I'll create hello.py with a hello function for you.", None, True)),
            updates(
                (
                    "",
                    [{"function": {"name": "write_file", "arguments": {"path": "hello.py", "content": "x = 1\n"}}}],
                    True,
                ),
            ),
            updates(("Created it.", None, True)),
        ]

        messages = [{"role": "user", "content": "create hello.py"}]
        gen = run_agent_turn(client, registry, messages)
        events = drive_agent_turn(gen, decisions=[True])

        assert any(e["type"] == "narration_redirected" for e in events)
        assert any(e["type"] == "change_applied" for e in events)

    def test_a_mutating_tool_call_first_means_no_redirect_even_if_final_text_matches(self, registry, project):
        """Once a mutating tool has actually run this turn, a later
        plain-text wrap-up must NOT be treated as unactioned narration even
        if it happens to contain similar phrasing -- it's a real completion
        report, not a stall."""
        client = mock.create_autospec(OllamaClient, instance=True)
        client.chat.side_effect = [
            updates(
                (
                    "",
                    [
                        {
                            "function": {
                                "name": "write_file",
                                "arguments": {"path": "hello.py", "content": "x = 1\n"},
                            }
                        }
                    ],
                    True,
                ),
            ),
            updates(("I created hello.py -- let me know if you would like anything else.", None, True)),
        ]

        messages = [{"role": "user", "content": "create hello.py"}]
        gen = run_agent_turn(client, registry, messages)
        events = drive_agent_turn(gen, decisions=[True])

        assert not any(e["type"] == "narration_redirected" for e in events)
        final_events = [e for e in events if e["type"] == "final"]
        assert len(final_events) == 1

    def test_repeated_narration_exhausts_retries_and_flags_task_incomplete(self, registry):
        client = mock.create_autospec(OllamaClient, instance=True)
        client.chat.side_effect = lambda *a, **k: updates(
            ("I'll create hello.py for you shortly.", None, True),
        )

        messages = [{"role": "user", "content": "create hello.py"}]
        events = list(run_agent_turn(client, registry, messages, max_iterations=10))

        redirected = [e for e in events if e["type"] == "narration_redirected"]
        final_events = [e for e in events if e["type"] == "final"]
        assert len(redirected) == 2  # MAX_NARRATION_RETRIES
        assert len(final_events) == 1
        assert final_events[0]["task_incomplete"] is True

    def test_genuine_informational_answer_is_not_flagged(self, registry):
        """A real answer to a real question (no mutation implied) must not
        be redirected just because it happens to share some phrasing."""
        client = mock.create_autospec(OllamaClient, instance=True)
        client.chat.return_value = updates(
            (
                "A list comprehension in Python is a concise way to build a list, e.g. "
                "[x * 2 for x in range(5)].",
                None,
                True,
            ),
        )

        messages = [{"role": "user", "content": "what is a list comprehension?"}]
        events = list(run_agent_turn(client, registry, messages))

        assert not any(e["type"] == "narration_redirected" for e in events)
        assert client.chat.call_count == 1


class TestEndToEndSortingScenario:
    """The representative end-to-end scenario this whole feature set was
    built for: create a new combined file, migrate the old implementation
    into it, update a reference to the old file, delete the old file, and
    run the tests -- all in one turn, driven entirely by the model's own
    tool calls with no human ever typing "call write_file now" or "call
    delete_file" mid-task. Exercises write_file, search_files, edit_file,
    delete_file, and run_command together against a real temp-dir project;
    only OllamaClient.chat and the run_command subprocess are mocked."""

    @pytest.fixture
    def sorting_project(self, tmp_path):
        (tmp_path / "bubble_sort.py").write_text(
            "def bubble_sort(items):\n"
            "    items = list(items)\n"
            "    for i in range(len(items)):\n"
            "        for j in range(len(items) - i - 1):\n"
            "            if items[j] > items[j + 1]:\n"
            "                items[j], items[j + 1] = items[j + 1], items[j]\n"
            "    return items\n"
        )
        (tmp_path / "main.py").write_text(
            "import bubble_sort\n\nprint(bubble_sort.bubble_sort([3, 1, 2]))\n"
        )
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_sorting.py").write_text("def test_placeholder():\n    assert True\n")
        return ProjectRoot(tmp_path)

    @pytest.fixture
    def sorting_task_state(self):
        return TaskState()

    @pytest.fixture
    def sorting_registry(self, sorting_project, sorting_task_state):
        return build_default_registry(sorting_project, task_state=sorting_task_state)

    def test_full_scenario_completes_without_manual_intervention(
        self, sorting_project, sorting_registry, sorting_task_state
    ):
        combined_content = (
            "def bubble_sort(items):\n    return sorted(items)\n\n"
            "def quick_sort(items):\n    return sorted(items)\n\n"
            "def merge_sort(items):\n    return sorted(items)\n\n"
            "def insertion_sort(items):\n    return sorted(items)\n\n"
            "def selection_sort(items):\n    return sorted(items)\n"
        )

        client = mock.create_autospec(OllamaClient, instance=True)
        client.chat.side_effect = [
            updates(("", [{"function": {"name": "list_files", "arguments": {"path": "."}}}], True)),
            updates(
                ("", [{"function": {"name": "read_file", "arguments": {"path": "bubble_sort.py"}}}], True)
            ),
            updates(
                (
                    "",
                    [
                        {
                            "function": {
                                "name": "write_file",
                                "arguments": {"path": "sorting.py", "content": combined_content},
                            }
                        }
                    ],
                    True,
                ),
            ),
            updates(
                ("", [{"function": {"name": "search_files", "arguments": {"query": "bubble_sort"}}}], True)
            ),
            updates(
                (
                    "",
                    [
                        {
                            "function": {
                                "name": "edit_file",
                                "arguments": {
                                    "path": "main.py",
                                    "old_text": "import bubble_sort\n\nprint(bubble_sort.bubble_sort([3, 1, 2]))\n",
                                    "new_text": "import sorting\n\nprint(sorting.bubble_sort([3, 1, 2]))\n",
                                },
                            }
                        }
                    ],
                    True,
                ),
            ),
            updates(
                ("", [{"function": {"name": "delete_file", "arguments": {"path": "bubble_sort.py"}}}], True)
            ),
            updates(
                (
                    "",
                    [{"function": {"name": "run_command", "arguments": {"program": "pytest", "args": []}}}],
                    True,
                ),
            ),
            updates(
                (
                    "Done: created sorting.py with all five sort functions, updated main.py's import, "
                    "removed bubble_sort.py, and the tests passed.",
                    None,
                    True,
                ),
            ),
        ]

        messages = [
            {
                "role": "user",
                "content": (
                    "Create a new file sorting.py containing bubble_sort, quick_sort, merge_sort, "
                    "insertion_sort, and selection_sort. Move the old bubble_sort.py implementation "
                    "into sorting.py, update any imports that reference bubble_sort.py, and remove "
                    "the old file. Run the tests."
                ),
            }
        ]

        with mock.patch("agent.tools.terminal.subprocess.Popen") as mock_popen:
            fake_proc = mock.Mock()
            fake_proc.communicate.return_value = ("2 passed in 0.01s\n", "")
            fake_proc.returncode = 0
            fake_proc.pid = 12345
            mock_popen.return_value = fake_proc

            gen = run_agent_turn(client, sorting_registry, messages, task_state=sorting_task_state)
            # decisions sent only for confirm-shaped events, in order:
            # write_file, edit_file, delete_file, run_command
            events = drive_agent_turn(gen, decisions=[True, True, True, True])

        # No manual nudging was needed: the model never triggered the
        # narration-correction path (it called a real tool every turn).
        assert not any(e["type"] == "narration_redirected" for e in events)

        # A genuine final answer was reached, not a stall/incomplete flag.
        final_events = [e for e in events if e["type"] == "final"]
        assert len(final_events) == 1
        assert not final_events[0].get("task_incomplete")

        # Real filesystem state matches the request -- not just the model's
        # narration of it (see CLAUDE.md's "chat transcript claiming success
        # is not evidence of success").
        assert not (sorting_project.root / "bubble_sort.py").exists()
        assert (sorting_project.root / "sorting.py").exists()
        sorting_content = (sorting_project.root / "sorting.py").read_text()
        for fn in ("bubble_sort", "quick_sort", "merge_sort", "insertion_sort", "selection_sort"):
            assert f"def {fn}(" in sorting_content
        main_content = (sorting_project.root / "main.py").read_text()
        assert "import sorting" in main_content
        assert "import bubble_sort" not in main_content

        # Tests were actually run (via the mocked subprocess), not just claimed.
        assert any(e["type"] == "command_result" for e in events)
        assert any(e["type"] == "file_op_applied" for e in events)  # the delete
        assert any(e["type"] == "change_applied" for e in events)  # the write + edit
