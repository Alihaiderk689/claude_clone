"""Tests for the tool-calling agent loop in agent/loop.py.

OllamaClient.chat is mocked throughout, so these tests run without a real
Ollama server or model.
"""
from __future__ import annotations

from unittest import mock

import pytest

from agent.loop import run_agent_turn
from agent.ollama_client import OllamaClient, OllamaConnectionError
from agent.project import ProjectRoot
from agent.tools import FileStateTracker, build_default_registry
from agent.tools.base import Tool, ToolResult
from agent.tools.registry import ToolRegistry


def updates(*items):
    """Helper: build a fake stream of chat() updates from simple tuples."""
    for content, tool_calls, done in items:
        yield {"content": content, "tool_calls": tool_calls, "done": done}


_CONFIRMATION_EVENT_TYPES = {"confirm", "confirm_command", "confirm_git_operation"}


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
def registry(project):
    return build_default_registry(project)


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
            ("", [{"function": {"name": "delete_file", "arguments": {"path": "x"}}}], True),
        )
        second_call = updates(("I can't do that yet.", None, True))
        client.chat.side_effect = [first_call, second_call]

        messages = [{"role": "user", "content": "delete a file"}]
        events = list(run_agent_turn(client, registry, messages))

        tool_error_events = [e for e in events if e["type"] == "tool_error"]
        assert len(tool_error_events) == 1
        assert "unknown tool" in tool_error_events[0]["message"].lower()

        final_events = [e for e in events if e["type"] == "final"]
        assert final_events[0]["text"] == "I can't do that yet."


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

    def test_connection_error_yields_error_event_and_stops(self, registry):
        client = mock.create_autospec(OllamaClient, instance=True)

        def raise_connection_error(*_args, **_kwargs):
            raise OllamaConnectionError("Could not connect to Ollama.")
            yield  # pragma: no cover - makes this a generator function

        client.chat.side_effect = raise_connection_error

        messages = [{"role": "user", "content": "hi"}]
        events = list(run_agent_turn(client, registry, messages))

        assert len(events) == 1
        assert events[0]["type"] == "error"
        assert "could not connect" in events[0]["message"].lower()


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


class TestRunCommandApprovalFlow:
    """Phase 4's core guarantee, same shape as Phase 3's: the model proposes
    a command, the loop pauses with a 'confirm_command' event, and nothing
    executes until the caller sends back an explicit True. subprocess.run is
    always mocked -- no real process runs in these tests.
    """

    def test_approved_command_executes(self, registry):
        client = mock.create_autospec(OllamaClient, instance=True)
        first_call = updates(
            ("", [{"function": {"name": "run_command", "arguments": {"program": "pytest", "args": []}}}], True),
        )
        second_call = updates(("Tests passed.", None, True))
        client.chat.side_effect = [first_call, second_call]

        fake_proc = mock.Mock(returncode=0, stdout="2 passed\n", stderr="")
        with mock.patch("agent.tools.terminal.subprocess.run", return_value=fake_proc) as mock_run:
            messages = [{"role": "user", "content": "run the tests"}]
            gen = run_agent_turn(client, registry, messages)
            events = drive_agent_turn(gen, decisions=[True])

        mock_run.assert_called_once()
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

        with mock.patch("agent.tools.terminal.subprocess.run") as mock_run:
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

        with mock.patch("agent.tools.terminal.subprocess.run") as mock_run:
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

        with mock.patch("agent.tools.terminal.subprocess.run") as mock_run:
            messages = [{"role": "user", "content": "run pytest; rm -rf ."}]
            gen = run_agent_turn(client, registry, messages)
            events = drive_agent_turn(gen)

        mock_run.assert_not_called()
        assert not any(e["type"] == "confirm_command" for e in events)

    def test_timed_out_command_reports_tool_error_too(self, registry):
        import subprocess

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

        with mock.patch(
            "agent.tools.terminal.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["pytest"], timeout=5),
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

        fake_proc = mock.Mock(returncode=0, stdout="ok\n", stderr="")
        with mock.patch("agent.tools.terminal.subprocess.run", return_value=fake_proc) as mock_run:
            messages = [{"role": "user", "content": "run tests and lint"}]
            gen = run_agent_turn(client, registry, messages)
            events = drive_agent_turn(gen, decisions=[True, False])  # approve pytest, reject ruff

        assert mock_run.call_count == 1  # only the approved one actually ran
        applied = [e for e in events if e["type"] == "command_result"]
        rejected = [e for e in events if e["type"] == "command_rejected"]
        assert len(applied) == 1
        assert len(rejected) == 1


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
