"""Stress and memory/resource tests (Phase 8 sections 29-30).

The goal here is not performance measurement -- it's "does the agent
remain stable" under conditions that are larger/longer/more repetitive
than the ordinary unit tests exercise: many tool calls in one turn, large
files/diffs, huge command output, many repeated calls, many plan steps,
repeated interruption cycles, and many sequential requests against the
real HTTP server. Everything here still runs with OllamaClient mocked and
completes in well under a second per test -- "stress" refers to the shape
of the workload, not wall-clock load testing.
"""
from __future__ import annotations

import json
import threading
import time
from unittest import mock

import pytest

from agent.loop import MAX_TOOL_ITERATIONS, run_agent_turn
from agent.ollama_client import OllamaClient
from agent.project import ProjectRoot
from agent.task_state import TaskState
from agent.tools import build_default_registry
from agent.tools.terminal import MAX_STDOUT_CHARS


def updates(*items):
    for content, tool_calls, done in items:
        yield {"content": content, "tool_calls": tool_calls, "done": done}


@pytest.fixture
def project(tmp_path):
    for i in range(30):
        (tmp_path / f"file_{i}.py").write_text(f"# file {i}\nvalue_{i} = {i}\n")
    return ProjectRoot(tmp_path)


@pytest.fixture
def task_state():
    return TaskState()


@pytest.fixture
def registry(project, task_state):
    return build_default_registry(project, task_state=task_state)


class TestManyToolCallsInOneTurn:
    def test_reaches_max_iterations_without_crashing(self, registry):
        """A model that never stops calling tools must be cut off cleanly
        at MAX_TOOL_ITERATIONS, not left running indefinitely or crash."""
        client = mock.create_autospec(OllamaClient, instance=True)
        call_index = {"n": 0}

        def infinite_tool_calls(*_a, **_k):
            call_index["n"] += 1
            path = f"file_{call_index['n'] % 30}.py"
            return updates(("", [{"function": {"name": "read_file", "arguments": {"path": path}}}], True))

        client.chat.side_effect = infinite_tool_calls

        messages = [{"role": "user", "content": "look at everything"}]
        start = time.monotonic()
        events = list(run_agent_turn(client, registry, messages))
        elapsed = time.monotonic() - start

        assert events[-1]["type"] == "max_iterations"
        assert client.chat.call_count == MAX_TOOL_ITERATIONS
        assert elapsed < 5  # no real I/O or sleeping involved -- should be near-instant

    def test_many_tool_calls_within_a_single_response(self, registry):
        """Ollama can return several tool_calls in one message; a large
        batch of them must all execute without issue."""
        client = mock.create_autospec(OllamaClient, instance=True)
        many_calls = [
            {"function": {"name": "read_file", "arguments": {"path": f"file_{i}.py"}}} for i in range(30)
        ]
        client.chat.side_effect = [
            updates(("", many_calls, True)),
            updates(("Looked at everything.", None, True)),
        ]

        messages = [{"role": "user", "content": "read all the files"}]
        events = list(run_agent_turn(client, registry, messages, max_iterations=MAX_TOOL_ITERATIONS))

        tool_call_events = [e for e in events if e["type"] == "tool_call"]
        assert len(tool_call_events) == 30
        assert events[-1]["type"] == "final"


class TestLargeFilesAndDiffs:
    def test_large_file_read_and_edit(self, tmp_path):
        big_content = "\n".join(f"line_{i} = {i}" for i in range(5000)) + "\n"
        (tmp_path / "big.py").write_text(big_content)
        project = ProjectRoot(tmp_path)
        task_state = TaskState()
        registry = build_default_registry(project, task_state=task_state)

        client = mock.create_autospec(OllamaClient, instance=True)
        client.chat.side_effect = [
            updates(("", [{"function": {"name": "read_file", "arguments": {"path": "big.py", "start_line": 1, "end_line": 800}}}], True)),
            updates(
                (
                    "",
                    [
                        {
                            "function": {
                                "name": "edit_file",
                                "arguments": {"path": "big.py", "old_text": "line_0 = 0\n", "new_text": "line_0 = 999\n"},
                            }
                        }
                    ],
                    True,
                )
            ),
            updates(("Updated the first line.", None, True)),
        ]

        messages = [{"role": "user", "content": "change the first value"}]
        gen = run_agent_turn(client, registry, messages, tracker=None, task_state=task_state)

        send_value = None
        events = []
        while True:
            try:
                event = gen.send(send_value)
            except StopIteration:
                break
            events.append(event)
            send_value = True if event["type"] == "confirm" else None

        assert any(e["type"] == "change_applied" for e in events)
        assert "line_0 = 999" in (tmp_path / "big.py").read_text()

    def test_large_diff_does_not_crash_rendering(self, tmp_path):
        """A change that touches a large fraction of a big file must still
        produce a valid diff/confirm event, not hang or crash."""
        from agent.diff import ProposedChange, unified_diff_text

        old_content = "\n".join(f"line {i}" for i in range(3000)) + "\n"
        new_content = "\n".join(f"line {i} MODIFIED" for i in range(3000)) + "\n"
        change = ProposedChange(
            path="big.py", resolved_path=tmp_path / "big.py", kind="edit",
            old_content=old_content, new_content=new_content,
        )
        start = time.monotonic()
        diff_text = unified_diff_text(change)
        elapsed = time.monotonic() - start

        assert "MODIFIED" in diff_text
        assert elapsed < 5


class TestLongCommandOutput:
    def test_extremely_long_output_is_still_bounded(self, tmp_path):
        from agent.command_policy import ApprovedCommand
        from agent.tools.terminal import execute_command, format_result_for_model

        cmd = ApprovedCommand(program="pytest", args=[], timeout=120, cwd=tmp_path)
        huge_output = "x" * (MAX_STDOUT_CHARS * 20)  # 20x the truncation limit
        fake_proc = mock.Mock()
        fake_proc.pid = 1
        fake_proc.returncode = 0
        fake_proc.communicate.return_value = (huge_output, "")

        with mock.patch("agent.tools.terminal.subprocess.Popen", return_value=fake_proc):
            result = execute_command(cmd)

        assert len(result.stdout) < len(huge_output)
        assert len(result.stdout) < MAX_STDOUT_CHARS * 2  # bounded regardless of real output size

        # The text sent to the model must also stay bounded -- this is
        # what actually protects an 8GB machine's context budget.
        model_text = format_result_for_model(result)
        assert len(model_text) < MAX_STDOUT_CHARS * 2


class TestRepeatedToolCallsAtScale:
    def test_twenty_identical_calls_only_execute_twice(self, registry, project):
        """Repetition detection (loop.py) must cap real executions at
        MAX_CONSECUTIVE_IDENTICAL_CALLS-1 regardless of how many times the
        model keeps asking. It must also not let the model keep re-issuing
        the identical call indefinitely just because interception isn't a
        real execution -- MAX_REPETITION_ESCALATIONS bounds how many times
        it will re-explain before giving up on the turn outright, rather
        than silently spending the rest of max_iterations on a call that has
        already proven it won't produce a different outcome."""
        client = mock.create_autospec(OllamaClient, instance=True)
        client.chat.side_effect = lambda *a, **k: updates(
            ("", [{"function": {"name": "read_file", "arguments": {"path": "file_0.py"}}}], True)
        )

        messages = [{"role": "user", "content": "keep reading the same file"}]
        events = list(run_agent_turn(client, registry, messages, max_iterations=20))

        tool_call_events = [e for e in events if e["type"] == "tool_call" and e["name"] == "read_file"]
        repetition_events = [e for e in events if e["type"] == "repetition_detected"]
        assert len(tool_call_events) == 2
        assert len(repetition_events) == 2
        assert events[-1]["type"] == "final"
        assert events[-1]["task_incomplete"] is True


class TestManyPlanSteps:
    def test_plan_near_the_maximum_step_count(self, registry, task_state):
        """agent/tools/planning.py caps steps at 10 -- a plan at that
        ceiling must still work through the full propose/approve/track
        cycle without issue."""
        client = mock.create_autospec(OllamaClient, instance=True)
        steps = [f"Step {i}" for i in range(10)]
        client.chat.side_effect = [
            updates(("", [{"function": {"name": "create_plan", "arguments": {"goal": "Big task", "steps": steps}}}], True)),
            updates(("Plan adopted.", None, True)),
        ]

        messages = [{"role": "user", "content": "do the big task"}]
        gen = run_agent_turn(client, registry, messages, task_state=task_state)
        send_value = None
        events = []
        while True:
            try:
                event = gen.send(send_value)
            except StopIteration:
                break
            events.append(event)
            send_value = True if event["type"] == "confirm_plan" else None

        assert task_state.plan is not None
        assert len(task_state.plan.steps) == 10
        assert events[-1]["type"] == "final"


class TestRepeatedInterruptionCycles:
    def test_stop_start_stop_start_leaves_session_usable(self, tmp_path, monkeypatch):
        """Repeated Stop Task / New message cycles must not degrade the
        session -- each cycle should cleanly release the lock and leave the
        next turn free to run."""
        import agent.server as server_module

        token_dir = tmp_path / "home" / ".code-agent"
        monkeypatch.setattr(server_module, "TOKEN_DIR", token_dir)
        monkeypatch.setattr(server_module, "TOKEN_FILE", token_dir / "server.json")
        monkeypatch.setattr(OllamaClient, "check_connection", lambda self: True)

        httpd = server_module.build_server(port=0)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        host, port = httpd.server_address[:2]
        base_url = f"http://{host}:{port}"
        token = server_module._Handler.token

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "f.py").write_text("x = 1\n")

        import requests

        def slow_loop(self, messages, tools=None, cancel_event=None):
            for _ in range(50):
                time.sleep(0.02)
                if cancel_event is not None and cancel_event.is_set():
                    from agent.ollama_client import OllamaCancelledError

                    raise OllamaCancelledError("cancelled")
            yield {"content": "done", "tool_calls": None, "done": True}

        try:
            with mock.patch.object(OllamaClient, "chat", slow_loop):
                for cycle in range(5):
                    t = threading.Thread(
                        target=lambda: requests.post(
                            f"{base_url}/chat",
                            json={"workspace_root": str(workspace), "message": f"cycle {cycle}"},
                            headers={"Authorization": f"Bearer {token}"},
                            timeout=10,
                        )
                    )
                    t.start()
                    time.sleep(0.03)
                    requests.post(
                        f"{base_url}/task/stop",
                        json={"workspace_root": str(workspace)},
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=5,
                    )
                    t.join(timeout=10)

            store = server_module._Handler.sessions
            session = store.get_or_create(str(workspace))
            assert not session.lock.locked()

            health = requests.get(f"{base_url}/health", timeout=5)
            assert health.status_code == 200
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=2)


class TestContextStaysBounded:
    """Phase 8 section 30: on an 8GB machine, the conversation sent to
    Ollama must not grow without bound across a long multi-turn task --
    this is what agent/context_manager.py's compaction exists for."""

    def test_message_history_size_stays_bounded_across_many_turns(self, registry, task_state):
        client = mock.create_autospec(OllamaClient, instance=True)
        messages = [{"role": "system", "content": "You are a coding assistant in /project"}]

        for i in range(25):
            client.chat.side_effect = [
                updates(("", [{"function": {"name": "read_file", "arguments": {"path": f"file_{i % 30}.py"}}}], True)),
                updates((f"Looked at file_{i % 30}.py.", None, True)),
            ]
            messages.append({"role": "user", "content": f"look at file {i}"})
            list(run_agent_turn(client, registry, messages, task_state=task_state))

        # Serialized conversation size must stay well below what 25 full
        # file-read tool results (each with headers/line numbers) would
        # cost without compaction -- compact_messages() rewrites stale/
        # duplicate read_file results down to short placeholders.
        serialized_size = len(json.dumps(messages))
        assert serialized_size < 50_000  # generous bound; would be far larger uncompacted

        # files_inspected itself is allowed to keep growing (it's only
        # short path strings) -- what must stay bounded is the rendered
        # summary actually injected into the system prompt every turn.
        summary_lines = task_state.summarize().splitlines()
        files_line = next(line for line in summary_lines if line.startswith("Files inspected:"))
        assert len(files_line.split(",")) <= 10  # bounded by MAX_FILES_IN_SUMMARY


class TestTaskNewDoesNotLeakOldState:
    def test_new_task_state_does_not_reference_old_data(self):
        from agent.cli import _fresh_session_state

        project_root = None
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            project = ProjectRoot(Path(tmp))
            tracker1, task_state1, registry1, messages1 = _fresh_session_state(project, Path(tmp))
            task_state1.note_file_inspected("secret_a.py")
            task_state1.goal = "Old goal"
            messages1.append({"role": "user", "content": "old conversation"})

            tracker2, task_state2, registry2, messages2 = _fresh_session_state(project, Path(tmp))

            # A fresh session must not carry over any of the old one's data
            # or even share the same mutable objects.
            assert task_state2 is not task_state1
            assert task_state2.goal is None
            assert task_state2.files_inspected == []
            assert messages2 is not messages1
            assert not any("old conversation" in m.get("content", "") for m in messages2)
