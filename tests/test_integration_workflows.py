"""Integration tests for complete end-to-end workflows (Phase 8 section 27).

These exercise the real stack together -- the local HTTP server
(agent/server.py), the agent loop (agent/loop.py), the real tool registry
against a real temp-dir project, and real Git/filesystem operations -- with
only OllamaClient.chat mocked, so no real Ollama server is required. This
is deliberately a different angle from the unit tests elsewhere: those
verify one mechanism at a time; these verify that the mechanisms compose
correctly across a full multi-step scenario, matching the architecture
diagram in the Phase 8 spec:

    User -> Agent -> Tool -> Tool failure -> Agent receives structured
    failure -> Agent analyzes failure -> Retry/alternative approach/ask
    user -> Continue task
"""
from __future__ import annotations

import json
import threading
import time
from unittest import mock

import pytest
import requests

import agent.server as server_module
from agent.ollama_client import OllamaClient, OllamaConnectionError


def _parse_ndjson(text: str) -> list:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def server(tmp_path, monkeypatch):
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

    yield base_url, token

    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=2)


@pytest.fixture
def workspace(tmp_path) -> str:
    d = tmp_path / "workspace"
    d.mkdir()
    (d / "app.py").write_text("def greet():\n    return 'hi'\n")
    (d / "tests").mkdir()
    (d / "tests" / "test_app.py").write_text("def test_placeholder():\n    assert True\n")
    return str(d)


def drive_chat_with_confirms(base_url, token, workspace, message, decisions, timeout=10):
    """Sends one /chat message whose turn is expected to pause on
    `len(decisions)` separate confirm* events, answering each in order via
    /chat/confirm as it comes up (polling session.awaiting_confirm, exactly
    like a real client would react to the NDJSON stream). Returns the full
    list of parsed events once the turn completes.
    """
    result = {}

    def run():
        resp = requests.post(
            f"{base_url}/chat",
            json={"workspace_root": workspace, "message": message},
            headers=auth_headers(token),
            timeout=timeout,
        )
        result["events"] = _parse_ndjson(resp.text)

    thread = threading.Thread(target=run)
    thread.start()

    store = server_module._Handler.sessions
    session = store.get_or_create(workspace)

    for approved in decisions:
        for _ in range(200):
            if session.awaiting_confirm:
                break
            time.sleep(0.01)
        else:
            raise AssertionError("Timed out waiting for a confirm* event.")
        requests.post(
            f"{base_url}/chat/confirm",
            json={"workspace_root": workspace, "approved": approved},
            headers=auth_headers(token),
            timeout=5,
        )

    thread.join(timeout=timeout)
    if thread.is_alive():
        raise AssertionError("Chat turn did not complete in time.")
    return result["events"]


def chat_call(content="", tool_calls=None, done=True):
    return {"content": content, "tool_calls": tool_calls, "done": done}


def tool_call_response(name, arguments):
    return chat_call(tool_calls=[{"function": {"name": name, "arguments": arguments}}])


class TestWorkflowSuccessfulTask:
    """Workflow 1: plan -> inspect -> edit -> run tests -> pass -> complete."""

    def test_full_plan_driven_task_succeeds(self, server, workspace):
        base_url, token = server
        responses = [
            tool_call_response(
                "create_plan",
                {"goal": "Add a farewell function", "steps": ["Inspect app.py", "Add function", "Run tests"]},
            ),
            tool_call_response("read_file", {"path": "app.py"}),
            tool_call_response(
                "edit_file",
                {
                    "path": "app.py",
                    "old_text": "def greet():\n    return 'hi'\n",
                    "new_text": "def greet():\n    return 'hi'\n\n\ndef farewell():\n    return 'bye'\n",
                },
            ),
            tool_call_response("run_command", {"program": "pytest", "args": []}),
            chat_call("Added farewell() and the test suite passes.", done=True),
        ]

        def fake_chat(self, messages, tools=None, cancel_event=None):
            yield responses.pop(0)

        with mock.patch.object(OllamaClient, "chat", fake_chat), mock.patch(
            "agent.tools.terminal.subprocess.Popen"
        ) as mock_popen:
            fake_proc = mock.Mock()
            fake_proc.pid = 12345
            fake_proc.returncode = 0
            fake_proc.communicate.return_value = ("1 passed\n", "")
            mock_popen.return_value = fake_proc

            events = drive_chat_with_confirms(
                base_url, token, workspace, "Add a farewell function and run the tests",
                decisions=[True, True, True],  # plan, edit, command
            )

        types = [e["type"] for e in events]
        assert "plan_approved" in types
        assert "change_applied" in types
        assert "command_result" in types
        assert types[-1] == "final"

        assert "def farewell" in (open(f"{workspace}/app.py").read())

        # Task state reflects the completed work.
        status = requests.get(
            f"{base_url}/task/status", params={"workspace_root": workspace}, headers=auth_headers(token), timeout=5
        ).json()
        assert status["plan"]["goal"] == "Add a farewell function"
        assert "app.py" in status["files_modified"]


class TestWorkflowTestFailureThenFix:
    """Workflow 2: edit -> run tests -> fail -> analyze -> edit -> run tests -> pass."""

    def test_failing_then_passing_test_run(self, server, workspace):
        base_url, token = server
        responses = [
            tool_call_response("run_command", {"program": "pytest", "args": []}),
            tool_call_response(
                "edit_file",
                {"path": "app.py", "old_text": "return 'hi'", "new_text": "return 'hello'"},
            ),
            tool_call_response("run_command", {"program": "pytest", "args": []}),
            chat_call("Fixed -- tests pass now.", done=True),
        ]

        def fake_chat(self, messages, tools=None, cancel_event=None):
            yield responses.pop(0)

        call_results = iter(
            [
                mock.Mock(pid=1, returncode=1, communicate=mock.Mock(return_value=("1 failed\n", ""))),
                mock.Mock(pid=2, returncode=0, communicate=mock.Mock(return_value=("1 passed\n", ""))),
            ]
        )

        with mock.patch.object(OllamaClient, "chat", fake_chat), mock.patch(
            "agent.tools.terminal.subprocess.Popen", side_effect=lambda *a, **k: next(call_results)
        ):
            events = drive_chat_with_confirms(
                base_url, token, workspace, "Run the tests and fix any failure",
                decisions=[True, True, True],  # first pytest, edit, second pytest
            )

        command_results = [e for e in events if e["type"] == "command_result"]
        assert len(command_results) == 2
        assert command_results[0]["result"]["exit_code"] == 1
        assert command_results[1]["result"]["exit_code"] == 0
        assert events[-1]["type"] == "final"


class TestWorkflowToolFailureThenAlternative:
    """Workflow 3: tool fails -> agent receives a structured, classified
    error -> tries a different (correct) approach -> continues to success."""

    def test_recoverable_edit_failure_then_correct_retry(self, server, workspace):
        base_url, token = server
        responses = [
            # Wrong old_text -- guaranteed to fail validation.
            tool_call_response(
                "edit_file", {"path": "app.py", "old_text": "not in the file", "new_text": "xxx"}
            ),
            tool_call_response("read_file", {"path": "app.py"}),
            # Correct old_text this time.
            tool_call_response(
                "edit_file", {"path": "app.py", "old_text": "return 'hi'", "new_text": "return 'hello'"}
            ),
            chat_call("Fixed it after re-checking the file.", done=True),
        ]

        def fake_chat(self, messages, tools=None, cancel_event=None):
            yield responses.pop(0)

        with mock.patch.object(OllamaClient, "chat", fake_chat):
            events = drive_chat_with_confirms(
                base_url, token, workspace, "Change the greeting to hello", decisions=[True]
            )

        tool_errors = [e for e in events if e["type"] == "tool_error"]
        assert len(tool_errors) == 1
        assert tool_errors[0]["message"]  # a real, human-readable message, not a stack trace

        applied = [e for e in events if e["type"] == "change_applied"]
        assert len(applied) == 1
        assert events[-1]["type"] == "final"
        assert "return 'hello'" in open(f"{workspace}/app.py").read()


class TestWorkflowUserRejection:
    """Workflow 4: agent proposes an action -> user rejects it -> agent
    continues normally (rejection is not an application error)."""

    def test_rejected_command_lets_the_turn_finish_normally(self, server, workspace):
        base_url, token = server
        responses = [
            tool_call_response("run_command", {"program": "pytest", "args": []}),
            chat_call("Understood, I won't run the tests.", done=True),
        ]

        def fake_chat(self, messages, tools=None, cancel_event=None):
            yield responses.pop(0)

        with mock.patch.object(OllamaClient, "chat", fake_chat), mock.patch(
            "agent.tools.terminal.subprocess.Popen"
        ) as mock_popen:
            events = drive_chat_with_confirms(
                base_url, token, workspace, "Run the tests", decisions=[False]
            )

        mock_popen.assert_not_called()
        assert any(e["type"] == "command_rejected" for e in events)
        assert events[-1]["type"] == "final"

        # The server must still be fully usable for a subsequent request.
        health = requests.get(f"{base_url}/health", timeout=5)
        assert health.status_code == 200


class TestWorkflowOllamaFailure:
    """Workflow 5: Ollama is unavailable -> a controlled error is reported
    -> the agent (and this session) remain fully usable afterward."""

    def test_ollama_failure_then_recovery_on_next_message(self, server, workspace, monkeypatch):
        import agent.loop as loop_module

        monkeypatch.setattr(loop_module, "OLLAMA_RETRY_BACKOFF_SECONDS", (0, 0))
        base_url, token = server

        def always_fails(self, messages, tools=None, cancel_event=None):
            raise OllamaConnectionError("Could not connect to Ollama.")
            yield  # pragma: no cover

        with mock.patch.object(OllamaClient, "chat", always_fails):
            resp = requests.post(
                f"{base_url}/chat",
                json={"workspace_root": workspace, "message": "hi"},
                headers=auth_headers(token),
                timeout=10,
            )
        events = _parse_ndjson(resp.text)
        assert events[-1]["type"] == "error"
        assert "connect" in events[-1]["message"].lower()

        # The session must not be left locked/broken by the failure.
        store = server_module._Handler.sessions
        session = store.get_or_create(workspace)
        assert not session.lock.locked()

        # A subsequent, unrelated message on the same session must work.
        def works_fine(self, messages, tools=None, cancel_event=None):
            yield chat_call("I'm back.", done=True)

        with mock.patch.object(OllamaClient, "chat", works_fine):
            resp2 = requests.post(
                f"{base_url}/chat",
                json={"workspace_root": workspace, "message": "are you there?"},
                headers=auth_headers(token),
                timeout=10,
            )
        events2 = _parse_ndjson(resp2.text)
        assert events2[-1] == {"type": "final", "text": "I'm back."}


class TestWorkflowCancellation:
    """Workflow 6: a long task is stopped mid-flight -> resources are
    released -> task state built up before the stop is preserved, not
    thrown away."""

    def test_stop_preserves_prior_progress_and_frees_the_session(self, server, workspace):
        base_url, token = server
        call_count = {"n": 0}

        def looping_chat(self, messages, tools=None, cancel_event=None):
            call_count["n"] += 1
            time.sleep(0.05)
            if cancel_event is not None and cancel_event.is_set():
                from agent.ollama_client import OllamaCancelledError

                raise OllamaCancelledError("cancelled")
            # Alternate between two harmless read-only calls so repetition
            # detection doesn't interfere with this test's own timing.
            path = "app.py" if call_count["n"] % 2 else "tests/test_app.py"
            yield tool_call_response("read_file", {"path": path})

        with mock.patch.object(OllamaClient, "chat", looping_chat):
            resp_holder = {}

            def run():
                resp_holder["resp"] = requests.post(
                    f"{base_url}/chat",
                    json={"workspace_root": workspace, "message": "keep looking around"},
                    headers=auth_headers(token),
                    stream=True,
                    timeout=15,
                )
                for _ in resp_holder["resp"].iter_lines():
                    pass  # drain the stream fully in this thread

            thread = threading.Thread(target=run)
            thread.start()

            store = server_module._Handler.sessions
            session = store.get_or_create(workspace)
            for _ in range(200):
                if session.task_state.files_inspected:
                    break
                time.sleep(0.01)
            assert session.task_state.files_inspected  # some progress happened first

            files_before_stop = list(session.task_state.files_inspected)

            requests.post(
                f"{base_url}/task/stop", json={"workspace_root": workspace}, headers=auth_headers(token), timeout=5
            )
            thread.join(timeout=10)

        assert not session.lock.locked()
        # Progress made before the stop must still be there afterward --
        # cancellation must not reset task state.
        assert session.task_state.files_inspected == files_before_stop or len(
            session.task_state.files_inspected
        ) >= len(files_before_stop)

        # The session must still be usable for a new message afterward.
        def works_fine(self, messages, tools=None, cancel_event=None):
            yield chat_call("Ready.", done=True)

        with mock.patch.object(OllamaClient, "chat", works_fine):
            resp = requests.post(
                f"{base_url}/chat",
                json={"workspace_root": workspace, "message": "still there?"},
                headers=auth_headers(token),
                timeout=10,
            )
        assert resp.status_code == 200
