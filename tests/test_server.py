"""Tests for the local HTTP agent server (agent/server.py) that the VS Code
extension talks to.

These start a real ThreadingHTTPServer on an OS-assigned loopback port and
drive it with real HTTP requests (via `requests`) -- the only thing mocked
out is OllamaClient.chat/check_connection, so no real Ollama server or model
download is required. This exercises the actual request parsing, chunked
NDJSON streaming, auth, and session-isolation code paths, not a reimplementation
of them.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest import mock

import pytest
import requests

import agent.server as server_module
from agent.ollama_client import OllamaCancelledError, OllamaClient


def _parse_ndjson(text: str) -> list:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


@pytest.fixture
def server(tmp_path, monkeypatch):
    """Starts a real server bound to 127.0.0.1:<ephemeral>, with its token
    file redirected into tmp_path so tests never touch the real
    ~/.code-agent/server.json. Yields (base_url, token). Torn down cleanly
    at the end of the test via httpd.shutdown().
    """
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
    (d / "README.md").write_text("hello\n")
    return str(d)


def auth_headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


class TestBindHost:
    def test_binds_only_to_loopback(self, server):
        base_url, _ = server
        assert base_url.startswith("http://127.0.0.1:")

    def test_never_binds_to_0_0_0_0(self):
        assert server_module.BIND_HOST == "127.0.0.1"


class TestHealthEndpoint:
    def test_health_reports_model_and_status(self, server):
        base_url, _ = server
        resp = requests.get(f"{base_url}/health", timeout=5)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["backend"] == "ollama"
        assert "model" in body
        assert body["ollama_connected"] is True

    def test_health_works_without_a_token(self, server):
        base_url, _ = server
        resp = requests.get(f"{base_url}/health", timeout=5)
        assert resp.status_code == 200

    def test_health_without_workspace_root_omits_workspace_field(self, server):
        base_url, _ = server
        resp = requests.get(f"{base_url}/health", timeout=5)
        assert "workspace" not in resp.json()

    def test_health_reports_workspace_ok(self, server, workspace):
        base_url, _ = server
        resp = requests.get(f"{base_url}/health", params={"workspace_root": workspace}, timeout=5)
        assert resp.json()["workspace"] == "ok"

    def test_health_reports_workspace_not_found(self, server, tmp_path):
        base_url, _ = server
        missing = str(tmp_path / "does-not-exist")
        resp = requests.get(f"{base_url}/health", params={"workspace_root": missing}, timeout=5)
        assert resp.json()["workspace"] == "not_found"


class TestTaskIdInStatus:
    def test_task_status_includes_a_task_id(self, server, workspace):
        base_url, token = server
        resp = requests.get(
            f"{base_url}/task/status",
            params={"workspace_root": workspace},
            headers=auth_headers(token),
            timeout=5,
        )
        body = resp.json()
        assert isinstance(body["task_id"], str)
        assert len(body["task_id"]) > 0

    def test_task_id_changes_after_task_new(self, server, workspace):
        base_url, token = server
        first = requests.get(
            f"{base_url}/task/status", params={"workspace_root": workspace}, headers=auth_headers(token), timeout=5
        ).json()["task_id"]

        requests.post(
            f"{base_url}/task/new", json={"workspace_root": workspace}, headers=auth_headers(token), timeout=5
        )

        second = requests.get(
            f"{base_url}/task/status", params={"workspace_root": workspace}, headers=auth_headers(token), timeout=5
        ).json()["task_id"]

        assert first != second


class TestAuthentication:
    @pytest.mark.parametrize(
        "method,path",
        [
            ("get", "/task/status?workspace_root=/tmp"),
            ("post", "/chat"),
            ("post", "/chat/confirm"),
            ("post", "/task/stop"),
            ("post", "/task/new"),
        ],
    )
    def test_missing_token_rejected(self, server, method, path):
        base_url, _ = server
        resp = getattr(requests, method)(f"{base_url}{path}", timeout=5)
        assert resp.status_code == 401

    def test_invalid_token_rejected(self, server, workspace):
        base_url, _ = server
        resp = requests.get(
            f"{base_url}/task/status",
            params={"workspace_root": workspace},
            headers=auth_headers("not-the-real-token"),
            timeout=5,
        )
        assert resp.status_code == 401

    def test_valid_token_accepted(self, server, workspace):
        base_url, token = server
        resp = requests.get(
            f"{base_url}/task/status",
            params={"workspace_root": workspace},
            headers=auth_headers(token),
            timeout=5,
        )
        assert resp.status_code == 200


class TestMalformedRequests:
    def test_chat_missing_workspace_root(self, server):
        base_url, token = server
        resp = requests.post(
            f"{base_url}/chat", json={"message": "hi"}, headers=auth_headers(token), timeout=5
        )
        assert resp.status_code == 400

    def test_chat_nonexistent_workspace_root(self, server, tmp_path):
        base_url, token = server
        resp = requests.post(
            f"{base_url}/chat",
            json={"workspace_root": str(tmp_path / "does-not-exist"), "message": "hi"},
            headers=auth_headers(token),
            timeout=5,
        )
        assert resp.status_code == 400

    def test_chat_missing_message(self, server, workspace):
        base_url, token = server
        resp = requests.post(
            f"{base_url}/chat",
            json={"workspace_root": workspace},
            headers=auth_headers(token),
            timeout=5,
        )
        assert resp.status_code == 400

    def test_invalid_json_body_rejected(self, server, workspace):
        base_url, token = server
        resp = requests.post(
            f"{base_url}/chat",
            data=b"{not valid json",
            headers={**auth_headers(token), "Content-Type": "application/json"},
            timeout=5,
        )
        assert resp.status_code == 400

    def test_unknown_path_404(self, server, workspace):
        base_url, token = server
        resp = requests.get(f"{base_url}/nope", headers=auth_headers(token), timeout=5)
        assert resp.status_code == 404


class TestChatEndpoint:
    def test_simple_reply_streams_final_event(self, server, workspace):
        base_url, token = server

        def fake_chat(self, messages, tools=None, cancel_event=None):
            yield {"content": "Hello there", "tool_calls": None, "done": True}

        with mock.patch.object(OllamaClient, "chat", fake_chat):
            resp = requests.post(
                f"{base_url}/chat",
                json={"workspace_root": workspace, "message": "hi"},
                headers=auth_headers(token),
                timeout=5,
            )

        assert resp.status_code == 200
        events = _parse_ndjson(resp.text)
        types = [e["type"] for e in events]
        assert "final" in types
        final = next(e for e in events if e["type"] == "final")
        assert final["text"] == "Hello there"

    def test_second_concurrent_chat_rejected_with_409(self, server, workspace):
        base_url, token = server
        started = threading.Event()
        release = threading.Event()

        def fake_chat(self, messages, tools=None, cancel_event=None):
            started.set()
            release.wait(timeout=5)
            yield {"content": "done", "tool_calls": None, "done": True}

        results = {}

        def do_first_chat():
            with mock.patch.object(OllamaClient, "chat", fake_chat):
                results["first"] = requests.post(
                    f"{base_url}/chat",
                    json={"workspace_root": workspace, "message": "hi"},
                    headers=auth_headers(token),
                    timeout=5,
                )

        t = threading.Thread(target=do_first_chat)
        t.start()
        started.wait(timeout=5)

        resp2 = requests.post(
            f"{base_url}/chat",
            json={"workspace_root": workspace, "message": "again"},
            headers=auth_headers(token),
            timeout=5,
        )
        assert resp2.status_code == 409

        release.set()
        t.join(timeout=5)
        assert results["first"].status_code == 200


class TestConnectionLoss:
    """Phase 8 section 22: a request that dies mid-stream (client closed
    the connection) must not corrupt session state -- the partial turn's
    user message must be rolled back, the session lock released, and the
    server must remain fully usable for the next request."""

    def test_client_disconnect_mid_stream_rolls_back_and_releases_lock(self, server, workspace):
        """Deterministically simulates a broken connection by making the
        write path itself raise BrokenPipeError partway through a turn --
        exercising exactly the exception-handling code in _handle_chat,
        without depending on real OS-level socket/RST timing (which proved
        flaky across environments in an earlier version of this test)."""
        base_url, token = server

        def fake_chat(self, messages, tools=None, cancel_event=None):
            yield {"content": "first chunk", "tool_calls": None, "done": False}
            yield {"content": "second chunk", "tool_calls": None, "done": True}

        call_count = {"n": 0}
        real_write_chunk_line = server_module._Handler._write_chunk_line

        def flaky_write_chunk_line(self_handler, payload):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return real_write_chunk_line(self_handler, payload)
            raise BrokenPipeError("simulated broken pipe")

        def flaky_end_chunks(self_handler):
            raise BrokenPipeError("simulated broken pipe")

        with mock.patch.object(OllamaClient, "chat", fake_chat), mock.patch.object(
            server_module._Handler, "_write_chunk_line", flaky_write_chunk_line
        ), mock.patch.object(server_module._Handler, "_end_chunks", flaky_end_chunks):
            with pytest.raises(requests.exceptions.RequestException):
                requests.post(
                    f"{base_url}/chat",
                    json={"workspace_root": workspace, "message": "hi"},
                    headers=auth_headers(token),
                    timeout=5,
                )

        store = server_module._Handler.sessions
        session = store.get_or_create(workspace)
        assert not session.lock.locked()  # lock was released, not left held forever
        assert session.messages[-1]["role"] != "user" or session.messages[-1]["content"] != "hi"

        # The server must still be fully usable afterward.
        resp = requests.get(f"{base_url}/health", timeout=5)
        assert resp.status_code == 200

        def works_fine(self, messages, tools=None, cancel_event=None):
            yield {"content": "back to normal", "tool_calls": None, "done": True}

        with mock.patch.object(OllamaClient, "chat", works_fine):
            resp2 = requests.post(
                f"{base_url}/chat",
                json={"workspace_root": workspace, "message": "still there?"},
                headers=auth_headers(token),
                timeout=5,
            )
        assert resp2.status_code == 200


class TestTaskStatus:
    def test_empty_task_state(self, server, workspace):
        base_url, token = server
        resp = requests.get(
            f"{base_url}/task/status",
            params={"workspace_root": workspace},
            headers=auth_headers(token),
            timeout=5,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["goal"] is None
        assert body["plan"] is None
        assert body["files_inspected"] == []


class TestWorkspaceIsolation:
    def test_sessions_are_independent(self, server, tmp_path):
        base_url, token = server
        ws_a = tmp_path / "project-a"
        ws_b = tmp_path / "project-b"
        ws_a.mkdir()
        ws_b.mkdir()

        store = server_module._Handler.sessions
        session_a = store.get_or_create(str(ws_a))
        session_b = store.get_or_create(str(ws_b))

        assert session_a is not session_b
        assert session_a.project.root != session_b.project.root

        session_a.task_state.goal = "Goal for project A only"
        session_a.task_state.note_file_inspected("secret_a.py")

        assert session_b.task_state.goal is None
        assert session_b.task_state.files_inspected == []

    def test_get_or_create_returns_same_session_for_same_path(self, server, workspace):
        store = server_module._Handler.sessions
        s1 = store.get_or_create(workspace)
        s2 = store.get_or_create(workspace)
        assert s1 is s2


class TestTaskStop:
    def test_stop_cancels_a_looping_turn(self, server, workspace):
        base_url, token = server
        call_count = {"n": 0}

        def fake_chat(self, messages, tools=None, cancel_event=None):
            call_count["n"] += 1
            time.sleep(0.05)
            if cancel_event is not None and cancel_event.is_set():
                raise OllamaCancelledError("cancelled")
            yield {
                "content": "",
                "tool_calls": [{"function": {"name": "list_files", "arguments": {}}}],
                "done": True,
            }

        with mock.patch.object(OllamaClient, "chat", fake_chat):
            resp = requests.post(
                f"{base_url}/chat",
                json={"workspace_root": workspace, "message": "loop forever"},
                headers=auth_headers(token),
                stream=True,
                timeout=15,
            )

            lines = resp.iter_lines(decode_unicode=True)
            first_line = next(lines)
            assert json.loads(first_line)["type"] == "tool_call"

            stop_resp = requests.post(
                f"{base_url}/task/stop",
                json={"workspace_root": workspace},
                headers=auth_headers(token),
                timeout=5,
            )
            assert stop_resp.status_code == 200

            remaining_types = []
            for line in lines:
                if line.strip():
                    remaining_types.append(json.loads(line)["type"])

        assert "cancelled" in remaining_types
        # Cancellation should have cut the loop well short of the 10-iteration
        # safety limit -- otherwise this test isn't actually proving anything.
        assert call_count["n"] < 10


class TestTaskNew:
    def test_resets_session_state(self, server, workspace):
        base_url, token = server
        store = server_module._Handler.sessions
        session = store.get_or_create(workspace)
        session.task_state.goal = "Some in-progress goal"
        session.task_state.note_file_inspected("x.py")

        resp = requests.post(
            f"{base_url}/task/new",
            json={"workspace_root": workspace},
            headers=auth_headers(token),
            timeout=5,
        )
        assert resp.status_code == 200

        refreshed = store.get_or_create(workspace)
        assert refreshed.task_state.goal is None
        assert refreshed.task_state.files_inspected == []


class TestTaskMode:
    def test_new_session_starts_in_manual_mode(self, server, workspace):
        base_url, token = server
        resp = requests.get(
            f"{base_url}/task/status", params={"workspace_root": workspace}, headers=auth_headers(token), timeout=5
        )
        assert resp.json()["mode"] == "manual"

    def test_setting_auto_mode_is_reflected_in_status(self, server, workspace):
        base_url, token = server
        resp = requests.post(
            f"{base_url}/task/mode",
            json={"workspace_root": workspace, "mode": "auto"},
            headers=auth_headers(token),
            timeout=5,
        )
        assert resp.status_code == 200
        assert resp.json() == {"ok": True, "mode": "auto"}

        status = requests.get(
            f"{base_url}/task/status", params={"workspace_root": workspace}, headers=auth_headers(token), timeout=5
        )
        assert status.json()["mode"] == "auto"

    def test_switching_back_to_manual_works(self, server, workspace):
        base_url, token = server
        requests.post(
            f"{base_url}/task/mode",
            json={"workspace_root": workspace, "mode": "auto"},
            headers=auth_headers(token),
            timeout=5,
        )
        resp = requests.post(
            f"{base_url}/task/mode",
            json={"workspace_root": workspace, "mode": "manual"},
            headers=auth_headers(token),
            timeout=5,
        )
        assert resp.json()["mode"] == "manual"

    def test_invalid_mode_value_rejected(self, server, workspace):
        base_url, token = server
        resp = requests.post(
            f"{base_url}/task/mode",
            json={"workspace_root": workspace, "mode": "yolo"},
            headers=auth_headers(token),
            timeout=5,
        )
        assert resp.status_code == 400

        # Rejected value must not have been applied.
        status = requests.get(
            f"{base_url}/task/status", params={"workspace_root": workspace}, headers=auth_headers(token), timeout=5
        )
        assert status.json()["mode"] == "manual"

    def test_missing_workspace_root_rejected(self, server):
        base_url, token = server
        resp = requests.post(
            f"{base_url}/task/mode", json={"mode": "auto"}, headers=auth_headers(token), timeout=5
        )
        assert resp.status_code == 400

    def test_requires_authentication(self, server, workspace):
        base_url, _ = server
        resp = requests.post(
            f"{base_url}/task/mode", json={"workspace_root": workspace, "mode": "auto"}, timeout=5
        )
        assert resp.status_code == 401

    def test_mode_survives_task_new(self, server, workspace):
        """Mode is a session preference, not task data -- /task/new must
        not reset it back to manual."""
        base_url, token = server
        requests.post(
            f"{base_url}/task/mode",
            json={"workspace_root": workspace, "mode": "auto"},
            headers=auth_headers(token),
            timeout=5,
        )
        requests.post(
            f"{base_url}/task/new", json={"workspace_root": workspace}, headers=auth_headers(token), timeout=5
        )
        status = requests.get(
            f"{base_url}/task/status", params={"workspace_root": workspace}, headers=auth_headers(token), timeout=5
        )
        assert status.json()["mode"] == "auto"

    def test_mode_is_isolated_per_workspace(self, server, tmp_path):
        base_url, token = server
        ws_a = tmp_path / "project-a"
        ws_b = tmp_path / "project-b"
        ws_a.mkdir()
        ws_b.mkdir()

        requests.post(
            f"{base_url}/task/mode",
            json={"workspace_root": str(ws_a), "mode": "auto"},
            headers=auth_headers(token),
            timeout=5,
        )

        status_a = requests.get(
            f"{base_url}/task/status", params={"workspace_root": str(ws_a)}, headers=auth_headers(token), timeout=5
        )
        status_b = requests.get(
            f"{base_url}/task/status", params={"workspace_root": str(ws_b)}, headers=auth_headers(token), timeout=5
        )
        assert status_a.json()["mode"] == "auto"
        assert status_b.json()["mode"] == "manual"


class TestConfirmFlow:
    def test_edit_approval_round_trip(self, server, tmp_path):
        base_url, token = server
        ws = tmp_path / "editable"
        ws.mkdir()
        (ws / "target.py").write_text("value = 1\n")

        call_count = {"n": 0}

        def fake_chat(self, messages, tools=None, cancel_event=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                yield {
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "edit_file",
                                "arguments": {
                                    "path": "target.py",
                                    "old_text": "value = 1\n",
                                    "new_text": "value = 2\n",
                                },
                            }
                        }
                    ],
                    "done": True,
                }
            else:
                yield {"content": "Done editing.", "tool_calls": None, "done": True}

        events_holder = {}

        def do_chat():
            with mock.patch.object(OllamaClient, "chat", fake_chat):
                resp = requests.post(
                    f"{base_url}/chat",
                    json={"workspace_root": str(ws), "message": "bump the value"},
                    headers=auth_headers(token),
                    timeout=10,
                )
            events_holder["events"] = _parse_ndjson(resp.text)

        t = threading.Thread(target=do_chat)
        t.start()

        store = server_module._Handler.sessions
        session = store.get_or_create(str(ws))
        for _ in range(200):
            if session.awaiting_confirm:
                break
            time.sleep(0.01)
        assert session.awaiting_confirm is True

        confirm_resp = requests.post(
            f"{base_url}/chat/confirm",
            json={"workspace_root": str(ws), "approved": True},
            headers=auth_headers(token),
            timeout=5,
        )
        assert confirm_resp.status_code == 200

        t.join(timeout=5)
        types = [e["type"] for e in events_holder["events"]]
        assert "confirm" in types
        assert "change_applied" in types
        assert (ws / "target.py").read_text() == "value = 2\n"

    def test_auto_mode_applies_edit_without_a_chat_confirm_call(self, server, tmp_path):
        """The end-to-end proof: with mode=auto, POST /chat alone (no
        /chat/confirm at all) is enough for the edit to actually land on
        disk."""
        base_url, token = server
        ws = tmp_path / "auto-editable"
        ws.mkdir()
        (ws / "target.py").write_text("value = 1\n")

        requests.post(
            f"{base_url}/task/mode",
            json={"workspace_root": str(ws), "mode": "auto"},
            headers=auth_headers(token),
            timeout=5,
        )

        call_count = {"n": 0}

        def fake_chat(self, messages, tools=None, cancel_event=None):
            call_count["n"] += 1
            if call_count["n"] == 1:
                yield {
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "edit_file",
                                "arguments": {
                                    "path": "target.py",
                                    "old_text": "value = 1\n",
                                    "new_text": "value = 2\n",
                                },
                            }
                        }
                    ],
                    "done": True,
                }
            else:
                yield {"content": "Done.", "tool_calls": None, "done": True}

        with mock.patch.object(OllamaClient, "chat", fake_chat):
            resp = requests.post(
                f"{base_url}/chat",
                json={"workspace_root": str(ws), "message": "bump the value"},
                headers=auth_headers(token),
                timeout=10,
            )

        events = _parse_ndjson(resp.text)
        confirm_event = next(e for e in events if e["type"] == "confirm")
        assert confirm_event["auto_approved"] is True
        assert any(e["type"] == "change_applied" for e in events)
        assert (ws / "target.py").read_text() == "value = 2\n"

    def test_confirm_with_no_pending_approval_rejected(self, server, workspace):
        base_url, token = server
        resp = requests.post(
            f"{base_url}/chat/confirm",
            json={"workspace_root": workspace, "approved": True},
            headers=auth_headers(token),
            timeout=5,
        )
        assert resp.status_code == 409


class TestOllamaUnavailable:
    def test_health_reports_ollama_disconnected(self, tmp_path, monkeypatch):
        token_dir = tmp_path / "home" / ".code-agent"
        monkeypatch.setattr(server_module, "TOKEN_DIR", token_dir)
        monkeypatch.setattr(server_module, "TOKEN_FILE", token_dir / "server.json")
        monkeypatch.setattr(OllamaClient, "check_connection", lambda self: False)

        httpd = server_module.build_server(port=0)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = httpd.server_address[:2]
            resp = requests.get(f"http://{host}:{port}/health", timeout=5)
            assert resp.status_code == 200
            assert resp.json()["ollama_connected"] is False
        finally:
            httpd.shutdown()
            httpd.server_close()
            thread.join(timeout=2)
