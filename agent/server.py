"""Local-only HTTP server exposing the existing agent loop to the VS Code
extension (Phase 7).

This module does not reimplement any agent logic. It drives the exact same
`run_agent_turn()` generator that agent/cli.py's `render_turn()` drives, and
builds sessions with the exact same `_fresh_session_state()` cli.py uses for
its own `/new` command -- the HTTP layer only adds request/response plumbing,
per-workspace session isolation, and local authentication around that
existing core.

Security posture (see CLAUDE.md for the full rationale):
  - The server binds ONLY to 127.0.0.1. There is no configuration option to
    change this -- it is a hardcoded constant, not something an env var or
    CLI flag can override, so this server can never be reachable from the
    LAN or the internet.
  - Every request must present the bearer token written to
    ~/.code-agent/server.json (chmod 0600) when this process started. The
    token is generated fresh per run with `secrets.token_hex` and is never
    hardcoded anywhere in this repo or in the VS Code extension source.
  - This server does not re-validate or relax any of the existing security
    boundaries (project-root confinement, sensitive-file protection, the
    run_command allowlist, the Git tool allowlist, or edit/command/Git
    approval). It is a thin transport around the same ProjectRoot, the same
    ToolRegistry, and the same run_agent_turn() the CLI already uses --
    every approval the CLI would ask for over stdin is asked for here too,
    just over HTTP instead of console.input().
"""
from __future__ import annotations

import json
import os
import queue
import secrets
import signal
import stat
import sys
import threading
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Optional
from urllib.parse import parse_qs, urlparse

from . import context_budget
from .cli import _fresh_session_state
from .loop import run_agent_turn
from .logging_config import get_logger
from .ollama_client import (
    DEFAULT_HOST,
    DEFAULT_MODEL,
    OllamaClient,
    keep_alive_from_env,
    temperature_from_env,
    timeout_from_env,
    top_p_from_env,
)
from .project import ProjectRoot
from .tools.state import FileStateTracker
from .task_state import TaskState

_logger = get_logger("server")

# Hardcoded, not configurable -- see module docstring. Never bind to
# 0.0.0.0 or any other interface, under any circumstance.
BIND_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

TOKEN_DIR = Path.home() / ".code-agent"
TOKEN_FILE = TOKEN_DIR / "server.json"

CONFIRM_EVENT_TYPES = {
    "confirm", "confirm_file_op", "confirm_command", "confirm_git_operation", "confirm_plan",
}

VALID_MODES = {"manual", "auto"}

# How long a /chat handler thread waits on the confirm queue before checking
# cancel_event again -- keeps Stop Task responsive even while paused on an
# approval that the extension never answers.
CONFIRM_POLL_SECONDS = 0.25
# How long /task/new waits for an in-flight turn to notice cancellation and
# release the session lock before giving up and resetting anyway.
NEW_TASK_DRAIN_TIMEOUT_SECONDS = 5.0


@dataclass
class Session:
    """All state for one VS Code workspace. Sessions never share any of
    this -- each is built from its own ProjectRoot rooted at its own
    workspace folder, so nothing here can leak between two open projects.
    """

    project: ProjectRoot
    project_root_path: Path
    tracker: FileStateTracker
    task_state: TaskState
    registry: object
    messages: list
    lock: threading.Lock = field(default_factory=threading.Lock)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    confirm_queue: "queue.Queue" = field(default_factory=queue.Queue)
    awaiting_confirm: bool = False
    # "manual" (default) or "auto" -- see loop.py's auto_approve_edits.
    # Deliberately NOT touched by SessionStore.reset() below, so it
    # survives /task/new as a UI preference rather than task data, but
    # every session still starts "manual" the first time it's created
    # (safety-by-default on every fresh `code-agent serve` start).
    mode: str = "manual"


class SessionStore:
    """Keyed by resolved absolute project root path. Guards session
    creation with its own lock; each Session then guards its own turn
    execution with its own lock, so unrelated workspaces never block each
    other and never touch each other's messages/task_state/tracker.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.Lock()

    def get_or_create(self, workspace_root: str) -> Session:
        resolved = str(Path(workspace_root).resolve())
        with self._lock:
            session = self._sessions.get(resolved)
            if session is None:
                project_root_path = Path(resolved)
                project = ProjectRoot(project_root_path)
                tracker, task_state, registry, messages = _fresh_session_state(
                    project, project_root_path
                )
                session = Session(
                    project=project,
                    project_root_path=project_root_path,
                    tracker=tracker,
                    task_state=task_state,
                    registry=registry,
                    messages=messages,
                )
                self._sessions[resolved] = session
            return session

    def reset(self, session: Session) -> None:
        tracker, task_state, registry, messages = _fresh_session_state(
            session.project, session.project_root_path
        )
        session.tracker = tracker
        session.task_state = task_state
        session.registry = registry
        session.messages = messages
        session.cancel_event = threading.Event()


def _to_jsonable(obj):
    """Recursively convert dataclasses/Path objects into plain JSON-safe
    structures. Every event run_agent_turn yields is built from this
    project's existing dataclasses (ProposedChange, ApprovedCommand,
    CommandExecutionResult, ProposedGitOperation, Plan/PlanStep) -- this
    function never invents new fields, it just makes the same data
    JSON-serializable.
    """
    if is_dataclass_instance(obj):
        return {f: _to_jsonable(v) for f, v in obj.__dict__.items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def is_dataclass_instance(obj) -> bool:
    return hasattr(obj, "__dataclass_fields__")


def _write_token_file(host: str, port: int, token: str) -> None:
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"host": host, "port": port, "token": token, "pid": os.getpid()}
    fd = os.open(str(TOKEN_FILE), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
    except Exception:
        os.close(fd)
        raise
    os.chmod(TOKEN_FILE, stat.S_IRUSR | stat.S_IWUSR)


def _remove_token_file() -> None:
    try:
        TOKEN_FILE.unlink()
    except FileNotFoundError:
        pass


def _bind_socket(port: int) -> ThreadingHTTPServer:
    """Try the requested port first; fall back to an OS-assigned ephemeral
    port if it's already in use, rather than crashing (see the spec's
    "port already in use" troubleshooting requirement)."""
    try:
        return ThreadingHTTPServer((BIND_HOST, port), _Handler)
    except OSError:
        return ThreadingHTTPServer((BIND_HOST, 0), _Handler)


def _task_status_payload(session: Session) -> dict:
    ts = session.task_state
    plan_payload = None
    if ts.plan is not None:
        plan_payload = {
            "goal": ts.plan.goal,
            "steps": [
                {"id": s.id, "description": s.description, "status": s.status, "note": s.note}
                for s in ts.plan.steps
            ],
            "is_complete": ts.plan.is_complete(),
            "is_blocked": ts.plan.is_blocked(),
        }
    return {
        "goal": ts.goal,
        "task_id": ts.task_id,
        "plan": plan_payload,
        "files_inspected": list(ts.files_inspected),
        "files_modified": list(ts.files_modified),
        "commands_executed": [
            {
                "display": c.display,
                "exit_code": c.exit_code,
                "outcome": c.outcome,
                "is_test": c.is_test,
            }
            for c in ts.commands_executed
        ],
        "errors_encountered": list(ts.errors_encountered),
        "git_operations": list(ts.git_operations),
        "turn_in_progress": session.lock.locked(),
        "mode": session.mode,
    }


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "code-agent-server/1.0"

    # Set by run_server() before the server ever accepts a connection.
    client: OllamaClient
    model: str
    ollama_host: str
    token: str
    sessions: SessionStore

    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        # Quiet by default; startup banner already told the user how to
        # find this server, and our own request logging below (via
        # logging_config, gated on --debug) covers the useful case. This
        # silences BaseHTTPRequestHandler's own unconditional stderr prints.
        pass

    def _new_request_id(self) -> str:
        # Never derived from anything secret (auth token, paths) -- purely
        # a random correlation handle for matching a client-visible error
        # to the corresponding server log line (spec section 24).
        request_id = uuid.uuid4().hex[:8]
        self._request_id = request_id  # noqa: attr-defined-outside-init
        return request_id

    # -- auth -----------------------------------------------------------

    def _authenticated(self) -> bool:
        header = self.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            return False
        supplied = header[len("Bearer "):].strip()
        return secrets.compare_digest(supplied, self.token)

    def _reject_unauthorized(self) -> None:
        _logger.warning(
            "[%s] Rejected unauthenticated request: %s %s",
            getattr(self, "_request_id", "-"), self.command, self.path,
        )
        self._send_json(401, {"error": "Missing or invalid Authorization bearer token."})

    # -- helpers ----------------------------------------------------------

    def _send_json(self, status: int, payload: dict) -> None:
        if status >= 400:
            body_payload = {**payload, "request_id": getattr(self, "_request_id", None)}
            _logger.info(
                "[%s] %s %s -> %d: %s",
                getattr(self, "_request_id", "-"), self.command, self.path, status, payload.get("error"),
            )
        else:
            body_payload = payload
        body = json.dumps(body_payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> Optional[dict]:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None

    def _session_or_400(self, workspace_root: Optional[str]) -> Optional[Session]:
        if not workspace_root or not isinstance(workspace_root, str):
            self._send_json(400, {"error": "workspace_root is required."})
            return None
        path = Path(workspace_root)
        if not path.is_absolute() or not path.is_dir():
            self._send_json(400, {"error": "workspace_root must be an existing absolute directory path."})
            return None
        return self.sessions.get_or_create(workspace_root)

    # -- routing ----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - stdlib method name
        self._new_request_id()
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            qs = parse_qs(parsed.query)
            workspace_root = (qs.get("workspace_root") or [None])[0]
            self._handle_health(workspace_root)
            return
        if not self._authenticated():
            self._reject_unauthorized()
            return
        if parsed.path == "/task/status":
            qs = parse_qs(parsed.query)
            workspace_root = (qs.get("workspace_root") or [None])[0]
            session = self._session_or_400(workspace_root)
            if session is None:
                return
            self._send_json(200, _task_status_payload(session))
            return
        self._send_json(404, {"error": "Not found."})

    def do_POST(self) -> None:  # noqa: N802 - stdlib method name
        self._new_request_id()
        parsed = urlparse(self.path)
        if not self._authenticated():
            self._reject_unauthorized()
            return

        body = self._read_json_body()
        if body is None:
            self._send_json(400, {"error": "Request body must be valid JSON."})
            return

        if parsed.path == "/chat":
            self._handle_chat(body)
            return
        if parsed.path == "/chat/confirm":
            self._handle_chat_confirm(body)
            return
        if parsed.path == "/task/stop":
            self._handle_task_stop(body)
            return
        if parsed.path == "/task/new":
            self._handle_task_new(body)
            return
        if parsed.path == "/task/mode":
            self._handle_task_mode(body)
            return
        self._send_json(404, {"error": "Not found."})

    # -- endpoint implementations ------------------------------------------

    def _handle_health(self, workspace_root: Optional[str] = None) -> None:
        payload = {
            "status": "ok",
            "backend": "ollama",
            "model": self.model,
            "ollama_host": self.ollama_host,
            "ollama_connected": self.client.check_connection(),
        }
        # workspace_root is optional -- omitted entirely when the caller
        # doesn't pass one (e.g. a general "is the agent up at all" check
        # before any workspace is known), so every existing caller of
        # GET /health is unaffected by this field's addition.
        if workspace_root:
            path = Path(workspace_root)
            payload["workspace"] = "ok" if path.is_absolute() and path.is_dir() else "not_found"
        self._send_json(200, payload)

    def _handle_chat(self, body: dict) -> None:
        workspace_root = body.get("workspace_root")
        message = body.get("message")
        session = self._session_or_400(workspace_root)
        if session is None:
            return
        if not message or not isinstance(message, str):
            self._send_json(400, {"error": "message is required."})
            return

        if not session.lock.acquire(blocking=False):
            self._send_json(409, {"error": "A turn is already in progress for this workspace."})
            return

        request_id = getattr(self, "_request_id", "-")
        _logger.info(
            "[%s] chat started: workspace=%s task_id=%s", request_id, workspace_root, session.task_state.task_id
        )
        event_counts: Dict[str, int] = {}
        try:
            session.cancel_event.clear()
            history_len_before = len(session.messages)
            session.messages.append({"role": "user", "content": message})

            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()

            gen = run_agent_turn(
                self.client,
                session.registry,
                session.messages,
                tracker=session.tracker,
                task_state=session.task_state,
                cancel_event=session.cancel_event,
                auto_approve_edits=(session.mode == "auto"),
                max_context_chars=self.max_context_chars,
            )

            send_value = None
            try:
                while True:
                    try:
                        event = gen.send(send_value)
                    except StopIteration:
                        break
                    send_value = None

                    event_counts[event["type"]] = event_counts.get(event["type"], 0) + 1
                    self._write_chunk_line(_to_jsonable(event))

                    if event["type"] in CONFIRM_EVENT_TYPES:
                        if event.get("auto_approved"):
                            # run_agent_turn already resolved this
                            # internally (Auto mode, edits/plans only) --
                            # no need to wait on the confirm queue.
                            send_value = None
                        else:
                            session.awaiting_confirm = True
                            send_value = self._wait_for_confirm(session)
                            session.awaiting_confirm = False
            except (BrokenPipeError, ConnectionResetError):
                _logger.warning("[%s] chat connection lost mid-stream; rolling back this turn.", request_id)
                del session.messages[history_len_before:]
            finally:
                try:
                    self._end_chunks()
                except (BrokenPipeError, ConnectionResetError):
                    # The connection was already gone (this is the normal
                    # case right after the except clause above ran) --
                    # nothing left to notify, and session state is already
                    # rolled back or was never touched.
                    pass
                _logger.info("[%s] chat ended: %s", request_id, event_counts)
        finally:
            session.lock.release()

    def _wait_for_confirm(self, session: Session) -> bool:
        while True:
            if session.cancel_event.is_set():
                return False
            try:
                return session.confirm_queue.get(timeout=CONFIRM_POLL_SECONDS)
            except queue.Empty:
                continue

    def _write_chunk_line(self, payload: dict) -> None:
        data = (json.dumps(payload) + "\n").encode("utf-8")
        self.wfile.write(f"{len(data):x}\r\n".encode("ascii"))
        self.wfile.write(data)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def _end_chunks(self) -> None:
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    def _handle_chat_confirm(self, body: dict) -> None:
        workspace_root = body.get("workspace_root")
        session = self._session_or_400(workspace_root)
        if session is None:
            return
        approved = bool(body.get("approved"))
        if not session.awaiting_confirm:
            self._send_json(409, {"error": "No pending approval for this workspace."})
            return
        try:
            session.confirm_queue.put_nowait(approved)
        except queue.Full:
            pass
        self._send_json(200, {"ok": True})

    def _handle_task_stop(self, body: dict) -> None:
        workspace_root = body.get("workspace_root")
        session = self._session_or_400(workspace_root)
        if session is None:
            return
        session.cancel_event.set()
        if session.awaiting_confirm:
            try:
                session.confirm_queue.put_nowait(False)
            except queue.Full:
                pass
        self._send_json(200, {"ok": True})

    def _handle_task_new(self, body: dict) -> None:
        workspace_root = body.get("workspace_root")
        session = self._session_or_400(workspace_root)
        if session is None:
            return

        session.cancel_event.set()
        if session.awaiting_confirm:
            try:
                session.confirm_queue.put_nowait(False)
            except queue.Full:
                pass

        acquired = session.lock.acquire(timeout=NEW_TASK_DRAIN_TIMEOUT_SECONDS)
        try:
            self.sessions.reset(session)
        finally:
            if acquired:
                session.lock.release()

        self._send_json(200, {"ok": True})

    def _handle_task_mode(self, body: dict) -> None:
        """Sets Manual/Auto mode for a workspace's session -- see
        Session.mode's docstring and loop.py's auto_approve_edits for what
        this actually changes (edits/plans only; commands and every Git
        operation always require individual approval no matter what this
        is set to).
        """
        workspace_root = body.get("workspace_root")
        session = self._session_or_400(workspace_root)
        if session is None:
            return
        mode = body.get("mode")
        if mode not in VALID_MODES:
            self._send_json(400, {"error": f"mode must be one of {sorted(VALID_MODES)}."})
            return
        session.mode = mode
        self._send_json(200, {"ok": True, "mode": session.mode})


def build_server(port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """Construct and bind the server (auth token generated, token file
    written, socket bound) without starting to serve requests. Split out
    from run_server() so tests -- and any other embedder -- can start and
    cleanly shut down a server instance without relying on serve_forever()
    blocking the whole process forever. Only ever binds to 127.0.0.1 -- see
    BIND_HOST above.
    """
    host = os.environ.get("OLLAMA_HOST", DEFAULT_HOST)
    model = os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
    client = OllamaClient(
        host=host,
        model=model,
        timeout=timeout_from_env(),
        keep_alive=keep_alive_from_env(),
        temperature=temperature_from_env(),
        top_p=top_p_from_env(),
    )

    token = secrets.token_hex(32)

    _Handler.client = client
    _Handler.model = model
    _Handler.ollama_host = host
    _Handler.token = token
    _Handler.sessions = SessionStore()
    _Handler.max_context_chars = context_budget.max_context_chars_from_env()

    httpd = _bind_socket(port)
    bound_host, bound_port = httpd.server_address[:2]

    _write_token_file(bound_host, bound_port, token)
    return httpd


def run_server(port: int = DEFAULT_PORT) -> None:
    """Start the local agent server and block forever."""
    httpd = build_server(port)
    bound_host, bound_port = httpd.server_address[:2]
    ollama_connected = _Handler.client.check_connection()

    print("Local Coding Agent -- server mode")
    print(f"  Server URL:    http://{bound_host}:{bound_port}")
    print(f"  Model:         {_Handler.model}")
    print(f"  Ollama status: {'connected' if ollama_connected else 'NOT CONNECTED (start it with: ollama serve)'}")
    print(f"  Auth token:    {TOKEN_FILE} (chmod 600)")
    print()
    print("Waiting for VS Code...")
    sys.stdout.flush()

    def _handle_sigterm(signum, frame):  # noqa: ARG001 - required signal handler signature
        # Make a plain `kill`/service-manager stop behave like Ctrl+C so the
        # token file is always cleaned up on shutdown, not just on SIGINT.
        raise KeyboardInterrupt()

    previous_sigterm_handler = signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
        _remove_token_file()
        httpd.server_close()
