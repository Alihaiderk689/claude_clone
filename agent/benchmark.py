"""Local, repeatable benchmark harness for the coding agent (Phase 8,
spec sections 31-32).

Runs a fixed set of small coding tasks against a REAL local Ollama server
and model, driving the exact same `run_agent_turn()` generator the CLI and
the HTTP server use -- this is not a synthetic benchmark, it exercises the
real agent loop, real tools, and a real temporary Git repository end to
end. Every confirm*/approval prompt is auto-approved (there is no human in
the loop for an unattended benchmark run) -- that is the one deliberate
difference from real usage; everything else (validation, execution, Git,
terminal commands) is completely real.

This is a baseline measurement, not telemetry: nothing leaves the machine,
no external service beyond a local Ollama server is required, and no
result is uploaded anywhere. Phase 8 does not use these numbers to change
the model or its prompting -- that's explicitly out of scope until Phase 9.

Usage:
    python -m agent.benchmark [--model MODEL] [--host HOST] [--output FILE]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .loop import run_agent_turn
from .ollama_client import DEFAULT_HOST, DEFAULT_MODEL, OllamaClient
from .project import ProjectRoot
from .task_state import TaskState
from .tools import FileStateTracker, build_default_registry

CONFIRM_EVENT_TYPES = {"confirm", "confirm_command", "confirm_git_operation", "confirm_plan"}


@dataclass
class TaskResult:
    name: str
    success: bool
    duration_seconds: float
    tool_calls: int
    retries: int
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    detail: str = ""


class _RecordingClient:
    """Wraps a real OllamaClient, forwarding every call unchanged, while
    keeping a running tally of tokens actually used. Purely observational
    -- changes no behavior run_agent_turn depends on.
    """

    def __init__(self, inner: OllamaClient) -> None:
        self._inner = inner
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def chat(self, messages, tools=None, cancel_event=None):
        for update in self._inner.chat(messages, tools=tools, cancel_event=cancel_event):
            if "prompt_eval_count" in update:
                self.prompt_tokens += update["prompt_eval_count"]
            if "eval_count" in update:
                self.completion_tokens += update["eval_count"]
            yield update


def _drive_turn_auto_approve(client, registry, messages, task_state) -> Tuple[list, int]:
    """Runs one turn to completion, auto-approving every confirm* event --
    there is no human in the loop for an unattended benchmark run. Returns
    (events, retry_count).
    """
    tracker = FileStateTracker()
    gen = run_agent_turn(client, registry, messages, tracker=tracker, task_state=task_state)
    events: list = []
    retries = 0
    send_value = None
    while True:
        try:
            event = gen.send(send_value)
        except StopIteration:
            break
        events.append(event)
        if event["type"] == "retry":
            retries += 1
        send_value = True if event["type"] in CONFIRM_EVENT_TYPES else None
    return events, retries


def _setup_fixture_project(root: Path) -> None:
    (root / "backend").mkdir()
    (root / "backend" / "auth.py").write_text(
        "class JWTAuth:\n"
        "    def __init__(self, secret):\n"
        "        self.secret = secret\n"
        "\n"
        "    def verify(self, token):\n"
        "        # BUG: always returns True regardless of token validity\n"
        "        return True\n"
    )
    (root / "backend" / "greeting.py").write_text(
        "def greet(name):\n"
        "    return 'Hello, ' + name\n"
    )
    (root / "tests").mkdir()
    (root / "tests" / "test_greeting.py").write_text(
        "from backend.greeting import greet\n\n\n"
        "def test_greet():\n"
        "    assert greet('World') == 'Hello, World'\n"
    )
    (root / "tests" / "test_auth.py").write_text(
        "from backend.auth import JWTAuth\n\n\n"
        "def test_verify_rejects_wrong_secret():\n"
        "    auth = JWTAuth('correct-secret')\n"
        "    assert auth.verify('correct-secret') is True\n"
        "    assert auth.verify('wrong-secret') is False\n"
    )
    (root / "README.md").write_text(
        "# Benchmark project\n\nA tiny fixture project: backend/ has an auth module with a "
        "known bug and a greeting module; tests/ covers both.\n"
    )
    subprocess.run(["git", "init", "-q"], cwd=str(root), check=True)
    subprocess.run(["git", "config", "user.email", "bench@example.com"], cwd=str(root), check=True)
    subprocess.run(["git", "config", "user.name", "Bench"], cwd=str(root), check=True)
    subprocess.run(["git", "add", "-A"], cwd=str(root), check=True)
    subprocess.run(["git", "commit", "-q", "-m", "Initial commit"], cwd=str(root), check=True)


def _new_session(root: Path):
    project = ProjectRoot(root)
    task_state = TaskState()
    registry = build_default_registry(project, task_state=task_state)
    system_prompt = (
        f"You are a coding assistant operating inside a specific project directory:\n\n{root}\n\n"
        "Use your tools to inspect and modify the project as needed. Before editing anything, "
        "read the relevant file first. Keep answers concise."
    )
    messages = [{"role": "system", "content": system_prompt}]
    return registry, task_state, messages


def _run_task(
    name: str,
    client_factory: Callable[[], OllamaClient],
    root: Path,
    user_message: str,
    check: Callable[[Path, list, TaskState], Tuple[bool, str]],
) -> TaskResult:
    registry, task_state, messages = _new_session(root)
    recording = _RecordingClient(client_factory())
    messages.append({"role": "user", "content": user_message})

    start = time.monotonic()
    try:
        events, retries = _drive_turn_auto_approve(recording, registry, messages, task_state)
    except Exception as exc:  # pragma: no cover - a benchmark run must never crash outright
        return TaskResult(name, False, time.monotonic() - start, 0, 0, None, None, detail=f"crashed: {exc}")
    duration = time.monotonic() - start

    tool_call_count = sum(1 for e in events if e["type"] == "tool_call")
    success, detail = check(root, events, task_state)
    return TaskResult(
        name=name,
        success=success,
        duration_seconds=duration,
        tool_calls=tool_call_count,
        retries=retries,
        prompt_tokens=recording.prompt_tokens or None,
        completion_tokens=recording.completion_tokens or None,
        detail=detail,
    )


def _has_clean_final_answer(events: list) -> bool:
    return any(e["type"] == "final" for e in events) and not any(
        e["type"] in ("error", "max_iterations") for e in events
    )


def _check_explain_structure(root: Path, events: list, task_state: TaskState) -> Tuple[bool, str]:
    if not _has_clean_final_answer(events):
        return False, "no clean final answer"
    final_text = next(e["text"] for e in events if e["type"] == "final")
    ok = bool(final_text.strip()) and any(e["type"] == "tool_call" for e in events)
    return ok, "" if ok else "final answer empty or no inspection happened"


def _check_find_bug(root: Path, events: list, task_state: TaskState) -> Tuple[bool, str]:
    if not _has_clean_final_answer(events):
        return False, "no clean final answer"
    final_text = next(e["text"] for e in events if e["type"] == "final").lower()
    ok = "verify" in final_text and ("bug" in final_text or "always" in final_text or "true" in final_text)
    return ok, "" if ok else "final answer didn't mention the known bug"


def _check_modify_function(root: Path, events: list, task_state: TaskState) -> Tuple[bool, str]:
    applied = any(e["type"] == "change_applied" for e in events)
    if not applied:
        return False, "no change was applied"
    content = (root / "backend" / "greeting.py").read_text()
    ok = "greet" in content and "Hello, ' + name" not in content
    return ok, "" if ok else "greeting.py wasn't actually changed"


def _check_add_test(root: Path, events: list, task_state: TaskState) -> Tuple[bool, str]:
    applied = any(e["type"] == "change_applied" for e in events)
    # The fixture already ships 2 test files -- a genuine new one means 3+.
    test_files_after = list((root / "tests").glob("*.py"))
    ok = applied and len(test_files_after) >= 3
    return ok, "" if ok else "no new test file was created"


def _check_run_and_fix_tests(root: Path, events: list, task_state: TaskState) -> Tuple[bool, str]:
    results = [e["result"] for e in events if e["type"] == "command_result"]
    ok = bool(results) and results[-1].exit_code == 0
    return ok, "" if ok else "final test run did not pass"


def _check_multi_step_change(root: Path, events: list, task_state: TaskState) -> Tuple[bool, str]:
    applied = sum(1 for e in events if e["type"] == "change_applied")
    ok = applied >= 1 and _has_clean_final_answer(events)
    return ok, "" if ok else "expected at least one applied change and a clean final answer"


TASKS: List[tuple] = [
    (
        "Explain repository structure",
        "Briefly explain what this project's structure looks like.",
        _check_explain_structure,
    ),
    (
        "Find a bug",
        "There's a known bug in backend/auth.py -- find it and explain what's wrong, but don't fix it yet.",
        _check_find_bug,
    ),
    (
        "Modify a function",
        "Change greet() in backend/greeting.py to return 'Hi, <name>!' instead of 'Hello, <name>'.",
        _check_modify_function,
    ),
    (
        "Add a test",
        "Add a new test file tests/test_extra.py with a simple passing test for greet() from backend/greeting.py.",
        _check_add_test,
    ),
    (
        "Run tests and fix a failure",
        "Run the test suite with pytest and fix any failing test you find.",
        _check_run_and_fix_tests,
    ),
    (
        "Multi-step change",
        "Fix the bug in backend/auth.py's verify() method so it actually checks the token against "
        "the secret, then run the tests.",
        _check_multi_step_change,
    ),
]


def run_benchmark(model: str, host: str, output: Optional[Path]) -> List[TaskResult]:
    def client_factory() -> OllamaClient:
        return OllamaClient(host=host, model=model)

    results: List[TaskResult] = []
    tmp_root = Path(tempfile.mkdtemp(prefix="code-agent-benchmark-"))
    try:
        for name, message, check in TASKS:
            task_root = tmp_root / name.lower().replace(" ", "_")
            task_root.mkdir()
            _setup_fixture_project(task_root)
            print(f"Running: {name} ...", file=sys.stderr)
            result = _run_task(name, client_factory, task_root, message, check)
            results.append(result)
            status = "PASS" if result.success else "FAIL"
            print(f"  {status} in {result.duration_seconds:.1f}s, {result.tool_calls} tool call(s)", file=sys.stderr)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    if output is not None:
        output.write_text(json.dumps([asdict(r) for r in results], indent=2))

    return results


def print_report(results: List[TaskResult], model: str) -> None:
    print(f"\nQwen benchmark report -- model: {model}\n")
    header = f"{'Task':<30} {'Result':<6} {'Time(s)':>8} {'Tools':>6} {'Retries':>8} {'PromptTok':>10} {'CompTok':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r.name:<30} {'PASS' if r.success else 'FAIL':<6} {r.duration_seconds:>8.1f} "
            f"{r.tool_calls:>6} {r.retries:>8} {r.prompt_tokens or '-':>10} {r.completion_tokens or '-':>8}"
        )
    passed = sum(1 for r in results if r.success)
    print(f"\n{passed}/{len(results)} tasks passed.")
    for r in results:
        if not r.success and r.detail:
            print(f"  - {r.name}: {r.detail}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m agent.benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--output", type=Path, default=None, help="Write a JSON report to this path.")
    args = parser.parse_args()

    client = OllamaClient(host=args.host, model=args.model)
    if not client.check_connection():
        print(f"Could not connect to Ollama at {args.host}. Start it with: ollama serve", file=sys.stderr)
        sys.exit(1)

    results = run_benchmark(args.model, args.host, args.output)
    print_report(results, args.model)


if __name__ == "__main__":
    main()
