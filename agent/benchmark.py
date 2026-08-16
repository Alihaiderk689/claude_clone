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
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .context_budget import estimate_tokens_from_chars
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
    # Phase 9 additions. All optional/default-safe so a report from before
    # Phase 9 still loads (and --compare degrades gracefully -- see below)
    # even though it never wrote these fields.
    llm_calls: int = 0
    peak_context_chars: int = 0
    estimated_peak_context_tokens: int = 0
    time_to_first_token_seconds: Optional[float] = None
    peak_rss_bytes: Optional[int] = None


class _RecordingClient:
    """Wraps a real OllamaClient, forwarding every call unchanged, while
    keeping a running tally of tokens actually used, the number of model
    round-trips, the largest context sent on any single call, and how long
    the very first call took to produce its first streamed update. Purely
    observational -- changes no behavior run_agent_turn depends on.
    """

    def __init__(self, inner: OllamaClient) -> None:
        self._inner = inner
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.llm_calls = 0
        self.peak_context_chars = 0
        self.time_to_first_token_seconds: Optional[float] = None

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def chat(self, messages, tools=None, cancel_event=None):
        self.llm_calls += 1
        context_chars = sum(len(m.get("content") or "") for m in messages)
        self.peak_context_chars = max(self.peak_context_chars, context_chars)

        call_start = time.monotonic()
        first_update_seen = False
        for update in self._inner.chat(messages, tools=tools, cancel_event=cancel_event):
            if not first_update_seen:
                first_update_seen = True
                if self.time_to_first_token_seconds is None:
                    self.time_to_first_token_seconds = time.monotonic() - call_start
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


# Path fragments that mark a tool call's path argument as one of Task E's
# deliberately irrelevant noise files -- never legitimate targets for a
# request that's purely about fixing authentication.
_NOISE_PATH_MARKERS = ("frontend/", "package-lock.json", "vendor/")


def _setup_relevance_fixture(root: Path) -> None:
    """Task E's fixture: the normal auth/greeting project PLUS several
    unrelated noise files a relevance-aware agent should never need to
    read or edit while fixing authentication."""
    _setup_fixture_project(root)
    (root / "frontend").mkdir()
    (root / "frontend" / "logo.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>\n")
    (root / "frontend" / "styles.css").write_text("body { margin: 0; }\n")
    (root / "package-lock.json").write_text(
        json.dumps({"name": "noise", "lockfileVersion": 3, "packages": {}}, indent=2)
    )
    (root / "vendor").mkdir()
    (root / "vendor" / "big_unrelated_module.py").write_text(
        "\n".join(f"def unrelated_function_{i}():\n    pass\n" for i in range(50))
    )
    subprocess.run(["git", "add", "-A"], cwd=str(root), check=True)
    subprocess.run(["git", "commit", "-q", "-m", "Add unrelated noise files"], cwd=str(root), check=True)


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


def _peak_rss_bytes() -> int:
    """ru_maxrss is a process-wide high-water mark, not per-task-isolated --
    reading it after each task still gives a useful monotonically-growing
    signal across a run, just not a clean per-task delta. Units differ by
    platform: bytes on macOS (the target hardware), KB on Linux -- callers
    displaying this should say so rather than assume bytes everywhere."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


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
        llm_calls=recording.llm_calls,
        peak_context_chars=recording.peak_context_chars,
        estimated_peak_context_tokens=estimate_tokens_from_chars(recording.peak_context_chars),
        time_to_first_token_seconds=recording.time_to_first_token_seconds,
        peak_rss_bytes=_peak_rss_bytes(),
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


def _check_relevant_file_selection(root: Path, events: list, task_state: TaskState) -> Tuple[bool, str]:
    """Task E: fixing authentication must never touch the deliberately
    irrelevant noise files planted by _setup_relevance_fixture -- listing
    the project (which will show their names) is fine; reading, editing, or
    writing INTO one of them is what this actually checks for."""
    if not _has_clean_final_answer(events):
        return False, "no clean final answer"
    touched_noise = []
    for e in events:
        if e["type"] != "tool_call":
            continue
        path = (e.get("args") or {}).get("path", "")
        if any(marker in path for marker in _NOISE_PATH_MARKERS):
            touched_noise.append(path)
    ok = not touched_noise
    detail = "" if ok else f"touched irrelevant noise file(s): {', '.join(touched_noise)}"
    return ok, detail


def _check_stuck_recovery(root: Path, events: list, task_state: TaskState) -> Tuple[bool, str]:
    """Task F: the request is designed so a naive first attempt (matching
    literal text that doesn't exist in the file) fails the same way if
    retried unchanged -- this checks the turn recovers (a clean final
    answer, whether or not repetition_detected actually fired) instead of
    burning the whole iteration budget stuck on the same failing call."""
    hit_max_iterations = any(e["type"] == "max_iterations" for e in events)
    ok = not hit_max_iterations and _has_clean_final_answer(events)
    detail = "" if ok else "turn got stuck (hit max_iterations without recovering)"
    return ok, detail


# (name, user_message, check, fixture_setup) -- fixture_setup defaults to
# _setup_fixture_project for every task except where a task specifically
# needs a different/extended fixture (Task E's noise files).
TASKS: List[tuple] = [
    (
        "Explain repository structure",
        "Briefly explain what this project's structure looks like.",
        _check_explain_structure,
        _setup_fixture_project,
    ),
    (
        "Find a bug",
        "There's a known bug in backend/auth.py -- find it and explain what's wrong, but don't fix it yet.",
        _check_find_bug,
        _setup_fixture_project,
    ),
    (
        "Modify a function",
        "Change greet() in backend/greeting.py to return 'Hi, <name>!' instead of 'Hello, <name>'.",
        _check_modify_function,
        _setup_fixture_project,
    ),
    (
        "Add a test",
        "Add a new test file tests/test_extra.py with a simple passing test for greet() from backend/greeting.py.",
        _check_add_test,
        _setup_fixture_project,
    ),
    (
        "Run tests and fix a failure",
        "Run the test suite with pytest and fix any failing test you find.",
        _check_run_and_fix_tests,
        _setup_fixture_project,
    ),
    (
        "Multi-step change",
        "Fix the bug in backend/auth.py's verify() method so it actually checks the token against "
        "the secret, then run the tests.",
        _check_multi_step_change,
        _setup_fixture_project,
    ),
    (
        "Relevant file selection",
        "There's a bug in this project's authentication code -- find it and fix it.",
        _check_relevant_file_selection,
        _setup_relevance_fixture,
    ),
    (
        "Stuck/repetition recovery",
        "Edit backend/greeting.py: replace the exact text 'this substring does not exist anywhere "
        "in the file' with 'unused'.",
        _check_stuck_recovery,
        _setup_fixture_project,
    ),
]


def run_benchmark(model: str, host: str, output: Optional[Path]) -> List[TaskResult]:
    def client_factory() -> OllamaClient:
        return OllamaClient(host=host, model=model)

    results: List[TaskResult] = []
    tmp_root = Path(tempfile.mkdtemp(prefix="code-agent-benchmark-"))
    try:
        for name, message, check, setup_fixture in TASKS:
            task_root = tmp_root / name.lower().replace(" ", "_").replace("/", "_")
            task_root.mkdir()
            setup_fixture(task_root)
            print(f"Running: {name} ...", file=sys.stderr)
            result = _run_task(name, client_factory, task_root, message, check)
            results.append(result)
            status = "PASS" if result.success else "FAIL"
            print(f"  {status} in {result.duration_seconds:.1f}s, {result.tool_calls} tool call(s)", file=sys.stderr)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    report = {
        "model": model,
        "host": host,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tasks": [asdict(r) for r in results],
    }
    if output is not None:
        output.write_text(json.dumps(report, indent=2))

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

    print("\nPhase 9 metrics:")
    ph9_header = f"{'Task':<30} {'LLMcalls':>8} {'CtxTok~':>8} {'TTFT(s)':>8} {'PeakRSS':>12}"
    print(ph9_header)
    print("-" * len(ph9_header))
    for r in results:
        ttft = f"{r.time_to_first_token_seconds:.2f}" if r.time_to_first_token_seconds is not None else "-"
        rss = f"{r.peak_rss_bytes:,}" if r.peak_rss_bytes is not None else "-"
        print(
            f"{r.name:<30} {r.llm_calls:>8} {r.estimated_peak_context_tokens:>8} {ttft:>8} {rss:>12}"
        )


def _load_report(path: Path) -> Tuple[Optional[str], List[dict]]:
    """Loads either the current {"model", "host", "timestamp", "tasks": [...]}
    report shape or the pre-Phase-9 bare-array shape (a real baseline
    captured before this format existed, per the plan's baseline-first
    requirement) -- returns (model_or_None, task_dicts)."""
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return None, data
    return data.get("model"), data.get("tasks", [])


def compare_reports(before_path: Path, after_path: Path) -> None:
    before_model, before_tasks = _load_report(before_path)
    after_model, after_tasks = _load_report(after_path)
    before_by_name = {t["name"]: t for t in before_tasks}
    after_by_name = {t["name"]: t for t in after_tasks}

    print(f"\nBefore: {before_path} (model: {before_model or 'unknown'})")
    print(f"After:  {after_path} (model: {after_model or 'unknown'})\n")

    # Metrics present in both the pre- and post-Phase-9 report shape --
    # comparable no matter which side is the older format.
    common_metrics = ["success", "duration_seconds", "tool_calls", "retries", "prompt_tokens", "completion_tokens"]
    # Phase 9-only metrics -- only meaningful when both sides have them.
    new_metrics = ["llm_calls", "estimated_peak_context_tokens", "time_to_first_token_seconds", "peak_rss_bytes"]

    all_names = list(dict.fromkeys(list(before_by_name) + list(after_by_name)))
    for name in all_names:
        before = before_by_name.get(name)
        after = after_by_name.get(name)
        print(f"## {name}")
        if before is None:
            print("  (new task -- no baseline to compare against)")
        elif after is None:
            print("  (present in before, missing from after)")
        else:
            for metric in common_metrics:
                b, a = before.get(metric), after.get(metric)
                print(f"  {metric}: {b} -> {a}")
            if all(m in before for m in new_metrics) and all(m in after for m in new_metrics):
                for metric in new_metrics:
                    print(f"  {metric}: {before.get(metric)} -> {after.get(metric)}  (Phase 9 metric)")
            else:
                print("  (Phase 9 metrics not available on both sides -- after-only, no historical baseline)")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m agent.benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--output", type=Path, default=None, help="Write a JSON report to this path.")
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BEFORE_JSON", "AFTER_JSON"),
        type=Path,
        default=None,
        help="Skip running the benchmark; print a before/after diff of two saved JSON reports.",
    )
    args = parser.parse_args()

    if args.compare is not None:
        compare_reports(*args.compare)
        return

    client = OllamaClient(host=args.host, model=args.model)
    if not client.check_connection():
        print(f"Could not connect to Ollama at {args.host}. Start it with: ollama serve", file=sys.stderr)
        sys.exit(1)

    results = run_benchmark(args.model, args.host, args.output)
    print_report(results, args.model)


if __name__ == "__main__":
    main()
