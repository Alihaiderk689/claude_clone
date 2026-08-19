"""The tool-calling agent loop.

Drives one user turn: ask the model, execute any tool calls it requests,
feed the results back, and repeat until it produces a plain-text final
answer or a safety limit is hit. Knows nothing about the terminal — it
yields structured events for a caller (cli.py) to render.
"""
from __future__ import annotations

import json
import re
import threading
import time
from typing import Dict, Iterator, List, Optional

from . import context_budget, context_manager
from .logging_config import get_logger
from .ollama_client import (
    OllamaAPIError,
    OllamaCancelledError,
    OllamaClient,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
)
from .task_state import TaskState
from .tools.editing import apply_change
from .tools.file_ops import apply_file_op
from .tools.git import apply_git_operation
from .tools.registry import ToolRegistry
from .tools.state import FileStateTracker
from .tools.terminal import describe_command, execute_command, format_result_for_model

# Raised from an earlier, much tighter 10: a multi-file task (read a file,
# create a replacement, update references, delete the original, run tests)
# routinely needs 8-10 tool calls on its own even when the model behaves
# perfectly first try; add narration-correction retries (see
# MAX_NARRATION_RETRIES below) or one re-read after an unexpected tool
# error and the old limit was routinely too tight for exactly the kind of
# task this agent is meant to handle end-to-end without a new user message.
MAX_TOOL_ITERATIONS = 25

# Retries only ever apply to OllamaConnectionError/OllamaTimeoutError -- both
# are transient (server hiccup, momentary network blip) and a second attempt
# with the exact same request is a reasonable thing to try. A model-not-found
# or a malformed-request-shaped API error is not transient: retrying an
# identical request would just fail identically, so those still fail
# immediately (see the except clauses below).
MAX_OLLAMA_RETRIES = 2  # total attempts = MAX_OLLAMA_RETRIES + 1
OLLAMA_RETRY_BACKOFF_SECONDS = (1.0, 2.0)

# Same repeated-call detection covers both "repetition" (spec section 9) and
# "failed approach" (section 10): if the exact same tool+arguments are
# attempted this many times in a row within one turn, the loop stops
# executing it again and tells the model to change approach instead.
MAX_CONSECUTIVE_IDENTICAL_CALLS = 3

# Tool names whose successful proposal represents real progress toward a
# requested change -- used only to gate narration-vs-tool-call detection
# below (see _looks_like_unactioned_narration). Read-only inspection tools
# (list_files, read_file, search_files, git_status/diff/log) deliberately do
# NOT count: calling them and then stopping to ask "what next?" is exactly
# the failure mode this guards against, observed live as
# read_file -> read_file[cached] -> "What would you like me to do next?"
# with zero mutating tool calls anywhere in the turn.
MUTATING_TOOL_NAMES = frozenset(
    {
        "edit_file", "write_file", "delete_file", "rename_file", "run_command",
        "git_create_branch", "git_stage", "git_commit", "git_push",
    }
)

# How many times a plain-text response that reads as "I'll do X" / "what
# would you like me to do next" (with no tool call anywhere yet this turn)
# gets a corrective nudge instead of being accepted as the final answer. A
# small model's own narration should never substitute for an actual
# edit_file/write_file/delete_file/rename_file/run_command/git_* call -- see
# _looks_like_unactioned_narration below. Bounded so a model that genuinely
# can't comply still gets a real answer back (with task_incomplete flagged)
# rather than looping forever.
MAX_NARRATION_RETRIES = 2

# Two independent signals, either one is enough to trigger a correction:
# (1) the model is deferring back to the user for a decision it doesn't
#     need ("what would you like me to do next?", "should I proceed?") when
#     the user's request already told it what to do, and
# (2) the model states a mutating intent in future/conditional tense
#     ("I'll create...", "I will delete...") without having actually called
#     the tool that would do it.
# Deliberately narrow (specific phrases / verb list) rather than a broad
# "contains I will" match -- a genuine informational answer to a real
# question can legitimately contain similar phrasing, and this only matters
# because it's bounded (MAX_NARRATION_RETRIES) and gated on no mutating tool
# having run yet this turn (see run_agent_turn), so a false positive costs
# at most a couple of extra, cheap local-model turns rather than corrupting
# behavior.
_DEFERRAL_RE = re.compile(
    r"what (?:would|do) you (?:like|want)|what should i do next|let me know if you|"
    r"should i (?:proceed|go ahead)|would you like me to|do you want me to",
    re.IGNORECASE,
)
_UNACTIONED_INTENT_RE = re.compile(
    r"\bi(?:'ll| will| am going to| plan to)\b[^.\n]{0,40}\b"
    r"(create|write|add|update|edit|modify|delete|remove|rename|move|implement)\b",
    re.IGNORECASE,
)


def _looks_like_unactioned_narration(content: str) -> bool:
    text = (content or "").strip()
    if not text:
        return False
    return bool(_DEFERRAL_RE.search(text) or _UNACTIONED_INTENT_RE.search(text))


_logger = get_logger("loop")

# qwen2.5-coder's own Ollama chat template documents <tool_call>{json}</tool_call>
# as its function-calling format (see `ollama show qwen2.5-coder:7b --template`).
# In practice (confirmed live, repeatedly, against a real local Ollama server),
# Ollama doesn't reliably lift that into the API's structured tool_calls field
# for this model -- it can arrive as plain assistant content instead, as bare
# JSON, tagged JSON, or prose followed by a JSON call with no tag at all. We
# recognize that same model-documented shape wherever it appears in the text,
# rather than inventing a new protocol or trusting tool_calls alone.
_TOOL_CALL_TAG_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)


def run_agent_turn(
    client: OllamaClient,
    registry: ToolRegistry,
    messages: List[Dict],
    tracker: Optional[FileStateTracker] = None,
    task_state: Optional[TaskState] = None,
    max_iterations: int = MAX_TOOL_ITERATIONS,
    cancel_event: Optional[threading.Event] = None,
    auto_approve_edits: bool = False,
    max_context_chars: int = context_manager.MAX_CONTEXT_CHARS,
) -> Iterator[dict]:
    """Drive one user turn to completion.

    `messages` is mutated in place as the assistant/tool messages are
    produced, so the caller's conversation history stays in sync even if
    iteration stops early. Yields events:

        {"type": "tool_call", "name", "args", "display"}
        {"type": "tool_error", "name", "message"}
        {"type": "confirm", "name", "change"}   -- edit_file/write_file proposed a
            change; the caller MUST send back True (apply) or False (reject) via
            generator.send() before iteration continues. THE MODEL NEVER WRITES
            TO DISK -- this pause, and an explicit True from the caller, is the
            only path to apply_change() actually touching the filesystem. If
            `auto_approve_edits` is set, the event carries an extra
            `"auto_approved": True` field and the approval is already resolved
            internally -- the caller does not need to (and should not need to)
            send anything back for it to apply; see the note on
            `auto_approve_edits` below.
        {"type": "change_applied", "name", "path"}   -- approved, written, verified
        {"type": "change_rejected", "name", "path"}  -- user declined; untouched
        {"type": "confirm_file_op", "name", "operation"}   -- delete_file/rename_file
            proposed a filesystem mutation (see file_ops.ProposedFileOp); the caller
            MUST send back True (apply) or False (reject) via generator.send(). THE
            MODEL NEVER TOUCHES THE FILESYSTEM -- this pause, and an explicit True,
            is the only path to apply_file_op() actually calling os.remove/os.rename.
            Same `auto_approved` behavior as `confirm` when `auto_approve_edits` is
            set -- delete/rename are treated as the same risk class as edit/write.
        {"type": "file_op_applied", "name", "operation"}   -- approved and done
        {"type": "file_op_rejected", "name", "operation"}  -- user declined; untouched
        {"type": "confirm_command", "name", "command"}   -- run_command proposed a
            command; the caller MUST send back True (run) or False (reject) via
            generator.send(). THE MODEL NEVER EXECUTES ANYTHING -- this pause,
            and an explicit True, is the only path to execute_command() actually
            spawning a subprocess.
        {"type": "command_started", "name", "display"}   -- approved; about to run
        {"type": "command_result", "name", "result"}      -- finished (see
            tools/terminal.py's CommandExecutionResult); a nonzero exit code is
            still a normal result, not a tool_error -- only a Python-level
            failure to even run the command is reported as tool_error.
        {"type": "command_rejected", "name"}              -- user declined; never ran
        {"type": "confirm_git_operation", "name", "operation"}   -- git_create_branch/
            git_stage/git_commit proposed a state-changing Git operation; the caller
            MUST send back True (do it) or False (reject) via generator.send().
            THE MODEL NEVER RUNS GIT DIRECTLY -- this pause is the only path to
            apply_git_operation() actually running `git branch`/`git add`/`git commit`.
        {"type": "git_operation_applied", "name", "message"}   -- approved and done
        {"type": "git_operation_rejected", "name", "operation"}  -- user declined
        {"type": "confirm_plan", "name", "plan"}   -- create_plan proposed a plan; the
            caller MUST send back True (adopt) or False (reject) via generator.send().
            Unlike every other confirm_* kind, the caller's UI should default to
            APPROVE on an empty answer -- a plan has no side effects of its own.
            Also auto-approved (with `"auto_approved": True`) when
            `auto_approve_edits` is set -- see below.
        {"type": "plan_approved", "name", "plan"}     -- adopted into task_state
        {"type": "plan_rejected", "name"}             -- user declined; discarded
        {"type": "content", "text"}   -- the final answer text to display
        {"type": "final", "text"}     -- the complete final answer (same text). Carries an
            extra `"task_incomplete": True` field when the turn ended on a response that
            still looks like unactioned narration (see MAX_NARRATION_RETRIES) after
            every corrective retry was exhausted -- a signal for the caller to flag
            this to the user, not just print it as an ordinary answer. Absent (not
            `False`) in the normal case, so existing callers that don't check for it
            behave exactly as before.
        {"type": "narration_redirected", "text", "attempt", "max_attempts"}  -- the model
            produced a plain-text response with no tool call that reads as unactioned
            narration ("I'll create...", "what would you like me to do next?") while no
            mutating tool has run yet this turn; a corrective instruction was injected
            and the turn continues (not a terminal event) -- see
            _looks_like_unactioned_narration and MUTATING_TOOL_NAMES above.
        {"type": "error", "message"}  -- fatal Ollama error; turn stops
        {"type": "max_iterations"}    -- safety limit hit without a final answer
        {"type": "cancelled"}         -- cancel_event was set; turn stopped early
        {"type": "retry", "reason", "attempt", "max_attempts", "wait_seconds"}  -- a transient
            Ollama connection/timeout error is being retried (see MAX_OLLAMA_RETRIES); the turn
            is not over, this is purely informational for the UI to show "Retrying (2/3)..."
        {"type": "repetition_detected", "name", "args", "count"}  -- the exact same tool call was
            requested MAX_CONSECUTIVE_IDENTICAL_CALLS times in a row; it was NOT executed/proposed
            again -- the model was told to change approach instead. The turn continues.

    Each iteration's response is fully buffered before being shown, because a
    tool call can arrive embedded in ordinary assistant content (see above)
    at an unpredictable position -- there's no safe prefix-only check that
    would let content stream live without risking a raw tool call leaking
    into the terminal as if it were the answer.

    If `task_state` is given, every tool result updates it with a compact
    record (see agent/context_manager.py) instead of the full output, and the
    conversation is compacted and re-summarized into the system prompt at the
    top of every iteration -- this is what keeps a long multi-step task from
    slowly resending every file it has ever read on every subsequent turn.

    If `cancel_event` is given, it's checked at the start of every iteration
    and passed through to client.chat() so a long model generation can be
    interrupted mid-stream. When cancellation is observed, the turn stops
    with {"type": "cancelled"} instead of "final" -- used by the local HTTP
    server's Stop Task endpoint (agent/server.py). Left as None (the
    default), this has no effect on existing callers.

    If `auto_approve_edits` is True, `confirm` (edit_file/write_file),
    `confirm_file_op` (delete_file/rename_file -- same risk class as an edit:
    reversible via Git, no command execution, no Git state change), and
    `confirm_plan` events resolve to an internal `True` *before* the yield,
    rather than depending on whatever the caller sends back -- so a caller
    that doesn't specifically handle `auto_approved` still can't
    accidentally skip a real approval or hang forever; at worst it shows a
    redundant prompt for something that already happened. This is "Auto
    mode" (see cli.py's `/auto`/`/manual` and server.py's `/task/mode`).
    It deliberately does NOT extend to `confirm_command` or
    `confirm_git_operation` -- every terminal command and every Git
    operation (including `git_push`) always pauses for individual human
    approval no matter what this flag is set to; there is no parameter
    that reaches those two blocks. Left as False (the default), this has
    no effect on existing callers.

    `max_context_chars` is passed straight through to
    context_manager.compact_messages() every iteration; left at its
    default (context_manager.MAX_CONTEXT_CHARS), behavior is unchanged.
    See agent/context_budget.py's `max_context_chars_from_env()` for how
    cli.py/server.py source an override from CODE_AGENT_MAX_CONTEXT_CHARS.
    """
    tools_schema = registry.schemas()

    # Repetition/failed-approach guard (spec sections 9-10): tracks the same
    # (tool name, arguments) signature across consecutive tool calls within
    # this turn -- across iterations, not just within one -- so a small
    # model stuck re-issuing an identical call (whether it keeps "succeeding"
    # with no progress, like re-reading the same file, or keeps failing the
    # same way, like re-proposing a rejected edit) gets stopped and told to
    # change approach instead of burning the whole iteration budget on it.
    last_call_signature: Optional[tuple] = None
    consecutive_call_count = 0

    # Narration-vs-tool-call guard (see MUTATING_TOOL_NAMES/MAX_NARRATION_RETRIES
    # above): tracked across the whole turn, not just one iteration, so a
    # response that comes back with zero tool calls after a read_file/
    # search_files-only prelude is still recognized as "nothing was actually
    # done yet" -- exactly the reported failure (two read_file calls, then
    # plain-text narration).
    mutating_tool_called_this_turn = False
    narration_retry_count = 0

    # Phase 9 debug-mode stats (agent/logging_config.py's --debug /
    # CODE_AGENT_DEBUG=1) -- accumulated across the whole turn and logged
    # once in the `finally` below, regardless of which of the many
    # return/yield-then-return exit paths above actually fires.
    llm_call_count = 0
    tool_call_count = 0
    cache_hits = 0
    cache_misses = 0
    compaction_totals = context_manager.CompactionStats()

    try:
        for _ in range(max_iterations):
            if cancel_event is not None and cancel_event.is_set():
                yield {"type": "cancelled"}
                return

            if task_state is not None:
                iteration_stats = context_manager.compact_messages(
                    messages, task_state, max_context_chars=max_context_chars
                )
                compaction_totals.superseded += iteration_stats.superseded
                compaction_totals.stale += iteration_stats.stale
                compaction_totals.trimmed += iteration_stats.trimmed
                context_manager.refresh_system_prompt(messages, task_state)

            content_parts: List[str] = []
            tool_calls: List[dict] = []
            retry_attempt = 0

            while True:
                content_parts = []
                tool_calls = []
                try:
                    for update in client.chat(messages, tools=tools_schema, cancel_event=cancel_event):
                        if update["content"]:
                            content_parts.append(update["content"])
                        if update["tool_calls"]:
                            tool_calls.extend(update["tool_calls"])
                    break  # a full response streamed successfully
                except OllamaCancelledError:
                    _logger.info("Turn cancelled during model generation.")
                    yield {"type": "cancelled"}
                    return
                except (OllamaConnectionError, OllamaTimeoutError) as exc:
                    if retry_attempt >= MAX_OLLAMA_RETRIES:
                        _logger.warning("Ollama request failed after %d retries: %s", retry_attempt, exc)
                        yield {"type": "error", "message": str(exc)}
                        return
                    backoff = OLLAMA_RETRY_BACKOFF_SECONDS[
                        min(retry_attempt, len(OLLAMA_RETRY_BACKOFF_SECONDS) - 1)
                    ]
                    retry_attempt += 1
                    _logger.warning(
                        "Ollama request failed (%s); retrying attempt %d/%d in %.1fs.",
                        exc, retry_attempt, MAX_OLLAMA_RETRIES + 1, backoff,
                    )
                    yield {
                        "type": "retry",
                        "reason": str(exc),
                        "attempt": retry_attempt,
                        "max_attempts": MAX_OLLAMA_RETRIES + 1,
                        "wait_seconds": backoff,
                    }
                    if _interruptible_sleep(backoff, cancel_event):
                        _logger.info("Cancelled while waiting to retry.")
                        yield {"type": "cancelled"}
                        return
                    continue
                except (OllamaModelNotFoundError, OllamaAPIError) as exc:
                    _logger.error("Non-retryable Ollama error: %s", exc)
                    yield {"type": "error", "message": str(exc)}
                    return

            llm_call_count += 1
            full_content = "".join(content_parts)

            if not tool_calls:
                tool_calls = _parse_fallback_tool_calls(full_content) or []
                if tool_calls:
                    full_content = ""  # it was a tool-call attempt, not a message to show

            if not tool_calls:
                needs_correction = (
                    not mutating_tool_called_this_turn
                    and narration_retry_count < MAX_NARRATION_RETRIES
                    and _looks_like_unactioned_narration(full_content)
                )
                if needs_correction:
                    narration_retry_count += 1
                    _logger.info(
                        "Narration without a tool call detected (attempt %d/%d); redirecting instead "
                        "of accepting it as the final answer.",
                        narration_retry_count, MAX_NARRATION_RETRIES,
                    )
                    messages.append({"role": "assistant", "content": full_content})
                    if narration_retry_count == 1:
                        correction = (
                            "The requested change has not been performed yet. Continue the task by "
                            "using the appropriate tool now (edit_file, write_file, delete_file, "
                            "rename_file, or run_command) instead of describing or asking about it."
                        )
                    else:
                        correction = (
                            "You still have not called a tool. Stop explaining what you would do and "
                            "call the tool that actually performs the change in your very next "
                            "response -- no further description or questions."
                        )
                    messages.append({"role": "user", "content": correction})
                    yield {
                        "type": "narration_redirected",
                        "text": full_content,
                        "attempt": narration_retry_count,
                        "max_attempts": MAX_NARRATION_RETRIES,
                    }
                    continue

                if full_content:
                    yield {"type": "content", "text": full_content}
                messages.append({"role": "assistant", "content": full_content})
                final_event = {"type": "final", "text": full_content}
                if narration_retry_count >= MAX_NARRATION_RETRIES and _looks_like_unactioned_narration(
                    full_content
                ):
                    final_event["task_incomplete"] = True
                yield final_event
                return

            messages.append(
                {"role": "assistant", "content": full_content, "tool_calls": tool_calls}
            )

            for call in tool_calls:
                tool_call_count += 1
                function = call.get("function", {}) or {}
                name = function.get("name", "")
                raw_args = function.get("arguments")
                if isinstance(raw_args, str):
                    # Most models send an object, but be forgiving of a JSON string.
                    try:
                        raw_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        raw_args = {}

                try:
                    signature_args = json.dumps(raw_args, sort_keys=True, default=str)
                except TypeError:
                    signature_args = repr(raw_args)
                signature = (name, signature_args)

                if signature == last_call_signature:
                    consecutive_call_count += 1
                else:
                    consecutive_call_count = 1
                    last_call_signature = signature

                if consecutive_call_count >= MAX_CONSECUTIVE_IDENTICAL_CALLS:
                    _logger.warning(
                        "Repetition detected: %s called %d times in a row with identical arguments.",
                        name, consecutive_call_count,
                    )
                    tool_output = (
                        f"This exact {name} call has been repeated {consecutive_call_count} times in a "
                        "row without a different outcome. Stop repeating it -- inspect the actual error "
                        "or result above, try a meaningfully different approach, or ask the user for "
                        "guidance instead of trying the same thing again."
                    )
                    yield {
                        "type": "repetition_detected",
                        "name": name,
                        "args": raw_args,
                        "count": consecutive_call_count,
                    }
                    messages.append({"role": "tool", "tool_name": name, "content": tool_output})
                    continue

                if name in MUTATING_TOOL_NAMES:
                    mutating_tool_called_this_turn = True

                result = registry.execute(name, raw_args)

                if name == "read_file" and result.ok:
                    if result.cache_hit:
                        cache_hits += 1
                    else:
                        cache_misses += 1

                if result.ok and result.pending_change is not None:
                    change = result.pending_change
                    if auto_approve_edits:
                        approved = True
                        yield {"type": "confirm", "name": name, "change": change, "auto_approved": True}
                    else:
                        approved = yield {"type": "confirm", "name": name, "change": change}
                    if approved:
                        apply_result = apply_change(change, tracker)
                        if apply_result.ok:
                            yield {"type": "change_applied", "name": name, "path": change.path}
                            context_manager.record_file_modified(task_state, change.path)
                        else:
                            yield {"type": "tool_error", "name": name, "message": apply_result.output}
                        tool_output = apply_result.output
                    else:
                        tool_output = "User rejected this change. The file was not modified."
                        yield {"type": "change_rejected", "name": name, "path": change.path}
                elif result.ok and result.pending_file_op is not None:
                    file_op = result.pending_file_op
                    if auto_approve_edits:
                        approved = True
                        yield {
                            "type": "confirm_file_op",
                            "name": name,
                            "operation": file_op,
                            "auto_approved": True,
                        }
                    else:
                        approved = yield {"type": "confirm_file_op", "name": name, "operation": file_op}
                    if approved:
                        apply_result = apply_file_op(file_op, tracker)
                        if apply_result.ok:
                            yield {"type": "file_op_applied", "name": name, "operation": file_op}
                            context_manager.record_file_removed(task_state, file_op.source_path)
                            if file_op.kind == "rename":
                                context_manager.record_file_modified(
                                    task_state, file_op.destination_path
                                )
                        else:
                            yield {"type": "tool_error", "name": name, "message": apply_result.output}
                        tool_output = apply_result.output
                    else:
                        tool_output = "User rejected this file operation. Nothing was changed."
                        yield {"type": "file_op_rejected", "name": name, "operation": file_op}
                elif result.ok and result.pending_command is not None:
                    cmd = result.pending_command
                    approved = yield {"type": "confirm_command", "name": name, "command": cmd}
                    if approved:
                        yield {"type": "command_started", "name": name, "display": describe_command(cmd)}
                        exec_result = execute_command(cmd, cancel_event=cancel_event)
                        yield {"type": "command_result", "name": name, "result": exec_result}
                        context_manager.record_command_result(task_state, cmd, exec_result)
                        if exec_result.cancelled:
                            yield {
                                "type": "tool_error",
                                "name": name,
                                "message": "Command was cancelled by the user and terminated.",
                            }
                        elif exec_result.timed_out:
                            yield {
                                "type": "tool_error",
                                "name": name,
                                "message": f"Command timed out after {cmd.timeout}s and was terminated.",
                            }
                        tool_output = format_result_for_model(exec_result)
                    else:
                        tool_output = "Command was not executed. The user rejected running it."
                        yield {"type": "command_rejected", "name": name}
                elif result.ok and result.pending_git_operation is not None:
                    git_op = result.pending_git_operation
                    approved = yield {"type": "confirm_git_operation", "name": name, "operation": git_op}
                    if approved:
                        apply_result = apply_git_operation(git_op)
                        if apply_result.ok:
                            yield {"type": "git_operation_applied", "name": name, "message": apply_result.output}
                            context_manager.record_git_result(task_state, git_op.kind, True, apply_result.output)
                        else:
                            yield {"type": "tool_error", "name": name, "message": apply_result.output}
                        tool_output = apply_result.output
                    else:
                        tool_output = "User rejected this Git operation. Nothing was changed."
                        yield {"type": "git_operation_rejected", "name": name, "operation": git_op}
                elif result.ok and result.pending_plan is not None:
                    plan = result.pending_plan
                    if auto_approve_edits:
                        approved = True
                        yield {"type": "confirm_plan", "name": name, "plan": plan, "auto_approved": True}
                    else:
                        approved = yield {"type": "confirm_plan", "name": name, "plan": plan}
                    if approved:
                        if task_state is not None:
                            task_state.goal = plan.goal
                            task_state.plan = plan
                        tool_output = "Plan approved by the user and is now the active plan:\n" + "\n".join(
                            plan.render_lines()
                        )
                        yield {"type": "plan_approved", "name": name, "plan": plan}
                    else:
                        tool_output = (
                            "The user did not approve this plan. Ask what they'd like changed, propose a "
                            "revised plan, or continue without one if they'd rather you proceed directly."
                        )
                        yield {"type": "plan_rejected", "name": name}
                else:
                    display = result.display or _describe_call(name, raw_args)
                    yield {"type": "tool_call", "name": name, "args": raw_args, "display": display}
                    if not result.ok:
                        _logger.info(
                            "Tool %s failed (%s, recoverable=%s): %s",
                            name, result.error_type, result.recoverable, result.output,
                        )
                        yield {"type": "tool_error", "name": name, "message": result.output}
                    elif name == "read_file":
                        context_manager.record_read_file(task_state, raw_args.get("path", ""))
                    tool_output = result.output

                tool_msg = {"role": "tool", "tool_name": name, "content": tool_output}
                if result.cache_hit:
                    tool_msg["cache_hit"] = True
                messages.append(tool_msg)

        _logger.warning("Turn stopped: reached max_iterations (%d) without a final answer.", max_iterations)
        yield {"type": "max_iterations"}
    finally:
        final_chars = sum(len(m.get("content") or "") for m in messages)
        estimated_tokens = context_budget.estimate_tokens_from_chars(final_chars)
        _logger.debug(
            "context=%d tokens (%d chars) llm_calls=%d tool_calls=%d cache=%d hits/%d misses "
            "compaction=%d superseded/%d stale/%d trimmed",
            estimated_tokens,
            final_chars,
            llm_call_count,
            tool_call_count,
            cache_hits,
            cache_misses,
            compaction_totals.superseded,
            compaction_totals.stale,
            compaction_totals.trimmed,
        )


def _interruptible_sleep(seconds: float, cancel_event: Optional[threading.Event]) -> bool:
    """Sleep in short increments so a retry backoff can still be cut short
    by Stop Task instead of blocking the whole wait. Returns True if
    cancellation was observed during the sleep.
    """
    if cancel_event is None:
        time.sleep(seconds)
        return False
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if cancel_event.is_set():
            return True
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    return cancel_event.is_set()


def _describe_call(name: str, args: dict) -> str:
    if not args:
        return f"{name}()"
    parts = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return f"{name}({parts})"


def _normalize_tool_call_obj(obj) -> Optional[dict]:
    if not isinstance(obj, dict):
        return None
    name = obj.get("name")
    arguments = obj.get("arguments", {})
    if not isinstance(name, str) or not name or not isinstance(arguments, dict):
        return None
    return {"function": {"name": name, "arguments": arguments}}


def _extract_json_objects(text: str) -> List[dict]:
    """Find every top-level {...} JSON object embedded anywhere in `text`,
    ignoring braces that appear inside quoted strings.
    """
    objects: List[dict] = []
    depth = 0
    start: Optional[int] = None
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidate = text[start : i + 1]
                    try:
                        objects.append(json.loads(candidate))
                    except json.JSONDecodeError:
                        pass
                    start = None

    return objects


def _parse_fallback_tool_calls(content: str) -> Optional[List[dict]]:
    text = content.strip()
    if not text:
        return None

    tag_blocks = _TOOL_CALL_TAG_RE.findall(text)
    search_space = "\n".join(tag_blocks) if tag_blocks else text

    calls = [
        normalized
        for obj in _extract_json_objects(search_space)
        if (normalized := _normalize_tool_call_obj(obj))
    ]
    return calls or None
