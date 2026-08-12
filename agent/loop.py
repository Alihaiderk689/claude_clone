"""The tool-calling agent loop.

Drives one user turn: ask the model, execute any tool calls it requests,
feed the results back, and repeat until it produces a plain-text final
answer or a safety limit is hit. Knows nothing about the terminal — it
yields structured events for a caller (cli.py) to render.
"""
from __future__ import annotations

import json
import re
from typing import Dict, Iterator, List, Optional

from .ollama_client import (
    OllamaAPIError,
    OllamaClient,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
)
from .tools.editing import apply_change
from .tools.git import apply_git_operation
from .tools.registry import ToolRegistry
from .tools.state import FileStateTracker
from .tools.terminal import describe_command, execute_command, format_result_for_model

MAX_TOOL_ITERATIONS = 10

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
    max_iterations: int = MAX_TOOL_ITERATIONS,
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
            only path to apply_change() actually touching the filesystem.
        {"type": "change_applied", "name", "path"}   -- approved, written, verified
        {"type": "change_rejected", "name", "path"}  -- user declined; untouched
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
        {"type": "content", "text"}   -- the final answer text to display
        {"type": "final", "text"}     -- the complete final answer (same text)
        {"type": "error", "message"}  -- fatal Ollama error; turn stops
        {"type": "max_iterations"}    -- safety limit hit without a final answer

    Each iteration's response is fully buffered before being shown, because a
    tool call can arrive embedded in ordinary assistant content (see above)
    at an unpredictable position -- there's no safe prefix-only check that
    would let content stream live without risking a raw tool call leaking
    into the terminal as if it were the answer.
    """
    tools_schema = registry.schemas()

    for _ in range(max_iterations):
        content_parts: List[str] = []
        tool_calls: List[dict] = []

        try:
            for update in client.chat(messages, tools=tools_schema):
                if update["content"]:
                    content_parts.append(update["content"])
                if update["tool_calls"]:
                    tool_calls.extend(update["tool_calls"])
        except (
            OllamaConnectionError,
            OllamaModelNotFoundError,
            OllamaTimeoutError,
            OllamaAPIError,
        ) as exc:
            yield {"type": "error", "message": str(exc)}
            return

        full_content = "".join(content_parts)

        if not tool_calls:
            tool_calls = _parse_fallback_tool_calls(full_content) or []
            if tool_calls:
                full_content = ""  # it was a tool-call attempt, not a message to show

        if not tool_calls:
            if full_content:
                yield {"type": "content", "text": full_content}
            messages.append({"role": "assistant", "content": full_content})
            yield {"type": "final", "text": full_content}
            return

        messages.append(
            {"role": "assistant", "content": full_content, "tool_calls": tool_calls}
        )

        for call in tool_calls:
            function = call.get("function", {}) or {}
            name = function.get("name", "")
            raw_args = function.get("arguments")
            if isinstance(raw_args, str):
                # Most models send an object, but be forgiving of a JSON string.
                try:
                    raw_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    raw_args = {}

            result = registry.execute(name, raw_args)

            if result.ok and result.pending_change is not None:
                change = result.pending_change
                approved = yield {"type": "confirm", "name": name, "change": change}
                if approved:
                    apply_result = apply_change(change, tracker)
                    if apply_result.ok:
                        yield {"type": "change_applied", "name": name, "path": change.path}
                    else:
                        yield {"type": "tool_error", "name": name, "message": apply_result.output}
                    tool_output = apply_result.output
                else:
                    tool_output = "User rejected this change. The file was not modified."
                    yield {"type": "change_rejected", "name": name, "path": change.path}
            elif result.ok and result.pending_command is not None:
                cmd = result.pending_command
                approved = yield {"type": "confirm_command", "name": name, "command": cmd}
                if approved:
                    yield {"type": "command_started", "name": name, "display": describe_command(cmd)}
                    exec_result = execute_command(cmd)
                    yield {"type": "command_result", "name": name, "result": exec_result}
                    if exec_result.timed_out:
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
                    else:
                        yield {"type": "tool_error", "name": name, "message": apply_result.output}
                    tool_output = apply_result.output
                else:
                    tool_output = "User rejected this Git operation. Nothing was changed."
                    yield {"type": "git_operation_rejected", "name": name, "operation": git_op}
            else:
                display = result.display or _describe_call(name, raw_args)
                yield {"type": "tool_call", "name": name, "args": raw_args, "display": display}
                if not result.ok:
                    yield {"type": "tool_error", "name": name, "message": result.output}
                tool_output = result.output

            messages.append(
                {"role": "tool", "tool_name": name, "content": tool_output}
            )

    yield {"type": "max_iterations"}


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
