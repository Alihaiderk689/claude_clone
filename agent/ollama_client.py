"""Thin client for talking to a local Ollama server's chat API.

This module knows nothing about the terminal UI. It only knows how to
send a conversation to Ollama and yield back streamed text chunks, or
raise a specific exception describing what went wrong.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Dict, Iterator, List, Optional

import requests

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5-coder:3b"
DEFAULT_TIMEOUT = 120  # generation on CPU can be slow; keep this generous
# Ollama's own server default is "5m" -- short enough that a normal pause
# between agent turns (reading a diff, deciding what to ask next) can let the
# model fall out of memory, so the *next* message pays a multi-second-to-
# tens-of-seconds reload. 30m keeps it resident for the length of a typical
# session instead.
DEFAULT_KEEP_ALIVE = "30m"
# Left unset before this: Ollama fell back to qwen2.5-coder's own Modelfile
# default (0.7-0.8 range), tuned for open-ended chat, not for reliably
# emitting well-formed tool-call JSON on every turn. A lower temperature
# measurably reduces exactly the kind of drift this codebase already fights
# elsewhere (see loop.py's fallback tool-call parsing and narration
# detection) -- more consistent formatting, less creative rewording of "call
# the tool" into "let me describe what I'd do." Not 0.0: a small amount of
# variance still helps the model recover after a rejected/failed call
# instead of deterministically repeating the exact same (wrong) attempt.
DEFAULT_TEMPERATURE = 0.2
DEFAULT_TOP_P = 0.9


def _float_from_env(var_name: str, default: float, *, minimum: float, maximum: float) -> float:
    """Same tolerant-fallback shape as timeout_from_env()/keep_alive_from_env():
    a missing, non-numeric, or out-of-range value degrades to the default
    instead of raising or silently sending Ollama a nonsensical setting."""
    raw = os.environ.get(var_name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if minimum <= value <= maximum else default


def temperature_from_env() -> float:
    return _float_from_env("OLLAMA_TEMPERATURE", DEFAULT_TEMPERATURE, minimum=0.0, maximum=2.0)


def top_p_from_env() -> float:
    return _float_from_env("OLLAMA_TOP_P", DEFAULT_TOP_P, minimum=0.0, maximum=1.0)


def keep_alive_from_env() -> str:
    """Reads OLLAMA_KEEP_ALIVE the same way OLLAMA_TIMEOUT is read. Value is
    passed through as-is (Ollama accepts durations like "30m"/"1h" or a
    plain number of seconds; "-1" means never unload). Falls back to
    DEFAULT_KEEP_ALIVE for a missing/empty value rather than raising."""
    raw = os.environ.get("OLLAMA_KEEP_ALIVE")
    return raw if raw else DEFAULT_KEEP_ALIVE


def timeout_from_env() -> float:
    """Reads OLLAMA_TIMEOUT (seconds) the same way host/model read
    OLLAMA_HOST/OLLAMA_MODEL. Falls back to DEFAULT_TIMEOUT for a missing,
    non-numeric, or non-positive value rather than raising -- a bad env var
    should degrade to the safe default, not crash startup."""
    raw = os.environ.get("OLLAMA_TIMEOUT")
    if not raw:
        return DEFAULT_TIMEOUT
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_TIMEOUT
    return value if value > 0 else DEFAULT_TIMEOUT


class OllamaError(Exception):
    """Base class for all errors raised by OllamaClient."""


class OllamaConnectionError(OllamaError):
    """Raised when the Ollama server cannot be reached at all."""


class OllamaTimeoutError(OllamaError):
    """Raised when a request to Ollama times out."""


class OllamaModelNotFoundError(OllamaError):
    """Raised when the configured model isn't available locally."""


class OllamaAPIError(OllamaError):
    """Raised for any other error the Ollama API reports."""


class OllamaCancelledError(OllamaError):
    """Raised when a caller-supplied cancel_event was set during streaming."""


class OllamaClient:
    """Minimal client for the Ollama /api/chat and /api/tags endpoints."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        keep_alive: str = DEFAULT_KEEP_ALIVE,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.temperature = temperature
        self.top_p = top_p

    def check_connection(self) -> bool:
        """Return True if the Ollama server responds, False otherwise."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def warm_up(self) -> None:
        """Block until the model is loaded into Ollama's memory.

        Sends a promptless /api/generate call, which Ollama loads the model
        for but never generates any tokens against. Called once at startup
        so the (potentially many-second) cold load happens up front with
        clear "loading" messaging, instead of silently during the user's
        first real chat message where it looks like a connection failure or
        random slowness. Raises the same OllamaError subclasses as chat().
        """
        url = f"{self.host}/api/generate"
        payload = {"model": self.model, "keep_alive": self.keep_alive}
        response = self._post_stream(url, payload)
        for _ in response.iter_lines():
            break

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        cancel_event: Optional[threading.Event] = None,
    ) -> Iterator[str]:
        """Send a conversation to Ollama and yield assistant text as it arrives.

        `messages` follows the standard {"role": ..., "content": ...} format
        with roles "system", "user", and "assistant". No tools are offered.
        """
        for update in self._chat_updates(messages, tools=None, cancel_event=cancel_event):
            if update["content"]:
                yield update["content"]

    def chat(
        self,
        messages: List[Dict],
        tools: Optional[List[dict]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Iterator[dict]:
        """Send a conversation to Ollama, optionally offering tools, and yield
        structured updates as they stream in:

            {"content": str, "tool_calls": list | None, "done": bool}

        `tool_calls`, when present, follows Ollama's native function-calling
        format: [{"function": {"name": ..., "arguments": {...}}}, ...].

        If `cancel_event` is given and becomes set while streaming, iteration
        stops early and raises OllamaCancelledError -- used by the local HTTP
        server's Stop Task endpoint. Left as None (the default), behavior is
        identical to before this parameter existed.
        """
        return self._chat_updates(messages, tools=tools, cancel_event=cancel_event)

    def _chat_updates(
        self,
        messages: List[Dict],
        tools: Optional[List[dict]],
        cancel_event: Optional[threading.Event] = None,
    ) -> Iterator[dict]:
        url = f"{self.host}/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {"temperature": self.temperature, "top_p": self.top_p},
        }
        if tools:
            payload["tools"] = tools

        response = self._post_stream(url, payload)

        try:
            for line in response.iter_lines():
                if cancel_event is not None and cancel_event.is_set():
                    raise OllamaCancelledError("Generation was cancelled.")
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(data, dict):
                    # Valid JSON but not the object shape the API contract
                    # promises (e.g. a bare string/number/list) -- treat as
                    # one malformed line, not a fatal error for the stream.
                    continue

                if "error" in data:
                    error_msg = str(data["error"])
                    if "not found" in error_msg.lower():
                        raise OllamaModelNotFoundError(
                            f"Model '{self.model}' was not found. "
                            f"Pull it first with: ollama pull {self.model}"
                        )
                    raise OllamaAPIError(f"Ollama returned an error: {error_msg}")

                message = data.get("message")
                if not isinstance(message, dict):
                    message = {}
                tool_calls = message.get("tool_calls")
                if not isinstance(tool_calls, list):
                    tool_calls = None
                content = message.get("content")
                if not isinstance(content, str):
                    content = ""
                update = {
                    "content": content,
                    "tool_calls": tool_calls,
                    "done": bool(data.get("done")),
                }
                # Ollama includes these only on the final chunk of a
                # response; surfaced for the local benchmark harness
                # (agent/benchmark.py) to report "tokens if available" --
                # purely informational, no behavior depends on them.
                if isinstance(data.get("eval_count"), int):
                    update["eval_count"] = data["eval_count"]
                if isinstance(data.get("prompt_eval_count"), int):
                    update["prompt_eval_count"] = data["prompt_eval_count"]
                yield update

                if data.get("done"):
                    break
        except requests.exceptions.ChunkedEncodingError as exc:
            raise OllamaConnectionError(
                "Connection to Ollama was interrupted while streaming the response."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise OllamaTimeoutError(f"Connection to Ollama at {self.host} timed out.") from exc

    def _post_stream(self, url: str, payload: dict) -> requests.Response:
        try:
            response = requests.post(url, json=payload, stream=True, timeout=self.timeout)
        except requests.exceptions.ConnectionError as exc:
            raise OllamaConnectionError(
                f"Could not connect to Ollama at {self.host}. Make sure Ollama is running."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise OllamaTimeoutError(f"Connection to Ollama at {self.host} timed out.") from exc
        except requests.exceptions.RequestException as exc:
            raise OllamaAPIError(f"Unexpected error contacting Ollama: {exc}") from exc

        if response.status_code == 404:
            raise OllamaModelNotFoundError(
                f"Model '{self.model}' was not found. Pull it first with: ollama pull {self.model}"
            )
        if response.status_code != 200:
            raise OllamaAPIError(
                f"Ollama returned an unexpected status code {response.status_code}: {response.text}"
            )
        return response
