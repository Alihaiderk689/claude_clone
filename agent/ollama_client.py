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
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def check_connection(self) -> bool:
        """Return True if the Ollama server responds, False otherwise."""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

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
        payload = {"model": self.model, "messages": messages, "stream": True}
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
