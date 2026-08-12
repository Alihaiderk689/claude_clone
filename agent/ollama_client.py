"""Thin client for talking to a local Ollama server's chat API.

This module knows nothing about the terminal UI. It only knows how to
send a conversation to Ollama and yield back streamed text chunks, or
raise a specific exception describing what went wrong.
"""
from __future__ import annotations

import json
from typing import Dict, Iterator, List, Optional

import requests

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5-coder:7b"
DEFAULT_TIMEOUT = 120  # generation on CPU can be slow; keep this generous


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

    def chat_stream(self, messages: List[Dict[str, str]]) -> Iterator[str]:
        """Send a conversation to Ollama and yield assistant text as it arrives.

        `messages` follows the standard {"role": ..., "content": ...} format
        with roles "system", "user", and "assistant". No tools are offered.
        """
        for update in self._chat_updates(messages, tools=None):
            if update["content"]:
                yield update["content"]

    def chat(self, messages: List[Dict], tools: Optional[List[dict]] = None) -> Iterator[dict]:
        """Send a conversation to Ollama, optionally offering tools, and yield
        structured updates as they stream in:

            {"content": str, "tool_calls": list | None, "done": bool}

        `tool_calls`, when present, follows Ollama's native function-calling
        format: [{"function": {"name": ..., "arguments": {...}}}, ...].
        """
        return self._chat_updates(messages, tools=tools)

    def _chat_updates(
        self, messages: List[Dict], tools: Optional[List[dict]]
    ) -> Iterator[dict]:
        url = f"{self.host}/api/chat"
        payload = {"model": self.model, "messages": messages, "stream": True}
        if tools:
            payload["tools"] = tools

        response = self._post_stream(url, payload)

        try:
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "error" in data:
                    error_msg = str(data["error"])
                    if "not found" in error_msg.lower():
                        raise OllamaModelNotFoundError(
                            f"Model '{self.model}' was not found. "
                            f"Pull it first with: ollama pull {self.model}"
                        )
                    raise OllamaAPIError(f"Ollama returned an error: {error_msg}")

                message = data.get("message") or {}
                yield {
                    "content": message.get("content") or "",
                    "tool_calls": message.get("tool_calls") or None,
                    "done": bool(data.get("done")),
                }

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
