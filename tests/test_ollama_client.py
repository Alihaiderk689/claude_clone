"""Unit tests for agent.ollama_client.

These tests mock the `requests` library entirely, so they do not
require a running Ollama server or a downloaded model.
"""
from __future__ import annotations

import json
import threading
from unittest import mock

import pytest
import requests

from agent.ollama_client import (
    OllamaAPIError,
    OllamaCancelledError,
    OllamaClient,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
)


def make_stream_response(lines, status_code=200):
    response = mock.Mock()
    response.status_code = status_code
    response.text = ""
    response.iter_lines.return_value = [
        line.encode("utf-8") if isinstance(line, str) else line for line in lines
    ]
    return response


class TestCheckConnection:
    @mock.patch("agent.ollama_client.requests.get")
    def test_returns_true_when_server_responds_ok(self, mock_get):
        mock_get.return_value = mock.Mock(status_code=200)
        client = OllamaClient()
        assert client.check_connection() is True

    @mock.patch("agent.ollama_client.requests.get")
    def test_returns_false_when_status_not_ok(self, mock_get):
        mock_get.return_value = mock.Mock(status_code=500)
        client = OllamaClient()
        assert client.check_connection() is False

    @mock.patch("agent.ollama_client.requests.get")
    def test_returns_false_on_connection_error(self, mock_get):
        mock_get.side_effect = requests.exceptions.ConnectionError()
        client = OllamaClient()
        assert client.check_connection() is False


class TestChatStream:
    @mock.patch("agent.ollama_client.requests.post")
    def test_yields_streamed_content_chunks(self, mock_post):
        lines = [
            json.dumps({"message": {"role": "assistant", "content": "Hello"}, "done": False}),
            json.dumps({"message": {"role": "assistant", "content": ", world"}, "done": False}),
            json.dumps({"message": {"role": "assistant", "content": ""}, "done": True}),
        ]
        mock_post.return_value = make_stream_response(lines)

        client = OllamaClient()
        chunks = list(client.chat_stream([{"role": "user", "content": "hi"}]))

        assert chunks == ["Hello", ", world"]

    @mock.patch("agent.ollama_client.requests.post")
    def test_skips_blank_and_malformed_lines(self, mock_post):
        lines = [
            "",
            "not json",
            json.dumps({"message": {"role": "assistant", "content": "ok"}, "done": True}),
        ]
        mock_post.return_value = make_stream_response(lines)

        client = OllamaClient()
        chunks = list(client.chat_stream([{"role": "user", "content": "hi"}]))

        assert chunks == ["ok"]

    @mock.patch("agent.ollama_client.requests.post")
    def test_connection_error_raises_ollama_connection_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.ConnectionError()
        client = OllamaClient()

        with pytest.raises(OllamaConnectionError):
            list(client.chat_stream([{"role": "user", "content": "hi"}]))

    @mock.patch("agent.ollama_client.requests.post")
    def test_timeout_raises_ollama_timeout_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.Timeout()
        client = OllamaClient()

        with pytest.raises(OllamaTimeoutError):
            list(client.chat_stream([{"role": "user", "content": "hi"}]))

    @mock.patch("agent.ollama_client.requests.post")
    def test_404_status_raises_model_not_found(self, mock_post):
        mock_post.return_value = make_stream_response([], status_code=404)
        client = OllamaClient(model="does-not-exist")

        with pytest.raises(OllamaModelNotFoundError):
            list(client.chat_stream([{"role": "user", "content": "hi"}]))

    @mock.patch("agent.ollama_client.requests.post")
    def test_other_bad_status_raises_api_error(self, mock_post):
        mock_post.return_value = make_stream_response([], status_code=500)
        client = OllamaClient()

        with pytest.raises(OllamaAPIError):
            list(client.chat_stream([{"role": "user", "content": "hi"}]))

    @mock.patch("agent.ollama_client.requests.post")
    def test_error_field_in_stream_raises_api_error(self, mock_post):
        lines = [json.dumps({"error": "something went wrong"})]
        mock_post.return_value = make_stream_response(lines)
        client = OllamaClient()

        with pytest.raises(OllamaAPIError):
            list(client.chat_stream([{"role": "user", "content": "hi"}]))

    @mock.patch("agent.ollama_client.requests.post")
    def test_error_field_mentioning_not_found_raises_model_not_found(self, mock_post):
        lines = [json.dumps({"error": "model 'foo' not found, try pulling it first"})]
        mock_post.return_value = make_stream_response(lines)
        client = OllamaClient(model="foo")

        with pytest.raises(OllamaModelNotFoundError):
            list(client.chat_stream([{"role": "user", "content": "hi"}]))

    def test_host_trailing_slash_is_stripped(self):
        client = OllamaClient(host="http://localhost:11434/")
        assert client.host == "http://localhost:11434"


class TestCancelEvent:
    """cancel_event is Phase 7's additive Stop Task mechanism -- must be a
    no-op when omitted/unset, and must interrupt streaming promptly when
    set, without needing the server to finish sending its response."""

    @mock.patch("agent.ollama_client.requests.post")
    def test_no_cancel_event_behaves_as_before(self, mock_post):
        lines = [json.dumps({"message": {"content": "hi"}, "done": True})]
        mock_post.return_value = make_stream_response(lines)
        client = OllamaClient()

        result = list(client.chat_stream([{"role": "user", "content": "hi"}]))
        assert result == ["hi"]

    @mock.patch("agent.ollama_client.requests.post")
    def test_unset_cancel_event_does_not_raise(self, mock_post):
        lines = [json.dumps({"message": {"content": "hi"}, "done": True})]
        mock_post.return_value = make_stream_response(lines)
        client = OllamaClient()

        result = list(
            client.chat_stream([{"role": "user", "content": "hi"}], cancel_event=threading.Event())
        )
        assert result == ["hi"]

    @mock.patch("agent.ollama_client.requests.post")
    def test_set_cancel_event_raises_before_processing_further_lines(self, mock_post):
        lines = [
            json.dumps({"message": {"content": "hi"}, "done": False}),
            json.dumps({"message": {"content": " there"}, "done": True}),
        ]
        mock_post.return_value = make_stream_response(lines)
        client = OllamaClient()

        cancel_event = threading.Event()
        cancel_event.set()

        with pytest.raises(OllamaCancelledError):
            list(
                client.chat_stream(
                    [{"role": "user", "content": "hi"}], cancel_event=cancel_event
                )
            )

    @mock.patch("agent.ollama_client.requests.post")
    def test_chat_forwards_cancel_event(self, mock_post):
        lines = [json.dumps({"message": {"content": "hi"}, "done": True})]
        mock_post.return_value = make_stream_response(lines)
        client = OllamaClient()
        cancel_event = threading.Event()
        cancel_event.set()

        with pytest.raises(OllamaCancelledError):
            list(client.chat([{"role": "user", "content": "hi"}], cancel_event=cancel_event))


class TestMalformedResponses:
    """Phase 8: a stream line that's valid JSON but not the object shape
    the API contract promises must be skipped, not crash the client with
    an AttributeError/TypeError from blindly calling .get() on it."""

    @mock.patch("agent.ollama_client.requests.post")
    def test_bare_json_list_line_is_skipped(self, mock_post):
        lines = [
            json.dumps([1, 2, 3]),
            json.dumps({"message": {"content": "hi"}, "done": True}),
        ]
        mock_post.return_value = make_stream_response(lines)
        client = OllamaClient()

        result = list(client.chat_stream([{"role": "user", "content": "hi"}]))
        assert result == ["hi"]

    @mock.patch("agent.ollama_client.requests.post")
    def test_bare_json_string_line_is_skipped(self, mock_post):
        lines = [json.dumps("just a string"), json.dumps({"message": {"content": "ok"}, "done": True})]
        mock_post.return_value = make_stream_response(lines)
        client = OllamaClient()

        result = list(client.chat_stream([{"role": "user", "content": "hi"}]))
        assert result == ["ok"]

    @mock.patch("agent.ollama_client.requests.post")
    def test_message_field_not_a_dict_does_not_crash(self, mock_post):
        lines = [json.dumps({"message": "not an object", "done": True})]
        mock_post.return_value = make_stream_response(lines)
        client = OllamaClient()

        result = list(client.chat_stream([{"role": "user", "content": "hi"}]))
        assert result == []  # no content extracted, but no crash

    @mock.patch("agent.ollama_client.requests.post")
    def test_tool_calls_field_not_a_list_is_ignored(self, mock_post):
        lines = [json.dumps({"message": {"content": "", "tool_calls": "not-a-list"}, "done": True})]
        mock_post.return_value = make_stream_response(lines)
        client = OllamaClient()

        updates_list = list(client.chat([{"role": "user", "content": "hi"}]))
        assert updates_list[0]["tool_calls"] is None

    @mock.patch("agent.ollama_client.requests.post")
    def test_content_field_not_a_string_is_ignored(self, mock_post):
        lines = [json.dumps({"message": {"content": 12345}, "done": True})]
        mock_post.return_value = make_stream_response(lines)
        client = OllamaClient()

        updates_list = list(client.chat([{"role": "user", "content": "hi"}]))
        assert updates_list[0]["content"] == ""
