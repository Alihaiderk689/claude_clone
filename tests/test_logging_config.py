"""Tests for agent/logging_config.py: secret redaction and the
debug/quiet logging toggle. Phase 8 section 23 explicitly requires that
secrets (tokens, passwords, API keys) never appear in logs.
"""
from __future__ import annotations

import io
import logging

from agent.logging_config import (
    LOGGER_NAME,
    configure_logging,
    debug_enabled_from_env,
    get_logger,
    redact,
)


class TestRedact:
    def test_redacts_bearer_token(self):
        text = "Authorization: Bearer bc22239861d5bf7ae5fe45c16b413a00635a30020"
        assert "bc22239861d5bf7ae5fe45c16b413a00635a30020" not in redact(text)
        assert "REDACTED" in redact(text)

    def test_redacts_token_key_value(self):
        text = '{"token": "supersecretvalue1234"}'
        assert "supersecretvalue1234" not in redact(text)

    def test_redacts_api_key(self):
        text = "api_key=sk-1234567890abcdef"
        assert "sk-1234567890abcdef" not in redact(text)

    def test_redacts_password(self):
        text = "password: hunter2hunter2"
        assert "hunter2hunter2" not in redact(text)

    def test_does_not_redact_plain_sentence_mentioning_token(self):
        """A false positive here would make ordinary log messages
        unreadable -- only secret-shaped values (long, token-like) should
        be caught, not the English word "token" itself."""
        text = "Missing or invalid Authorization bearer token."
        assert redact(text) == text

    def test_does_not_redact_short_benign_values(self):
        text = "status: ok"
        assert redact(text) == text

    def test_leaves_unrelated_text_untouched(self):
        text = "Reading backend/auth.py (42 lines)"
        assert redact(text) == text


class TestConfigureLogging:
    def test_debug_false_sets_warning_level(self):
        logger = configure_logging(debug=False)
        assert logger.level == logging.WARNING

    def test_debug_true_sets_debug_level(self):
        logger = configure_logging(debug=True)
        assert logger.level == logging.DEBUG

    def test_reconfiguring_does_not_accumulate_handlers(self):
        configure_logging(debug=True)
        configure_logging(debug=True)
        logger = configure_logging(debug=True)
        assert len(logger.handlers) == 1

    def test_get_logger_returns_child_of_root_logger_name(self):
        logger = get_logger("loop")
        assert logger.name == f"{LOGGER_NAME}.loop"

    def test_secrets_never_reach_the_configured_handler_output(self):
        logger = configure_logging(debug=True)
        stream = io.StringIO()
        logger.handlers[0].stream = stream

        child = get_logger("test")
        child.warning("Authorization: Bearer bc22239861d5bf7ae5fe45c16b413a00635a30020cd")

        output = stream.getvalue()
        assert "bc22239861d5bf7ae5fe45c16b413a00635a30020cd" not in output
        assert "REDACTED" in output


class TestDebugEnabledFromEnv:
    def test_unset_is_false(self, monkeypatch):
        monkeypatch.delenv("CODE_AGENT_DEBUG", raising=False)
        assert debug_enabled_from_env() is False

    def test_one_is_true(self, monkeypatch):
        monkeypatch.setenv("CODE_AGENT_DEBUG", "1")
        assert debug_enabled_from_env() is True

    def test_true_string_is_true(self, monkeypatch):
        monkeypatch.setenv("CODE_AGENT_DEBUG", "true")
        assert debug_enabled_from_env() is True

    def test_zero_is_false(self, monkeypatch):
        monkeypatch.setenv("CODE_AGENT_DEBUG", "0")
        assert debug_enabled_from_env() is False
