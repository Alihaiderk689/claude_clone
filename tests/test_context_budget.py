"""Unit tests for agent.context_budget."""
from __future__ import annotations

from agent.context_budget import (
    CHARS_PER_TOKEN_ESTIMATE,
    estimate_tokens,
    estimate_tokens_from_chars,
    max_context_chars_from_env,
)
from agent.context_manager import MAX_CONTEXT_CHARS


class TestEstimateTokens:
    def test_empty_string_is_zero(self):
        assert estimate_tokens("") == 0

    def test_rounds_up(self):
        # 5 chars / 4 per token -> 2 tokens (rounded up), not 1
        assert estimate_tokens("abcde") == 2

    def test_exact_multiple(self):
        assert estimate_tokens("a" * (CHARS_PER_TOKEN_ESTIMATE * 3)) == 3

    def test_scales_with_length(self):
        assert estimate_tokens("a" * 4000) == 1000


class TestEstimateTokensFromChars:
    def test_zero_is_zero(self):
        assert estimate_tokens_from_chars(0) == 0

    def test_negative_is_zero(self):
        assert estimate_tokens_from_chars(-5) == 0

    def test_rounds_up(self):
        assert estimate_tokens_from_chars(5) == 2

    def test_matches_estimate_tokens_for_equivalent_text(self):
        text = "a" * 4321
        assert estimate_tokens_from_chars(len(text)) == estimate_tokens(text)


class TestMaxContextCharsFromEnv:
    def test_returns_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("CODE_AGENT_MAX_CONTEXT_CHARS", raising=False)
        assert max_context_chars_from_env() == MAX_CONTEXT_CHARS

    def test_returns_default_for_empty_string(self, monkeypatch):
        monkeypatch.setenv("CODE_AGENT_MAX_CONTEXT_CHARS", "")
        assert max_context_chars_from_env() == MAX_CONTEXT_CHARS

    def test_parses_a_valid_value(self, monkeypatch):
        monkeypatch.setenv("CODE_AGENT_MAX_CONTEXT_CHARS", "20000")
        assert max_context_chars_from_env() == 20000

    def test_returns_default_for_non_numeric_value(self, monkeypatch):
        monkeypatch.setenv("CODE_AGENT_MAX_CONTEXT_CHARS", "not-a-number")
        assert max_context_chars_from_env() == MAX_CONTEXT_CHARS

    def test_returns_default_for_zero_or_negative_value(self, monkeypatch):
        monkeypatch.setenv("CODE_AGENT_MAX_CONTEXT_CHARS", "0")
        assert max_context_chars_from_env() == MAX_CONTEXT_CHARS
        monkeypatch.setenv("CODE_AGENT_MAX_CONTEXT_CHARS", "-500")
        assert max_context_chars_from_env() == MAX_CONTEXT_CHARS
