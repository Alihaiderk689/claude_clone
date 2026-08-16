"""Approximate context-size reporting for a 3B model on 8GB unified memory.

This module is deliberately a reporting/debug layer only -- it does not
replace `context_manager.py`'s proven character-count budget enforcement
(`MAX_CONTEXT_CHARS`). No tokenizer dependency is added; `estimate_tokens`
is a simple, cheap heuristic good enough to show roughly how large the
context is getting, not to drive any hard cutoff itself.
"""
from __future__ import annotations

import os

from .context_manager import MAX_CONTEXT_CHARS

CHARS_PER_TOKEN_ESTIMATE = 4


def estimate_tokens_from_chars(chars: int) -> int:
    """Same heuristic as estimate_tokens(), for callers that already have a
    character count (e.g. loop.py's debug logging, benchmark.py's peak
    context size) and shouldn't build/re-scan a string just to measure it."""
    if chars <= 0:
        return 0
    return -(-chars // CHARS_PER_TOKEN_ESTIMATE)


def estimate_tokens(text: str) -> int:
    """Rough token count: len(text) / 4, rounded up. Good enough to show
    whether context is growing, not precise enough to enforce a hard limit
    against -- that's still done in chars in context_manager.py."""
    return estimate_tokens_from_chars(len(text)) if text else 0


def max_context_chars_from_env() -> int:
    """Reads CODE_AGENT_MAX_CONTEXT_CHARS the same way OLLAMA_TIMEOUT reads
    its env var (see ollama_client.timeout_from_env). Falls back to
    context_manager.MAX_CONTEXT_CHARS for a missing, non-numeric, or
    non-positive value rather than raising."""
    raw = os.environ.get("CODE_AGENT_MAX_CONTEXT_CHARS")
    if not raw:
        return MAX_CONTEXT_CHARS
    try:
        value = int(raw)
    except ValueError:
        return MAX_CONTEXT_CHARS
    return value if value > 0 else MAX_CONTEXT_CHARS
