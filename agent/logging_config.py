"""Structured application logging (Phase 8).

A thin wrapper around the stdlib `logging` module -- no new dependency.
Quiet by default (WARNING and above, to stderr, so it never interleaves
with the Rich-rendered conversation on stdout); pass `debug=True` (wired to
`code-agent --debug` / `code-agent serve --debug`, or the CODE_AGENT_DEBUG=1
environment variable) to get DEBUG-level detail. Every log line is passed
through `redact()` first -- see its docstring for exactly what that catches
and, more importantly, what it doesn't substitute for.
"""
from __future__ import annotations

import logging
import os
import re
import sys
from typing import Optional

LOGGER_NAME = "code_agent"

# Best-effort scrub of common secret *shapes* appearing in a log message,
# as defense in depth. This is NOT the reason secrets stay out of the
# logs -- the actual guarantee is that nothing in this codebase logs raw
# request bodies, environment dumps, or file contents in the first place
# (see agent/tools/terminal.py's _minimal_env, which is what keeps a
# subprocess's own environment out of captured output to begin with).
_SECRET_PATTERNS = [
    # Real bearer tokens (this project's own auth token is 64 hex chars)
    # are always much longer than any plain-English word that happens to
    # follow "bearer" in a log message describing the requirement itself
    # (e.g. "missing or invalid ... bearer token.") -- an 8-char minimum
    # avoids redacting that kind of sentence while still catching anything
    # secret-shaped.
    re.compile(r"(Bearer\s+)([A-Za-z0-9._-]{8,})", re.IGNORECASE),
    re.compile(r"((?:api[_-]?key|token|secret|password)[\"']?\s*[:=]\s*[\"']?)([A-Za-z0-9._-]{8,})", re.IGNORECASE),
]


def redact(text: str) -> str:
    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(lambda m: m.group(1) + "***REDACTED***", redacted)
    return redacted


class _RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def debug_enabled_from_env() -> bool:
    return os.environ.get("CODE_AGENT_DEBUG", "").strip().lower() in ("1", "true", "yes")


def configure_logging(debug: bool = False) -> logging.Logger:
    """Set up (or reconfigure) the root `code_agent` logger. Safe to call
    more than once -- e.g. tests reconfiguring between cases -- since it
    replaces rather than accumulates handlers.
    """
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG if debug else logging.WARNING)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_RedactingFormatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    full_name = LOGGER_NAME if not name else f"{LOGGER_NAME}.{name}"
    return logging.getLogger(full_name)


# A silent logger by default (no handlers, WARNING level, matching stdlib's
# own default) until configure_logging() is explicitly called by an entry
# point (cli.py's main()/_run_serve_command). Library code (loop.py,
# server.py) can safely call get_logger() and log at import time without
# forcing output on a caller who never opted in -- e.g. every existing test
# that imports agent.loop directly, unaffected by this module's existence.
