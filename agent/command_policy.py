"""Decides whether a proposed run_command invocation is allowed.

This module only validates -- it never executes anything (see
agent/tools/terminal.py for execution, which only ever runs a command
that has already passed validate_command() *and* been approved by the
user). The model is never trusted: the executable and every argument are
checked structurally against an explicit allowlist, never just "does this
program exist on PATH."
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

from .project import PathSecurityError, ProjectRoot

MIN_TIMEOUT_SECONDS = 1
DEFAULT_TIMEOUT_SECONDS = 120
MAX_TIMEOUT_SECONDS = 300

# Named explicitly (rather than relying only on the allowlist below) so
# rejections are clear and so this list is easy to audit and test on its
# own. Not exhaustive -- the real gate is that _VALIDATORS is a strict
# allowlist and anything not in it is rejected regardless of this list.
DENYLISTED_PROGRAMS = frozenset(
    {
        "rm", "rmdir", "sudo", "su", "shutdown", "reboot", "halt", "mkfs",
        "dd", "chmod", "chown", "chgrp", "curl", "wget", "ssh", "scp",
        "sftp", "nc", "netcat", "ncat", "telnet", "kill", "killall",
        "pkill", "diskutil", "launchctl", "systemctl", "service",
        "bash", "sh", "zsh", "fish", "ksh", "osascript", "pip", "pip3",
        "pipx", "brew", "apt", "apt-get", "dpkg", "yum", "dnf", "port",
        "nohup", "setsid", "disown", "env", "printenv", "eval", "exec",
        "xargs", "git", "docker", "vagrant",
    }
)

# Checked as literal substrings of any individual argument token. Since we
# never use shell=True, none of these can actually trigger shell parsing --
# this is deliberate defense in depth (see Phase 4 spec item 7), not the
# only thing standing between the model and shell access.
_SHELL_METACHARACTERS = (";", "&&", "||", "|", ">", "<", "$(", "`", "&", "\n", "\r", "\x00")

_PROGRAM_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")
_NPM_SCRIPT_RE = re.compile(r"^[A-Za-z0-9_:.\-]+$")

_DANGEROUS_NPM_SCRIPTS = frozenset(
    {
        "install", "i", "add", "ci", "publish", "link", "unlink", "audit",
        "update", "upgrade", "exec", "dlx", "init", "config", "outdated",
        "prune", "rebuild", "cache", "login", "logout", "adduser", "owner",
        "deprecate", "star", "org", "team", "token", "hook", "access",
        "postinstall", "preinstall",
    }
)

_PYTHON_EVAL_FLAGS = frozenset({"-c", "--command"})
_NODE_EVAL_FLAGS = frozenset({"-e", "--eval", "-p", "--print", "-i", "--interactive", "-r", "--require"})


class CommandPolicyError(Exception):
    """Raised with a model-facing explanation when a proposed command is rejected."""


@dataclass(frozen=True)
class ApprovedCommand:
    """A command that passed validate_command(). Carries its own cwd so
    agent/tools/terminal.py's execute_command() doesn't need the caller to
    thread a ProjectRoot through the agent loop separately.
    """

    program: str
    args: List[str]
    timeout: int
    cwd: Path


def _contains_shell_metacharacters(token: str) -> bool:
    return any(bad in token for bad in _SHELL_METACHARACTERS)


def _check_no_injection(program: str, args: List[str]) -> None:
    for token in (program, *args):
        if _contains_shell_metacharacters(token):
            raise CommandPolicyError(
                f"Rejected: {token!r} contains shell syntax. Commands run directly as a program "
                "plus a list of arguments, never through a shell, so this can never be interpreted "
                "the way a shell pipeline would be -- provide separate, plain arguments instead."
            )
        if "nohup" in token.lower():
            raise CommandPolicyError("Rejected: background/detached execution is not allowed.")


def _check_path_like_args_stay_in_project(project: ProjectRoot, args: List[str]) -> None:
    """Any argument that isn't a flag is treated as a path-like argument and
    must resolve inside the project root. Bareword option values (like a
    -k test name) harmlessly resolve to a nonexistent path under the root
    and pass; the only thing this actually rejects is genuine traversal.
    """
    for token in args:
        if token.startswith("-"):
            continue
        try:
            project.resolve(token)
        except PathSecurityError:
            raise CommandPolicyError(
                f"Rejected: argument {token!r} resolves outside the project root."
            )


def _validate_generic(project: ProjectRoot, args: List[str]) -> None:
    """Shared policy for tools that are just flags + paths (pytest, mypy)."""
    _check_path_like_args_stay_in_project(project, args)


def _validate_ruff(project: ProjectRoot, args: List[str]) -> None:
    if args and not args[0].startswith("-") and args[0] not in {"check", "format"}:
        raise CommandPolicyError(
            f"Rejected: unsupported ruff subcommand {args[0]!r}. Allowed: check, format."
        )
    _check_path_like_args_stay_in_project(project, args)


def _validate_npm(project: ProjectRoot, args: List[str]) -> None:
    if not args:
        raise CommandPolicyError("Rejected: npm requires a subcommand, e.g. npm test.")

    subcommand = args[0]
    if subcommand == "test":
        if len(args) > 1:
            raise CommandPolicyError("Rejected: npm test takes no additional arguments.")
        return

    if subcommand == "run":
        if len(args) != 2:
            raise CommandPolicyError("Rejected: npm run requires exactly one script name.")
        script = args[1]
        if not _NPM_SCRIPT_RE.match(script):
            raise CommandPolicyError(f"Rejected: invalid npm script name {script!r}.")
        if script.lower() in _DANGEROUS_NPM_SCRIPTS:
            raise CommandPolicyError(
                f"Rejected: npm run {script!r} is not allowed. Installing, publishing, and "
                "changing npm configuration are not permitted."
            )
        return

    raise CommandPolicyError(
        f"Rejected: unsupported npm subcommand {subcommand!r}. Allowed: test, run <script>."
    )


def _validate_python(project: ProjectRoot, args: List[str]) -> None:
    if not args:
        raise CommandPolicyError(
            "Rejected: python requires an argument, e.g. -m pytest or a project script."
        )
    if any(a in _PYTHON_EVAL_FLAGS for a in args):
        raise CommandPolicyError("Rejected: inline code execution (-c) is not allowed.")

    if args[0] == "-m":
        if len(args) < 2:
            raise CommandPolicyError("Rejected: -m requires a module name.")
        module = args[1]
        if module not in {"pytest", "unittest"}:
            raise CommandPolicyError(
                f"Rejected: unsupported module {module!r} for python -m. Allowed: pytest, unittest."
            )
        _check_path_like_args_stay_in_project(project, args[2:])
        return

    script = args[0]
    if script.startswith("-"):
        raise CommandPolicyError(f"Rejected: unsupported python flag {script!r}.")
    if not script.endswith(".py"):
        raise CommandPolicyError(
            "Rejected: python must be invoked as `-m pytest`/`-m unittest`, or with an existing "
            "project .py script (e.g. manage.py)."
        )
    try:
        resolved = project.resolve(script)
    except PathSecurityError:
        raise CommandPolicyError(f"Rejected: {script!r} resolves outside the project root.")
    if not resolved.is_file():
        raise CommandPolicyError(f"Rejected: script {script!r} does not exist in the project.")

    _check_path_like_args_stay_in_project(project, args[1:])


def _validate_node(project: ProjectRoot, args: List[str]) -> None:
    if not args:
        raise CommandPolicyError("Rejected: node requires a project script argument.")
    if any(a in _NODE_EVAL_FLAGS for a in args):
        raise CommandPolicyError(
            "Rejected: inline code (-e/-p) and module-preloading (-r) flags are not allowed."
        )

    script = args[0]
    if script.startswith("-"):
        raise CommandPolicyError(f"Rejected: unsupported node flag {script!r}.")
    if not script.endswith((".js", ".mjs", ".cjs")):
        raise CommandPolicyError("Rejected: node must be invoked with an existing project script file.")
    try:
        resolved = project.resolve(script)
    except PathSecurityError:
        raise CommandPolicyError(f"Rejected: {script!r} resolves outside the project root.")
    if not resolved.is_file():
        raise CommandPolicyError(f"Rejected: script {script!r} does not exist in the project.")

    _check_path_like_args_stay_in_project(project, args[1:])


_VALIDATORS: Dict[str, Callable[[ProjectRoot, List[str]], None]] = {
    "pytest": _validate_generic,
    "mypy": _validate_generic,
    "ruff": _validate_ruff,
    "npm": _validate_npm,
    "python": _validate_python,
    "python3": _validate_python,
    "node": _validate_node,
}


def validate_command(project: ProjectRoot, program: str, args: List[str], timeout: int) -> ApprovedCommand:
    """Validate a proposed command. Raises CommandPolicyError with a
    model-facing explanation if anything about it is disallowed. Returns an
    ApprovedCommand ready to show the user and, if they approve, execute --
    never executes anything itself.
    """
    if not _PROGRAM_TOKEN_RE.match(program):
        raise CommandPolicyError(f"Rejected: {program!r} is not a valid program name.")

    if program in DENYLISTED_PROGRAMS or program not in _VALIDATORS:
        raise CommandPolicyError(
            f"Rejected: {program!r} is not allowed. This agent can only run a small allowlist of "
            f"local development/test commands: {', '.join(sorted(_VALIDATORS))}."
        )

    if not (MIN_TIMEOUT_SECONDS <= timeout <= MAX_TIMEOUT_SECONDS):
        raise CommandPolicyError(
            f"Rejected: timeout must be between {MIN_TIMEOUT_SECONDS} and {MAX_TIMEOUT_SECONDS} seconds."
        )

    _check_no_injection(program, args)
    _VALIDATORS[program](project, args)

    return ApprovedCommand(program=program, args=list(args), timeout=timeout, cwd=project.root)
