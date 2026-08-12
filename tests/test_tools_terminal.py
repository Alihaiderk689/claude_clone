"""Tests for the run_command tool: proposal (validation, no execution) and
execute_command (execution, with subprocess mocked so no real process ever
runs during tests).
"""
from __future__ import annotations

import subprocess
from unittest import mock

import pytest

from agent.command_policy import ApprovedCommand
from agent.project import ProjectRoot
from agent.tools.terminal import (
    MAX_STDERR_CHARS,
    MAX_STDOUT_CHARS,
    build_run_command_tool,
    execute_command,
    format_result_for_model,
)


@pytest.fixture
def project(tmp_path):
    (tmp_path / "tests").mkdir()
    return ProjectRoot(tmp_path)


@pytest.fixture
def run_command_tool(project):
    return build_run_command_tool(project)


class TestProposal:
    def test_valid_command_is_proposed_without_executing(self, run_command_tool):
        with mock.patch("agent.tools.terminal.subprocess.run") as mock_run:
            result = run_command_tool.execute({"program": "pytest", "args": ["-v"]})

        assert result.ok
        assert result.pending_command is not None
        assert result.pending_command.program == "pytest"
        assert result.pending_command.args == ["-v"]
        mock_run.assert_not_called()

    def test_default_timeout_applied(self, run_command_tool):
        result = run_command_tool.execute({"program": "pytest", "args": []})
        assert result.pending_command.timeout == 120

    def test_rejected_command_has_no_pending_command(self, run_command_tool):
        result = run_command_tool.execute({"program": "rm", "args": ["-rf", "."]})
        assert not result.ok
        assert result.pending_command is None

    def test_shell_injection_rejected_at_proposal(self, run_command_tool):
        result = run_command_tool.execute({"program": "pytest", "args": ["&&", "rm", "-rf", "."]})
        assert not result.ok

    def test_malformed_arguments_rejected(self, run_command_tool):
        result = run_command_tool.execute({})  # missing required "program"
        assert not result.ok

    def test_timeout_out_of_range_rejected_by_schema(self, run_command_tool):
        result = run_command_tool.execute({"program": "pytest", "args": [], "timeout": 9999})
        assert not result.ok


class TestExecuteCommandSuccess:
    def _cmd(self, tmp_path, program="pytest", args=None, timeout=120):
        return ApprovedCommand(program=program, args=args or [], timeout=timeout, cwd=tmp_path)

    def test_successful_run_returns_structured_result(self, tmp_path):
        cmd = self._cmd(tmp_path)
        fake_proc = mock.Mock(returncode=0, stdout="2 passed\n", stderr="")
        with mock.patch("agent.tools.terminal.subprocess.run", return_value=fake_proc) as mock_run:
            result = execute_command(cmd)

        assert result.exit_code == 0
        assert result.stdout == "2 passed\n"
        assert result.stderr == ""
        assert not result.timed_out

        called_kwargs = mock_run.call_args.kwargs
        assert called_kwargs["shell"] is False
        assert called_kwargs["cwd"] == str(tmp_path)
        assert called_kwargs["timeout"] == 120

    def test_nonzero_exit_code_is_still_a_normal_result(self, tmp_path):
        cmd = self._cmd(tmp_path)
        fake_proc = mock.Mock(returncode=1, stdout="1 failed, 1 passed\n", stderr="")
        with mock.patch("agent.tools.terminal.subprocess.run", return_value=fake_proc):
            result = execute_command(cmd)

        assert result.exit_code == 1
        assert not result.timed_out
        assert "1 failed" in result.stdout

    def test_never_uses_shell_true(self, tmp_path):
        cmd = self._cmd(tmp_path, args=["-v"])
        fake_proc = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch("agent.tools.terminal.subprocess.run", return_value=fake_proc) as mock_run:
            execute_command(cmd)

        args_passed, kwargs_passed = mock_run.call_args
        assert args_passed[0] == ["pytest", "-v"]  # argv list, never a joined string
        assert kwargs_passed["shell"] is False

    def test_environment_is_stripped_to_safe_vars(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SUPER_SECRET_TOKEN", "sk-should-not-leak")
        monkeypatch.setenv("PATH", "/usr/bin")
        cmd = self._cmd(tmp_path)
        fake_proc = mock.Mock(returncode=0, stdout="", stderr="")
        with mock.patch("agent.tools.terminal.subprocess.run", return_value=fake_proc) as mock_run:
            execute_command(cmd)

        env_passed = mock_run.call_args.kwargs["env"]
        assert "SUPER_SECRET_TOKEN" not in env_passed
        assert env_passed.get("PATH") == "/usr/bin"


class TestExecuteCommandTimeout:
    def test_timeout_returns_structured_timed_out_result(self, tmp_path):
        cmd = ApprovedCommand(program="pytest", args=[], timeout=5, cwd=tmp_path)
        with mock.patch(
            "agent.tools.terminal.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["pytest"], timeout=5, output="partial", stderr=""),
        ):
            result = execute_command(cmd)

        assert result.timed_out
        assert result.exit_code is None
        assert result.stdout == "partial"

    def test_timeout_process_does_not_hang_the_caller(self, tmp_path):
        """subprocess.run itself is responsible for killing the process on
        timeout; we just verify we handle the exception rather than letting
        it propagate or blocking indefinitely."""
        cmd = ApprovedCommand(program="pytest", args=[], timeout=1, cwd=tmp_path)
        with mock.patch(
            "agent.tools.terminal.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["pytest"], timeout=1),
        ):
            result = execute_command(cmd)  # must return, not raise
        assert result.timed_out


class TestExecuteCommandNotInstalled:
    def test_missing_executable_reports_clearly(self, tmp_path):
        cmd = ApprovedCommand(program="pytest", args=[], timeout=120, cwd=tmp_path)
        with mock.patch("agent.tools.terminal.subprocess.run", side_effect=FileNotFoundError()):
            result = execute_command(cmd)

        assert not result.timed_out
        assert result.exit_code is None
        assert "not installed" in result.stderr.lower()
        assert "install it manually" in result.stderr.lower()


class TestOutputTruncation:
    def test_stdout_truncated_beyond_limit(self, tmp_path):
        cmd = ApprovedCommand(program="pytest", args=[], timeout=120, cwd=tmp_path)
        huge_stdout = "x" * (MAX_STDOUT_CHARS + 500)
        fake_proc = mock.Mock(returncode=0, stdout=huge_stdout, stderr="")
        with mock.patch("agent.tools.terminal.subprocess.run", return_value=fake_proc):
            result = execute_command(cmd)

        assert len(result.stdout) < len(huge_stdout)
        assert "truncated" in result.stdout.lower()
        assert "stdout truncated" in result.stdout.lower()

    def test_stderr_truncated_beyond_limit(self, tmp_path):
        cmd = ApprovedCommand(program="pytest", args=[], timeout=120, cwd=tmp_path)
        huge_stderr = "e" * (MAX_STDERR_CHARS + 500)
        fake_proc = mock.Mock(returncode=1, stdout="", stderr=huge_stderr)
        with mock.patch("agent.tools.terminal.subprocess.run", return_value=fake_proc):
            result = execute_command(cmd)

        assert len(result.stderr) < len(huge_stderr)
        assert "stderr truncated" in result.stderr.lower()

    def test_short_output_not_truncated(self, tmp_path):
        cmd = ApprovedCommand(program="pytest", args=[], timeout=120, cwd=tmp_path)
        fake_proc = mock.Mock(returncode=0, stdout="short\n", stderr="")
        with mock.patch("agent.tools.terminal.subprocess.run", return_value=fake_proc):
            result = execute_command(cmd)
        assert result.stdout == "short\n"
        assert "truncated" not in result.stdout


class TestFormatResultForModel:
    def test_includes_exit_code_and_streams(self, tmp_path):
        cmd = ApprovedCommand(program="pytest", args=["-v"], timeout=120, cwd=tmp_path)
        fake_proc = mock.Mock(returncode=1, stdout="1 failed\n", stderr="warning\n")
        with mock.patch("agent.tools.terminal.subprocess.run", return_value=fake_proc):
            result = execute_command(cmd)
        text = format_result_for_model(result)
        assert "exit code: 1" in text
        assert "1 failed" in text
        assert "warning" in text

    def test_timed_out_is_clearly_labeled(self, tmp_path):
        cmd = ApprovedCommand(program="pytest", args=[], timeout=1, cwd=tmp_path)
        with mock.patch(
            "agent.tools.terminal.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["pytest"], timeout=1),
        ):
            result = execute_command(cmd)
        text = format_result_for_model(result)
        assert "TIMED OUT" in text
