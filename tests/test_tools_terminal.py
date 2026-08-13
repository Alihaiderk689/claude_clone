"""Tests for the run_command tool: proposal (validation, no execution) and
execute_command (execution, with subprocess.Popen mocked so no real process
runs during most tests -- see TestRealSubprocessTermination at the bottom
for the handful of tests that deliberately run a real short-lived process
to prove cancellation/timeout actually kill it).
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
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


def _fake_proc(returncode=0, stdout="", stderr="", pid=99999):
    """A stand-in for subprocess.Popen: .communicate() returns immediately
    (no real process, no real waiting) unless a test overrides
    .communicate.side_effect to simulate a hang."""
    proc = mock.Mock()
    proc.pid = pid
    proc.returncode = returncode
    proc.communicate.return_value = (stdout, stderr)
    return proc


class TestProposal:
    def test_valid_command_is_proposed_without_executing(self, run_command_tool):
        with mock.patch("agent.tools.terminal.subprocess.Popen") as mock_popen:
            result = run_command_tool.execute({"program": "pytest", "args": ["-v"]})

        assert result.ok
        assert result.pending_command is not None
        assert result.pending_command.program == "pytest"
        assert result.pending_command.args == ["-v"]
        mock_popen.assert_not_called()

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
        fake = _fake_proc(returncode=0, stdout="2 passed\n", stderr="")
        with mock.patch("agent.tools.terminal.subprocess.Popen", return_value=fake) as mock_popen:
            result = execute_command(cmd)

        assert result.exit_code == 0
        assert result.stdout == "2 passed\n"
        assert result.stderr == ""
        assert not result.timed_out
        assert not result.cancelled

        called_args, called_kwargs = mock_popen.call_args
        assert called_args[0] == ["pytest"]
        assert called_kwargs["shell"] is False
        assert called_kwargs["cwd"] == str(tmp_path)
        assert called_kwargs["start_new_session"] is True

    def test_nonzero_exit_code_is_still_a_normal_result(self, tmp_path):
        cmd = self._cmd(tmp_path)
        fake = _fake_proc(returncode=1, stdout="1 failed, 1 passed\n", stderr="")
        with mock.patch("agent.tools.terminal.subprocess.Popen", return_value=fake):
            result = execute_command(cmd)

        assert result.exit_code == 1
        assert not result.timed_out
        assert "1 failed" in result.stdout

    def test_never_uses_shell_true(self, tmp_path):
        cmd = self._cmd(tmp_path, args=["-v"])
        fake = _fake_proc(returncode=0)
        with mock.patch("agent.tools.terminal.subprocess.Popen", return_value=fake) as mock_popen:
            execute_command(cmd)

        args_passed, kwargs_passed = mock_popen.call_args
        assert args_passed[0] == ["pytest", "-v"]  # argv list, never a joined string
        assert kwargs_passed["shell"] is False

    def test_environment_is_stripped_to_safe_vars(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SUPER_SECRET_TOKEN", "sk-should-not-leak")
        monkeypatch.setenv("PATH", "/usr/bin")
        cmd = self._cmd(tmp_path)
        fake = _fake_proc(returncode=0)
        with mock.patch("agent.tools.terminal.subprocess.Popen", return_value=fake) as mock_popen:
            execute_command(cmd)

        env_passed = mock_popen.call_args.kwargs["env"]
        assert "SUPER_SECRET_TOKEN" not in env_passed
        assert env_passed.get("PATH") == "/usr/bin"

    def test_runs_in_its_own_process_group(self, tmp_path):
        """start_new_session=True is what makes _kill_process_group able to
        terminate descendants (e.g. a test runner's own worker processes),
        not just the direct child."""
        cmd = self._cmd(tmp_path)
        fake = _fake_proc(returncode=0)
        with mock.patch("agent.tools.terminal.subprocess.Popen", return_value=fake) as mock_popen:
            execute_command(cmd)
        assert mock_popen.call_args.kwargs["start_new_session"] is True


class TestExecuteCommandTimeout:
    def _cmd(self, tmp_path, timeout=5):
        return ApprovedCommand(program="pytest", args=[], timeout=timeout, cwd=tmp_path)

    def _drive_timeout(self, tmp_path, final_output=("partial", "")):
        """Deterministically simulates a hung process without any real
        sleeping: time.monotonic() is scripted so the very first remaining-
        time check already reads as expired, and .communicate() only
        succeeds on the post-kill cleanup call."""
        cmd = self._cmd(tmp_path)
        fake = _fake_proc(returncode=None)
        fake.communicate.side_effect = [subprocess.TimeoutExpired(cmd=["pytest"], timeout=cmd.timeout)] * 0 + [
            final_output
        ]
        # First monotonic() call is `start`; second is the first remaining
        # check -- returning start+timeout already makes remaining <= 0.
        monotonic_values = iter([0.0, cmd.timeout])
        with mock.patch("agent.tools.terminal.subprocess.Popen", return_value=fake), mock.patch(
            "agent.tools.terminal.time.monotonic", side_effect=lambda: next(monotonic_values)
        ), mock.patch("agent.tools.terminal.os.getpgid", return_value=4242), mock.patch(
            "agent.tools.terminal.os.killpg"
        ) as mock_killpg, mock.patch.object(
            fake, "wait", return_value=None
        ):
            result = execute_command(cmd)
        return result, mock_killpg, fake

    def test_timeout_returns_structured_timed_out_result(self, tmp_path):
        result, mock_killpg, _ = self._drive_timeout(tmp_path, final_output=("partial", ""))
        assert result.timed_out
        assert not result.cancelled
        assert result.exit_code is None
        assert result.stdout == "partial"
        assert mock_killpg.called

    def test_timeout_process_does_not_hang_the_caller(self, tmp_path):
        """execute_command must return, not raise or block indefinitely,
        when the process doesn't exit in time."""
        result, _, _ = self._drive_timeout(tmp_path)
        assert result.timed_out

    def test_timeout_kills_the_process_group_not_just_the_child(self, tmp_path):
        """SIGTERM is sent to the whole process group (os.killpg), not just
        proc.kill() on the direct child -- this is what lets a timeout
        clean up a test runner's own worker subprocesses too."""
        import signal

        _, mock_killpg, _ = self._drive_timeout(tmp_path)
        mock_killpg.assert_any_call(4242, signal.SIGTERM)


class TestExecuteCommandCancellation:
    """Phase 8: Stop Task must actually terminate the running process group,
    not just stop waiting for it."""

    def _cmd(self, tmp_path, timeout=120):
        return ApprovedCommand(program="pytest", args=[], timeout=timeout, cwd=tmp_path)

    def test_cancel_event_set_before_start_terminates_immediately(self, tmp_path):
        cmd = self._cmd(tmp_path)
        fake = _fake_proc(returncode=None)
        fake.communicate.return_value = ("", "")
        cancel_event = threading.Event()
        cancel_event.set()

        with mock.patch("agent.tools.terminal.subprocess.Popen", return_value=fake), mock.patch(
            "agent.tools.terminal.os.getpgid", return_value=4242
        ), mock.patch("agent.tools.terminal.os.killpg") as mock_killpg, mock.patch.object(
            fake, "wait", return_value=None
        ):
            result = execute_command(cmd, cancel_event=cancel_event)

        assert result.cancelled
        assert not result.timed_out
        assert mock_killpg.called

    def test_cancelled_result_is_not_reported_as_timed_out(self, tmp_path):
        cmd = self._cmd(tmp_path)
        fake = _fake_proc(returncode=None)
        fake.communicate.return_value = ("partial output", "")
        cancel_event = threading.Event()
        cancel_event.set()

        with mock.patch("agent.tools.terminal.subprocess.Popen", return_value=fake), mock.patch(
            "agent.tools.terminal.os.getpgid", return_value=4242
        ), mock.patch("agent.tools.terminal.os.killpg"), mock.patch.object(fake, "wait", return_value=None):
            result = execute_command(cmd, cancel_event=cancel_event)

        text = format_result_for_model(result)
        assert "CANCELLED BY USER" in text
        assert "TIMED OUT" not in text

    def test_no_cancel_event_is_unaffected(self, tmp_path):
        """cancel_event is optional and additive -- omitting it must behave
        exactly as before this feature existed."""
        cmd = self._cmd(tmp_path)
        fake = _fake_proc(returncode=0, stdout="ok\n")
        with mock.patch("agent.tools.terminal.subprocess.Popen", return_value=fake):
            result = execute_command(cmd)
        assert not result.cancelled
        assert result.exit_code == 0


class TestExecuteCommandNotInstalled:
    def test_missing_executable_reports_clearly(self, tmp_path):
        cmd = ApprovedCommand(program="pytest", args=[], timeout=120, cwd=tmp_path)
        with mock.patch("agent.tools.terminal.subprocess.Popen", side_effect=FileNotFoundError()):
            result = execute_command(cmd)

        assert not result.timed_out
        assert result.exit_code is None
        assert "not installed" in result.stderr.lower()
        assert "install it manually" in result.stderr.lower()

    def test_os_error_reports_clearly(self, tmp_path):
        cmd = ApprovedCommand(program="pytest", args=[], timeout=120, cwd=tmp_path)
        with mock.patch(
            "agent.tools.terminal.subprocess.Popen", side_effect=OSError("no permission")
        ):
            result = execute_command(cmd)

        assert result.exit_code is None
        assert "no permission" in result.stderr.lower()


class TestOutputTruncation:
    def test_stdout_truncated_beyond_limit(self, tmp_path):
        cmd = ApprovedCommand(program="pytest", args=[], timeout=120, cwd=tmp_path)
        huge_stdout = "x" * (MAX_STDOUT_CHARS + 500)
        fake = _fake_proc(returncode=0, stdout=huge_stdout, stderr="")
        with mock.patch("agent.tools.terminal.subprocess.Popen", return_value=fake):
            result = execute_command(cmd)

        assert len(result.stdout) < len(huge_stdout)
        assert "truncated" in result.stdout.lower()
        assert "stdout truncated" in result.stdout.lower()

    def test_stderr_truncated_beyond_limit(self, tmp_path):
        cmd = ApprovedCommand(program="pytest", args=[], timeout=120, cwd=tmp_path)
        huge_stderr = "e" * (MAX_STDERR_CHARS + 500)
        fake = _fake_proc(returncode=1, stdout="", stderr=huge_stderr)
        with mock.patch("agent.tools.terminal.subprocess.Popen", return_value=fake):
            result = execute_command(cmd)

        assert len(result.stderr) < len(huge_stderr)
        assert "stderr truncated" in result.stderr.lower()

    def test_short_output_not_truncated(self, tmp_path):
        cmd = ApprovedCommand(program="pytest", args=[], timeout=120, cwd=tmp_path)
        fake = _fake_proc(returncode=0, stdout="short\n", stderr="")
        with mock.patch("agent.tools.terminal.subprocess.Popen", return_value=fake):
            result = execute_command(cmd)
        assert result.stdout == "short\n"
        assert "truncated" not in result.stdout


class TestFormatResultForModel:
    def test_includes_exit_code_and_streams(self, tmp_path):
        cmd = ApprovedCommand(program="pytest", args=["-v"], timeout=120, cwd=tmp_path)
        fake = _fake_proc(returncode=1, stdout="1 failed\n", stderr="warning\n")
        with mock.patch("agent.tools.terminal.subprocess.Popen", return_value=fake):
            result = execute_command(cmd)
        text = format_result_for_model(result)
        assert "exit code: 1" in text
        assert "1 failed" in text
        assert "warning" in text


class TestRealSubprocessTermination:
    """A handful of tests that run a genuine short-lived process (bypassing
    command_policy's allowlist entirely, since execute_command itself
    trusts whatever ApprovedCommand it's given -- policy enforcement
    already happened earlier in the pipeline) to prove timeout/cancellation
    actually kill the OS process, not just stop waiting for it."""

    def _pid_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:  # pragma: no cover - would mean it's alive but not ours
            return True
        return True

    def test_timeout_actually_kills_a_real_sleeping_process(self, tmp_path):
        cmd = ApprovedCommand(program="sleep", args=["30"], timeout=0.3, cwd=tmp_path)

        started = time.monotonic()
        result = execute_command(cmd)
        elapsed = time.monotonic() - started

        assert result.timed_out
        assert elapsed < 5  # nowhere near the real 30s sleep duration

    def test_cancel_event_actually_kills_a_real_sleeping_process(self, tmp_path):
        cmd = ApprovedCommand(program="sleep", args=["30"], timeout=120, cwd=tmp_path)
        cancel_event = threading.Event()

        result_holder = {}

        def run():
            result_holder["result"] = execute_command(cmd, cancel_event=cancel_event)

        thread = threading.Thread(target=run)
        thread.start()
        time.sleep(0.2)  # let the process actually start
        cancel_event.set()
        thread.join(timeout=10)

        assert not thread.is_alive()
        assert result_holder["result"].cancelled

    def test_killed_process_is_no_longer_running(self, tmp_path):
        cmd = ApprovedCommand(program="sleep", args=["30"], timeout=0.3, cwd=tmp_path)

        pid_holder = {}
        real_popen = subprocess.Popen

        def spying_popen(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            pid_holder["pid"] = proc.pid
            return proc

        with mock.patch("agent.tools.terminal.subprocess.Popen", side_effect=spying_popen):
            result = execute_command(cmd)

        assert result.timed_out
        assert "pid" in pid_holder
        time.sleep(0.2)  # give the OS a moment to reap it
        assert not self._pid_alive(pid_holder["pid"])
