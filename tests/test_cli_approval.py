"""Tests for cli.py's approval-prompt parsing: _ask_approval/_handle_confirm
(file changes), _ask_command_approval/_handle_command_confirm (run_command),
and _ask_git_approval/_handle_git_confirm (Git operations). These are pure
enough to test directly against a mocked Rich Console, without needing a
real terminal or model.
"""
from __future__ import annotations

from pathlib import Path
from unittest import mock

from agent.cli import (
    _ask_approval,
    _ask_command_approval,
    _ask_git_approval,
    _handle_command_confirm,
    _handle_confirm,
    _handle_git_confirm,
)
from agent.command_policy import ApprovedCommand
from agent.diff import ProposedChange
from agent.git_policy import ProposedGitOperation


def make_console(input_value: str):
    console = mock.Mock()
    console.input.return_value = input_value
    console.width = 80
    return console


def _all_console_output(console) -> str:
    """Everything shown to the user: console.print() calls plus the prompt
    text passed to console.input() (the [y/N]-style question itself goes
    through input(), not print())."""
    printed = " ".join(str(call.args[0]) for call in console.print.call_args_list if call.args)
    inputs = " ".join(str(call.args[0]) for call in console.input.call_args_list if call.args)
    return f"{printed} {inputs}"


class TestAskApproval:
    def test_y_means_yes(self):
        assert _ask_approval(make_console("y")) == "yes"

    def test_yes_means_yes(self):
        assert _ask_approval(make_console("yes")) == "yes"

    def test_uppercase_y_means_yes(self):
        assert _ask_approval(make_console("Y")) == "yes"

    def test_n_means_no(self):
        assert _ask_approval(make_console("n")) == "no"

    def test_no_means_no(self):
        assert _ask_approval(make_console("no")) == "no"

    def test_empty_input_defaults_to_no(self):
        assert _ask_approval(make_console("")) == "no"

    def test_whitespace_only_defaults_to_no(self):
        assert _ask_approval(make_console("   ")) == "no"

    def test_garbage_input_defaults_to_no(self):
        assert _ask_approval(make_console("sure whatever")) == "no"

    def test_a_means_all(self):
        assert _ask_approval(make_console("a")) == "all"


class TestHandleConfirm:
    def _change(self):
        return ProposedChange(
            path="x.py",
            resolved_path=Path("/tmp/x.py"),
            kind="edit",
            old_content="old\n",
            new_content="new\n",
        )

    def test_approve_all_flag_skips_prompt_and_approves(self):
        console = make_console("this should never be read")
        turn_state = {"approve_all": True}

        approved = _handle_confirm(console, self._change(), turn_state)

        assert approved is True
        console.input.assert_not_called()

    def test_yes_approves_without_setting_approve_all(self):
        console = make_console("y")
        turn_state = {"approve_all": False}

        approved = _handle_confirm(console, self._change(), turn_state)

        assert approved is True
        assert turn_state["approve_all"] is False

    def test_no_rejects(self):
        console = make_console("n")
        turn_state = {"approve_all": False}

        approved = _handle_confirm(console, self._change(), turn_state)

        assert approved is False

    def test_all_approves_and_sets_approve_all_for_rest_of_turn(self):
        console = make_console("a")
        turn_state = {"approve_all": False}

        approved = _handle_confirm(console, self._change(), turn_state)

        assert approved is True
        assert turn_state["approve_all"] is True

    def test_default_reject_does_not_leak_across_turns(self):
        """approve_all must be a fresh dict per turn in cli.py's render_turn
        -- this just documents/locks the contract that _handle_confirm reads
        and writes whatever dict it's given, nothing global."""
        turn_state_a = {"approve_all": True}
        turn_state_b = {"approve_all": False}

        assert _handle_confirm(make_console("n"), self._change(), turn_state_a) is True
        assert _handle_confirm(make_console("n"), self._change(), turn_state_b) is False


class TestAskCommandApproval:
    def test_y_approves(self):
        assert _ask_command_approval(make_console("y")) is True

    def test_yes_approves(self):
        assert _ask_command_approval(make_console("yes")) is True

    def test_uppercase_y_approves(self):
        assert _ask_command_approval(make_console("Y")) is True

    def test_n_rejects(self):
        assert _ask_command_approval(make_console("n")) is False

    def test_empty_input_defaults_to_reject(self):
        assert _ask_command_approval(make_console("")) is False

    def test_garbage_defaults_to_reject(self):
        assert _ask_command_approval(make_console("sure")) is False

    def test_a_is_not_special_for_commands(self):
        """Unlike file-edit approval, run_command has no 'approve all'
        shortcut -- every command is confirmed individually (per spec)."""
        assert _ask_command_approval(make_console("a")) is False


class TestHandleCommandConfirm:
    def _cmd(self, tmp_path):
        return ApprovedCommand(program="pytest", args=["-v"], timeout=120, cwd=tmp_path)

    def test_approved(self, tmp_path):
        console = make_console("y")
        assert _handle_command_confirm(console, self._cmd(tmp_path)) is True

    def test_rejected(self, tmp_path):
        console = make_console("n")
        assert _handle_command_confirm(console, self._cmd(tmp_path)) is False

    def test_bare_enter_rejects(self, tmp_path):
        console = make_console("")
        assert _handle_command_confirm(console, self._cmd(tmp_path)) is False

    def test_prompt_shows_program_and_cwd(self, tmp_path):
        console = make_console("n")
        _handle_command_confirm(console, self._cmd(tmp_path))
        printed = " ".join(str(call.args[0]) for call in console.print.call_args_list if call.args)
        assert "pytest" in printed
        assert str(tmp_path) in printed


class TestAskGitApproval:
    def test_y_approves(self):
        assert _ask_git_approval(make_console("y"), "Create this branch? [y/N]:") is True

    def test_yes_approves(self):
        assert _ask_git_approval(make_console("yes"), "Create this branch? [y/N]:") is True

    def test_n_rejects(self):
        assert _ask_git_approval(make_console("n"), "Create this branch? [y/N]:") is False

    def test_empty_input_defaults_to_reject(self):
        assert _ask_git_approval(make_console(""), "Create this branch? [y/N]:") is False

    def test_garbage_defaults_to_reject(self):
        assert _ask_git_approval(make_console("sure"), "Create this branch? [y/N]:") is False


class TestHandleGitConfirm:
    def _branch_op(self):
        return ProposedGitOperation(kind="create_branch", repo_root=Path("/tmp/repo"), branch_name="feature/x")

    def _stage_op(self, sensitive=False):
        return ProposedGitOperation(
            kind="stage",
            repo_root=Path("/tmp/repo"),
            paths=[".env"] if sensitive else ["README.md"],
            sensitive_paths=[".env"] if sensitive else [],
        )

    def _commit_op(self):
        return ProposedGitOperation(
            kind="commit",
            repo_root=Path("/tmp/repo"),
            message="Fix bug",
            expected_staged_files=["README.md"],
        )

    def test_create_branch_approved(self):
        console = make_console("y")
        assert _handle_git_confirm(console, self._branch_op()) is True

    def test_create_branch_rejected_on_bare_enter(self):
        console = make_console("")
        assert _handle_git_confirm(console, self._branch_op()) is False

    def test_create_branch_prompt_shows_name(self):
        console = make_console("n")
        _handle_git_confirm(console, self._branch_op())
        printed = _all_console_output(console)
        assert "feature/x" in printed
        assert "Create this branch?" in printed

    def test_stage_approved(self):
        console = make_console("y")
        assert _handle_git_confirm(console, self._stage_op()) is True

    def test_stage_prompt_shows_paths(self):
        console = make_console("n")
        _handle_git_confirm(console, self._stage_op())
        printed = _all_console_output(console)
        assert "README.md" in printed
        assert "Stage these files?" in printed

    def test_stage_sensitive_file_shows_warning_prompt(self):
        console = make_console("n")
        _handle_git_confirm(console, self._stage_op(sensitive=True))
        printed = _all_console_output(console)
        assert "WARNING" in printed
        assert ".env" in printed
        assert "Are you sure you want to stage it?" in printed

    def test_stage_sensitive_file_defaults_to_reject(self):
        console = make_console("")
        assert _handle_git_confirm(console, self._stage_op(sensitive=True)) is False

    def test_stage_sensitive_file_requires_explicit_yes(self):
        console = make_console("y")
        assert _handle_git_confirm(console, self._stage_op(sensitive=True)) is True

    def test_commit_approved(self):
        console = make_console("y")
        assert _handle_git_confirm(console, self._commit_op()) is True

    def test_commit_prompt_shows_message_and_staged_files(self):
        console = make_console("n")
        _handle_git_confirm(console, self._commit_op())
        printed = _all_console_output(console)
        assert "Fix bug" in printed
        assert "README.md" in printed
        assert "Create this commit?" in printed

    def test_commit_rejected_on_bare_enter(self):
        console = make_console("")
        assert _handle_git_confirm(console, self._commit_op()) is False
