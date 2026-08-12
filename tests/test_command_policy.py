"""Tests for the command validation layer in agent/command_policy.py.

This module only validates -- nothing here ever spawns a process. See
tests/test_tools_terminal.py for execution-layer tests (with subprocess
mocked) and tests/test_loop.py for the full propose/confirm/apply flow.
"""
from __future__ import annotations

import pytest

from agent.command_policy import (
    DEFAULT_TIMEOUT_SECONDS,
    MAX_TIMEOUT_SECONDS,
    MIN_TIMEOUT_SECONDS,
    CommandPolicyError,
    validate_command,
)
from agent.project import ProjectRoot


@pytest.fixture
def sample_project(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    (tmp_path / "manage.py").write_text("# django manage.py\n")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "check.js").write_text("console.log('ok');\n")
    return tmp_path


@pytest.fixture
def project(sample_project):
    return ProjectRoot(sample_project)


def validate(project, program, args, timeout=DEFAULT_TIMEOUT_SECONDS):
    return validate_command(project, program, args, timeout)


class TestAllowedCommands:
    def test_bare_pytest(self, project):
        approved = validate(project, "pytest", [])
        assert approved.program == "pytest"
        assert approved.cwd == project.root

    def test_pytest_with_flags_and_path(self, project):
        approved = validate(project, "pytest", ["-v", "tests"])
        assert approved.args == ["-v", "tests"]

    def test_python_module_pytest(self, project):
        approved = validate(project, "python", ["-m", "pytest"])
        assert approved.args == ["-m", "pytest"]

    def test_python3_module_pytest(self, project):
        validate(project, "python3", ["-m", "pytest", "-v"])

    def test_ruff_check_dot(self, project):
        approved = validate(project, "ruff", ["check", "."])
        assert approved.args == ["check", "."]

    def test_ruff_format(self, project):
        validate(project, "ruff", ["format", "."])

    def test_mypy_dot(self, project):
        validate(project, "mypy", ["."])

    def test_npm_test(self, project):
        approved = validate(project, "npm", ["test"])
        assert approved.args == ["test"]

    def test_npm_run_test(self, project):
        validate(project, "npm", ["run", "test"])

    def test_npm_run_lint(self, project):
        validate(project, "npm", ["run", "lint"])

    def test_npm_run_build(self, project):
        validate(project, "npm", ["run", "build"])

    def test_npm_run_typecheck(self, project):
        validate(project, "npm", ["run", "typecheck"])

    def test_python_manage_py_test(self, project):
        approved = validate(project, "python", ["manage.py", "test"])
        assert approved.args == ["manage.py", "test"]

    def test_node_project_script(self, project):
        approved = validate(project, "node", ["scripts/check.js"])
        assert approved.args == ["scripts/check.js"]

    def test_custom_timeout_within_bounds(self, project):
        approved = validate(project, "pytest", [], timeout=60)
        assert approved.timeout == 60

    def test_min_and_max_timeout_bounds_accepted(self, project):
        validate(project, "pytest", [], timeout=MIN_TIMEOUT_SECONDS)
        validate(project, "pytest", [], timeout=MAX_TIMEOUT_SECONDS)


class TestRejectedDangerousExecutables:
    @pytest.mark.parametrize(
        "program",
        [
            "rm", "sudo", "shutdown", "reboot", "mkfs", "dd", "chmod",
            "chown", "curl", "wget", "ssh", "scp", "nc", "kill", "pkill",
            "diskutil", "launchctl", "bash", "sh", "git",
        ],
    )
    def test_rejected(self, project, program):
        with pytest.raises(CommandPolicyError):
            validate(project, program, [])

    def test_unknown_program_rejected(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "ls", [])

    def test_env_dumping_rejected(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "env", [])

    def test_printenv_rejected(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "printenv", [])


class TestShellInjectionRejected:
    def test_semicolon_chain(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "pytest", [";", "rm", "-rf", "."])

    def test_semicolon_embedded_in_token(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "pytest", ["tests; rm -rf ."])

    def test_and_and_chain(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "pytest", ["&&", "malicious_command"])

    def test_pipe_chain(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "pytest", ["|", "malicious_command"])

    def test_or_or_chain(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "pytest", ["||", "malicious_command"])

    def test_command_substitution_dollar(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "pytest", ["$(malicious_command)"])

    def test_command_substitution_backtick(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "pytest", ["`malicious_command`"])

    def test_redirect_out(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "pytest", [">", "/tmp/x"])

    def test_redirect_in(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "pytest", ["<", "/etc/passwd"])

    def test_background_ampersand(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "pytest", ["&"])

    def test_nohup_rejected_as_program(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "nohup", ["pytest"])

    def test_nohup_rejected_as_arg(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "pytest", ["nohup"])

    def test_injection_in_program_field_itself(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "pytest; rm -rf .", [])


class TestProjectRootRestriction:
    def test_pytest_path_argument_outside_project_rejected(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "pytest", ["../../etc"])

    def test_pytest_absolute_path_outside_project_rejected(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "pytest", ["/etc/passwd"])

    def test_bareword_option_values_are_not_falsely_rejected(self, project):
        # -k <testname> is a value, not a path -- must not be misflagged.
        validate(project, "pytest", ["-k", "test_login"])

    def test_python_script_outside_project_rejected(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "python", ["../evil.py"])

    def test_node_script_outside_project_rejected(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "node", ["../evil.js"])


class TestTimeoutValidation:
    def test_zero_timeout_rejected(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "pytest", [], timeout=0)

    def test_negative_timeout_rejected(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "pytest", [], timeout=-10)

    def test_over_max_timeout_rejected(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "pytest", [], timeout=MAX_TIMEOUT_SECONDS + 1)

    def test_absurdly_large_timeout_rejected(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "pytest", [], timeout=10_000_000)


class TestDependencyInstallationBlocked:
    def test_npm_install_rejected(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "npm", ["install"])

    def test_npm_i_rejected(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "npm", ["i"])

    def test_npm_ci_rejected(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "npm", ["ci"])

    def test_npm_run_install_script_rejected(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "npm", ["run", "install"])

    def test_pip_rejected_outright(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "pip", ["install", "requests"])


class TestArbitraryCodeExecutionBlocked:
    def test_python_dash_c_rejected(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "python", ["-c", "import os; os.system('rm -rf .')"])

    def test_python3_dash_c_rejected(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "python3", ["-c", "print(1)"])

    def test_node_eval_rejected(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "node", ["-e", "require('child_process').exec('rm -rf .')"])

    def test_node_require_flag_rejected(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "node", ["scripts/check.js", "-r", "/etc/passwd"])

    def test_python_arbitrary_non_project_flag_rejected(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "python", ["--version"])

    def test_python_script_must_exist(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "python", ["does_not_exist.py"])

    def test_ruff_unsupported_subcommand_rejected(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "ruff", ["clean"])


class TestMalformedProgram:
    def test_program_with_spaces_rejected(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "py test", [])

    def test_empty_program_rejected(self, project):
        with pytest.raises(CommandPolicyError):
            validate(project, "", [])
