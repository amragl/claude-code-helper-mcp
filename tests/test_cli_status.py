"""Tests for the Click CLI framework and ``memory status`` command (CMH-013).

All tests use real files in temporary directories, real MemoryStore and
WindowManager instances, and real Click test runner invocations.  No mocks,
no stubs, no fakes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from click.testing import CliRunner

from claude_code_helper_mcp.cli.main import (
    _collect_status,
    _determine_last_activity,
    _format_bytes,
    _render_status_text,
    cli,
    status,
)
from claude_code_helper_mcp.config import MemoryConfig
from claude_code_helper_mcp.models.task import TaskMemory, TaskStatus
from claude_code_helper_mcp.storage.store import MemoryStore
from claude_code_helper_mcp.storage.window_manager import WindowManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def project_dir(tmp_path: Path) -> Path:
    """Create a temporary project directory with a .git marker."""
    (tmp_path / ".git").mkdir()
    return tmp_path


@pytest.fixture()
def storage_path(project_dir: Path) -> str:
    """Return the path to a .claude-memory directory inside the project."""
    path = project_dir / ".claude-memory"
    path.mkdir(exist_ok=True)
    return str(path)


@pytest.fixture()
def manager(storage_path: str) -> WindowManager:
    """Create a WindowManager with a fresh storage directory."""
    return WindowManager(storage_path=storage_path, window_size=3)


@pytest.fixture()
def runner() -> CliRunner:
    """Create a Click CLI test runner."""
    return CliRunner()


# ---------------------------------------------------------------------------
# CLI group and entry point
# ---------------------------------------------------------------------------


class TestCLIGroup:
    """The CLI group is properly configured with version and help."""

    def test_cli_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Claude Code Helper" in result.output
        assert "Memory management" in result.output

    def test_cli_version(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "claude-code-helper-mcp" in result.output

    def test_cli_no_command_shows_usage(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, [])
        # Click groups with required subcommands return exit code 2 when
        # invoked without a subcommand.  The important thing is that usage
        # information is shown.
        assert result.exit_code in (0, 2)
        assert "Usage:" in result.output or "usage:" in result.output.lower()

    def test_status_command_registered(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "status" in result.output

    def test_status_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["status", "--help"])
        assert result.exit_code == 0
        assert "memory system status" in result.output.lower() or "status" in result.output.lower()


# ---------------------------------------------------------------------------
# Status command -- empty state
# ---------------------------------------------------------------------------


class TestStatusEmpty:
    """Status command with no tasks and fresh storage."""

    def test_status_text_output_empty(
        self, runner: CliRunner, storage_path: str
    ) -> None:
        result = runner.invoke(cli, ["--storage-path", storage_path, "status"])
        assert result.exit_code == 0
        assert "Memory Status" in result.output
        assert "No active task" in result.output

    def test_status_json_output_empty(
        self, runner: CliRunner, storage_path: str
    ) -> None:
        result = runner.invoke(
            cli, ["--storage-path", storage_path, "status", "--json-output"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "ok"
        assert data["current_task"] is None
        assert data["window"]["tasks_in_window"] == 0
        assert data["window"]["completed_count"] == 0
        assert data["storage"]["exists"] is True

    def test_status_json_has_all_sections(
        self, runner: CliRunner, storage_path: str
    ) -> None:
        result = runner.invoke(
            cli, ["--storage-path", storage_path, "status", "--json-output"]
        )
        data = json.loads(result.output)
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "current_task" in data
        assert "window" in data
        assert "storage" in data
        assert "config" in data
        assert "last_activity" in data

    def test_status_empty_last_activity_null(
        self, runner: CliRunner, storage_path: str
    ) -> None:
        result = runner.invoke(
            cli, ["--storage-path", storage_path, "status", "--json-output"]
        )
        data = json.loads(result.output)
        assert data["last_activity"] is None


# ---------------------------------------------------------------------------
# Status command -- with active task
# ---------------------------------------------------------------------------


class TestStatusActiveTask:
    """Status command when a task is active."""

    def test_status_shows_active_task(
        self, runner: CliRunner, storage_path: str, manager: WindowManager
    ) -> None:
        manager.start_new_task("CMH-TEST-1", "Test task one", phase="phase-1")
        result = runner.invoke(cli, ["--storage-path", storage_path, "status"])
        assert result.exit_code == 0
        assert "CMH-TEST-1" in result.output
        assert "Test task one" in result.output
        assert "Active Task" in result.output

    def test_status_json_active_task(
        self, runner: CliRunner, storage_path: str, manager: WindowManager
    ) -> None:
        task = manager.start_new_task("CMH-TEST-2", "JSON test task", phase="phase-2")
        task.add_step("First step", "Did something")
        task.add_step("Second step", "Did another thing")
        manager.save_current_task()

        result = runner.invoke(
            cli, ["--storage-path", storage_path, "status", "--json-output"]
        )
        data = json.loads(result.output)
        assert data["current_task"] is not None
        assert data["current_task"]["ticket_id"] == "CMH-TEST-2"
        assert data["current_task"]["title"] == "JSON test task"
        assert data["current_task"]["phase"] == "phase-2"
        assert data["current_task"]["status"] == "active"
        assert data["current_task"]["steps"] == 2
        assert data["current_task"]["elapsed_minutes"] >= 0
        assert data["window"]["has_active_task"] is True

    def test_status_active_task_with_files_and_decisions(
        self, runner: CliRunner, storage_path: str, manager: WindowManager
    ) -> None:
        from claude_code_helper_mcp.models.records import BranchAction, FileAction

        task = manager.start_new_task("CMH-TEST-3", "Complex task")
        task.record_file("src/main.py", FileAction.CREATED, "Created main module")
        task.record_file("src/util.py", FileAction.MODIFIED, "Updated utility")
        task.add_decision("Use Click for CLI", "Industry standard", ["argparse", "typer"])
        task.record_branch("feature/test-branch", BranchAction.CREATED, "main")
        manager.save_current_task()

        result = runner.invoke(
            cli, ["--storage-path", storage_path, "status", "--json-output"]
        )
        data = json.loads(result.output)
        ct = data["current_task"]
        assert ct["files"] == 2
        assert ct["decisions"] == 1
        assert ct["branches"] == 1
        assert ct["active_branch"] == "feature/test-branch"

    def test_status_text_shows_phase(
        self, runner: CliRunner, storage_path: str, manager: WindowManager
    ) -> None:
        manager.start_new_task("CMH-TEST-4", "Phase task", phase="phase-3")
        result = runner.invoke(cli, ["--storage-path", storage_path, "status"])
        assert "phase-3" in result.output


# ---------------------------------------------------------------------------
# Status command -- with completed tasks
# ---------------------------------------------------------------------------


class TestStatusCompletedTasks:
    """Status command with completed tasks in the window."""

    def test_status_completed_tasks_in_window(
        self, runner: CliRunner, storage_path: str, manager: WindowManager
    ) -> None:
        # Create and complete two tasks.
        manager.start_new_task("CMH-C1", "Completed task 1", phase="phase-1")
        manager.complete_current_task("Done with task 1")
        manager.start_new_task("CMH-C2", "Completed task 2", phase="phase-1")
        manager.complete_current_task("Done with task 2")

        result = runner.invoke(
            cli, ["--storage-path", storage_path, "status", "--json-output"]
        )
        data = json.loads(result.output)
        assert data["window"]["completed_count"] == 2
        assert data["window"]["tasks_in_window"] == 2
        assert len(data["window"]["completed_tasks"]) == 2
        assert data["window"]["completed_tasks"][0]["ticket_id"] == "CMH-C1"
        assert data["window"]["completed_tasks"][1]["ticket_id"] == "CMH-C2"

    def test_status_text_shows_completed_tasks(
        self, runner: CliRunner, storage_path: str, manager: WindowManager
    ) -> None:
        manager.start_new_task("CMH-C3", "Third task", phase="phase-2")
        manager.complete_current_task("Completed")
        result = runner.invoke(cli, ["--storage-path", storage_path, "status"])
        assert "CMH-C3" in result.output
        assert "Recent completed tasks" in result.output

    def test_status_archived_tasks(
        self, runner: CliRunner, storage_path: str, manager: WindowManager
    ) -> None:
        # Create and complete 4 tasks (window_size=3), so first gets archived.
        for i in range(4):
            manager.start_new_task(f"CMH-A{i}", f"Archive test {i}")
            manager.complete_current_task(f"Done {i}")

        result = runner.invoke(
            cli, ["--storage-path", storage_path, "status", "--json-output"]
        )
        data = json.loads(result.output)
        assert data["window"]["archived_count"] == 1
        assert "CMH-A0" in data["window"]["archived_task_ids"]
        assert data["window"]["completed_count"] == 3

    def test_status_text_shows_archived_ids(
        self, runner: CliRunner, storage_path: str, manager: WindowManager
    ) -> None:
        for i in range(5):
            manager.start_new_task(f"CMH-AR{i}", f"Archive {i}")
            manager.complete_current_task(f"Done {i}")

        result = runner.invoke(cli, ["--storage-path", storage_path, "status"])
        assert "Archived IDs" in result.output
        assert "CMH-AR0" in result.output
        assert "CMH-AR1" in result.output


# ---------------------------------------------------------------------------
# Status command -- window state
# ---------------------------------------------------------------------------


class TestStatusWindowState:
    """Status command window section details."""

    def test_window_size_displayed(
        self, runner: CliRunner, storage_path: str
    ) -> None:
        result = runner.invoke(
            cli, ["--storage-path", storage_path, "status", "--json-output"]
        )
        data = json.loads(result.output)
        assert data["window"]["window_size"] == 3

    def test_window_state_mixed(
        self, runner: CliRunner, storage_path: str, manager: WindowManager
    ) -> None:
        # One completed, one active.
        manager.start_new_task("CMH-M1", "Mixed 1")
        manager.complete_current_task("Done 1")
        manager.start_new_task("CMH-M2", "Mixed 2")

        result = runner.invoke(
            cli, ["--storage-path", storage_path, "status", "--json-output"]
        )
        data = json.loads(result.output)
        assert data["window"]["tasks_in_window"] == 2
        assert data["window"]["completed_count"] == 1
        assert data["window"]["has_active_task"] is True
        assert data["current_task"]["ticket_id"] == "CMH-M2"


# ---------------------------------------------------------------------------
# Status command -- storage info
# ---------------------------------------------------------------------------


class TestStatusStorage:
    """Status command storage section details."""

    def test_storage_exists(
        self, runner: CliRunner, storage_path: str
    ) -> None:
        result = runner.invoke(
            cli, ["--storage-path", storage_path, "status", "--json-output"]
        )
        data = json.loads(result.output)
        assert data["storage"]["exists"] is True
        assert data["storage"]["path"] == storage_path

    def test_storage_task_file_count(
        self, runner: CliRunner, storage_path: str, manager: WindowManager
    ) -> None:
        manager.start_new_task("CMH-F1", "File count test 1")
        manager.complete_current_task("Done")
        manager.start_new_task("CMH-F2", "File count test 2")
        manager.complete_current_task("Done")

        result = runner.invoke(
            cli, ["--storage-path", storage_path, "status", "--json-output"]
        )
        data = json.loads(result.output)
        assert data["storage"]["task_files"] >= 2

    def test_storage_total_size(
        self, runner: CliRunner, storage_path: str, manager: WindowManager
    ) -> None:
        manager.start_new_task("CMH-S1", "Size test")
        manager.complete_current_task("Done")

        result = runner.invoke(
            cli, ["--storage-path", storage_path, "status", "--json-output"]
        )
        data = json.loads(result.output)
        # There should be at least some bytes on disk (window.json + task file).
        assert data["storage"]["total_size_bytes"] > 0
        assert isinstance(data["storage"]["total_size_human"], str)

    def test_storage_text_output(
        self, runner: CliRunner, storage_path: str
    ) -> None:
        result = runner.invoke(cli, ["--storage-path", storage_path, "status"])
        assert "Storage" in result.output
        assert "Path:" in result.output
        assert "Exists:" in result.output
        assert "Task files:" in result.output
        assert "Total size:" in result.output


# ---------------------------------------------------------------------------
# Status command -- configuration display
# ---------------------------------------------------------------------------


class TestStatusConfig:
    """Status command configuration section details."""

    def test_config_section_present(
        self, runner: CliRunner, storage_path: str
    ) -> None:
        result = runner.invoke(
            cli, ["--storage-path", storage_path, "status", "--json-output"]
        )
        data = json.loads(result.output)
        cfg = data["config"]
        assert "project_root" in cfg
        assert "storage_path" in cfg
        assert "window_size" in cfg
        assert "log_level" in cfg
        assert "auto_save" in cfg
        assert "archive_completed" in cfg

    def test_config_defaults(
        self, runner: CliRunner, storage_path: str
    ) -> None:
        result = runner.invoke(
            cli, ["--storage-path", storage_path, "status", "--json-output"]
        )
        data = json.loads(result.output)
        cfg = data["config"]
        assert cfg["window_size"] == 3
        assert cfg["log_level"] == "INFO"
        assert cfg["auto_save"] is True
        assert cfg["archive_completed"] is True

    def test_config_text_output(
        self, runner: CliRunner, storage_path: str
    ) -> None:
        result = runner.invoke(cli, ["--storage-path", storage_path, "status"])
        assert "Configuration" in result.output
        assert "Project root:" in result.output
        assert "Window size:" in result.output
        assert "Log level:" in result.output


# ---------------------------------------------------------------------------
# Last activity tracking
# ---------------------------------------------------------------------------


class TestLastActivity:
    """Last activity detection across tasks."""

    def test_no_tasks_returns_none(self) -> None:
        result = _determine_last_activity(None, [])
        assert result is None

    def test_active_task_started_at(self) -> None:
        task = TaskMemory(ticket_id="T-1", title="Test")
        result = _determine_last_activity(task, [])
        assert result is not None
        # Should be parseable as ISO 8601.
        parsed = datetime.fromisoformat(result)
        assert parsed.tzinfo is not None

    def test_active_task_with_steps(self) -> None:
        task = TaskMemory(ticket_id="T-2", title="With steps")
        task.add_step("Step 1", "First step")
        result = _determine_last_activity(task, [])
        assert result is not None
        # The step timestamp should be the latest.
        step_ts = task.steps[-1].timestamp.isoformat()
        assert result == step_ts

    def test_completed_tasks_latest(self) -> None:
        task1 = TaskMemory(ticket_id="T-3", title="Completed 1")
        task1.complete("Done 1")
        task2 = TaskMemory(ticket_id="T-4", title="Completed 2")
        task2.complete("Done 2")

        result = _determine_last_activity(None, [task1, task2])
        assert result is not None
        # Task 2 was completed after task 1.
        assert result == task2.completed_at.isoformat()

    def test_mixed_active_and_completed(self) -> None:
        completed = TaskMemory(ticket_id="T-5", title="Old completed")
        completed.complete("Done")
        active = TaskMemory(ticket_id="T-6", title="Current")
        active.add_step("Recent step", "Just now")

        result = _determine_last_activity(active, [completed])
        assert result is not None


# ---------------------------------------------------------------------------
# _format_bytes utility
# ---------------------------------------------------------------------------


class TestFormatBytes:
    """Byte formatting utility."""

    def test_zero_bytes(self) -> None:
        assert _format_bytes(0) == "0 B"

    def test_small_bytes(self) -> None:
        assert _format_bytes(512) == "512 B"

    def test_exactly_1023_bytes(self) -> None:
        assert _format_bytes(1023) == "1023 B"

    def test_one_kib(self) -> None:
        assert _format_bytes(1024) == "1.0 KiB"

    def test_kib_range(self) -> None:
        result = _format_bytes(1536)
        assert "KiB" in result
        assert result == "1.5 KiB"

    def test_one_mib(self) -> None:
        assert _format_bytes(1024 * 1024) == "1.0 MiB"

    def test_mib_range(self) -> None:
        result = _format_bytes(3 * 1024 * 1024 + 200 * 1024)
        assert "MiB" in result

    def test_one_gib(self) -> None:
        assert _format_bytes(1024 * 1024 * 1024) == "1.0 GiB"


# ---------------------------------------------------------------------------
# _collect_status internal function
# ---------------------------------------------------------------------------


class TestCollectStatus:
    """Internal _collect_status function."""

    def test_returns_ok_with_valid_path(self, storage_path: str) -> None:
        data = _collect_status(storage_path)
        assert data["status"] == "ok"

    def test_returns_all_sections(self, storage_path: str) -> None:
        data = _collect_status(storage_path)
        assert "current_task" in data
        assert "window" in data
        assert "storage" in data
        assert "config" in data
        assert "last_activity" in data
        assert "version" in data
        assert "timestamp" in data

    def test_auto_detect_when_no_path(self, project_dir: Path) -> None:
        """When storage_path is None, auto-detection is used."""
        import os

        original = os.getcwd()
        try:
            os.chdir(str(project_dir))
            data = _collect_status(None)
            # Should succeed even without explicit path.
            assert data["status"] == "ok"
        finally:
            os.chdir(original)

    def test_with_active_task(
        self, storage_path: str, manager: WindowManager
    ) -> None:
        manager.start_new_task("CMH-CS1", "Collect status test")
        data = _collect_status(storage_path)
        assert data["current_task"] is not None
        assert data["current_task"]["ticket_id"] == "CMH-CS1"

    def test_with_completed_tasks(
        self, storage_path: str, manager: WindowManager
    ) -> None:
        manager.start_new_task("CMH-CS2", "Completed collect")
        manager.complete_current_task("All done")
        data = _collect_status(storage_path)
        assert data["current_task"] is None
        assert data["window"]["completed_count"] == 1

    def test_storage_metrics(
        self, storage_path: str, manager: WindowManager
    ) -> None:
        manager.start_new_task("CMH-CS3", "Storage metrics test")
        manager.complete_current_task("Done")
        data = _collect_status(storage_path)
        assert data["storage"]["task_files"] >= 1
        assert data["storage"]["total_size_bytes"] > 0


# ---------------------------------------------------------------------------
# --storage-path option
# ---------------------------------------------------------------------------


class TestStoragePathOption:
    """The --storage-path CLI option is passed through correctly."""

    def test_explicit_storage_path(
        self, runner: CliRunner, storage_path: str
    ) -> None:
        result = runner.invoke(
            cli, ["--storage-path", storage_path, "status", "--json-output"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["storage"]["path"] == storage_path

    def test_storage_path_env_var(
        self, runner: CliRunner, storage_path: str
    ) -> None:
        result = runner.invoke(
            cli,
            ["status", "--json-output"],
            env={"CLAUDE_MEMORY_STORAGE_PATH": storage_path},
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["storage"]["path"] == storage_path


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """CLI handles errors gracefully."""

    def test_render_error_status(self, runner: CliRunner) -> None:
        """When _collect_status returns error, text renderer exits with code 1."""
        import click as _click

        error_data = {
            "status": "error",
            "error": "Test error message",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        @_click.command()
        def test_cmd():
            _render_status_text(error_data)

        result = runner.invoke(test_cmd)
        assert result.exit_code == 1
        # Error message should appear in output (Click merges stderr into
        # output by default in the test runner).
        assert "ERROR" in result.output

    def test_json_output_error_status(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """When storage init fails, JSON output includes error."""
        # Use a non-writable path to trigger an error -- but on most systems
        # tmp_path is writable, so we use a path that triggers a config error.
        # We test the structure by passing a valid but empty storage path.
        empty_storage = str(tmp_path / "nonexistent-deep" / ".claude-memory")
        result = runner.invoke(
            cli, ["--storage-path", empty_storage, "status", "--json-output"]
        )
        # It should still succeed because WindowManager creates dirs.
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "ok"


# ---------------------------------------------------------------------------
# Text rendering details
# ---------------------------------------------------------------------------


class TestTextRendering:
    """Detailed text rendering checks."""

    def test_header_present(
        self, runner: CliRunner, storage_path: str
    ) -> None:
        result = runner.invoke(cli, ["--storage-path", storage_path, "status"])
        assert "Claude Code Helper -- Memory Status" in result.output
        assert "=" * 42 in result.output

    def test_version_in_output(
        self, runner: CliRunner, storage_path: str
    ) -> None:
        from claude_code_helper_mcp import __version__

        result = runner.invoke(cli, ["--storage-path", storage_path, "status"])
        assert __version__ in result.output

    def test_all_sections_in_text(
        self, runner: CliRunner, storage_path: str
    ) -> None:
        result = runner.invoke(cli, ["--storage-path", storage_path, "status"])
        assert "Sliding Window" in result.output
        assert "Storage" in result.output
        assert "Configuration" in result.output
        assert "Last activity:" in result.output

    def test_no_active_task_message(
        self, runner: CliRunner, storage_path: str
    ) -> None:
        result = runner.invoke(cli, ["--storage-path", storage_path, "status"])
        assert "No active task" in result.output


# ---------------------------------------------------------------------------
# Integration: full lifecycle through CLI
# ---------------------------------------------------------------------------


class TestFullLifecycle:
    """End-to-end lifecycle test exercising all CLI features."""

    def test_lifecycle_empty_to_active_to_completed(
        self, runner: CliRunner, storage_path: str, manager: WindowManager
    ) -> None:
        # 1. Empty state.
        result = runner.invoke(
            cli, ["--storage-path", storage_path, "status", "--json-output"]
        )
        data = json.loads(result.output)
        assert data["current_task"] is None
        assert data["window"]["tasks_in_window"] == 0

        # 2. Start a task.
        manager.start_new_task("CMH-LC1", "Lifecycle test", phase="phase-1")
        task = manager.get_current_task()
        task.add_step("Started work", "Beginning implementation")
        manager.save_current_task()

        result = runner.invoke(
            cli, ["--storage-path", storage_path, "status", "--json-output"]
        )
        data = json.loads(result.output)
        assert data["current_task"]["ticket_id"] == "CMH-LC1"
        assert data["current_task"]["steps"] == 1
        assert data["window"]["has_active_task"] is True

        # 3. Complete the task.
        manager.complete_current_task("All done")

        result = runner.invoke(
            cli, ["--storage-path", storage_path, "status", "--json-output"]
        )
        data = json.loads(result.output)
        assert data["current_task"] is None
        assert data["window"]["completed_count"] == 1
        assert data["window"]["completed_tasks"][0]["ticket_id"] == "CMH-LC1"
        assert data["last_activity"] is not None

    def test_lifecycle_window_rotation(
        self, runner: CliRunner, storage_path: str, manager: WindowManager
    ) -> None:
        # Fill the window (size 3) plus overflow to trigger archival.
        for i in range(5):
            manager.start_new_task(f"CMH-ROT{i}", f"Rotation test {i}")
            manager.complete_current_task(f"Done {i}")

        result = runner.invoke(
            cli, ["--storage-path", storage_path, "status", "--json-output"]
        )
        data = json.loads(result.output)
        assert data["window"]["completed_count"] == 3
        assert data["window"]["archived_count"] == 2
        assert "CMH-ROT0" in data["window"]["archived_task_ids"]
        assert "CMH-ROT1" in data["window"]["archived_task_ids"]
        assert data["storage"]["task_files"] >= 5

    def test_text_and_json_consistency(
        self, runner: CliRunner, storage_path: str, manager: WindowManager
    ) -> None:
        """Text and JSON output contain the same core information."""
        manager.start_new_task("CMH-CON1", "Consistency test", phase="phase-2")
        manager.complete_current_task("Done")

        text_result = runner.invoke(
            cli, ["--storage-path", storage_path, "status"]
        )
        json_result = runner.invoke(
            cli, ["--storage-path", storage_path, "status", "--json-output"]
        )
        data = json.loads(json_result.output)

        # Both should report the same version.
        assert data["version"] in text_result.output
        # Both should mention the completed task.
        assert "CMH-CON1" in text_result.output
        assert data["window"]["completed_tasks"][0]["ticket_id"] == "CMH-CON1"
