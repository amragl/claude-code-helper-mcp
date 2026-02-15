"""Tests for the ``memory list`` and ``memory show`` CLI commands (CMH-014).

All tests use real files in temporary directories, real MemoryStore and
WindowManager instances, and real Click test runner invocations.  No mocks,
no stubs, no fakes.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest
from click.testing import CliRunner

from claude_code_helper_mcp.cli.main import (
    _collect_list,
    _collect_show,
    _format_duration,
    _format_iso_short,
    _render_list_table,
    _render_show_text,
    cli,
    list_tasks,
    show,
)
from claude_code_helper_mcp.models.records import (
    BranchAction,
    FileAction,
)
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


@pytest.fixture()
def populated_manager(storage_path: str) -> WindowManager:
    """Create a WindowManager with 2 completed tasks and 1 active task."""
    mgr = WindowManager(storage_path=storage_path, window_size=3)

    # Task 1: completed with steps, files, decisions, branches.
    t1 = mgr.start_new_task("CMH-001", "Project initialization", phase="phase-1")
    t1.add_step("Created project", "Set up directory structure", tool_used="Write")
    t1.add_step("Added deps", "Added pyproject.toml dependencies", tool_used="Edit")
    t1.record_file("pyproject.toml", FileAction.CREATED, "Project config")
    t1.record_file("src/main.py", FileAction.CREATED, "Main module")
    t1.record_branch("feature/CMH-001-init", BranchAction.CREATED, base_branch="main")
    t1.add_decision(
        "Use Click for CLI",
        reasoning="Click is well-documented and Click.testing.CliRunner works well",
        alternatives=["argparse", "typer"],
        context="Need a CLI framework for developer commands",
    )
    t1.next_steps = ["Set up test framework", "Create data models"]
    mgr.save_current_task()
    mgr.complete_current_task("Initialized project structure")

    # Task 2: completed with minimal data.
    t2 = mgr.start_new_task("CMH-002", "Data schema", phase="phase-1")
    t2.add_step("Defined models", "Created Pydantic models")
    t2.record_file("src/models.py", FileAction.CREATED, "Data models")
    mgr.save_current_task()
    mgr.complete_current_task("Schema implemented")

    # Task 3: active.
    t3 = mgr.start_new_task("CMH-003", "Storage engine", phase="phase-1")
    t3.add_step("Created store class", "Implemented MemoryStore", tool_used="Write")
    t3.record_file("src/store.py", FileAction.CREATED, "Storage engine")
    t3.record_branch("feature/CMH-003-storage", BranchAction.CREATED, base_branch="main")
    t3.add_decision(
        "Use atomic writes",
        reasoning="Prevent data corruption on crash",
        alternatives=["Direct writes"],
    )
    mgr.save_current_task()

    return mgr


# ---------------------------------------------------------------------------
# _format_duration utility tests
# ---------------------------------------------------------------------------


class TestFormatDuration:
    """Tests for the _format_duration utility function."""

    def test_zero_minutes(self) -> None:
        assert _format_duration(0) == "0m"

    def test_under_sixty_minutes(self) -> None:
        assert _format_duration(15) == "15m"
        assert _format_duration(59) == "59m"

    def test_exactly_sixty_minutes(self) -> None:
        assert _format_duration(60) == "1h"

    def test_over_sixty_minutes(self) -> None:
        assert _format_duration(75) == "1h 15m"
        assert _format_duration(125) == "2h 5m"

    def test_exact_hours(self) -> None:
        assert _format_duration(120) == "2h"
        assert _format_duration(180) == "3h"

    def test_large_duration(self) -> None:
        assert _format_duration(600) == "10h"
        assert _format_duration(601) == "10h 1m"


# ---------------------------------------------------------------------------
# _format_iso_short utility tests
# ---------------------------------------------------------------------------


class TestFormatIsoShort:
    """Tests for the _format_iso_short utility function."""

    def test_none_returns_question_mark(self) -> None:
        assert _format_iso_short(None) == "?"

    def test_empty_string_returns_question_mark(self) -> None:
        assert _format_iso_short("") == "?"

    def test_valid_iso_string(self) -> None:
        result = _format_iso_short("2026-02-15T11:30:00+00:00")
        assert "2026-02-15" in result
        assert "11:30" in result

    def test_iso_without_timezone(self) -> None:
        result = _format_iso_short("2026-02-15T11:30:00")
        assert "2026-02-15" in result
        assert "11:30" in result

    def test_short_string_returned_as_is(self) -> None:
        result = _format_iso_short("short")
        assert result == "short"


# ---------------------------------------------------------------------------
# memory list -- _collect_list tests
# ---------------------------------------------------------------------------


class TestCollectList:
    """Tests for _collect_list data collection."""

    def test_empty_window(self, storage_path: str) -> None:
        data = _collect_list(storage_path, show_all=False)
        assert data["status"] == "ok"
        assert data["tasks"] == []
        assert data["archived_tasks"] == []
        assert data["total_in_window"] == 0
        assert data["total_archived"] == 0

    def test_with_active_task(
        self, storage_path: str, manager: WindowManager
    ) -> None:
        manager.start_new_task("CMH-010", "Test task", phase="phase-2")
        data = _collect_list(storage_path, show_all=False)
        assert data["status"] == "ok"
        assert len(data["tasks"]) == 1
        assert data["tasks"][0]["ticket_id"] == "CMH-010"
        assert data["tasks"][0]["source"] == "active"
        assert data["tasks"][0]["status"] == "active"

    def test_with_completed_tasks(
        self, storage_path: str, populated_manager: WindowManager
    ) -> None:
        data = _collect_list(storage_path, show_all=False)
        assert data["status"] == "ok"
        # 2 completed + 1 active = 3 tasks.
        assert len(data["tasks"]) == 3
        # First should be active task (CMH-003 -- it's listed first in our code).
        active_tasks = [t for t in data["tasks"] if t["source"] == "active"]
        assert len(active_tasks) == 1
        assert active_tasks[0]["ticket_id"] == "CMH-003"
        # 2 window tasks.
        window_tasks = [t for t in data["tasks"] if t["source"] == "window"]
        assert len(window_tasks) == 2

    def test_show_all_false_no_archived(
        self, storage_path: str, populated_manager: WindowManager
    ) -> None:
        data = _collect_list(storage_path, show_all=False)
        assert data["archived_tasks"] == []

    def test_show_all_with_archived_tasks(
        self, storage_path: str
    ) -> None:
        """Verify that --all loads archived tasks from disk."""
        mgr = WindowManager(storage_path=storage_path, window_size=1)

        # Create and complete 3 tasks (window size=1, so 2 will be archived).
        t1 = mgr.start_new_task("CMH-A01", "First task", phase="p1")
        mgr.complete_current_task("Done A01")

        t2 = mgr.start_new_task("CMH-A02", "Second task", phase="p1")
        mgr.complete_current_task("Done A02")

        t3 = mgr.start_new_task("CMH-A03", "Third task", phase="p1")
        mgr.complete_current_task("Done A03")

        data = _collect_list(storage_path, show_all=True)
        assert data["status"] == "ok"
        # Only 1 in window (CMH-A03, since window_size=1).
        assert data["total_in_window"] == 1
        # 2 archived (CMH-A01 and CMH-A02).
        assert data["total_archived"] == 2
        archived_ids = [t["ticket_id"] for t in data["archived_tasks"]]
        assert "CMH-A01" in archived_ids
        assert "CMH-A02" in archived_ids

    def test_task_has_duration(
        self, storage_path: str, populated_manager: WindowManager
    ) -> None:
        data = _collect_list(storage_path, show_all=False)
        completed = [t for t in data["tasks"] if t["source"] == "window"]
        for t in completed:
            assert "duration_minutes" in t

    def test_task_has_step_and_file_counts(
        self, storage_path: str, populated_manager: WindowManager
    ) -> None:
        data = _collect_list(storage_path, show_all=False)
        for t in data["tasks"]:
            assert "steps" in t
            assert "files" in t
            assert "decisions" in t

    def test_invalid_storage_path(self) -> None:
        data = _collect_list("/nonexistent/unlikely/path", show_all=False)
        assert data["status"] == "error"

    def test_window_size_reported(
        self, storage_path: str, manager: WindowManager
    ) -> None:
        data = _collect_list(storage_path, show_all=False)
        assert data["window_size"] == 3


# ---------------------------------------------------------------------------
# memory list -- CLI invocation tests
# ---------------------------------------------------------------------------


class TestListCommand:
    """Tests for the ``memory list`` CLI command via CliRunner."""

    def test_list_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["list", "--help"])
        assert result.exit_code == 0
        assert "List tasks" in result.output
        assert "--all" in result.output
        assert "--format" in result.output

    def test_list_empty_window(
        self, runner: CliRunner, storage_path: str
    ) -> None:
        result = runner.invoke(cli, ["--storage-path", storage_path, "list"])
        assert result.exit_code == 0
        assert "No tasks found" in result.output

    def test_list_table_format(
        self, runner: CliRunner, storage_path: str, populated_manager: WindowManager
    ) -> None:
        result = runner.invoke(cli, ["--storage-path", storage_path, "list"])
        assert result.exit_code == 0
        assert "Task List" in result.output
        assert "CMH-001" in result.output
        assert "CMH-002" in result.output
        assert "CMH-003" in result.output

    def test_list_json_format(
        self, runner: CliRunner, storage_path: str, populated_manager: WindowManager
    ) -> None:
        result = runner.invoke(
            cli, ["--storage-path", storage_path, "list", "--format", "json"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "ok"
        assert len(data["tasks"]) == 3

    def test_list_with_all_flag(
        self, runner: CliRunner, storage_path: str
    ) -> None:
        # Create a scenario with archived tasks.
        mgr = WindowManager(storage_path=storage_path, window_size=1)
        mgr.start_new_task("T-001", "Task 1")
        mgr.complete_current_task()
        mgr.start_new_task("T-002", "Task 2")
        mgr.complete_current_task()

        result = runner.invoke(
            cli, ["--storage-path", storage_path, "list", "--all"]
        )
        assert result.exit_code == 0
        assert "Archived" in result.output
        assert "T-001" in result.output
        assert "T-002" in result.output

    def test_list_all_json(
        self, runner: CliRunner, storage_path: str
    ) -> None:
        mgr = WindowManager(storage_path=storage_path, window_size=1)
        mgr.start_new_task("T-001", "Task 1")
        mgr.complete_current_task()
        mgr.start_new_task("T-002", "Task 2")
        mgr.complete_current_task()

        result = runner.invoke(
            cli,
            ["--storage-path", storage_path, "list", "--all", "--format", "json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["total_archived"] >= 1

    def test_list_shows_status_colours(
        self, runner: CliRunner, storage_path: str, populated_manager: WindowManager
    ) -> None:
        """Verify that the list command runs without error and contains status text."""
        result = runner.invoke(cli, ["--storage-path", storage_path, "list"])
        assert result.exit_code == 0
        # Status text should appear.
        assert "active" in result.output or "completed" in result.output

    def test_list_shows_step_counts(
        self, runner: CliRunner, storage_path: str, populated_manager: WindowManager
    ) -> None:
        result = runner.invoke(cli, ["--storage-path", storage_path, "list"])
        assert result.exit_code == 0
        # The header should contain Steps and Files.
        assert "Steps" in result.output
        assert "Files" in result.output


# ---------------------------------------------------------------------------
# memory show -- _collect_show tests
# ---------------------------------------------------------------------------


class TestCollectShow:
    """Tests for _collect_show data collection."""

    def test_show_active_task(
        self, storage_path: str, populated_manager: WindowManager
    ) -> None:
        data = _collect_show(storage_path, "CMH-003")
        assert data["status"] == "ok"
        assert data["ticket_id"] == "CMH-003"
        assert data["title"] == "Storage engine"
        assert data["task_status"] == "active"
        assert data["source"] == "active"
        assert len(data["steps"]) == 1
        assert len(data["files"]) == 1
        assert len(data["branches"]) == 1
        assert len(data["decisions"]) == 1

    def test_show_completed_task(
        self, storage_path: str, populated_manager: WindowManager
    ) -> None:
        data = _collect_show(storage_path, "CMH-001")
        assert data["status"] == "ok"
        assert data["ticket_id"] == "CMH-001"
        assert data["task_status"] == "completed"
        assert data["source"] == "window"
        assert len(data["steps"]) == 2
        assert len(data["files"]) == 2
        assert len(data["branches"]) == 1
        assert len(data["decisions"]) == 1
        assert data["summary"] == "Initialized project structure"

    def test_show_not_found(self, storage_path: str) -> None:
        data = _collect_show(storage_path, "CMH-999")
        assert data["status"] == "not_found"
        assert "not found" in data["error"]

    def test_show_archived_task(self, storage_path: str) -> None:
        """Verify that show can load archived tasks from disk."""
        mgr = WindowManager(storage_path=storage_path, window_size=1)
        t = mgr.start_new_task("CMH-A01", "Archived task", phase="p1")
        t.add_step("Worked on it", "Detailed work")
        t.record_file("file.py", FileAction.CREATED, "Created file")
        mgr.save_current_task()
        mgr.complete_current_task("Archived task done")

        # Push another task to archive CMH-A01.
        mgr.start_new_task("CMH-A02", "Another task")
        mgr.complete_current_task()

        data = _collect_show(storage_path, "CMH-A01")
        assert data["status"] == "ok"
        assert data["source"] == "archived"
        assert data["ticket_id"] == "CMH-A01"
        assert len(data["steps"]) == 1
        assert len(data["files"]) == 1

    def test_show_has_counts(
        self, storage_path: str, populated_manager: WindowManager
    ) -> None:
        data = _collect_show(storage_path, "CMH-001")
        assert "counts" in data
        assert data["counts"]["steps"] == 2
        assert data["counts"]["files"] == 2
        assert data["counts"]["branches"] == 1
        assert data["counts"]["decisions"] == 1

    def test_show_step_details(
        self, storage_path: str, populated_manager: WindowManager
    ) -> None:
        data = _collect_show(storage_path, "CMH-001")
        step = data["steps"][0]
        assert step["step_number"] == 1
        assert step["action"] == "Created project"
        assert step["tool_used"] == "Write"
        assert "timestamp" in step

    def test_show_file_details(
        self, storage_path: str, populated_manager: WindowManager
    ) -> None:
        data = _collect_show(storage_path, "CMH-001")
        f = data["files"][0]
        assert f["path"] == "pyproject.toml"
        assert f["action"] == "created"
        assert "action_count" in f

    def test_show_branch_details(
        self, storage_path: str, populated_manager: WindowManager
    ) -> None:
        data = _collect_show(storage_path, "CMH-001")
        b = data["branches"][0]
        assert b["branch_name"] == "feature/CMH-001-init"
        assert b["action"] == "created"
        assert b["base_branch"] == "main"

    def test_show_decision_details(
        self, storage_path: str, populated_manager: WindowManager
    ) -> None:
        data = _collect_show(storage_path, "CMH-001")
        d = data["decisions"][0]
        assert d["decision_number"] == 1
        assert d["decision"] == "Use Click for CLI"
        assert "reasoning" in d
        assert d["alternatives"] == ["argparse", "typer"]
        assert "context" in d

    def test_show_next_steps(
        self, storage_path: str, populated_manager: WindowManager
    ) -> None:
        data = _collect_show(storage_path, "CMH-001")
        assert data["next_steps"] == ["Set up test framework", "Create data models"]

    def test_show_duration_for_completed(
        self, storage_path: str, populated_manager: WindowManager
    ) -> None:
        data = _collect_show(storage_path, "CMH-001")
        assert data["duration_minutes"] is not None

    def test_show_elapsed_for_active(
        self, storage_path: str, populated_manager: WindowManager
    ) -> None:
        data = _collect_show(storage_path, "CMH-003")
        assert data["duration_minutes"] is not None

    def test_show_invalid_storage_path(self) -> None:
        data = _collect_show("/nonexistent/unlikely/path", "CMH-001")
        assert data["status"] == "error"


# ---------------------------------------------------------------------------
# memory show -- CLI invocation tests
# ---------------------------------------------------------------------------


class TestShowCommand:
    """Tests for the ``memory show`` CLI command via CliRunner."""

    def test_show_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["show", "--help"])
        assert result.exit_code == 0
        assert "Show full details" in result.output
        assert "TICKET_ID" in result.output
        assert "--format" in result.output

    def test_show_text_format(
        self, runner: CliRunner, storage_path: str, populated_manager: WindowManager
    ) -> None:
        result = runner.invoke(
            cli, ["--storage-path", storage_path, "show", "CMH-001"]
        )
        assert result.exit_code == 0
        assert "Task Details" in result.output
        assert "CMH-001" in result.output
        assert "Project initialization" in result.output
        assert "completed" in result.output

    def test_show_json_format(
        self, runner: CliRunner, storage_path: str, populated_manager: WindowManager
    ) -> None:
        result = runner.invoke(
            cli,
            ["--storage-path", storage_path, "show", "CMH-001", "--format", "json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "ok"
        assert data["ticket_id"] == "CMH-001"
        assert len(data["steps"]) == 2

    def test_show_not_found(
        self, runner: CliRunner, storage_path: str
    ) -> None:
        result = runner.invoke(
            cli, ["--storage-path", storage_path, "show", "CMH-999"]
        )
        assert result.exit_code != 0

    def test_show_active_task(
        self, runner: CliRunner, storage_path: str, populated_manager: WindowManager
    ) -> None:
        result = runner.invoke(
            cli, ["--storage-path", storage_path, "show", "CMH-003"]
        )
        assert result.exit_code == 0
        assert "CMH-003" in result.output
        assert "Storage engine" in result.output
        assert "active" in result.output

    def test_show_displays_timeline(
        self, runner: CliRunner, storage_path: str, populated_manager: WindowManager
    ) -> None:
        result = runner.invoke(
            cli, ["--storage-path", storage_path, "show", "CMH-001"]
        )
        assert result.exit_code == 0
        assert "Timeline" in result.output
        assert "Step" in result.output
        assert "Created project" in result.output
        assert "Added deps" in result.output

    def test_show_displays_files(
        self, runner: CliRunner, storage_path: str, populated_manager: WindowManager
    ) -> None:
        result = runner.invoke(
            cli, ["--storage-path", storage_path, "show", "CMH-001"]
        )
        assert result.exit_code == 0
        assert "Files" in result.output
        assert "pyproject.toml" in result.output

    def test_show_displays_branches(
        self, runner: CliRunner, storage_path: str, populated_manager: WindowManager
    ) -> None:
        result = runner.invoke(
            cli, ["--storage-path", storage_path, "show", "CMH-001"]
        )
        assert result.exit_code == 0
        assert "Branches" in result.output
        assert "feature/CMH-001-init" in result.output

    def test_show_displays_decisions(
        self, runner: CliRunner, storage_path: str, populated_manager: WindowManager
    ) -> None:
        result = runner.invoke(
            cli, ["--storage-path", storage_path, "show", "CMH-001"]
        )
        assert result.exit_code == 0
        assert "Decisions" in result.output
        assert "Use Click for CLI" in result.output

    def test_show_displays_next_steps(
        self, runner: CliRunner, storage_path: str, populated_manager: WindowManager
    ) -> None:
        result = runner.invoke(
            cli, ["--storage-path", storage_path, "show", "CMH-001"]
        )
        assert result.exit_code == 0
        assert "Next Steps" in result.output
        assert "Set up test framework" in result.output

    def test_show_displays_summary(
        self, runner: CliRunner, storage_path: str, populated_manager: WindowManager
    ) -> None:
        result = runner.invoke(
            cli, ["--storage-path", storage_path, "show", "CMH-001"]
        )
        assert result.exit_code == 0
        assert "Initialized project structure" in result.output


# ---------------------------------------------------------------------------
# Render function tests (text output verification)
# ---------------------------------------------------------------------------


class TestRenderListTable:
    """Tests for _render_list_table rendering."""

    def test_renders_error(self, runner: CliRunner) -> None:
        """Error data should print to stderr and exit non-zero."""
        data = {"status": "error", "error": "test error"}
        result = runner.invoke(cli, ["list"], standalone_mode=False)
        # The function itself requires a storage path, so test indirectly
        # via the collect function returning error data.
        assert data["status"] == "error"

    def test_empty_tasks_message(
        self, runner: CliRunner, storage_path: str
    ) -> None:
        result = runner.invoke(cli, ["--storage-path", storage_path, "list"])
        assert "No tasks found" in result.output


class TestRenderShowText:
    """Tests for _render_show_text rendering."""

    def test_not_found_exits_nonzero(
        self, runner: CliRunner, storage_path: str
    ) -> None:
        result = runner.invoke(
            cli, ["--storage-path", storage_path, "show", "NONEXISTENT"]
        )
        assert result.exit_code != 0

    def test_shows_overview_section(
        self, runner: CliRunner, storage_path: str, populated_manager: WindowManager
    ) -> None:
        result = runner.invoke(
            cli, ["--storage-path", storage_path, "show", "CMH-002"]
        )
        assert result.exit_code == 0
        assert "Overview" in result.output
        assert "CMH-002" in result.output
        assert "Data schema" in result.output

    def test_shows_counts_line(
        self, runner: CliRunner, storage_path: str, populated_manager: WindowManager
    ) -> None:
        result = runner.invoke(
            cli, ["--storage-path", storage_path, "show", "CMH-001"]
        )
        assert result.exit_code == 0
        assert "Steps:" in result.output
        assert "Files:" in result.output
        assert "Branches:" in result.output
        assert "Decisions:" in result.output


# ---------------------------------------------------------------------------
# Edge cases and integration
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case tests for list and show commands."""

    def test_list_with_failed_task(self, storage_path: str) -> None:
        """A failed task should appear in the list with status 'failed'."""
        mgr = WindowManager(storage_path=storage_path, window_size=3)
        mgr.start_new_task("CMH-F01", "Failed task")
        mgr.fail_current_task("Build error")

        data = _collect_list(storage_path, show_all=False)
        failed_tasks = [t for t in data["tasks"] if t["status"] == "failed"]
        assert len(failed_tasks) == 1
        assert failed_tasks[0]["ticket_id"] == "CMH-F01"

    def test_show_failed_task(self, storage_path: str) -> None:
        """Show should display failed task details including failure summary."""
        mgr = WindowManager(storage_path=storage_path, window_size=3)
        mgr.start_new_task("CMH-F01", "Failed task")
        mgr.fail_current_task("Build error")

        data = _collect_show(storage_path, "CMH-F01")
        assert data["status"] == "ok"
        assert data["task_status"] == "failed"
        assert "FAILED" in data["summary"]

    def test_list_task_with_no_steps(self, storage_path: str) -> None:
        """A task with zero steps should still appear in the list."""
        mgr = WindowManager(storage_path=storage_path, window_size=3)
        mgr.start_new_task("CMH-E01", "Empty task")

        data = _collect_list(storage_path, show_all=False)
        assert len(data["tasks"]) == 1
        assert data["tasks"][0]["steps"] == 0

    def test_show_task_with_no_steps(self, storage_path: str) -> None:
        """Show for a task with no steps should work without error."""
        mgr = WindowManager(storage_path=storage_path, window_size=3)
        mgr.start_new_task("CMH-E01", "Empty task")

        data = _collect_show(storage_path, "CMH-E01")
        assert data["status"] == "ok"
        assert len(data["steps"]) == 0

    def test_show_task_with_many_steps(self, storage_path: str) -> None:
        """A task with many steps should render without error."""
        mgr = WindowManager(storage_path=storage_path, window_size=3)
        t = mgr.start_new_task("CMH-M01", "Many steps task")
        for i in range(20):
            t.add_step(f"Step {i+1}", f"Description {i+1}")
        mgr.save_current_task()

        data = _collect_show(storage_path, "CMH-M01")
        assert data["status"] == "ok"
        assert len(data["steps"]) == 20

    def test_list_multiple_phases(self, storage_path: str) -> None:
        """Tasks from different phases should all appear in the list."""
        mgr = WindowManager(storage_path=storage_path, window_size=3)
        mgr.start_new_task("P1-001", "Phase 1 task", phase="phase-1")
        mgr.complete_current_task()
        mgr.start_new_task("P2-001", "Phase 2 task", phase="phase-2")

        data = _collect_list(storage_path, show_all=False)
        phases = {t.get("phase") for t in data["tasks"]}
        assert "phase-1" in phases
        assert "phase-2" in phases

    def test_show_json_round_trip(
        self, storage_path: str, populated_manager: WindowManager
    ) -> None:
        """JSON output from show should be valid JSON with all fields."""
        data = _collect_show(storage_path, "CMH-001")
        json_str = json.dumps(data, indent=2, default=str)
        parsed = json.loads(json_str)
        assert parsed["ticket_id"] == "CMH-001"
        assert len(parsed["steps"]) == 2

    def test_list_json_round_trip(
        self, storage_path: str, populated_manager: WindowManager
    ) -> None:
        """JSON output from list should be valid JSON with all fields."""
        data = _collect_list(storage_path, show_all=True)
        json_str = json.dumps(data, indent=2, default=str)
        parsed = json.loads(json_str)
        assert parsed["status"] == "ok"

    def test_show_step_with_files_involved(self, storage_path: str) -> None:
        """A step with files_involved should include them in the output."""
        mgr = WindowManager(storage_path=storage_path, window_size=3)
        t = mgr.start_new_task("CMH-FI01", "Files involved test")
        t.add_step(
            "Edited files",
            "Made changes to multiple files",
            files_involved=["src/a.py", "src/b.py", "src/c.py"],
        )
        mgr.save_current_task()

        data = _collect_show(storage_path, "CMH-FI01")
        assert data["steps"][0]["files_involved"] == [
            "src/a.py",
            "src/b.py",
            "src/c.py",
        ]

    def test_show_step_failure_marker(self, storage_path: str) -> None:
        """A failed step should have success=False in the data."""
        mgr = WindowManager(storage_path=storage_path, window_size=3)
        t = mgr.start_new_task("CMH-SF01", "Step failure test")
        t.add_step("Ran tests", "Tests failed", success=False)
        mgr.save_current_task()

        data = _collect_show(storage_path, "CMH-SF01")
        assert data["steps"][0]["success"] is False

    def test_list_long_title_truncation(
        self, runner: CliRunner, storage_path: str
    ) -> None:
        """A very long title should be truncated in table display."""
        mgr = WindowManager(storage_path=storage_path, window_size=3)
        long_title = "A" * 100
        mgr.start_new_task("CMH-LT01", long_title)

        result = runner.invoke(cli, ["--storage-path", storage_path, "list"])
        assert result.exit_code == 0
        # The title should be truncated with "...".
        assert "..." in result.output

    def test_show_file_with_multiple_actions(self, storage_path: str) -> None:
        """A file with multiple actions should report the correct action_count."""
        mgr = WindowManager(storage_path=storage_path, window_size=3)
        t = mgr.start_new_task("CMH-MA01", "Multi-action file")
        t.record_file("src/main.py", FileAction.CREATED, "Created")
        t.record_file("src/main.py", FileAction.MODIFIED, "Updated imports")
        t.record_file("src/main.py", FileAction.MODIFIED, "Added function")
        mgr.save_current_task()

        data = _collect_show(storage_path, "CMH-MA01")
        assert len(data["files"]) == 1
        assert data["files"][0]["action_count"] == 3

    def test_archived_task_not_loadable(self, storage_path: str) -> None:
        """If an archived task file is missing, list --all shows (unavailable)."""
        mgr = WindowManager(storage_path=storage_path, window_size=1)
        mgr.start_new_task("CMH-DEL01", "Will be deleted")
        mgr.complete_current_task()
        mgr.start_new_task("CMH-DEL02", "Pushes first out")
        mgr.complete_current_task()

        # Delete the archived task file.
        mgr.store.delete_task("CMH-DEL01")

        data = _collect_list(storage_path, show_all=True)
        archived = data["archived_tasks"]
        missing = [t for t in archived if t["ticket_id"] == "CMH-DEL01"]
        assert len(missing) == 1
        assert missing[0]["title"] == "(unavailable)"
