"""Tests for the MarkdownGenerator and generate_summary MCP tool (CMH-010).

Verifies:
- MarkdownGenerator.generate_task_summary() with various task states.
- MarkdownGenerator.generate_current_summary() for active tasks.
- MarkdownGenerator.generate_window_index() for window overview.
- generate_summary MCP tool with all summary_type modes.
- Edge cases: empty tasks, no active task, archived tasks, formatting.

All tests use real MemoryStore/WindowManager instances with temporary
directories.  No mocks.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from claude_code_helper_mcp.mcp.markdown import MarkdownGenerator
from claude_code_helper_mcp.mcp.server import (
    create_server,
    get_window_manager,
    reset_server,
)
from claude_code_helper_mcp.models.records import (
    BranchAction,
    FileAction,
)
from claude_code_helper_mcp.models.task import TaskMemory, TaskStatus
from claude_code_helper_mcp.models.window import MemoryWindow
from claude_code_helper_mcp.storage.window_manager import WindowManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_tool_result(result) -> dict:
    """Extract a dict from a FastMCP ToolResult."""
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        return json.loads(result)
    if hasattr(result, "content") and result.content:
        text = result.content[0].text
        return json.loads(text)
    raise TypeError(f"Cannot parse tool result of type {type(result)}")


def _make_task(
    ticket_id: str = "TEST-001",
    title: str = "Test task",
    phase: str = "phase-1",
    status: TaskStatus = TaskStatus.ACTIVE,
    with_steps: int = 0,
    with_files: int = 0,
    with_branches: int = 0,
    with_decisions: int = 0,
    summary: str = "",
    completed: bool = False,
) -> TaskMemory:
    """Create a TaskMemory with optional pre-populated records."""
    started = datetime(2026, 2, 14, 10, 0, 0, tzinfo=timezone.utc)
    task = TaskMemory(
        ticket_id=ticket_id,
        title=title,
        phase=phase,
        status=status,
        started_at=started,
        summary=summary,
    )

    for i in range(with_steps):
        task.add_step(
            action=f"Step action {i + 1}",
            description=f"Detailed description for step {i + 1}",
            tool_used="Write" if i % 2 == 0 else "Bash",
            result_summary=f"Step {i + 1} completed successfully",
            files_involved=[f"src/file_{i + 1}.py"] if i < 3 else [],
            success=True,
        )

    for i in range(with_files):
        task.record_file(
            path=f"src/module_{i + 1}.py",
            action=FileAction.CREATED if i % 2 == 0 else FileAction.MODIFIED,
            description=f"File {i + 1} action description",
        )

    for i in range(with_branches):
        task.record_branch(
            branch_name=f"feature/{ticket_id.lower()}-branch-{i + 1}",
            action=BranchAction.CREATED if i == 0 else BranchAction.PUSHED,
            base_branch="main" if i == 0 else None,
        )

    for i in range(with_decisions):
        task.add_decision(
            decision=f"Decision {i + 1} text",
            reasoning=f"Reasoning for decision {i + 1}",
            alternatives=[f"Alt A for {i + 1}", f"Alt B for {i + 1}"],
            context=f"Context for decision {i + 1}",
        )

    if completed:
        task.complete(summary or "Task completed successfully")

    return task


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_server():
    """Reset the server singleton before and after each test."""
    reset_server()
    yield
    reset_server()


@pytest.fixture
def project_dir():
    """Create a temporary project directory with a .git marker."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / ".git").mkdir()
        yield tmpdir


@pytest.fixture
def gen():
    """Return a fresh MarkdownGenerator instance."""
    return MarkdownGenerator()


@pytest.fixture
def server(project_dir):
    """Create a server instance with a temp project directory."""
    return create_server(project_root=project_dir)


# ===========================================================================
# MarkdownGenerator.generate_task_summary() tests
# ===========================================================================


class TestGenerateTaskSummary:
    """Tests for full task summary generation."""

    def test_minimal_active_task(self, gen):
        """A minimal active task produces a valid markdown document."""
        task = _make_task()
        md = gen.generate_task_summary(task)

        assert "# Task Summary: TEST-001" in md
        assert "**Test task**" in md
        assert "IN PROGRESS" in md
        assert "| Ticket | TEST-001 |" in md
        assert "| Phase | phase-1 |" in md

    def test_completed_task_with_duration(self, gen):
        """Completed tasks show duration in metadata."""
        task = _make_task(completed=True, summary="All done")

        md = gen.generate_task_summary(task)

        assert "DONE" in md
        assert "| Duration |" in md
        assert "All done" in md
        assert "## Summary" in md

    def test_steps_timeline_section(self, gen):
        """Steps are rendered as a numbered timeline."""
        task = _make_task(with_steps=3)
        md = gen.generate_task_summary(task)

        assert "## Steps Timeline" in md
        assert "1. [+] **Step action 1**" in md
        assert "2. [+] **Step action 2**" in md
        assert "3. [+] **Step action 3**" in md
        assert "(tool: `Write`)" in md
        assert "(tool: `Bash`)" in md
        assert "Result: Step 1 completed successfully" in md

    def test_files_modified_table(self, gen):
        """Files are rendered in a markdown table."""
        task = _make_task(with_files=3)
        md = gen.generate_task_summary(task)

        assert "## Files Modified" in md
        assert "| `src/module_1.py` | created |" in md
        assert "| `src/module_2.py` | modified |" in md
        assert "| `src/module_3.py` | created |" in md

    def test_file_action_history_section(self, gen):
        """Files with action history show detailed history."""
        task = _make_task()
        task.record_file("src/main.py", FileAction.CREATED, "Initial creation")
        task.record_file("src/main.py", FileAction.MODIFIED, "Added imports")
        task.record_file("src/main.py", FileAction.MODIFIED, "Added class")

        md = gen.generate_task_summary(task)

        assert "### File Action History" in md
        assert "`src/main.py`" in md
        assert "3 actions" in md

    def test_branches_table(self, gen):
        """Branches are rendered in a markdown table."""
        task = _make_task(with_branches=2)
        md = gen.generate_task_summary(task)

        assert "## Branches" in md
        assert "| `feature/test-001-branch-1` | created | main |" in md
        assert "| `feature/test-001-branch-2` | pushed |" in md

    def test_decisions_section(self, gen):
        """Decisions are rendered with reasoning and alternatives."""
        task = _make_task(with_decisions=2)
        md = gen.generate_task_summary(task)

        assert "## Decisions" in md
        assert "### Decision 1: Decision 1 text" in md
        assert "**Reasoning:** Reasoning for decision 1" in md
        assert "- Alt A for 1" in md
        assert "- Alt B for 1" in md
        assert "**Context:** Context for decision 1" in md
        assert "### Decision 2: Decision 2 text" in md

    def test_next_steps_section(self, gen):
        """Next steps are rendered as a bullet list."""
        task = _make_task()
        task.next_steps = ["Write tests", "Create PR", "Run validation"]

        md = gen.generate_task_summary(task)

        assert "## Next Steps" in md
        assert "- Write tests" in md
        assert "- Create PR" in md
        assert "- Run validation" in md

    def test_metadata_section(self, gen):
        """Non-empty metadata is rendered as a JSON block."""
        task = _make_task()
        task.metadata = {"description": "Task scope details", "pr_number": 42}

        md = gen.generate_task_summary(task)

        # There will be two "## Metadata" sections (one for the table, one for json).
        # The JSON section contains the metadata content.
        assert '"description": "Task scope details"' in md
        assert '"pr_number": 42' in md

    def test_empty_optional_sections_omitted(self, gen):
        """Sections with no data are not rendered."""
        task = _make_task()  # No steps, files, branches, decisions

        md = gen.generate_task_summary(task)

        assert "## Steps Timeline" not in md
        assert "## Files Modified" not in md
        assert "## Branches" not in md
        assert "## Decisions" not in md
        assert "## Next Steps" not in md

    def test_failed_step_marker(self, gen):
        """Failed steps show [x] marker instead of [+]."""
        task = _make_task()
        task.add_step(action="Run tests", success=True)
        task.add_step(action="Build failed", success=False)

        md = gen.generate_task_summary(task)

        assert "1. [+] **Run tests**" in md
        assert "2. [x] **Build failed**" in md

    def test_failed_task_status(self, gen):
        """Failed tasks show FAILED status badge."""
        task = _make_task(status=TaskStatus.FAILED)
        md = gen.generate_task_summary(task)

        assert "FAILED" in md

    def test_no_phase_omits_phase_row(self, gen):
        """Tasks without a phase omit the Phase row."""
        task = _make_task(phase=None)
        md = gen.generate_task_summary(task)

        assert "| Phase |" not in md

    def test_full_task_all_sections(self, gen):
        """A fully populated task renders all sections correctly."""
        task = _make_task(
            ticket_id="CMH-010",
            title="Markdown summary generation",
            phase="phase-2",
            with_steps=5,
            with_files=3,
            with_branches=2,
            with_decisions=2,
            completed=True,
            summary="Implemented MarkdownGenerator with full test suite.",
        )
        task.next_steps = ["Review PR", "Merge to main"]

        md = gen.generate_task_summary(task)

        assert "# Task Summary: CMH-010" in md
        assert "## Summary" in md
        assert "## Steps Timeline" in md
        assert "## Files Modified" in md
        assert "## Branches" in md
        assert "## Decisions" in md
        assert "## Next Steps" in md


# ===========================================================================
# MarkdownGenerator.generate_current_summary() tests
# ===========================================================================


class TestGenerateCurrentSummary:
    """Tests for current task progress summary."""

    def test_minimal_active_task(self, gen):
        """A minimal active task produces a valid current summary."""
        task = _make_task()
        md = gen.generate_current_summary(task)

        assert "# Current Task: TEST-001" in md
        assert "**Test task**" in md
        assert "IN PROGRESS" in md
        assert "**Phase:** phase-1" in md

    def test_includes_step_count(self, gen):
        """Summary shows the number of steps taken."""
        task = _make_task(with_steps=7)
        md = gen.generate_current_summary(task)

        assert "**Steps taken:** 7" in md

    def test_recent_steps_limited_to_10(self, gen):
        """Only the last 10 steps are shown in recent steps."""
        task = _make_task(with_steps=15)
        md = gen.generate_current_summary(task)

        assert "## Recent Steps (last 10)" in md
        # Step 6 should not appear (it is beyond the last 10 of 15)
        assert "**Step action 6**" in md  # step 6 is within last 10 (steps 6-15)
        assert "**Step action 1**" not in md  # step 1 is too old

    def test_files_touched_section(self, gen):
        """Files are listed as a bullet list."""
        task = _make_task(with_files=3)
        md = gen.generate_current_summary(task)

        assert "## Files Touched" in md
        assert "- `src/module_1.py` (created)" in md

    def test_key_decisions_limited_to_5(self, gen):
        """Only the last 5 decisions are shown."""
        task = _make_task(with_decisions=8)
        md = gen.generate_current_summary(task)

        assert "## Key Decisions (last 5)" in md
        assert "**D4:**" in md  # decision 4 is within last 5 (4-8)
        assert "**D3:**" not in md  # decision 3 is too old

    def test_active_branch_shown(self, gen):
        """The active branch is displayed when present."""
        task = _make_task(with_branches=1)
        md = gen.generate_current_summary(task)

        assert "**Active branch:**" in md
        assert "feature/test-001-branch-1" in md

    def test_elapsed_time_shown(self, gen):
        """Elapsed time since task start is displayed."""
        task = _make_task()
        md = gen.generate_current_summary(task)

        assert "**Elapsed:**" in md

    def test_next_steps_shown(self, gen):
        """Next steps are rendered when present."""
        task = _make_task()
        task.next_steps = ["Write more tests", "Update docs"]

        md = gen.generate_current_summary(task)

        assert "## Next Steps" in md
        assert "- Write more tests" in md

    def test_no_steps_omits_section(self, gen):
        """A task with no steps omits the Recent Steps section."""
        task = _make_task(with_steps=0)
        md = gen.generate_current_summary(task)

        assert "## Recent Steps" not in md


# ===========================================================================
# MarkdownGenerator.generate_window_index() tests
# ===========================================================================


class TestGenerateWindowIndex:
    """Tests for window index generation."""

    def test_empty_window(self, gen):
        """An empty window produces a valid index."""
        window = MemoryWindow()
        md = gen.generate_window_index(window)

        assert "# Memory Window Index" in md
        assert "*No active task.*" in md
        assert "Tasks in window: 0" in md

    def test_window_with_current_task(self, gen):
        """A window with an active task shows the current task section."""
        window = MemoryWindow()
        window.start_task("TEST-001", "Active task", "phase-1")

        md = gen.generate_window_index(window)

        assert "**TEST-001**: Active task" in md
        assert "IN PROGRESS" in md
        assert "Tasks in window: 1" in md

    def test_window_with_completed_tasks(self, gen):
        """Completed tasks appear in the table."""
        window = MemoryWindow(window_size=5)
        task1 = window.start_task("T-001", "First task")
        window.complete_current_task("Done 1")
        task2 = window.start_task("T-002", "Second task")
        window.complete_current_task("Done 2")

        md = gen.generate_window_index(window)

        assert "## Completed Tasks" in md
        assert "T-001" in md
        assert "T-002" in md
        assert "DONE" in md

    def test_window_with_archived_tasks(self, gen):
        """Archived task IDs are listed."""
        window = MemoryWindow(window_size=1)
        window.start_task("T-001", "Task 1")
        window.complete_current_task("Done 1")
        window.start_task("T-002", "Task 2")
        window.complete_current_task("Done 2")
        # T-001 should be archived since window_size is 1

        md = gen.generate_window_index(window)

        assert "## Archived Tasks" in md
        assert "T-001" in md

    def test_statistics_section(self, gen):
        """Statistics section shows correct counts."""
        window = MemoryWindow(window_size=3)
        window.start_task("T-001", "Task 1")
        window.complete_current_task("Done")
        window.start_task("T-002", "Active")

        md = gen.generate_window_index(window)

        assert "## Statistics" in md
        assert "Tasks in window: 2" in md
        assert "Completed: 1" in md

    def test_custom_completed_tasks_list(self, gen):
        """Passing a custom completed_tasks list uses those tasks."""
        window = MemoryWindow()
        custom_tasks = [
            _make_task(ticket_id="C-001", title="Custom 1", completed=True),
            _make_task(ticket_id="C-002", title="Custom 2", completed=True),
        ]

        md = gen.generate_window_index(window, completed_tasks=custom_tasks)

        assert "C-001" in md
        assert "C-002" in md


# ===========================================================================
# Formatting helper tests
# ===========================================================================


class TestFormattingHelpers:
    """Tests for the private formatting helpers."""

    def test_format_dt(self, gen):
        """Datetime formatting produces expected string."""
        dt = datetime(2026, 2, 14, 15, 30, 45, tzinfo=timezone.utc)
        result = gen._format_dt(dt)
        assert result == "2026-02-14 15:30:45 UTC"

    def test_format_dt_naive(self, gen):
        """Naive datetimes are treated as UTC."""
        dt = datetime(2026, 1, 1, 0, 0, 0)
        result = gen._format_dt(dt)
        assert result == "2026-01-01 00:00:00 UTC"

    def test_format_duration_hours(self, gen):
        """Duration formatting with hours."""
        start = datetime(2026, 2, 14, 10, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 2, 14, 12, 30, 15, tzinfo=timezone.utc)
        result = gen._format_duration(start, end)
        assert result == "2h 30m 15s"

    def test_format_duration_minutes_only(self, gen):
        """Duration formatting with just minutes."""
        start = datetime(2026, 2, 14, 10, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 2, 14, 10, 15, 0, tzinfo=timezone.utc)
        result = gen._format_duration(start, end)
        assert result == "15m"

    def test_format_duration_seconds_only(self, gen):
        """Duration formatting with just seconds."""
        start = datetime(2026, 2, 14, 10, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 2, 14, 10, 0, 45, tzinfo=timezone.utc)
        result = gen._format_duration(start, end)
        assert result == "45s"

    def test_format_duration_zero(self, gen):
        """Zero duration returns '0s'."""
        dt = datetime(2026, 2, 14, 10, 0, 0, tzinfo=timezone.utc)
        result = gen._format_duration(dt, dt)
        assert result == "0s"

    def test_format_duration_negative(self, gen):
        """Negative duration returns '0s'."""
        start = datetime(2026, 2, 14, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 2, 14, 10, 0, 0, tzinfo=timezone.utc)
        result = gen._format_duration(start, end)
        assert result == "0s"

    def test_status_badge_active(self, gen):
        """Active status badge."""
        assert gen._status_badge(TaskStatus.ACTIVE) == "IN PROGRESS"

    def test_status_badge_completed(self, gen):
        """Completed status badge."""
        assert gen._status_badge(TaskStatus.COMPLETED) == "DONE"

    def test_status_badge_failed(self, gen):
        """Failed status badge."""
        assert gen._status_badge(TaskStatus.FAILED) == "FAILED"

    def test_status_badge_archived(self, gen):
        """Archived status badge."""
        assert gen._status_badge(TaskStatus.ARCHIVED) == "ARCHIVED"


# ===========================================================================
# generate_summary MCP tool tests
# ===========================================================================


class TestGenerateSummaryTool:
    """Tests for the generate_summary MCP tool via the FastMCP server."""

    def test_auto_no_task_returns_index(self, server):
        """Auto mode with no active task returns a window index."""
        wm = get_window_manager()

        # Call the tool function directly via the server.
        tool = None
        for t in server._tool_manager._tools.values():
            if t.name == "generate_summary":
                tool = t
                break
        assert tool is not None, "generate_summary tool not registered"

        result = asyncio.get_event_loop().run_until_complete(
            tool.run({"ticket_id": "", "summary_type": "auto"})
        )
        data = _parse_tool_result(result)

        assert data["error"] is False
        assert data["summary_type"] == "index"
        assert "# Memory Window Index" in data["markdown"]

    def test_auto_with_active_task_returns_current(self, server):
        """Auto mode with an active task returns current summary."""
        wm = get_window_manager()
        wm.start_new_task("TEST-001", "Test task", "phase-1")

        tool = None
        for t in server._tool_manager._tools.values():
            if t.name == "generate_summary":
                tool = t
                break

        result = asyncio.get_event_loop().run_until_complete(
            tool.run({"ticket_id": "", "summary_type": "auto"})
        )
        data = _parse_tool_result(result)

        assert data["error"] is False
        assert data["summary_type"] == "current"
        assert "TEST-001" in data["markdown"]

    def test_auto_with_ticket_id_returns_task(self, server):
        """Auto mode with a ticket_id returns task summary."""
        wm = get_window_manager()
        wm.start_new_task("TEST-001", "Test task", "phase-1")

        tool = None
        for t in server._tool_manager._tools.values():
            if t.name == "generate_summary":
                tool = t
                break

        result = asyncio.get_event_loop().run_until_complete(
            tool.run({"ticket_id": "TEST-001", "summary_type": "auto"})
        )
        data = _parse_tool_result(result)

        assert data["error"] is False
        assert data["summary_type"] == "task"
        assert "# Task Summary: TEST-001" in data["markdown"]

    def test_explicit_task_type(self, server):
        """Explicit 'task' type returns task summary."""
        wm = get_window_manager()
        wm.start_new_task("TEST-002", "Another task", "phase-1")

        tool = None
        for t in server._tool_manager._tools.values():
            if t.name == "generate_summary":
                tool = t
                break

        result = asyncio.get_event_loop().run_until_complete(
            tool.run({"ticket_id": "TEST-002", "summary_type": "task"})
        )
        data = _parse_tool_result(result)

        assert data["error"] is False
        assert data["summary_type"] == "task"
        assert "TEST-002" in data["ticket_id"]

    def test_explicit_current_type(self, server):
        """Explicit 'current' type returns current summary."""
        wm = get_window_manager()
        wm.start_new_task("TEST-003", "Current task", "phase-2")

        tool = None
        for t in server._tool_manager._tools.values():
            if t.name == "generate_summary":
                tool = t
                break

        result = asyncio.get_event_loop().run_until_complete(
            tool.run({"ticket_id": "", "summary_type": "current"})
        )
        data = _parse_tool_result(result)

        assert data["error"] is False
        assert data["summary_type"] == "current"
        assert "# Current Task: TEST-003" in data["markdown"]

    def test_explicit_index_type(self, server):
        """Explicit 'index' type returns window index."""
        tool = None
        for t in server._tool_manager._tools.values():
            if t.name == "generate_summary":
                tool = t
                break

        result = asyncio.get_event_loop().run_until_complete(
            tool.run({"ticket_id": "", "summary_type": "index"})
        )
        data = _parse_tool_result(result)

        assert data["error"] is False
        assert data["summary_type"] == "index"
        assert "# Memory Window Index" in data["markdown"]

    def test_invalid_summary_type(self, server):
        """Invalid summary_type returns an error."""
        tool = None
        for t in server._tool_manager._tools.values():
            if t.name == "generate_summary":
                tool = t
                break

        result = asyncio.get_event_loop().run_until_complete(
            tool.run({"ticket_id": "", "summary_type": "invalid"})
        )
        data = _parse_tool_result(result)

        assert data["error"] is True
        assert "Invalid summary_type" in data["message"]

    def test_task_not_found_error(self, server):
        """Requesting a non-existent task returns an error."""
        tool = None
        for t in server._tool_manager._tools.values():
            if t.name == "generate_summary":
                tool = t
                break

        result = asyncio.get_event_loop().run_until_complete(
            tool.run({"ticket_id": "DOES-NOT-EXIST", "summary_type": "task"})
        )
        data = _parse_tool_result(result)

        assert data["error"] is True
        assert "not found" in data["message"]

    def test_current_type_no_active_task_error(self, server):
        """'current' type with no active task returns an error."""
        tool = None
        for t in server._tool_manager._tools.values():
            if t.name == "generate_summary":
                tool = t
                break

        result = asyncio.get_event_loop().run_until_complete(
            tool.run({"ticket_id": "", "summary_type": "current"})
        )
        data = _parse_tool_result(result)

        assert data["error"] is True
        assert "No active task" in data["message"]

    def test_markdown_length_in_response(self, server):
        """Response includes markdown_length field."""
        tool = None
        for t in server._tool_manager._tools.values():
            if t.name == "generate_summary":
                tool = t
                break

        result = asyncio.get_event_loop().run_until_complete(
            tool.run({"ticket_id": "", "summary_type": "index"})
        )
        data = _parse_tool_result(result)

        assert data["error"] is False
        assert data["markdown_length"] == len(data["markdown"])
        assert data["markdown_length"] > 0

    def test_completed_task_summary_via_tool(self, server):
        """Completed task summary is retrievable via the tool."""
        wm = get_window_manager()
        wm.start_new_task("DONE-001", "Completed task", "phase-1")
        task = wm.get_current_task()
        task.add_step(action="Wrote code", tool_used="Write")
        task.add_decision(decision="Used Pydantic", reasoning="Type safety")
        wm.save_current_task()
        wm.complete_current_task("All implemented and tested.")

        tool = None
        for t in server._tool_manager._tools.values():
            if t.name == "generate_summary":
                tool = t
                break

        result = asyncio.get_event_loop().run_until_complete(
            tool.run({"ticket_id": "DONE-001", "summary_type": "task"})
        )
        data = _parse_tool_result(result)

        assert data["error"] is False
        assert data["summary_type"] == "task"
        assert "DONE" in data["markdown"]
        assert "All implemented and tested." in data["markdown"]
        assert "Wrote code" in data["markdown"]
        assert "Used Pydantic" in data["markdown"]


# ===========================================================================
# Edge case and integration tests
# ===========================================================================


class TestEdgeCases:
    """Edge cases for markdown generation."""

    def test_long_title_in_index_truncated(self, gen):
        """Long titles in the window index are truncated."""
        task = _make_task(
            title="A" * 60,
            completed=True,
        )
        window = MemoryWindow(window_size=3)
        md = gen.generate_window_index(window, completed_tasks=[task])

        # Title should be truncated to 40 chars + "..."
        assert "A" * 40 + "..." in md

    def test_short_title_not_truncated(self, gen):
        """Short titles in the window index are not truncated."""
        task = _make_task(title="Short title", completed=True)
        window = MemoryWindow(window_size=3)
        md = gen.generate_window_index(window, completed_tasks=[task])

        assert "Short title" in md
        assert "Short title..." not in md

    def test_file_description_truncated(self, gen):
        """Long file descriptions are truncated in the table."""
        task = _make_task()
        task.record_file(
            path="src/long.py",
            action=FileAction.CREATED,
            description="X" * 100,
        )

        md = gen.generate_task_summary(task)

        # Description should be truncated to 80 chars + "..."
        assert "X" * 80 + "..." in md

    def test_output_is_valid_string(self, gen):
        """All generators return non-empty strings."""
        task = _make_task(with_steps=2, with_files=1, with_decisions=1)
        window = MemoryWindow()

        md1 = gen.generate_task_summary(task)
        md2 = gen.generate_current_summary(task)
        md3 = gen.generate_window_index(window)

        assert isinstance(md1, str) and len(md1) > 0
        assert isinstance(md2, str) and len(md2) > 0
        assert isinstance(md3, str) and len(md3) > 0

    def test_multiple_file_actions_tracked(self, gen):
        """Multiple actions on the same file are tracked in history."""
        task = _make_task()
        task.record_file("src/main.py", FileAction.CREATED, "Created file")
        task.record_file("src/main.py", FileAction.MODIFIED, "Added function")
        task.record_file("src/main.py", FileAction.MODIFIED, "Fixed bug")

        md = gen.generate_task_summary(task)

        assert "3 actions" in md
        assert "src/main.py" in md

    def test_generate_summary_returns_string_output(self, gen):
        """The markdown output is a plain string, not wrapped in any object."""
        task = _make_task(with_steps=3, with_files=2, with_branches=1, with_decisions=1)
        md = gen.generate_task_summary(task)
        assert isinstance(md, str)
        # Verify it starts with a markdown header
        assert md.startswith("# Task Summary:")
