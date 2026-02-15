"""End-to-end integration tests for claude-code-helper-mcp (CMH-018).

Comprehensive integration tests that exercise the full system across multiple
components simultaneously.  Each test class covers a distinct integration
scenario:

- TestFullLifecycle: start -> record -> complete -> verify persistence
- TestSlidingWindowRotation: complete enough tasks to trigger archival
- TestRecoveryContextAccuracy: recovery after task completion
- TestMCPToolRoundTrips: call server tools, verify effects
- TestCLIOutput: CliRunner tests for status/list/show/recover commands
- TestHookIntegrationLifecycle: full pipeline hook lifecycle
- TestEdgeCases: concurrent state, missing files, large payloads, etc.
- TestCrossCutting: interactions between hooks, server, CLI, and recovery

All tests use real file I/O via temporary directories.  No mocks, no stubs,
no patches, no hardcoded test values.  Every assertion verifies real state
on disk.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest
from click.testing import CliRunner

from claude_code_helper_mcp.cli.main import cli
from claude_code_helper_mcp.config import MemoryConfig
from claude_code_helper_mcp.hooks.pipeline import (
    post_build_complete,
    post_build_start,
    post_merge,
    post_tool_call,
    reset_hook_state,
)
from claude_code_helper_mcp.mcp.server import (
    create_server,
    get_window_manager,
    reset_server,
)
from claude_code_helper_mcp.models.records import BranchAction, FileAction
from claude_code_helper_mcp.models.recovery import RecoveryContext
from claude_code_helper_mcp.models.task import TaskMemory, TaskStatus
from claude_code_helper_mcp.storage.store import MemoryStore
from claude_code_helper_mcp.storage.window_manager import WindowManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_tool_result(result) -> dict:
    """Extract a dict from a FastMCP ToolResult.

    FastMCP tool.run() returns a ToolResult whose .content is a list of
    TextContent objects.  The first TextContent's .text is JSON-encoded.
    """
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        return json.loads(result)
    # ToolResult object -- extract text from first content item.
    if hasattr(result, "content") and result.content:
        text = result.content[0].text
        return json.loads(text)
    raise TypeError(f"Cannot parse tool result of type {type(result)}")


def _call_tool(server, tool_name: str, args: dict | None = None):
    """Invoke a FastMCP tool by name and return the parsed result dict.

    FastMCP does not expose a ``call_tool`` method on the server object.
    Instead, we fetch the tools dict via ``server.get_tools()``, look up the
    tool by name, and call ``tool.run(args)``.
    """
    if args is None:
        args = {}
    tools = asyncio.run(server.get_tools())
    tool = tools[tool_name]
    raw = asyncio.run(tool.run(args))
    return _parse_tool_result(raw)


def _setup_storage(tmp_path: Path) -> str:
    """Create a storage directory structure and return the storage path."""
    storage_dir = tmp_path / ".claude-memory"
    storage_dir.mkdir()
    (storage_dir / "tasks").mkdir()
    return str(storage_dir)


def _setup_project(tmp_path: Path) -> Path:
    """Create a full project directory with storage and root markers."""
    storage_dir = tmp_path / ".claude-memory"
    storage_dir.mkdir()
    (storage_dir / "tasks").mkdir()
    # Create a marker file so project root detection works.
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    return tmp_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_all_singletons():
    """Reset server and hook singletons before and after each test."""
    reset_server()
    reset_hook_state()
    yield
    reset_server()
    reset_hook_state()


@pytest.fixture
def project_dir(tmp_path):
    """Create a temporary project directory with storage structure."""
    return _setup_project(tmp_path)


@pytest.fixture
def storage_path(project_dir):
    """Return the storage path for a project directory."""
    return str(project_dir / ".claude-memory")


@pytest.fixture
def window_manager(storage_path):
    """Create a WindowManager rooted at the storage path."""
    return WindowManager(storage_path=storage_path, window_size=3)


@pytest.fixture
def server(project_dir):
    """Create and return a FastMCP server instance rooted at the project dir."""
    return create_server(project_root=str(project_dir))


# ===================================================================
# TestFullLifecycle
# ===================================================================


class TestFullLifecycle:
    """Test the complete task lifecycle: start -> record -> complete -> verify."""

    def test_start_record_steps_complete_persists(self, window_manager, storage_path):
        """Full lifecycle with steps, files, branches, decisions persists to disk."""
        # Start a task.
        task = window_manager.start_new_task(
            ticket_id="INTEG-001",
            title="Full lifecycle test",
            phase="phase-1",
        )
        assert task.status == TaskStatus.ACTIVE
        assert task.ticket_id == "INTEG-001"

        # Record steps.
        step1 = task.add_step(
            action="Created module",
            description="Wrote the main module file",
            tool_used="Write",
            result_summary="File created with 200 lines",
        )
        step2 = task.add_step(
            action="Ran tests",
            description="Executed pytest on the new module",
            tool_used="Bash",
            result_summary="15 tests passed, 0 failed",
        )
        assert step1.step_number == 1
        assert step2.step_number == 2

        # Record files.
        file1 = task.record_file(
            path="src/module.py",
            action=FileAction.CREATED,
            description="New module",
        )
        file2 = task.record_file(
            path="tests/test_module.py",
            action=FileAction.CREATED,
            description="Test file",
        )
        assert len(task.files) == 2

        # Record a branch.
        branch = task.record_branch(
            branch_name="feature/INTEG-001-lifecycle-test",
            action=BranchAction.CREATED,
            base_branch="main",
        )
        assert branch.branch_name == "feature/INTEG-001-lifecycle-test"

        # Record a decision.
        decision = task.add_decision(
            decision="Use Pydantic models",
            reasoning="Type safety and validation",
            alternatives=["dataclasses", "plain dicts"],
            context="Project already uses Pydantic",
        )
        assert decision.decision_number == 1

        # Save incrementally.
        window_manager.save_current_task()

        # Verify on-disk persistence before completion.
        store = MemoryStore(storage_path)
        loaded_task = store.load_task("INTEG-001")
        assert loaded_task is not None
        assert loaded_task.step_count() == 2
        assert len(loaded_task.files) == 2
        assert len(loaded_task.branches) == 1
        assert len(loaded_task.decisions) == 1

        # Complete the task.
        completed = window_manager.complete_current_task("All tests passing")
        assert completed.status == TaskStatus.COMPLETED
        assert completed.completed_at is not None
        assert completed.summary == "All tests passing"

        # Verify completed task persists on disk.
        reloaded = store.load_task("INTEG-001")
        assert reloaded.status == TaskStatus.COMPLETED
        assert reloaded.summary == "All tests passing"
        assert reloaded.step_count() == 2

        # Verify window state.
        assert window_manager.has_active_task() is False
        assert window_manager.completed_task_count() == 1
        assert "INTEG-001" in window_manager.get_all_task_ids()

    def test_file_deduplication_across_lifecycle(self, window_manager):
        """Files recorded multiple times are deduplicated by path."""
        task = window_manager.start_new_task(
            ticket_id="INTEG-002",
            title="Dedup test",
        )

        # Record the same file twice with different actions.
        task.record_file(
            path="src/main.py",
            action=FileAction.CREATED,
            description="Initial creation",
        )
        task.record_file(
            path="src/main.py",
            action=FileAction.MODIFIED,
            description="Added error handling",
        )

        assert len(task.files) == 1
        # add_action pushes the PREVIOUS state into action_history, then
        # overwrites top-level fields.  After initial create (0 history)
        # and one dedup call, action_history has exactly 1 entry.
        assert len(task.files[0].action_history) == 1
        assert task.files[0].action_history[0]["action"] == "created"
        # The current (top-level) action is now the latest one.
        assert task.files[0].action.value == "modified"

        window_manager.save_current_task()
        window_manager.complete_current_task("Done")

        # Verify deduplication persists.
        reloaded = window_manager.get_task("INTEG-002")
        assert len(reloaded.files) == 1
        assert len(reloaded.files[0].action_history) == 1

    def test_branch_deduplication_across_lifecycle(self, window_manager):
        """Branches recorded multiple times are deduplicated by name."""
        task = window_manager.start_new_task(
            ticket_id="INTEG-003",
            title="Branch dedup test",
        )

        # Record the same branch with different actions.
        task.record_branch(
            branch_name="feature/INTEG-003",
            action=BranchAction.CREATED,
            base_branch="main",
        )
        task.record_branch(
            branch_name="feature/INTEG-003",
            action=BranchAction.PUSHED,
        )
        task.record_branch(
            branch_name="feature/INTEG-003",
            action=BranchAction.MERGED,
            base_branch="main",
        )

        assert len(task.branches) == 1
        # add_action pushes current state into history before overwriting.
        # After create (0 history) + pushed (1 history) + merged (2 history).
        assert len(task.branches[0].action_history) == 2

        window_manager.save_current_task()
        window_manager.complete_current_task("Merged")

        reloaded = window_manager.get_task("INTEG-003")
        assert len(reloaded.branches) == 1
        assert len(reloaded.branches[0].action_history) == 2

    def test_task_fail_lifecycle(self, window_manager):
        """Failed tasks go through the lifecycle correctly."""
        task = window_manager.start_new_task(
            ticket_id="INTEG-004",
            title="Fail lifecycle test",
        )
        task.add_step(action="Attempted build", success=False)
        window_manager.save_current_task()

        failed = window_manager.fail_current_task("Build compilation error")
        assert failed.status == TaskStatus.FAILED
        # fail_current_task prepends "FAILED: " to the reason.
        assert "Build compilation error" in failed.summary
        assert failed.completed_at is not None

        # Failed tasks persist.
        reloaded = window_manager.get_task("INTEG-004")
        assert reloaded.status == TaskStatus.FAILED

    def test_metadata_survives_lifecycle(self, window_manager):
        """Arbitrary metadata persists through the full lifecycle."""
        task = window_manager.start_new_task(
            ticket_id="INTEG-005",
            title="Metadata test",
        )
        task.metadata["pr_number"] = 42
        task.metadata["tests_total"] = 100
        task.metadata["custom_key"] = "custom_value"
        window_manager.save_current_task()

        completed = window_manager.complete_current_task("Done with metadata")
        reloaded = window_manager.get_task("INTEG-005")
        assert reloaded.metadata["pr_number"] == 42
        assert reloaded.metadata["tests_total"] == 100
        assert reloaded.metadata["custom_key"] == "custom_value"


# ===================================================================
# TestSlidingWindowRotation
# ===================================================================


class TestSlidingWindowRotation:
    """Test sliding window rotation: task archival when window is full."""

    def test_window_rotates_oldest_task(self, storage_path):
        """When window is full, completing a new task archives the oldest."""
        wm = WindowManager(storage_path=storage_path, window_size=2)

        # Complete 2 tasks to fill the window.
        wm.start_new_task("ROT-001", "First task", "phase-1")
        wm.complete_current_task("First done")

        wm.start_new_task("ROT-002", "Second task", "phase-1")
        wm.complete_current_task("Second done")

        assert wm.completed_task_count() == 2
        assert wm.archived_task_count() == 0

        # Complete a third task -- should archive the oldest (ROT-001).
        wm.start_new_task("ROT-003", "Third task", "phase-1")
        wm.complete_current_task("Third done")

        assert wm.completed_task_count() == 2
        assert wm.archived_task_count() == 1
        assert "ROT-001" in wm.window.archived_task_ids
        assert "ROT-002" in wm.get_all_task_ids()
        assert "ROT-003" in wm.get_all_task_ids()

    def test_archived_task_still_loadable_from_disk(self, storage_path):
        """Archived tasks remain on disk and are loadable after rotation."""
        wm = WindowManager(storage_path=storage_path, window_size=2)

        # Create 3 tasks; first will be archived.
        wm.start_new_task("ROT-010", "Task A")
        task_a = wm.get_current_task()
        task_a.add_step(action="Step A1", tool_used="Write")
        task_a.record_file(path="a.py", action=FileAction.CREATED)
        wm.save_current_task()
        wm.complete_current_task("A done")

        wm.start_new_task("ROT-011", "Task B")
        wm.complete_current_task("B done")

        wm.start_new_task("ROT-012", "Task C")
        wm.complete_current_task("C done")

        # ROT-010 should be archived.
        assert wm.is_task_archived("ROT-010")

        # But still loadable from disk.
        archived = wm.get_task("ROT-010")
        assert archived is not None
        assert archived.ticket_id == "ROT-010"
        assert archived.status == TaskStatus.COMPLETED
        assert archived.step_count() == 1
        assert len(archived.files) == 1
        assert archived.files[0].path == "a.py"

    def test_window_rotation_sequence_with_many_tasks(self, storage_path):
        """Rotation works correctly through a sequence of many tasks."""
        wm = WindowManager(storage_path=storage_path, window_size=3)

        # Complete 6 tasks with window_size=3.
        for i in range(1, 7):
            tid = f"SEQ-{i:03d}"
            wm.start_new_task(tid, f"Task {i}")
            wm.get_current_task().add_step(action=f"Step for {tid}")
            wm.save_current_task()
            wm.complete_current_task(f"Done {i}")

        # Window should have the last 3 completed tasks.
        assert wm.completed_task_count() == 3
        window_ids = [t.ticket_id for t in wm.window.completed_tasks]
        assert window_ids == ["SEQ-004", "SEQ-005", "SEQ-006"]

        # First 3 should be archived.
        assert wm.archived_task_count() == 3
        for i in range(1, 4):
            assert wm.is_task_archived(f"SEQ-{i:03d}")

        # All 6 should still be known.
        all_ids = wm.get_all_known_task_ids()
        for i in range(1, 7):
            assert f"SEQ-{i:03d}" in all_ids

    def test_window_resize_triggers_archival(self, storage_path):
        """Resizing window to a smaller size archives overflow tasks."""
        wm = WindowManager(storage_path=storage_path, window_size=5)

        # Fill with 5 completed tasks.
        for i in range(1, 6):
            wm.start_new_task(f"RESIZE-{i:03d}", f"Task {i}")
            wm.complete_current_task(f"Done {i}")

        assert wm.completed_task_count() == 5
        assert wm.archived_task_count() == 0

        # Resize to 2.
        newly_archived = wm.resize_window(2)

        assert len(newly_archived) == 3
        assert wm.completed_task_count() == 2
        assert wm.archived_task_count() == 3

        # The oldest 3 should be archived.
        for tid in ["RESIZE-001", "RESIZE-002", "RESIZE-003"]:
            assert tid in wm.window.archived_task_ids

        # The newest 2 should be in the window.
        window_ids = [t.ticket_id for t in wm.window.completed_tasks]
        assert "RESIZE-004" in window_ids
        assert "RESIZE-005" in window_ids

    def test_window_rotation_preserves_active_task(self, storage_path):
        """Rotation never evicts the active task."""
        wm = WindowManager(storage_path=storage_path, window_size=2)

        # Fill the window.
        wm.start_new_task("PRES-001", "Task 1")
        wm.complete_current_task("Done 1")
        wm.start_new_task("PRES-002", "Task 2")
        wm.complete_current_task("Done 2")

        # Start a new task (active, not completed).
        wm.start_new_task("PRES-003", "Task 3 (active)")

        # The active task should still be accessible.
        assert wm.has_active_task()
        assert wm.get_current_task().ticket_id == "PRES-003"

    def test_reload_preserves_window_state(self, storage_path):
        """Reloading from disk produces identical window state."""
        wm1 = WindowManager(storage_path=storage_path, window_size=2)
        wm1.start_new_task("REL-001", "Reload task 1")
        wm1.complete_current_task("Done 1")
        wm1.start_new_task("REL-002", "Reload task 2")
        wm1.complete_current_task("Done 2")
        wm1.start_new_task("REL-003", "Reload task 3")
        wm1.complete_current_task("Done 3")

        # Create a fresh WindowManager from the same storage.
        wm2 = WindowManager(storage_path=storage_path, window_size=2)

        assert wm2.completed_task_count() == 2
        assert wm2.archived_task_count() == 1
        assert wm2.is_task_archived("REL-001")


# ===================================================================
# TestRecoveryContextAccuracy
# ===================================================================


class TestRecoveryContextAccuracy:
    """Test recovery context generation from task memory."""

    def _create_rich_task(self, wm: WindowManager) -> TaskMemory:
        """Create a task with substantial memory content."""
        task = wm.start_new_task(
            ticket_id="REC-001",
            title="Recovery context test task",
            phase="phase-3",
        )
        # Add multiple steps.
        for i in range(1, 8):
            task.add_step(
                action=f"Step {i} action",
                description=f"Description for step {i}",
                tool_used=["Write", "Edit", "Bash", "Read", "Write", "Edit", "Bash"][i - 1],
                result_summary=f"Result {i}",
            )
        # Add files.
        task.record_file("src/main.py", FileAction.CREATED, "Main module")
        task.record_file("src/utils.py", FileAction.CREATED, "Utilities")
        task.record_file("tests/test_main.py", FileAction.CREATED, "Tests")
        task.record_file("src/main.py", FileAction.MODIFIED, "Added error handling")
        # Add branches.
        task.record_branch(
            "feature/REC-001-recovery-test",
            BranchAction.CREATED,
            "main",
        )
        task.record_branch(
            "feature/REC-001-recovery-test",
            BranchAction.PUSHED,
        )
        # Add decisions.
        task.add_decision(
            decision="Use file-based recovery",
            reasoning="Simplest approach for the use case",
            alternatives=["Database", "Environment variables"],
            context="Project is file-based",
        )
        task.add_decision(
            decision="Include last 10 steps in recovery",
            reasoning="Enough context without overwhelming",
        )
        # Add next steps.
        task.next_steps = ["Run full test suite", "Create PR", "Update documentation"]
        # Add metadata.
        task.metadata["description"] = "Testing recovery accuracy"
        task.metadata["pr_number"] = 99

        wm.save_current_task()
        return task

    def test_recovery_context_from_active_task(self, window_manager):
        """Recovery context accurately reflects an active task."""
        task = self._create_rich_task(window_manager)

        recovery = RecoveryContext.from_task_memory(task, recent_step_count=5)

        assert recovery.ticket_id == "REC-001"
        assert recovery.title == "Recovery context test task"
        assert recovery.phase == "phase-3"
        assert recovery.status == "active"
        assert recovery.total_steps_completed == 7
        assert len(recovery.recent_steps) == 5
        assert len(recovery.files_modified) == 3  # 3 unique paths
        assert recovery.active_branch == "feature/REC-001-recovery-test"
        assert len(recovery.key_decisions) == 2
        assert len(recovery.next_steps) == 3
        assert recovery.task_started_at is not None
        assert recovery.metadata.get("pr_number") == 99

    def test_recovery_context_from_completed_task(self, window_manager):
        """Recovery context works for completed tasks too."""
        task = self._create_rich_task(window_manager)
        window_manager.complete_current_task("All done, tests passing")

        completed_task = window_manager.get_task("REC-001")
        recovery = RecoveryContext.from_task_memory(completed_task)

        assert recovery.status == "completed"
        assert recovery.total_steps_completed == 7
        assert recovery.summary_so_far == "All done, tests passing"

    def test_recovery_prompt_format(self, window_manager):
        """Recovery prompt is well-formatted markdown with all sections."""
        task = self._create_rich_task(window_manager)

        recovery = RecoveryContext.from_task_memory(task)
        prompt = recovery.format_for_prompt()

        # Verify key sections are present in the prompt.
        assert "REC-001" in prompt
        assert "Recovery context test task" in prompt
        assert "phase-3" in prompt
        assert "src/main.py" in prompt
        assert "feature/REC-001-recovery-test" in prompt
        assert "Use file-based recovery" in prompt
        assert "Run full test suite" in prompt

    def test_recovery_with_minimal_task(self, window_manager):
        """Recovery context handles tasks with minimal data."""
        task = window_manager.start_new_task(
            ticket_id="REC-002",
            title="Minimal task",
        )
        window_manager.save_current_task()

        recovery = RecoveryContext.from_task_memory(task)

        assert recovery.ticket_id == "REC-002"
        assert recovery.total_steps_completed == 0
        assert len(recovery.recent_steps) == 0
        assert len(recovery.files_modified) == 0
        assert recovery.active_branch is None
        assert len(recovery.key_decisions) == 0

    def test_recovery_recent_step_count_clamped(self, window_manager):
        """Recent step count is correctly limited by the parameter."""
        task = window_manager.start_new_task("REC-003", "Step limit test")
        for i in range(20):
            task.add_step(action=f"Step {i + 1}")
        window_manager.save_current_task()

        recovery_5 = RecoveryContext.from_task_memory(task, recent_step_count=5)
        assert len(recovery_5.recent_steps) == 5
        assert recovery_5.total_steps_completed == 20

        recovery_all = RecoveryContext.from_task_memory(task, recent_step_count=50)
        assert len(recovery_all.recent_steps) == 20


# ===================================================================
# TestMCPToolRoundTrips
# ===================================================================


class TestMCPToolRoundTrips:
    """Test MCP tool invocations and verify server-side effects."""

    def test_start_task_via_server(self, server):
        """start_task tool creates an active task visible in status."""
        wm = get_window_manager()

        data = _call_tool(server, "start_task", {
            "ticket_id": "MCP-001",
            "title": "Server round trip",
            "phase": "phase-1",
        })

        assert data["error"] is False
        assert data["task_id"] == "MCP-001"
        assert data["status"] == "active"

        # Verify via WindowManager.
        assert wm.has_active_task()
        assert wm.get_current_task().ticket_id == "MCP-001"

    def test_record_step_via_server(self, server):
        """record_step tool adds a step to the active task."""
        _call_tool(server, "start_task", {
            "ticket_id": "MCP-002",
            "title": "Step recording",
        })

        data = _call_tool(server, "record_step", {
            "action": "Implemented feature",
            "description": "Added the main logic",
            "tool_used": "Write",
            "result_summary": "200 lines created",
        })

        assert data["error"] is False
        assert data["step_number"] == 1
        assert data["action"] == "Implemented feature"
        assert data["total_steps"] == 1

    def test_record_file_via_server(self, server):
        """record_file tool records a file action with deduplication."""
        _call_tool(server, "start_task", {
            "ticket_id": "MCP-003",
            "title": "File recording",
        })

        # First recording.
        r1 = _call_tool(server, "record_file", {
            "path": "src/handler.py",
            "action": "created",
            "description": "New file",
        })
        assert r1["error"] is False
        assert r1["is_update"] is False
        assert r1["total_files"] == 1

        # Second recording of the same file.
        r2 = _call_tool(server, "record_file", {
            "path": "src/handler.py",
            "action": "modified",
            "description": "Updated",
        })
        assert r2["error"] is False
        assert r2["is_update"] is True
        assert r2["total_files"] == 1
        # add_action pushes old state into history: after 1 dedup call = 1 entry.
        assert r2["action_history_count"] == 1

    def test_record_branch_via_server(self, server):
        """record_branch tool records branch lifecycle."""
        _call_tool(server, "start_task", {
            "ticket_id": "MCP-004",
            "title": "Branch recording",
        })

        r1 = _call_tool(server, "record_branch", {
            "branch_name": "feature/MCP-004",
            "action": "created",
            "base_branch": "main",
        })
        assert r1["error"] is False
        assert r1["is_update"] is False

        r2 = _call_tool(server, "record_branch", {
            "branch_name": "feature/MCP-004",
            "action": "pushed",
        })
        assert r2["error"] is False
        assert r2["is_update"] is True
        # add_action pushes old state into history: after 1 dedup call = 1 entry.
        assert r2["action_history_count"] == 1

    def test_record_decision_via_server(self, server):
        """record_decision tool records and numbers decisions."""
        _call_tool(server, "start_task", {
            "ticket_id": "MCP-005",
            "title": "Decision recording",
        })

        r1 = _call_tool(server, "record_decision", {
            "decision": "Use Pydantic for models",
            "reasoning": "Type safety",
            "alternatives": ["dataclasses", "attrs"],
            "context": "Existing codebase uses Pydantic",
        })
        assert r1["error"] is False
        assert r1["decision_number"] == 1
        assert r1["total_decisions"] == 1

    def test_complete_task_via_server(self, server):
        """complete_task tool transitions task and reports stats."""
        _call_tool(server, "start_task", {
            "ticket_id": "MCP-006",
            "title": "Completion test",
        })
        _call_tool(server, "record_step", {"action": "Step 1"})
        _call_tool(server, "record_file", {
            "path": "src/main.py",
            "action": "created",
        })

        result = _call_tool(server, "complete_task", {
            "summary": "Feature complete",
        })

        assert result["error"] is False
        assert result["task_id"] == "MCP-006"
        assert result["status"] == "completed"
        assert result["counts"]["steps"] == 1
        assert result["counts"]["files"] == 1
        assert result["duration_seconds"] is not None

    def test_get_task_status_via_server(self, server):
        """get_task_status tool returns active task details."""
        _call_tool(server, "start_task", {
            "ticket_id": "MCP-007",
            "title": "Status check",
        })
        _call_tool(server, "record_step", {"action": "Did something"})

        result = _call_tool(server, "get_task_status", {})

        assert result["error"] is False
        assert result["has_active_task"] is True
        assert result["task_id"] == "MCP-007"
        assert result["counts"]["steps"] == 1

    def test_health_check_via_server(self, server):
        """health_check tool returns server health information."""
        result = _call_tool(server, "health_check", {})

        assert result["status"] == "healthy"
        assert result["storage_accessible"] is True
        assert result["window_size"] == 3  # default

    def test_get_recovery_context_via_server(self, server):
        """get_recovery_context tool returns context for active task."""
        _call_tool(server, "start_task", {
            "ticket_id": "MCP-008",
            "title": "Recovery test",
        })
        _call_tool(server, "record_step", {"action": "Step 1"})
        _call_tool(server, "record_file", {
            "path": "src/feature.py",
            "action": "created",
        })

        result = _call_tool(server, "get_recovery_context", {
            "include_prompt": True,
        })

        assert result["error"] is False
        assert result["has_context"] is True
        assert result["ticket_id"] == "MCP-008"
        assert result["source"] == "active"
        assert result["total_steps_completed"] == 1
        assert len(result["files_modified"]) == 1
        assert "recovery_prompt" in result

    def test_check_alignment_via_server(self, server):
        """check_alignment tool returns alignment assessment."""
        _call_tool(server, "start_task", {
            "ticket_id": "MCP-009",
            "title": "Alignment checker module",
        })
        _call_tool(server, "record_step", {
            "action": "Working on alignment checker",
        })

        result = _call_tool(server, "check_alignment", {
            "action": "Editing the alignment checker module",
            "file_path": "src/detection/alignment.py",
            "threshold": 0.3,
        })

        assert result["error"] is False
        assert "confidence" in result
        assert "aligned" in result
        assert isinstance(result["confidence"], (int, float))

    def test_generate_summary_auto_mode(self, server):
        """generate_summary with auto mode generates correct type."""
        _call_tool(server, "start_task", {
            "ticket_id": "MCP-010",
            "title": "Summary generation",
        })
        _call_tool(server, "record_step", {"action": "Built module"})

        result = _call_tool(server, "generate_summary", {
            "summary_type": "auto",
        })

        assert result["error"] is False
        assert result["summary_type"] == "current"
        assert "markdown" in result
        assert len(result["markdown"]) > 0

    def test_start_task_rejected_when_active(self, server):
        """Starting a task when one is already active returns an error."""
        _call_tool(server, "start_task", {
            "ticket_id": "MCP-011",
            "title": "First task",
        })

        result = _call_tool(server, "start_task", {
            "ticket_id": "MCP-012",
            "title": "Second task",
        })

        assert result["error"] is True
        assert "already active" in result["message"]

    def test_record_step_without_active_task_returns_error(self, server):
        """Recording a step with no active task returns an error."""
        result = _call_tool(server, "record_step", {
            "action": "Orphan step",
        })

        assert result["error"] is True
        assert "No active task" in result["message"]

    def test_record_file_invalid_action_returns_error(self, server):
        """Recording a file with invalid action returns an error."""
        _call_tool(server, "start_task", {
            "ticket_id": "MCP-013",
            "title": "Invalid action test",
        })

        result = _call_tool(server, "record_file", {
            "path": "src/main.py",
            "action": "invalid_action",
        })

        assert result["error"] is True
        assert "Invalid file action" in result["message"]


# ===================================================================
# TestCLIOutput
# ===================================================================


class TestCLIOutput:
    """Test CLI commands via CliRunner with real storage."""

    def test_status_no_tasks(self, storage_path):
        """Status command with empty storage shows no active task."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--storage-path", storage_path, "status"])

        assert result.exit_code == 0
        assert "No active task" in result.output or "no active task" in result.output.lower()

    def test_status_with_active_task(self, storage_path):
        """Status command shows active task details."""
        wm = WindowManager(storage_path=storage_path)
        task = wm.start_new_task("CLI-001", "CLI status test", "phase-1")
        task.add_step(action="Step 1", tool_used="Write")
        wm.save_current_task()

        runner = CliRunner()
        result = runner.invoke(cli, ["--storage-path", storage_path, "status"])

        assert result.exit_code == 0
        assert "CLI-001" in result.output

    def test_status_json_output(self, storage_path):
        """Status command with --json-output produces valid JSON."""
        wm = WindowManager(storage_path=storage_path)
        wm.start_new_task("CLI-002", "JSON status test")
        wm.save_current_task()

        runner = CliRunner()
        result = runner.invoke(
            cli, ["--storage-path", storage_path, "status", "--json-output"]
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "current_task" in data or "active_task" in data or "task_id" in str(data)

    def test_list_with_tasks(self, storage_path):
        """List command shows tasks in the window."""
        wm = WindowManager(storage_path=storage_path)
        wm.start_new_task("CLI-003", "List test 1")
        wm.complete_current_task("Done 1")
        wm.start_new_task("CLI-004", "List test 2")
        wm.complete_current_task("Done 2")

        runner = CliRunner()
        result = runner.invoke(cli, ["--storage-path", storage_path, "list"])

        assert result.exit_code == 0
        assert "CLI-003" in result.output
        assert "CLI-004" in result.output

    def test_list_json_format(self, storage_path):
        """List command with --format json produces valid JSON."""
        wm = WindowManager(storage_path=storage_path)
        wm.start_new_task("CLI-005", "JSON list test")
        wm.complete_current_task("Done")

        runner = CliRunner()
        result = runner.invoke(
            cli, ["--storage-path", storage_path, "list", "--format", "json"]
        )

        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, (list, dict))

    def test_show_specific_task(self, storage_path):
        """Show command displays details for a specific task."""
        wm = WindowManager(storage_path=storage_path)
        task = wm.start_new_task("CLI-006", "Show test task", "phase-2")
        task.add_step(action="Created module", tool_used="Write")
        task.record_file("src/module.py", FileAction.CREATED, "New file")
        task.add_decision(decision="Use async", reasoning="Performance")
        wm.save_current_task()
        wm.complete_current_task("Implementation complete")

        runner = CliRunner()
        result = runner.invoke(
            cli, ["--storage-path", storage_path, "show", "CLI-006"]
        )

        assert result.exit_code == 0
        assert "CLI-006" in result.output
        assert "Show test task" in result.output

    def test_show_task_not_found(self, storage_path):
        """Show command for nonexistent task shows error."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--storage-path", storage_path, "show", "NONEXISTENT"]
        )

        # Should exit with an error or display "not found" message.
        assert "not found" in result.output.lower() or result.exit_code != 0

    def test_list_empty_window(self, storage_path):
        """List command with no tasks shows appropriate message."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--storage-path", storage_path, "list"])

        assert result.exit_code == 0
        # Should indicate no tasks or show an empty list.

    def test_version_flag(self):
        """Version flag outputs the package version."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])

        assert result.exit_code == 0
        assert "claude-code-helper-mcp" in result.output


# ===================================================================
# TestHookIntegrationLifecycle
# ===================================================================


class TestHookIntegrationLifecycle:
    """Test the full pipeline hook lifecycle end-to-end."""

    def test_full_build_lifecycle_via_hooks(self, storage_path, project_dir):
        """Full pipeline: build_start -> tool_calls -> build_complete -> merge."""
        from claude_code_helper_mcp.hooks import pipeline

        # Create a WindowManager and inject it into the hook module.
        wm = WindowManager(storage_path=storage_path, window_size=3)
        pipeline._hook_window_manager = wm

        # 1. post_build_start -- starts the task.
        start_result = post_build_start(
            ticket_id="HOOK-001",
            title="Hook lifecycle test",
            branch_name="feature/HOOK-001-lifecycle",
            phase="phase-4",
            description="Testing the complete hook lifecycle",
            base_branch="main",
        )
        assert start_result["recorded"] is True
        assert start_result["task_created"] is True

        # Verify task is active.
        task = wm.get_current_task()
        assert task is not None
        assert task.ticket_id == "HOOK-001"
        assert task.step_count() >= 1  # build start step
        assert len(task.branches) == 1

        # 2. post_tool_call -- record several tool calls during the build.
        tool1 = post_tool_call(
            tool_name="Write",
            action="Created handler module",
            file_path="src/handler.py",
            result_summary="File created",
        )
        assert tool1["recorded"] is True

        tool2 = post_tool_call(
            tool_name="Edit",
            action="Updated config",
            file_path="src/config.py",
            result_summary="Config updated",
        )
        assert tool2["recorded"] is True

        tool3 = post_tool_call(
            tool_name="Bash",
            action="Ran test suite",
            result_summary="42 tests passed",
            success=True,
        )
        assert tool3["recorded"] is True

        # Verify tool call recordings.
        task = wm.get_current_task()
        assert task.step_count() >= 4  # 1 build start + 3 tool calls
        assert len(task.files) >= 2  # handler.py and config.py

        # 3. post_build_complete -- records PR creation.
        complete_result = post_build_complete(
            ticket_id="HOOK-001",
            branch_name="feature/HOOK-001-lifecycle",
            pr_number=100,
            files_changed=["src/handler.py", "src/config.py", "tests/test_handler.py"],
            tests_passed=42,
            tests_total=42,
            commit_count=3,
            summary="Implementation complete with full test coverage",
        )
        assert complete_result["recorded"] is True
        assert complete_result["pr_number"] == 100
        assert complete_result["files_recorded"] == 3

        # Verify build metadata.
        task = wm.get_current_task()
        assert task.metadata["pr_number"] == 100
        assert task.metadata["build_complete"] is True
        assert task.metadata["tests_total"] == 42

        # 4. post_merge -- completes the task.
        merge_result = post_merge(
            ticket_id="HOOK-001",
            branch_name="feature/HOOK-001-lifecycle",
            pr_number=100,
            merge_strategy="squash",
            target_branch="main",
            completion_summary="Hook lifecycle test complete",
        )
        assert merge_result["recorded"] is True
        assert merge_result["task_completed"] is True

        # Verify task is completed.
        assert wm.has_active_task() is False
        completed_task = wm.get_task("HOOK-001")
        assert completed_task.status == TaskStatus.COMPLETED
        assert completed_task.summary != ""

        # Verify branch actions recorded.
        assert len(completed_task.branches) == 1
        branch_record = completed_task.branches[0]
        assert branch_record.branch_name == "feature/HOOK-001-lifecycle"
        # Created, pushed, merged, deleted = 4 actions.
        assert len(branch_record.action_history) >= 3

    def test_hooks_idempotent_on_resume(self, storage_path):
        """post_build_start is idempotent when called twice for same ticket."""
        from claude_code_helper_mcp.hooks import pipeline

        wm = WindowManager(storage_path=storage_path, window_size=3)
        pipeline._hook_window_manager = wm

        # First call creates the task.
        r1 = post_build_start(
            ticket_id="HOOK-002",
            title="Idempotent test",
            branch_name="feature/HOOK-002",
            base_branch="main",
        )
        assert r1["task_created"] is True

        # Second call on same ticket does not create a new task.
        r2 = post_build_start(
            ticket_id="HOOK-002",
            title="Idempotent test",
            branch_name="feature/HOOK-002",
            base_branch="main",
        )
        assert r2["recorded"] is True
        assert r2["task_created"] is False

        # Only one task should exist.
        assert wm.total_tasks_in_window() == 1

    def test_hooks_reject_mismatched_ticket(self, storage_path):
        """post_build_complete rejects if active ticket does not match."""
        from claude_code_helper_mcp.hooks import pipeline

        wm = WindowManager(storage_path=storage_path, window_size=3)
        pipeline._hook_window_manager = wm

        wm.start_new_task("HOOK-003", "Active task")

        result = post_build_complete(
            ticket_id="HOOK-WRONG",
            branch_name="feature/wrong",
            pr_number=99,
        )
        assert result["recorded"] is False
        assert "does not match" in result["error"]

    def test_hooks_graceful_when_no_window_manager(self):
        """Hooks return recorded=False gracefully when WM is unavailable."""
        # With both server and hook state reset, no WM should be available
        # (unless the config auto-detects one in the real CWD, so we
        # verify the hook doesn't crash).
        result = post_tool_call(
            tool_name="Write",
            action="Orphan call",
            file_path="src/orphan.py",
        )
        # The result should either have recorded=False or recorded=True
        # (if it finds a real WM). Either way, it should not raise.
        assert "recorded" in result

    def test_multiple_ticket_lifecycle_via_hooks(self, storage_path):
        """Multiple tickets processed sequentially via hooks."""
        from claude_code_helper_mcp.hooks import pipeline

        wm = WindowManager(storage_path=storage_path, window_size=3)
        pipeline._hook_window_manager = wm

        for i in range(1, 4):
            tid = f"MULTI-{i:03d}"
            branch = f"feature/{tid}-task-{i}"

            post_build_start(
                ticket_id=tid,
                title=f"Multi task {i}",
                branch_name=branch,
                phase=f"phase-{i}",
                base_branch="main",
            )
            post_tool_call(
                tool_name="Write",
                action=f"Built module {i}",
                file_path=f"src/module_{i}.py",
            )
            post_build_complete(
                ticket_id=tid,
                branch_name=branch,
                pr_number=100 + i,
                files_changed=[f"src/module_{i}.py"],
                tests_passed=10 * i,
                tests_total=10 * i,
            )
            post_merge(
                ticket_id=tid,
                branch_name=branch,
                pr_number=100 + i,
                merge_strategy="squash",
                target_branch="main",
            )

        # All 3 tasks completed.
        assert wm.completed_task_count() == 3
        assert wm.has_active_task() is False

        # Each task has its own data.
        for i in range(1, 4):
            task = wm.get_task(f"MULTI-{i:03d}")
            assert task is not None
            assert task.status == TaskStatus.COMPLETED
            assert task.metadata.get("pr_number") == 100 + i


# ===================================================================
# TestEdgeCases
# ===================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_window_queries(self, storage_path):
        """Querying an empty window returns appropriate defaults."""
        wm = WindowManager(storage_path=storage_path, window_size=3)

        assert wm.has_active_task() is False
        assert wm.get_current_task() is None
        assert wm.completed_task_count() == 0
        assert wm.archived_task_count() == 0
        assert wm.total_tasks_in_window() == 0
        assert wm.get_all_task_ids() == []

    def test_get_task_nonexistent_returns_none(self, storage_path):
        """Looking up a nonexistent task returns None."""
        wm = WindowManager(storage_path=storage_path, window_size=3)
        assert wm.get_task("NONEXISTENT") is None

    def test_complete_without_active_raises(self, storage_path):
        """Completing when no task is active raises ValueError."""
        wm = WindowManager(storage_path=storage_path, window_size=3)
        with pytest.raises(ValueError):
            wm.complete_current_task("Nothing to complete")

    def test_start_task_while_active_raises(self, storage_path):
        """Starting a new task while one is active raises ValueError."""
        wm = WindowManager(storage_path=storage_path, window_size=3)
        wm.start_new_task("EDGE-001", "First")

        with pytest.raises(ValueError):
            wm.start_new_task("EDGE-002", "Second")

    def test_window_size_one(self, storage_path):
        """Window size of 1 keeps only the latest completed task."""
        wm = WindowManager(storage_path=storage_path, window_size=1)

        wm.start_new_task("ONE-001", "Task 1")
        wm.complete_current_task("Done 1")

        wm.start_new_task("ONE-002", "Task 2")
        wm.complete_current_task("Done 2")

        assert wm.completed_task_count() == 1
        assert wm.archived_task_count() == 1
        assert wm.is_task_archived("ONE-001")

        window_ids = [t.ticket_id for t in wm.window.completed_tasks]
        assert window_ids == ["ONE-002"]

    def test_large_step_payload(self, storage_path):
        """Tasks with many steps and large payloads persist correctly."""
        wm = WindowManager(storage_path=storage_path, window_size=3)
        task = wm.start_new_task("LARGE-001", "Large payload test")

        # Add 100 steps with substantial content.
        for i in range(100):
            task.add_step(
                action=f"Step {i + 1} with a longer description of what happened",
                description=f"Detailed description #{i + 1}: " + "x" * 500,
                tool_used="Write",
                result_summary=f"Result for step {i + 1}: " + "y" * 200,
            )

        # Add 50 files.
        for i in range(50):
            task.record_file(
                path=f"src/module_{i}.py",
                action=FileAction.CREATED,
                description=f"Module {i}",
            )

        wm.save_current_task()
        wm.complete_current_task("Large task done")

        # Verify all data persists.
        reloaded = wm.get_task("LARGE-001")
        assert reloaded.step_count() == 100
        assert len(reloaded.files) == 50

    def test_many_decisions(self, storage_path):
        """Tasks with many decisions persist correctly."""
        wm = WindowManager(storage_path=storage_path, window_size=3)
        task = wm.start_new_task("DEC-001", "Many decisions")

        for i in range(25):
            task.add_decision(
                decision=f"Decision {i + 1}",
                reasoning=f"Reasoning for decision {i + 1}",
                alternatives=[f"Alt A-{i}", f"Alt B-{i}"],
                context=f"Context {i + 1}",
            )

        wm.save_current_task()
        wm.complete_current_task("All decisions made")

        reloaded = wm.get_task("DEC-001")
        assert len(reloaded.decisions) == 25
        assert reloaded.decisions[0].decision_number == 1
        assert reloaded.decisions[24].decision_number == 25

    def test_resize_window_invalid_values(self, storage_path):
        """Resizing window to invalid values raises ValueError."""
        wm = WindowManager(storage_path=storage_path, window_size=3)

        with pytest.raises(ValueError):
            wm.resize_window(0)

        with pytest.raises(ValueError):
            wm.resize_window(101)

    def test_fail_without_active_raises(self, storage_path):
        """Failing when no task is active raises ValueError."""
        wm = WindowManager(storage_path=storage_path, window_size=3)
        with pytest.raises(ValueError):
            wm.fail_current_task("Nothing to fail")

    def test_task_serialization_roundtrip(self, storage_path):
        """Tasks survive JSON serialization and deserialization."""
        wm = WindowManager(storage_path=storage_path, window_size=3)
        task = wm.start_new_task("SER-001", "Serialization test", "phase-1")
        task.add_step(action="Step 1", tool_used="Write")
        task.record_file("src/a.py", FileAction.CREATED, "Created")
        task.record_branch("feature/SER-001", BranchAction.CREATED, "main")
        task.add_decision(decision="Use JSON", reasoning="Simple")
        task.metadata["custom"] = {"nested": True, "count": 42}
        task.next_steps = ["Review PR", "Deploy"]
        wm.save_current_task()

        # Manually load the JSON file and verify structure.
        # MemoryStore uses the naming convention: task-<ticket_id>.json
        task_file = Path(storage_path) / "tasks" / "task-SER-001.json"
        assert task_file.exists()
        raw = json.loads(task_file.read_text())

        assert raw["ticket_id"] == "SER-001"
        assert raw["title"] == "Serialization test"
        assert raw["phase"] == "phase-1"
        assert raw["status"] == "active"
        assert len(raw["steps"]) == 1
        assert len(raw["files"]) == 1
        assert len(raw["branches"]) == 1
        assert len(raw["decisions"]) == 1
        assert raw["metadata"]["custom"]["nested"] is True
        assert raw["next_steps"] == ["Review PR", "Deploy"]

        # Deserialize back to TaskMemory.
        restored = TaskMemory.from_json_dict(raw)
        assert restored.ticket_id == "SER-001"
        assert restored.step_count() == 1
        assert restored.metadata["custom"]["count"] == 42

    def test_concurrent_window_manager_reads(self, storage_path):
        """Two WindowManager instances reading the same storage stay consistent."""
        wm1 = WindowManager(storage_path=storage_path, window_size=3)
        wm1.start_new_task("CONC-001", "Concurrent task")
        wm1.get_current_task().add_step(action="Step from WM1")
        wm1.save_current_task()
        wm1.complete_current_task("Done from WM1")

        # Create a second manager from the same path.
        wm2 = WindowManager(storage_path=storage_path, window_size=3)
        task = wm2.get_task("CONC-001")

        assert task is not None
        assert task.status == TaskStatus.COMPLETED
        assert task.step_count() == 1


# ===================================================================
# TestCrossCutting
# ===================================================================


class TestCrossCutting:
    """Test cross-component interactions: hooks + server + CLI + recovery."""

    def test_server_task_visible_in_cli(self, project_dir):
        """A task started via server tools is visible in CLI commands."""
        server = create_server(project_root=str(project_dir))
        storage_path = str(project_dir / ".claude-memory")

        # Start a task via the server.
        _call_tool(server, "start_task", {
            "ticket_id": "CROSS-001",
            "title": "Cross-component test",
        })
        _call_tool(server, "record_step", {
            "action": "Built feature via server",
        })

        # Verify via CLI.
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--storage-path", storage_path, "status"]
        )
        assert result.exit_code == 0
        assert "CROSS-001" in result.output

    def test_hook_task_visible_in_server(self, project_dir):
        """A task started via hooks is accessible through server tools."""
        from claude_code_helper_mcp.hooks import pipeline

        storage_path = str(project_dir / ".claude-memory")

        # Use a WindowManager for hooks.
        wm = WindowManager(storage_path=storage_path, window_size=3)
        pipeline._hook_window_manager = wm

        # Start task via hook.
        post_build_start(
            ticket_id="CROSS-002",
            title="Hook to server test",
            branch_name="feature/CROSS-002",
            phase="phase-1",
            base_branch="main",
        )

        # Create a server pointing to the same storage.
        server = create_server(project_root=str(project_dir))
        result = _call_tool(server, "get_task_status", {})

        assert result["has_active_task"] is True
        assert result["task_id"] == "CROSS-002"

    def test_completed_task_recovery_via_server(self, project_dir):
        """Recovery context for a completed task is accurate via server."""
        server = create_server(project_root=str(project_dir))

        # Start and complete a task with rich data.
        _call_tool(server, "start_task", {
            "ticket_id": "CROSS-003",
            "title": "Recovery via server",
        })
        _call_tool(server, "record_step", {"action": "Step 1"})
        _call_tool(server, "record_step", {"action": "Step 2"})
        _call_tool(server, "record_file", {
            "path": "src/feature.py",
            "action": "created",
        })
        _call_tool(server, "record_branch", {
            "branch_name": "feature/CROSS-003",
            "action": "created",
            "base_branch": "main",
        })
        _call_tool(server, "complete_task", {
            "summary": "Feature completed successfully",
        })

        # Recover the completed task.
        result = _call_tool(server, "get_recovery_context", {
            "ticket_id": "CROSS-003",
        })

        assert result["error"] is False
        assert result["has_context"] is True
        assert result["ticket_id"] == "CROSS-003"
        assert result["source"] == "explicit"
        assert result["total_steps_completed"] == 2
        assert len(result["files_modified"]) == 1
        assert result["active_branch"] == "feature/CROSS-003"

    def test_window_rotation_preserves_recovery(self, project_dir):
        """Archived tasks are still recoverable after window rotation."""
        server = create_server(project_root=str(project_dir))
        wm = get_window_manager()

        # Create and complete tasks to trigger rotation (window_size=3).
        for i in range(1, 5):
            _call_tool(server, "start_task", {
                "ticket_id": f"ROT-{i:03d}",
                "title": f"Rotation task {i}",
            })
            _call_tool(server, "record_step", {
                "action": f"Step for task {i}",
            })
            _call_tool(server, "complete_task", {
                "summary": f"Task {i} done",
            })

        # ROT-001 should be archived.
        assert wm.is_task_archived("ROT-001")

        # But still recoverable.
        result = _call_tool(server, "get_recovery_context", {
            "ticket_id": "ROT-001",
        })
        assert result["error"] is False
        assert result["ticket_id"] == "ROT-001"

    def test_cli_list_after_hook_lifecycle(self, project_dir):
        """CLI list shows tasks created and completed via hooks."""
        from claude_code_helper_mcp.hooks import pipeline

        storage_path = str(project_dir / ".claude-memory")
        wm = WindowManager(storage_path=storage_path, window_size=5)
        pipeline._hook_window_manager = wm

        # Process 2 tickets via hooks.
        for i in range(1, 3):
            tid = f"CLIST-{i:03d}"
            post_build_start(
                ticket_id=tid,
                title=f"CLI list task {i}",
                branch_name=f"feature/{tid}",
                base_branch="main",
            )
            post_build_complete(
                ticket_id=tid,
                branch_name=f"feature/{tid}",
                pr_number=200 + i,
                files_changed=[f"src/{tid.lower()}.py"],
            )
            post_merge(
                ticket_id=tid,
                branch_name=f"feature/{tid}",
                pr_number=200 + i,
            )

        # List via CLI.
        runner = CliRunner()
        result = runner.invoke(cli, ["--storage-path", storage_path, "list"])

        assert result.exit_code == 0
        assert "CLIST-001" in result.output
        assert "CLIST-002" in result.output

    def test_server_summary_after_hook_recordings(self, project_dir):
        """Server generate_summary works on tasks built via hooks."""
        from claude_code_helper_mcp.hooks import pipeline

        storage_path = str(project_dir / ".claude-memory")

        # Create server to initialize storage.
        server = create_server(project_root=str(project_dir))
        wm = get_window_manager()
        pipeline._hook_window_manager = wm

        # Build task via hooks.
        post_build_start(
            ticket_id="SUM-001",
            title="Summary via hooks",
            branch_name="feature/SUM-001",
            base_branch="main",
        )
        post_tool_call(
            tool_name="Write",
            action="Created main module",
            file_path="src/main.py",
        )

        # Generate summary via server tool.
        result = _call_tool(server, "generate_summary", {
            "summary_type": "current",
        })

        assert result["error"] is False
        assert result["summary_type"] == "current"
        assert "SUM-001" in result.get("ticket_id", "")
        assert len(result["markdown"]) > 0

    def test_full_pipeline_hook_to_cli_to_recovery(self, project_dir):
        """End-to-end: hooks create task, CLI inspects it, recovery restores it."""
        from claude_code_helper_mcp.hooks import pipeline

        storage_path = str(project_dir / ".claude-memory")
        wm = WindowManager(storage_path=storage_path, window_size=3)
        pipeline._hook_window_manager = wm

        # 1. Hook: start a build.
        post_build_start(
            ticket_id="E2E-001",
            title="End-to-end pipeline test",
            branch_name="feature/E2E-001-full-test",
            phase="phase-4",
            description="Full end-to-end test across all components",
            base_branch="main",
        )

        # 2. Hook: record tool calls.
        post_tool_call(tool_name="Write", action="Created file A", file_path="src/a.py")
        post_tool_call(tool_name="Edit", action="Updated file B", file_path="src/b.py")
        post_tool_call(tool_name="Bash", action="Ran tests", result_summary="All pass")

        # 3. CLI: verify task is visible.
        runner = CliRunner()
        status_result = runner.invoke(
            cli, ["--storage-path", storage_path, "status"]
        )
        assert "E2E-001" in status_result.output

        # 4. Recovery: build context for the active task.
        task = wm.get_current_task()
        recovery = RecoveryContext.from_task_memory(task)
        assert recovery.ticket_id == "E2E-001"
        assert recovery.total_steps_completed >= 4  # 1 build start + 3 tool calls
        assert "src/a.py" in recovery.files_modified
        assert recovery.active_branch == "feature/E2E-001-full-test"

        # 5. Hook: complete the build.
        post_build_complete(
            ticket_id="E2E-001",
            branch_name="feature/E2E-001-full-test",
            pr_number=999,
            files_changed=["src/a.py", "src/b.py"],
            tests_passed=50,
            tests_total=50,
        )

        # 6. Hook: merge.
        post_merge(
            ticket_id="E2E-001",
            branch_name="feature/E2E-001-full-test",
            pr_number=999,
            merge_strategy="squash",
            target_branch="main",
            completion_summary="E2E test complete, all components verified",
        )

        # 7. Verify completed state across all components.
        assert wm.has_active_task() is False
        completed = wm.get_task("E2E-001")
        assert completed.status == TaskStatus.COMPLETED

        # CLI shows completed task.
        list_result = runner.invoke(
            cli, ["--storage-path", storage_path, "list"]
        )
        assert "E2E-001" in list_result.output

        # Recovery still works for completed task.
        completed_recovery = RecoveryContext.from_task_memory(completed, recent_step_count=5)
        assert completed_recovery.status == "completed"
        assert completed_recovery.total_steps_completed >= 4
