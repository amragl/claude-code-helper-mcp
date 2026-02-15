"""Comprehensive tests for Agent Forge hook integration (CMH-015).

Tests all four pipeline hooks (post_tool_call, post_build_start,
post_build_complete, post_merge) with real file I/O, real WindowManager
instances, and real MemoryStore persistence.  No mocks, no stubs.

Test organization:
- TestPostToolCall: post_tool_call hook functionality
- TestPostBuildStart: post_build_start hook functionality
- TestPostBuildComplete: post_build_complete hook functionality
- TestPostMerge: post_merge hook functionality
- TestHookHelpers: _infer_file_action and _get_window_manager
- TestHookIdempotency: idempotent behavior for pipeline resume scenarios
- TestHookErrorHandling: failure tolerance and graceful degradation
- TestHookIntegration: full lifecycle (start -> tool calls -> complete -> merge)
"""

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from claude_code_helper_mcp.config import MemoryConfig
from claude_code_helper_mcp.hooks.pipeline import (
    _get_window_manager,
    _infer_file_action,
    post_build_complete,
    post_build_start,
    post_merge,
    post_tool_call,
    reset_hook_state,
)
from claude_code_helper_mcp.models.records import BranchAction, FileAction
from claude_code_helper_mcp.models.task import TaskMemory, TaskStatus
from claude_code_helper_mcp.storage.store import MemoryStore
from claude_code_helper_mcp.storage.window_manager import WindowManager


@pytest.fixture
def tmp_project(tmp_path):
    """Create a temporary project directory with .claude-memory storage."""
    storage_dir = tmp_path / ".claude-memory"
    storage_dir.mkdir()
    (storage_dir / "tasks").mkdir()
    # Create a pyproject.toml marker so project root detection works.
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    return tmp_path


@pytest.fixture
def window_manager(tmp_project):
    """Create a WindowManager rooted at the tmp_project."""
    storage_path = str(tmp_project / ".claude-memory")
    return WindowManager(storage_path=storage_path, window_size=3)


@pytest.fixture(autouse=True)
def reset_hooks():
    """Reset hook module state before and after each test."""
    reset_hook_state()
    yield
    reset_hook_state()


@pytest.fixture
def active_task(window_manager):
    """Start a task and return the WindowManager with an active task."""
    window_manager.start_new_task(
        ticket_id="CMH-015",
        title="Agent Forge hook integration",
        phase="phase-4",
    )
    return window_manager


# ===================================================================
# TestPostToolCall
# ===================================================================


class TestPostToolCall:
    """Tests for the post_tool_call hook."""

    def test_records_step_for_active_task(self, active_task):
        """post_tool_call should record a step when a task is active."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = active_task

        result = post_tool_call(
            tool_name="Write",
            action="Created hooks module",
            file_path="src/hooks/pipeline.py",
            result_summary="File created successfully",
            success=True,
        )

        assert result["recorded"] is True
        assert result["task_id"] == "CMH-015"
        assert result["step_number"] == 1
        assert result["action"] == "Created hooks module"
        assert result["tool_name"] == "Write"
        assert result["file_recorded"] is True
        assert result["file_path"] == "src/hooks/pipeline.py"
        assert "timestamp" in result

    def test_records_step_without_file(self, active_task):
        """post_tool_call should work without a file_path."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = active_task

        result = post_tool_call(
            tool_name="Bash",
            action="Ran test suite",
            result_summary="All 15 tests passed",
            success=True,
        )

        assert result["recorded"] is True
        assert result["file_recorded"] is False
        assert result["file_path"] is None

    def test_records_failed_step(self, active_task):
        """post_tool_call should record a step even when success is False."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = active_task

        result = post_tool_call(
            tool_name="Bash",
            action="Ran tests",
            result_summary="3 tests failed",
            success=False,
        )

        assert result["recorded"] is True
        task = active_task.get_current_task()
        assert task.steps[-1].success is False

    def test_returns_error_when_no_active_task(self, window_manager):
        """post_tool_call should return recorded=False when no task is active."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = window_manager

        result = post_tool_call(
            tool_name="Write",
            action="Some action",
        )

        assert result["recorded"] is False
        assert "No active task" in result["error"]

    def test_returns_error_when_wm_unavailable(self):
        """post_tool_call should return recorded=False when WM is not available."""
        # No WM initialized, no server running, invalid project root.
        result = post_tool_call(
            tool_name="Write",
            action="Some action",
            project_root="/nonexistent/path",
        )

        assert result["recorded"] is False
        assert "error" in result

    def test_sequential_steps_increment_number(self, active_task):
        """Multiple post_tool_call invocations should increment step numbers."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = active_task

        r1 = post_tool_call(tool_name="Write", action="Step 1")
        r2 = post_tool_call(tool_name="Edit", action="Step 2")
        r3 = post_tool_call(tool_name="Bash", action="Step 3")

        assert r1["step_number"] == 1
        assert r2["step_number"] == 2
        assert r3["step_number"] == 3

    def test_file_deduplication(self, active_task):
        """Recording the same file twice should update, not duplicate."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = active_task

        post_tool_call(
            tool_name="Write",
            action="Created file",
            file_path="src/hooks/pipeline.py",
        )
        post_tool_call(
            tool_name="Edit",
            action="Edited file",
            file_path="src/hooks/pipeline.py",
        )

        task = active_task.get_current_task()
        # Should have 1 file record (deduplicated), not 2.
        assert len(task.files) == 1
        # But 2 steps.
        assert len(task.steps) == 2
        # File should have action history.
        assert len(task.files[0].action_history) == 1

    def test_persists_to_disk(self, active_task):
        """post_tool_call should persist the task to disk."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = active_task

        post_tool_call(
            tool_name="Write",
            action="Created module",
            file_path="src/module.py",
        )

        # Reload from disk and verify.
        reloaded = active_task.store.load_task("CMH-015")
        assert reloaded is not None
        assert len(reloaded.steps) == 1
        assert reloaded.steps[0].action == "Created module"


# ===================================================================
# TestPostBuildStart
# ===================================================================


class TestPostBuildStart:
    """Tests for the post_build_start hook."""

    def test_creates_new_task(self, window_manager):
        """post_build_start should create a new task when none is active."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = window_manager

        result = post_build_start(
            ticket_id="CMH-015",
            title="Hook integration",
            branch_name="feature/CMH-015-hooks",
            phase="phase-4",
            description="Implement Agent Forge hook integration.",
            base_branch="main",
        )

        assert result["recorded"] is True
        assert result["task_id"] == "CMH-015"
        assert result["task_created"] is True
        assert result["branch_name"] == "feature/CMH-015-hooks"
        assert result["phase"] == "phase-4"

        # Verify the task was actually created.
        task = window_manager.get_current_task()
        assert task is not None
        assert task.ticket_id == "CMH-015"
        assert task.title == "Hook integration"
        assert task.phase == "phase-4"
        assert task.metadata.get("description") == "Implement Agent Forge hook integration."

    def test_records_branch_creation(self, window_manager):
        """post_build_start should record the branch creation."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = window_manager

        post_build_start(
            ticket_id="CMH-015",
            title="Hook integration",
            branch_name="feature/CMH-015-hooks",
            base_branch="main",
        )

        task = window_manager.get_current_task()
        assert len(task.branches) == 1
        assert task.branches[0].branch_name == "feature/CMH-015-hooks"
        assert task.branches[0].action == BranchAction.CREATED
        assert task.branches[0].base_branch == "main"

    def test_records_initial_step(self, window_manager):
        """post_build_start should record a build-start step."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = window_manager

        post_build_start(
            ticket_id="CMH-015",
            title="Hook integration",
            branch_name="feature/CMH-015-hooks",
        )

        task = window_manager.get_current_task()
        assert len(task.steps) == 1
        assert "Build started" in task.steps[0].action
        assert task.steps[0].tool_used == "agent-forge-build"

    def test_idempotent_when_task_already_active(self, active_task):
        """post_build_start should be idempotent when the same task is active."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = active_task

        result = post_build_start(
            ticket_id="CMH-015",
            title="Hook integration",
            branch_name="feature/CMH-015-hooks",
        )

        assert result["recorded"] is True
        assert result["task_created"] is False

    def test_error_when_different_task_active(self, active_task):
        """post_build_start should error when a different task is active."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = active_task

        result = post_build_start(
            ticket_id="CMH-016",
            title="Different ticket",
            branch_name="feature/CMH-016-other",
        )

        assert result["recorded"] is False
        assert "CMH-015" in result["error"]

    def test_no_description(self, window_manager):
        """post_build_start should work without a description."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = window_manager

        result = post_build_start(
            ticket_id="CMH-015",
            title="Hook integration",
            branch_name="feature/CMH-015-hooks",
        )

        assert result["recorded"] is True
        task = window_manager.get_current_task()
        assert "description" not in task.metadata

    def test_persists_to_disk(self, window_manager):
        """post_build_start should persist everything to disk."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = window_manager

        post_build_start(
            ticket_id="CMH-015",
            title="Hook integration",
            branch_name="feature/CMH-015-hooks",
            phase="phase-4",
        )

        reloaded = window_manager.store.load_task("CMH-015")
        assert reloaded is not None
        assert reloaded.ticket_id == "CMH-015"
        assert len(reloaded.branches) == 1
        assert len(reloaded.steps) == 1


# ===================================================================
# TestPostBuildComplete
# ===================================================================


class TestPostBuildComplete:
    """Tests for the post_build_complete hook."""

    def test_records_build_completion(self, active_task):
        """post_build_complete should record PR creation and files."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = active_task

        result = post_build_complete(
            ticket_id="CMH-015",
            branch_name="feature/CMH-015-hooks",
            pr_number=42,
            files_changed=["src/hooks/pipeline.py", "tests/test_hooks.py"],
            tests_passed=15,
            tests_total=15,
            commit_count=3,
            summary="Implemented all 4 hook functions",
        )

        assert result["recorded"] is True
        assert result["task_id"] == "CMH-015"
        assert result["pr_number"] == 42
        assert result["files_recorded"] == 2

    def test_records_files_changed(self, active_task):
        """post_build_complete should create FileRecords for each changed file."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = active_task

        post_build_complete(
            ticket_id="CMH-015",
            branch_name="feature/CMH-015-hooks",
            pr_number=42,
            files_changed=["file1.py", "file2.py", "file3.py"],
        )

        task = active_task.get_current_task()
        assert len(task.files) == 3
        file_paths = [f.path for f in task.files]
        assert "file1.py" in file_paths
        assert "file2.py" in file_paths
        assert "file3.py" in file_paths

    def test_records_branch_push(self, active_task):
        """post_build_complete should record a branch push action."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = active_task

        post_build_complete(
            ticket_id="CMH-015",
            branch_name="feature/CMH-015-hooks",
            pr_number=42,
        )

        task = active_task.get_current_task()
        pushed_branches = [
            b for b in task.branches if b.action == BranchAction.PUSHED
        ]
        assert len(pushed_branches) == 1
        assert pushed_branches[0].branch_name == "feature/CMH-015-hooks"

    def test_stores_metadata(self, active_task):
        """post_build_complete should store build metadata in the task."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = active_task

        post_build_complete(
            ticket_id="CMH-015",
            branch_name="feature/CMH-015-hooks",
            pr_number=42,
            tests_passed=15,
            tests_total=15,
            commit_count=3,
        )

        task = active_task.get_current_task()
        assert task.metadata["pr_number"] == 42
        assert task.metadata["branch_name"] == "feature/CMH-015-hooks"
        assert task.metadata["build_complete"] is True
        assert task.metadata["tests_passed"] == 15
        assert task.metadata["tests_total"] == 15
        assert task.metadata["commit_count"] == 3

    def test_records_decision(self, active_task):
        """post_build_complete should record a build decision."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = active_task

        post_build_complete(
            ticket_id="CMH-015",
            branch_name="feature/CMH-015-hooks",
            pr_number=42,
            summary="Implemented with event-driven architecture",
        )

        task = active_task.get_current_task()
        assert len(task.decisions) == 1
        assert "CMH-015" in task.decisions[0].decision
        assert "event-driven" in task.decisions[0].reasoning

    def test_error_when_no_active_task(self, window_manager):
        """post_build_complete should error when no task is active."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = window_manager

        result = post_build_complete(
            ticket_id="CMH-015",
            branch_name="feature/CMH-015-hooks",
            pr_number=42,
        )

        assert result["recorded"] is False
        assert "No active task" in result["error"]

    def test_error_when_ticket_mismatch(self, active_task):
        """post_build_complete should error when ticket doesn't match."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = active_task

        result = post_build_complete(
            ticket_id="CMH-999",
            branch_name="feature/CMH-999-other",
            pr_number=42,
        )

        assert result["recorded"] is False
        assert "does not match" in result["error"]

    def test_no_files_changed(self, active_task):
        """post_build_complete should work without files_changed."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = active_task

        result = post_build_complete(
            ticket_id="CMH-015",
            branch_name="feature/CMH-015-hooks",
            pr_number=42,
        )

        assert result["recorded"] is True
        assert result["files_recorded"] == 0


# ===================================================================
# TestPostMerge
# ===================================================================


class TestPostMerge:
    """Tests for the post_merge hook."""

    def test_completes_task_on_merge(self, active_task):
        """post_merge should complete the memory task."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = active_task

        result = post_merge(
            ticket_id="CMH-015",
            branch_name="feature/CMH-015-hooks",
            pr_number=42,
            merge_strategy="squash",
            target_branch="main",
        )

        assert result["recorded"] is True
        assert result["task_completed"] is True
        assert result["pr_number"] == 42
        assert result["merge_strategy"] == "squash"

        # Task should now be completed.
        assert active_task.get_current_task() is None
        assert active_task.completed_task_count() == 1

    def test_records_merge_and_delete_branch(self, active_task):
        """post_merge should record branch merge and deletion."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = active_task

        post_merge(
            ticket_id="CMH-015",
            branch_name="feature/CMH-015-hooks",
            pr_number=42,
        )

        # The task is now completed -- retrieve it from completed list.
        completed = active_task.window.completed_tasks[-1]
        branch_actions = [
            (b.branch_name, b.action) for b in completed.branches
        ]
        # Should have both merge and delete recorded.
        # Note: branch deduplication means one BranchRecord with action history.
        hook_branch = [b for b in completed.branches if b.branch_name == "feature/CMH-015-hooks"]
        assert len(hook_branch) == 1
        # Latest action should be DELETED.
        assert hook_branch[0].action == BranchAction.DELETED
        # History should include MERGED.
        merged_in_history = any(
            entry.get("action") == "merged"
            for entry in hook_branch[0].action_history
        )
        assert merged_in_history

    def test_records_merge_step(self, active_task):
        """post_merge should record a merge step."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = active_task

        post_merge(
            ticket_id="CMH-015",
            branch_name="feature/CMH-015-hooks",
            pr_number=42,
            merge_strategy="squash",
        )

        completed = active_task.window.completed_tasks[-1]
        merge_steps = [
            s for s in completed.steps if "merged" in s.action.lower()
        ]
        assert len(merge_steps) == 1
        assert "squash" in merge_steps[0].action.lower()

    def test_sets_completion_summary(self, active_task):
        """post_merge should set a completion summary on the task."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = active_task

        post_merge(
            ticket_id="CMH-015",
            branch_name="feature/CMH-015-hooks",
            pr_number=42,
            completion_summary="All hooks implemented and tested.",
        )

        completed = active_task.window.completed_tasks[-1]
        assert completed.summary == "All hooks implemented and tested."
        assert completed.status == TaskStatus.COMPLETED

    def test_auto_generates_summary(self, active_task):
        """post_merge should auto-generate a summary when none is provided."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = active_task

        # Add some metadata from build phase.
        task = active_task.get_current_task()
        task.metadata["tests_total"] = 15
        task.metadata["tests_passed"] = 15
        active_task.save_current_task()

        post_merge(
            ticket_id="CMH-015",
            branch_name="feature/CMH-015-hooks",
            pr_number=42,
        )

        completed = active_task.window.completed_tasks[-1]
        assert "PR #42" in completed.summary
        assert "Tests: 15/15" in completed.summary

    def test_window_state_returned(self, active_task):
        """post_merge should return window state information."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = active_task

        result = post_merge(
            ticket_id="CMH-015",
            branch_name="feature/CMH-015-hooks",
            pr_number=42,
        )

        assert "window_state" in result
        assert result["window_state"]["completed_tasks"] == 1

    def test_error_when_no_active_task(self, window_manager):
        """post_merge should error when no task is active."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = window_manager

        result = post_merge(
            ticket_id="CMH-015",
            branch_name="feature/CMH-015-hooks",
            pr_number=42,
        )

        assert result["recorded"] is False
        assert "No active task" in result["error"]

    def test_error_when_ticket_mismatch(self, active_task):
        """post_merge should error when ticket doesn't match."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = active_task

        result = post_merge(
            ticket_id="CMH-999",
            branch_name="feature/CMH-999-other",
            pr_number=42,
        )

        assert result["recorded"] is False
        assert "does not match" in result["error"]

    def test_task_summary_stats(self, active_task):
        """post_merge should return task summary statistics."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = active_task

        # Add some data to the task.
        task = active_task.get_current_task()
        task.add_step(action="Step 1")
        task.add_step(action="Step 2")
        task.record_file(path="file1.py", action=FileAction.CREATED)
        task.add_decision(decision="Decision 1")
        active_task.save_current_task()

        result = post_merge(
            ticket_id="CMH-015",
            branch_name="feature/CMH-015-hooks",
            pr_number=42,
        )

        assert result["task_summary"]["steps"] >= 2  # At least the 2 manually added steps
        assert result["task_summary"]["files"] == 1
        assert result["task_summary"]["decisions"] == 1


# ===================================================================
# TestHookHelpers
# ===================================================================


class TestHookHelpers:
    """Tests for helper functions."""

    def test_infer_file_action_write(self):
        """_infer_file_action should return CREATED for Write tool."""
        assert _infer_file_action("Write") == FileAction.CREATED

    def test_infer_file_action_edit(self):
        """_infer_file_action should return MODIFIED for Edit tool."""
        assert _infer_file_action("Edit") == FileAction.MODIFIED

    def test_infer_file_action_read(self):
        """_infer_file_action should return READ for Read tool."""
        assert _infer_file_action("Read") == FileAction.READ

    def test_infer_file_action_bash(self):
        """_infer_file_action should return MODIFIED for Bash tool."""
        assert _infer_file_action("Bash") == FileAction.MODIFIED

    def test_infer_file_action_unknown(self):
        """_infer_file_action should default to MODIFIED for unknown tools."""
        assert _infer_file_action("UnknownTool") == FileAction.MODIFIED

    def test_infer_file_action_case_insensitive(self):
        """_infer_file_action should be case-insensitive."""
        assert _infer_file_action("write") == FileAction.CREATED
        assert _infer_file_action("WRITE") == FileAction.CREATED
        assert _infer_file_action("edit") == FileAction.MODIFIED
        assert _infer_file_action("READ") == FileAction.READ

    def test_reset_hook_state(self, window_manager):
        """reset_hook_state should clear the cached WindowManager."""
        from claude_code_helper_mcp.hooks import pipeline

        pipeline._hook_window_manager = window_manager
        assert pipeline._hook_window_manager is not None

        reset_hook_state()
        assert pipeline._hook_window_manager is None

    def test_get_window_manager_caches(self, tmp_project):
        """_get_window_manager should cache the standalone WindowManager."""
        from claude_code_helper_mcp.hooks import pipeline

        wm1 = _get_window_manager(str(tmp_project))
        wm2 = _get_window_manager(str(tmp_project))
        assert wm1 is wm2


# ===================================================================
# TestHookIdempotency
# ===================================================================


class TestHookIdempotency:
    """Tests for idempotent hook behavior in resume scenarios."""

    def test_post_build_start_idempotent(self, window_manager):
        """Calling post_build_start twice for the same ticket should not error."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = window_manager

        r1 = post_build_start(
            ticket_id="CMH-015",
            title="Hook integration",
            branch_name="feature/CMH-015-hooks",
        )
        r2 = post_build_start(
            ticket_id="CMH-015",
            title="Hook integration",
            branch_name="feature/CMH-015-hooks",
        )

        assert r1["recorded"] is True
        assert r1["task_created"] is True
        assert r2["recorded"] is True
        assert r2["task_created"] is False

    def test_tool_call_deduplicates_files(self, active_task):
        """Multiple tool calls on the same file should deduplicate."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = active_task

        for i in range(5):
            post_tool_call(
                tool_name="Edit",
                action=f"Edit iteration {i}",
                file_path="src/hooks/pipeline.py",
            )

        task = active_task.get_current_task()
        assert len(task.files) == 1  # Deduplicated
        assert len(task.steps) == 5  # All steps recorded
        assert len(task.files[0].action_history) == 4  # 4 updates to original


# ===================================================================
# TestHookErrorHandling
# ===================================================================


class TestHookErrorHandling:
    """Tests for failure tolerance and graceful degradation."""

    def test_post_tool_call_returns_error_dict_on_exception(self):
        """post_tool_call should catch exceptions and return error dict."""
        # Force an exception by providing invalid project root.
        result = post_tool_call(
            tool_name="Write",
            action="Some action",
            project_root="/nonexistent/invalid/path",
        )

        assert result["recorded"] is False
        assert "error" in result

    def test_post_build_start_returns_error_dict_on_exception(self):
        """post_build_start should catch exceptions and return error dict."""
        result = post_build_start(
            ticket_id="CMH-015",
            title="Hook integration",
            branch_name="feature/CMH-015-hooks",
            project_root="/nonexistent/invalid/path",
        )

        assert result["recorded"] is False
        assert "error" in result

    def test_post_build_complete_returns_error_dict_on_exception(self):
        """post_build_complete should catch exceptions and return error dict."""
        result = post_build_complete(
            ticket_id="CMH-015",
            branch_name="feature/CMH-015-hooks",
            pr_number=42,
            project_root="/nonexistent/invalid/path",
        )

        assert result["recorded"] is False
        assert "error" in result

    def test_post_merge_returns_error_dict_on_exception(self):
        """post_merge should catch exceptions and return error dict."""
        result = post_merge(
            ticket_id="CMH-015",
            branch_name="feature/CMH-015-hooks",
            pr_number=42,
            project_root="/nonexistent/invalid/path",
        )

        assert result["recorded"] is False
        assert "error" in result


# ===================================================================
# TestHookIntegration
# ===================================================================


class TestHookIntegration:
    """Integration tests: full lifecycle from build start to merge."""

    def test_full_lifecycle(self, window_manager):
        """Test the complete hook lifecycle: start -> tools -> complete -> merge."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = window_manager

        # 1. Build starts.
        r_start = post_build_start(
            ticket_id="CMH-015",
            title="Agent Forge hook integration",
            branch_name="feature/CMH-015-hooks",
            phase="phase-4",
            description="Implement hook integration for automatic recording.",
        )
        assert r_start["recorded"] is True
        assert r_start["task_created"] is True

        # 2. Tool calls during build.
        r_write1 = post_tool_call(
            tool_name="Write",
            action="Created hooks/__init__.py",
            file_path="src/hooks/__init__.py",
        )
        assert r_write1["recorded"] is True

        r_write2 = post_tool_call(
            tool_name="Write",
            action="Created hooks/pipeline.py",
            file_path="src/hooks/pipeline.py",
        )
        assert r_write2["recorded"] is True

        r_write3 = post_tool_call(
            tool_name="Write",
            action="Created tests/test_hooks.py",
            file_path="tests/test_hooks.py",
        )
        assert r_write3["recorded"] is True

        r_bash = post_tool_call(
            tool_name="Bash",
            action="Ran test suite",
            result_summary="60 tests passed, 0 failed",
        )
        assert r_bash["recorded"] is True

        # 3. Build complete.
        r_complete = post_build_complete(
            ticket_id="CMH-015",
            branch_name="feature/CMH-015-hooks",
            pr_number=42,
            files_changed=[
                "src/hooks/__init__.py",
                "src/hooks/pipeline.py",
                "tests/test_hooks.py",
            ],
            tests_passed=60,
            tests_total=60,
            commit_count=3,
            summary="Implemented all 4 hook functions with comprehensive tests.",
        )
        assert r_complete["recorded"] is True
        assert r_complete["files_recorded"] == 3

        # Verify task state before merge.
        task = window_manager.get_current_task()
        assert task.ticket_id == "CMH-015"
        assert task.metadata["pr_number"] == 42
        assert task.metadata["build_complete"] is True
        assert len(task.steps) >= 6  # start + 4 tools + complete
        assert len(task.files) == 3  # 3 unique files
        assert len(task.decisions) == 1  # build decision

        # 4. Merge.
        r_merge = post_merge(
            ticket_id="CMH-015",
            branch_name="feature/CMH-015-hooks",
            pr_number=42,
            merge_strategy="squash",
            target_branch="main",
        )
        assert r_merge["recorded"] is True
        assert r_merge["task_completed"] is True

        # Verify final state.
        assert window_manager.get_current_task() is None
        assert window_manager.completed_task_count() == 1

        completed = window_manager.window.completed_tasks[0]
        assert completed.ticket_id == "CMH-015"
        assert completed.status == TaskStatus.COMPLETED
        assert completed.completed_at is not None
        assert "PR #42" in completed.summary
        assert "Tests: 60/60" in completed.summary

    def test_lifecycle_persists_across_reload(self, tmp_project):
        """Hook data should survive a WindowManager reload from disk."""
        from claude_code_helper_mcp.hooks import pipeline

        storage_path = str(tmp_project / ".claude-memory")
        wm = WindowManager(storage_path=storage_path, window_size=3)
        pipeline._hook_window_manager = wm

        # Create task and record some data.
        post_build_start(
            ticket_id="CMH-015",
            title="Hook integration",
            branch_name="feature/CMH-015-hooks",
            phase="phase-4",
        )
        post_tool_call(
            tool_name="Write",
            action="Created module",
            file_path="src/hooks/pipeline.py",
        )

        # Simulate a WindowManager restart (e.g., after /clear).
        reset_hook_state()
        wm2 = WindowManager(storage_path=storage_path, window_size=3)
        pipeline._hook_window_manager = wm2

        # Task should still be active after reload.
        task = wm2.get_current_task()
        assert task is not None
        assert task.ticket_id == "CMH-015"
        assert len(task.steps) >= 1
        assert len(task.files) >= 1

    def test_window_rotation_after_multiple_merges(self, window_manager):
        """Completing multiple tasks should trigger window rotation."""
        from claude_code_helper_mcp.hooks import pipeline
        pipeline._hook_window_manager = window_manager

        # Complete 4 tasks (window size is 3, so the oldest should be archived).
        for i in range(4):
            tid = f"CMH-{100 + i}"
            post_build_start(
                ticket_id=tid,
                title=f"Task {i}",
                branch_name=f"feature/{tid}",
            )
            post_merge(
                ticket_id=tid,
                branch_name=f"feature/{tid}",
                pr_number=100 + i,
            )

        # Window should have 3 completed tasks (window_size=3).
        assert window_manager.completed_task_count() == 3
        # First task should have been archived.
        assert window_manager.archived_task_count() >= 1
        assert "CMH-100" in window_manager.window.archived_task_ids
