"""Tests for WindowManager -- sliding window retention lifecycle coordinator.

All tests use real files in temporary directories with real TaskMemory,
MemoryWindow, and MemoryStore objects.  No mocks, no stubs, no fakes.
"""

from pathlib import Path

import pytest

from claude_code_helper_mcp.models.task import TaskMemory, TaskStatus
from claude_code_helper_mcp.models.window import MemoryWindow
from claude_code_helper_mcp.storage.store import MemoryStore
from claude_code_helper_mcp.storage.window_manager import WindowManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def storage_path(tmp_path: Path) -> str:
    """Return a path string for a temp storage directory."""
    return str(tmp_path / ".claude-memory")


@pytest.fixture()
def manager(storage_path: str) -> WindowManager:
    """Return a WindowManager with a fresh temp store."""
    return WindowManager(storage_path=storage_path)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    """WindowManager initializes store and loads/creates a window."""

    def test_creates_store_directory(self, storage_path: str) -> None:
        mgr = WindowManager(storage_path=storage_path)
        assert mgr.store.storage_root.is_dir()

    def test_creates_empty_window_on_fresh_store(self, manager: WindowManager) -> None:
        assert manager.window is not None
        assert manager.get_current_task() is None
        assert manager.completed_task_count() == 0

    def test_default_window_size_is_3(self, manager: WindowManager) -> None:
        assert manager.window_size == 3

    def test_custom_window_size(self, storage_path: str) -> None:
        mgr = WindowManager(storage_path=storage_path, window_size=5)
        assert mgr.window_size == 5

    def test_loads_existing_window_on_restart(self, storage_path: str) -> None:
        mgr1 = WindowManager(storage_path=storage_path)
        mgr1.start_new_task("CMH-INIT-1", "First task")
        mgr1.complete_current_task("Done")

        # Create a new manager on the same path -- should load persisted state.
        mgr2 = WindowManager(storage_path=storage_path)
        assert mgr2.completed_task_count() == 1
        assert mgr2.get_current_task() is None

    def test_window_size_override_on_load(self, storage_path: str) -> None:
        """Loading with a smaller window_size archives overflow tasks."""
        mgr1 = WindowManager(storage_path=storage_path, window_size=5)
        for i in range(5):
            mgr1.start_new_task(f"T-{i}", f"Task {i}")
            mgr1.complete_current_task()

        # Re-open with smaller window -- should archive 3 tasks.
        mgr2 = WindowManager(storage_path=storage_path, window_size=2)
        assert mgr2.completed_task_count() == 2
        assert mgr2.archived_task_count() == 3


# ---------------------------------------------------------------------------
# start_new_task
# ---------------------------------------------------------------------------


class TestStartNewTask:
    """Starting a new task creates it in the window and persists it."""

    def test_returns_task_memory(self, manager: WindowManager) -> None:
        task = manager.start_new_task("CMH-100", "Test task")
        assert isinstance(task, TaskMemory)
        assert task.ticket_id == "CMH-100"
        assert task.title == "Test task"
        assert task.status == TaskStatus.ACTIVE

    def test_sets_current_task(self, manager: WindowManager) -> None:
        manager.start_new_task("CMH-101", "Current task", phase="phase-1")
        current = manager.get_current_task()
        assert current is not None
        assert current.ticket_id == "CMH-101"
        assert current.phase == "phase-1"

    def test_persists_task_file(self, manager: WindowManager) -> None:
        manager.start_new_task("CMH-102", "Persisted task")
        assert manager.store.task_exists("CMH-102")

    def test_persists_window_state(self, storage_path: str) -> None:
        mgr = WindowManager(storage_path=storage_path)
        mgr.start_new_task("CMH-103", "Persist window")

        # Reload from disk to verify persistence.
        mgr2 = WindowManager(storage_path=storage_path)
        assert mgr2.get_current_task() is not None
        assert mgr2.get_current_task().ticket_id == "CMH-103"

    def test_raises_if_task_already_active(self, manager: WindowManager) -> None:
        manager.start_new_task("CMH-104", "First")
        with pytest.raises(ValueError, match="still active"):
            manager.start_new_task("CMH-105", "Second")

    def test_can_start_after_complete(self, manager: WindowManager) -> None:
        manager.start_new_task("CMH-106", "First")
        manager.complete_current_task()
        task = manager.start_new_task("CMH-107", "Second")
        assert task.ticket_id == "CMH-107"

    def test_can_start_after_fail(self, manager: WindowManager) -> None:
        manager.start_new_task("CMH-108", "First")
        manager.fail_current_task("oops")
        task = manager.start_new_task("CMH-109", "Second")
        assert task.ticket_id == "CMH-109"

    def test_has_active_task_is_true(self, manager: WindowManager) -> None:
        assert manager.has_active_task() is False
        manager.start_new_task("CMH-110", "Active")
        assert manager.has_active_task() is True


# ---------------------------------------------------------------------------
# complete_current_task
# ---------------------------------------------------------------------------


class TestCompleteCurrentTask:
    """Completing a task marks it done, rotates the window, and persists."""

    def test_returns_completed_task(self, manager: WindowManager) -> None:
        manager.start_new_task("CMH-200", "To complete")
        completed = manager.complete_current_task("All done")
        assert completed.status == TaskStatus.COMPLETED
        assert completed.summary == "All done"
        assert completed.completed_at is not None

    def test_clears_current_task(self, manager: WindowManager) -> None:
        manager.start_new_task("CMH-201", "Clear after complete")
        manager.complete_current_task()
        assert manager.get_current_task() is None
        assert manager.has_active_task() is False

    def test_adds_to_completed_list(self, manager: WindowManager) -> None:
        manager.start_new_task("CMH-202", "First")
        manager.complete_current_task()
        assert manager.completed_task_count() == 1

    def test_persists_completed_task(self, manager: WindowManager) -> None:
        manager.start_new_task("CMH-203", "Persist")
        manager.complete_current_task("saved")
        loaded = manager.store.load_task("CMH-203")
        assert loaded is not None
        assert loaded.status == TaskStatus.COMPLETED

    def test_raises_if_no_current_task(self, manager: WindowManager) -> None:
        with pytest.raises(ValueError, match="No current task"):
            manager.complete_current_task()

    def test_window_state_persisted(self, storage_path: str) -> None:
        mgr = WindowManager(storage_path=storage_path)
        mgr.start_new_task("CMH-204", "Persist window")
        mgr.complete_current_task()

        mgr2 = WindowManager(storage_path=storage_path)
        assert mgr2.completed_task_count() == 1
        assert mgr2.get_current_task() is None


# ---------------------------------------------------------------------------
# Sliding window rotation and archival
# ---------------------------------------------------------------------------


class TestSlidingWindowRotation:
    """When the window overflows, the oldest completed task is archived."""

    def test_window_enforces_size_limit(self, storage_path: str) -> None:
        mgr = WindowManager(storage_path=storage_path, window_size=2)
        for i in range(3):
            mgr.start_new_task(f"ROT-{i}", f"Task {i}")
            mgr.complete_current_task()

        # Window holds 2 completed tasks; oldest should be archived.
        assert mgr.completed_task_count() == 2
        assert mgr.archived_task_count() == 1
        assert mgr.is_task_archived("ROT-0")
        assert not mgr.is_task_archived("ROT-1")
        assert not mgr.is_task_archived("ROT-2")

    def test_archived_task_still_on_disk(self, storage_path: str) -> None:
        mgr = WindowManager(storage_path=storage_path, window_size=1)
        mgr.start_new_task("ARCH-1", "First")
        mgr.complete_current_task()
        mgr.start_new_task("ARCH-2", "Second")
        mgr.complete_current_task()

        # ARCH-1 should be archived but its file is still on disk.
        assert mgr.is_task_archived("ARCH-1")
        assert mgr.store.task_exists("ARCH-1")

    def test_archived_task_retrievable_via_get_task(self, storage_path: str) -> None:
        mgr = WindowManager(storage_path=storage_path, window_size=1)
        mgr.start_new_task("ARCH-A", "First")
        mgr.complete_current_task("Completed first")
        mgr.start_new_task("ARCH-B", "Second")
        mgr.complete_current_task("Completed second")

        # ARCH-A is archived but still retrievable from disk.
        task = mgr.get_task("ARCH-A")
        assert task is not None
        assert task.ticket_id == "ARCH-A"
        assert task.status == TaskStatus.COMPLETED

    def test_multiple_rotations(self, storage_path: str) -> None:
        mgr = WindowManager(storage_path=storage_path, window_size=2)
        for i in range(6):
            mgr.start_new_task(f"MULTI-{i}", f"Task {i}")
            mgr.complete_current_task()

        assert mgr.completed_task_count() == 2
        assert mgr.archived_task_count() == 4

        # The last 2 (MULTI-4, MULTI-5) are in the window.
        window_ids = mgr.get_all_task_ids()
        assert "MULTI-4" in window_ids
        assert "MULTI-5" in window_ids

        # The first 4 are archived.
        for i in range(4):
            assert mgr.is_task_archived(f"MULTI-{i}")

    def test_window_size_1(self, storage_path: str) -> None:
        """Edge case: window size of 1 retains only the most recent completed task."""
        mgr = WindowManager(storage_path=storage_path, window_size=1)
        for i in range(5):
            mgr.start_new_task(f"TINY-{i}", f"Task {i}")
            mgr.complete_current_task()

        assert mgr.completed_task_count() == 1
        assert mgr.archived_task_count() == 4
        # Only the last one is in the window.
        assert mgr.window.completed_tasks[0].ticket_id == "TINY-4"

    def test_rotation_persists_across_restarts(self, storage_path: str) -> None:
        mgr = WindowManager(storage_path=storage_path, window_size=2)
        for i in range(4):
            mgr.start_new_task(f"PERS-{i}", f"Task {i}")
            mgr.complete_current_task()

        mgr2 = WindowManager(storage_path=storage_path)
        assert mgr2.completed_task_count() == 2
        assert mgr2.archived_task_count() == 2
        assert mgr2.is_task_archived("PERS-0")
        assert mgr2.is_task_archived("PERS-1")


# ---------------------------------------------------------------------------
# fail_current_task
# ---------------------------------------------------------------------------


class TestFailCurrentTask:
    """Failing a task marks it failed and adds it to the completed window."""

    def test_returns_failed_task(self, manager: WindowManager) -> None:
        manager.start_new_task("FAIL-1", "Will fail")
        failed = manager.fail_current_task("Build error")
        assert failed.status == TaskStatus.FAILED
        assert "FAILED: Build error" in failed.summary

    def test_clears_current_task(self, manager: WindowManager) -> None:
        manager.start_new_task("FAIL-2", "Will fail")
        manager.fail_current_task()
        assert manager.get_current_task() is None

    def test_adds_to_completed_count(self, manager: WindowManager) -> None:
        manager.start_new_task("FAIL-3", "Will fail")
        manager.fail_current_task("reason")
        assert manager.completed_task_count() == 1

    def test_persists_failed_task(self, manager: WindowManager) -> None:
        manager.start_new_task("FAIL-4", "Will fail")
        manager.fail_current_task("persist check")
        loaded = manager.store.load_task("FAIL-4")
        assert loaded is not None
        assert loaded.status == TaskStatus.FAILED

    def test_raises_if_no_current_task(self, manager: WindowManager) -> None:
        with pytest.raises(ValueError, match="No current task"):
            manager.fail_current_task()

    def test_failed_task_counts_toward_window_limit(self, storage_path: str) -> None:
        mgr = WindowManager(storage_path=storage_path, window_size=2)
        mgr.start_new_task("FW-1", "Complete")
        mgr.complete_current_task()
        mgr.start_new_task("FW-2", "Fail")
        mgr.fail_current_task("nope")
        mgr.start_new_task("FW-3", "Complete again")
        mgr.complete_current_task()

        # Window holds 2: FW-2 (failed) and FW-3 (completed). FW-1 is archived.
        assert mgr.completed_task_count() == 2
        assert mgr.archived_task_count() == 1
        assert mgr.is_task_archived("FW-1")


# ---------------------------------------------------------------------------
# save_current_task (incremental persistence)
# ---------------------------------------------------------------------------


class TestSaveCurrentTask:
    """Incremental saves persist the current task without status change."""

    def test_saves_task_with_added_steps(self, manager: WindowManager) -> None:
        task = manager.start_new_task("INC-1", "Incremental")
        task.add_step("First step", "analysis")
        manager.save_current_task()

        loaded = manager.store.load_task("INC-1")
        assert loaded is not None
        assert len(loaded.steps) == 1
        assert loaded.steps[0].action == "First step"

    def test_save_when_no_current_returns_none(self, manager: WindowManager) -> None:
        result = manager.save_current_task()
        assert result is None

    def test_preserves_task_status(self, manager: WindowManager) -> None:
        task = manager.start_new_task("INC-2", "Active")
        manager.save_current_task()
        loaded = manager.store.load_task("INC-2")
        assert loaded is not None
        assert loaded.status == TaskStatus.ACTIVE

    def test_multiple_incremental_saves(self, manager: WindowManager) -> None:
        task = manager.start_new_task("INC-3", "Multi-save")
        task.add_step("Step 1", "first")
        manager.save_current_task()
        task.add_step("Step 2", "second")
        manager.save_current_task()

        loaded = manager.store.load_task("INC-3")
        assert loaded is not None
        assert len(loaded.steps) == 2


# ---------------------------------------------------------------------------
# get_task
# ---------------------------------------------------------------------------


class TestGetTask:
    """get_task searches current, completed, and archived tasks."""

    def test_get_current_task(self, manager: WindowManager) -> None:
        manager.start_new_task("GET-1", "Current")
        task = manager.get_task("GET-1")
        assert task is not None
        assert task.ticket_id == "GET-1"

    def test_get_completed_task_in_window(self, manager: WindowManager) -> None:
        manager.start_new_task("GET-2", "Complete me")
        manager.complete_current_task()
        task = manager.get_task("GET-2")
        assert task is not None
        assert task.status == TaskStatus.COMPLETED

    def test_get_archived_task(self, storage_path: str) -> None:
        mgr = WindowManager(storage_path=storage_path, window_size=1)
        mgr.start_new_task("GET-A1", "Archived")
        mgr.complete_current_task()
        mgr.start_new_task("GET-A2", "In window")
        mgr.complete_current_task()

        task = mgr.get_task("GET-A1")
        assert task is not None
        assert task.ticket_id == "GET-A1"

    def test_get_nonexistent_returns_none(self, manager: WindowManager) -> None:
        assert manager.get_task("NOPE-999") is None


# ---------------------------------------------------------------------------
# get_all_known_task_ids
# ---------------------------------------------------------------------------


class TestGetAllKnownTaskIds:
    """get_all_known_task_ids returns a sorted, deduplicated list."""

    def test_empty_manager(self, manager: WindowManager) -> None:
        assert manager.get_all_known_task_ids() == []

    def test_includes_current_and_completed(self, manager: WindowManager) -> None:
        manager.start_new_task("KNOWN-1", "First")
        manager.complete_current_task()
        manager.start_new_task("KNOWN-2", "Current")

        ids = manager.get_all_known_task_ids()
        assert "KNOWN-1" in ids
        assert "KNOWN-2" in ids

    def test_includes_archived(self, storage_path: str) -> None:
        mgr = WindowManager(storage_path=storage_path, window_size=1)
        for i in range(3):
            mgr.start_new_task(f"KN-{i}", f"Task {i}")
            mgr.complete_current_task()

        ids = mgr.get_all_known_task_ids()
        assert "KN-0" in ids
        assert "KN-1" in ids
        assert "KN-2" in ids

    def test_sorted_and_deduplicated(self, storage_path: str) -> None:
        mgr = WindowManager(storage_path=storage_path, window_size=2)
        mgr.start_new_task("Z-3", "Third")
        mgr.complete_current_task()
        mgr.start_new_task("A-1", "First")
        mgr.complete_current_task()

        ids = mgr.get_all_known_task_ids()
        assert ids == sorted(ids)
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# resize_window
# ---------------------------------------------------------------------------


class TestResizeWindow:
    """Resizing the window archives overflow tasks or opens capacity."""

    def test_shrink_archives_overflow(self, storage_path: str) -> None:
        mgr = WindowManager(storage_path=storage_path, window_size=5)
        for i in range(5):
            mgr.start_new_task(f"RSZ-{i}", f"Task {i}")
            mgr.complete_current_task()

        archived = mgr.resize_window(2)
        assert len(archived) == 3
        assert mgr.completed_task_count() == 2

    def test_grow_does_not_lose_tasks(self, storage_path: str) -> None:
        mgr = WindowManager(storage_path=storage_path, window_size=2)
        for i in range(3):
            mgr.start_new_task(f"GROW-{i}", f"Task {i}")
            mgr.complete_current_task()

        # 2 in window, 1 archived.
        assert mgr.completed_task_count() == 2
        assert mgr.archived_task_count() == 1

        # Growing doesn't un-archive, but it accepts future tasks.
        archived = mgr.resize_window(5)
        assert archived == []
        assert mgr.window_size == 5

    def test_resize_to_same_size(self, manager: WindowManager) -> None:
        manager.start_new_task("SAME-1", "Task")
        manager.complete_current_task()
        archived = manager.resize_window(3)
        assert archived == []

    def test_resize_invalid_low(self, manager: WindowManager) -> None:
        with pytest.raises(ValueError, match="between 1 and 100"):
            manager.resize_window(0)

    def test_resize_invalid_high(self, manager: WindowManager) -> None:
        with pytest.raises(ValueError, match="between 1 and 100"):
            manager.resize_window(101)

    def test_resize_persists(self, storage_path: str) -> None:
        mgr = WindowManager(storage_path=storage_path, window_size=5)
        for i in range(4):
            mgr.start_new_task(f"RP-{i}", f"Task {i}")
            mgr.complete_current_task()
        mgr.resize_window(2)

        mgr2 = WindowManager(storage_path=storage_path)
        assert mgr2.window_size == 2
        assert mgr2.completed_task_count() == 2


# ---------------------------------------------------------------------------
# reload
# ---------------------------------------------------------------------------


class TestReload:
    """Reload discards in-memory state and reads from disk."""

    def test_reload_picks_up_external_changes(self, storage_path: str) -> None:
        mgr = WindowManager(storage_path=storage_path)
        mgr.start_new_task("RELOAD-1", "Original")
        mgr.complete_current_task()

        # Simulate external change: create a second manager and modify state.
        mgr2 = WindowManager(storage_path=storage_path)
        mgr2.start_new_task("RELOAD-2", "External")
        mgr2.complete_current_task()

        # Original manager doesn't see the change yet.
        assert mgr.completed_task_count() == 1

        # After reload, it picks up the external change.
        mgr.reload()
        assert mgr.completed_task_count() == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Various edge cases and boundary conditions."""

    def test_start_complete_many_tasks_sequentially(self, storage_path: str) -> None:
        """Process many tasks through the full lifecycle."""
        mgr = WindowManager(storage_path=storage_path, window_size=3)
        for i in range(20):
            task = mgr.start_new_task(f"SEQ-{i:03d}", f"Task {i}")
            task.add_step(f"Step for task {i}", "implementation")
            mgr.save_current_task()
            mgr.complete_current_task(f"Completed task {i}")

        assert mgr.completed_task_count() == 3
        assert mgr.archived_task_count() == 17

        # All tasks should be retrievable.
        for i in range(20):
            t = mgr.get_task(f"SEQ-{i:03d}")
            assert t is not None
            assert t.status == TaskStatus.COMPLETED

    def test_interleaved_complete_and_fail(self, storage_path: str) -> None:
        mgr = WindowManager(storage_path=storage_path, window_size=2)
        mgr.start_new_task("MIX-1", "Complete")
        mgr.complete_current_task()
        mgr.start_new_task("MIX-2", "Fail")
        mgr.fail_current_task("broken")
        mgr.start_new_task("MIX-3", "Complete")
        mgr.complete_current_task()
        mgr.start_new_task("MIX-4", "Complete")
        mgr.complete_current_task()

        # Window: MIX-3 and MIX-4. Archived: MIX-1 and MIX-2.
        assert mgr.completed_task_count() == 2
        assert mgr.archived_task_count() == 2
        assert mgr.get_task("MIX-2").status == TaskStatus.FAILED

    def test_total_tasks_in_window(self, manager: WindowManager) -> None:
        assert manager.total_tasks_in_window() == 0
        manager.start_new_task("TOT-1", "Current")
        assert manager.total_tasks_in_window() == 1
        manager.complete_current_task()
        assert manager.total_tasks_in_window() == 1  # 1 completed, 0 current
        manager.start_new_task("TOT-2", "Another")
        assert manager.total_tasks_in_window() == 2  # 1 completed + 1 current

    def test_task_with_all_record_types(self, manager: WindowManager) -> None:
        """A task with steps, files, branches, and decisions round-trips."""
        from claude_code_helper_mcp.models.records import BranchAction, FileAction

        task = manager.start_new_task("RICH-1", "Rich task", phase="phase-1")
        task.add_step("Analysed", "analysis")
        task.add_step("Built", "build")
        task.record_file("src/main.py", FileAction.CREATED, "New file")
        task.record_file("src/main.py", FileAction.MODIFIED, "Updated")
        task.record_branch("feature/RICH-1", BranchAction.CREATED, "main")
        task.add_decision("Use Python", "Matches project stack", ["Go", "Rust"])
        manager.save_current_task()
        manager.complete_current_task("Rich task done")

        loaded = manager.get_task("RICH-1")
        assert loaded is not None
        assert len(loaded.steps) == 2
        assert len(loaded.files) == 1
        assert len(loaded.files[0].action_history) == 1
        assert len(loaded.branches) == 1
        assert len(loaded.decisions) == 1
