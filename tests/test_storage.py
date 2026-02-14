"""Tests for MemoryStore -- file-based storage engine.

All tests use real files in temporary directories with real TaskMemory and
MemoryWindow objects.  No mocks, no stubs, no fakes.
"""

import json
import os
import stat
import tempfile
from pathlib import Path

import pytest

from claude_code_helper_mcp.models.task import TaskMemory
from claude_code_helper_mcp.models.window import MemoryWindow
from claude_code_helper_mcp.storage.store import (
    DEFAULT_STORAGE_DIR,
    TASK_FILE_EXTENSION,
    TASK_FILE_PREFIX,
    TASKS_SUBDIR,
    WINDOW_STATE_FILE,
    MemoryStore,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_store(tmp_path: Path) -> MemoryStore:
    """Return a MemoryStore rooted in a fresh temp directory."""
    return MemoryStore(str(tmp_path / ".claude-memory"))


@pytest.fixture()
def sample_task() -> TaskMemory:
    """Return a minimal TaskMemory for testing."""
    return TaskMemory(ticket_id="CMH-100", title="Test storage", phase="phase-1")


@pytest.fixture()
def rich_task() -> TaskMemory:
    """Return a TaskMemory with steps, files, decisions populated."""
    task = TaskMemory(ticket_id="CMH-200", title="Rich task", phase="phase-2")
    task.add_step("Analysed the requirements", "analysis")
    task.add_step("Implemented the feature", "implementation")
    task.record_file("src/main.py", "modified", "Updated main entry point")
    task.record_branch("feature/CMH-200", "created", "Feature branch")
    task.add_decision(
        decision="Use JSON for storage",
        reasoning="Simple and human-readable",
        alternatives=["SQLite", "TOML"],
    )
    return task


# ---------------------------------------------------------------------------
# Directory creation
# ---------------------------------------------------------------------------


class TestDirectoryCreation:
    """MemoryStore auto-creates the directory tree on init."""

    def test_creates_root_directory(self, tmp_path: Path) -> None:
        store = MemoryStore(str(tmp_path / "new-dir"))
        assert store.storage_root.is_dir()

    def test_creates_tasks_subdirectory(self, tmp_path: Path) -> None:
        store = MemoryStore(str(tmp_path / "new-dir"))
        assert store.tasks_directory.is_dir()
        assert store.tasks_directory.name == TASKS_SUBDIR

    def test_nested_path_creation(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c" / ".memory"
        store = MemoryStore(str(deep))
        assert store.storage_root.is_dir()
        assert store.tasks_directory.is_dir()

    def test_idempotent_on_existing_directory(self, tmp_store: MemoryStore) -> None:
        """Creating a second MemoryStore on the same path must not fail."""
        second = MemoryStore(str(tmp_store.storage_root))
        assert second.storage_root == tmp_store.storage_root

    def test_default_storage_dir_name(self) -> None:
        assert DEFAULT_STORAGE_DIR == ".claude-memory"


# ---------------------------------------------------------------------------
# Task save / load round-trip
# ---------------------------------------------------------------------------


class TestTaskSaveLoad:
    """Persist and recover TaskMemory objects."""

    def test_save_returns_path(
        self, tmp_store: MemoryStore, sample_task: TaskMemory
    ) -> None:
        result = tmp_store.save_task(sample_task)
        assert isinstance(result, Path)
        assert result.is_file()

    def test_saved_file_is_valid_json(
        self, tmp_store: MemoryStore, sample_task: TaskMemory
    ) -> None:
        path = tmp_store.save_task(sample_task)
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        assert data["ticket_id"] == "CMH-100"

    def test_load_round_trip(
        self, tmp_store: MemoryStore, sample_task: TaskMemory
    ) -> None:
        tmp_store.save_task(sample_task)
        loaded = tmp_store.load_task("CMH-100")
        assert loaded is not None
        assert loaded.ticket_id == sample_task.ticket_id
        assert loaded.title == sample_task.title
        assert loaded.phase == sample_task.phase

    def test_rich_task_round_trip(
        self, tmp_store: MemoryStore, rich_task: TaskMemory
    ) -> None:
        tmp_store.save_task(rich_task)
        loaded = tmp_store.load_task("CMH-200")
        assert loaded is not None
        assert len(loaded.steps) == 2
        assert len(loaded.files) == 1
        assert len(loaded.branches) == 1
        assert len(loaded.decisions) == 1
        assert loaded.decisions[0].decision == "Use JSON for storage"

    def test_overwrite_existing_task(
        self, tmp_store: MemoryStore, sample_task: TaskMemory
    ) -> None:
        tmp_store.save_task(sample_task)
        sample_task.add_step("Extra step", "extra")
        tmp_store.save_task(sample_task)
        loaded = tmp_store.load_task("CMH-100")
        assert loaded is not None
        assert len(loaded.steps) == 1

    def test_load_nonexistent_returns_none(self, tmp_store: MemoryStore) -> None:
        assert tmp_store.load_task("DOES-NOT-EXIST") is None


# ---------------------------------------------------------------------------
# Task listing
# ---------------------------------------------------------------------------


class TestTaskList:
    """list_tasks scans the tasks directory for persisted IDs."""

    def test_empty_store(self, tmp_store: MemoryStore) -> None:
        assert tmp_store.list_tasks() == []

    def test_single_task(
        self, tmp_store: MemoryStore, sample_task: TaskMemory
    ) -> None:
        tmp_store.save_task(sample_task)
        ids = tmp_store.list_tasks()
        assert ids == ["CMH-100"]

    def test_multiple_tasks_sorted(self, tmp_store: MemoryStore) -> None:
        for tid in ("CMH-003", "CMH-001", "CMH-002"):
            task = TaskMemory(ticket_id=tid, title=f"Task {tid}", phase="phase-1")
            tmp_store.save_task(task)
        assert tmp_store.list_tasks() == ["CMH-001", "CMH-002", "CMH-003"]

    def test_ignores_non_task_files(self, tmp_store: MemoryStore) -> None:
        # Drop a rogue file into the tasks directory
        rogue = tmp_store.tasks_directory / "notes.txt"
        rogue.write_text("not a task")
        assert tmp_store.list_tasks() == []

    def test_ignores_directories(self, tmp_store: MemoryStore) -> None:
        subdir = tmp_store.tasks_directory / f"{TASK_FILE_PREFIX}fake{TASK_FILE_EXTENSION}"
        subdir.mkdir()
        # Directories matching the name pattern must be excluded
        assert tmp_store.list_tasks() == []


# ---------------------------------------------------------------------------
# Task deletion
# ---------------------------------------------------------------------------


class TestTaskDelete:
    """delete_task removes a task file from disk."""

    def test_delete_existing(
        self, tmp_store: MemoryStore, sample_task: TaskMemory
    ) -> None:
        tmp_store.save_task(sample_task)
        assert tmp_store.delete_task("CMH-100") is True
        assert not tmp_store.task_exists("CMH-100")

    def test_delete_nonexistent(self, tmp_store: MemoryStore) -> None:
        assert tmp_store.delete_task("GHOST") is False

    def test_delete_then_list(
        self, tmp_store: MemoryStore, sample_task: TaskMemory
    ) -> None:
        tmp_store.save_task(sample_task)
        tmp_store.delete_task("CMH-100")
        assert tmp_store.list_tasks() == []


# ---------------------------------------------------------------------------
# task_exists
# ---------------------------------------------------------------------------


class TestTaskExists:
    """task_exists returns True iff the file is on disk."""

    def test_exists_true(
        self, tmp_store: MemoryStore, sample_task: TaskMemory
    ) -> None:
        tmp_store.save_task(sample_task)
        assert tmp_store.task_exists("CMH-100") is True

    def test_exists_false(self, tmp_store: MemoryStore) -> None:
        assert tmp_store.task_exists("NOPE") is False


# ---------------------------------------------------------------------------
# Window save / load
# ---------------------------------------------------------------------------


class TestWindowSaveLoad:
    """Persist and recover MemoryWindow state."""

    def test_save_returns_path(self, tmp_store: MemoryStore) -> None:
        window = MemoryWindow()
        path = tmp_store.save_window(window)
        assert isinstance(path, Path)
        assert path.is_file()

    def test_load_round_trip(self, tmp_store: MemoryStore) -> None:
        window = MemoryWindow()
        window.start_task("CMH-050", "Window task")
        tmp_store.save_window(window)
        loaded = tmp_store.load_window()
        assert loaded.current_task is not None
        assert loaded.current_task.ticket_id == "CMH-050"

    def test_load_missing_returns_empty_window(self, tmp_path: Path) -> None:
        store = MemoryStore(str(tmp_path / "empty"))
        window = store.load_window()
        assert isinstance(window, MemoryWindow)
        assert window.current_task is None
        assert window.completed_tasks == []

    def test_window_exists_true(self, tmp_store: MemoryStore) -> None:
        tmp_store.save_window(MemoryWindow())
        assert tmp_store.window_exists() is True

    def test_window_exists_false(self, tmp_path: Path) -> None:
        store = MemoryStore(str(tmp_path / "empty2"))
        assert store.window_exists() is False


# ---------------------------------------------------------------------------
# Corrupt / malformed file handling
# ---------------------------------------------------------------------------


class TestCorruptFileHandling:
    """Graceful degradation when files are corrupt or unreadable."""

    def test_load_corrupt_task_returns_none(self, tmp_store: MemoryStore) -> None:
        corrupt = tmp_store.tasks_directory / f"{TASK_FILE_PREFIX}BAD{TASK_FILE_EXTENSION}"
        corrupt.write_text("{{{invalid json", encoding="utf-8")
        assert tmp_store.load_task("BAD") is None

    def test_load_empty_task_file_returns_none(self, tmp_store: MemoryStore) -> None:
        empty = tmp_store.tasks_directory / f"{TASK_FILE_PREFIX}EMPTY{TASK_FILE_EXTENSION}"
        empty.write_text("", encoding="utf-8")
        assert tmp_store.load_task("EMPTY") is None

    def test_load_task_wrong_schema_returns_none(
        self, tmp_store: MemoryStore
    ) -> None:
        wrong = tmp_store.tasks_directory / f"{TASK_FILE_PREFIX}WRONG{TASK_FILE_EXTENSION}"
        wrong.write_text('{"not_a_task": true}', encoding="utf-8")
        assert tmp_store.load_task("WRONG") is None

    def test_load_corrupt_window_returns_new_window(
        self, tmp_store: MemoryStore
    ) -> None:
        window_path = tmp_store.storage_root / WINDOW_STATE_FILE
        window_path.write_text("<<<not json>>>", encoding="utf-8")
        window = tmp_store.load_window()
        assert isinstance(window, MemoryWindow)
        assert window.current_task is None


# ---------------------------------------------------------------------------
# Atomic write behaviour
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    """The _atomic_write method guarantees atomicity via temp+rename."""

    def test_file_has_trailing_newline(
        self, tmp_store: MemoryStore, sample_task: TaskMemory
    ) -> None:
        path = tmp_store.save_task(sample_task)
        content = path.read_text(encoding="utf-8")
        assert content.endswith("\n")

    def test_file_is_formatted_json(
        self, tmp_store: MemoryStore, sample_task: TaskMemory
    ) -> None:
        path = tmp_store.save_task(sample_task)
        raw = path.read_text(encoding="utf-8")
        # Formatted with indent=2 means the second line starts with spaces
        lines = raw.strip().splitlines()
        assert len(lines) > 1
        assert lines[1].startswith("  ")

    def test_no_temp_files_left_on_success(
        self, tmp_store: MemoryStore, sample_task: TaskMemory
    ) -> None:
        tmp_store.save_task(sample_task)
        tmp_files = list(tmp_store.tasks_directory.glob(".tmp_*"))
        assert tmp_files == []


# ---------------------------------------------------------------------------
# Filename sanitisation
# ---------------------------------------------------------------------------


class TestFilenameSanitisation:
    """Ticket IDs with special characters are safely sanitised."""

    def test_slashes_replaced(self) -> None:
        assert "/" not in MemoryStore._sanitise_filename("a/b")
        assert "\\" not in MemoryStore._sanitise_filename("a\\b")

    def test_colons_replaced(self) -> None:
        assert ":" not in MemoryStore._sanitise_filename("a:b")

    def test_angle_brackets_replaced(self) -> None:
        result = MemoryStore._sanitise_filename("<id>")
        assert "<" not in result
        assert ">" not in result

    def test_whitespace_stripped(self) -> None:
        assert MemoryStore._sanitise_filename("  CMH-001  ") == "CMH-001"

    def test_normal_id_unchanged(self) -> None:
        assert MemoryStore._sanitise_filename("CMH-003") == "CMH-003"

    def test_round_trip_with_sanitised_id(self, tmp_store: MemoryStore) -> None:
        """A task with a special-char ID can be saved and loaded back."""
        task = TaskMemory(ticket_id="MY/PROJ:42", title="Odd ID", phase="phase-1")
        tmp_store.save_task(task)
        loaded = tmp_store.load_task("MY/PROJ:42")
        assert loaded is not None
        assert loaded.ticket_id == "MY/PROJ:42"


# ---------------------------------------------------------------------------
# Introspection properties
# ---------------------------------------------------------------------------


class TestIntrospection:
    """storage_root and tasks_directory properties."""

    def test_storage_root(self, tmp_store: MemoryStore) -> None:
        assert tmp_store.storage_root.is_dir()

    def test_tasks_directory_is_subdir_of_root(
        self, tmp_store: MemoryStore
    ) -> None:
        assert tmp_store.tasks_directory.parent == tmp_store.storage_root

    def test_window_file_location(self, tmp_store: MemoryStore) -> None:
        tmp_store.save_window(MemoryWindow())
        window_path = tmp_store.storage_root / WINDOW_STATE_FILE
        assert window_path.is_file()


# ---------------------------------------------------------------------------
# Integration: task + window together
# ---------------------------------------------------------------------------


class TestIntegration:
    """MemoryStore handles tasks and window side-by-side."""

    def test_save_task_and_window_together(self, tmp_store: MemoryStore) -> None:
        task = TaskMemory(ticket_id="CMH-INT-1", title="Integration", phase="phase-1")
        window = MemoryWindow()
        window.start_task("CMH-INT-1", "Integration")

        tmp_store.save_task(task)
        tmp_store.save_window(window)

        loaded_task = tmp_store.load_task("CMH-INT-1")
        loaded_window = tmp_store.load_window()

        assert loaded_task is not None
        assert loaded_task.ticket_id == "CMH-INT-1"
        assert loaded_window.current_task is not None
        assert loaded_window.current_task.ticket_id == "CMH-INT-1"

    def test_multiple_tasks_with_window(self, tmp_store: MemoryStore) -> None:
        window = MemoryWindow()

        # Start and complete a task
        window.start_task("CMH-A", "First")
        t1 = TaskMemory(ticket_id="CMH-A", title="First", phase="phase-1")
        t1.complete("Done with first task")
        tmp_store.save_task(t1)
        window.complete_current_task()

        # Start a second task
        window.start_task("CMH-B", "Second")
        t2 = TaskMemory(ticket_id="CMH-B", title="Second", phase="phase-1")
        tmp_store.save_task(t2)

        tmp_store.save_window(window)

        # Verify everything round-trips
        ids = tmp_store.list_tasks()
        assert "CMH-A" in ids
        assert "CMH-B" in ids
        loaded_window = tmp_store.load_window()
        assert loaded_window.current_task.ticket_id == "CMH-B"
        assert len(loaded_window.completed_tasks) == 1
