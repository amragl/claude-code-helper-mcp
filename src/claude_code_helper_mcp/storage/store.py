"""MemoryStore -- file-based storage engine for task memory persistence.

Handles all file I/O for saving and loading TaskMemory and MemoryWindow
objects. Uses atomic writes (write-to-temp + rename) to prevent data
corruption on crashes, and creates the `.claude-memory/` directory tree
automatically on first use.

Typical usage::

    store = MemoryStore()                    # uses .claude-memory/ in cwd
    store = MemoryStore("/path/to/project")  # explicit project root

    # Task operations
    store.save_task(task_memory)
    task = store.load_task("CMH-003")
    task_ids = store.list_tasks()
    store.delete_task("CMH-001")

    # Window operations
    store.save_window(window)
    window = store.load_window()
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from claude_code_helper_mcp.models.task import TaskMemory
from claude_code_helper_mcp.models.window import MemoryWindow

logger = logging.getLogger(__name__)

# Default directory name for memory storage, placed at the project root.
DEFAULT_STORAGE_DIR = ".claude-memory"

# Subdirectories within the storage root.
TASKS_SUBDIR = "tasks"

# File name for the persisted window state.
WINDOW_STATE_FILE = "window.json"

# File prefix and extension for individual task files.
TASK_FILE_PREFIX = "task-"
TASK_FILE_EXTENSION = ".json"


class MemoryStore:
    """File-based storage engine for task memory.

    Persists ``TaskMemory`` objects as individual JSON files in
    ``<storage_path>/tasks/`` and the ``MemoryWindow`` state as
    ``<storage_path>/window.json``.

    All writes are atomic: data is first written to a temporary file in the
    same directory and then renamed into place. This guarantees that a crash
    mid-write never produces a truncated or corrupt file.

    Parameters
    ----------
    storage_path:
        Absolute or relative path to the storage directory.  When *None*,
        defaults to ``<cwd>/.claude-memory/``.
    """

    def __init__(self, storage_path: Optional[str] = None) -> None:
        if storage_path is not None:
            self._root = Path(storage_path).resolve()
        else:
            self._root = Path.cwd() / DEFAULT_STORAGE_DIR

        self._tasks_dir = self._root / TASKS_SUBDIR
        self._window_path = self._root / WINDOW_STATE_FILE

        # Ensure the directory tree exists the first time we touch it.
        self._ensure_directories()

    # ------------------------------------------------------------------
    # Public API -- Task operations
    # ------------------------------------------------------------------

    def save_task(self, task: TaskMemory) -> Path:
        """Persist a TaskMemory to disk.

        Writes ``<tasks_dir>/task-<ticket_id>.json`` atomically.

        Parameters
        ----------
        task:
            The task memory to save.

        Returns
        -------
        Path
            The path to the written file.
        """
        file_path = self._task_path(task.ticket_id)
        data = task.to_json_dict()
        self._atomic_write(file_path, data)
        logger.info("Saved task %s to %s", task.ticket_id, file_path)
        return file_path

    def load_task(self, ticket_id: str) -> Optional[TaskMemory]:
        """Load a TaskMemory from disk.

        Parameters
        ----------
        ticket_id:
            The ticket identifier (e.g., ``"CMH-003"``).

        Returns
        -------
        TaskMemory or None
            The loaded task, or *None* if the file does not exist or is corrupt.
        """
        file_path = self._task_path(ticket_id)
        data = self._safe_read_json(file_path)
        if data is None:
            return None
        try:
            return TaskMemory.from_json_dict(data)
        except Exception:
            logger.warning(
                "Failed to deserialize task from %s. File may be corrupt.",
                file_path,
                exc_info=True,
            )
            return None

    def list_tasks(self) -> list[str]:
        """Return a sorted list of all persisted ticket IDs.

        Scans the tasks directory for files matching the naming convention
        ``task-<ticket_id>.json`` and extracts the ticket IDs.

        Returns
        -------
        list[str]
            Sorted ticket IDs found on disk.
        """
        if not self._tasks_dir.is_dir():
            return []

        ids: list[str] = []
        for entry in self._tasks_dir.iterdir():
            if (
                entry.is_file()
                and entry.name.startswith(TASK_FILE_PREFIX)
                and entry.name.endswith(TASK_FILE_EXTENSION)
            ):
                ticket_id = entry.name[
                    len(TASK_FILE_PREFIX) : -len(TASK_FILE_EXTENSION)
                ]
                if ticket_id:
                    ids.append(ticket_id)

        return sorted(ids)

    def delete_task(self, ticket_id: str) -> bool:
        """Delete a persisted task file.

        Parameters
        ----------
        ticket_id:
            The ticket identifier to delete.

        Returns
        -------
        bool
            *True* if the file existed and was deleted, *False* otherwise.
        """
        file_path = self._task_path(ticket_id)
        try:
            file_path.unlink()
            logger.info("Deleted task file %s", file_path)
            return True
        except FileNotFoundError:
            logger.debug("Task file %s does not exist, nothing to delete.", file_path)
            return False
        except OSError:
            logger.warning(
                "Failed to delete task file %s",
                file_path,
                exc_info=True,
            )
            return False

    def task_exists(self, ticket_id: str) -> bool:
        """Check whether a task file exists on disk."""
        return self._task_path(ticket_id).is_file()

    # ------------------------------------------------------------------
    # Public API -- Window operations
    # ------------------------------------------------------------------

    def save_window(self, window: MemoryWindow) -> Path:
        """Persist the MemoryWindow state to disk.

        Writes ``<storage_root>/window.json`` atomically.

        Parameters
        ----------
        window:
            The window state to save.

        Returns
        -------
        Path
            The path to the written file.
        """
        data = window.to_json_dict()
        self._atomic_write(self._window_path, data)
        logger.info("Saved window state to %s", self._window_path)
        return self._window_path

    def load_window(self) -> MemoryWindow:
        """Load the MemoryWindow state from disk.

        If no window file exists, returns a new empty ``MemoryWindow``.
        If the file is corrupt, logs a warning and returns a new window.

        Returns
        -------
        MemoryWindow
            The loaded or freshly created window.
        """
        data = self._safe_read_json(self._window_path)
        if data is None:
            logger.info(
                "No window state found at %s. Creating new empty window.",
                self._window_path,
            )
            return MemoryWindow()
        try:
            return MemoryWindow.from_json_dict(data)
        except Exception:
            logger.warning(
                "Failed to deserialize window from %s. Returning new empty window.",
                self._window_path,
                exc_info=True,
            )
            return MemoryWindow()

    def window_exists(self) -> bool:
        """Check whether a window state file exists on disk."""
        return self._window_path.is_file()

    # ------------------------------------------------------------------
    # Public API -- Introspection
    # ------------------------------------------------------------------

    @property
    def storage_root(self) -> Path:
        """The resolved root directory used for storage."""
        return self._root

    @property
    def tasks_directory(self) -> Path:
        """The directory where individual task files are stored."""
        return self._tasks_dir

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_directories(self) -> None:
        """Create the storage directory tree if it does not exist."""
        self._tasks_dir.mkdir(parents=True, exist_ok=True)

    def _task_path(self, ticket_id: str) -> Path:
        """Return the file path for a given ticket ID."""
        sanitised = self._sanitise_filename(ticket_id)
        return self._tasks_dir / f"{TASK_FILE_PREFIX}{sanitised}{TASK_FILE_EXTENSION}"

    @staticmethod
    def _sanitise_filename(ticket_id: str) -> str:
        """Sanitise a ticket ID for safe use as a filename component.

        Replaces characters that are problematic on common file systems
        (``/``, ``\\``, ``:``, ``*``, ``?``, ``"``, ``<``, ``>``, ``|``)
        with hyphens and strips leading/trailing whitespace.
        """
        sanitised = ticket_id.strip()
        for ch in r'/\:*?"<>|':
            sanitised = sanitised.replace(ch, "-")
        return sanitised

    def _atomic_write(self, target: Path, data: dict) -> None:
        """Write *data* as formatted JSON to *target* atomically.

        The strategy is write-to-temp-then-rename:

        1. Create a temporary file in the **same directory** as *target* (so
           the rename is guaranteed to be atomic on POSIX systems).
        2. Write the JSON payload to the temp file.
        3. Flush and ``fsync`` the file descriptor to ensure the data reaches
           the storage device.
        4. Rename the temp file over the target path.

        If anything fails before the rename, the temp file is cleaned up and
        the original target file (if any) is left untouched.
        """
        target.parent.mkdir(parents=True, exist_ok=True)

        fd = None
        tmp_path: Optional[str] = None
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=str(target.parent),
                prefix=".tmp_",
                suffix=".json",
            )
            with os.fdopen(fd, "w", encoding="utf-8") as fp:
                fd = None  # os.fdopen takes ownership of the fd
                json.dump(data, fp, indent=2, ensure_ascii=False)
                fp.write("\n")  # trailing newline for POSIX friendliness
                fp.flush()
                os.fsync(fp.fileno())

            # Atomic rename (POSIX guarantees atomicity for same-filesystem rename)
            os.replace(tmp_path, str(target))
            tmp_path = None  # rename succeeded; nothing to clean up

        except BaseException:
            # Clean up the temp file on any error.
            if fd is not None:
                os.close(fd)
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            raise

    def _safe_read_json(self, path: Path) -> Optional[dict]:
        """Read and parse a JSON file, returning *None* on any failure.

        Handles missing files, permission errors, and malformed JSON
        gracefully -- logs a warning and returns *None* instead of raising.
        """
        if not path.is_file():
            return None
        try:
            with open(path, "r", encoding="utf-8") as fp:
                return json.load(fp)
        except json.JSONDecodeError:
            logger.warning(
                "Corrupt JSON in %s. The file will be ignored.",
                path,
                exc_info=True,
            )
            return None
        except OSError:
            logger.warning(
                "Could not read %s.",
                path,
                exc_info=True,
            )
            return None
