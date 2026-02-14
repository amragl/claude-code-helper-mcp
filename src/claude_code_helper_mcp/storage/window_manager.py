"""WindowManager -- high-level coordinator for the sliding window lifecycle.

Bridges the MemoryWindow (in-memory model) and MemoryStore (persistence) to
provide a single API for task lifecycle management.  Every mutation is
automatically persisted so callers never need to remember to save.

Responsibilities:
- Starting new tasks (with automatic persistence of both task and window).
- Completing tasks (with automatic window rotation and archival persistence).
- Failing tasks (same archival handling as completion).
- Retrieving tasks from the live window or from the archive on disk.
- Reconfiguring the window size at runtime.

Typical usage::

    manager = WindowManager(storage_path="/path/to/project/.claude-memory")
    task = manager.start_new_task("CMH-004", "Sliding window manager", phase="phase-1")
    task.add_step("Implemented the class", "implementation")
    manager.save_current_task()         # persist incremental progress
    manager.complete_current_task("All done!")

All methods that mutate state persist immediately.  Read-only methods
(``get_task``, ``get_current_task``, etc.) do not trigger writes.
"""

from __future__ import annotations

import logging
from typing import Optional

from claude_code_helper_mcp.models.task import TaskMemory, TaskStatus
from claude_code_helper_mcp.models.window import MemoryWindow
from claude_code_helper_mcp.storage.store import MemoryStore

logger = logging.getLogger(__name__)


class WindowManager:
    """High-level manager for the sliding window task lifecycle.

    Wraps a :class:`MemoryStore` and a :class:`MemoryWindow`, keeping them
    in sync.  The manager owns the canonical ``MemoryWindow`` instance; all
    mutations go through the manager so that persistence is guaranteed.

    Parameters
    ----------
    storage_path:
        Path passed to :class:`MemoryStore`.  When *None*, uses the default
        ``<cwd>/.claude-memory/`` directory.
    window_size:
        Override the default window size (number of completed tasks to
        retain).  When *None*, the persisted window's ``window_size`` is used,
        or ``3`` for a fresh window.
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        window_size: Optional[int] = None,
    ) -> None:
        self._store = MemoryStore(storage_path)
        self._window = self._store.load_window()

        if window_size is not None:
            self._window.window_size = window_size
            # Enforce the new limit in case the loaded window had more tasks.
            self._window._enforce_window_size()
            self._persist_window()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def store(self) -> MemoryStore:
        """The underlying :class:`MemoryStore`."""
        return self._store

    @property
    def window(self) -> MemoryWindow:
        """The current in-memory :class:`MemoryWindow`."""
        return self._window

    @property
    def window_size(self) -> int:
        """The configured maximum number of completed tasks in the window."""
        return self._window.window_size

    # ------------------------------------------------------------------
    # Task lifecycle
    # ------------------------------------------------------------------

    def start_new_task(
        self,
        ticket_id: str,
        title: str,
        phase: Optional[str] = None,
    ) -> TaskMemory:
        """Start a new task and make it the current active task.

        Creates the task in the window, persists the task file, and saves
        the updated window state.

        Parameters
        ----------
        ticket_id:
            The ticket identifier (e.g., ``"CMH-004"``).
        title:
            Human-readable title.
        phase:
            Optional roadmap phase.

        Returns
        -------
        TaskMemory
            The newly created and persisted task.

        Raises
        ------
        ValueError
            If there is already an active task that has not been completed
            or failed.
        """
        task = self._window.start_task(ticket_id, title, phase)
        self._store.save_task(task)
        self._persist_window()
        logger.info("Started new task %s ('%s').", ticket_id, title)
        return task

    def complete_current_task(self, summary: str = "") -> TaskMemory:
        """Complete the current task and rotate the window.

        The task is marked as completed, the window rotation is applied
        (archiving the oldest completed task if the window overflows),
        and all affected state is persisted.

        Archived tasks are preserved on disk even after they leave the
        window -- the individual task files remain in the store.

        Parameters
        ----------
        summary:
            Optional completion summary.

        Returns
        -------
        TaskMemory
            The completed task.

        Raises
        ------
        ValueError
            If there is no active current task.
        """
        # Capture IDs before rotation so we can detect newly archived tasks.
        ids_before = set(self._window.archived_task_ids)

        completed_task = self._window.complete_current_task(summary)

        # Persist the completed task (with updated status and timestamp).
        self._store.save_task(completed_task)

        # Detect tasks that were just archived by the rotation.
        ids_after = set(self._window.archived_task_ids)
        newly_archived = ids_after - ids_before
        for tid in newly_archived:
            logger.info(
                "Task %s archived (rotated out of window).", tid
            )

        self._persist_window()
        logger.info(
            "Completed task %s. Window: %d completed, %d archived total.",
            completed_task.ticket_id,
            len(self._window.completed_tasks),
            len(self._window.archived_task_ids),
        )
        return completed_task

    def fail_current_task(self, reason: str = "") -> TaskMemory:
        """Mark the current task as failed and rotate the window.

        Failed tasks occupy window slots just like completed ones.  If the
        window overflows, the oldest completed task is archived.

        Parameters
        ----------
        reason:
            Reason for the failure.

        Returns
        -------
        TaskMemory
            The failed task.

        Raises
        ------
        ValueError
            If there is no active current task.
        """
        ids_before = set(self._window.archived_task_ids)

        failed_task = self._window.fail_current_task(reason)

        self._store.save_task(failed_task)

        ids_after = set(self._window.archived_task_ids)
        newly_archived = ids_after - ids_before
        for tid in newly_archived:
            logger.info("Task %s archived during fail rotation.", tid)

        self._persist_window()
        logger.info("Failed task %s. Reason: %s", failed_task.ticket_id, reason or "(none)")
        return failed_task

    def save_current_task(self) -> Optional[TaskMemory]:
        """Persist the current task without changing its status.

        Useful for saving incremental progress (new steps, files, decisions)
        during task execution.

        Returns
        -------
        TaskMemory or None
            The saved task, or *None* if there is no current task.
        """
        if self._window.current_task is None:
            logger.debug("No current task to save.")
            return None

        self._store.save_task(self._window.current_task)
        self._persist_window()
        logger.debug("Saved incremental progress for %s.", self._window.current_task.ticket_id)
        return self._window.current_task

    # ------------------------------------------------------------------
    # Task retrieval
    # ------------------------------------------------------------------

    def get_current_task(self) -> Optional[TaskMemory]:
        """Return the current active task, or *None*.

        This returns the in-memory task -- no disk read required.
        """
        return self._window.current_task

    def get_task(self, ticket_id: str) -> Optional[TaskMemory]:
        """Retrieve a task by ticket ID.

        Searches in this order:
        1. Current task (in-memory)
        2. Completed tasks in the window (in-memory)
        3. Archived task files on disk

        Parameters
        ----------
        ticket_id:
            The ticket identifier to look up.

        Returns
        -------
        TaskMemory or None
            The task if found anywhere, *None* otherwise.
        """
        # 1. Check in-memory window first (fast path).
        task = self._window.get_task(ticket_id)
        if task is not None:
            return task

        # 2. Fall back to disk (archived tasks).
        return self._store.load_task(ticket_id)

    def has_active_task(self) -> bool:
        """Check whether there is an active (in-progress) task."""
        return self._window.has_active_task()

    def is_task_archived(self, ticket_id: str) -> bool:
        """Check whether a task has been archived (rotated out of the window)."""
        return self._window.is_task_archived(ticket_id)

    # ------------------------------------------------------------------
    # Window queries
    # ------------------------------------------------------------------

    def get_all_task_ids(self) -> list[str]:
        """Return all task IDs currently in the window (completed + current)."""
        return self._window.get_all_task_ids()

    def get_all_known_task_ids(self) -> list[str]:
        """Return all task IDs the manager knows about.

        Includes tasks in the window, archived IDs, and any task files
        found on disk.  The result is a sorted, deduplicated list.
        """
        ids = set(self._window.get_all_task_ids())
        ids.update(self._window.archived_task_ids)
        ids.update(self._store.list_tasks())
        return sorted(ids)

    def total_tasks_in_window(self) -> int:
        """Return the count of tasks currently in the window (completed + current)."""
        return self._window.total_tasks_in_window()

    def completed_task_count(self) -> int:
        """Return the count of completed tasks in the window."""
        return len(self._window.completed_tasks)

    def archived_task_count(self) -> int:
        """Return the count of archived task IDs."""
        return len(self._window.archived_task_ids)

    # ------------------------------------------------------------------
    # Window reconfiguration
    # ------------------------------------------------------------------

    def resize_window(self, new_size: int) -> list[str]:
        """Change the window size, archiving overflow tasks if necessary.

        Parameters
        ----------
        new_size:
            The new maximum number of completed tasks to retain.
            Must be >= 1 and <= 100.

        Returns
        -------
        list[str]
            Ticket IDs of tasks that were archived as a result of the resize.

        Raises
        ------
        ValueError
            If *new_size* is outside the allowed range [1, 100].
        """
        if new_size < 1 or new_size > 100:
            raise ValueError(f"Window size must be between 1 and 100, got {new_size}.")

        ids_before = set(self._window.archived_task_ids)

        self._window.window_size = new_size
        self._window._enforce_window_size()

        ids_after = set(self._window.archived_task_ids)
        newly_archived = sorted(ids_after - ids_before)

        self._persist_window()
        logger.info(
            "Resized window to %d. Newly archived: %s.",
            new_size,
            newly_archived or "(none)",
        )
        return newly_archived

    # ------------------------------------------------------------------
    # Reload
    # ------------------------------------------------------------------

    def reload(self) -> None:
        """Reload the window state from disk.

        Discards the in-memory window and reads the latest persisted state.
        Useful if external processes may have modified the storage.
        """
        self._window = self._store.load_window()
        logger.info("Reloaded window state from disk.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _persist_window(self) -> None:
        """Save the current window state to disk."""
        self._store.save_window(self._window)
