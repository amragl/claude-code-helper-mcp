"""MemoryWindow model -- the sliding window container for task memories.

The MemoryWindow enforces a fixed capacity: it holds the current active task
plus the N most recently completed tasks (default N=3). When capacity is
exceeded, the oldest completed task is archived.

This structure is the top-level persistence unit: the storage engine saves
and loads MemoryWindow instances.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from claude_code_helper_mcp.models.task import TaskMemory, TaskStatus


class MemoryWindow(BaseModel):
    """Sliding window of task memories.

    Maintains a current task (if any) plus a bounded list of recently
    completed tasks. The window enforces its size limit: when a task is
    completed and the completed list exceeds the configured size, the
    oldest completed task is removed and its ID is recorded in the
    archived list.

    The default window size is 3 completed tasks + 1 current.
    """

    window_size: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Maximum number of completed tasks to retain (not counting current).",
    )
    current_task: Optional[TaskMemory] = Field(
        default=None,
        description="The currently active task, or None if no task is in progress.",
    )
    completed_tasks: list[TaskMemory] = Field(
        default_factory=list,
        description="Recently completed tasks, ordered oldest to newest.",
    )
    archived_task_ids: list[str] = Field(
        default_factory=list,
        description="IDs of tasks that were rotated out of the window.",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this memory window was first created (UTC).",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this memory window was last updated (UTC).",
    )


    @model_validator(mode="after")
    def validate_completed_within_limit(self) -> "MemoryWindow":
        """Ensure completed_tasks does not exceed window_size on load."""
        while len(self.completed_tasks) > self.window_size:
            oldest = self.completed_tasks.pop(0)
            if oldest.ticket_id not in self.archived_task_ids:
                self.archived_task_ids.append(oldest.ticket_id)
        return self

    def start_task(self, ticket_id: str, title: str, phase: Optional[str] = None) -> TaskMemory:
        """Start a new task, making it the current task.

        If there is already an active task, it must be completed first.

        Args:
            ticket_id: The ticket identifier.
            title: Human-readable task title.
            phase: Optional roadmap phase.

        Returns:
            The newly created TaskMemory.

        Raises:
            ValueError: If there is already an active (uncompleted) task.
        """
        if self.current_task is not None and self.current_task.status == TaskStatus.ACTIVE:
            raise ValueError(
                f"Cannot start a new task while task '{self.current_task.ticket_id}' "
                f"is still active. Complete or fail it first."
            )

        task = TaskMemory(
            ticket_id=ticket_id,
            title=title,
            phase=phase,
        )
        self.current_task = task
        self._touch()
        return task

    def complete_current_task(self, summary: str = "") -> TaskMemory:
        """Complete the current task and move it into the completed window.

        The task is marked as completed and added to the end of the
        completed_tasks list. If the list exceeds window_size, the oldest
        task is archived.

        Args:
            summary: Optional completion summary.

        Returns:
            The completed TaskMemory.

        Raises:
            ValueError: If there is no active current task.
        """
        if self.current_task is None:
            raise ValueError("No current task to complete.")
        if self.current_task.status != TaskStatus.ACTIVE:
            raise ValueError(
                f"Current task '{self.current_task.ticket_id}' is not active "
                f"(status: {self.current_task.status.value})."
            )

        self.current_task.complete(summary)
        self.completed_tasks.append(self.current_task)
        completed_task = self.current_task
        self.current_task = None

        self._enforce_window_size()
        self._touch()
        return completed_task

    def fail_current_task(self, reason: str = "") -> TaskMemory:
        """Mark the current task as failed and move it into the completed window.

        Failed tasks still count toward the window size.

        Args:
            reason: Reason for the failure.

        Returns:
            The failed TaskMemory.

        Raises:
            ValueError: If there is no active current task.
        """
        if self.current_task is None:
            raise ValueError("No current task to fail.")

        self.current_task.status = TaskStatus.FAILED
        self.current_task.completed_at = datetime.now(timezone.utc)
        if reason:
            self.current_task.summary = f"FAILED: {reason}"
        self.completed_tasks.append(self.current_task)
        failed_task = self.current_task
        self.current_task = None

        self._enforce_window_size()
        self._touch()
        return failed_task

    def get_task(self, ticket_id: str) -> Optional[TaskMemory]:
        """Look up a task by ticket ID.

        Searches the current task and completed tasks.

        Args:
            ticket_id: The ticket ID to search for.

        Returns:
            The TaskMemory if found, None otherwise.
        """
        if self.current_task and self.current_task.ticket_id == ticket_id:
            return self.current_task

        for task in self.completed_tasks:
            if task.ticket_id == ticket_id:
                return task

        return None

    def get_all_task_ids(self) -> list[str]:
        """Return all task IDs currently in the window (including current)."""
        ids = [t.ticket_id for t in self.completed_tasks]
        if self.current_task:
            ids.append(self.current_task.ticket_id)
        return ids

    def has_active_task(self) -> bool:
        """Check whether there is a current active task."""
        return self.current_task is not None and self.current_task.status == TaskStatus.ACTIVE

    def is_task_archived(self, ticket_id: str) -> bool:
        """Check whether a task has been archived (rotated out of the window)."""
        return ticket_id in self.archived_task_ids

    def total_tasks_in_window(self) -> int:
        """Return the total number of tasks in the window (completed + current)."""
        count = len(self.completed_tasks)
        if self.current_task is not None:
            count += 1
        return count

    def _enforce_window_size(self) -> None:
        """Remove the oldest completed tasks if the window exceeds its size."""
        while len(self.completed_tasks) > self.window_size:
            oldest = self.completed_tasks.pop(0)
            if oldest.ticket_id not in self.archived_task_ids:
                self.archived_task_ids.append(oldest.ticket_id)

    def _touch(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now(timezone.utc)

    def to_json_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_json_dict(cls, data: dict) -> "MemoryWindow":
        """Deserialize from a JSON-compatible dictionary."""
        return cls.model_validate(data)
