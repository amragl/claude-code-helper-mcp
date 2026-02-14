"""TaskMemory model -- the primary container for all memory about a single task.

A TaskMemory aggregates all steps, files, branches, and decisions recorded
during execution of one ticket. It is the unit of persistence and the unit
of the sliding window.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator

from claude_code_helper_mcp.models.records import (
    BranchRecord,
    DecisionRecord,
    FileAction,
    FileRecord,
    StepRecord,
)


class TaskStatus(str, Enum):
    """Lifecycle status of a task memory."""

    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"
    FAILED = "failed"


class TaskMemory(BaseModel):
    """Complete memory record for a single task/ticket.

    This is the primary data structure: one TaskMemory per ticket. It
    contains all steps taken, files touched, branches used, and decisions
    made during that ticket's execution.

    The memory is created when a task starts and updated throughout execution.
    On completion, it is finalized with a summary and moved into the sliding
    window's completed list.
    """

    ticket_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="The ticket identifier (e.g., 'CMH-002').",
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Human-readable title of the task.",
    )
    phase: Optional[str] = Field(
        default=None,
        max_length=50,
        description="The roadmap phase this task belongs to (e.g., 'phase-1').",
    )
    status: TaskStatus = Field(
        default=TaskStatus.ACTIVE,
        description="Current lifecycle status of this task memory.",
    )
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this task was started (UTC).",
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="When this task was completed (UTC). Null if still active.",
    )
    summary: str = Field(
        default="",
        max_length=5000,
        description="Post-completion summary of what was accomplished.",
    )
    steps: list[StepRecord] = Field(
        default_factory=list,
        description="Ordered list of steps taken during this task.",
    )
    files: list[FileRecord] = Field(
        default_factory=list,
        description="Files touched during this task (deduplicated by path).",
    )
    branches: list[BranchRecord] = Field(
        default_factory=list,
        description="Git branches involved in this task.",
    )
    decisions: list[DecisionRecord] = Field(
        default_factory=list,
        description="Significant decisions made during this task.",
    )
    next_steps: list[str] = Field(
        default_factory=list,
        description="Planned next actions (useful for recovery after /clear).",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata for extensibility.",
    )


    @model_validator(mode="after")
    def validate_completion_state(self) -> "TaskMemory":
        """Ensure completed_at is set when status is completed."""
        if self.status == TaskStatus.COMPLETED and self.completed_at is None:
            self.completed_at = datetime.now(timezone.utc)
        return self

    def add_step(
        self,
        action: str,
        description: str = "",
        tool_used: Optional[str] = None,
        result_summary: Optional[str] = None,
        files_involved: Optional[list[str]] = None,
        success: bool = True,
    ) -> StepRecord:
        """Record a new step in this task.

        Auto-assigns the next sequential step number and current timestamp.

        Returns:
            The created StepRecord.
        """
        step = StepRecord(
            step_number=len(self.steps) + 1,
            action=action,
            description=description,
            tool_used=tool_used,
            result_summary=result_summary,
            files_involved=files_involved or [],
            success=success,
        )
        self.steps.append(step)
        return step

    def record_file(
        self,
        path: str,
        action: FileAction,
        description: str = "",
    ) -> FileRecord:
        """Record a file action, deduplicating by path.

        If the file has already been recorded, appends to its action history
        instead of creating a new record.

        Returns:
            The FileRecord (existing or newly created).
        """
        for existing in self.files:
            if existing.path == path:
                existing.add_action(action, description)
                return existing

        record = FileRecord(
            path=path,
            action=action,
            description=description,
        )
        self.files.append(record)
        return record

    def record_branch(
        self,
        branch_name: str,
        action: "BranchAction",
        base_branch: Optional[str] = None,
    ) -> BranchRecord:
        """Record a branch action, deduplicating by branch name.

        If the branch has already been recorded, appends to its action history
        instead of creating a new record.

        Returns:
            The BranchRecord (existing or newly created).
        """
        from claude_code_helper_mcp.models.records import BranchAction as BA

        for existing in self.branches:
            if existing.branch_name == branch_name:
                existing.add_action(action, base_branch)
                return existing

        record = BranchRecord(
            branch_name=branch_name,
            action=action,
            base_branch=base_branch,
        )
        self.branches.append(record)
        return record

    def add_decision(
        self,
        decision: str,
        reasoning: str = "",
        alternatives: Optional[list[str]] = None,
        context: str = "",
    ) -> DecisionRecord:
        """Record a new decision in this task.

        Auto-assigns the next sequential decision number and current timestamp.

        Returns:
            The created DecisionRecord.
        """
        record = DecisionRecord(
            decision_number=len(self.decisions) + 1,
            decision=decision,
            reasoning=reasoning,
            alternatives=alternatives or [],
            context=context,
        )
        self.decisions.append(record)
        return record

    def complete(self, summary: str = "") -> None:
        """Mark this task as completed.

        Sets the status to COMPLETED, records the completion timestamp,
        and optionally sets a completion summary.
        """
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        if summary:
            self.summary = summary

    def get_file_paths(self) -> list[str]:
        """Return a list of all file paths touched in this task."""
        return [f.path for f in self.files]

    def get_active_branch(self) -> Optional[str]:
        """Return the most recently acted-upon branch name, or None."""
        if not self.branches:
            return None
        return self.branches[-1].branch_name

    def step_count(self) -> int:
        """Return the total number of steps recorded."""
        return len(self.steps)

    def to_json_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary.

        Uses Pydantic's model_dump with mode='json' for proper datetime
        serialization and enum value conversion.
        """
        return self.model_dump(mode="json")

    @classmethod
    def from_json_dict(cls, data: dict) -> "TaskMemory":
        """Deserialize from a JSON-compatible dictionary.

        Handles loading from files where datetimes are ISO strings and
        enums are stored as their string values.
        """
        return cls.model_validate(data)
