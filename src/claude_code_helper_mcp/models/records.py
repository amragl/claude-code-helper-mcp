"""Individual record models for steps, files, branches, and decisions.

These models represent the atomic units of memory that are collected during
task execution and aggregated into a TaskMemory.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class FileAction(str, Enum):
    """Actions that can be performed on a file."""

    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    RENAMED = "renamed"
    READ = "read"


class BranchAction(str, Enum):
    """Actions that can be performed on a git branch."""

    CREATED = "created"
    CHECKED_OUT = "checked_out"
    MERGED = "merged"
    DELETED = "deleted"
    PUSHED = "pushed"
    PULLED = "pulled"


class StepRecord(BaseModel):
    """A single recorded step during task execution.

    Steps are the most granular unit of memory: each tool call, edit, or
    significant action is recorded as a step with its outcome.
    """

    step_number: int = Field(
        ...,
        ge=1,
        description="Sequential step number within the task, starting at 1.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this step was recorded (UTC).",
    )
    action: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Short description of the action taken (e.g., 'Created file', 'Ran tests').",
    )
    description: str = Field(
        default="",
        max_length=2000,
        description="Detailed description of what happened during this step.",
    )
    tool_used: Optional[str] = Field(
        default=None,
        max_length=100,
        description="The tool or command that was used (e.g., 'Write', 'Bash', 'Edit').",
    )
    result_summary: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Summary of the result (e.g., 'File created successfully', '3 tests passed').",
    )
    files_involved: list[str] = Field(
        default_factory=list,
        description="Paths of files involved in this step.",
    )
    success: bool = Field(
        default=True,
        description="Whether this step completed successfully.",
    )



class FileRecord(BaseModel):
    """A record of a file that was touched during task execution.

    Files are deduplicated by path: if the same file is modified multiple
    times, its action history is appended rather than creating duplicate records.
    """

    path: str = Field(
        ...,
        min_length=1,
        description="Relative file path from the project root.",
    )
    action: FileAction = Field(
        ...,
        description="The most recent action performed on this file.",
    )
    description: str = Field(
        default="",
        max_length=1000,
        description="Description of what was done to this file.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this file action was recorded (UTC).",
    )
    action_history: list[dict] = Field(
        default_factory=list,
        description="History of all actions on this file: [{action, description, timestamp}].",
    )

    def add_action(self, action: FileAction, description: str = "") -> None:
        """Record an additional action on this file.

        Updates the top-level action and appends to the action history.
        """
        now = datetime.now(timezone.utc)
        self.action_history.append(
            {
                "action": self.action.value,
                "description": self.description,
                "timestamp": self.timestamp.isoformat(),
            }
        )
        self.action = action
        self.description = description
        self.timestamp = now


class BranchRecord(BaseModel):
    """A record of a git branch involved in task execution.

    Tracks the lifecycle of branches: creation, checkout, push, merge, deletion.
    """

    branch_name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="The full branch name (e.g., 'feature/CMH-002-data-models').",
    )
    action: BranchAction = Field(
        ...,
        description="The most recent action performed on this branch.",
    )
    base_branch: Optional[str] = Field(
        default=None,
        max_length=200,
        description="The branch this was created from or merged into.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this branch action was recorded (UTC).",
    )
    action_history: list[dict] = Field(
        default_factory=list,
        description="History of all actions on this branch: [{action, base_branch, timestamp}].",
    )

    def add_action(
        self,
        action: BranchAction,
        base_branch: Optional[str] = None,
    ) -> None:
        """Record an additional action on this branch.

        Updates the top-level action and appends to the action history.
        """
        now = datetime.now(timezone.utc)
        self.action_history.append(
            {
                "action": self.action.value,
                "base_branch": self.base_branch,
                "timestamp": self.timestamp.isoformat(),
            }
        )
        self.action = action
        if base_branch is not None:
            self.base_branch = base_branch
        self.timestamp = now


class DecisionRecord(BaseModel):
    """A record of a significant decision made during task execution.

    Decisions capture the reasoning behind choices: why a particular approach
    was taken, what alternatives were considered, and under what context.
    """

    decision_number: int = Field(
        ...,
        ge=1,
        description="Sequential decision number within the task, starting at 1.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this decision was recorded (UTC).",
    )
    decision: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The decision that was made.",
    )
    reasoning: str = Field(
        default="",
        max_length=2000,
        description="Why this decision was made.",
    )
    alternatives: list[str] = Field(
        default_factory=list,
        description="Alternative approaches that were considered but not chosen.",
    )
    context: str = Field(
        default="",
        max_length=1000,
        description="Relevant context that informed the decision.",
    )

