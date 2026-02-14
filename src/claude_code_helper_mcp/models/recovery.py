"""RecoveryContext model -- the context bundle returned after /clear.

When a Claude Code session is cleared, all context is lost. The RecoveryContext
provides everything needed to resume work: what ticket was active, what files
were being modified, what branch was checked out, what decisions were made,
and what the next planned steps were.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class RecoveryContext(BaseModel):
    """Full task context for recovery after /clear.

    This model is returned by the get_recovery_context MCP tool. It contains
    a curated snapshot of the most important information needed to resume
    work on a task without re-reading the entire codebase.
    """

    ticket_id: str = Field(
        ...,
        min_length=1,
        description="The ticket being worked on when context was lost.",
    )
    title: str = Field(
        ...,
        min_length=1,
        description="Human-readable title of the active task.",
    )
    phase: Optional[str] = Field(
        default=None,
        description="The roadmap phase the task belongs to.",
    )
    status: str = Field(
        ...,
        description="Task status at time of recovery (e.g., 'active').",
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this recovery context was generated (UTC).",
    )
    last_step: Optional[dict] = Field(
        default=None,
        description="The most recent step record (serialized StepRecord).",
    )
    recent_steps: list[dict] = Field(
        default_factory=list,
        description="The last N steps taken (most recent first), serialized.",
    )
    files_modified: list[str] = Field(
        default_factory=list,
        description="Paths of all files modified during this task.",
    )
    active_branch: Optional[str] = Field(
        default=None,
        description="The git branch that was being used.",
    )
    key_decisions: list[dict] = Field(
        default_factory=list,
        description="All decisions made during this task, serialized.",
    )
    next_steps: list[str] = Field(
        default_factory=list,
        description="Planned next actions that were recorded before /clear.",
    )
    summary_so_far: str = Field(
        default="",
        description="A brief summary of progress so far on this task.",
    )
    total_steps_completed: int = Field(
        default=0,
        ge=0,
        description="Total number of steps completed before /clear.",
    )
    task_started_at: Optional[datetime] = Field(
        default=None,
        description="When the task was originally started (UTC).",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata from the task.",
    )


    @classmethod
    def from_task_memory(
        cls,
        task: "TaskMemory",
        recent_step_count: int = 10,
    ) -> "RecoveryContext":
        """Build a RecoveryContext from a TaskMemory.

        Extracts the most relevant information from the full task memory
        into a focused recovery snapshot.

        Args:
            task: The TaskMemory to create recovery context from.
            recent_step_count: How many recent steps to include (default 10).

        Returns:
            A RecoveryContext with the task's key information.
        """
        from claude_code_helper_mcp.models.task import TaskMemory

        recent_steps_raw = task.steps[-recent_step_count:] if task.steps else []
        recent_steps = [s.model_dump(mode="json") for s in reversed(recent_steps_raw)]

        last_step = None
        if task.steps:
            last_step = task.steps[-1].model_dump(mode="json")

        key_decisions = [d.model_dump(mode="json") for d in task.decisions]

        return cls(
            ticket_id=task.ticket_id,
            title=task.title,
            phase=task.phase,
            status=task.status.value,
            last_step=last_step,
            recent_steps=recent_steps,
            files_modified=task.get_file_paths(),
            active_branch=task.get_active_branch(),
            key_decisions=key_decisions,
            next_steps=list(task.next_steps),
            summary_so_far=task.summary,
            total_steps_completed=task.step_count(),
            task_started_at=task.started_at,
            metadata=dict(task.metadata),
        )

    def to_json_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        return self.model_dump(mode="json")

    @classmethod
    def from_json_dict(cls, data: dict) -> "RecoveryContext":
        """Deserialize from a JSON-compatible dictionary."""
        return cls.model_validate(data)

    def format_for_prompt(self) -> str:
        """Format this recovery context as a human-readable prompt.

        Produces a structured text block suitable for injecting into a
        Claude Code session to restore task awareness.
        """
        lines = [
            f"# Recovery Context for {self.ticket_id}: {self.title}",
            "",
            f"**Phase:** {self.phase or 'N/A'}",
            f"**Status:** {self.status}",
            f"**Steps completed:** {self.total_steps_completed}",
            f"**Started:** {self.task_started_at.isoformat() if self.task_started_at else 'N/A'}",
            "",
        ]

        if self.active_branch:
            lines.append(f"**Active branch:** `{self.active_branch}`")
            lines.append("")

        if self.summary_so_far:
            lines.append("## Progress So Far")
            lines.append(self.summary_so_far)
            lines.append("")

        if self.files_modified:
            lines.append("## Files Modified")
            for f in self.files_modified:
                lines.append(f"- `{f}`")
            lines.append("")

        if self.key_decisions:
            lines.append("## Key Decisions")
            for d in self.key_decisions:
                lines.append(f"- **{d.get('decision', 'N/A')}**: {d.get('reasoning', '')}")
            lines.append("")

        if self.last_step:
            lines.append("## Last Step")
            lines.append(
                f"Step {self.last_step.get('step_number', '?')}: "
                f"{self.last_step.get('action', 'N/A')}"
            )
            if self.last_step.get("result_summary"):
                lines.append(f"Result: {self.last_step['result_summary']}")
            lines.append("")

        if self.next_steps:
            lines.append("## Planned Next Steps")
            for i, step in enumerate(self.next_steps, 1):
                lines.append(f"{i}. {step}")
            lines.append("")

        return "\n".join(lines)
