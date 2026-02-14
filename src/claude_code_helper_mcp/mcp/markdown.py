"""MarkdownGenerator -- produces human-readable .md summaries of task memory.

Generates structured markdown documents from TaskMemory objects:

- **Task summary**: full details of a single task (active or completed).
- **Current summary**: live progress snapshot for the active task.
- **Window index**: overview of all tasks in the sliding window.

All output is deterministic given the same input (timestamps are formatted,
not regenerated).  The generator is pure -- it has no side effects and does
not perform any I/O.  Callers are responsible for writing the output to disk.

Typical usage::

    from claude_code_helper_mcp.mcp.markdown import MarkdownGenerator

    gen = MarkdownGenerator()
    md = gen.generate_task_summary(task_memory)
    md = gen.generate_current_summary(task_memory)
    md = gen.generate_window_index(window, completed_tasks)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from claude_code_helper_mcp.models.task import TaskMemory, TaskStatus
from claude_code_helper_mcp.models.window import MemoryWindow


class MarkdownGenerator:
    """Produces human-readable markdown summaries from task memory data.

    This is a stateless utility class.  All methods are pure functions that
    accept data models and return markdown strings.  No filesystem access
    or network calls are made.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_task_summary(self, task: TaskMemory) -> str:
        """Generate a full markdown summary for a single task.

        Produces a comprehensive document with all recorded information:
        metadata, steps timeline, files modified, branches used, decisions
        made, and completion summary.

        Parameters
        ----------
        task:
            The task memory to summarize.

        Returns
        -------
        str
            The formatted markdown string.
        """
        sections: list[str] = []

        # Header
        status_badge = self._status_badge(task.status)
        sections.append(f"# Task Summary: {task.ticket_id}\n")
        sections.append(f"**{task.title}**\n")

        # Metadata table
        sections.append("## Metadata\n")
        sections.append("| Field | Value |")
        sections.append("| --- | --- |")
        sections.append(f"| Ticket | {task.ticket_id} |")
        sections.append(f"| Status | {status_badge} |")
        if task.phase:
            sections.append(f"| Phase | {task.phase} |")
        sections.append(f"| Started | {self._format_dt(task.started_at)} |")
        if task.completed_at:
            sections.append(f"| Completed | {self._format_dt(task.completed_at)} |")
            duration = self._format_duration(task.started_at, task.completed_at)
            sections.append(f"| Duration | {duration} |")
        sections.append(f"| Steps | {task.step_count()} |")
        sections.append(f"| Files | {len(task.files)} |")
        sections.append(f"| Branches | {len(task.branches)} |")
        sections.append(f"| Decisions | {len(task.decisions)} |")
        sections.append("")

        # Summary section (if present)
        if task.summary:
            sections.append("## Summary\n")
            sections.append(task.summary)
            sections.append("")

        # Steps timeline
        if task.steps:
            sections.append("## Steps Timeline\n")
            for step in task.steps:
                success_marker = "+" if step.success else "x"
                line = f"{step.step_number}. [{success_marker}] **{step.action}**"
                if step.tool_used:
                    line += f" (tool: `{step.tool_used}`)"
                line += f" -- {self._format_dt(step.timestamp)}"
                sections.append(line)
                if step.description:
                    sections.append(f"   {step.description}")
                if step.result_summary:
                    sections.append(f"   Result: {step.result_summary}")
                if step.files_involved:
                    files_str = ", ".join(f"`{f}`" for f in step.files_involved)
                    sections.append(f"   Files: {files_str}")
            sections.append("")

        # Files modified
        if task.files:
            sections.append("## Files Modified\n")
            sections.append("| File | Action | Description |")
            sections.append("| --- | --- | --- |")
            for f in task.files:
                desc = f.description[:80] + "..." if len(f.description) > 80 else f.description
                sections.append(f"| `{f.path}` | {f.action.value} | {desc} |")
            sections.append("")

            # Detailed file action history (if any file has history)
            files_with_history = [f for f in task.files if f.action_history]
            if files_with_history:
                sections.append("### File Action History\n")
                for f in files_with_history:
                    sections.append(f"**`{f.path}`** ({len(f.action_history) + 1} actions):\n")
                    for i, entry in enumerate(f.action_history, 1):
                        sections.append(
                            f"  {i}. {entry.get('action', 'unknown')}"
                            f" -- {entry.get('timestamp', 'unknown')}"
                        )
                        if entry.get("description"):
                            sections.append(f"     {entry['description']}")
                    # Current (latest) action
                    sections.append(
                        f"  {len(f.action_history) + 1}. {f.action.value}"
                        f" -- {self._format_dt(f.timestamp)} (current)"
                    )
                    if f.description:
                        sections.append(f"     {f.description}")
                    sections.append("")

        # Branches
        if task.branches:
            sections.append("## Branches\n")
            sections.append("| Branch | Action | Base |")
            sections.append("| --- | --- | --- |")
            for b in task.branches:
                base = b.base_branch or "-"
                sections.append(f"| `{b.branch_name}` | {b.action.value} | {base} |")
            sections.append("")

        # Decisions
        if task.decisions:
            sections.append("## Decisions\n")
            for dec in task.decisions:
                sections.append(
                    f"### Decision {dec.decision_number}: {dec.decision}\n"
                )
                sections.append(f"*Recorded: {self._format_dt(dec.timestamp)}*\n")
                if dec.reasoning:
                    sections.append(f"**Reasoning:** {dec.reasoning}\n")
                if dec.alternatives:
                    sections.append("**Alternatives considered:**\n")
                    for alt in dec.alternatives:
                        sections.append(f"- {alt}")
                    sections.append("")
                if dec.context:
                    sections.append(f"**Context:** {dec.context}\n")

        # Next steps (if present)
        if task.next_steps:
            sections.append("## Next Steps\n")
            for ns in task.next_steps:
                sections.append(f"- {ns}")
            sections.append("")

        # Metadata (if non-empty)
        if task.metadata:
            sections.append("## Metadata\n")
            sections.append("```json")
            import json

            sections.append(json.dumps(task.metadata, indent=2))
            sections.append("```")
            sections.append("")

        return "\n".join(sections)

    def generate_current_summary(self, task: TaskMemory) -> str:
        """Generate a live progress summary for the active task.

        Produces a concise document showing ongoing progress: what has been
        done so far, which files have been touched, the active branch, and
        key decisions.  Designed for quick orientation.

        Parameters
        ----------
        task:
            The active task memory.

        Returns
        -------
        str
            The formatted markdown string.
        """
        sections: list[str] = []

        sections.append(f"# Current Task: {task.ticket_id}\n")
        sections.append(f"**{task.title}**\n")

        # Quick stats
        sections.append("## Status\n")
        sections.append(f"- **Status:** {self._status_badge(task.status)}")
        if task.phase:
            sections.append(f"- **Phase:** {task.phase}")
        sections.append(f"- **Started:** {self._format_dt(task.started_at)}")
        elapsed = self._format_elapsed(task.started_at)
        sections.append(f"- **Elapsed:** {elapsed}")
        sections.append(f"- **Steps taken:** {task.step_count()}")
        sections.append(f"- **Files touched:** {len(task.files)}")
        sections.append(f"- **Decisions made:** {len(task.decisions)}")
        active_branch = task.get_active_branch()
        if active_branch:
            sections.append(f"- **Active branch:** `{active_branch}`")
        sections.append("")

        # Recent steps (last 10)
        if task.steps:
            recent = task.steps[-10:]
            sections.append(f"## Recent Steps (last {len(recent)})\n")
            for step in recent:
                success_marker = "+" if step.success else "x"
                line = f"{step.step_number}. [{success_marker}] **{step.action}**"
                if step.tool_used:
                    line += f" (`{step.tool_used}`)"
                sections.append(line)
                if step.result_summary:
                    sections.append(f"   {step.result_summary}")
            sections.append("")

        # Files touched
        if task.files:
            sections.append("## Files Touched\n")
            for f in task.files:
                sections.append(f"- `{f.path}` ({f.action.value})")
            sections.append("")

        # Key decisions (last 5)
        if task.decisions:
            recent_dec = task.decisions[-5:]
            sections.append(f"## Key Decisions (last {len(recent_dec)})\n")
            for dec in recent_dec:
                sections.append(f"- **D{dec.decision_number}:** {dec.decision}")
                if dec.reasoning:
                    sections.append(f"  *{dec.reasoning}*")
            sections.append("")

        # Next steps
        if task.next_steps:
            sections.append("## Next Steps\n")
            for ns in task.next_steps:
                sections.append(f"- {ns}")
            sections.append("")

        return "\n".join(sections)

    def generate_window_index(
        self,
        window: MemoryWindow,
        completed_tasks: Optional[list[TaskMemory]] = None,
    ) -> str:
        """Generate an index overview of all tasks in the sliding window.

        Produces a summary table and per-task quick reference.

        Parameters
        ----------
        window:
            The memory window containing completed task references.
        completed_tasks:
            Optional list of completed TaskMemory objects for richer output.
            If not provided, only basic info from the window is shown.

        Returns
        -------
        str
            The formatted markdown string.
        """
        sections: list[str] = []

        sections.append("# Memory Window Index\n")
        sections.append(f"*Window size: {window.window_size} completed + current*\n")
        sections.append(f"*Updated: {self._format_dt(window.updated_at)}*\n")

        # Current task
        sections.append("## Current Task\n")
        if window.current_task:
            ct = window.current_task
            sections.append(f"- **{ct.ticket_id}**: {ct.title}")
            sections.append(f"  - Status: {self._status_badge(ct.status)}")
            if ct.phase:
                sections.append(f"  - Phase: {ct.phase}")
            sections.append(f"  - Steps: {ct.step_count()}, Files: {len(ct.files)}")
            sections.append(f"  - Started: {self._format_dt(ct.started_at)}")
        else:
            sections.append("*No active task.*")
        sections.append("")

        # Completed tasks table
        tasks_list = completed_tasks if completed_tasks else window.completed_tasks
        if tasks_list:
            sections.append("## Completed Tasks\n")
            sections.append(
                "| Ticket | Title | Status | Steps | Files | Duration |"
            )
            sections.append("| --- | --- | --- | --- | --- | --- |")
            for t in tasks_list:
                duration = "-"
                if t.completed_at and t.started_at:
                    duration = self._format_duration(t.started_at, t.completed_at)
                title_short = (
                    t.title[:40] + "..." if len(t.title) > 40 else t.title
                )
                sections.append(
                    f"| {t.ticket_id} | {title_short} "
                    f"| {self._status_badge(t.status)} "
                    f"| {t.step_count()} | {len(t.files)} | {duration} |"
                )
            sections.append("")

        # Archived tasks
        if window.archived_task_ids:
            sections.append("## Archived Tasks\n")
            sections.append(
                f"*{len(window.archived_task_ids)} task(s) have been "
                f"rotated out of the window:*\n"
            )
            for tid in window.archived_task_ids:
                sections.append(f"- {tid}")
            sections.append("")

        # Window stats
        sections.append("## Statistics\n")
        total_in_window = window.total_tasks_in_window()
        completed_count = len(window.completed_tasks)
        archived_count = len(window.archived_task_ids)
        sections.append(f"- Tasks in window: {total_in_window}")
        sections.append(f"- Completed: {completed_count}")
        sections.append(f"- Archived: {archived_count}")
        sections.append(
            f"- Total tasks tracked: {total_in_window + archived_count}"
        )
        sections.append("")

        return "\n".join(sections)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_dt(dt: datetime) -> str:
        """Format a datetime as a human-readable UTC string."""
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")

    @staticmethod
    def _format_duration(start: datetime, end: datetime) -> str:
        """Format the duration between two datetimes as a human-readable string."""
        delta = end - start
        total_seconds = int(delta.total_seconds())
        if total_seconds < 0:
            return "0s"

        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60

        parts: list[str] = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if seconds > 0 or not parts:
            parts.append(f"{seconds}s")

        return " ".join(parts)

    @staticmethod
    def _format_elapsed(start: datetime) -> str:
        """Format elapsed time from start to now."""
        now = datetime.now(timezone.utc)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        delta = now - start
        total_seconds = int(delta.total_seconds())
        if total_seconds < 0:
            return "just started"

        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60

        if hours > 0:
            return f"{hours}h {minutes}m"
        if minutes > 0:
            return f"{minutes}m"
        return f"{total_seconds}s"

    @staticmethod
    def _status_badge(status: TaskStatus) -> str:
        """Return a text badge for a task status."""
        badges = {
            TaskStatus.ACTIVE: "IN PROGRESS",
            TaskStatus.COMPLETED: "DONE",
            TaskStatus.ARCHIVED: "ARCHIVED",
            TaskStatus.FAILED: "FAILED",
        }
        return badges.get(status, status.value.upper())
