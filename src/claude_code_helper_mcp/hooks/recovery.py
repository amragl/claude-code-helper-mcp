"""Post-/clear recovery workflow for Claude Code Helper.

Provides the :class:`RecoveryWorkflow` class which orchestrates the complete
recovery process after a ``/clear`` command wipes Claude Code session context.
The workflow integrates three systems:

1. **Clear event detection** -- Detects that a ``/clear`` has occurred by
   comparing the current session state against persisted memory.  A clear is
   inferred when the memory system has an active task but no session-level
   context exists (i.e., the agent has no knowledge of the current task).

2. **Recovery prompt generation** -- Produces a structured, human-readable
   recovery prompt that can be injected into the new session.  The prompt
   includes ticket ID, phase, branch, files modified, decisions made, recent
   steps, and planned next steps.

3. **Pipeline resumption integration** -- Reads Agent Forge pipeline state
   to enrich the recovery context with pipeline-specific information: the
   current pipeline step, build status, PR number, and which steps have
   already been completed.

Usage::

    from claude_code_helper_mcp.hooks.recovery import RecoveryWorkflow

    workflow = RecoveryWorkflow(
        project_root="/path/to/project",
        storage_path="/path/to/.claude-memory",
    )

    # Detect a clear event
    is_clear = workflow.detect_clear_event()

    # Generate the full recovery context
    result = workflow.recover()

    # Get just the prompt text
    prompt = workflow.generate_recovery_prompt()

CLI integration is provided via the ``memory recover`` command (registered
in :mod:`claude_code_helper_mcp.cli.main`).
"""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from claude_code_helper_mcp.config import MemoryConfig
from claude_code_helper_mcp.hooks.state_reader import StateReader, TicketContext
from claude_code_helper_mcp.models.recovery import RecoveryContext
from claude_code_helper_mcp.models.task import TaskMemory, TaskStatus
from claude_code_helper_mcp.storage.window_manager import WindowManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Marker file written after a successful recovery to avoid duplicate prompts.
RECOVERY_MARKER_FILE = "last-recovery.json"

# Maximum age (in seconds) for considering a recovery marker as "recent".
# If the last recovery happened within this window, detect_clear_event
# returns False to avoid spamming recovery prompts.
RECOVERY_COOLDOWN_SECONDS = 60


# ---------------------------------------------------------------------------
# RecoveryWorkflow
# ---------------------------------------------------------------------------


class RecoveryWorkflow:
    """Orchestrates the complete post-/clear recovery workflow.

    This class ties together clear event detection, recovery context
    generation from the memory system, pipeline state enrichment from
    Agent Forge, and recovery prompt formatting.

    Parameters
    ----------
    project_root:
        The root directory of the project.  Used to locate both the
        ``.claude-memory/`` storage and ``.agent-forge/`` state directories.
        When *None*, auto-detection is used via :class:`MemoryConfig`.
    storage_path:
        Explicit path to the ``.claude-memory/`` storage directory.
        When *None*, derived from *project_root*.
    window_size:
        Override for the window size.  When *None*, uses the persisted
        or default value.
    """

    def __init__(
        self,
        project_root: Optional[str] = None,
        storage_path: Optional[str] = None,
        window_size: Optional[int] = None,
    ) -> None:
        # Load configuration.
        if storage_path:
            self._config = MemoryConfig(storage_path=storage_path)
        elif project_root:
            self._config = MemoryConfig.load(project_root=project_root)
        else:
            self._config = MemoryConfig.load()

        self._project_root = Path(
            project_root or self._config.project_root
        ).resolve()

        # Initialise the window manager for memory access.
        self._window_manager = WindowManager(
            storage_path=self._config.storage_path,
            window_size=window_size,
        )

        # Initialise the state reader for pipeline context.
        self._state_reader = StateReader(
            project_root=str(self._project_root),
            cache_ttl_seconds=0,  # No caching -- always fresh reads.
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def project_root(self) -> Path:
        """The resolved project root path."""
        return self._project_root

    @property
    def config(self) -> MemoryConfig:
        """The loaded memory configuration."""
        return self._config

    @property
    def window_manager(self) -> WindowManager:
        """The window manager used for memory access."""
        return self._window_manager

    @property
    def state_reader(self) -> StateReader:
        """The state reader used for pipeline context."""
        return self._state_reader

    # ------------------------------------------------------------------
    # Clear event detection
    # ------------------------------------------------------------------

    def detect_clear_event(self) -> bool:
        """Detect whether a ``/clear`` has likely occurred.

        A clear event is inferred when ALL of the following are true:

        1. The memory system has an active task (a task with status "active").
        2. The pipeline state shows a running or blocked pipeline (work was
           in progress when /clear happened).
        3. No recent recovery marker exists (to avoid duplicate detection
           within the cooldown window).

        Returns
        -------
        bool
            True if a clear event is detected, False otherwise.
        """
        # Check 1: Is there an active task in memory?
        current_task = self._window_manager.get_current_task()
        if current_task is None:
            logger.debug(
                "RecoveryWorkflow.detect_clear_event: No active task. "
                "Not a clear event."
            )
            return False

        # Check 2: Is the pipeline in a state suggesting active work?
        pipeline = self._state_reader.read_pipeline()
        if pipeline is not None:
            if pipeline.status not in ("running", "blocked", "idle"):
                logger.debug(
                    "RecoveryWorkflow.detect_clear_event: Pipeline status "
                    "is '%s', not indicative of active work.",
                    pipeline.status,
                )
                # Still might be a clear -- task is active but pipeline idle
                # after completion.  We allow detection in this case too.

        # Check 3: Has a recent recovery already been performed?
        if self._has_recent_recovery():
            logger.debug(
                "RecoveryWorkflow.detect_clear_event: Recent recovery "
                "marker found. Suppressing duplicate detection."
            )
            return False

        logger.info(
            "RecoveryWorkflow.detect_clear_event: Clear event detected. "
            "Active task: %s.",
            current_task.ticket_id,
        )
        return True

    # ------------------------------------------------------------------
    # Recovery execution
    # ------------------------------------------------------------------

    def recover(
        self,
        ticket_id: Optional[str] = None,
        recent_step_count: int = 10,
        include_pipeline_context: bool = True,
    ) -> dict[str, Any]:
        """Execute the full recovery workflow and return the result.

        This is the primary entry point.  It:

        1. Resolves the target task (explicit ticket ID, active task,
           or most recently completed task).
        2. Builds a :class:`RecoveryContext` from the task memory.
        3. Optionally enriches it with pipeline state from Agent Forge.
        4. Generates a formatted recovery prompt.
        5. Detects the current git branch (live from the filesystem).
        6. Writes a recovery marker to prevent duplicate detections.
        7. Returns the full recovery result dictionary.

        Parameters
        ----------
        ticket_id:
            Explicit ticket ID to recover context for.  When *None*,
            auto-detects (active task first, then most recent completed).
        recent_step_count:
            Number of recent steps to include.  Default: 10.
        include_pipeline_context:
            Whether to enrich the recovery with Agent Forge pipeline
            state.  Default: True.

        Returns
        -------
        dict
            A dictionary containing:
            - ``status``: ``"recovered"`` or ``"no_context"`` or ``"error"``
            - ``recovery_context``: The serialised RecoveryContext (if available)
            - ``pipeline_context``: Pipeline enrichment data (if available)
            - ``recovery_prompt``: Formatted text prompt for session injection
            - ``git_branch``: The current git branch (if detectable)
            - ``timestamp``: When the recovery was performed
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # Step 1: Resolve the target task.
        task, source = self._resolve_task(ticket_id)
        if task is None:
            return {
                "status": "no_context",
                "message": (
                    "No task found for recovery. "
                    + (
                        f"Ticket '{ticket_id}' not found in memory."
                        if ticket_id
                        else "No active or recently completed tasks in the memory window."
                    )
                ),
                "available_tasks": self._window_manager.get_all_known_task_ids(),
                "timestamp": timestamp,
            }

        # Step 2: Build the RecoveryContext from task memory.
        recent_step_count = max(1, min(50, recent_step_count))
        recovery_ctx = RecoveryContext.from_task_memory(
            task, recent_step_count=recent_step_count
        )

        # Step 3: Detect the current git branch.
        git_branch = self._detect_git_branch()
        if git_branch and not recovery_ctx.active_branch:
            # Update the recovery context with the live branch.
            recovery_ctx = recovery_ctx.model_copy(
                update={"active_branch": git_branch}
            )

        # Step 4: Enrich with pipeline context.
        pipeline_context = None
        if include_pipeline_context:
            pipeline_context = self._get_pipeline_enrichment(task.ticket_id)

        # Step 5: Generate the recovery prompt.
        recovery_prompt = self._generate_prompt(
            recovery_ctx, pipeline_context, source
        )

        # Step 6: Write the recovery marker.
        self._write_recovery_marker(task.ticket_id, source)

        # Step 7: Assemble and return the result.
        result: dict[str, Any] = {
            "status": "recovered",
            "ticket_id": task.ticket_id,
            "title": task.title,
            "source": source,
            "recovery_context": recovery_ctx.to_json_dict(),
            "recovery_prompt": recovery_prompt,
            "git_branch": git_branch,
            "timestamp": timestamp,
        }

        if pipeline_context is not None:
            result["pipeline_context"] = pipeline_context

        return result

    def generate_recovery_prompt(
        self,
        ticket_id: Optional[str] = None,
        recent_step_count: int = 10,
    ) -> str:
        """Generate just the recovery prompt text without the full workflow.

        This is a convenience method for callers that only need the prompt
        text (e.g., for injection into a session).

        Parameters
        ----------
        ticket_id:
            Explicit ticket ID, or None for auto-detection.
        recent_step_count:
            Number of recent steps to include.

        Returns
        -------
        str
            The formatted recovery prompt, or an error message if no
            task is available for recovery.
        """
        result = self.recover(
            ticket_id=ticket_id,
            recent_step_count=recent_step_count,
        )

        if result["status"] == "recovered":
            return result["recovery_prompt"]
        else:
            return f"[Recovery unavailable: {result.get('message', 'Unknown error')}]"

    # ------------------------------------------------------------------
    # Pipeline enrichment
    # ------------------------------------------------------------------

    def _get_pipeline_enrichment(
        self, ticket_id: str
    ) -> Optional[dict[str, Any]]:
        """Read Agent Forge pipeline state and extract relevant context.

        Parameters
        ----------
        ticket_id:
            The ticket ID to match against the pipeline state.

        Returns
        -------
        dict or None
            Pipeline enrichment data, or None if pipeline state is
            unavailable or does not match the given ticket.
        """
        ticket_ctx = self._state_reader.get_current_ticket_context()

        if ticket_ctx is None:
            logger.debug(
                "RecoveryWorkflow._get_pipeline_enrichment: "
                "No ticket context available from state reader."
            )
            return None

        # Build enrichment from the ticket context.
        enrichment: dict[str, Any] = {
            "pipeline_status": ticket_ctx.pipeline_status,
            "pipeline_step": ticket_ctx.pipeline_step,
            "pipeline_agent": ticket_ctx.pipeline_agent,
            "last_completed_step": ticket_ctx.last_completed_step,
        }

        # If the ticket context matches our recovery ticket, include
        # build-specific details.
        if ticket_ctx.ticket_id == ticket_id:
            enrichment["matched_ticket"] = True
            enrichment["branch"] = ticket_ctx.branch
            enrichment["pr_number"] = ticket_ctx.pr_number
            enrichment["build_status"] = ticket_ctx.build_status
            enrichment["files_changed"] = ticket_ctx.files_changed
            enrichment["phase"] = ticket_ctx.phase
            enrichment["phase_name"] = ticket_ctx.phase_name
        else:
            enrichment["matched_ticket"] = False
            enrichment["pipeline_ticket"] = ticket_ctx.ticket_id

        # Read step tracking for more detail.
        pipeline = self._state_reader.read_pipeline()
        if pipeline is not None and pipeline.raw:
            step_tracking = pipeline.raw.get("step_tracking", {})
            enrichment["steps_completed"] = step_tracking.get(
                "steps_completed", []
            )
            enrichment["steps_remaining"] = step_tracking.get(
                "steps_remaining", []
            )

            # Include blocked/failed info if relevant.
            if pipeline.blocked_reason:
                enrichment["blocked_reason"] = pipeline.blocked_reason
            if pipeline.failed_step:
                enrichment["failed_step"] = pipeline.failed_step
            if pipeline.failure_reason:
                enrichment["failure_reason"] = pipeline.failure_reason

        return enrichment

    # ------------------------------------------------------------------
    # Prompt generation
    # ------------------------------------------------------------------

    def _generate_prompt(
        self,
        recovery_ctx: RecoveryContext,
        pipeline_context: Optional[dict[str, Any]],
        source: str,
    ) -> str:
        """Generate a formatted recovery prompt for session injection.

        Combines the RecoveryContext's built-in formatting with pipeline-
        specific sections to produce a comprehensive recovery prompt.

        Parameters
        ----------
        recovery_ctx:
            The recovery context from the memory system.
        pipeline_context:
            Optional pipeline enrichment data.
        source:
            How the task was resolved (``"active"``, ``"explicit"``,
            ``"most_recent_completed"``).

        Returns
        -------
        str
            The complete recovery prompt text.
        """
        lines: list[str] = []

        # Header.
        lines.append(
            f"# Recovery Context for {recovery_ctx.ticket_id}: "
            f"{recovery_ctx.title}"
        )
        lines.append("")
        lines.append(
            f"> This context was recovered from persistent memory after "
            f"a /clear event."
        )
        lines.append(
            f"> Source: {source} task | Generated: "
            f"{recovery_ctx.generated_at.isoformat()}"
        )
        lines.append("")

        # Overview section.
        lines.append("## Task Overview")
        lines.append(f"- **Ticket:** {recovery_ctx.ticket_id}")
        lines.append(f"- **Title:** {recovery_ctx.title}")
        lines.append(f"- **Phase:** {recovery_ctx.phase or 'N/A'}")
        lines.append(f"- **Status:** {recovery_ctx.status}")
        lines.append(
            f"- **Steps completed:** {recovery_ctx.total_steps_completed}"
        )
        if recovery_ctx.task_started_at:
            lines.append(
                f"- **Started:** {recovery_ctx.task_started_at.isoformat()}"
            )
        lines.append("")

        # Git branch.
        if recovery_ctx.active_branch:
            lines.append(f"## Active Branch")
            lines.append(f"`{recovery_ctx.active_branch}`")
            lines.append("")

        # Pipeline context (if available).
        if pipeline_context is not None:
            lines.append("## Pipeline State")
            ps = pipeline_context.get("pipeline_status", "unknown")
            lines.append(f"- **Pipeline status:** {ps}")

            step = pipeline_context.get("pipeline_step")
            if step:
                lines.append(f"- **Current step:** {step}")

            last_step = pipeline_context.get("last_completed_step")
            if last_step:
                lines.append(f"- **Last completed step:** {last_step}")

            completed = pipeline_context.get("steps_completed", [])
            if completed:
                lines.append(
                    f"- **Steps done:** {', '.join(completed)}"
                )

            remaining = pipeline_context.get("steps_remaining", [])
            if remaining:
                lines.append(
                    f"- **Steps remaining:** {', '.join(remaining)}"
                )

            pr = pipeline_context.get("pr_number")
            if pr:
                lines.append(f"- **PR:** #{pr}")

            branch = pipeline_context.get("branch")
            if branch:
                lines.append(f"- **Build branch:** {branch}")

            blocked = pipeline_context.get("blocked_reason")
            if blocked:
                lines.append(f"- **BLOCKED:** {blocked}")

            failed_step = pipeline_context.get("failed_step")
            if failed_step:
                lines.append(f"- **Failed step:** {failed_step}")

            lines.append("")

        # Progress summary.
        if recovery_ctx.summary_so_far:
            lines.append("## Progress So Far")
            lines.append(recovery_ctx.summary_so_far)
            lines.append("")

        # Files modified.
        if recovery_ctx.files_modified:
            lines.append("## Files Modified")
            for f in recovery_ctx.files_modified:
                lines.append(f"- `{f}`")
            lines.append("")

        # Key decisions.
        if recovery_ctx.key_decisions:
            lines.append("## Key Decisions")
            for d in recovery_ctx.key_decisions:
                decision_text = d.get("decision", "N/A")
                reasoning = d.get("reasoning", "")
                lines.append(f"- **{decision_text}**")
                if reasoning:
                    lines.append(f"  Reasoning: {reasoning}")
            lines.append("")

        # Recent steps.
        if recovery_ctx.recent_steps:
            lines.append("## Recent Steps")
            for s in recovery_ctx.recent_steps:
                num = s.get("step_number", "?")
                action = s.get("action", "N/A")
                result = s.get("result_summary", "")
                success = s.get("success", True)
                marker = "+" if success else "x"
                lines.append(f"- [{marker}] Step {num}: {action}")
                if result:
                    lines.append(f"  Result: {result}")
            lines.append("")

        # Next steps.
        if recovery_ctx.next_steps:
            lines.append("## Planned Next Steps")
            for i, step in enumerate(recovery_ctx.next_steps, 1):
                lines.append(f"{i}. {step}")
            lines.append("")

        # Resumption instructions.
        lines.append("## How to Resume")
        lines.append(
            "This recovery context has been loaded from persistent memory. "
            "You can continue working on this task from where you left off."
        )
        if recovery_ctx.active_branch:
            lines.append(
                f"1. Verify you are on branch: `{recovery_ctx.active_branch}`"
            )
        lines.append(
            f"{'2' if recovery_ctx.active_branch else '1'}. "
            f"Review the files modified and recent steps above."
        )
        lines.append(
            f"{'3' if recovery_ctx.active_branch else '2'}. "
            f"Continue from the planned next steps."
        )
        lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Task resolution
    # ------------------------------------------------------------------

    def _resolve_task(
        self, ticket_id: Optional[str]
    ) -> tuple[Optional[TaskMemory], str]:
        """Resolve the target task for recovery.

        Search order:
        1. If ticket_id is given, look up that specific task.
        2. If no ticket_id, prefer the active task.
        3. Fall back to the most recently completed task.

        Parameters
        ----------
        ticket_id:
            Explicit ticket ID, or None for auto-detection.

        Returns
        -------
        tuple[TaskMemory | None, str]
            The resolved task and the source label (``"explicit"``,
            ``"active"``, ``"most_recent_completed"``, or ``""``).
        """
        wm = self._window_manager

        if ticket_id:
            task = wm.get_task(ticket_id)
            if task is not None:
                return task, "explicit"
            logger.warning(
                "RecoveryWorkflow._resolve_task: Ticket '%s' not found.",
                ticket_id,
            )
            return None, ""

        # Auto-detect: active task first.
        current = wm.get_current_task()
        if current is not None:
            return current, "active"

        # Fall back to most recently completed.
        if wm.window.completed_tasks:
            return wm.window.completed_tasks[-1], "most_recent_completed"

        return None, ""

    # ------------------------------------------------------------------
    # Git branch detection
    # ------------------------------------------------------------------

    def _detect_git_branch(self) -> Optional[str]:
        """Detect the current git branch from the filesystem.

        Runs ``git rev-parse --abbrev-ref HEAD`` in the project root
        directory to determine the active branch.

        Returns
        -------
        str or None
            The current branch name, or None if git is unavailable
            or the project is not a git repository.
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(self._project_root),
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                branch = result.stdout.strip()
                if branch:
                    return branch
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            logger.debug(
                "RecoveryWorkflow._detect_git_branch: "
                "Could not detect git branch.",
                exc_info=True,
            )
        return None

    # ------------------------------------------------------------------
    # Recovery marker management
    # ------------------------------------------------------------------

    def _has_recent_recovery(self) -> bool:
        """Check if a recent recovery marker exists within the cooldown.

        Returns
        -------
        bool
            True if a recovery was performed within RECOVERY_COOLDOWN_SECONDS.
        """
        marker_path = Path(self._config.storage_path) / RECOVERY_MARKER_FILE
        if not marker_path.is_file():
            return False

        try:
            with open(marker_path, "r", encoding="utf-8") as fp:
                marker = json.load(fp)
            recovered_at = marker.get("recovered_at", "")
            if recovered_at:
                recovered_dt = datetime.fromisoformat(recovered_at)
                age = (
                    datetime.now(timezone.utc) - recovered_dt
                ).total_seconds()
                return age < RECOVERY_COOLDOWN_SECONDS
        except (json.JSONDecodeError, ValueError, OSError, KeyError):
            logger.debug(
                "RecoveryWorkflow._has_recent_recovery: "
                "Could not read recovery marker.",
                exc_info=True,
            )

        return False

    def _write_recovery_marker(
        self, ticket_id: str, source: str
    ) -> None:
        """Write a recovery marker to prevent duplicate detections.

        Parameters
        ----------
        ticket_id:
            The ticket that was recovered.
        source:
            How the task was resolved.
        """
        marker_path = Path(self._config.storage_path) / RECOVERY_MARKER_FILE
        marker = {
            "recovered_at": datetime.now(timezone.utc).isoformat(),
            "ticket_id": ticket_id,
            "source": source,
        }

        try:
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            with open(marker_path, "w", encoding="utf-8") as fp:
                json.dump(marker, fp, indent=2)
        except OSError:
            logger.warning(
                "RecoveryWorkflow._write_recovery_marker: "
                "Could not write recovery marker to %s.",
                marker_path,
                exc_info=True,
            )

    def clear_recovery_marker(self) -> bool:
        """Remove the recovery marker file.

        Returns
        -------
        bool
            True if the marker was removed, False if it did not exist
            or could not be removed.
        """
        marker_path = Path(self._config.storage_path) / RECOVERY_MARKER_FILE
        try:
            if marker_path.is_file():
                marker_path.unlink()
                return True
        except OSError:
            logger.warning(
                "RecoveryWorkflow.clear_recovery_marker: "
                "Could not remove marker at %s.",
                marker_path,
                exc_info=True,
            )
        return False
