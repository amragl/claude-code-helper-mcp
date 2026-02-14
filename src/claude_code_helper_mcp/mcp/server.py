"""FastMCP server bootstrap for Claude Code Helper.

Sets up the FastMCP server instance with stdio transport, integrates the
MemoryStore and WindowManager, and registers a health check tool.

The server is the central hub for all MCP tools. Recording tools (record_step,
record_decision, etc.) will be registered in subsequent tickets (CMH-007
through CMH-010).

Typical usage as an MCP server entry point::

    # Via the registered entry point (pyproject.toml):
    # [project.entry-points."mcp.servers"]
    # claude-code-helper = "claude_code_helper_mcp.mcp:create_server"

    # Or programmatically:
    from claude_code_helper_mcp.mcp.server import create_server
    server = create_server()
    server.run(transport="stdio")

The server provides these foundational capabilities:
- MemoryStore and WindowManager lifecycle management
- Health check tool for verifying server status
- Structured error handling and logging
- Configuration-driven storage path and window size
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastmcp import FastMCP

from claude_code_helper_mcp import __version__
from claude_code_helper_mcp.config import MemoryConfig
from claude_code_helper_mcp.detection.alignment import AlignmentChecker
from claude_code_helper_mcp.mcp.markdown import MarkdownGenerator
from claude_code_helper_mcp.models.records import BranchAction, FileAction
from claude_code_helper_mcp.models.recovery import RecoveryContext
from claude_code_helper_mcp.storage.store import MemoryStore
from claude_code_helper_mcp.storage.window_manager import WindowManager

logger = logging.getLogger(__name__)

# Module-level reference to the server singleton.  Created on first call to
# create_server() or get_server().  This ensures all tools share the same
# MemoryStore and WindowManager instances.
_server_instance: Optional[FastMCP] = None
_window_manager: Optional[WindowManager] = None
_config: Optional[MemoryConfig] = None


def create_server(
    project_root: Optional[str] = None,
    config_path: Optional[str] = None,
) -> FastMCP:
    """Create and configure the FastMCP server instance.

    This is the factory function registered in pyproject.toml as the
    ``mcp.servers`` entry point.  It:

    1. Loads configuration (MemoryConfig) with project root auto-detection.
    2. Creates the MemoryStore and WindowManager.
    3. Instantiates the FastMCP server with metadata.
    4. Registers the health_check tool.
    5. Returns the server (ready to ``run()``).

    Parameters
    ----------
    project_root:
        Explicit project root path.  When None, MemoryConfig auto-detects
        the project root by walking up from the current directory.
    config_path:
        Explicit config file path.  When None, looks for
        ``<project_root>/.claude-memory/config.json``.

    Returns
    -------
    FastMCP
        The configured server instance.
    """
    global _server_instance, _window_manager, _config

    # Load configuration.
    _config = MemoryConfig.load(
        project_root=project_root,
        config_path=config_path,
    )
    _config.configure_logging()

    logger.info(
        "Initializing Claude Code Helper MCP server v%s",
        __version__,
    )
    logger.info("Storage path: %s", _config.storage_path)
    logger.info("Window size: %d", _config.window_size)
    logger.info("Project root: %s", _config.project_root)

    # Create the storage engine and window manager.
    _window_manager = WindowManager(
        storage_path=_config.storage_path,
        window_size=_config.window_size,
    )

    logger.info(
        "WindowManager initialized. Tasks in window: %d, Archived: %d",
        _window_manager.total_tasks_in_window(),
        _window_manager.archived_task_count(),
    )

    # Create the FastMCP server.
    _server_instance = FastMCP(
        name="claude-code-helper",
        instructions=(
            "Claude Code Helper provides structured memory for Claude Code "
            "sessions. Use the recording tools to track steps, decisions, "
            "files, and branches during task execution. Use "
            "get_recovery_context after /clear to restore full task awareness."
        ),
        version=__version__,
    )

    # Register tools.
    _register_tools(_server_instance)

    logger.info("FastMCP server created successfully. Tools registered.")

    return _server_instance


def get_server() -> FastMCP:
    """Return the existing server instance, creating it if necessary.

    This is a convenience accessor for code that needs the server reference
    after initial creation (e.g., test fixtures, CLI commands).

    Returns
    -------
    FastMCP
        The singleton server instance.
    """
    global _server_instance
    if _server_instance is None:
        return create_server()
    return _server_instance


def get_window_manager() -> WindowManager:
    """Return the WindowManager instance used by the server.

    Raises
    ------
    RuntimeError
        If the server has not been created yet.
    """
    if _window_manager is None:
        raise RuntimeError(
            "Server has not been initialized. Call create_server() first."
        )
    return _window_manager


def get_config() -> MemoryConfig:
    """Return the MemoryConfig instance used by the server.

    Raises
    ------
    RuntimeError
        If the server has not been created yet.
    """
    if _config is None:
        raise RuntimeError(
            "Server has not been initialized. Call create_server() first."
        )
    return _config


def reset_server() -> None:
    """Reset the server singleton (primarily for testing).

    Clears the module-level references so that the next call to
    ``create_server()`` or ``get_server()`` creates a fresh instance.
    """
    global _server_instance, _window_manager, _config
    _server_instance = None
    _window_manager = None
    _config = None
    logger.debug("Server singleton reset.")


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------


def _register_tools(server: FastMCP) -> None:
    """Register all MCP tools on the server instance.

    Currently registers:
    - health_check -- server health and status information
    - record_step -- record a step taken during the current task
    - record_decision -- record a significant decision during the current task
    - record_file -- record a file action during the current task (CMH-008)
    - record_branch -- record a branch action during the current task (CMH-008)
    - start_task -- start a new task, making it the active task (CMH-009)
    - complete_task -- complete the active task with optional summary (CMH-009)
    - get_task_status -- get details of the current active task (CMH-009)
    - generate_summary -- generate markdown summary for tasks (CMH-010)
    - get_recovery_context -- recover full task context after /clear (CMH-011)
    - check_alignment -- check if an action is aligned with the task scope (CMH-012)
    """

    @server.tool()
    def health_check() -> dict:
        """Check the health and status of the Claude Code Helper MCP server.

        Returns a dictionary with server version, storage status, window
        state, and configuration summary. Use this to verify the server
        is running and connected to the correct project.

        Returns:
            A dictionary with health status information including:
            - server_version: The server version string
            - status: "healthy" or "degraded"
            - storage_path: Path to the .claude-memory directory
            - storage_accessible: Whether the storage directory is readable
            - window_size: Configured sliding window size
            - tasks_in_window: Number of tasks currently in the window
            - current_task: The active task ticket ID or null
            - completed_tasks: Number of completed tasks in the window
            - archived_tasks: Number of archived task IDs
            - project_root: The detected project root path
            - timestamp: ISO 8601 timestamp of this health check
        """
        wm = get_window_manager()
        cfg = get_config()

        storage_accessible = wm.store.storage_root.is_dir()
        current_task_id = None
        if wm.get_current_task() is not None:
            current_task_id = wm.get_current_task().ticket_id

        status = "healthy"
        if not storage_accessible:
            status = "degraded"

        return {
            "server_version": __version__,
            "status": status,
            "storage_path": str(wm.store.storage_root),
            "storage_accessible": storage_accessible,
            "window_size": wm.window_size,
            "tasks_in_window": wm.total_tasks_in_window(),
            "current_task": current_task_id,
            "completed_tasks": wm.completed_task_count(),
            "archived_tasks": wm.archived_task_count(),
            "project_root": cfg.project_root,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @server.tool()
    def record_step(
        action: str,
        description: str = "",
        tool_used: str = "",
        result_summary: str = "",
    ) -> dict:
        """Record a step taken during the current task.

        Each call creates a new StepRecord with an auto-assigned sequential
        step number and UTC timestamp.  The step is appended to the current
        task's step list and persisted immediately.

        An active task must exist (created via start_task).  If no task is
        active, returns an error response.

        Args:
            action: Short description of the action taken (e.g., "Created file",
                "Ran tests", "Edited function").  Required, 1-200 characters.
            description: Detailed description of what happened during this step.
                Optional, up to 2000 characters.
            tool_used: The tool or command that was used (e.g., "Write", "Bash",
                "Edit").  Optional, up to 100 characters.
            result_summary: Summary of the result (e.g., "File created
                successfully", "3 tests passed").  Optional, up to 1000 characters.

        Returns:
            A dictionary with the recorded step details including step_number,
            timestamp, and confirmation, or an error if no active task exists.
        """
        wm = get_window_manager()
        current = wm.get_current_task()

        if current is None:
            logger.warning("record_step called with no active task.")
            return {
                "error": True,
                "message": "No active task. Start a task first with start_task.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        step = current.add_step(
            action=action,
            description=description,
            tool_used=tool_used if tool_used else None,
            result_summary=result_summary if result_summary else None,
        )

        # Persist the updated task to disk.
        wm.save_current_task()

        logger.info(
            "Recorded step #%d for task %s: %s",
            step.step_number,
            current.ticket_id,
            action,
        )

        return {
            "error": False,
            "task_id": current.ticket_id,
            "step_number": step.step_number,
            "action": step.action,
            "description": step.description,
            "tool_used": step.tool_used,
            "result_summary": step.result_summary,
            "timestamp": step.timestamp.isoformat(),
            "total_steps": current.step_count(),
        }

    @server.tool()
    def record_decision(
        decision: str,
        reasoning: str = "",
        alternatives: list[str] | None = None,
        context: str = "",
    ) -> dict:
        """Record a significant decision made during the current task.

        Each call creates a new DecisionRecord with an auto-assigned sequential
        decision number and UTC timestamp.  The decision is appended to the
        current task's decision list and persisted immediately.

        An active task must exist (created via start_task).  If no task is
        active, returns an error response.

        Use this to capture *why* a particular approach was taken, what
        alternatives were considered, and under what context.  Decisions are
        especially valuable for post-/clear recovery -- they help reconstruct
        the reasoning that led to the current state.

        Args:
            decision: The decision that was made (e.g., "Use Pydantic for
                validation", "Split into two modules").  Required, 1-500
                characters.
            reasoning: Why this decision was made.  Optional, up to 2000
                characters.
            alternatives: A list of alternative approaches that were considered
                but not chosen.  Optional.
            context: Relevant context that informed the decision (e.g., "The
                existing codebase already uses Pydantic for models").  Optional,
                up to 1000 characters.

        Returns:
            A dictionary with the recorded decision details including
            decision_number, timestamp, and confirmation, or an error if no
            active task exists.
        """
        wm = get_window_manager()
        current = wm.get_current_task()

        if current is None:
            logger.warning("record_decision called with no active task.")
            return {
                "error": True,
                "message": "No active task. Start a task first with start_task.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        record = current.add_decision(
            decision=decision,
            reasoning=reasoning,
            alternatives=alternatives or [],
            context=context,
        )

        # Persist the updated task to disk.
        wm.save_current_task()

        logger.info(
            "Recorded decision #%d for task %s: %s",
            record.decision_number,
            current.ticket_id,
            decision,
        )

        return {
            "error": False,
            "task_id": current.ticket_id,
            "decision_number": record.decision_number,
            "decision": record.decision,
            "reasoning": record.reasoning,
            "alternatives": record.alternatives,
            "context": record.context,
            "timestamp": record.timestamp.isoformat(),
            "total_decisions": len(current.decisions),
        }

    @server.tool()
    def record_file(
        path: str,
        action: str,
        description: str = "",
    ) -> dict:
        """Record a file action during the current task.

        Tracks files that are created, modified, deleted, renamed, or read
        during task execution.  Files are deduplicated by path: if the same
        file is recorded multiple times, subsequent calls update the existing
        record and append to its action history rather than creating duplicates.

        An active task must exist (created via start_task).  If no task is
        active, returns an error response.

        Args:
            path: Relative file path from the project root (e.g.,
                "src/models/task.py", "tests/test_storage.py").  Required.
            action: The file action performed.  Must be one of: "created",
                "modified", "deleted", "renamed", "read".  Required.
            description: Description of what was done to this file (e.g.,
                "Added TaskMemory model with Pydantic validation").  Optional,
                up to 1000 characters.

        Returns:
            A dictionary with the recorded file details including path, action,
            whether the record was new or updated (deduplicated), action history
            count, total files tracked, and confirmation timestamp.  Returns an
            error if no active task exists or if the action is invalid.
        """
        wm = get_window_manager()
        current = wm.get_current_task()

        if current is None:
            logger.warning("record_file called with no active task.")
            return {
                "error": True,
                "message": "No active task. Start a task first with start_task.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Validate the action string against the FileAction enum.
        try:
            file_action = FileAction(action)
        except ValueError:
            valid_actions = [a.value for a in FileAction]
            logger.warning(
                "record_file called with invalid action '%s'. Valid: %s",
                action,
                valid_actions,
            )
            return {
                "error": True,
                "message": (
                    f"Invalid file action '{action}'. "
                    f"Must be one of: {valid_actions}"
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Check if this file already exists in the task (for dedup reporting).
        existing_before = any(f.path == path for f in current.files)

        # Record the file action (TaskMemory handles deduplication).
        record = current.record_file(
            path=path,
            action=file_action,
            description=description,
        )

        # Persist the updated task to disk.
        wm.save_current_task()

        is_update = existing_before
        logger.info(
            "Recorded file %s for task %s: %s (%s)",
            "update" if is_update else "new",
            current.ticket_id,
            path,
            file_action.value,
        )

        return {
            "error": False,
            "task_id": current.ticket_id,
            "path": record.path,
            "action": record.action.value,
            "description": record.description,
            "is_update": is_update,
            "action_history_count": len(record.action_history),
            "timestamp": record.timestamp.isoformat(),
            "total_files": len(current.files),
        }

    @server.tool()
    def record_branch(
        branch_name: str,
        action: str,
        base_branch: str = "",
    ) -> dict:
        """Record a git branch action during the current task.

        Tracks the lifecycle of branches used during task execution: creation,
        checkout, push, merge, deletion, and pull.  Branches are deduplicated
        by name: if the same branch is recorded multiple times, subsequent
        calls update the existing record and append to its action history.

        An active task must exist (created via start_task).  If no task is
        active, returns an error response.

        Args:
            branch_name: The full branch name (e.g.,
                "feature/CMH-008-record-file-tools").  Required, 1-200
                characters.
            action: The branch action performed.  Must be one of: "created",
                "checked_out", "merged", "deleted", "pushed", "pulled".
                Required.
            base_branch: The branch this was created from or merged into
                (e.g., "main").  Optional.

        Returns:
            A dictionary with the recorded branch details including
            branch_name, action, base_branch, whether the record was new or
            updated (deduplicated), action history count, total branches
            tracked, and confirmation timestamp.  Returns an error if no
            active task exists or if the action is invalid.
        """
        wm = get_window_manager()
        current = wm.get_current_task()

        if current is None:
            logger.warning("record_branch called with no active task.")
            return {
                "error": True,
                "message": "No active task. Start a task first with start_task.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Validate the action string against the BranchAction enum.
        try:
            branch_action = BranchAction(action)
        except ValueError:
            valid_actions = [a.value for a in BranchAction]
            logger.warning(
                "record_branch called with invalid action '%s'. Valid: %s",
                action,
                valid_actions,
            )
            return {
                "error": True,
                "message": (
                    f"Invalid branch action '{action}'. "
                    f"Must be one of: {valid_actions}"
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Check if this branch already exists in the task (for dedup reporting).
        existing_before = any(
            b.branch_name == branch_name for b in current.branches
        )

        # Record the branch action (TaskMemory handles deduplication).
        record = current.record_branch(
            branch_name=branch_name,
            action=branch_action,
            base_branch=base_branch if base_branch else None,
        )

        # Persist the updated task to disk.
        wm.save_current_task()

        is_update = existing_before
        logger.info(
            "Recorded branch %s for task %s: %s (%s)",
            "update" if is_update else "new",
            current.ticket_id,
            branch_name,
            branch_action.value,
        )

        return {
            "error": False,
            "task_id": current.ticket_id,
            "branch_name": record.branch_name,
            "action": record.action.value,
            "base_branch": record.base_branch,
            "is_update": is_update,
            "action_history_count": len(record.action_history),
            "timestamp": record.timestamp.isoformat(),
            "total_branches": len(current.branches),
        }

    @server.tool()
    def start_task(
        ticket_id: str,
        title: str,
        description: str = "",
        phase: str = "",
    ) -> dict:
        """Start a new task and make it the active task in memory.

        Creates a new TaskMemory via the WindowManager, which handles
        persistence and window lifecycle management.  Only one task can be
        active at a time -- if a task is already active, returns an error
        instructing the caller to complete or fail the current task first.

        Args:
            ticket_id: The ticket identifier (e.g., "CMH-009").  Required,
                1-50 characters.
            title: Human-readable title of the task (e.g., "Task lifecycle
                management").  Required, 1-200 characters.
            description: Optional description of the task scope and goals.
                Up to 2000 characters.
            phase: Optional roadmap phase (e.g., "phase-2").  Up to 50
                characters.

        Returns:
            A dictionary with the new task details including ticket_id,
            title, phase, started_at timestamp, task status, and window
            state summary.  Returns an error if a task is already active.
        """
        wm = get_window_manager()

        # Check if there is already an active task.
        if wm.has_active_task():
            current = wm.get_current_task()
            logger.warning(
                "start_task called but task %s is already active.",
                current.ticket_id,
            )
            return {
                "error": True,
                "message": (
                    f"Cannot start new task: task '{current.ticket_id}' "
                    f"('{current.title}') is already active. Complete or "
                    f"fail the current task first."
                ),
                "current_task_id": current.ticket_id,
                "current_task_title": current.title,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Start the new task via WindowManager (handles persistence).
        task = wm.start_new_task(
            ticket_id=ticket_id,
            title=title,
            phase=phase if phase else None,
        )

        # Store the description in metadata if provided.
        if description:
            task.metadata["description"] = description
            wm.save_current_task()

        logger.info(
            "Started task %s ('%s') via MCP tool. Phase: %s",
            ticket_id,
            title,
            phase or "(none)",
        )

        return {
            "error": False,
            "task_id": task.ticket_id,
            "title": task.title,
            "phase": task.phase,
            "description": description,
            "started_at": task.started_at.isoformat(),
            "status": task.status.value,
            "window_state": {
                "tasks_in_window": wm.total_tasks_in_window(),
                "completed_tasks": wm.completed_task_count(),
                "archived_tasks": wm.archived_task_count(),
                "window_size": wm.window_size,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @server.tool()
    def complete_task(
        summary: str = "",
    ) -> dict:
        """Complete the current active task and rotate the sliding window.

        Marks the current task as completed, sets the completion timestamp,
        and triggers the WindowManager's window rotation.  If the window is
        full, the oldest completed task is archived (rotated out).

        The completed task remains on disk in the storage directory even
        after it leaves the window.

        Args:
            summary: Optional completion summary describing what was
                accomplished.  Up to 5000 characters.

        Returns:
            A dictionary with the completed task summary including ticket_id,
            title, duration, step/decision/file/branch counts, archived
            tasks (if any were rotated out), and updated window state.
            Returns an error if no task is active.
        """
        wm = get_window_manager()
        current = wm.get_current_task()

        if current is None:
            logger.warning("complete_task called with no active task.")
            return {
                "error": True,
                "message": "No active task to complete. Start a task first with start_task.",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Capture pre-completion state for reporting.
        task_id = current.ticket_id
        task_title = current.title
        task_phase = current.phase
        step_count = current.step_count()
        decision_count = len(current.decisions)
        file_count = len(current.files)
        branch_count = len(current.branches)
        started_at = current.started_at

        # Capture archived IDs before rotation to detect newly archived.
        archived_before = set(wm.window.archived_task_ids)

        # Complete the task (WindowManager handles persistence and rotation).
        completed_task = wm.complete_current_task(summary)

        # Detect newly archived tasks from window rotation.
        archived_after = set(wm.window.archived_task_ids)
        newly_archived = sorted(archived_after - archived_before)

        # Calculate duration.
        duration_seconds = None
        if completed_task.completed_at and started_at:
            delta = completed_task.completed_at - started_at
            duration_seconds = int(delta.total_seconds())

        logger.info(
            "Completed task %s ('%s') via MCP tool. Steps: %d, Decisions: %d, "
            "Files: %d, Branches: %d. Duration: %ss. Archived: %s.",
            task_id,
            task_title,
            step_count,
            decision_count,
            file_count,
            branch_count,
            duration_seconds,
            newly_archived or "(none)",
        )

        return {
            "error": False,
            "task_id": task_id,
            "title": task_title,
            "phase": task_phase,
            "summary": summary,
            "status": completed_task.status.value,
            "started_at": started_at.isoformat(),
            "completed_at": completed_task.completed_at.isoformat(),
            "duration_seconds": duration_seconds,
            "counts": {
                "steps": step_count,
                "decisions": decision_count,
                "files": file_count,
                "branches": branch_count,
            },
            "newly_archived": newly_archived,
            "window_state": {
                "tasks_in_window": wm.total_tasks_in_window(),
                "completed_tasks": wm.completed_task_count(),
                "archived_tasks": wm.archived_task_count(),
                "window_size": wm.window_size,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @server.tool()
    def get_task_status() -> dict:
        """Get the status and details of the current active task.

        Returns a comprehensive snapshot of the active task including its
        ticket ID, title, phase, current step/decision/file/branch counts,
        start time, and a list of recent steps and key decisions.

        This is useful for orientation after a /clear command or when
        resuming work.  If no task is active, returns a message indicating
        so along with counts of completed and archived tasks.

        Returns:
            A dictionary with the current task details, or a message
            indicating no task is active.  When active, includes ticket_id,
            title, phase, status, started_at, step_count, decision_count,
            file_count, branch_count, recent_steps (last 5), recent_decisions
            (last 3), file_paths, active_branch, and metadata.
        """
        wm = get_window_manager()
        current = wm.get_current_task()

        if current is None:
            logger.info("get_task_status: no active task.")
            return {
                "error": False,
                "has_active_task": False,
                "message": "No active task. Use start_task to begin a new task.",
                "window_state": {
                    "tasks_in_window": wm.total_tasks_in_window(),
                    "completed_tasks": wm.completed_task_count(),
                    "archived_tasks": wm.archived_task_count(),
                    "window_size": wm.window_size,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Build recent steps (last 5).
        recent_steps = []
        for step in current.steps[-5:]:
            recent_steps.append({
                "step_number": step.step_number,
                "action": step.action,
                "tool_used": step.tool_used,
                "timestamp": step.timestamp.isoformat(),
            })

        # Build recent decisions (last 3).
        recent_decisions = []
        for dec in current.decisions[-3:]:
            recent_decisions.append({
                "decision_number": dec.decision_number,
                "decision": dec.decision,
                "timestamp": dec.timestamp.isoformat(),
            })

        logger.info(
            "get_task_status: task %s active. Steps: %d, Decisions: %d, "
            "Files: %d, Branches: %d.",
            current.ticket_id,
            current.step_count(),
            len(current.decisions),
            len(current.files),
            len(current.branches),
        )

        return {
            "error": False,
            "has_active_task": True,
            "task_id": current.ticket_id,
            "title": current.title,
            "phase": current.phase,
            "status": current.status.value,
            "started_at": current.started_at.isoformat(),
            "counts": {
                "steps": current.step_count(),
                "decisions": len(current.decisions),
                "files": len(current.files),
                "branches": len(current.branches),
            },
            "recent_steps": recent_steps,
            "recent_decisions": recent_decisions,
            "file_paths": current.get_file_paths(),
            "active_branch": current.get_active_branch(),
            "next_steps": current.next_steps,
            "metadata": current.metadata,
            "window_state": {
                "tasks_in_window": wm.total_tasks_in_window(),
                "completed_tasks": wm.completed_task_count(),
                "archived_tasks": wm.archived_task_count(),
                "window_size": wm.window_size,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @server.tool()
    def generate_summary(
        ticket_id: str = "",
        summary_type: str = "auto",
    ) -> dict:
        """Generate a markdown summary for a task or the window index.

        Produces a human-readable markdown document from task memory data.
        The output can be used for documentation, review, or post-/clear
        recovery orientation.

        The summary_type parameter controls what is generated:
        - "auto": If a ticket_id is given, generates a task summary for that
          specific task. If no ticket_id is given but an active task exists,
          generates a current progress summary. Otherwise generates a window
          index.
        - "task": Full task summary (requires ticket_id or active task).
        - "current": Current progress summary for the active task.
        - "index": Window index showing all tasks in the sliding window.

        Args:
            ticket_id: Optional ticket ID to generate summary for. If empty
                and summary_type is "auto" or "task", uses the active task.
            summary_type: The type of summary to generate. Must be one of:
                "auto", "task", "current", "index". Default: "auto".

        Returns:
            A dictionary with the generated markdown content, the summary
            type that was produced, and metadata about the generation.
            Returns an error if the requested task is not found or if no
            task is active when one is required.
        """
        wm = get_window_manager()
        gen = MarkdownGenerator()

        valid_types = ("auto", "task", "current", "index")
        if summary_type not in valid_types:
            return {
                "error": True,
                "message": (
                    f"Invalid summary_type '{summary_type}'. "
                    f"Must be one of: {list(valid_types)}"
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Resolve the effective summary type for "auto" mode.
        effective_type = summary_type
        if summary_type == "auto":
            if ticket_id:
                effective_type = "task"
            elif wm.has_active_task():
                effective_type = "current"
            else:
                effective_type = "index"

        # Generate based on effective type.
        if effective_type == "task":
            # Resolve the task.
            if ticket_id:
                task = wm.get_task(ticket_id)
            else:
                task = wm.get_current_task()

            if task is None:
                target = ticket_id if ticket_id else "(active task)"
                logger.warning(
                    "generate_summary: task '%s' not found.", target
                )
                return {
                    "error": True,
                    "message": (
                        f"Task '{target}' not found. Check the ticket ID "
                        f"or start a task first."
                    ),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            markdown = gen.generate_task_summary(task)
            logger.info(
                "Generated task summary for %s (%d chars).",
                task.ticket_id,
                len(markdown),
            )
            return {
                "error": False,
                "summary_type": "task",
                "ticket_id": task.ticket_id,
                "title": task.title,
                "markdown": markdown,
                "markdown_length": len(markdown),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        elif effective_type == "current":
            task = wm.get_current_task()
            if task is None:
                logger.warning(
                    "generate_summary: no active task for 'current' summary."
                )
                return {
                    "error": True,
                    "message": (
                        "No active task. Start a task first with start_task, "
                        "or use summary_type='index' for a window overview."
                    ),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

            markdown = gen.generate_current_summary(task)
            logger.info(
                "Generated current summary for %s (%d chars).",
                task.ticket_id,
                len(markdown),
            )
            return {
                "error": False,
                "summary_type": "current",
                "ticket_id": task.ticket_id,
                "title": task.title,
                "markdown": markdown,
                "markdown_length": len(markdown),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        else:  # effective_type == "index"
            markdown = gen.generate_window_index(
                wm.window,
                completed_tasks=wm.window.completed_tasks,
            )
            logger.info(
                "Generated window index (%d chars, %d tasks in window).",
                len(markdown),
                wm.total_tasks_in_window(),
            )
            return {
                "error": False,
                "summary_type": "index",
                "ticket_id": None,
                "title": "Memory Window Index",
                "markdown": markdown,
                "markdown_length": len(markdown),
                "tasks_in_window": wm.total_tasks_in_window(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    @server.tool()
    def get_recovery_context(
        ticket_id: str = "",
        recent_step_count: int = 10,
        include_prompt: bool = True,
    ) -> dict:
        """Recover full task context after a /clear command.

        This is the primary recovery tool.  After a /clear wipes all session
        context, calling get_recovery_context reconstructs everything needed to
        resume work: the ticket being worked on, what files were modified, which
        branch was active, what decisions were made, what the last step was, and
        what the planned next steps are.

        The tool searches for the target task in this order:

        1. If ``ticket_id`` is given, looks up that specific task (active,
           completed in window, or archived on disk).
        2. If ``ticket_id`` is empty and an active task exists, uses the active
           task.
        3. If ``ticket_id`` is empty and no active task exists, uses the most
           recently completed task in the window (the one you were most likely
           working on before /clear).
        4. If no tasks exist at all, returns an informational message.

        Args:
            ticket_id: Optional ticket ID to recover context for.  If empty,
                auto-detects the most relevant task (active first, then most
                recently completed).
            recent_step_count: Number of recent steps to include in the
                recovery context.  Default: 10.  Range: 1-50.
            include_prompt: Whether to include a pre-formatted human-readable
                recovery prompt in the response.  Default: True.

        Returns:
            A dictionary with the full recovery context including ticket_id,
            title, phase, status, last_step, recent_steps, files_modified,
            active_branch, key_decisions, next_steps, summary_so_far,
            total_steps_completed, task_started_at, metadata, and optionally
            a formatted recovery_prompt.  Returns an error if the specified
            ticket is not found, or an informational message if no tasks exist.
        """
        wm = get_window_manager()

        # Clamp recent_step_count to a safe range.
        recent_step_count = max(1, min(50, recent_step_count))

        # --- Resolve the target task ---
        task = None
        source = ""

        if ticket_id:
            # Explicit ticket ID requested.
            task = wm.get_task(ticket_id)
            if task is None:
                logger.warning(
                    "get_recovery_context: task '%s' not found.", ticket_id
                )
                return {
                    "error": True,
                    "message": (
                        f"Task '{ticket_id}' not found in the memory window "
                        f"or archive. Available tasks: "
                        f"{wm.get_all_known_task_ids()}"
                    ),
                    "available_tasks": wm.get_all_known_task_ids(),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            source = "explicit"
        else:
            # Auto-detect: prefer active task, then most recent completed.
            current = wm.get_current_task()
            if current is not None:
                task = current
                source = "active"
            elif wm.window.completed_tasks:
                # Most recently completed task is at the end of the list.
                task = wm.window.completed_tasks[-1]
                source = "most_recent_completed"
            else:
                # No tasks at all.
                logger.info(
                    "get_recovery_context: no tasks in window or archive."
                )
                return {
                    "error": False,
                    "has_context": False,
                    "message": (
                        "No tasks found in the memory window. "
                        "Use start_task to begin a new task, or check if "
                        "tasks exist in the archive."
                    ),
                    "archived_task_ids": wm.window.archived_task_ids,
                    "window_state": {
                        "tasks_in_window": wm.total_tasks_in_window(),
                        "completed_tasks": wm.completed_task_count(),
                        "archived_tasks": wm.archived_task_count(),
                        "window_size": wm.window_size,
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

        # --- Build the RecoveryContext ---
        recovery = RecoveryContext.from_task_memory(
            task, recent_step_count=recent_step_count
        )

        # Build the response dict from the RecoveryContext.
        result = {
            "error": False,
            "has_context": True,
            "source": source,
            "ticket_id": recovery.ticket_id,
            "title": recovery.title,
            "phase": recovery.phase,
            "status": recovery.status,
            "generated_at": recovery.generated_at.isoformat(),
            "last_step": recovery.last_step,
            "recent_steps": recovery.recent_steps,
            "files_modified": recovery.files_modified,
            "active_branch": recovery.active_branch,
            "key_decisions": recovery.key_decisions,
            "next_steps": recovery.next_steps,
            "summary_so_far": recovery.summary_so_far,
            "total_steps_completed": recovery.total_steps_completed,
            "task_started_at": (
                recovery.task_started_at.isoformat()
                if recovery.task_started_at
                else None
            ),
            "metadata": recovery.metadata,
            "window_state": {
                "tasks_in_window": wm.total_tasks_in_window(),
                "completed_tasks": wm.completed_task_count(),
                "archived_tasks": wm.archived_task_count(),
                "window_size": wm.window_size,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if include_prompt:
            result["recovery_prompt"] = recovery.format_for_prompt()

        logger.info(
            "Generated recovery context for task %s (source: %s). "
            "Steps: %d, Files: %d, Decisions: %d, Branch: %s.",
            recovery.ticket_id,
            source,
            recovery.total_steps_completed,
            len(recovery.files_modified),
            len(recovery.key_decisions),
            recovery.active_branch or "(none)",
        )

        return result

    @server.tool()
    def check_alignment(
        action: str,
        file_path: str = "",
        threshold: float = 0.5,
    ) -> dict:
        """Check whether an action is aligned with the current task's scope.

        Compares the proposed or in-progress action against the active task's
        title, description, recorded files, recorded steps, and phase to
        produce an alignment report.  Use this to detect scope drift before
        it happens -- for example, before editing a file that is unrelated
        to the current ticket.

        The alignment score is a float between 0.0 (completely unrelated) and
        1.0 (perfectly aligned).  Actions scoring below the threshold are
        flagged with warnings.

        An active task must exist (created via start_task).  If no task is
        active, returns an error response.

        Args:
            action: Description of the action being checked (e.g., "Adding
                error handling to the alignment checker", "Editing the
                database migration script").  Required, 1-500 characters.
            file_path: Optional file path the action targets (e.g.,
                "src/detection/alignment.py").  Providing a file path
                improves the accuracy of the alignment check by enabling
                file-scope analysis.
            threshold: Minimum confidence score to consider the action
                aligned.  Default: 0.5.  Range: 0.0-1.0.  Lower values
                are more permissive; higher values are stricter.

        Returns:
            A dictionary with the alignment assessment including confidence
            score, aligned (boolean), warnings (list of strings), scope_info
            (task context used for comparison), and action_analysis (scoring
            breakdown).  Returns an error if no active task exists.
        """
        wm = get_window_manager()
        current = wm.get_current_task()

        if current is None:
            logger.warning("check_alignment called with no active task.")
            return {
                "error": True,
                "message": (
                    "No active task. Start a task first with start_task. "
                    "Alignment checking requires an active task to compare against."
                ),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        # Clamp threshold to valid range.
        effective_threshold = max(0.0, min(1.0, threshold))

        # Gather task context for the checker.
        task_description = current.metadata.get("description", "")
        task_file_paths = current.get_file_paths()
        task_step_actions = [s.action for s in current.steps]

        # Run the alignment check.
        checker = AlignmentChecker(threshold=effective_threshold)
        report = checker.check(
            action=action,
            file_path=file_path if file_path else None,
            task_title=current.title,
            task_description=task_description,
            task_phase=current.phase,
            task_files=task_file_paths,
            task_steps=task_step_actions,
            task_ticket_id=current.ticket_id,
        )

        logger.info(
            "Alignment check for task %s: confidence=%.3f, aligned=%s, "
            "warnings=%d. Action: '%s', File: '%s'.",
            current.ticket_id,
            report.confidence,
            report.aligned,
            len(report.warnings),
            action[:80],
            file_path or "(none)",
        )

        return {
            "error": False,
            "task_id": current.ticket_id,
            "confidence": round(report.confidence, 3),
            "aligned": report.aligned,
            "warnings": report.warnings,
            "scope_info": report.scope_info,
            "action_analysis": report.action_analysis,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
