"""Pipeline lifecycle hooks for Agent Forge integration.

These hooks are called by the Agent Forge pipeline orchestrator at specific
lifecycle events during ticket execution.  Each hook records structured memory
entries so that task context is automatically preserved without requiring
explicit MCP tool calls from the agents.

Hook contract:
- Every hook accepts keyword arguments describing the event.
- Every hook returns a dict with ``recorded: bool`` and event-specific details.
- Hooks NEVER raise exceptions to the caller -- all errors are caught, logged,
  and returned as ``recorded: False`` with an ``error`` field.
- Hooks are idempotent where possible (deduplication of file/branch records).

The hooks operate on the WindowManager instance that is either:
1. Obtained from the running MCP server (if the server is initialized), or
2. Created fresh from the project's storage path (for standalone hook usage).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from claude_code_helper_mcp.config import MemoryConfig
from claude_code_helper_mcp.models.records import BranchAction, FileAction
from claude_code_helper_mcp.storage.window_manager import WindowManager

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level WindowManager cache
# ---------------------------------------------------------------------------

_hook_window_manager: Optional[WindowManager] = None


def _get_window_manager(
    project_root: Optional[str] = None,
) -> Optional[WindowManager]:
    """Obtain a WindowManager for hook operations.

    Attempts to reuse the MCP server's WindowManager if the server has been
    initialized.  Otherwise falls back to creating a standalone instance
    from the project's storage path.

    Returns None if no WindowManager can be obtained (e.g., storage path
    does not exist).
    """
    global _hook_window_manager

    # Try the MCP server's WindowManager first (if server is running).
    try:
        from claude_code_helper_mcp.mcp.server import (
            get_window_manager as server_wm,
        )

        wm = server_wm()
        return wm
    except RuntimeError:
        # Server not initialized -- fall through to standalone mode.
        pass

    # Use cached standalone WindowManager if available.
    if _hook_window_manager is not None:
        return _hook_window_manager

    # Create a standalone WindowManager from config.
    try:
        config = MemoryConfig.load(project_root=project_root)
        _hook_window_manager = WindowManager(
            storage_path=config.storage_path,
            window_size=config.window_size,
        )
        logger.info(
            "Hook WindowManager initialized from %s", config.storage_path
        )
        return _hook_window_manager
    except Exception:
        logger.warning(
            "Failed to initialize hook WindowManager. Hooks will be inactive.",
            exc_info=True,
        )
        return None


def reset_hook_state() -> None:
    """Reset the module-level hook state (primarily for testing).

    Clears the cached standalone WindowManager so that the next hook call
    creates a fresh instance.
    """
    global _hook_window_manager
    _hook_window_manager = None
    logger.debug("Hook state reset.")


# ---------------------------------------------------------------------------
# Hook: post_tool_call
# ---------------------------------------------------------------------------


def post_tool_call(
    *,
    tool_name: str,
    action: str,
    file_path: Optional[str] = None,
    result_summary: Optional[str] = None,
    success: bool = True,
    project_root: Optional[str] = None,
) -> dict[str, Any]:
    """Record a memory step after a tool invocation.

    Called after each significant tool call during pipeline execution (e.g.,
    after Write, Edit, Bash, Read).  Creates a StepRecord and optionally a
    FileRecord if a file_path is provided.

    Parameters
    ----------
    tool_name:
        The tool that was invoked (e.g., "Write", "Edit", "Bash", "Read").
    action:
        Short description of the action taken (e.g., "Created models file",
        "Ran test suite").
    file_path:
        Optional relative file path involved in this tool call.  If provided,
        a FileRecord is also created/updated.
    result_summary:
        Optional summary of the tool call result.
    success:
        Whether the tool call succeeded.  Default: True.
    project_root:
        Optional project root path for standalone WindowManager initialization.

    Returns
    -------
    dict
        ``{"recorded": True, "step_number": N, ...}`` on success,
        ``{"recorded": False, "error": "..."}`` on failure or no active task.
    """
    try:
        wm = _get_window_manager(project_root)
        if wm is None:
            return {
                "recorded": False,
                "error": "WindowManager not available. Memory system may not be initialized.",
            }

        current = wm.get_current_task()
        if current is None:
            return {
                "recorded": False,
                "error": "No active task. Start a task before recording tool calls.",
            }

        # Record the step.
        step = current.add_step(
            action=action,
            description=f"Tool: {tool_name}",
            tool_used=tool_name,
            result_summary=result_summary,
            files_involved=[file_path] if file_path else [],
            success=success,
        )

        # Optionally record the file action.
        file_recorded = False
        if file_path:
            file_action = _infer_file_action(tool_name)
            current.record_file(
                path=file_path,
                action=file_action,
                description=action,
            )
            file_recorded = True

        # Persist.
        wm.save_current_task()

        logger.debug(
            "post_tool_call: step #%d recorded for task %s (tool=%s, file=%s).",
            step.step_number,
            current.ticket_id,
            tool_name,
            file_path or "(none)",
        )

        return {
            "recorded": True,
            "task_id": current.ticket_id,
            "step_number": step.step_number,
            "action": action,
            "tool_name": tool_name,
            "file_recorded": file_recorded,
            "file_path": file_path,
            "timestamp": step.timestamp.isoformat(),
        }

    except Exception as exc:
        logger.warning(
            "post_tool_call hook failed: %s", exc, exc_info=True
        )
        return {
            "recorded": False,
            "error": f"Hook error: {exc}",
        }


# ---------------------------------------------------------------------------
# Hook: post_build_start
# ---------------------------------------------------------------------------


def post_build_start(
    *,
    ticket_id: str,
    title: str,
    branch_name: str,
    phase: Optional[str] = None,
    description: Optional[str] = None,
    base_branch: str = "main",
    project_root: Optional[str] = None,
) -> dict[str, Any]:
    """Record the start of a build phase.

    Called when the Build Agent begins work on a ticket.  This hook:
    1. Starts a new memory task (via WindowManager).
    2. Records the feature branch creation.
    3. Records an initial step noting the build start.

    If a task with the given ticket_id is already active, the hook skips task
    creation and records only the branch and step (idempotent behavior for
    pipeline resume scenarios).

    Parameters
    ----------
    ticket_id:
        The ticket identifier (e.g., "CMH-015").
    title:
        Human-readable title of the ticket.
    branch_name:
        The feature branch name (e.g., "feature/CMH-015-hook-integration").
    phase:
        Optional roadmap phase (e.g., "phase-4").
    description:
        Optional description of the ticket scope.
    base_branch:
        The branch the feature branch was created from.  Default: "main".
    project_root:
        Optional project root path for standalone WindowManager initialization.

    Returns
    -------
    dict
        ``{"recorded": True, "task_id": "...", ...}`` on success,
        ``{"recorded": False, "error": "..."}`` on failure.
    """
    try:
        wm = _get_window_manager(project_root)
        if wm is None:
            return {
                "recorded": False,
                "error": "WindowManager not available. Memory system may not be initialized.",
            }

        # Check if a task is already active (resume scenario).
        task_created = False
        current = wm.get_current_task()

        if current is not None and current.ticket_id == ticket_id:
            # Task already exists for this ticket -- idempotent path.
            logger.info(
                "post_build_start: task %s already active. Recording branch only.",
                ticket_id,
            )
        elif current is not None and current.ticket_id != ticket_id:
            # Different task is active -- this is an unexpected state.
            return {
                "recorded": False,
                "error": (
                    f"Cannot start task '{ticket_id}': task "
                    f"'{current.ticket_id}' is already active. "
                    f"Complete the current task first."
                ),
            }
        else:
            # No active task -- start a new one.
            task = wm.start_new_task(
                ticket_id=ticket_id,
                title=title,
                phase=phase,
            )
            if description:
                task.metadata["description"] = description
                wm.save_current_task()
            task_created = True
            current = task

        # Record the branch creation.
        current.record_branch(
            branch_name=branch_name,
            action=BranchAction.CREATED,
            base_branch=base_branch,
        )

        # Record the build start step.
        current.add_step(
            action=f"Build started for {ticket_id}",
            description=(
                f"Branch '{branch_name}' created from '{base_branch}'. "
                f"Phase: {phase or '(none)'}."
            ),
            tool_used="agent-forge-build",
            result_summary="Build phase initialized",
        )

        wm.save_current_task()

        logger.info(
            "post_build_start: recorded for task %s (branch=%s, created=%s).",
            ticket_id,
            branch_name,
            task_created,
        )

        return {
            "recorded": True,
            "task_id": ticket_id,
            "task_created": task_created,
            "branch_name": branch_name,
            "phase": phase,
            "window_state": {
                "tasks_in_window": wm.total_tasks_in_window(),
                "completed_tasks": wm.completed_task_count(),
                "archived_tasks": wm.archived_task_count(),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as exc:
        logger.warning(
            "post_build_start hook failed: %s", exc, exc_info=True
        )
        return {
            "recorded": False,
            "error": f"Hook error: {exc}",
        }


# ---------------------------------------------------------------------------
# Hook: post_build_complete
# ---------------------------------------------------------------------------


def post_build_complete(
    *,
    ticket_id: str,
    branch_name: str,
    pr_number: int,
    files_changed: Optional[list[str]] = None,
    tests_passed: Optional[int] = None,
    tests_total: Optional[int] = None,
    commit_count: Optional[int] = None,
    summary: Optional[str] = None,
    project_root: Optional[str] = None,
) -> dict[str, Any]:
    """Record the completion of a build phase.

    Called when the Build Agent finishes implementation and creates a PR.
    This hook:
    1. Records all changed files as FileRecords.
    2. Records the branch push.
    3. Records a build-complete step with PR and test details.
    4. Stores build metadata for later use by other hooks/agents.

    Parameters
    ----------
    ticket_id:
        The ticket identifier.
    branch_name:
        The feature branch that was pushed.
    pr_number:
        The GitHub PR number that was created.
    files_changed:
        Optional list of file paths that were modified in the build.
    tests_passed:
        Optional count of tests that passed.
    tests_total:
        Optional total count of tests run.
    commit_count:
        Optional number of commits in the PR.
    summary:
        Optional build summary.
    project_root:
        Optional project root path for standalone WindowManager initialization.

    Returns
    -------
    dict
        ``{"recorded": True, "files_recorded": N, ...}`` on success,
        ``{"recorded": False, "error": "..."}`` on failure.
    """
    try:
        wm = _get_window_manager(project_root)
        if wm is None:
            return {
                "recorded": False,
                "error": "WindowManager not available.",
            }

        current = wm.get_current_task()
        if current is None:
            return {
                "recorded": False,
                "error": "No active task. Expected active task for build completion.",
            }

        if current.ticket_id != ticket_id:
            return {
                "recorded": False,
                "error": (
                    f"Active task '{current.ticket_id}' does not match "
                    f"expected ticket '{ticket_id}'."
                ),
            }

        # Record files changed.
        files_recorded = 0
        if files_changed:
            for fpath in files_changed:
                current.record_file(
                    path=fpath,
                    action=FileAction.MODIFIED,
                    description=f"Modified in build for {ticket_id}",
                )
                files_recorded += 1

        # Record branch push.
        current.record_branch(
            branch_name=branch_name,
            action=BranchAction.PUSHED,
        )

        # Build the test result string.
        test_info = ""
        if tests_total is not None:
            test_info = f" Tests: {tests_passed or 0}/{tests_total}."

        # Record the build-complete step.
        step_description = (
            f"PR #{pr_number} created on branch '{branch_name}'. "
            f"{files_recorded} files changed."
            f"{test_info}"
        )
        if commit_count:
            step_description += f" Commits: {commit_count}."

        current.add_step(
            action=f"Build complete for {ticket_id}",
            description=step_description,
            tool_used="agent-forge-build",
            result_summary=summary or f"PR #{pr_number} created",
        )

        # Store build metadata for downstream hooks.
        current.metadata["pr_number"] = pr_number
        current.metadata["branch_name"] = branch_name
        current.metadata["build_complete"] = True
        if tests_total is not None:
            current.metadata["tests_passed"] = tests_passed
            current.metadata["tests_total"] = tests_total
        if commit_count is not None:
            current.metadata["commit_count"] = commit_count

        # Record a decision about the build approach.
        current.add_decision(
            decision=f"Build approach for {ticket_id}",
            reasoning=summary or "Implementation completed per ticket specification.",
            context=f"PR #{pr_number} on branch {branch_name}",
        )

        wm.save_current_task()

        logger.info(
            "post_build_complete: recorded for task %s (PR=#%d, files=%d).",
            ticket_id,
            pr_number,
            files_recorded,
        )

        return {
            "recorded": True,
            "task_id": ticket_id,
            "pr_number": pr_number,
            "files_recorded": files_recorded,
            "branch_name": branch_name,
            "test_info": test_info.strip() if test_info else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as exc:
        logger.warning(
            "post_build_complete hook failed: %s", exc, exc_info=True
        )
        return {
            "recorded": False,
            "error": f"Hook error: {exc}",
        }


# ---------------------------------------------------------------------------
# Hook: post_merge
# ---------------------------------------------------------------------------


def post_merge(
    *,
    ticket_id: str,
    branch_name: str,
    pr_number: int,
    merge_strategy: str = "squash",
    target_branch: str = "main",
    completion_summary: Optional[str] = None,
    project_root: Optional[str] = None,
) -> dict[str, Any]:
    """Record the merge event and complete the memory task.

    Called after the orchestrator auto-merges the PR.  This hook:
    1. Records the branch merge action.
    2. Records the branch deletion (if applicable).
    3. Adds a final merge step.
    4. Completes the memory task with a comprehensive summary.
    5. Triggers window rotation (archival of oldest task if window is full).

    After this hook completes, the memory task transitions from ``active``
    to ``completed`` and the sliding window may rotate.

    Parameters
    ----------
    ticket_id:
        The ticket identifier.
    branch_name:
        The feature branch that was merged.
    pr_number:
        The GitHub PR number that was merged.
    merge_strategy:
        The merge strategy used ("squash", "merge", "rebase").
    target_branch:
        The branch the PR was merged into.  Default: "main".
    completion_summary:
        Optional summary of the overall ticket completion.
    project_root:
        Optional project root path for standalone WindowManager initialization.

    Returns
    -------
    dict
        ``{"recorded": True, "task_completed": True, ...}`` on success,
        ``{"recorded": False, "error": "..."}`` on failure.
    """
    try:
        wm = _get_window_manager(project_root)
        if wm is None:
            return {
                "recorded": False,
                "error": "WindowManager not available.",
            }

        current = wm.get_current_task()
        if current is None:
            return {
                "recorded": False,
                "error": "No active task. Expected active task for merge recording.",
            }

        if current.ticket_id != ticket_id:
            return {
                "recorded": False,
                "error": (
                    f"Active task '{current.ticket_id}' does not match "
                    f"expected ticket '{ticket_id}'."
                ),
            }

        # Record the branch merge.
        current.record_branch(
            branch_name=branch_name,
            action=BranchAction.MERGED,
            base_branch=target_branch,
        )

        # Record the branch deletion (standard post-merge cleanup).
        current.record_branch(
            branch_name=branch_name,
            action=BranchAction.DELETED,
        )

        # Build a comprehensive summary.
        step_count = current.step_count()
        file_count = len(current.files)
        decision_count = len(current.decisions)
        branch_count = len(current.branches)

        merge_summary = (
            f"PR #{pr_number} merged to '{target_branch}' via {merge_strategy}. "
            f"Branch '{branch_name}' deleted. "
            f"Task stats: {step_count} steps, {file_count} files, "
            f"{decision_count} decisions, {branch_count} branches."
        )

        # Record the merge step.
        current.add_step(
            action=f"PR #{pr_number} merged ({merge_strategy})",
            description=merge_summary,
            tool_used="agent-forge-orchestrator",
            result_summary=f"Merged to {target_branch}",
        )

        # Persist the step before completing (complete_current_task changes status).
        wm.save_current_task()

        # Build the completion summary.
        full_summary = completion_summary or ""
        if not full_summary:
            test_info = ""
            if "tests_total" in current.metadata:
                test_info = (
                    f" Tests: {current.metadata.get('tests_passed', 0)}/"
                    f"{current.metadata['tests_total']}."
                )
            full_summary = (
                f"{current.title}. PR #{pr_number} merged via {merge_strategy}. "
                f"{step_count} steps, {file_count} files modified, "
                f"{decision_count} decisions.{test_info}"
            )

        # Complete the task -- this triggers window rotation.
        completed_task = wm.complete_current_task(full_summary)

        logger.info(
            "post_merge: task %s completed and memory finalized "
            "(PR=#%d, steps=%d, files=%d).",
            ticket_id,
            pr_number,
            step_count,
            file_count,
        )

        return {
            "recorded": True,
            "task_id": ticket_id,
            "task_completed": True,
            "pr_number": pr_number,
            "merge_strategy": merge_strategy,
            "target_branch": target_branch,
            "task_summary": {
                "steps": step_count,
                "files": file_count,
                "decisions": decision_count,
                "branches": branch_count,
            },
            "window_state": {
                "tasks_in_window": wm.total_tasks_in_window(),
                "completed_tasks": wm.completed_task_count(),
                "archived_tasks": wm.archived_task_count(),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as exc:
        logger.warning(
            "post_merge hook failed: %s", exc, exc_info=True
        )
        return {
            "recorded": False,
            "error": f"Hook error: {exc}",
        }


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _infer_file_action(tool_name: str) -> FileAction:
    """Infer the file action from a tool name.

    Maps common Claude Code tool names to FileAction enum values.
    """
    tool_lower = tool_name.lower()

    if tool_lower in ("write",):
        return FileAction.CREATED
    elif tool_lower in ("edit",):
        return FileAction.MODIFIED
    elif tool_lower in ("read",):
        return FileAction.READ
    elif tool_lower in ("bash",):
        # Bash commands could be anything -- default to modified.
        return FileAction.MODIFIED
    else:
        return FileAction.MODIFIED
