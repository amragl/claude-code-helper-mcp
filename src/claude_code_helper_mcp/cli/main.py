"""Main Click CLI entry point for the memory command.

Provides the ``memory`` CLI group and the ``status``, ``list``, and
``show`` subcommands.  The CLI exposes developer-facing commands for
inspecting and managing the structured memory system.

Entry point registered in pyproject.toml::

    [project.scripts]
    memory = "claude_code_helper_mcp.cli.main:cli"

Usage examples::

    memory --version
    memory status
    memory status --json
    memory status --storage-path /path/to/.claude-memory
    memory list
    memory list --all
    memory list --format json
    memory show CMH-014
    memory show CMH-014 --format json
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click

from claude_code_helper_mcp import __version__


@click.group()
@click.version_option(version=__version__, prog_name="claude-code-helper-mcp")
@click.option(
    "--storage-path",
    type=click.Path(exists=False),
    default=None,
    envvar="CLAUDE_MEMORY_STORAGE_PATH",
    help="Path to the .claude-memory storage directory. Auto-detected if not set.",
)
@click.pass_context
def cli(ctx: click.Context, storage_path: Optional[str]) -> None:
    """Claude Code Helper -- Memory management for Claude Code sessions."""
    ctx.ensure_object(dict)
    ctx.obj["storage_path"] = storage_path


@cli.command()
@click.option(
    "--json-output",
    "output_json",
    is_flag=True,
    default=False,
    help="Output status as JSON instead of human-readable text.",
)
@click.pass_context
def status(ctx: click.Context, output_json: bool) -> None:
    """Show memory system status: current task, window state, storage info.

    Displays a comprehensive overview of the memory system including the
    active task (if any), sliding window contents, storage directory health,
    and configuration. Use --json-output for machine-readable output.
    """
    storage_path = ctx.obj.get("storage_path")

    status_data = _collect_status(storage_path)

    if output_json:
        click.echo(json.dumps(status_data, indent=2, default=str))
    else:
        _render_status_text(status_data)


@cli.command("list")
@click.option(
    "--all",
    "show_all",
    is_flag=True,
    default=False,
    help="Include archived tasks (loaded from disk) in addition to window tasks.",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["table", "json"], case_sensitive=False),
    default="table",
    help="Output format: table (default) or json.",
)
@click.pass_context
def list_tasks(ctx: click.Context, show_all: bool, output_format: str) -> None:
    """List tasks in the memory window.

    Shows a table of all tasks currently in the sliding window (completed
    tasks plus the active task, if any).  Use --all to also include archived
    tasks that have been rotated out of the window.  Use --format json for
    machine-readable output.
    """
    storage_path = ctx.obj.get("storage_path")
    list_data = _collect_list(storage_path, show_all)

    if output_format == "json":
        click.echo(json.dumps(list_data, indent=2, default=str))
    else:
        _render_list_table(list_data, show_all)


@cli.command()
@click.argument("ticket_id")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format: text (default) or json.",
)
@click.pass_context
def show(ctx: click.Context, ticket_id: str, output_format: str) -> None:
    """Show full details for a specific task.

    Displays the complete memory record for TICKET_ID including status,
    timeline of all steps, file modifications, branch activity, and
    decisions made.  The task is looked up in the window first, then
    in archived task files on disk.

    Use --format json for machine-readable output.
    """
    storage_path = ctx.obj.get("storage_path")
    show_data = _collect_show(storage_path, ticket_id)

    if output_format == "json":
        click.echo(json.dumps(show_data, indent=2, default=str))
    else:
        _render_show_text(show_data)


# ---------------------------------------------------------------------------
# Status data collection
# ---------------------------------------------------------------------------


def _collect_status(storage_path: Optional[str]) -> dict:
    """Collect all status information into a structured dictionary.

    Initialises the MemoryConfig, MemoryStore, and WindowManager using
    the provided storage path (or auto-detection), then gathers window
    state, current task info, storage metrics, and configuration summary.

    Parameters
    ----------
    storage_path:
        Explicit storage path.  When None, auto-detection is used.

    Returns
    -------
    dict
        Complete status data suitable for JSON serialisation or text rendering.
    """
    from claude_code_helper_mcp.config import MemoryConfig
    from claude_code_helper_mcp.storage.store import MemoryStore
    from claude_code_helper_mcp.storage.window_manager import WindowManager

    # Load configuration.
    try:
        if storage_path:
            # When a storage path is given, we infer project root from it.
            config = MemoryConfig(storage_path=storage_path)
        else:
            config = MemoryConfig.load()
    except Exception as exc:
        return {
            "status": "error",
            "error": f"Failed to load configuration: {exc}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # Initialise storage and window manager.
    try:
        store = MemoryStore(config.storage_path)
        manager = WindowManager(
            storage_path=config.storage_path,
            window_size=config.window_size,
        )
    except Exception as exc:
        return {
            "status": "error",
            "error": f"Failed to initialise storage: {exc}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # Collect storage metrics.
    storage_root = Path(config.storage_path)
    storage_exists = storage_root.is_dir()
    tasks_dir = storage_root / "tasks"
    task_file_count = 0
    total_size_bytes = 0

    if storage_exists:
        for entry in storage_root.rglob("*"):
            if entry.is_file():
                total_size_bytes += entry.stat().st_size
        if tasks_dir.is_dir():
            task_file_count = sum(
                1 for f in tasks_dir.iterdir()
                if f.is_file() and f.suffix == ".json"
            )

    # Window state.
    window = manager.window
    current_task = manager.get_current_task()

    current_task_info = None
    if current_task is not None:
        elapsed = datetime.now(timezone.utc) - current_task.started_at
        elapsed_minutes = int(elapsed.total_seconds() / 60)
        current_task_info = {
            "ticket_id": current_task.ticket_id,
            "title": current_task.title,
            "phase": current_task.phase,
            "status": current_task.status.value,
            "started_at": current_task.started_at.isoformat(),
            "elapsed_minutes": elapsed_minutes,
            "steps": current_task.step_count(),
            "files": len(current_task.files),
            "decisions": len(current_task.decisions),
            "branches": len(current_task.branches),
            "active_branch": current_task.get_active_branch(),
        }

    # Completed tasks in window.
    completed_tasks_info = []
    for task in window.completed_tasks:
        completed_tasks_info.append({
            "ticket_id": task.ticket_id,
            "title": task.title,
            "status": task.status.value,
            "completed_at": (
                task.completed_at.isoformat() if task.completed_at else None
            ),
            "steps": task.step_count(),
            "files": len(task.files),
        })

    # Last activity: find the most recent timestamp across all tasks.
    last_activity = _determine_last_activity(current_task, window.completed_tasks)

    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": __version__,
        "current_task": current_task_info,
        "window": {
            "window_size": window.window_size,
            "tasks_in_window": manager.total_tasks_in_window(),
            "completed_count": manager.completed_task_count(),
            "archived_count": manager.archived_task_count(),
            "has_active_task": current_task is not None,
            "completed_tasks": completed_tasks_info,
            "archived_task_ids": window.archived_task_ids,
        },
        "storage": {
            "path": str(storage_root),
            "exists": storage_exists,
            "task_files": task_file_count,
            "total_size_bytes": total_size_bytes,
            "total_size_human": _format_bytes(total_size_bytes),
        },
        "config": {
            "project_root": config.project_root,
            "storage_path": config.storage_path,
            "window_size": config.window_size,
            "log_level": config.log_level,
            "auto_save": config.auto_save,
            "archive_completed": config.archive_completed,
        },
        "last_activity": last_activity,
    }


def _determine_last_activity(
    current_task,
    completed_tasks: list,
) -> Optional[str]:
    """Find the most recent timestamp across all known tasks.

    Checks the current task's started_at and latest step timestamp, and
    all completed tasks' completed_at timestamps.  Returns the most
    recent one as an ISO 8601 string, or None if no tasks exist.
    """
    candidates: list[datetime] = []

    if current_task is not None:
        candidates.append(current_task.started_at)
        if current_task.steps:
            candidates.append(current_task.steps[-1].timestamp)

    for task in completed_tasks:
        if task.completed_at is not None:
            candidates.append(task.completed_at)
        if task.steps:
            candidates.append(task.steps[-1].timestamp)

    if not candidates:
        return None

    return max(candidates).isoformat()


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------


def _render_status_text(data: dict) -> None:
    """Render status data as formatted text to stdout.

    Produces a human-readable dashboard-style output with sections for
    current task, window state, storage info, and configuration.
    """
    if data.get("status") == "error":
        click.secho("ERROR: " + data.get("error", "Unknown error"), fg="red", err=True)
        sys.exit(1)

    click.secho("Claude Code Helper -- Memory Status", fg="cyan", bold=True)
    click.secho("=" * 42, fg="cyan")
    click.echo(f"Version: {data.get('version', 'unknown')}")
    click.echo()

    # Current task section.
    task = data.get("current_task")
    if task is not None:
        click.secho("Active Task", fg="green", bold=True)
        click.secho("-" * 20, fg="green")
        click.echo(f"  Ticket:   {task['ticket_id']}")
        click.echo(f"  Title:    {task['title']}")
        if task.get("phase"):
            click.echo(f"  Phase:    {task['phase']}")
        click.echo(f"  Status:   {task['status']}")
        click.echo(f"  Elapsed:  {task['elapsed_minutes']} minutes")
        click.echo(f"  Steps:    {task['steps']}")
        click.echo(f"  Files:    {task['files']}")
        click.echo(f"  Decisions:{' ' + str(task['decisions'])}")
        if task.get("active_branch"):
            click.echo(f"  Branch:   {task['active_branch']}")
    else:
        click.secho("No active task", fg="yellow")
    click.echo()

    # Window section.
    win = data.get("window", {})
    click.secho("Sliding Window", fg="blue", bold=True)
    click.secho("-" * 20, fg="blue")
    click.echo(f"  Window size:     {win.get('window_size', '?')}")
    click.echo(f"  Tasks in window: {win.get('tasks_in_window', 0)}")
    click.echo(f"  Completed:       {win.get('completed_count', 0)}")
    click.echo(f"  Archived total:  {win.get('archived_count', 0)}")

    completed = win.get("completed_tasks", [])
    if completed:
        click.echo()
        click.echo("  Recent completed tasks:")
        for ct in completed:
            completed_at = ct.get("completed_at", "?")
            if completed_at and completed_at != "?":
                # Show only date portion for readability.
                completed_at = completed_at[:10]
            click.echo(
                f"    {ct['ticket_id']:10s} {ct['title'][:40]:40s} "
                f"{ct['status']:10s} {completed_at}"
            )

    archived_ids = win.get("archived_task_ids", [])
    if archived_ids:
        click.echo(f"  Archived IDs:    {', '.join(archived_ids)}")
    click.echo()

    # Storage section.
    stor = data.get("storage", {})
    click.secho("Storage", fg="magenta", bold=True)
    click.secho("-" * 20, fg="magenta")
    click.echo(f"  Path:       {stor.get('path', '?')}")
    click.echo(f"  Exists:     {'Yes' if stor.get('exists') else 'No'}")
    click.echo(f"  Task files: {stor.get('task_files', 0)}")
    click.echo(f"  Total size: {stor.get('total_size_human', '0 B')}")
    click.echo()

    # Config section.
    cfg = data.get("config", {})
    click.secho("Configuration", fg="white", bold=True)
    click.secho("-" * 20, fg="white")
    click.echo(f"  Project root:      {cfg.get('project_root', '?')}")
    click.echo(f"  Window size:       {cfg.get('window_size', '?')}")
    click.echo(f"  Log level:         {cfg.get('log_level', '?')}")
    click.echo(f"  Auto-save:         {cfg.get('auto_save', '?')}")
    click.echo(f"  Archive completed: {cfg.get('archive_completed', '?')}")
    click.echo()

    # Last activity.
    last = data.get("last_activity")
    if last:
        click.echo(f"Last activity: {last}")
    else:
        click.echo("Last activity: (none)")


# ---------------------------------------------------------------------------
# List data collection
# ---------------------------------------------------------------------------


def _collect_list(storage_path: Optional[str], show_all: bool) -> dict:
    """Collect task list information for the ``list`` command.

    Gathers tasks from the sliding window.  When *show_all* is True, also
    loads archived tasks from disk.

    Parameters
    ----------
    storage_path:
        Explicit storage path.  When None, auto-detection is used.
    show_all:
        If True, include archived tasks loaded from disk.

    Returns
    -------
    dict
        List data suitable for JSON serialisation or table rendering.
    """
    from claude_code_helper_mcp.config import MemoryConfig
    from claude_code_helper_mcp.storage.window_manager import WindowManager

    try:
        if storage_path:
            config = MemoryConfig(storage_path=storage_path)
        else:
            config = MemoryConfig.load()
    except Exception as exc:
        return {
            "status": "error",
            "error": f"Failed to load configuration: {exc}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    try:
        manager = WindowManager(
            storage_path=config.storage_path,
            window_size=config.window_size,
        )
    except Exception as exc:
        return {
            "status": "error",
            "error": f"Failed to initialise storage: {exc}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    tasks: list[dict] = []

    # Current active task (if any).
    current = manager.get_current_task()
    if current is not None:
        elapsed = datetime.now(timezone.utc) - current.started_at
        elapsed_minutes = int(elapsed.total_seconds() / 60)
        tasks.append({
            "ticket_id": current.ticket_id,
            "title": current.title,
            "phase": current.phase,
            "status": current.status.value,
            "started_at": current.started_at.isoformat(),
            "completed_at": None,
            "steps": current.step_count(),
            "files": len(current.files),
            "decisions": len(current.decisions),
            "elapsed_minutes": elapsed_minutes,
            "source": "active",
        })

    # Completed tasks in the window (oldest first).
    for task in manager.window.completed_tasks:
        duration_minutes = None
        if task.completed_at is not None:
            duration = task.completed_at - task.started_at
            duration_minutes = int(duration.total_seconds() / 60)
        tasks.append({
            "ticket_id": task.ticket_id,
            "title": task.title,
            "phase": task.phase,
            "status": task.status.value,
            "started_at": task.started_at.isoformat(),
            "completed_at": (
                task.completed_at.isoformat() if task.completed_at else None
            ),
            "steps": task.step_count(),
            "files": len(task.files),
            "decisions": len(task.decisions),
            "duration_minutes": duration_minutes,
            "source": "window",
        })

    # Archived tasks (loaded from disk) when --all is requested.
    archived_tasks: list[dict] = []
    if show_all:
        archived_ids = manager.window.archived_task_ids
        for tid in archived_ids:
            archived_task = manager.store.load_task(tid)
            if archived_task is not None:
                duration_minutes = None
                if archived_task.completed_at is not None:
                    duration = archived_task.completed_at - archived_task.started_at
                    duration_minutes = int(duration.total_seconds() / 60)
                archived_tasks.append({
                    "ticket_id": archived_task.ticket_id,
                    "title": archived_task.title,
                    "phase": archived_task.phase,
                    "status": archived_task.status.value,
                    "started_at": archived_task.started_at.isoformat(),
                    "completed_at": (
                        archived_task.completed_at.isoformat()
                        if archived_task.completed_at
                        else None
                    ),
                    "steps": archived_task.step_count(),
                    "files": len(archived_task.files),
                    "decisions": len(archived_task.decisions),
                    "duration_minutes": duration_minutes,
                    "source": "archived",
                })
            else:
                # Task file exists in archived list but cannot be loaded.
                archived_tasks.append({
                    "ticket_id": tid,
                    "title": "(unavailable)",
                    "phase": None,
                    "status": "archived",
                    "started_at": None,
                    "completed_at": None,
                    "steps": 0,
                    "files": 0,
                    "decisions": 0,
                    "duration_minutes": None,
                    "source": "archived",
                })

    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "window_size": manager.window_size,
        "tasks": tasks,
        "archived_tasks": archived_tasks,
        "total_in_window": len(tasks),
        "total_archived": len(archived_tasks),
    }


def _render_list_table(data: dict, show_all: bool) -> None:
    """Render the task list as a formatted table.

    Parameters
    ----------
    data:
        List data from ``_collect_list``.
    show_all:
        Whether archived tasks were requested.
    """
    if data.get("status") == "error":
        click.secho(
            "ERROR: " + data.get("error", "Unknown error"), fg="red", err=True
        )
        sys.exit(1)

    tasks = data.get("tasks", [])
    archived = data.get("archived_tasks", [])
    window_size = data.get("window_size", "?")

    click.secho("Claude Code Helper -- Task List", fg="cyan", bold=True)
    click.secho("=" * 42, fg="cyan")
    click.echo(
        f"Window: {data.get('total_in_window', 0)} task(s) "
        f"(max {window_size} completed)"
    )

    if show_all:
        click.echo(f"Archived: {data.get('total_archived', 0)} task(s)")
    click.echo()

    if not tasks and not archived:
        click.secho("No tasks found.", fg="yellow")
        return

    # Table header.
    header = (
        f"{'Ticket':<12s} {'Title':<35s} {'Status':<12s} "
        f"{'Steps':>5s} {'Files':>5s} {'Time':>8s}"
    )
    click.secho(header, bold=True)
    click.secho("-" * len(header), dim=True)

    # Window tasks (active + completed).
    for t in tasks:
        _render_list_row(t)

    # Separator between window and archived tasks.
    if archived:
        click.echo()
        click.secho("Archived Tasks", fg="yellow", bold=True)
        click.secho("-" * len(header), dim=True)
        for t in archived:
            _render_list_row(t)

    click.echo()


def _render_list_row(task: dict) -> None:
    """Render a single task row in the list table.

    Parameters
    ----------
    task:
        Task dictionary from the list data.
    """
    ticket = task.get("ticket_id", "?")
    title = task.get("title", "")
    if len(title) > 33:
        title = title[:32] + "..."
    status_val = task.get("status", "?")

    steps = str(task.get("steps", 0))
    files = str(task.get("files", 0))

    # Time display: for active tasks show elapsed, for completed show duration.
    time_str = ""
    if task.get("source") == "active":
        elapsed = task.get("elapsed_minutes")
        if elapsed is not None:
            time_str = _format_duration(elapsed)
    else:
        duration = task.get("duration_minutes")
        if duration is not None:
            time_str = _format_duration(duration)

    # Colour the status.
    status_colours = {
        "active": "green",
        "completed": "blue",
        "failed": "red",
        "archived": "yellow",
    }
    colour = status_colours.get(status_val, None)

    row = (
        f"{ticket:<12s} {title:<35s} "
    )
    click.echo(row, nl=False)
    click.secho(f"{status_val:<12s}", fg=colour, nl=False)
    click.echo(f"{steps:>5s} {files:>5s} {time_str:>8s}")


# ---------------------------------------------------------------------------
# Show data collection
# ---------------------------------------------------------------------------


def _collect_show(storage_path: Optional[str], ticket_id: str) -> dict:
    """Collect full task details for the ``show`` command.

    Looks up the task by ticket ID in the window first, then on disk.

    Parameters
    ----------
    storage_path:
        Explicit storage path.  When None, auto-detection is used.
    ticket_id:
        The ticket ID to look up (e.g., ``"CMH-014"``).

    Returns
    -------
    dict
        Complete task data suitable for JSON serialisation or text rendering.
    """
    from claude_code_helper_mcp.config import MemoryConfig
    from claude_code_helper_mcp.storage.window_manager import WindowManager

    try:
        if storage_path:
            config = MemoryConfig(storage_path=storage_path)
        else:
            config = MemoryConfig.load()
    except Exception as exc:
        return {
            "status": "error",
            "error": f"Failed to load configuration: {exc}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    try:
        manager = WindowManager(
            storage_path=config.storage_path,
            window_size=config.window_size,
        )
    except Exception as exc:
        return {
            "status": "error",
            "error": f"Failed to initialise storage: {exc}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # Look up the task (window first, then disk).
    task = manager.get_task(ticket_id)
    if task is None:
        return {
            "status": "not_found",
            "error": f"Task '{ticket_id}' not found in window or on disk.",
            "ticket_id": ticket_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # Determine the source.
    source = "archived"
    if manager.window.current_task and manager.window.current_task.ticket_id == ticket_id:
        source = "active"
    else:
        for ct in manager.window.completed_tasks:
            if ct.ticket_id == ticket_id:
                source = "window"
                break

    # Duration calculation.
    duration_minutes = None
    if task.completed_at is not None:
        duration = task.completed_at - task.started_at
        duration_minutes = int(duration.total_seconds() / 60)
    elif task.status.value == "active":
        elapsed = datetime.now(timezone.utc) - task.started_at
        duration_minutes = int(elapsed.total_seconds() / 60)

    # Build step timeline.
    steps_data = []
    for step in task.steps:
        steps_data.append({
            "step_number": step.step_number,
            "timestamp": step.timestamp.isoformat(),
            "action": step.action,
            "description": step.description,
            "tool_used": step.tool_used,
            "result_summary": step.result_summary,
            "files_involved": step.files_involved,
            "success": step.success,
        })

    # Build file list.
    files_data = []
    for f in task.files:
        files_data.append({
            "path": f.path,
            "action": f.action.value,
            "description": f.description,
            "timestamp": f.timestamp.isoformat(),
            "action_count": 1 + len(f.action_history),
        })

    # Build branch list.
    branches_data = []
    for b in task.branches:
        branches_data.append({
            "branch_name": b.branch_name,
            "action": b.action.value,
            "base_branch": b.base_branch,
            "timestamp": b.timestamp.isoformat(),
            "action_count": 1 + len(b.action_history),
        })

    # Build decision list.
    decisions_data = []
    for d in task.decisions:
        decisions_data.append({
            "decision_number": d.decision_number,
            "timestamp": d.timestamp.isoformat(),
            "decision": d.decision,
            "reasoning": d.reasoning,
            "alternatives": d.alternatives,
            "context": d.context,
        })

    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticket_id": task.ticket_id,
        "title": task.title,
        "phase": task.phase,
        "task_status": task.status.value,
        "source": source,
        "started_at": task.started_at.isoformat(),
        "completed_at": (
            task.completed_at.isoformat() if task.completed_at else None
        ),
        "duration_minutes": duration_minutes,
        "summary": task.summary,
        "next_steps": task.next_steps,
        "metadata": task.metadata,
        "steps": steps_data,
        "files": files_data,
        "branches": branches_data,
        "decisions": decisions_data,
        "counts": {
            "steps": len(steps_data),
            "files": len(files_data),
            "branches": len(branches_data),
            "decisions": len(decisions_data),
        },
    }


def _render_show_text(data: dict) -> None:
    """Render full task details as formatted text.

    Parameters
    ----------
    data:
        Show data from ``_collect_show``.
    """
    if data.get("status") == "error":
        click.secho(
            "ERROR: " + data.get("error", "Unknown error"), fg="red", err=True
        )
        sys.exit(1)

    if data.get("status") == "not_found":
        click.secho(
            f"Task not found: {data.get('ticket_id', '?')}", fg="red", err=True
        )
        sys.exit(1)

    # Header.
    click.secho(
        f"Claude Code Helper -- Task Details: {data['ticket_id']}",
        fg="cyan",
        bold=True,
    )
    click.secho("=" * 50, fg="cyan")
    click.echo()

    # Overview section.
    click.secho("Overview", fg="green", bold=True)
    click.secho("-" * 20, fg="green")
    click.echo(f"  Ticket:     {data['ticket_id']}")
    click.echo(f"  Title:      {data['title']}")
    if data.get("phase"):
        click.echo(f"  Phase:      {data['phase']}")
    click.echo(f"  Status:     {data['task_status']}")
    click.echo(f"  Source:     {data['source']}")
    click.echo(f"  Started:    {_format_iso_short(data['started_at'])}")
    if data.get("completed_at"):
        click.echo(f"  Completed:  {_format_iso_short(data['completed_at'])}")
    if data.get("duration_minutes") is not None:
        label = "Elapsed" if data["task_status"] == "active" else "Duration"
        click.echo(
            f"  {label + ':':12s}{_format_duration(data['duration_minutes'])}"
        )
    if data.get("summary"):
        click.echo(f"  Summary:    {data['summary']}")
    click.echo()

    # Counts summary line.
    counts = data.get("counts", {})
    click.echo(
        f"  Steps: {counts.get('steps', 0)}  |  "
        f"Files: {counts.get('files', 0)}  |  "
        f"Branches: {counts.get('branches', 0)}  |  "
        f"Decisions: {counts.get('decisions', 0)}"
    )
    click.echo()

    # Steps timeline.
    steps = data.get("steps", [])
    if steps:
        click.secho("Timeline (Steps)", fg="blue", bold=True)
        click.secho("-" * 20, fg="blue")
        for s in steps:
            success_marker = "+" if s.get("success", True) else "x"
            time_part = _format_iso_short(s["timestamp"])
            click.echo(
                f"  [{success_marker}] Step {s['step_number']:>3d}  "
                f"{time_part}  {s['action']}"
            )
            if s.get("description"):
                click.echo(f"      {s['description'][:80]}")
            if s.get("tool_used"):
                click.echo(f"      Tool: {s['tool_used']}")
            if s.get("result_summary"):
                click.echo(f"      Result: {s['result_summary'][:80]}")
            if s.get("files_involved"):
                for fp in s["files_involved"][:5]:
                    click.echo(f"        -> {fp}")
                if len(s["files_involved"]) > 5:
                    click.echo(
                        f"        ... and {len(s['files_involved']) - 5} more"
                    )
        click.echo()

    # Files section.
    files = data.get("files", [])
    if files:
        click.secho("Files Modified", fg="magenta", bold=True)
        click.secho("-" * 20, fg="magenta")
        for f in files:
            action_count = f.get("action_count", 1)
            suffix = f" ({action_count} actions)" if action_count > 1 else ""
            click.echo(f"  [{f['action']}] {f['path']}{suffix}")
            if f.get("description"):
                click.echo(f"    {f['description'][:80]}")
        click.echo()

    # Branches section.
    branches = data.get("branches", [])
    if branches:
        click.secho("Branches", fg="yellow", bold=True)
        click.secho("-" * 20, fg="yellow")
        for b in branches:
            base = f" (from {b['base_branch']})" if b.get("base_branch") else ""
            click.echo(f"  [{b['action']}] {b['branch_name']}{base}")
        click.echo()

    # Decisions section.
    decisions = data.get("decisions", [])
    if decisions:
        click.secho("Decisions", fg="white", bold=True)
        click.secho("-" * 20, fg="white")
        for d in decisions:
            time_part = _format_iso_short(d["timestamp"])
            click.echo(
                f"  #{d['decision_number']}  {time_part}  {d['decision']}"
            )
            if d.get("reasoning"):
                click.echo(f"    Reasoning: {d['reasoning'][:100]}")
            if d.get("alternatives"):
                alts = ", ".join(d["alternatives"][:3])
                click.echo(f"    Alternatives: {alts}")
            if d.get("context"):
                click.echo(f"    Context: {d['context'][:100]}")
        click.echo()

    # Next steps.
    next_steps = data.get("next_steps", [])
    if next_steps:
        click.secho("Next Steps", fg="green", bold=True)
        click.secho("-" * 20, fg="green")
        for i, ns in enumerate(next_steps, 1):
            click.echo(f"  {i}. {ns}")
        click.echo()


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _format_duration(minutes: int) -> str:
    """Format a duration in minutes into a human-readable string.

    For durations under 60 minutes, returns "Nm".  For durations of 60
    minutes or more, returns "Nh Mm".

    Parameters
    ----------
    minutes:
        The duration in minutes.

    Returns
    -------
    str
        Human-readable duration string.
    """
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    remaining = minutes % 60
    if remaining == 0:
        return f"{hours}h"
    return f"{hours}h {remaining}m"


def _format_iso_short(iso_str: Optional[str]) -> str:
    """Format an ISO 8601 timestamp string into a shorter display form.

    Converts ``"2026-02-15T11:30:00+00:00"`` into ``"2026-02-15 11:30"``.
    Returns ``"?"`` if the input is None or unparseable.

    Parameters
    ----------
    iso_str:
        An ISO 8601 timestamp string, or None.

    Returns
    -------
    str
        Shortened display string.
    """
    if not iso_str:
        return "?"
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        # If parsing fails, try truncating the raw string.
        if len(iso_str) >= 16:
            return iso_str[:10] + " " + iso_str[11:16]
        return iso_str


def _format_bytes(size_bytes: int) -> str:
    """Format a byte count into a human-readable string.

    Uses binary units (KiB, MiB, GiB) for sizes >= 1024 bytes.

    Parameters
    ----------
    size_bytes:
        The number of bytes.

    Returns
    -------
    str
        Human-readable size string (e.g., "1.5 KiB", "3.2 MiB").
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KiB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MiB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GiB"


if __name__ == "__main__":
    cli()
