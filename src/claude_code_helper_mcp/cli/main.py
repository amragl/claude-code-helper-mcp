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
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click

from claude_code_helper_mcp import __version__

logger = logging.getLogger(__name__)


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


@cli.command()
@click.argument("ticket_id", required=False, default=None)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json", "prompt"], case_sensitive=False),
    default="text",
    help="Output format: text (default human-readable), json (machine-readable), "
    "or prompt (raw prompt text suitable for session injection).",
)
@click.option(
    "--steps",
    "recent_step_count",
    type=int,
    default=10,
    show_default=True,
    help="Number of recent steps to include in the recovery context.",
)
@click.option(
    "--no-pipeline",
    is_flag=True,
    default=False,
    help="Skip pipeline state enrichment (useful when .agent-forge/ is unavailable).",
)
@click.option(
    "--detect",
    is_flag=True,
    default=False,
    help="Only detect whether a /clear event has occurred. Outputs 'yes' or 'no'.",
)
@click.pass_context
def recover(
    ctx: click.Context,
    ticket_id: Optional[str],
    output_format: str,
    recent_step_count: int,
    no_pipeline: bool,
    detect: bool,
) -> None:
    """Recover task context after a /clear event.

    Loads the most recent task memory and generates a recovery prompt
    that restores full awareness of the task being worked on.  If
    TICKET_ID is provided, recovers that specific task; otherwise
    auto-detects the active task or the most recently completed one.

    The recovery prompt includes: ticket ID, phase, branch, files
    modified, decisions made, recent steps, planned next steps, and
    (when available) Agent Forge pipeline state.

    Use --detect to check whether a /clear event occurred without
    performing recovery.

    Examples::

        memory recover                  # Auto-detect and recover
        memory recover CMH-017          # Recover specific ticket
        memory recover --format prompt  # Raw prompt for session injection
        memory recover --detect         # Check for /clear event
        memory recover --format json    # Full JSON recovery data
    """
    storage_path = ctx.obj.get("storage_path")

    if detect:
        _run_detect(storage_path)
        return

    recover_data = _collect_recover(
        storage_path,
        ticket_id,
        recent_step_count,
        include_pipeline=not no_pipeline,
    )

    if output_format == "json":
        click.echo(json.dumps(recover_data, indent=2, default=str))
    elif output_format == "prompt":
        if recover_data.get("status") == "recovered":
            click.echo(recover_data["recovery_prompt"])
        else:
            click.secho(
                recover_data.get("message", "No recovery context available."),
                fg="red",
                err=True,
            )
            sys.exit(1)
    else:
        _render_recover_text(recover_data)


@cli.command()
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    default=False,
    help="Output analysis as JSON instead of human-readable text.",
)
@click.option(
    "--since",
    type=click.DateTime(formats=["%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"]),
    default=None,
    help="Only analyze tasks started on or after this date (format: YYYY-MM-DD or ISO8601).",
)
@click.pass_context
def analytics(ctx: click.Context, output_json: bool, since: Optional[datetime]) -> None:
    """Analyze patterns across all tasks in memory.

    Generates insights about task execution patterns:
    - Average steps and time per ticket
    - Frequently modified files
    - Common error patterns
    - Decision patterns and reasoning trends

    Use --json for machine-readable output.  Use --since to filter by
    task start date (e.g., --since 2026-02-15).

    Examples::

        memory analytics
        memory analytics --json
        memory analytics --since 2026-02-15
        memory analytics --json --since 2026-02-14T10:00:00
    """
    from claude_code_helper_mcp.analytics import MemoryAnalytics

    storage_path = ctx.obj.get("storage_path")

    try:
        analyzer = MemoryAnalytics(storage_path)
        analyzer.analyze(since=since)
        analysis_data = analyzer.to_json_dict()

        if output_json:
            click.echo(json.dumps(analysis_data, indent=2))
        else:
            _render_analytics_text(analysis_data)

    except Exception as exc:
        click.secho(
            f"Analysis failed: {exc}",
            fg="red",
            err=True,
        )
        logger.exception("Analytics error")
        sys.exit(1)


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
# Recover data collection
# ---------------------------------------------------------------------------


def _run_detect(storage_path: Optional[str]) -> None:
    """Run clear event detection and output the result.

    Parameters
    ----------
    storage_path:
        Explicit storage path, or None for auto-detection.
    """
    from claude_code_helper_mcp.hooks.recovery import RecoveryWorkflow

    try:
        kwargs: dict = {}
        if storage_path:
            kwargs["storage_path"] = storage_path
        workflow = RecoveryWorkflow(**kwargs)
        detected = workflow.detect_clear_event()
    except Exception as exc:
        click.secho(f"ERROR: {exc}", fg="red", err=True)
        sys.exit(1)

    if detected:
        click.secho("yes", fg="yellow")
        click.echo(
            "A /clear event was detected. Run 'memory recover' to restore context.",
            err=True,
        )
    else:
        click.secho("no", fg="green")


def _collect_recover(
    storage_path: Optional[str],
    ticket_id: Optional[str],
    recent_step_count: int,
    include_pipeline: bool,
) -> dict:
    """Collect recovery data using the RecoveryWorkflow.

    Parameters
    ----------
    storage_path:
        Explicit storage path, or None for auto-detection.
    ticket_id:
        Explicit ticket ID, or None for auto-detection.
    recent_step_count:
        Number of recent steps to include.
    include_pipeline:
        Whether to include pipeline state enrichment.

    Returns
    -------
    dict
        Recovery result from RecoveryWorkflow.recover().
    """
    from claude_code_helper_mcp.hooks.recovery import RecoveryWorkflow

    try:
        kwargs: dict = {}
        if storage_path:
            kwargs["storage_path"] = storage_path
        workflow = RecoveryWorkflow(**kwargs)
        return workflow.recover(
            ticket_id=ticket_id,
            recent_step_count=recent_step_count,
            include_pipeline_context=include_pipeline,
        )
    except Exception as exc:
        return {
            "status": "error",
            "error": f"Recovery failed: {exc}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


def _render_recover_text(data: dict) -> None:
    """Render recovery data as formatted text.

    Parameters
    ----------
    data:
        Recovery result from _collect_recover().
    """
    if data.get("status") == "error":
        click.secho(
            "ERROR: " + data.get("error", "Unknown error"), fg="red", err=True
        )
        sys.exit(1)

    if data.get("status") == "no_context":
        click.secho(
            "No context available for recovery.", fg="yellow", err=True
        )
        message = data.get("message", "")
        if message:
            click.echo(f"  {message}", err=True)
        available = data.get("available_tasks", [])
        if available:
            click.echo(f"  Available tasks: {', '.join(available)}", err=True)
        sys.exit(1)

    # Recovered successfully.
    click.secho(
        "Claude Code Helper -- Recovery Context", fg="cyan", bold=True
    )
    click.secho("=" * 50, fg="cyan")
    click.echo()

    # Overview.
    click.secho("Task", fg="green", bold=True)
    click.secho("-" * 20, fg="green")
    click.echo(f"  Ticket:  {data.get('ticket_id', '?')}")
    click.echo(f"  Title:   {data.get('title', '?')}")
    click.echo(f"  Source:  {data.get('source', '?')}")

    git_branch = data.get("git_branch")
    if git_branch:
        click.echo(f"  Branch:  {git_branch}")
    click.echo()

    # Pipeline context.
    pipeline_ctx = data.get("pipeline_context")
    if pipeline_ctx:
        click.secho("Pipeline State", fg="blue", bold=True)
        click.secho("-" * 20, fg="blue")
        click.echo(
            f"  Status:          {pipeline_ctx.get('pipeline_status', '?')}"
        )
        step = pipeline_ctx.get("pipeline_step")
        if step:
            click.echo(f"  Current step:    {step}")
        last = pipeline_ctx.get("last_completed_step")
        if last:
            click.echo(f"  Last completed:  {last}")
        completed = pipeline_ctx.get("steps_completed", [])
        if completed:
            click.echo(f"  Steps done:      {', '.join(completed)}")
        remaining = pipeline_ctx.get("steps_remaining", [])
        if remaining:
            click.echo(f"  Steps remaining: {', '.join(remaining)}")
        pr = pipeline_ctx.get("pr_number")
        if pr:
            click.echo(f"  PR:              #{pr}")
        blocked = pipeline_ctx.get("blocked_reason")
        if blocked:
            click.secho(f"  BLOCKED:         {blocked}", fg="red")
        click.echo()

    # Recovery context summary.
    rc = data.get("recovery_context", {})
    files = rc.get("files_modified", [])
    if files:
        click.secho("Files Modified", fg="magenta", bold=True)
        click.secho("-" * 20, fg="magenta")
        for f in files[:20]:
            click.echo(f"  {f}")
        if len(files) > 20:
            click.echo(f"  ... and {len(files) - 20} more")
        click.echo()

    decisions = rc.get("key_decisions", [])
    if decisions:
        click.secho("Key Decisions", fg="white", bold=True)
        click.secho("-" * 20, fg="white")
        for d in decisions[:10]:
            click.echo(f"  - {d.get('decision', 'N/A')}")
        click.echo()

    steps = rc.get("recent_steps", [])
    if steps:
        click.secho("Recent Steps", fg="blue", bold=True)
        click.secho("-" * 20, fg="blue")
        for s in steps[:10]:
            num = s.get("step_number", "?")
            action = s.get("action", "N/A")
            success = s.get("success", True)
            marker = "+" if success else "x"
            click.echo(f"  [{marker}] Step {num}: {action}")
        click.echo()

    next_steps = rc.get("next_steps", [])
    if next_steps:
        click.secho("Planned Next Steps", fg="green", bold=True)
        click.secho("-" * 20, fg="green")
        for i, ns in enumerate(next_steps, 1):
            click.echo(f"  {i}. {ns}")
        click.echo()

    # Footer.
    click.secho(
        "Recovery prompt generated. Use --format prompt to get the raw "
        "prompt text.",
        fg="cyan",
    )


def _render_analytics_text(analysis_data: dict) -> None:
    """Render analysis results as human-readable text.

    Parameters
    ----------
    analysis_data:
        The JSON analysis output from MemoryAnalytics.to_json_dict().
    """
    summary = analysis_data.get("summary", {})

    click.secho("\nMemory Analytics Report", fg="cyan", bold=True)
    click.secho("=" * 50, fg="cyan")

    # Summary section
    click.echo()
    click.secho("Summary", fg="green", bold=True)
    click.secho("-" * 50, fg="green")

    total_tasks = summary.get("total_tasks_analyzed", 0)
    avg_steps = summary.get("avg_steps_per_ticket", 0)
    avg_time = summary.get("avg_time_per_ticket_seconds", 0)
    total_files = summary.get("total_files_modified", 0)
    total_decisions = summary.get("total_decisions", 0)

    click.echo(f"  Total tasks analyzed:        {total_tasks}")
    click.echo(f"  Avg steps per ticket:        {avg_steps:.1f}")
    click.echo(f"  Avg time per ticket:         {_format_seconds(avg_time)}")
    click.echo(f"  Total files modified:        {total_files}")
    click.echo(f"  Total decisions recorded:    {total_decisions}")

    # Status breakdown
    status_breakdown = summary.get("status_breakdown", {})
    if status_breakdown:
        click.echo()
        click.secho("Task Status Breakdown", fg="green", bold=True)
        for status, count in status_breakdown.items():
            click.echo(f"  {status.capitalize()}: {count}")

    # Top files
    top_files = summary.get("top_files", [])
    if top_files:
        click.echo()
        click.secho("Most Modified Files", fg="green", bold=True)
        click.secho("-" * 50, fg="green")
        for i, file_info in enumerate(top_files[:5], 1):
            path = file_info.get("path", "?")
            count = file_info.get("modification_count", 0)
            click.echo(f"  {i}. {path} ({count} times)")

    # Decision types
    decision_types = analysis_data.get("decision_types", {})
    if decision_types:
        click.echo()
        click.secho("Top Decision Types", fg="green", bold=True)
        click.secho("-" * 50, fg="green")
        for i, (dec_type, count) in enumerate(list(decision_types.items())[:5], 1):
            click.echo(f"  {i}. {dec_type}: {count} decisions")

    # Error patterns
    error_patterns = analysis_data.get("error_patterns", {})
    if error_patterns:
        click.echo()
        click.secho("Top Error Patterns", fg="yellow", bold=True)
        click.secho("-" * 50, fg="yellow")
        for i, (error, count) in enumerate(list(error_patterns.items())[:5], 1):
            error_short = error[:60] + ("..." if len(error) > 60 else "")
            click.echo(f"  {i}. {error_short} ({count} times)")

    click.echo()
    analyzed_at = analysis_data.get("analyzed_at", "?")
    click.secho(f"Analyzed at: {analyzed_at}", fg="cyan")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _format_seconds(seconds: float) -> str:
    """Format seconds into a human-readable duration string.

    Parameters
    ----------
    seconds:
        The duration in seconds.

    Returns
    -------
    str
        Human-readable duration (e.g., "1h 30m 45s", "45s").
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        remaining = seconds % 3600
        minutes = int(remaining // 60)
        secs = int(remaining % 60)
        if secs == 0:
            return f"{hours}h {minutes}m"
        return f"{hours}h {minutes}m {secs}s"


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


@cli.command()
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    default=False,
    help="Output dashboard as JSON instead of human-readable text.",
)
@click.option(
    "--section",
    type=click.Choice(["timeline", "decisions", "heatmap", "interventions", "window", "all"], case_sensitive=False),
    default="all",
    help="Display only a specific dashboard section.",
)
@click.pass_context
def dashboard(ctx: click.Context, output_json: bool, section: str) -> None:
    """Display comprehensive memory dashboard for developer review.

    Shows a complete overview of memory system state including:
    - Task timeline with chronological task status and metrics
    - Decision tree visualization showing all decisions made
    - File modification heat map (most-touched files)
    - Intervention summary (detected drift, errors, confusion, scope creep)
    - Window state (current sliding window status and occupancy)

    Use --json for machine-readable output. Use --section to view a specific
    section only (timeline, decisions, heatmap, interventions, window).

    Examples::

        memory dashboard
        memory dashboard --json
        memory dashboard --section timeline
        memory dashboard --section heatmap --json
    """
    from claude_code_helper_mcp.analytics.dashboard import DeveloperDashboard

    storage_path = ctx.obj.get("storage_path")

    try:
        dashboard_obj = DeveloperDashboard(storage_path)
        dashboard_obj.generate()
        dashboard_data = dashboard_obj.to_json_dict()

        if output_json:
            # Filter by section if requested
            if section != "all":
                filtered_data = {
                    "generated_at": dashboard_data["generated_at"],
                    section: dashboard_data.get(section, {}),
                }
                click.echo(json.dumps(filtered_data, indent=2))
            else:
                click.echo(json.dumps(dashboard_data, indent=2))
        else:
            _render_dashboard_text(dashboard_data, section)

    except Exception as exc:
        click.secho(
            f"ERROR: Failed to generate dashboard: {exc}",
            fg="red",
            err=True,
        )
        logger.exception("Dashboard generation failed")
        sys.exit(1)


def _render_dashboard_text(dashboard_data: dict, section: str) -> None:
    """Render dashboard data as human-readable text.

    Parameters
    ----------
    dashboard_data:
        Complete dashboard dictionary from DeveloperDashboard.to_json_dict().
    section:
        Which section to render: all, timeline, decisions, heatmap, interventions, window.
    """
    click.secho("\n", nl=False)
    click.secho("=" * 80, fg="cyan", bold=True)
    click.secho("CLAUDE CODE HELPER - MEMORY DASHBOARD", fg="cyan", bold=True)
    click.secho(f"Generated: {dashboard_data['generated_at']}", fg="cyan")
    click.secho("=" * 80, fg="cyan", bold=True)
    click.echo()

    summary = dashboard_data.get("summary", {})
    click.secho("SUMMARY", fg="yellow", bold=True)
    click.secho("-" * 40, fg="yellow")
    click.echo(f"  Total Tasks: {summary.get('total_tasks', 0)}")
    click.echo(f"  Total Decisions: {summary.get('total_decisions', 0)}")
    click.echo(f"  Files Tracked: {summary.get('files_tracked', 0)}")
    click.echo(f"  Detections: {summary.get('detections_count', 0)}")
    click.echo()

    # Timeline section
    if section in ("all", "timeline"):
        _render_timeline_section(dashboard_data.get("timeline", []))

    # Decision tree section
    if section in ("all", "decisions"):
        _render_decision_section(dashboard_data.get("decision_tree", []))

    # Heat map section
    if section in ("all", "heatmap"):
        _render_heatmap_section(dashboard_data.get("file_heat_map", []))

    # Interventions section
    if section in ("all", "interventions"):
        _render_interventions_section(dashboard_data.get("interventions", {}))

    # Window state section
    if section in ("all", "window"):
        _render_window_section(dashboard_data.get("window_state", {}))

    click.echo()
    click.secho("=" * 80, fg="cyan")


def _render_timeline_section(timeline: list[dict]) -> None:
    """Render task timeline section."""
    click.secho("TASK TIMELINE", fg="green", bold=True)
    click.secho("-" * 80, fg="green")

    if not timeline:
        click.echo("  No tasks in timeline.")
        click.echo()
        return

    # Calculate column widths
    max_ticket_len = max(len(t["ticket_id"]) for t in timeline) if timeline else 10
    max_title_len = max(len(t["title"]) for t in timeline) if timeline else 30
    max_ticket_len = max(max_ticket_len, 10)
    max_title_len = min(max_title_len, 40)

    # Header
    ticket_col = "TICKET".ljust(max_ticket_len)
    title_col = "TITLE".ljust(max_title_len)
    click.echo(f"  {ticket_col} {title_col} STATUS      STEPS DURATION")
    click.echo(f"  {'-' * max_ticket_len} {'-' * max_title_len} {'-' * 10} {'-' * 5} {'-' * 10}")

    # Rows
    for entry in timeline:
        ticket = entry["ticket_id"].ljust(max_ticket_len)
        title = entry["title"][:max_title_len].ljust(max_title_len)
        status = entry["status"].ljust(10)
        steps = str(entry["step_count"]).ljust(5)
        duration_sec = entry["duration_seconds"]
        if duration_sec < 60:
            duration_str = f"{int(duration_sec)}s"
        elif duration_sec < 3600:
            duration_str = f"{int(duration_sec / 60)}m"
        else:
            duration_str = f"{int(duration_sec / 3600)}h"
        duration = duration_str.ljust(10)

        click.echo(f"  {ticket} {title} {status} {steps} {duration}")

    click.echo()


def _render_decision_section(decision_tree: list[dict]) -> None:
    """Render decision tree section."""
    click.secho("DECISION TREE", fg="blue", bold=True)
    click.secho("-" * 80, fg="blue")

    if not decision_tree:
        click.echo("  No decisions recorded.")
        click.echo()
        return

    click.echo(f"  Total Decisions: {len(decision_tree)}\n")

    # Show first 10 decisions as samples
    for i, decision in enumerate(decision_tree[:10], 1):
        click.secho(f"  [{i}] {decision['task_id']}", fg="blue")
        click.echo(f"      Decision: {decision['decision']}")
        click.echo(f"      Reasoning: {decision['reasoning']}")
        if decision["alternatives"]:
            click.echo(f"      Alternatives: {', '.join(decision['alternatives'])}")
        click.echo()

    if len(decision_tree) > 10:
        click.echo(f"  ... and {len(decision_tree) - 10} more decisions")
        click.echo()


def _render_heatmap_section(heat_map: list[dict]) -> None:
    """Render file modification heat map section."""
    click.secho("FILE MODIFICATION HEAT MAP", fg="red", bold=True)
    click.secho("-" * 80, fg="red")

    if not heat_map:
        click.echo("  No files modified.")
        click.echo()
        return

    # Show top 15 files
    for entry in heat_map[:15]:
        heat_color = "red" if entry["heat_score"] == "hot" else (
            "yellow" if entry["heat_score"] == "warm" else (
                "cyan" if entry["heat_score"] == "cool" else "white"
            )
        )
        heat_icon = "🔥" if entry["heat_score"] == "hot" else (
            "💥" if entry["heat_score"] == "warm" else (
                "❄️ " if entry["heat_score"] == "cool" else "·"
            )
        )
        heat_score = entry["heat_score"].upper().ljust(6)
        count = str(entry["modification_count"]).ljust(3)
        click.secho(f"  {heat_icon} {heat_score} ({count}x) {entry['file_path']}", fg=heat_color)

    if len(heat_map) > 15:
        click.echo(f"  ... and {len(heat_map) - 15} more files")

    click.echo()


def _render_interventions_section(interventions: dict) -> None:
    """Render intervention summary section."""
    click.secho("INTERVENTION SUMMARY", fg="magenta", bold=True)
    click.secho("-" * 80, fg="magenta")

    total = interventions.get("total_detections", 0)

    if total == 0:
        click.echo("  No interventions detected. System is healthy.")
        click.echo()
        return

    click.echo(f"  Total Detections: {total}\n")

    drift = interventions.get("drift_detections", [])
    if drift:
        click.secho(f"  Drift Detections ({len(drift)}):", fg="magenta", bold=True)
        for d in drift[:5]:
            click.echo(f"    • {d['ticket_id']}: {d['severity']} - {d['details']}")
        if len(drift) > 5:
            click.echo(f"    ... and {len(drift) - 5} more")
        click.echo()

    errors = interventions.get("error_loop_detections", [])
    if errors:
        click.secho(f"  Error Loop Detections ({len(errors)}):", fg="magenta", bold=True)
        for e in errors[:5]:
            click.echo(f"    • {e['ticket_id']}: {e['action']} ({e['consecutive_failures']} failures)")
        if len(errors) > 5:
            click.echo(f"    ... and {len(errors) - 5} more")
        click.echo()

    confusion = interventions.get("confusion_detections", [])
    if confusion:
        click.secho(f"  Confusion Detections ({len(confusion)}):", fg="magenta", bold=True)
        for c in confusion[:5]:
            click.echo(f"    • {c['ticket_id']}: {c['confusion_type']} - {c['details']}")
        if len(confusion) > 5:
            click.echo(f"    ... and {len(confusion) - 5} more")
        click.echo()

    scope = interventions.get("scope_creep_detections", [])
    if scope:
        click.secho(f"  Scope Creep Detections ({len(scope)}):", fg="magenta", bold=True)
        for s in scope[:5]:
            click.echo(f"    • {s['ticket_id']}: {s['file_path']} - {s['reason']}")
        if len(scope) > 5:
            click.echo(f"    ... and {len(scope) - 5} more")
        click.echo()


def _render_window_section(window: dict) -> None:
    """Render window state section."""
    click.secho("WINDOW STATE", fg="white", bold=True)
    click.secho("-" * 80, fg="white")

    if not window:
        click.echo("  Window state unavailable.")
        click.echo()
        return

    click.echo(f"  Window Size: {window.get('window_size', 'unknown')}")
    click.echo(f"  Occupancy: {window.get('window_occupancy', 'unknown')}")
    click.echo(f"  Active Task: {window.get('active_task', 'none')}")

    completed = window.get("completed_tasks_in_window", [])
    if completed:
        click.echo(f"  Completed Tasks in Window: {len(completed)}")
        for task_id in completed:
            click.echo(f"    • {task_id}")
    else:
        click.echo("  Completed Tasks in Window: none")

    click.echo()


@cli.command()
@click.argument("output_path", type=click.Path())
@click.option(
    "--all",
    "export_all",
    is_flag=True,
    default=True,
    help="Export all tasks (default). Use without flag to export only window tasks.",
)
@click.pass_context
def export(ctx: click.Context, output_path: str, export_all: bool) -> None:
    """Export memory to portable JSON format.

    Exports all tasks (current, completed, and archived) to a portable JSON file
    with format version for forward compatibility. The exported JSON includes
    metadata about the export and a complete copy of all task data.

    This is useful for:
    - Backing up memory across sessions
    - Sharing task context across projects
    - Archiving completed work

    Example::

        memory export ./memory-backup.json
        memory export /path/to/memory-export.json
    """
    from claude_code_helper_mcp.storage.export_import import ExportManager

    storage_path = ctx.obj.get("storage_path")

    try:
        exporter = ExportManager(storage_path)
        result = exporter.export_all(output_path)

        if result.get("status") == "success":
            click.secho(
                f"Export successful: {result['exported_count']} tasks exported",
                fg="green",
            )
            click.echo(f"File: {result['file_path']}")
            click.echo(f"Format version: {result['format_version']}")
            click.echo(f"Timestamp: {result['timestamp']}")
        else:
            click.secho(f"Export failed: {result.get('error')}", fg="red", err=True)
            sys.exit(1)

    except Exception as exc:
        click.secho(f"Export error: {exc}", fg="red", err=True)
        logger.exception("Export command failed")
        sys.exit(1)


@cli.command("import")
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--mode",
    type=click.Choice(["merge", "replace"], case_sensitive=False),
    default="merge",
    help="Import mode: 'merge' adds imported tasks alongside existing, "
    "'replace' overwrites matching tasks. Default: merge.",
)
@click.option(
    "--allow-version-mismatch",
    is_flag=True,
    default=False,
    help="Allow importing from files with different format versions.",
)
@click.option(
    "--validate",
    "only_validate",
    is_flag=True,
    default=False,
    help="Only validate the import file without importing.",
)
@click.option(
    "--json-output",
    "output_json",
    is_flag=True,
    default=False,
    help="Output result as JSON instead of human-readable text.",
)
@click.pass_context
def import_(ctx: click.Context, input_path: str, mode: str, allow_version_mismatch: bool, only_validate: bool, output_json: bool) -> None:
    """Import memory from portable JSON format.

    Imports tasks from a portable JSON file (exported via 'memory export')
    with validation and compatibility checks.

    Import modes:
    - merge (default): Add imported tasks alongside existing ones
    - replace: Overwrite any existing tasks with the same ticket ID

    Use --validate to check if an import file is valid without importing.

    Examples::

        memory import ./memory-backup.json
        memory import ./memory-backup.json --mode replace
        memory import ./memory-export.json --validate
        memory import ./export.json --json-output
    """
    from claude_code_helper_mcp.storage.export_import import ImportManager

    storage_path = ctx.obj.get("storage_path")

    try:
        importer = ImportManager(storage_path)

        if only_validate:
            result = importer.validate_import_file(input_path)

            if output_json:
                click.echo(json.dumps(result, indent=2))
            else:
                if result.get("status") == "valid":
                    click.secho("Validation passed", fg="green")
                    click.echo(f"Format version: {result.get('format_version')}")
                    click.echo(f"Tasks in file: {result.get('task_count')}")
                else:
                    click.secho("Validation failed", fg="red")
                    errors = result.get("validation_errors", [])
                    if errors:
                        click.echo("Errors:")
                        for error in errors:
                            click.echo(f"  - {error}")
            return

        # Perform import
        result = importer.import_from_file(
            file_path=input_path,
            mode=mode,
            validate_compatibility=True,
            allow_version_mismatch=allow_version_mismatch,
        )

        if output_json:
            click.echo(json.dumps(result, indent=2))
        else:
            if result.get("status") == "success":
                click.secho("Import successful", fg="green")
                click.echo(f"Imported: {result['imported_count']} tasks")
                if result.get("skipped_count", 0) > 0:
                    click.echo(f"Skipped: {result['skipped_count']} tasks")
                click.echo(f"Format version: {result.get('format_version')}")
                click.echo(f"Mode: {mode}")
            elif result.get("status") == "compatibility_error":
                click.secho("Compatibility error", fg="red")
                click.echo(f"Error: {result.get('error')}")
                click.echo(
                    "Use --allow-version-mismatch to force import from different version.",
                    err=True,
                )
                sys.exit(1)
            elif result.get("status") == "validation_error":
                click.secho("Validation error", fg="red")
                click.echo(f"Error: {result.get('error')}")
                errors = result.get("validation_errors", [])
                if errors:
                    click.echo("Details:")
                    for error in errors:
                        click.echo(f"  - {error}")
                sys.exit(1)
            else:
                click.secho(f"Import failed: {result.get('error')}", fg="red", err=True)
                sys.exit(1)

    except Exception as exc:
        click.secho(f"Import error: {exc}", fg="red", err=True)
        logger.exception("Import command failed")
        sys.exit(1)


if __name__ == "__main__":
    cli()
