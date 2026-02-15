"""Main Click CLI entry point for the memory command.

Provides the ``memory`` CLI group and the ``status`` subcommand.
The CLI exposes developer-facing commands for inspecting and managing
the structured memory system.

Entry point registered in pyproject.toml::

    [project.scripts]
    memory = "claude_code_helper_mcp.cli.main:cli"

Usage examples::

    memory --version
    memory status
    memory status --json
    memory status --storage-path /path/to/.claude-memory
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
# Utilities
# ---------------------------------------------------------------------------


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
