"""Click CLI commands for developer-facing memory management.

Provides the ``memory`` CLI entry point with subcommands:
- ``memory status``  -- Show current task, window state, storage info, and last activity.
- ``memory list``    -- List tasks in the sliding window (with --all and --format options).
- ``memory show``    -- Show full details for a specific task by ticket ID.
- ``memory recover`` -- Recover task context after a /clear event.
"""

from claude_code_helper_mcp.cli.main import cli, list_tasks, recover, show, status

__all__ = ["cli", "list_tasks", "recover", "show", "status"]
