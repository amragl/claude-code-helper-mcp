"""Click CLI commands for developer-facing memory management.

Provides the ``memory`` CLI entry point with subcommands:
- ``memory status`` -- Show current task, window state, storage info, and last activity.
"""

from claude_code_helper_mcp.cli.main import cli, status

__all__ = ["cli", "status"]
