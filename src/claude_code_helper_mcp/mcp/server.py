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

    Future tickets will add:
    - record_step, record_decision (CMH-007)
    - record_file, record_branch (CMH-008)
    - start_task, complete_task, get_task_status (CMH-009)
    - generate_summary (CMH-010)
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
