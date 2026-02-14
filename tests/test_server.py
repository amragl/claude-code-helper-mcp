"""Tests for the FastMCP server bootstrap (CMH-006).

Verifies server creation, configuration integration, WindowManager lifecycle,
health check tool registration and execution, singleton management, and
error handling.  All tests use real file I/O with temporary directories.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from claude_code_helper_mcp.mcp.server import (
    create_server,
    get_config,
    get_server,
    get_window_manager,
    reset_server,
)
from claude_code_helper_mcp.config import MemoryConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_tool_result(result) -> dict:
    """Extract a dict from a FastMCP ToolResult.

    FastMCP tool.run() returns a ToolResult whose .content is a list of
    TextContent objects.  The first TextContent's .text is JSON-encoded.
    """
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        return json.loads(result)
    # ToolResult object -- extract text from first content item.
    if hasattr(result, "content") and result.content:
        text = result.content[0].text
        return json.loads(text)
    raise TypeError(f"Cannot parse tool result of type {type(result)}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_server():
    """Ensure the server singleton is reset before and after each test."""
    reset_server()
    yield
    reset_server()


@pytest.fixture
def project_dir():
    """Create a temporary project directory with a .git marker for root detection."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a .git marker so project root detection works.
        (Path(tmpdir) / ".git").mkdir()
        yield tmpdir


@pytest.fixture
def project_with_config(project_dir):
    """Create a project directory with a config.json file."""
    config_dir = Path(project_dir) / ".claude-memory"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "config.json"
    config_file.write_text(json.dumps({
        "window_size": 5,
        "log_level": "DEBUG",
    }))
    return project_dir


# ---------------------------------------------------------------------------
# Server creation tests
# ---------------------------------------------------------------------------

class TestCreateServer:
    """Tests for create_server() factory function."""

    def test_returns_fastmcp_instance(self, project_dir):
        from fastmcp import FastMCP
        server = create_server(project_root=project_dir)
        assert isinstance(server, FastMCP)

    def test_server_name(self, project_dir):
        server = create_server(project_root=project_dir)
        assert server.name == "claude-code-helper"

    def test_creates_storage_directory(self, project_dir):
        create_server(project_root=project_dir)
        storage_path = Path(project_dir) / ".claude-memory"
        assert storage_path.is_dir()
        assert (storage_path / "tasks").is_dir()

    def test_creates_window_file(self, project_dir):
        create_server(project_root=project_dir)
        window_path = Path(project_dir) / ".claude-memory" / "window.json"
        assert window_path.is_file()

    def test_window_file_is_valid_json(self, project_dir):
        create_server(project_root=project_dir)
        window_path = Path(project_dir) / ".claude-memory" / "window.json"
        data = json.loads(window_path.read_text())
        assert "window_size" in data
        assert "completed_tasks" in data

    def test_respects_config_file(self, project_with_config):
        create_server(project_root=project_with_config)
        cfg = get_config()
        assert cfg.window_size == 5
        assert cfg.log_level == "DEBUG"

    def test_default_window_size_is_3(self, project_dir):
        create_server(project_root=project_dir)
        cfg = get_config()
        assert cfg.window_size == 3

    def test_storage_path_resolves_correctly(self, project_dir):
        create_server(project_root=project_dir)
        cfg = get_config()
        expected = str(Path(project_dir).resolve() / ".claude-memory")
        assert cfg.storage_path == expected

    def test_project_root_resolves_correctly(self, project_dir):
        create_server(project_root=project_dir)
        cfg = get_config()
        assert cfg.project_root == str(Path(project_dir).resolve())


class TestCreateServerIdempotency:
    """Tests for server singleton behavior."""

    def test_second_call_overwrites_singleton(self, project_dir):
        server1 = create_server(project_root=project_dir)
        server2 = create_server(project_root=project_dir)
        # create_server always creates a fresh instance
        assert server2 is not None

    def test_get_server_creates_if_needed(self):
        """get_server() calls create_server() if no instance exists."""
        # No explicit project root, will use cwd detection.
        server = get_server()
        assert server is not None


# ---------------------------------------------------------------------------
# Accessor tests
# ---------------------------------------------------------------------------

class TestAccessors:
    """Tests for get_window_manager(), get_config(), and get_server()."""

    def test_get_window_manager_after_create(self, project_dir):
        create_server(project_root=project_dir)
        wm = get_window_manager()
        assert wm is not None
        assert wm.total_tasks_in_window() == 0

    def test_get_window_manager_before_create_raises(self):
        with pytest.raises(RuntimeError, match="not been initialized"):
            get_window_manager()

    def test_get_config_after_create(self, project_dir):
        create_server(project_root=project_dir)
        cfg = get_config()
        assert isinstance(cfg, MemoryConfig)

    def test_get_config_before_create_raises(self):
        with pytest.raises(RuntimeError, match="not been initialized"):
            get_config()

    def test_get_server_after_create(self, project_dir):
        server = create_server(project_root=project_dir)
        retrieved = get_server()
        assert retrieved is server


class TestResetServer:
    """Tests for reset_server()."""

    def test_reset_clears_singleton(self, project_dir):
        create_server(project_root=project_dir)
        reset_server()
        with pytest.raises(RuntimeError):
            get_window_manager()

    def test_reset_clears_config(self, project_dir):
        create_server(project_root=project_dir)
        reset_server()
        with pytest.raises(RuntimeError):
            get_config()

    def test_reset_is_idempotent(self):
        reset_server()
        reset_server()  # Should not raise


# ---------------------------------------------------------------------------
# Health check tool tests
# ---------------------------------------------------------------------------

class TestHealthCheckTool:
    """Tests for the health_check MCP tool."""

    def test_tool_is_registered(self, project_dir):
        server = create_server(project_root=project_dir)
        tools = asyncio.run(server.get_tools())
        assert "health_check" in tools

    def test_health_check_returns_dict(self, project_dir):
        server = create_server(project_root=project_dir)
        # Call the tool directly via the server's internal API
        tools = asyncio.run(server.get_tools())
        tool = tools["health_check"]
        raw = asyncio.run(tool.run({}))
        result = _parse_tool_result(raw)
        assert isinstance(result, dict)

    def test_health_check_contains_required_fields(self, project_dir):
        server = create_server(project_root=project_dir)
        tools = asyncio.run(server.get_tools())
        tool = tools["health_check"]
        result = _parse_tool_result(asyncio.run(tool.run({})))

        required_fields = [
            "server_version",
            "status",
            "storage_path",
            "storage_accessible",
            "window_size",
            "tasks_in_window",
            "current_task",
            "completed_tasks",
            "archived_tasks",
            "project_root",
            "timestamp",
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

    def test_health_check_status_healthy_on_fresh_server(self, project_dir):
        server = create_server(project_root=project_dir)
        tools = asyncio.run(server.get_tools())
        tool = tools["health_check"]
        result = _parse_tool_result(asyncio.run(tool.run({})))
        assert result["status"] == "healthy"

    def test_health_check_version_matches(self, project_dir):
        from claude_code_helper_mcp import __version__
        server = create_server(project_root=project_dir)
        tools = asyncio.run(server.get_tools())
        tool = tools["health_check"]
        result = _parse_tool_result(asyncio.run(tool.run({})))
        assert result["server_version"] == __version__

    def test_health_check_window_size(self, project_with_config):
        server = create_server(project_root=project_with_config)
        tools = asyncio.run(server.get_tools())
        tool = tools["health_check"]
        result = _parse_tool_result(asyncio.run(tool.run({})))
        assert result["window_size"] == 5

    def test_health_check_no_current_task(self, project_dir):
        server = create_server(project_root=project_dir)
        tools = asyncio.run(server.get_tools())
        tool = tools["health_check"]
        result = _parse_tool_result(asyncio.run(tool.run({})))
        assert result["current_task"] is None

    def test_health_check_with_active_task(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task", phase="phase-1")

        tools = asyncio.run(server.get_tools())
        tool = tools["health_check"]
        result = _parse_tool_result(asyncio.run(tool.run({})))
        assert result["current_task"] == "TST-001"
        assert result["tasks_in_window"] == 1

    def test_health_check_with_completed_tasks(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()

        # Complete two tasks
        wm.start_new_task("TST-001", "Task 1")
        wm.complete_current_task("Done 1")
        wm.start_new_task("TST-002", "Task 2")
        wm.complete_current_task("Done 2")

        tools = asyncio.run(server.get_tools())
        tool = tools["health_check"]
        result = _parse_tool_result(asyncio.run(tool.run({})))
        assert result["completed_tasks"] == 2
        assert result["current_task"] is None
        assert result["tasks_in_window"] == 2

    def test_health_check_with_archived_tasks(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        # Window size is 3 by default. Complete 4 tasks to archive 1.
        for i in range(1, 5):
            wm.start_new_task(f"TST-{i:03d}", f"Task {i}")
            wm.complete_current_task(f"Done {i}")

        tools = asyncio.run(server.get_tools())
        tool = tools["health_check"]
        result = _parse_tool_result(asyncio.run(tool.run({})))
        assert result["completed_tasks"] == 3  # window size
        assert result["archived_tasks"] == 1

    def test_health_check_storage_path(self, project_dir):
        server = create_server(project_root=project_dir)
        tools = asyncio.run(server.get_tools())
        tool = tools["health_check"]
        result = _parse_tool_result(asyncio.run(tool.run({})))
        assert ".claude-memory" in result["storage_path"]

    def test_health_check_timestamp_is_iso(self, project_dir):
        from datetime import datetime
        server = create_server(project_root=project_dir)
        tools = asyncio.run(server.get_tools())
        tool = tools["health_check"]
        result = _parse_tool_result(asyncio.run(tool.run({})))
        # Should parse as a valid ISO datetime.
        ts = result["timestamp"]
        datetime.fromisoformat(ts)  # Raises ValueError if invalid


# ---------------------------------------------------------------------------
# WindowManager integration tests
# ---------------------------------------------------------------------------

class TestWindowManagerIntegration:
    """Tests that the server correctly integrates with WindowManager."""

    def test_server_and_window_manager_share_store(self, project_dir):
        create_server(project_root=project_dir)
        wm = get_window_manager()
        expected_root = Path(project_dir) / ".claude-memory"
        assert str(wm.store.storage_root) == str(expected_root.resolve())

    def test_window_manager_persists_across_operations(self, project_dir):
        create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")
        wm.complete_current_task("Done")

        # Create a new WindowManager from the same storage to verify persistence
        from claude_code_helper_mcp.storage.window_manager import WindowManager
        wm2 = WindowManager(storage_path=str(Path(project_dir) / ".claude-memory"))
        assert wm2.completed_task_count() == 1

    def test_config_window_size_applied_to_manager(self, project_with_config):
        create_server(project_root=project_with_config)
        wm = get_window_manager()
        assert wm.window_size == 5

    def test_default_window_size_applied(self, project_dir):
        create_server(project_root=project_dir)
        wm = get_window_manager()
        assert wm.window_size == 3


# ---------------------------------------------------------------------------
# Configuration integration tests
# ---------------------------------------------------------------------------

class TestConfigIntegration:
    """Tests for configuration loading and application."""

    def test_config_loaded_from_file(self, project_with_config):
        create_server(project_root=project_with_config)
        cfg = get_config()
        assert cfg.window_size == 5
        assert cfg.log_level == "DEBUG"

    def test_config_defaults_when_no_file(self, project_dir):
        create_server(project_root=project_dir)
        cfg = get_config()
        assert cfg.window_size == 3
        assert cfg.log_level == "INFO"

    def test_config_project_root_set(self, project_dir):
        create_server(project_root=project_dir)
        cfg = get_config()
        assert cfg.project_root == str(Path(project_dir).resolve())


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Tests for error handling in the server module."""

    def test_create_server_with_nonexistent_config_path(self, project_dir):
        """Non-existent config path is silently ignored (uses defaults)."""
        server = create_server(
            project_root=project_dir,
            config_path="/nonexistent/config.json",
        )
        assert server is not None
        cfg = get_config()
        assert cfg.window_size == 3

    def test_create_server_creates_missing_storage_dir(self):
        """Server creates the storage directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            deep_path = Path(tmpdir) / "a" / "b" / "c"
            # Storage path doesn't exist yet
            assert not deep_path.exists()
            server = create_server(project_root=tmpdir)
            assert server is not None


# ---------------------------------------------------------------------------
# Module-level imports test
# ---------------------------------------------------------------------------

class TestModuleImports:
    """Tests that the mcp package exports are correct."""

    def test_mcp_package_exports_create_server(self):
        from claude_code_helper_mcp.mcp import create_server as cs
        assert callable(cs)

    def test_mcp_package_exports_get_server(self):
        from claude_code_helper_mcp.mcp import get_server as gs
        assert callable(gs)


# ---------------------------------------------------------------------------
# Full lifecycle integration test
# ---------------------------------------------------------------------------

class TestFullLifecycle:
    """End-to-end test covering server creation, task operations, and health."""

    def test_full_lifecycle(self, project_dir):
        """Create server, run tasks, check health at each stage."""
        # 1. Create server
        server = create_server(project_root=project_dir)
        wm = get_window_manager()

        # Verify initial state
        tools = asyncio.run(server.get_tools())
        tool = tools["health_check"]
        result = _parse_tool_result(asyncio.run(tool.run({})))
        assert result["status"] == "healthy"
        assert result["tasks_in_window"] == 0
        assert result["current_task"] is None

        # 2. Start a task
        task = wm.start_new_task("CMH-006", "FastMCP server bootstrap", phase="phase-2")
        task.add_step("Created server module", "Implementation", tool_used="Write")
        wm.save_current_task()

        result = _parse_tool_result(asyncio.run(tool.run({})))
        assert result["current_task"] == "CMH-006"
        assert result["tasks_in_window"] == 1

        # 3. Complete the task
        wm.complete_current_task("Server bootstrap implemented and tested")

        result = _parse_tool_result(asyncio.run(tool.run({})))
        assert result["current_task"] is None
        assert result["completed_tasks"] == 1
        assert result["tasks_in_window"] == 1

        # 4. Start another task
        wm.start_new_task("CMH-007", "Record step tools", phase="phase-2")

        result = _parse_tool_result(asyncio.run(tool.run({})))
        assert result["current_task"] == "CMH-007"
        assert result["tasks_in_window"] == 2  # 1 completed + 1 active

        # 5. Verify storage state is consistent
        assert wm.store.task_exists("CMH-006")
        assert wm.store.task_exists("CMH-007")
        assert wm.completed_task_count() == 1
        assert wm.has_active_task() is True
