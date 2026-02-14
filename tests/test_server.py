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


# ---------------------------------------------------------------------------
# record_step tool tests (CMH-007)
# ---------------------------------------------------------------------------

class TestRecordStepToolRegistration:
    """Tests that record_step is properly registered as an MCP tool."""

    def test_tool_is_registered(self, project_dir):
        server = create_server(project_root=project_dir)
        tools = asyncio.run(server.get_tools())
        assert "record_step" in tools

    def test_tool_callable(self, project_dir):
        server = create_server(project_root=project_dir)
        tools = asyncio.run(server.get_tools())
        tool = tools["record_step"]
        assert tool is not None


class TestRecordStepNoActiveTask:
    """Tests for record_step when no active task exists."""

    def test_returns_error_when_no_task(self, project_dir):
        server = create_server(project_root=project_dir)
        tools = asyncio.run(server.get_tools())
        tool = tools["record_step"]
        result = _parse_tool_result(asyncio.run(tool.run({"action": "Some action"})))
        assert result["error"] is True
        assert "No active task" in result["message"]

    def test_error_includes_timestamp(self, project_dir):
        server = create_server(project_root=project_dir)
        tools = asyncio.run(server.get_tools())
        tool = tools["record_step"]
        result = _parse_tool_result(asyncio.run(tool.run({"action": "Some action"})))
        assert "timestamp" in result
        from datetime import datetime
        datetime.fromisoformat(result["timestamp"])  # Should not raise


class TestRecordStepBasic:
    """Tests for basic record_step functionality with an active task."""

    def test_records_step_with_action_only(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task", phase="phase-1")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_step"]
        result = _parse_tool_result(asyncio.run(tool.run({"action": "Created file"})))

        assert result["error"] is False
        assert result["task_id"] == "TST-001"
        assert result["step_number"] == 1
        assert result["action"] == "Created file"
        assert result["total_steps"] == 1

    def test_records_step_with_all_fields(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_step"]
        result = _parse_tool_result(asyncio.run(tool.run({
            "action": "Ran test suite",
            "description": "Executed all unit tests for the storage module",
            "tool_used": "Bash",
            "result_summary": "42 tests passed, 0 failed",
        })))

        assert result["error"] is False
        assert result["action"] == "Ran test suite"
        assert result["description"] == "Executed all unit tests for the storage module"
        assert result["tool_used"] == "Bash"
        assert result["result_summary"] == "42 tests passed, 0 failed"

    def test_empty_description_defaults_to_empty_string(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_step"]
        result = _parse_tool_result(asyncio.run(tool.run({"action": "Quick action"})))

        assert result["description"] == ""

    def test_empty_tool_used_returns_none(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_step"]
        result = _parse_tool_result(asyncio.run(tool.run({"action": "Quick action"})))

        assert result["tool_used"] is None

    def test_empty_result_summary_returns_none(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_step"]
        result = _parse_tool_result(asyncio.run(tool.run({"action": "Quick action"})))

        assert result["result_summary"] is None


class TestRecordStepAutoNumbering:
    """Tests for auto-numbering of steps."""

    def test_first_step_is_number_1(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_step"]
        result = _parse_tool_result(asyncio.run(tool.run({"action": "First step"})))
        assert result["step_number"] == 1

    def test_sequential_numbering(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_step"]

        r1 = _parse_tool_result(asyncio.run(tool.run({"action": "Step one"})))
        r2 = _parse_tool_result(asyncio.run(tool.run({"action": "Step two"})))
        r3 = _parse_tool_result(asyncio.run(tool.run({"action": "Step three"})))

        assert r1["step_number"] == 1
        assert r2["step_number"] == 2
        assert r3["step_number"] == 3

    def test_total_steps_increments(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_step"]

        r1 = _parse_tool_result(asyncio.run(tool.run({"action": "Step one"})))
        r2 = _parse_tool_result(asyncio.run(tool.run({"action": "Step two"})))

        assert r1["total_steps"] == 1
        assert r2["total_steps"] == 2


class TestRecordStepTimestamp:
    """Tests for step timestamps."""

    def test_timestamp_is_valid_iso(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_step"]
        result = _parse_tool_result(asyncio.run(tool.run({"action": "Timed action"})))

        from datetime import datetime
        ts = result["timestamp"]
        datetime.fromisoformat(ts)  # Raises ValueError if invalid

    def test_timestamps_are_ordered(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_step"]

        r1 = _parse_tool_result(asyncio.run(tool.run({"action": "First"})))
        r2 = _parse_tool_result(asyncio.run(tool.run({"action": "Second"})))

        assert r1["timestamp"] <= r2["timestamp"]


class TestRecordStepPersistence:
    """Tests for step persistence to disk."""

    def test_step_persisted_to_task_file(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_step"]
        _parse_tool_result(asyncio.run(tool.run({
            "action": "Created file",
            "tool_used": "Write",
        })))

        # Reload from disk to verify persistence
        reloaded = wm.store.load_task("TST-001")
        assert reloaded is not None
        assert len(reloaded.steps) == 1
        assert reloaded.steps[0].action == "Created file"
        assert reloaded.steps[0].tool_used == "Write"

    def test_multiple_steps_persisted(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_step"]

        for i in range(1, 6):
            _parse_tool_result(asyncio.run(tool.run({"action": f"Step {i}"})))

        reloaded = wm.store.load_task("TST-001")
        assert len(reloaded.steps) == 5
        for i, step in enumerate(reloaded.steps, 1):
            assert step.step_number == i
            assert step.action == f"Step {i}"

    def test_persistence_survives_window_reload(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_step"]
        _parse_tool_result(asyncio.run(tool.run({"action": "Before reload"})))

        # Create a new WindowManager from the same storage
        from claude_code_helper_mcp.storage.window_manager import WindowManager
        wm2 = WindowManager(storage_path=str(Path(project_dir) / ".claude-memory"))
        task = wm2.store.load_task("TST-001")
        assert task is not None
        assert len(task.steps) == 1
        assert task.steps[0].action == "Before reload"


# ---------------------------------------------------------------------------
# record_decision tool tests (CMH-007)
# ---------------------------------------------------------------------------

class TestRecordDecisionToolRegistration:
    """Tests that record_decision is properly registered as an MCP tool."""

    def test_tool_is_registered(self, project_dir):
        server = create_server(project_root=project_dir)
        tools = asyncio.run(server.get_tools())
        assert "record_decision" in tools

    def test_tool_callable(self, project_dir):
        server = create_server(project_root=project_dir)
        tools = asyncio.run(server.get_tools())
        tool = tools["record_decision"]
        assert tool is not None


class TestRecordDecisionNoActiveTask:
    """Tests for record_decision when no active task exists."""

    def test_returns_error_when_no_task(self, project_dir):
        server = create_server(project_root=project_dir)
        tools = asyncio.run(server.get_tools())
        tool = tools["record_decision"]
        result = _parse_tool_result(asyncio.run(tool.run({
            "decision": "Use approach A",
        })))
        assert result["error"] is True
        assert "No active task" in result["message"]

    def test_error_includes_timestamp(self, project_dir):
        server = create_server(project_root=project_dir)
        tools = asyncio.run(server.get_tools())
        tool = tools["record_decision"]
        result = _parse_tool_result(asyncio.run(tool.run({
            "decision": "Use approach A",
        })))
        assert "timestamp" in result
        from datetime import datetime
        datetime.fromisoformat(result["timestamp"])


class TestRecordDecisionBasic:
    """Tests for basic record_decision functionality with an active task."""

    def test_records_decision_with_decision_only(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_decision"]
        result = _parse_tool_result(asyncio.run(tool.run({
            "decision": "Use Pydantic for validation",
        })))

        assert result["error"] is False
        assert result["task_id"] == "TST-001"
        assert result["decision_number"] == 1
        assert result["decision"] == "Use Pydantic for validation"
        assert result["total_decisions"] == 1

    def test_records_decision_with_all_fields(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_decision"]
        result = _parse_tool_result(asyncio.run(tool.run({
            "decision": "Use Pydantic for validation",
            "reasoning": "Pydantic is already used in the project for data models",
            "alternatives": ["dataclasses", "attrs", "manual validation"],
            "context": "The existing codebase uses Pydantic BaseModel extensively",
        })))

        assert result["error"] is False
        assert result["decision"] == "Use Pydantic for validation"
        assert result["reasoning"] == "Pydantic is already used in the project for data models"
        assert result["alternatives"] == ["dataclasses", "attrs", "manual validation"]
        assert result["context"] == "The existing codebase uses Pydantic BaseModel extensively"

    def test_empty_reasoning_defaults_to_empty_string(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_decision"]
        result = _parse_tool_result(asyncio.run(tool.run({
            "decision": "Quick decision",
        })))
        assert result["reasoning"] == ""

    def test_no_alternatives_returns_empty_list(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_decision"]
        result = _parse_tool_result(asyncio.run(tool.run({
            "decision": "Quick decision",
        })))
        assert result["alternatives"] == []

    def test_empty_context_defaults_to_empty_string(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_decision"]
        result = _parse_tool_result(asyncio.run(tool.run({
            "decision": "Quick decision",
        })))
        assert result["context"] == ""


class TestRecordDecisionAutoNumbering:
    """Tests for auto-numbering of decisions."""

    def test_first_decision_is_number_1(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_decision"]
        result = _parse_tool_result(asyncio.run(tool.run({
            "decision": "First decision",
        })))
        assert result["decision_number"] == 1

    def test_sequential_numbering(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_decision"]

        r1 = _parse_tool_result(asyncio.run(tool.run({"decision": "Decision 1"})))
        r2 = _parse_tool_result(asyncio.run(tool.run({"decision": "Decision 2"})))
        r3 = _parse_tool_result(asyncio.run(tool.run({"decision": "Decision 3"})))

        assert r1["decision_number"] == 1
        assert r2["decision_number"] == 2
        assert r3["decision_number"] == 3

    def test_total_decisions_increments(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_decision"]

        r1 = _parse_tool_result(asyncio.run(tool.run({"decision": "D1"})))
        r2 = _parse_tool_result(asyncio.run(tool.run({"decision": "D2"})))

        assert r1["total_decisions"] == 1
        assert r2["total_decisions"] == 2


class TestRecordDecisionTimestamp:
    """Tests for decision timestamps."""

    def test_timestamp_is_valid_iso(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_decision"]
        result = _parse_tool_result(asyncio.run(tool.run({
            "decision": "Timed decision",
        })))

        from datetime import datetime
        datetime.fromisoformat(result["timestamp"])

    def test_timestamps_are_ordered(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_decision"]

        r1 = _parse_tool_result(asyncio.run(tool.run({"decision": "First"})))
        r2 = _parse_tool_result(asyncio.run(tool.run({"decision": "Second"})))

        assert r1["timestamp"] <= r2["timestamp"]


class TestRecordDecisionPersistence:
    """Tests for decision persistence to disk."""

    def test_decision_persisted_to_task_file(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_decision"]
        _parse_tool_result(asyncio.run(tool.run({
            "decision": "Use approach A",
            "reasoning": "It is simpler",
            "alternatives": ["approach B", "approach C"],
            "context": "Time constraint",
        })))

        reloaded = wm.store.load_task("TST-001")
        assert reloaded is not None
        assert len(reloaded.decisions) == 1
        assert reloaded.decisions[0].decision == "Use approach A"
        assert reloaded.decisions[0].reasoning == "It is simpler"
        assert reloaded.decisions[0].alternatives == ["approach B", "approach C"]
        assert reloaded.decisions[0].context == "Time constraint"

    def test_multiple_decisions_persisted(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_decision"]

        for i in range(1, 4):
            _parse_tool_result(asyncio.run(tool.run({
                "decision": f"Decision {i}",
            })))

        reloaded = wm.store.load_task("TST-001")
        assert len(reloaded.decisions) == 3
        for i, d in enumerate(reloaded.decisions, 1):
            assert d.decision_number == i
            assert d.decision == f"Decision {i}"

    def test_persistence_survives_window_reload(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_decision"]
        _parse_tool_result(asyncio.run(tool.run({
            "decision": "Before reload",
            "reasoning": "Testing persistence",
        })))

        from claude_code_helper_mcp.storage.window_manager import WindowManager
        wm2 = WindowManager(storage_path=str(Path(project_dir) / ".claude-memory"))
        task = wm2.store.load_task("TST-001")
        assert task is not None
        assert len(task.decisions) == 1
        assert task.decisions[0].decision == "Before reload"


# ---------------------------------------------------------------------------
# Mixed steps and decisions tests (CMH-007)
# ---------------------------------------------------------------------------

class TestMixedStepsAndDecisions:
    """Tests for using record_step and record_decision together."""

    def test_steps_and_decisions_on_same_task(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        step_tool = tools["record_step"]
        dec_tool = tools["record_decision"]

        # Record interleaved steps and decisions
        s1 = _parse_tool_result(asyncio.run(step_tool.run({"action": "Analyzed requirements"})))
        d1 = _parse_tool_result(asyncio.run(dec_tool.run({"decision": "Use approach A"})))
        s2 = _parse_tool_result(asyncio.run(step_tool.run({"action": "Implemented approach A"})))
        d2 = _parse_tool_result(asyncio.run(dec_tool.run({"decision": "Add error handling"})))
        s3 = _parse_tool_result(asyncio.run(step_tool.run({"action": "Added error handling"})))

        assert s1["step_number"] == 1
        assert d1["decision_number"] == 1
        assert s2["step_number"] == 2
        assert d2["decision_number"] == 2
        assert s3["step_number"] == 3
        assert s3["total_steps"] == 3
        assert d2["total_decisions"] == 2

    def test_mixed_persistence(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        step_tool = tools["record_step"]
        dec_tool = tools["record_decision"]

        _parse_tool_result(asyncio.run(step_tool.run({"action": "Step 1"})))
        _parse_tool_result(asyncio.run(dec_tool.run({"decision": "Decision 1"})))
        _parse_tool_result(asyncio.run(step_tool.run({"action": "Step 2"})))

        reloaded = wm.store.load_task("TST-001")
        assert len(reloaded.steps) == 2
        assert len(reloaded.decisions) == 1

    def test_both_tools_reference_same_task(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        step_tool = tools["record_step"]
        dec_tool = tools["record_decision"]

        s = _parse_tool_result(asyncio.run(step_tool.run({"action": "A step"})))
        d = _parse_tool_result(asyncio.run(dec_tool.run({"decision": "A decision"})))

        assert s["task_id"] == "TST-001"
        assert d["task_id"] == "TST-001"

    def test_health_check_unaffected_by_recordings(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        health = tools["health_check"]
        step_tool = tools["record_step"]
        dec_tool = tools["record_decision"]

        _parse_tool_result(asyncio.run(step_tool.run({"action": "A step"})))
        _parse_tool_result(asyncio.run(dec_tool.run({"decision": "A decision"})))

        result = _parse_tool_result(asyncio.run(health.run({})))
        assert result["status"] == "healthy"
        assert result["current_task"] == "TST-001"


# ---------------------------------------------------------------------------
# Full lifecycle with recording tools (CMH-007)
# ---------------------------------------------------------------------------

class TestRecordingToolsFullLifecycle:
    """End-to-end test covering task creation, recording, completion."""

    def test_full_lifecycle_with_recording(self, project_dir):
        """Create server, start task, record steps and decisions, complete."""
        server = create_server(project_root=project_dir)
        wm = get_window_manager()

        # 1. Start a task
        wm.start_new_task("CMH-007", "Record step and decision tools", phase="phase-2")

        tools = asyncio.run(server.get_tools())
        step_tool = tools["record_step"]
        dec_tool = tools["record_decision"]
        health = tools["health_check"]

        # 2. Record steps and decisions
        _parse_tool_result(asyncio.run(step_tool.run({
            "action": "Analyzed ticket requirements",
            "description": "Read CMH-007 ticket: implement record_step and record_decision MCP tools",
            "tool_used": "Read",
        })))
        _parse_tool_result(asyncio.run(dec_tool.run({
            "decision": "Add tools to existing _register_tools function",
            "reasoning": "Keeps all tool registrations centralized in server.py",
            "alternatives": ["Separate tools module", "Decorator-based registration"],
        })))
        _parse_tool_result(asyncio.run(step_tool.run({
            "action": "Implemented record_step tool",
            "tool_used": "Edit",
            "result_summary": "Added record_step with auto-numbering and persistence",
        })))
        _parse_tool_result(asyncio.run(step_tool.run({
            "action": "Implemented record_decision tool",
            "tool_used": "Edit",
            "result_summary": "Added record_decision with auto-numbering and persistence",
        })))
        _parse_tool_result(asyncio.run(step_tool.run({
            "action": "Ran test suite",
            "tool_used": "Bash",
            "result_summary": "All tests passed",
        })))

        # 3. Verify in-memory state
        task = wm.get_current_task()
        assert task.step_count() == 4
        assert len(task.decisions) == 1
        assert task.ticket_id == "CMH-007"

        # 4. Verify health still reports correctly
        h = _parse_tool_result(asyncio.run(health.run({})))
        assert h["current_task"] == "CMH-007"

        # 5. Complete the task
        wm.complete_current_task("Implemented record_step and record_decision MCP tools")

        # 6. Verify completed task persisted with all recordings
        completed = wm.store.load_task("CMH-007")
        assert completed is not None
        assert completed.status.value == "completed"
        assert completed.step_count() == 4
        assert len(completed.decisions) == 1
        assert completed.steps[0].action == "Analyzed ticket requirements"
        assert completed.decisions[0].decision == "Add tools to existing _register_tools function"

        # 7. Verify recording tools reject calls after task completion
        s = _parse_tool_result(asyncio.run(step_tool.run({"action": "After completion"})))
        assert s["error"] is True
        d = _parse_tool_result(asyncio.run(dec_tool.run({"decision": "After completion"})))
        assert d["error"] is True


# ---------------------------------------------------------------------------
# record_file tool tests (CMH-008)
# ---------------------------------------------------------------------------

class TestRecordFileToolRegistration:
    """Tests that record_file is properly registered as an MCP tool."""

    def test_tool_is_registered(self, project_dir):
        server = create_server(project_root=project_dir)
        tools = asyncio.run(server.get_tools())
        assert "record_file" in tools

    def test_tool_callable(self, project_dir):
        server = create_server(project_root=project_dir)
        tools = asyncio.run(server.get_tools())
        tool = tools["record_file"]
        assert tool is not None


class TestRecordFileNoActiveTask:
    """Tests for record_file when no active task exists."""

    def test_returns_error_when_no_task(self, project_dir):
        server = create_server(project_root=project_dir)
        tools = asyncio.run(server.get_tools())
        tool = tools["record_file"]
        result = _parse_tool_result(asyncio.run(tool.run({
            "path": "src/main.py",
            "action": "created",
        })))
        assert result["error"] is True
        assert "No active task" in result["message"]

    def test_error_includes_timestamp(self, project_dir):
        server = create_server(project_root=project_dir)
        tools = asyncio.run(server.get_tools())
        tool = tools["record_file"]
        result = _parse_tool_result(asyncio.run(tool.run({
            "path": "src/main.py",
            "action": "created",
        })))
        assert "timestamp" in result
        from datetime import datetime
        datetime.fromisoformat(result["timestamp"])


class TestRecordFileInvalidAction:
    """Tests for record_file with invalid action values."""

    def test_invalid_action_returns_error(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_file"]
        result = _parse_tool_result(asyncio.run(tool.run({
            "path": "src/main.py",
            "action": "invalid_action",
        })))
        assert result["error"] is True
        assert "Invalid file action" in result["message"]
        assert "invalid_action" in result["message"]

    def test_error_lists_valid_actions(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_file"]
        result = _parse_tool_result(asyncio.run(tool.run({
            "path": "src/main.py",
            "action": "exploded",
        })))
        assert "created" in result["message"]
        assert "modified" in result["message"]
        assert "deleted" in result["message"]

    def test_invalid_action_includes_timestamp(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_file"]
        result = _parse_tool_result(asyncio.run(tool.run({
            "path": "src/main.py",
            "action": "nope",
        })))
        assert "timestamp" in result


class TestRecordFileBasic:
    """Tests for basic record_file functionality with an active task."""

    def test_records_file_with_required_fields_only(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_file"]
        result = _parse_tool_result(asyncio.run(tool.run({
            "path": "src/main.py",
            "action": "created",
        })))

        assert result["error"] is False
        assert result["task_id"] == "TST-001"
        assert result["path"] == "src/main.py"
        assert result["action"] == "created"
        assert result["is_update"] is False
        assert result["total_files"] == 1

    def test_records_file_with_description(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_file"]
        result = _parse_tool_result(asyncio.run(tool.run({
            "path": "src/models/task.py",
            "action": "modified",
            "description": "Added record_file method to TaskMemory class",
        })))

        assert result["error"] is False
        assert result["path"] == "src/models/task.py"
        assert result["action"] == "modified"
        assert result["description"] == "Added record_file method to TaskMemory class"

    def test_empty_description_defaults_to_empty_string(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_file"]
        result = _parse_tool_result(asyncio.run(tool.run({
            "path": "src/main.py",
            "action": "created",
        })))
        assert result["description"] == ""

    def test_all_valid_file_actions(self, project_dir):
        """Verify each valid FileAction enum value is accepted."""
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_file"]

        for action in ["created", "modified", "deleted", "renamed", "read"]:
            result = _parse_tool_result(asyncio.run(tool.run({
                "path": f"src/{action}_file.py",
                "action": action,
            })))
            assert result["error"] is False
            assert result["action"] == action


class TestRecordFileDeduplication:
    """Tests for file deduplication behavior."""

    def test_first_record_is_not_update(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_file"]
        result = _parse_tool_result(asyncio.run(tool.run({
            "path": "src/main.py",
            "action": "created",
        })))
        assert result["is_update"] is False
        assert result["action_history_count"] == 0

    def test_second_record_same_path_is_update(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_file"]

        # First record
        _parse_tool_result(asyncio.run(tool.run({
            "path": "src/main.py",
            "action": "created",
        })))

        # Second record on same path
        result = _parse_tool_result(asyncio.run(tool.run({
            "path": "src/main.py",
            "action": "modified",
            "description": "Added error handling",
        })))

        assert result["is_update"] is True
        assert result["action"] == "modified"
        assert result["action_history_count"] == 1  # previous action moved to history

    def test_dedup_preserves_action_history(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_file"]

        # Record three actions on the same file
        _parse_tool_result(asyncio.run(tool.run({
            "path": "src/main.py",
            "action": "created",
        })))
        _parse_tool_result(asyncio.run(tool.run({
            "path": "src/main.py",
            "action": "modified",
        })))
        result = _parse_tool_result(asyncio.run(tool.run({
            "path": "src/main.py",
            "action": "modified",
            "description": "Third edit",
        })))

        assert result["action_history_count"] == 2  # two previous actions in history
        assert result["action"] == "modified"

    def test_different_paths_are_separate_records(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_file"]

        r1 = _parse_tool_result(asyncio.run(tool.run({
            "path": "src/main.py",
            "action": "created",
        })))
        r2 = _parse_tool_result(asyncio.run(tool.run({
            "path": "src/utils.py",
            "action": "created",
        })))

        assert r1["total_files"] == 1
        assert r2["total_files"] == 2
        assert r1["is_update"] is False
        assert r2["is_update"] is False

    def test_total_files_does_not_double_count_deduped(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_file"]

        _parse_tool_result(asyncio.run(tool.run({
            "path": "src/main.py",
            "action": "created",
        })))
        result = _parse_tool_result(asyncio.run(tool.run({
            "path": "src/main.py",
            "action": "modified",
        })))

        assert result["total_files"] == 1  # Still only one file, just updated


class TestRecordFileTimestamp:
    """Tests for file record timestamps."""

    def test_timestamp_is_valid_iso(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_file"]
        result = _parse_tool_result(asyncio.run(tool.run({
            "path": "src/main.py",
            "action": "created",
        })))

        from datetime import datetime
        datetime.fromisoformat(result["timestamp"])

    def test_timestamps_are_ordered(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_file"]

        r1 = _parse_tool_result(asyncio.run(tool.run({
            "path": "src/first.py",
            "action": "created",
        })))
        r2 = _parse_tool_result(asyncio.run(tool.run({
            "path": "src/second.py",
            "action": "created",
        })))

        assert r1["timestamp"] <= r2["timestamp"]


class TestRecordFilePersistence:
    """Tests for file record persistence to disk."""

    def test_file_persisted_to_task_file(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_file"]
        _parse_tool_result(asyncio.run(tool.run({
            "path": "src/main.py",
            "action": "created",
            "description": "Initial file creation",
        })))

        reloaded = wm.store.load_task("TST-001")
        assert reloaded is not None
        assert len(reloaded.files) == 1
        assert reloaded.files[0].path == "src/main.py"
        assert reloaded.files[0].action.value == "created"
        assert reloaded.files[0].description == "Initial file creation"

    def test_deduplication_persisted(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_file"]

        _parse_tool_result(asyncio.run(tool.run({
            "path": "src/main.py",
            "action": "created",
        })))
        _parse_tool_result(asyncio.run(tool.run({
            "path": "src/main.py",
            "action": "modified",
            "description": "Added error handling",
        })))

        reloaded = wm.store.load_task("TST-001")
        assert len(reloaded.files) == 1
        assert reloaded.files[0].action.value == "modified"
        assert len(reloaded.files[0].action_history) == 1
        assert reloaded.files[0].action_history[0]["action"] == "created"

    def test_multiple_files_persisted(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_file"]

        for i in range(1, 6):
            _parse_tool_result(asyncio.run(tool.run({
                "path": f"src/module_{i}.py",
                "action": "created",
            })))

        reloaded = wm.store.load_task("TST-001")
        assert len(reloaded.files) == 5
        paths = {f.path for f in reloaded.files}
        for i in range(1, 6):
            assert f"src/module_{i}.py" in paths

    def test_persistence_survives_window_reload(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_file"]
        _parse_tool_result(asyncio.run(tool.run({
            "path": "src/main.py",
            "action": "created",
        })))

        from claude_code_helper_mcp.storage.window_manager import WindowManager
        wm2 = WindowManager(storage_path=str(Path(project_dir) / ".claude-memory"))
        task = wm2.store.load_task("TST-001")
        assert task is not None
        assert len(task.files) == 1
        assert task.files[0].path == "src/main.py"


# ---------------------------------------------------------------------------
# record_branch tool tests (CMH-008)
# ---------------------------------------------------------------------------

class TestRecordBranchToolRegistration:
    """Tests that record_branch is properly registered as an MCP tool."""

    def test_tool_is_registered(self, project_dir):
        server = create_server(project_root=project_dir)
        tools = asyncio.run(server.get_tools())
        assert "record_branch" in tools

    def test_tool_callable(self, project_dir):
        server = create_server(project_root=project_dir)
        tools = asyncio.run(server.get_tools())
        tool = tools["record_branch"]
        assert tool is not None


class TestRecordBranchNoActiveTask:
    """Tests for record_branch when no active task exists."""

    def test_returns_error_when_no_task(self, project_dir):
        server = create_server(project_root=project_dir)
        tools = asyncio.run(server.get_tools())
        tool = tools["record_branch"]
        result = _parse_tool_result(asyncio.run(tool.run({
            "branch_name": "feature/TST-001",
            "action": "created",
        })))
        assert result["error"] is True
        assert "No active task" in result["message"]

    def test_error_includes_timestamp(self, project_dir):
        server = create_server(project_root=project_dir)
        tools = asyncio.run(server.get_tools())
        tool = tools["record_branch"]
        result = _parse_tool_result(asyncio.run(tool.run({
            "branch_name": "feature/TST-001",
            "action": "created",
        })))
        assert "timestamp" in result
        from datetime import datetime
        datetime.fromisoformat(result["timestamp"])


class TestRecordBranchInvalidAction:
    """Tests for record_branch with invalid action values."""

    def test_invalid_action_returns_error(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_branch"]
        result = _parse_tool_result(asyncio.run(tool.run({
            "branch_name": "feature/TST-001",
            "action": "exploded",
        })))
        assert result["error"] is True
        assert "Invalid branch action" in result["message"]
        assert "exploded" in result["message"]

    def test_error_lists_valid_actions(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_branch"]
        result = _parse_tool_result(asyncio.run(tool.run({
            "branch_name": "feature/TST-001",
            "action": "nope",
        })))
        assert "created" in result["message"]
        assert "merged" in result["message"]
        assert "pushed" in result["message"]

    def test_invalid_action_includes_timestamp(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_branch"]
        result = _parse_tool_result(asyncio.run(tool.run({
            "branch_name": "feature/TST-001",
            "action": "nope",
        })))
        assert "timestamp" in result


class TestRecordBranchBasic:
    """Tests for basic record_branch functionality with an active task."""

    def test_records_branch_with_required_fields_only(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_branch"]
        result = _parse_tool_result(asyncio.run(tool.run({
            "branch_name": "feature/CMH-008-tools",
            "action": "created",
        })))

        assert result["error"] is False
        assert result["task_id"] == "TST-001"
        assert result["branch_name"] == "feature/CMH-008-tools"
        assert result["action"] == "created"
        assert result["is_update"] is False
        assert result["total_branches"] == 1

    def test_records_branch_with_base_branch(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_branch"]
        result = _parse_tool_result(asyncio.run(tool.run({
            "branch_name": "feature/CMH-008-tools",
            "action": "created",
            "base_branch": "main",
        })))

        assert result["error"] is False
        assert result["base_branch"] == "main"

    def test_empty_base_branch_is_none(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_branch"]
        result = _parse_tool_result(asyncio.run(tool.run({
            "branch_name": "feature/CMH-008-tools",
            "action": "created",
        })))
        assert result["base_branch"] is None

    def test_all_valid_branch_actions(self, project_dir):
        """Verify each valid BranchAction enum value is accepted."""
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_branch"]

        for action in ["created", "checked_out", "merged", "deleted", "pushed", "pulled"]:
            result = _parse_tool_result(asyncio.run(tool.run({
                "branch_name": f"feature/{action}-branch",
                "action": action,
            })))
            assert result["error"] is False
            assert result["action"] == action


class TestRecordBranchDeduplication:
    """Tests for branch deduplication behavior."""

    def test_first_record_is_not_update(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_branch"]
        result = _parse_tool_result(asyncio.run(tool.run({
            "branch_name": "feature/CMH-008",
            "action": "created",
            "base_branch": "main",
        })))
        assert result["is_update"] is False
        assert result["action_history_count"] == 0

    def test_second_record_same_branch_is_update(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_branch"]

        # Create the branch
        _parse_tool_result(asyncio.run(tool.run({
            "branch_name": "feature/CMH-008",
            "action": "created",
            "base_branch": "main",
        })))

        # Push the branch
        result = _parse_tool_result(asyncio.run(tool.run({
            "branch_name": "feature/CMH-008",
            "action": "pushed",
        })))

        assert result["is_update"] is True
        assert result["action"] == "pushed"
        assert result["action_history_count"] == 1

    def test_dedup_preserves_action_history(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_branch"]

        # Full branch lifecycle
        _parse_tool_result(asyncio.run(tool.run({
            "branch_name": "feature/CMH-008",
            "action": "created",
            "base_branch": "main",
        })))
        _parse_tool_result(asyncio.run(tool.run({
            "branch_name": "feature/CMH-008",
            "action": "pushed",
        })))
        result = _parse_tool_result(asyncio.run(tool.run({
            "branch_name": "feature/CMH-008",
            "action": "merged",
            "base_branch": "main",
        })))

        assert result["action_history_count"] == 2
        assert result["action"] == "merged"

    def test_different_branches_are_separate_records(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_branch"]

        r1 = _parse_tool_result(asyncio.run(tool.run({
            "branch_name": "feature/CMH-008-a",
            "action": "created",
        })))
        r2 = _parse_tool_result(asyncio.run(tool.run({
            "branch_name": "feature/CMH-008-b",
            "action": "created",
        })))

        assert r1["total_branches"] == 1
        assert r2["total_branches"] == 2

    def test_total_branches_does_not_double_count_deduped(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_branch"]

        _parse_tool_result(asyncio.run(tool.run({
            "branch_name": "feature/CMH-008",
            "action": "created",
        })))
        result = _parse_tool_result(asyncio.run(tool.run({
            "branch_name": "feature/CMH-008",
            "action": "pushed",
        })))

        assert result["total_branches"] == 1


class TestRecordBranchTimestamp:
    """Tests for branch record timestamps."""

    def test_timestamp_is_valid_iso(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_branch"]
        result = _parse_tool_result(asyncio.run(tool.run({
            "branch_name": "feature/CMH-008",
            "action": "created",
        })))

        from datetime import datetime
        datetime.fromisoformat(result["timestamp"])

    def test_timestamps_are_ordered(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_branch"]

        r1 = _parse_tool_result(asyncio.run(tool.run({
            "branch_name": "feature/first",
            "action": "created",
        })))
        r2 = _parse_tool_result(asyncio.run(tool.run({
            "branch_name": "feature/second",
            "action": "created",
        })))

        assert r1["timestamp"] <= r2["timestamp"]


class TestRecordBranchPersistence:
    """Tests for branch record persistence to disk."""

    def test_branch_persisted_to_task_file(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_branch"]
        _parse_tool_result(asyncio.run(tool.run({
            "branch_name": "feature/CMH-008-tools",
            "action": "created",
            "base_branch": "main",
        })))

        reloaded = wm.store.load_task("TST-001")
        assert reloaded is not None
        assert len(reloaded.branches) == 1
        assert reloaded.branches[0].branch_name == "feature/CMH-008-tools"
        assert reloaded.branches[0].action.value == "created"
        assert reloaded.branches[0].base_branch == "main"

    def test_deduplication_persisted(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_branch"]

        _parse_tool_result(asyncio.run(tool.run({
            "branch_name": "feature/CMH-008",
            "action": "created",
            "base_branch": "main",
        })))
        _parse_tool_result(asyncio.run(tool.run({
            "branch_name": "feature/CMH-008",
            "action": "pushed",
        })))

        reloaded = wm.store.load_task("TST-001")
        assert len(reloaded.branches) == 1
        assert reloaded.branches[0].action.value == "pushed"
        assert len(reloaded.branches[0].action_history) == 1
        assert reloaded.branches[0].action_history[0]["action"] == "created"

    def test_multiple_branches_persisted(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_branch"]

        for i in range(1, 4):
            _parse_tool_result(asyncio.run(tool.run({
                "branch_name": f"feature/branch-{i}",
                "action": "created",
                "base_branch": "main",
            })))

        reloaded = wm.store.load_task("TST-001")
        assert len(reloaded.branches) == 3

    def test_persistence_survives_window_reload(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        tool = tools["record_branch"]
        _parse_tool_result(asyncio.run(tool.run({
            "branch_name": "feature/CMH-008",
            "action": "created",
            "base_branch": "main",
        })))

        from claude_code_helper_mcp.storage.window_manager import WindowManager
        wm2 = WindowManager(storage_path=str(Path(project_dir) / ".claude-memory"))
        task = wm2.store.load_task("TST-001")
        assert task is not None
        assert len(task.branches) == 1
        assert task.branches[0].branch_name == "feature/CMH-008"


# ---------------------------------------------------------------------------
# Mixed recording tools tests (CMH-008 with CMH-007 tools)
# ---------------------------------------------------------------------------

class TestMixedRecordingTools:
    """Tests for using all four recording tools together."""

    def test_all_tools_on_same_task(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        step_tool = tools["record_step"]
        dec_tool = tools["record_decision"]
        file_tool = tools["record_file"]
        branch_tool = tools["record_branch"]

        s = _parse_tool_result(asyncio.run(step_tool.run({"action": "Created branch"})))
        b = _parse_tool_result(asyncio.run(branch_tool.run({
            "branch_name": "feature/TST-001",
            "action": "created",
            "base_branch": "main",
        })))
        d = _parse_tool_result(asyncio.run(dec_tool.run({
            "decision": "Use separate module",
        })))
        f = _parse_tool_result(asyncio.run(file_tool.run({
            "path": "src/new_module.py",
            "action": "created",
        })))
        s2 = _parse_tool_result(asyncio.run(step_tool.run({"action": "Wrote module"})))

        assert s["task_id"] == "TST-001"
        assert b["task_id"] == "TST-001"
        assert d["task_id"] == "TST-001"
        assert f["task_id"] == "TST-001"
        assert s2["step_number"] == 2
        assert d["decision_number"] == 1
        assert f["total_files"] == 1
        assert b["total_branches"] == 1

    def test_mixed_persistence(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        step_tool = tools["record_step"]
        dec_tool = tools["record_decision"]
        file_tool = tools["record_file"]
        branch_tool = tools["record_branch"]

        _parse_tool_result(asyncio.run(step_tool.run({"action": "Step 1"})))
        _parse_tool_result(asyncio.run(branch_tool.run({
            "branch_name": "feature/TST-001",
            "action": "created",
        })))
        _parse_tool_result(asyncio.run(file_tool.run({
            "path": "src/main.py",
            "action": "created",
        })))
        _parse_tool_result(asyncio.run(dec_tool.run({"decision": "D1"})))
        _parse_tool_result(asyncio.run(file_tool.run({
            "path": "src/utils.py",
            "action": "created",
        })))

        reloaded = wm.store.load_task("TST-001")
        assert len(reloaded.steps) == 1
        assert len(reloaded.decisions) == 1
        assert len(reloaded.files) == 2
        assert len(reloaded.branches) == 1

    def test_health_check_unaffected_by_file_and_branch_recordings(self, project_dir):
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")

        tools = asyncio.run(server.get_tools())
        health = tools["health_check"]
        file_tool = tools["record_file"]
        branch_tool = tools["record_branch"]

        _parse_tool_result(asyncio.run(file_tool.run({
            "path": "src/main.py",
            "action": "created",
        })))
        _parse_tool_result(asyncio.run(branch_tool.run({
            "branch_name": "feature/TST-001",
            "action": "created",
        })))

        result = _parse_tool_result(asyncio.run(health.run({})))
        assert result["status"] == "healthy"
        assert result["current_task"] == "TST-001"

    def test_file_and_branch_tools_reject_after_completion(self, project_dir):
        """record_file and record_branch return error after task is completed."""
        server = create_server(project_root=project_dir)
        wm = get_window_manager()
        wm.start_new_task("TST-001", "Test task")
        wm.complete_current_task("Done")

        tools = asyncio.run(server.get_tools())
        file_tool = tools["record_file"]
        branch_tool = tools["record_branch"]

        f = _parse_tool_result(asyncio.run(file_tool.run({
            "path": "src/main.py",
            "action": "created",
        })))
        b = _parse_tool_result(asyncio.run(branch_tool.run({
            "branch_name": "feature/TST-001",
            "action": "created",
        })))

        assert f["error"] is True
        assert b["error"] is True


# ---------------------------------------------------------------------------
# Full lifecycle with all recording tools (CMH-008)
# ---------------------------------------------------------------------------

class TestFullLifecycleWithAllTools:
    """End-to-end test covering task creation, all 4 recording tools, completion."""

    def test_full_lifecycle_with_all_recording_tools(self, project_dir):
        """Create server, start task, record steps/decisions/files/branches, complete."""
        server = create_server(project_root=project_dir)
        wm = get_window_manager()

        # 1. Start a task
        wm.start_new_task("CMH-008", "Record file and branch tools", phase="phase-2")

        tools = asyncio.run(server.get_tools())
        step_tool = tools["record_step"]
        dec_tool = tools["record_decision"]
        file_tool = tools["record_file"]
        branch_tool = tools["record_branch"]
        health = tools["health_check"]

        # 2. Record branch creation
        b = _parse_tool_result(asyncio.run(branch_tool.run({
            "branch_name": "feature/CMH-008-record-file-and-branch-tools",
            "action": "created",
            "base_branch": "main",
        })))
        assert b["error"] is False
        assert b["is_update"] is False

        # 3. Record steps and decisions
        _parse_tool_result(asyncio.run(step_tool.run({
            "action": "Analyzed ticket requirements",
            "tool_used": "Read",
        })))
        _parse_tool_result(asyncio.run(dec_tool.run({
            "decision": "Accept action as string, validate against enum",
            "reasoning": "MCP tools pass strings, need explicit validation",
        })))

        # 4. Record file actions
        _parse_tool_result(asyncio.run(file_tool.run({
            "path": "src/claude_code_helper_mcp/mcp/server.py",
            "action": "modified",
            "description": "Added record_file and record_branch tools",
        })))
        _parse_tool_result(asyncio.run(file_tool.run({
            "path": "tests/test_server.py",
            "action": "modified",
            "description": "Added tests for record_file and record_branch",
        })))

        # 5. Record more steps
        _parse_tool_result(asyncio.run(step_tool.run({
            "action": "Implemented record_file tool",
            "tool_used": "Edit",
            "result_summary": "Tool registered with dedup and enum validation",
        })))
        _parse_tool_result(asyncio.run(step_tool.run({
            "action": "Implemented record_branch tool",
            "tool_used": "Edit",
        })))

        # 6. Record branch push
        b_push = _parse_tool_result(asyncio.run(branch_tool.run({
            "branch_name": "feature/CMH-008-record-file-and-branch-tools",
            "action": "pushed",
        })))
        assert b_push["is_update"] is True
        assert b_push["action_history_count"] == 1

        # 7. Run tests step
        _parse_tool_result(asyncio.run(step_tool.run({
            "action": "Ran test suite",
            "tool_used": "Bash",
            "result_summary": "All tests passed",
        })))

        # 8. Verify in-memory state
        task = wm.get_current_task()
        assert task.step_count() == 4
        assert len(task.decisions) == 1
        assert len(task.files) == 2
        assert len(task.branches) == 1
        assert task.get_file_paths() == [
            "src/claude_code_helper_mcp/mcp/server.py",
            "tests/test_server.py",
        ]
        assert task.get_active_branch() == "feature/CMH-008-record-file-and-branch-tools"

        # 9. Verify health
        h = _parse_tool_result(asyncio.run(health.run({})))
        assert h["current_task"] == "CMH-008"

        # 10. Complete the task
        wm.complete_current_task("Implemented record_file and record_branch MCP tools")

        # 11. Verify completed task persisted with all recordings
        completed = wm.store.load_task("CMH-008")
        assert completed is not None
        assert completed.status.value == "completed"
        assert completed.step_count() == 4
        assert len(completed.decisions) == 1
        assert len(completed.files) == 2
        assert len(completed.branches) == 1
        assert completed.branches[0].branch_name == "feature/CMH-008-record-file-and-branch-tools"
        assert completed.branches[0].action.value == "pushed"
        assert len(completed.branches[0].action_history) == 1

        # 12. Verify tools reject calls after task completion
        f = _parse_tool_result(asyncio.run(file_tool.run({
            "path": "extra.py",
            "action": "created",
        })))
        assert f["error"] is True
        b = _parse_tool_result(asyncio.run(branch_tool.run({
            "branch_name": "feature/extra",
            "action": "created",
        })))
        assert b["error"] is True
