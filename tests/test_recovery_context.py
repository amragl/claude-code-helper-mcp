"""Tests for the get_recovery_context MCP tool (CMH-011).

Verifies recovery context generation for active tasks, recently completed tasks,
explicit ticket ID lookups, edge cases (no tasks, archived tasks, empty steps),
the formatted recovery prompt, and the configurable recent_step_count parameter.

All tests use real file I/O with temporary directories -- zero mocks.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from claude_code_helper_mcp.mcp.server import (
    create_server,
    get_window_manager,
    reset_server,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_tool_result(result) -> dict:
    """Extract a dict from a FastMCP ToolResult."""
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        return json.loads(result)
    if hasattr(result, "content") and result.content:
        text = result.content[0].text
        return json.loads(text)
    raise TypeError(f"Cannot parse tool result of type {type(result)}")


def _get_tools(project_dir: str) -> dict:
    """Create a server and return its tools dict."""
    server = create_server(project_root=project_dir)
    return asyncio.run(server.get_tools())


def _run_tool(tools: dict, name: str, args: dict | None = None) -> dict:
    """Run a tool and return the parsed result dict."""
    tool = tools[name]
    result = asyncio.run(tool.run(args or {}))
    return _parse_tool_result(result)


def _start_task(tools: dict, ticket_id: str, title: str,
                phase: str = "", description: str = "") -> dict:
    """Helper to start a task via the start_task tool."""
    args = {"ticket_id": ticket_id, "title": title}
    if phase:
        args["phase"] = phase
    if description:
        args["description"] = description
    return _run_tool(tools, "start_task", args)


def _complete_task(tools: dict, summary: str = "") -> dict:
    """Helper to complete the current task."""
    return _run_tool(tools, "complete_task", {"summary": summary})


def _record_step(tools: dict, action: str, description: str = "",
                 tool_used: str = "", result_summary: str = "") -> dict:
    """Helper to record a step."""
    return _run_tool(tools, "record_step", {
        "action": action,
        "description": description,
        "tool_used": tool_used,
        "result_summary": result_summary,
    })


def _record_decision(tools: dict, decision: str, reasoning: str = "",
                     alternatives: list[str] | None = None,
                     context: str = "") -> dict:
    """Helper to record a decision."""
    args = {"decision": decision, "reasoning": reasoning, "context": context}
    if alternatives:
        args["alternatives"] = alternatives
    return _run_tool(tools, "record_decision", args)


def _record_file(tools: dict, path: str, action: str,
                 description: str = "") -> dict:
    """Helper to record a file action."""
    return _run_tool(tools, "record_file", {
        "path": path, "action": action, "description": description,
    })


def _record_branch(tools: dict, branch_name: str, action: str,
                   base_branch: str = "") -> dict:
    """Helper to record a branch action."""
    return _run_tool(tools, "record_branch", {
        "branch_name": branch_name, "action": action,
        "base_branch": base_branch,
    })


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
    """Create a temporary project directory with a .git marker."""
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / ".git").mkdir()
        yield tmpdir


@pytest.fixture
def tools(project_dir):
    """Create a server and return tools dict."""
    return _get_tools(project_dir)


# ---------------------------------------------------------------------------
# Tool registration tests
# ---------------------------------------------------------------------------

class TestGetRecoveryContextRegistration:
    """Tests that get_recovery_context is properly registered."""

    def test_tool_is_registered(self, tools):
        assert "get_recovery_context" in tools

    def test_tool_callable(self, tools):
        tool = tools["get_recovery_context"]
        assert tool is not None


# ---------------------------------------------------------------------------
# No tasks -- empty window
# ---------------------------------------------------------------------------

class TestGetRecoveryContextNoTasks:
    """Tests for get_recovery_context when no tasks exist."""

    def test_returns_no_context_message(self, tools):
        result = _run_tool(tools, "get_recovery_context")
        assert result["error"] is False
        assert result["has_context"] is False
        assert "No tasks found" in result["message"]

    def test_includes_window_state(self, tools):
        result = _run_tool(tools, "get_recovery_context")
        assert "window_state" in result
        assert result["window_state"]["tasks_in_window"] == 0

    def test_includes_timestamp(self, tools):
        result = _run_tool(tools, "get_recovery_context")
        assert "timestamp" in result
        assert result["timestamp"]  # non-empty

    def test_includes_archived_task_ids(self, tools):
        result = _run_tool(tools, "get_recovery_context")
        assert "archived_task_ids" in result
        assert isinstance(result["archived_task_ids"], list)

    def test_explicit_nonexistent_ticket_returns_error(self, tools):
        result = _run_tool(tools, "get_recovery_context", {
            "ticket_id": "CMH-999"
        })
        assert result["error"] is True
        assert "CMH-999" in result["message"]
        assert "not found" in result["message"]

    def test_explicit_nonexistent_ticket_includes_available_tasks(self, tools):
        result = _run_tool(tools, "get_recovery_context", {
            "ticket_id": "CMH-999"
        })
        assert "available_tasks" in result
        assert isinstance(result["available_tasks"], list)


# ---------------------------------------------------------------------------
# Active task -- auto-detection
# ---------------------------------------------------------------------------

class TestGetRecoveryContextActiveTask:
    """Tests for recovery context with an active task (auto-detected)."""

    def test_recovers_active_task(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool", phase="phase-3")
        result = _run_tool(tools, "get_recovery_context")
        assert result["error"] is False
        assert result["has_context"] is True
        assert result["ticket_id"] == "CMH-011"
        assert result["source"] == "active"

    def test_includes_title(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        result = _run_tool(tools, "get_recovery_context")
        assert result["title"] == "Recovery context tool"

    def test_includes_phase(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool", phase="phase-3")
        result = _run_tool(tools, "get_recovery_context")
        assert result["phase"] == "phase-3"

    def test_includes_status(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        result = _run_tool(tools, "get_recovery_context")
        assert result["status"] == "active"

    def test_includes_generated_at(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        result = _run_tool(tools, "get_recovery_context")
        assert "generated_at" in result
        assert result["generated_at"]

    def test_includes_task_started_at(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        result = _run_tool(tools, "get_recovery_context")
        assert "task_started_at" in result
        assert result["task_started_at"] is not None

    def test_empty_steps_when_none_recorded(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        result = _run_tool(tools, "get_recovery_context")
        assert result["recent_steps"] == []
        assert result["last_step"] is None
        assert result["total_steps_completed"] == 0

    def test_empty_files_when_none_recorded(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        result = _run_tool(tools, "get_recovery_context")
        assert result["files_modified"] == []

    def test_empty_decisions_when_none_recorded(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        result = _run_tool(tools, "get_recovery_context")
        assert result["key_decisions"] == []

    def test_no_active_branch_when_none_recorded(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        result = _run_tool(tools, "get_recovery_context")
        assert result["active_branch"] is None

    def test_includes_metadata(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool",
                    description="A real description")
        result = _run_tool(tools, "get_recovery_context")
        assert "metadata" in result
        assert result["metadata"].get("description") == "A real description"

    def test_includes_window_state(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        result = _run_tool(tools, "get_recovery_context")
        assert "window_state" in result
        assert result["window_state"]["tasks_in_window"] == 1


# ---------------------------------------------------------------------------
# Active task with recorded data
# ---------------------------------------------------------------------------

class TestGetRecoveryContextWithRecordedData:
    """Tests for recovery context with steps, files, decisions, and branches."""

    def test_recovers_steps(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        _record_step(tools, "Created models", "Added Pydantic models", "Write")
        _record_step(tools, "Wrote tests", "Created test_recovery.py", "Write")
        _record_step(tools, "Ran tests", "All passing", "Bash", "3 tests passed")

        result = _run_tool(tools, "get_recovery_context")
        assert result["total_steps_completed"] == 3
        assert len(result["recent_steps"]) == 3

    def test_last_step_is_most_recent(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        _record_step(tools, "Step one", "", "Write")
        _record_step(tools, "Step two", "", "Bash")
        _record_step(tools, "Step three", "", "Edit")

        result = _run_tool(tools, "get_recovery_context")
        assert result["last_step"] is not None
        assert result["last_step"]["action"] == "Step three"
        assert result["last_step"]["step_number"] == 3

    def test_recent_steps_in_reverse_order(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        _record_step(tools, "Step A")
        _record_step(tools, "Step B")
        _record_step(tools, "Step C")

        result = _run_tool(tools, "get_recovery_context")
        # RecoveryContext.from_task_memory reverses: most recent first
        actions = [s["action"] for s in result["recent_steps"]]
        assert actions == ["Step C", "Step B", "Step A"]

    def test_recovers_files(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        _record_file(tools, "src/mcp/server.py", "modified", "Added tool")
        _record_file(tools, "tests/test_recovery.py", "created", "New tests")

        result = _run_tool(tools, "get_recovery_context")
        assert len(result["files_modified"]) == 2
        assert "src/mcp/server.py" in result["files_modified"]
        assert "tests/test_recovery.py" in result["files_modified"]

    def test_recovers_decisions(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        _record_decision(
            tools,
            "Use RecoveryContext.from_task_memory",
            reasoning="Model already has the factory method",
            alternatives=["Build context manually", "New model class"],
            context="RecoveryContext was designed in CMH-002",
        )

        result = _run_tool(tools, "get_recovery_context")
        assert len(result["key_decisions"]) == 1
        d = result["key_decisions"][0]
        assert d["decision"] == "Use RecoveryContext.from_task_memory"
        assert "Model already has" in d["reasoning"]
        assert len(d["alternatives"]) == 2

    def test_recovers_active_branch(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        _record_branch(tools, "feature/CMH-011-recovery", "created",
                       base_branch="main")

        result = _run_tool(tools, "get_recovery_context")
        assert result["active_branch"] == "feature/CMH-011-recovery"

    def test_recovers_branch_after_multiple_actions(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        _record_branch(tools, "feature/CMH-011-recovery", "created", "main")
        _record_branch(tools, "feature/CMH-011-recovery", "pushed")

        result = _run_tool(tools, "get_recovery_context")
        assert result["active_branch"] == "feature/CMH-011-recovery"


# ---------------------------------------------------------------------------
# Recent step count parameter
# ---------------------------------------------------------------------------

class TestGetRecoveryContextRecentStepCount:
    """Tests for the configurable recent_step_count parameter."""

    def test_limits_recent_steps(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        for i in range(15):
            _record_step(tools, f"Step {i + 1}")

        result = _run_tool(tools, "get_recovery_context", {
            "recent_step_count": 5
        })
        assert len(result["recent_steps"]) == 5
        assert result["total_steps_completed"] == 15

    def test_default_recent_step_count_is_10(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        for i in range(15):
            _record_step(tools, f"Step {i + 1}")

        result = _run_tool(tools, "get_recovery_context")
        assert len(result["recent_steps"]) == 10

    def test_returns_all_if_fewer_than_limit(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        _record_step(tools, "Step 1")
        _record_step(tools, "Step 2")

        result = _run_tool(tools, "get_recovery_context", {
            "recent_step_count": 10
        })
        assert len(result["recent_steps"]) == 2

    def test_clamps_minimum_to_1(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        _record_step(tools, "Step 1")
        _record_step(tools, "Step 2")

        result = _run_tool(tools, "get_recovery_context", {
            "recent_step_count": 0
        })
        assert len(result["recent_steps"]) >= 1

    def test_clamps_maximum_to_50(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        for i in range(60):
            _record_step(tools, f"Step {i + 1}")

        result = _run_tool(tools, "get_recovery_context", {
            "recent_step_count": 100
        })
        assert len(result["recent_steps"]) == 50


# ---------------------------------------------------------------------------
# Recovery prompt
# ---------------------------------------------------------------------------

class TestGetRecoveryContextPrompt:
    """Tests for the formatted recovery prompt."""

    def test_includes_prompt_by_default(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool", phase="phase-3")
        result = _run_tool(tools, "get_recovery_context")
        assert "recovery_prompt" in result
        assert isinstance(result["recovery_prompt"], str)
        assert len(result["recovery_prompt"]) > 0

    def test_prompt_contains_ticket_id(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        result = _run_tool(tools, "get_recovery_context")
        assert "CMH-011" in result["recovery_prompt"]

    def test_prompt_contains_title(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        result = _run_tool(tools, "get_recovery_context")
        assert "Recovery context tool" in result["recovery_prompt"]

    def test_prompt_contains_phase(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool", phase="phase-3")
        result = _run_tool(tools, "get_recovery_context")
        assert "phase-3" in result["recovery_prompt"]

    def test_prompt_contains_files(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        _record_file(tools, "src/mcp/server.py", "modified")
        result = _run_tool(tools, "get_recovery_context")
        assert "src/mcp/server.py" in result["recovery_prompt"]

    def test_prompt_contains_decisions(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        _record_decision(tools, "Use Pydantic for validation",
                        reasoning="Consistent with existing patterns")
        result = _run_tool(tools, "get_recovery_context")
        assert "Use Pydantic for validation" in result["recovery_prompt"]

    def test_prompt_contains_branch(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        _record_branch(tools, "feature/CMH-011-recovery", "created", "main")
        result = _run_tool(tools, "get_recovery_context")
        assert "feature/CMH-011-recovery" in result["recovery_prompt"]

    def test_prompt_contains_last_step(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        _record_step(tools, "Implemented get_recovery_context tool")
        result = _run_tool(tools, "get_recovery_context")
        assert "Implemented get_recovery_context tool" in result["recovery_prompt"]

    def test_exclude_prompt_when_false(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        result = _run_tool(tools, "get_recovery_context", {
            "include_prompt": False
        })
        assert "recovery_prompt" not in result


# ---------------------------------------------------------------------------
# Completed task -- auto-detection (most recently completed)
# ---------------------------------------------------------------------------

class TestGetRecoveryContextCompletedTask:
    """Tests for recovery from a completed (but not active) task."""

    def test_recovers_most_recently_completed_task(self, tools):
        _start_task(tools, "CMH-009", "Task lifecycle management")
        _record_step(tools, "Implemented start_task")
        _complete_task(tools, "All lifecycle tools done")

        result = _run_tool(tools, "get_recovery_context")
        assert result["error"] is False
        assert result["has_context"] is True
        assert result["ticket_id"] == "CMH-009"
        assert result["source"] == "most_recent_completed"
        assert result["status"] == "completed"

    def test_prefers_active_over_completed(self, tools):
        # Complete one task, then start another.
        _start_task(tools, "CMH-009", "Task lifecycle management")
        _complete_task(tools, "Done")

        _start_task(tools, "CMH-011", "Recovery context tool")

        result = _run_tool(tools, "get_recovery_context")
        assert result["ticket_id"] == "CMH-011"
        assert result["source"] == "active"

    def test_recovers_last_completed_when_multiple(self, tools):
        # Complete two tasks.
        _start_task(tools, "CMH-009", "Task lifecycle")
        _complete_task(tools, "Done 1")

        _start_task(tools, "CMH-010", "Markdown summary")
        _record_step(tools, "Generated summaries")
        _complete_task(tools, "Done 2")

        result = _run_tool(tools, "get_recovery_context")
        assert result["ticket_id"] == "CMH-010"
        assert result["source"] == "most_recent_completed"
        assert result["total_steps_completed"] == 1


# ---------------------------------------------------------------------------
# Explicit ticket ID lookup
# ---------------------------------------------------------------------------

class TestGetRecoveryContextExplicitTicket:
    """Tests for recovering context for a specific ticket by ID."""

    def test_recovers_specific_active_task(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        result = _run_tool(tools, "get_recovery_context", {
            "ticket_id": "CMH-011"
        })
        assert result["ticket_id"] == "CMH-011"
        assert result["source"] == "explicit"

    def test_recovers_specific_completed_task(self, tools):
        _start_task(tools, "CMH-009", "Task lifecycle")
        _record_step(tools, "Implemented tools")
        _complete_task(tools, "All done")

        _start_task(tools, "CMH-011", "Recovery context tool")

        # Explicitly request the completed task, not the active one.
        result = _run_tool(tools, "get_recovery_context", {
            "ticket_id": "CMH-009"
        })
        assert result["ticket_id"] == "CMH-009"
        assert result["source"] == "explicit"
        assert result["status"] == "completed"
        assert result["total_steps_completed"] == 1

    def test_recovers_archived_task(self, project_dir):
        # Use a small window to force archival.
        config_dir = Path(project_dir) / ".claude-memory"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "config.json").write_text(json.dumps({
            "window_size": 1
        }))
        tools = _get_tools(project_dir)

        # Create and complete tasks to overflow the window.
        _start_task(tools, "CMH-001", "Task A")
        _record_step(tools, "Did A stuff")
        _complete_task(tools, "A done")

        _start_task(tools, "CMH-002", "Task B")
        _complete_task(tools, "B done")

        # CMH-001 should now be archived. Explicitly recover it.
        result = _run_tool(tools, "get_recovery_context", {
            "ticket_id": "CMH-001"
        })
        assert result["error"] is False
        assert result["has_context"] is True
        assert result["ticket_id"] == "CMH-001"
        assert result["source"] == "explicit"
        assert result["total_steps_completed"] == 1


# ---------------------------------------------------------------------------
# Full lifecycle integration
# ---------------------------------------------------------------------------

class TestGetRecoveryContextFullLifecycle:
    """Integration test: start task, record data, recover context."""

    def test_full_lifecycle_recovery(self, tools):
        # Start task with full context.
        _start_task(tools, "CMH-011", "get_recovery_context MCP tool",
                    phase="phase-3",
                    description="Implement recovery context tool")

        # Record steps.
        _record_step(tools, "Analyzed RecoveryContext model",
                     "Read models/recovery.py", "Read",
                     "Model has from_task_memory and format_for_prompt")
        _record_step(tools, "Added import to server.py",
                     "Imported RecoveryContext", "Edit",
                     "Import added successfully")
        _record_step(tools, "Implemented get_recovery_context tool",
                     "Added tool to _register_tools", "Write",
                     "140 lines of production code")
        _record_step(tools, "Created test file",
                     "tests/test_recovery_context.py", "Write",
                     "50 test cases")
        _record_step(tools, "Ran tests",
                     "All tests pass", "Bash",
                     "50 passed, 0 failed")

        # Record files.
        _record_file(tools, "src/claude_code_helper_mcp/mcp/server.py",
                     "modified", "Added get_recovery_context tool")
        _record_file(tools, "tests/test_recovery_context.py",
                     "created", "50 comprehensive tests")

        # Record decisions.
        _record_decision(
            tools,
            "Reuse RecoveryContext.from_task_memory",
            reasoning="The model already has the factory method; no need to duplicate logic",
            alternatives=["Build context manually in the tool function"],
            context="RecoveryContext was designed specifically for this use case in CMH-002",
        )
        _record_decision(
            tools,
            "Auto-detect active task when no ticket_id given",
            reasoning="Most common /clear scenario is during an active task",
            alternatives=["Always require explicit ticket_id"],
        )

        # Record branch.
        _record_branch(tools, "feature/CMH-011-get-recovery-context-mcp-tool",
                        "created", "main")
        _record_branch(tools, "feature/CMH-011-get-recovery-context-mcp-tool",
                        "pushed")

        # Now simulate /clear by calling recovery.
        result = _run_tool(tools, "get_recovery_context")

        # Verify full context was recovered.
        assert result["error"] is False
        assert result["has_context"] is True
        assert result["ticket_id"] == "CMH-011"
        assert result["title"] == "get_recovery_context MCP tool"
        assert result["phase"] == "phase-3"
        assert result["status"] == "active"
        assert result["source"] == "active"

        # Steps
        assert result["total_steps_completed"] == 5
        assert len(result["recent_steps"]) == 5
        assert result["last_step"]["action"] == "Ran tests"
        assert result["last_step"]["step_number"] == 5

        # Files
        assert len(result["files_modified"]) == 2
        assert "src/claude_code_helper_mcp/mcp/server.py" in result["files_modified"]

        # Decisions
        assert len(result["key_decisions"]) == 2
        assert result["key_decisions"][0]["decision"] == "Reuse RecoveryContext.from_task_memory"

        # Branch
        assert result["active_branch"] == "feature/CMH-011-get-recovery-context-mcp-tool"

        # Metadata
        assert result["metadata"]["description"] == "Implement recovery context tool"

        # Prompt
        assert "recovery_prompt" in result
        assert "CMH-011" in result["recovery_prompt"]
        assert "phase-3" in result["recovery_prompt"]
        assert "server.py" in result["recovery_prompt"]
        assert "Reuse RecoveryContext.from_task_memory" in result["recovery_prompt"]

        # Window state
        assert result["window_state"]["tasks_in_window"] == 1

    def test_recovery_after_task_completion(self, tools):
        """After completing a task, recovery returns the completed task."""
        _start_task(tools, "CMH-011", "Recovery tool", phase="phase-3")
        _record_step(tools, "Built everything")
        _record_file(tools, "server.py", "modified")
        _record_decision(tools, "Used existing model")
        _record_branch(tools, "feature/CMH-011", "created", "main")
        _complete_task(tools, "CMH-011 complete. 50 tests pass.")

        result = _run_tool(tools, "get_recovery_context")
        assert result["error"] is False
        assert result["has_context"] is True
        assert result["ticket_id"] == "CMH-011"
        assert result["status"] == "completed"
        assert result["source"] == "most_recent_completed"
        assert result["total_steps_completed"] == 1
        assert result["summary_so_far"] == "CMH-011 complete. 50 tests pass."


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestGetRecoveryContextEdgeCases:
    """Edge case tests for robustness."""

    def test_task_with_no_phase(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        result = _run_tool(tools, "get_recovery_context")
        assert result["phase"] is None

    def test_task_with_empty_summary(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        result = _run_tool(tools, "get_recovery_context")
        assert result["summary_so_far"] == ""

    def test_task_with_many_steps(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        for i in range(25):
            _record_step(tools, f"Step {i + 1}")

        result = _run_tool(tools, "get_recovery_context")
        assert result["total_steps_completed"] == 25
        # Default is 10 recent steps
        assert len(result["recent_steps"]) == 10

    def test_task_with_many_files(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        for i in range(10):
            _record_file(tools, f"src/module_{i}.py", "created")

        result = _run_tool(tools, "get_recovery_context")
        assert len(result["files_modified"]) == 10

    def test_task_with_many_decisions(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        for i in range(5):
            _record_decision(tools, f"Decision {i + 1}")

        result = _run_tool(tools, "get_recovery_context")
        assert len(result["key_decisions"]) == 5

    def test_multiple_branches_returns_last(self, tools):
        _start_task(tools, "CMH-011", "Recovery context tool")
        _record_branch(tools, "feature/CMH-011", "created", "main")
        _record_branch(tools, "fix/CMH-011-hotfix", "created", "main")

        result = _run_tool(tools, "get_recovery_context")
        assert result["active_branch"] == "fix/CMH-011-hotfix"

    def test_file_dedup_in_recovery(self, tools):
        """Multiple actions on the same file should produce one entry."""
        _start_task(tools, "CMH-011", "Recovery context tool")
        _record_file(tools, "server.py", "created")
        _record_file(tools, "server.py", "modified")

        result = _run_tool(tools, "get_recovery_context")
        assert result["files_modified"].count("server.py") == 1
