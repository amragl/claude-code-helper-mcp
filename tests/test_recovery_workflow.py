"""Tests for the post-/clear recovery workflow (CMH-017).

Tests cover the RecoveryWorkflow class, the ``memory recover`` CLI command,
clear event detection, recovery prompt generation, pipeline state enrichment,
recovery marker management, and edge cases.

All tests use real file I/O via temporary directories -- no mocks.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pytest
from click.testing import CliRunner

from claude_code_helper_mcp.cli.main import cli
from claude_code_helper_mcp.config import MemoryConfig
from claude_code_helper_mcp.hooks.recovery import (
    RECOVERY_COOLDOWN_SECONDS,
    RECOVERY_MARKER_FILE,
    RecoveryWorkflow,
)
from claude_code_helper_mcp.models.records import BranchAction, FileAction
from claude_code_helper_mcp.models.recovery import RecoveryContext
from claude_code_helper_mcp.models.task import TaskMemory, TaskStatus
from claude_code_helper_mcp.storage.store import MemoryStore
from claude_code_helper_mcp.storage.window_manager import WindowManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_active_task(storage_path: str) -> TaskMemory:
    """Create an active task with steps, files, branches, and decisions."""
    wm = WindowManager(storage_path=storage_path)
    task = wm.start_new_task(
        ticket_id="CMH-017",
        title="Post-clear recovery workflow",
        phase="phase-4",
    )
    task.add_step(
        action="Implemented RecoveryWorkflow class",
        description="Created the main recovery orchestrator",
        tool_used="Write",
        result_summary="recovery.py created with 400 lines",
    )
    task.add_step(
        action="Added CLI recover command",
        description="Integrated recover command into Click CLI",
        tool_used="Edit",
        result_summary="cli/main.py updated with recover subcommand",
    )
    task.record_file(
        path="src/hooks/recovery.py",
        action=FileAction.CREATED,
        description="New recovery workflow module",
    )
    task.record_file(
        path="src/cli/main.py",
        action=FileAction.MODIFIED,
        description="Added recover CLI command",
    )
    task.record_branch(
        branch_name="feature/CMH-017-post-clear-recovery-workflow",
        action=BranchAction.CREATED,
        base_branch="main",
    )
    task.add_decision(
        decision="Use marker file for clear detection",
        reasoning="Simple file-based approach avoids session state dependency",
        alternatives=["Environment variable", "In-memory flag"],
        context="Recovery must work across session boundaries",
    )
    task.next_steps = ["Write tests", "Run validation"]
    task.summary = "Core recovery workflow implemented"
    wm.save_current_task()
    return task


def _make_workflow(project_root: str, storage_path: str) -> RecoveryWorkflow:
    """Create a fresh RecoveryWorkflow that reads latest state from disk."""
    return RecoveryWorkflow(
        project_root=project_root,
        storage_path=storage_path,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_project(tmp_path):
    """Create a temporary project directory with .claude-memory/ and .agent-forge/ structure."""
    project_root = tmp_path / "test-project"
    project_root.mkdir()

    storage_dir = project_root / ".claude-memory"
    storage_dir.mkdir()
    (storage_dir / "tasks").mkdir()

    agent_forge = project_root / ".agent-forge"
    agent_forge.mkdir()
    (agent_forge / "state").mkdir(parents=True)
    (agent_forge / "plans").mkdir(parents=True)

    project_json = {
        "project_name": "test-project",
        "repo_owner": "test",
        "repo_name": "test-project",
        "repo_path": str(project_root),
    }
    (agent_forge / "project.json").write_text(
        json.dumps(project_json, indent=2)
    )

    return project_root


@pytest.fixture
def storage_path(tmp_project):
    """Return the storage path for the temporary project."""
    return str(tmp_project / ".claude-memory")


@pytest.fixture
def pipeline_state(tmp_project):
    """Write a pipeline.json state file."""
    pipeline = {
        "status": "running",
        "current_phase": "phase-4",
        "current_ticket": "CMH-017",
        "current_agent": "build",
        "current_step": "build",
        "last_completed_step": "plan",
        "failed_step": None,
        "failure_reason": None,
        "blocked_reason": None,
        "last_run": datetime.now(timezone.utc).isoformat(),
        "step_tracking": {
            "step_order": ["plan", "build", "validate", "test", "monitor"],
            "current_step": "build",
            "steps_completed": ["plan"],
            "steps_remaining": ["build", "validate", "test", "monitor"],
        },
    }
    state_path = tmp_project / ".agent-forge" / "state" / "pipeline.json"
    state_path.write_text(json.dumps(pipeline, indent=2))
    return pipeline


@pytest.fixture
def build_output(tmp_project):
    """Write a build-output.json state file."""
    build = {
        "status": "success",
        "branch": "feature/CMH-017-post-clear-recovery-workflow",
        "pr_number": 44,
        "files_changed": ["src/hooks/recovery.py", "src/cli/main.py"],
        "ticket_id": "CMH-017",
        "summary": "Recovery workflow implemented",
    }
    state_path = tmp_project / ".agent-forge" / "state" / "build-output.json"
    state_path.write_text(json.dumps(build, indent=2))
    return build


@pytest.fixture
def backlog_state(tmp_project):
    """Write a backlog.json file."""
    backlog = {
        "project": "claude-code-helper-mcp",
        "phases": [
            {
                "id": "phase-4",
                "name": "Agent Forge Integration",
                "description": "Integration with Agent Forge pipeline",
                "milestone": "Full pipeline integration",
                "status": "in-progress",
                "tickets": ["CMH-015", "CMH-016", "CMH-017", "CMH-018"],
            }
        ],
        "tickets": [
            {
                "id": "CMH-017",
                "github_issue": 17,
                "title": "Post-clear recovery workflow",
                "description": "Implement complete post-/clear recovery workflow",
                "phase": "phase-4",
                "priority": "critical",
                "type": "feature",
                "status": "in-progress",
                "dependencies": ["CMH-011", "CMH-016"],
                "assigned_agent": "build",
                "estimated_complexity": "large",
                "pr_number": None,
                "completed_at": None,
            }
        ],
    }
    backlog_path = tmp_project / ".agent-forge" / "plans" / "backlog.json"
    backlog_path.write_text(json.dumps(backlog, indent=2))
    return backlog


# ---------------------------------------------------------------------------
# RecoveryWorkflow -- Initialization
# ---------------------------------------------------------------------------


class TestRecoveryWorkflowInit:
    """Test RecoveryWorkflow initialization."""

    def test_init_with_project_root(self, tmp_project, storage_path):
        wf = RecoveryWorkflow(project_root=str(tmp_project), storage_path=storage_path)
        assert wf.project_root == tmp_project.resolve()
        assert wf.window_manager is not None
        assert wf.state_reader is not None

    def test_init_with_storage_path_only(self, storage_path):
        wf = RecoveryWorkflow(storage_path=storage_path)
        assert wf.config.storage_path == storage_path

    def test_properties_accessible(self, tmp_project, storage_path):
        wf = _make_workflow(str(tmp_project), storage_path)
        assert wf.project_root is not None
        assert wf.config is not None
        assert wf.window_manager is not None
        assert wf.state_reader is not None


# ---------------------------------------------------------------------------
# Clear Event Detection
# ---------------------------------------------------------------------------


class TestClearEventDetection:
    """Test clear event detection logic."""

    def test_no_active_task_returns_false(self, tmp_project, storage_path):
        wf = _make_workflow(str(tmp_project), storage_path)
        assert wf.detect_clear_event() is False

    def test_active_task_returns_true(self, tmp_project, storage_path):
        _create_active_task(storage_path)
        wf = _make_workflow(str(tmp_project), storage_path)
        assert wf.detect_clear_event() is True

    def test_recent_recovery_suppresses_detection(self, tmp_project, storage_path):
        _create_active_task(storage_path)
        marker_path = Path(storage_path) / RECOVERY_MARKER_FILE
        marker = {
            "recovered_at": datetime.now(timezone.utc).isoformat(),
            "ticket_id": "CMH-017",
            "source": "active",
        }
        marker_path.write_text(json.dumps(marker, indent=2))

        wf = _make_workflow(str(tmp_project), storage_path)
        assert wf.detect_clear_event() is False

    def test_old_recovery_marker_does_not_suppress(self, tmp_project, storage_path):
        _create_active_task(storage_path)
        marker_path = Path(storage_path) / RECOVERY_MARKER_FILE
        old_time = datetime.now(timezone.utc) - timedelta(
            seconds=RECOVERY_COOLDOWN_SECONDS + 10
        )
        marker = {
            "recovered_at": old_time.isoformat(),
            "ticket_id": "CMH-017",
            "source": "active",
        }
        marker_path.write_text(json.dumps(marker, indent=2))

        wf = _make_workflow(str(tmp_project), storage_path)
        assert wf.detect_clear_event() is True

    def test_corrupt_recovery_marker_does_not_suppress(self, tmp_project, storage_path):
        _create_active_task(storage_path)
        marker_path = Path(storage_path) / RECOVERY_MARKER_FILE
        marker_path.write_text("not valid json {{{{")

        wf = _make_workflow(str(tmp_project), storage_path)
        assert wf.detect_clear_event() is True

    def test_completed_task_no_active_returns_false(self, tmp_project, storage_path):
        wm = WindowManager(storage_path=storage_path)
        wm.start_new_task(ticket_id="CMH-016", title="State readers", phase="phase-4")
        wm.complete_current_task("Done!")

        wf = _make_workflow(str(tmp_project), storage_path)
        assert wf.detect_clear_event() is False


# ---------------------------------------------------------------------------
# Recovery Execution
# ---------------------------------------------------------------------------


class TestRecoveryExecution:
    """Test the full recovery workflow."""

    def test_recover_active_task(self, tmp_project, storage_path):
        _create_active_task(storage_path)
        wf = _make_workflow(str(tmp_project), storage_path)
        result = wf.recover()

        assert result["status"] == "recovered"
        assert result["ticket_id"] == "CMH-017"
        assert result["title"] == "Post-clear recovery workflow"
        assert result["source"] == "active"
        assert "recovery_context" in result
        assert "recovery_prompt" in result
        assert "timestamp" in result

    def test_recover_explicit_ticket_id(self, tmp_project, storage_path):
        _create_active_task(storage_path)
        wf = _make_workflow(str(tmp_project), storage_path)
        result = wf.recover(ticket_id="CMH-017")

        assert result["status"] == "recovered"
        assert result["ticket_id"] == "CMH-017"
        assert result["source"] == "explicit"

    def test_recover_nonexistent_ticket(self, tmp_project, storage_path):
        _create_active_task(storage_path)
        wf = _make_workflow(str(tmp_project), storage_path)
        result = wf.recover(ticket_id="CMH-999")

        assert result["status"] == "no_context"
        assert "not found" in result["message"]

    def test_recover_no_tasks_at_all(self, tmp_project, storage_path):
        wf = _make_workflow(str(tmp_project), storage_path)
        result = wf.recover()

        assert result["status"] == "no_context"
        assert "No active or recently completed" in result["message"]

    def test_recover_most_recent_completed(self, tmp_project, storage_path):
        wm = WindowManager(storage_path=storage_path)
        task = wm.start_new_task(
            ticket_id="CMH-016", title="Pipeline state readers", phase="phase-4"
        )
        task.add_step(action="Implemented StateReader", description="Created the reader")
        wm.complete_current_task("All done!")

        wf = _make_workflow(str(tmp_project), storage_path)
        result = wf.recover()

        assert result["status"] == "recovered"
        assert result["ticket_id"] == "CMH-016"
        assert result["source"] == "most_recent_completed"

    def test_recovery_context_contents(self, tmp_project, storage_path):
        _create_active_task(storage_path)
        wf = _make_workflow(str(tmp_project), storage_path)
        result = wf.recover()
        rc = result["recovery_context"]

        assert rc["ticket_id"] == "CMH-017"
        assert rc["title"] == "Post-clear recovery workflow"
        assert rc["phase"] == "phase-4"
        assert rc["status"] == "active"
        assert rc["total_steps_completed"] == 2
        assert len(rc["files_modified"]) == 2
        assert len(rc["key_decisions"]) == 1
        assert len(rc["recent_steps"]) == 2
        assert len(rc["next_steps"]) == 2

    def test_recovery_writes_marker(self, tmp_project, storage_path):
        _create_active_task(storage_path)
        marker_path = Path(storage_path) / RECOVERY_MARKER_FILE
        assert not marker_path.exists()

        wf = _make_workflow(str(tmp_project), storage_path)
        wf.recover()

        assert marker_path.exists()
        marker = json.loads(marker_path.read_text())
        assert marker["ticket_id"] == "CMH-017"
        assert marker["source"] == "active"
        assert "recovered_at" in marker

    def test_recovery_prompt_contains_key_sections(self, tmp_project, storage_path):
        _create_active_task(storage_path)
        wf = _make_workflow(str(tmp_project), storage_path)
        result = wf.recover()
        prompt = result["recovery_prompt"]

        assert "Recovery Context for CMH-017" in prompt
        assert "Post-clear recovery workflow" in prompt
        assert "Task Overview" in prompt
        assert "phase-4" in prompt
        assert "Files Modified" in prompt
        assert "src/hooks/recovery.py" in prompt
        assert "Key Decisions" in prompt
        assert "marker file" in prompt
        assert "Recent Steps" in prompt
        assert "Implemented RecoveryWorkflow" in prompt
        assert "Planned Next Steps" in prompt
        assert "Write tests" in prompt
        assert "How to Resume" in prompt

    def test_recovery_with_pipeline_context(
        self, tmp_project, storage_path, pipeline_state, build_output, backlog_state
    ):
        _create_active_task(storage_path)
        wf = _make_workflow(str(tmp_project), storage_path)
        result = wf.recover(include_pipeline_context=True)

        assert result["status"] == "recovered"
        assert "pipeline_context" in result
        pc = result["pipeline_context"]
        assert pc["pipeline_status"] == "running"
        assert pc["steps_completed"] == ["plan"]
        assert "build" in pc["steps_remaining"]

    def test_recovery_without_pipeline_context(self, tmp_project, storage_path):
        _create_active_task(storage_path)
        wf = _make_workflow(str(tmp_project), storage_path)
        result = wf.recover(include_pipeline_context=False)

        assert result["status"] == "recovered"
        assert "pipeline_context" not in result

    def test_recovery_step_count_clamping(self, tmp_project, storage_path):
        _create_active_task(storage_path)
        wf = _make_workflow(str(tmp_project), storage_path)

        result_low = wf.recover(recent_step_count=-5)
        assert result_low["status"] == "recovered"

        result_high = wf.recover(recent_step_count=100)
        assert result_high["status"] == "recovered"


# ---------------------------------------------------------------------------
# Pipeline Enrichment
# ---------------------------------------------------------------------------


class TestPipelineEnrichment:
    """Test pipeline state enrichment."""

    def test_enrichment_with_matching_ticket(
        self, tmp_project, storage_path, pipeline_state, build_output, backlog_state
    ):
        _create_active_task(storage_path)
        wf = _make_workflow(str(tmp_project), storage_path)
        result = wf.recover()
        pc = result.get("pipeline_context", {})

        assert pc.get("matched_ticket") is True
        assert pc.get("branch") == "feature/CMH-017-post-clear-recovery-workflow"
        assert pc.get("pr_number") == 44
        assert pc.get("build_status") == "success"

    def test_enrichment_without_agent_forge(self, tmp_project, storage_path):
        _create_active_task(storage_path)
        wf = _make_workflow(str(tmp_project), storage_path)
        result = wf.recover()
        assert result.get("pipeline_context") is None

    def test_enrichment_with_blocked_pipeline(
        self, tmp_project, storage_path, backlog_state
    ):
        _create_active_task(storage_path)
        pipeline = {
            "status": "blocked",
            "current_phase": "phase-4",
            "current_ticket": "CMH-017",
            "current_agent": None,
            "current_step": "validate",
            "last_completed_step": "build",
            "failed_step": "validate",
            "failure_reason": "VALIDATE_CRITICAL_ISSUES: 2 critical issues found.",
            "blocked_reason": "Pipeline blocked at validate step.",
            "step_tracking": {
                "steps_completed": ["plan", "build"],
                "steps_remaining": ["validate", "test", "monitor"],
            },
        }
        state_path = tmp_project / ".agent-forge" / "state" / "pipeline.json"
        state_path.write_text(json.dumps(pipeline, indent=2))

        wf = _make_workflow(str(tmp_project), storage_path)
        result = wf.recover()
        pc = result.get("pipeline_context", {})

        assert pc.get("pipeline_status") == "blocked"
        assert pc.get("blocked_reason") is not None
        assert pc.get("failed_step") == "validate"


# ---------------------------------------------------------------------------
# Recovery Prompt Generation
# ---------------------------------------------------------------------------


class TestRecoveryPromptGeneration:
    """Test recovery prompt generation."""

    def test_generate_recovery_prompt_convenience_method(self, tmp_project, storage_path):
        _create_active_task(storage_path)
        wf = _make_workflow(str(tmp_project), storage_path)
        prompt = wf.generate_recovery_prompt()
        assert "Recovery Context for CMH-017" in prompt
        assert len(prompt) > 100

    def test_generate_recovery_prompt_for_missing_task(self, tmp_project, storage_path):
        wf = _make_workflow(str(tmp_project), storage_path)
        prompt = wf.generate_recovery_prompt(ticket_id="CMH-999")
        assert "[Recovery unavailable:" in prompt

    def test_prompt_includes_branch_info(self, tmp_project, storage_path):
        _create_active_task(storage_path)
        wf = _make_workflow(str(tmp_project), storage_path)
        result = wf.recover()
        prompt = result["recovery_prompt"]
        assert "feature/CMH-017-post-clear-recovery-workflow" in prompt

    def test_prompt_includes_pipeline_sections(
        self, tmp_project, storage_path, pipeline_state, build_output, backlog_state
    ):
        _create_active_task(storage_path)
        wf = _make_workflow(str(tmp_project), storage_path)
        result = wf.recover()
        prompt = result["recovery_prompt"]
        assert "Pipeline State" in prompt
        assert "running" in prompt
        assert "#44" in prompt


# ---------------------------------------------------------------------------
# Recovery Marker Management
# ---------------------------------------------------------------------------


class TestRecoveryMarker:
    """Test recovery marker file management."""

    def test_clear_recovery_marker(self, tmp_project, storage_path):
        _create_active_task(storage_path)
        wf = _make_workflow(str(tmp_project), storage_path)
        wf.recover()
        marker_path = Path(storage_path) / RECOVERY_MARKER_FILE
        assert marker_path.exists()

        assert wf.clear_recovery_marker() is True
        assert not marker_path.exists()

    def test_clear_nonexistent_marker(self, tmp_project, storage_path):
        wf = _make_workflow(str(tmp_project), storage_path)
        assert wf.clear_recovery_marker() is False

    def test_marker_prevents_duplicate_detection(self, tmp_project, storage_path):
        _create_active_task(storage_path)
        wf = _make_workflow(str(tmp_project), storage_path)
        assert wf.detect_clear_event() is True

        wf.recover()
        # Re-create workflow to get fresh state.
        wf2 = _make_workflow(str(tmp_project), storage_path)
        assert wf2.detect_clear_event() is False


# ---------------------------------------------------------------------------
# Git Branch Detection
# ---------------------------------------------------------------------------


class TestGitBranchDetection:
    """Test git branch detection."""

    def test_detect_git_branch_returns_string_or_none(self, tmp_project, storage_path):
        wf = _make_workflow(str(tmp_project), storage_path)
        branch = wf._detect_git_branch()
        assert branch is None or isinstance(branch, str)

    def test_recovery_includes_git_branch(self, tmp_project, storage_path):
        _create_active_task(storage_path)
        wf = _make_workflow(str(tmp_project), storage_path)
        result = wf.recover()
        assert "git_branch" in result


# ---------------------------------------------------------------------------
# CLI -- memory recover
# ---------------------------------------------------------------------------


class TestRecoverCLI:
    """Test the ``memory recover`` CLI command."""

    def test_recover_no_tasks_exits_with_error(self, storage_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["--storage-path", storage_path, "recover"])
        assert result.exit_code == 1

    def test_recover_active_task_text_format(self, storage_path):
        _create_active_task(storage_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["--storage-path", storage_path, "recover"])
        assert result.exit_code == 0
        assert "Recovery Context" in result.output
        assert "CMH-017" in result.output

    def test_recover_json_format(self, storage_path):
        _create_active_task(storage_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--storage-path", storage_path, "recover", "--format", "json"]
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "recovered"
        assert data["ticket_id"] == "CMH-017"

    def test_recover_prompt_format(self, storage_path):
        _create_active_task(storage_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--storage-path", storage_path, "recover", "--format", "prompt"]
        )
        assert result.exit_code == 0
        assert "Recovery Context for CMH-017" in result.output
        assert "How to Resume" in result.output

    def test_recover_explicit_ticket(self, storage_path):
        _create_active_task(storage_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--storage-path", storage_path, "recover", "CMH-017", "--format", "json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["ticket_id"] == "CMH-017"
        assert data["source"] == "explicit"

    def test_recover_nonexistent_ticket_json(self, storage_path):
        _create_active_task(storage_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--storage-path", storage_path, "recover", "CMH-999", "--format", "json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["status"] == "no_context"

    def test_recover_no_pipeline_flag(self, storage_path):
        _create_active_task(storage_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--storage-path", storage_path, "recover", "--no-pipeline", "--format", "json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "pipeline_context" not in data

    def test_recover_steps_option(self, storage_path):
        _create_active_task(storage_path)
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["--storage-path", storage_path, "recover", "--steps", "1", "--format", "json"],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        rc = data["recovery_context"]
        assert len(rc["recent_steps"]) <= 1

    def test_detect_flag_no_active_task(self, storage_path):
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--storage-path", storage_path, "recover", "--detect"]
        )
        assert result.exit_code == 0
        assert "no" in result.output

    def test_detect_flag_with_active_task(self, storage_path):
        _create_active_task(storage_path)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["--storage-path", storage_path, "recover", "--detect"]
        )
        assert result.exit_code == 0
        assert "yes" in result.output


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_recover_after_task_completion(self, tmp_project, storage_path):
        wm = WindowManager(storage_path=storage_path)
        task = wm.start_new_task(
            ticket_id="CMH-016", title="Pipeline state readers", phase="phase-4"
        )
        task.add_step(action="Implemented StateReader", description="Full implementation")
        wm.complete_current_task("Done!")

        wf = _make_workflow(str(tmp_project), storage_path)
        result = wf.recover()
        assert result["status"] == "recovered"
        assert result["ticket_id"] == "CMH-016"
        assert result["source"] == "most_recent_completed"

    def test_recover_multiple_completed_tasks(self, tmp_project, storage_path):
        wm = WindowManager(storage_path=storage_path)
        wm.start_new_task(ticket_id="CMH-015", title="Hook integration", phase="phase-4")
        wm.complete_current_task("Done with 015!")

        wm.start_new_task(ticket_id="CMH-016", title="State readers", phase="phase-4")
        wm.complete_current_task("Done with 016!")

        wf = _make_workflow(str(tmp_project), storage_path)
        result = wf.recover()
        assert result["status"] == "recovered"
        assert result["ticket_id"] == "CMH-016"
        assert result["source"] == "most_recent_completed"

    def test_recover_active_preferred_over_completed(self, tmp_project, storage_path):
        wm = WindowManager(storage_path=storage_path)
        wm.start_new_task(ticket_id="CMH-015", title="Hook integration", phase="phase-4")
        wm.complete_current_task("Done!")

        wm.start_new_task(ticket_id="CMH-017", title="Recovery workflow", phase="phase-4")
        wm.save_current_task()

        wf = _make_workflow(str(tmp_project), storage_path)
        result = wf.recover()
        assert result["ticket_id"] == "CMH-017"
        assert result["source"] == "active"

    def test_recovery_prompt_with_empty_fields(self, tmp_project, storage_path):
        wm = WindowManager(storage_path=storage_path)
        wm.start_new_task(ticket_id="CMH-099", title="Minimal task")
        wm.save_current_task()

        wf = _make_workflow(str(tmp_project), storage_path)
        result = wf.recover()
        assert result["status"] == "recovered"
        prompt = result["recovery_prompt"]
        assert "CMH-099" in prompt
        assert "Minimal task" in prompt

    def test_recovery_workflow_with_missing_agent_forge(self, tmp_project, storage_path):
        import shutil
        agent_forge = tmp_project / ".agent-forge"
        if agent_forge.exists():
            shutil.rmtree(agent_forge)

        _create_active_task(storage_path)
        wf = _make_workflow(str(tmp_project), storage_path)
        result = wf.recover()
        assert result["status"] == "recovered"
        assert result.get("pipeline_context") is None

    def test_recovery_with_corrupt_pipeline_json(self, tmp_project, storage_path):
        _create_active_task(storage_path)
        state_path = tmp_project / ".agent-forge" / "state" / "pipeline.json"
        state_path.write_text("{{not valid json")

        wf = _make_workflow(str(tmp_project), storage_path)
        result = wf.recover()
        assert result["status"] == "recovered"
        assert result.get("pipeline_context") is None

    def test_recovery_context_serialization_roundtrip(self, tmp_project, storage_path):
        _create_active_task(storage_path)
        wf = _make_workflow(str(tmp_project), storage_path)
        result = wf.recover()
        rc_dict = result["recovery_context"]

        rc = RecoveryContext.from_json_dict(rc_dict)
        assert rc.ticket_id == "CMH-017"
        assert rc.total_steps_completed == 2
        assert len(rc.files_modified) == 2

    def test_concurrent_recovery_marker_writes(self, tmp_project, storage_path):
        _create_active_task(storage_path)
        wf = _make_workflow(str(tmp_project), storage_path)
        for _ in range(5):
            wf.recover()

        marker_path = Path(storage_path) / RECOVERY_MARKER_FILE
        assert marker_path.exists()
        marker = json.loads(marker_path.read_text())
        assert marker["ticket_id"] == "CMH-017"


# ---------------------------------------------------------------------------
# Integration: Pipeline + Memory Recovery
# ---------------------------------------------------------------------------


class TestPipelineMemoryIntegration:
    """Test integration between pipeline state and memory recovery."""

    def test_full_recovery_with_all_state_files(
        self, tmp_project, storage_path, pipeline_state, build_output, backlog_state
    ):
        _create_active_task(storage_path)
        wf = _make_workflow(str(tmp_project), storage_path)
        result = wf.recover()

        assert result["status"] == "recovered"
        assert result["ticket_id"] == "CMH-017"

        rc = result["recovery_context"]
        assert rc["total_steps_completed"] == 2
        assert len(rc["files_modified"]) == 2

        pc = result["pipeline_context"]
        assert pc["pipeline_status"] == "running"
        assert pc["matched_ticket"] is True
        assert pc["pr_number"] == 44
        assert pc["branch"] == "feature/CMH-017-post-clear-recovery-workflow"
        assert "plan" in pc["steps_completed"]

        prompt = result["recovery_prompt"]
        assert "Pipeline State" in prompt
        assert "#44" in prompt
        assert "src/hooks/recovery.py" in prompt

    def test_recovery_prompt_for_blocked_pipeline(
        self, tmp_project, storage_path, build_output, backlog_state
    ):
        _create_active_task(storage_path)
        pipeline = {
            "status": "blocked",
            "current_phase": "phase-4",
            "current_ticket": "CMH-017",
            "current_agent": None,
            "current_step": "test",
            "last_completed_step": "validate",
            "failed_step": "test",
            "failure_reason": "TEST_FAILURE: 3 test failures",
            "blocked_reason": "Pipeline blocked at test step. 3 failures.",
            "step_tracking": {
                "steps_completed": ["plan", "build", "validate"],
                "steps_remaining": ["test", "monitor"],
            },
        }
        state_path = tmp_project / ".agent-forge" / "state" / "pipeline.json"
        state_path.write_text(json.dumps(pipeline, indent=2))

        wf = _make_workflow(str(tmp_project), storage_path)
        result = wf.recover()
        prompt = result["recovery_prompt"]
        assert "BLOCKED" in prompt
        assert "test" in prompt.lower()
