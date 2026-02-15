"""Tests for the StateReader class (CMH-016: Pipeline state file readers).

Tests verify that the StateReader correctly reads, parses, and caches
Agent Forge state files (pipeline.json, build-output.json, backlog.json,
plan-output.json).  All tests use real filesystem I/O with temporary
directories -- no mocks, no stubs.

Test categories:
    TestStateReaderInit -- construction and path resolution
    TestReadPipeline -- reading pipeline.json
    TestReadBuildOutput -- reading build-output.json
    TestReadPlanOutput -- reading plan-output.json
    TestReadBacklog -- reading backlog.json
    TestBacklogQueries -- BacklogState query methods
    TestGetCurrentTicketContext -- the convenience aggregator
    TestCaching -- cache behaviour, TTL, invalidation
    TestGracefulDegradation -- missing files, malformed JSON, edge cases
    TestDataClasses -- data class construction and from_dict
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from claude_code_helper_mcp.hooks.state_reader import (
    AGENT_FORGE_DIR,
    BACKLOG_PATH,
    BUILD_OUTPUT_PATH,
    DEFAULT_CACHE_TTL_SECONDS,
    PIPELINE_STATE_PATH,
    PLAN_OUTPUT_PATH,
    BacklogPhase,
    BacklogState,
    BacklogTicket,
    BuildOutput,
    PipelineState,
    PlanOutput,
    StateReader,
    TicketContext,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def project_dir(tmp_path: Path) -> Path:
    """Create a temporary project directory with .agent-forge/ structure."""
    af_dir = tmp_path / AGENT_FORGE_DIR
    (af_dir / "state").mkdir(parents=True)
    (af_dir / "plans").mkdir(parents=True)
    return tmp_path


def _write_json(project_dir: Path, relative_path: str, data: dict) -> Path:
    """Helper to write a JSON file at the given relative path under .agent-forge/."""
    file_path = project_dir / AGENT_FORGE_DIR / relative_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2)
    return file_path


def _sample_pipeline_data(**overrides: object) -> dict:
    """Return a minimal valid pipeline.json structure."""
    data = {
        "status": "running",
        "current_phase": "phase-4",
        "current_ticket": "CMH-016",
        "current_agent": "build",
        "current_step": "build",
        "last_completed_step": "plan",
        "failed_step": None,
        "failure_reason": None,
        "blocked_reason": None,
        "last_run": "2026-02-15T15:00:00Z",
    }
    data.update(overrides)
    return data


def _sample_build_output(**overrides: object) -> dict:
    """Return a minimal valid build-output.json structure."""
    data = {
        "status": "success",
        "timestamp": "2026-02-15T15:05:00Z",
        "agent": "build",
        "ticket_id": "CMH-016",
        "branch": "feature/CMH-016-pipeline-state-file-readers",
        "pr_number": 43,
        "files_changed": [
            "src/claude_code_helper_mcp/hooks/state_reader.py",
            "tests/test_state_reader.py",
        ],
        "summary": "StateReader class implemented",
        "self_check": {
            "no_mocks": True,
            "no_placeholders": True,
        },
    }
    data.update(overrides)
    return data


def _sample_plan_output(**overrides: object) -> dict:
    """Return a minimal valid plan-output.json structure."""
    data = {
        "status": "success",
        "timestamp": "2026-02-15T15:00:30Z",
        "agent": "plan",
        "ticket_id": "CMH-016",
        "next_ticket": "CMH-016",
        "next_ticket_details": {
            "id": "CMH-016",
            "title": "Pipeline state file readers",
            "phase": "phase-4",
            "priority": "high",
        },
        "backlog_summary": {
            "total_tickets": 27,
            "done": 15,
            "in_progress": 1,
            "planned": 11,
        },
    }
    data.update(overrides)
    return data


def _sample_backlog_data(**overrides: object) -> dict:
    """Return a minimal valid backlog.json structure."""
    data = {
        "project": "claude-code-helper-mcp",
        "updated_at": "2026-02-15T15:00:00Z",
        "phases": [
            {
                "id": "phase-1",
                "name": "Foundation",
                "description": "Core infrastructure",
                "milestone": "M-1",
                "status": "completed",
                "tickets": ["CMH-001", "CMH-002"],
            },
            {
                "id": "phase-4",
                "name": "Agent Forge Integration",
                "description": "Hook integration, state readers, recovery",
                "milestone": "M-4",
                "status": "in-progress",
                "tickets": ["CMH-015", "CMH-016", "CMH-017", "CMH-018"],
            },
        ],
        "tickets": [
            {
                "id": "CMH-001",
                "github_issue": 1,
                "title": "Project initialization",
                "description": "Set up project structure",
                "phase": "phase-1",
                "priority": "critical",
                "type": "feature",
                "status": "done",
                "dependencies": [],
                "assigned_agent": "build",
                "estimated_complexity": "small",
                "pr_number": 28,
                "completed_at": "2026-02-14T18:20:00Z",
            },
            {
                "id": "CMH-002",
                "github_issue": 2,
                "title": "Memory data schema",
                "description": "Define Pydantic data models",
                "phase": "phase-1",
                "priority": "critical",
                "type": "feature",
                "status": "done",
                "dependencies": ["CMH-001"],
                "assigned_agent": "build",
                "estimated_complexity": "medium",
                "pr_number": 29,
                "completed_at": "2026-02-14T20:16:00Z",
            },
            {
                "id": "CMH-015",
                "github_issue": 15,
                "title": "Agent Forge hook integration",
                "description": "Implement pipeline hooks",
                "phase": "phase-4",
                "priority": "critical",
                "type": "feature",
                "status": "done",
                "dependencies": ["CMH-006", "CMH-007"],
                "assigned_agent": "build",
                "estimated_complexity": "large",
                "pr_number": 42,
                "completed_at": "2026-02-15T14:18:00Z",
            },
            {
                "id": "CMH-016",
                "github_issue": 16,
                "title": "Pipeline state file readers",
                "description": "Implement StateReader class",
                "phase": "phase-4",
                "priority": "high",
                "type": "feature",
                "status": "in-progress",
                "dependencies": ["CMH-005", "CMH-015"],
                "assigned_agent": "build",
                "estimated_complexity": "medium",
                "pr_number": None,
                "completed_at": None,
            },
            {
                "id": "CMH-017",
                "github_issue": 17,
                "title": "Post-clear recovery workflow",
                "description": "Implement recovery workflow",
                "phase": "phase-4",
                "priority": "critical",
                "type": "feature",
                "status": "planned",
                "dependencies": ["CMH-011", "CMH-016"],
                "assigned_agent": "build",
                "estimated_complexity": "large",
                "pr_number": None,
                "completed_at": None,
            },
        ],
    }
    data.update(overrides)
    return data


# ===========================================================================
# TestStateReaderInit
# ===========================================================================


class TestStateReaderInit:
    """Tests for StateReader construction and path resolution."""

    def test_init_with_valid_project_root(self, project_dir: Path) -> None:
        """StateReader initializes correctly with a valid project root."""
        reader = StateReader(str(project_dir))
        assert reader.project_root == project_dir
        assert reader.agent_forge_dir == project_dir / AGENT_FORGE_DIR

    def test_init_resolves_relative_path(self, project_dir: Path) -> None:
        """StateReader resolves relative paths to absolute."""
        # Create a reader with a relative path that resolves to project_dir.
        reader = StateReader(str(project_dir))
        assert reader.project_root.is_absolute()

    def test_init_with_missing_agent_forge_dir(self, tmp_path: Path) -> None:
        """StateReader initializes even without .agent-forge/ (logs warning)."""
        # tmp_path has no .agent-forge/ directory.
        reader = StateReader(str(tmp_path))
        assert reader.project_root == tmp_path
        # Reads should return None gracefully.
        assert reader.read_pipeline() is None

    def test_default_cache_ttl(self, project_dir: Path) -> None:
        """Default cache TTL matches the module constant."""
        reader = StateReader(str(project_dir))
        assert reader.cache_ttl == DEFAULT_CACHE_TTL_SECONDS

    def test_custom_cache_ttl(self, project_dir: Path) -> None:
        """Custom cache TTL is stored correctly."""
        reader = StateReader(str(project_dir), cache_ttl_seconds=60.0)
        assert reader.cache_ttl == 60.0

    def test_cache_ttl_setter(self, project_dir: Path) -> None:
        """Cache TTL can be updated via property setter."""
        reader = StateReader(str(project_dir))
        reader.cache_ttl = 120.0
        assert reader.cache_ttl == 120.0

    def test_cache_ttl_setter_rejects_negative(self, project_dir: Path) -> None:
        """Cache TTL setter rejects negative values."""
        reader = StateReader(str(project_dir))
        with pytest.raises(ValueError, match="non-negative"):
            reader.cache_ttl = -1.0

    def test_zero_cache_ttl_disables_caching(self, project_dir: Path) -> None:
        """With cache_ttl=0, every read goes to disk."""
        _write_json(project_dir, PIPELINE_STATE_PATH, _sample_pipeline_data())
        reader = StateReader(str(project_dir), cache_ttl_seconds=0)

        result1 = reader.read_pipeline()
        assert result1 is not None
        # Cache should never report as cached with TTL=0.
        assert not reader.is_cached(PIPELINE_STATE_PATH)


# ===========================================================================
# TestReadPipeline
# ===========================================================================


class TestReadPipeline:
    """Tests for reading pipeline.json."""

    def test_read_valid_pipeline(self, project_dir: Path) -> None:
        """Reads and parses a valid pipeline.json correctly."""
        data = _sample_pipeline_data()
        _write_json(project_dir, PIPELINE_STATE_PATH, data)
        reader = StateReader(str(project_dir))

        result = reader.read_pipeline()
        assert result is not None
        assert result.status == "running"
        assert result.current_phase == "phase-4"
        assert result.current_ticket == "CMH-016"
        assert result.current_agent == "build"
        assert result.current_step == "build"
        assert result.last_completed_step == "plan"
        assert result.failed_step is None
        assert result.failure_reason is None
        assert result.blocked_reason is None
        assert result.last_run == "2026-02-15T15:00:00Z"

    def test_read_pipeline_idle(self, project_dir: Path) -> None:
        """Reads an idle pipeline correctly (null fields)."""
        data = _sample_pipeline_data(
            status="idle",
            current_phase=None,
            current_ticket=None,
            current_agent=None,
            current_step=None,
            last_completed_step=None,
        )
        _write_json(project_dir, PIPELINE_STATE_PATH, data)
        reader = StateReader(str(project_dir))

        result = reader.read_pipeline()
        assert result is not None
        assert result.status == "idle"
        assert result.current_ticket is None
        assert result.current_step is None

    def test_read_pipeline_blocked(self, project_dir: Path) -> None:
        """Reads a blocked pipeline with failure info."""
        data = _sample_pipeline_data(
            status="blocked",
            failed_step="validate",
            failure_reason="Critical issue found",
            blocked_reason="Validation failed",
        )
        _write_json(project_dir, PIPELINE_STATE_PATH, data)
        reader = StateReader(str(project_dir))

        result = reader.read_pipeline()
        assert result is not None
        assert result.status == "blocked"
        assert result.failed_step == "validate"
        assert result.failure_reason == "Critical issue found"
        assert result.blocked_reason == "Validation failed"

    def test_read_pipeline_preserves_raw(self, project_dir: Path) -> None:
        """The raw dict is preserved for advanced consumers."""
        data = _sample_pipeline_data()
        data["custom_field"] = "custom_value"
        _write_json(project_dir, PIPELINE_STATE_PATH, data)
        reader = StateReader(str(project_dir))

        result = reader.read_pipeline()
        assert result is not None
        assert result.raw["custom_field"] == "custom_value"
        assert result.raw["status"] == "running"

    def test_read_missing_pipeline_returns_none(self, project_dir: Path) -> None:
        """Returns None when pipeline.json does not exist."""
        reader = StateReader(str(project_dir))
        result = reader.read_pipeline()
        assert result is None

    def test_read_pipeline_with_extra_fields(self, project_dir: Path) -> None:
        """Extra fields in pipeline.json do not cause errors."""
        data = _sample_pipeline_data()
        data["step_tracking"] = {"step_order": ["plan", "build"]}
        data["agents"] = {"plan": {"status": "idle"}}
        _write_json(project_dir, PIPELINE_STATE_PATH, data)
        reader = StateReader(str(project_dir))

        result = reader.read_pipeline()
        assert result is not None
        assert result.status == "running"
        assert "step_tracking" in result.raw

    def test_read_pipeline_with_minimal_fields(self, project_dir: Path) -> None:
        """A pipeline.json with only status field still parses."""
        _write_json(project_dir, PIPELINE_STATE_PATH, {"status": "idle"})
        reader = StateReader(str(project_dir))

        result = reader.read_pipeline()
        assert result is not None
        assert result.status == "idle"
        assert result.current_ticket is None


# ===========================================================================
# TestReadBuildOutput
# ===========================================================================


class TestReadBuildOutput:
    """Tests for reading build-output.json."""

    def test_read_valid_build_output(self, project_dir: Path) -> None:
        """Reads and parses a valid build-output.json correctly."""
        data = _sample_build_output()
        _write_json(project_dir, BUILD_OUTPUT_PATH, data)
        reader = StateReader(str(project_dir))

        result = reader.read_build_output()
        assert result is not None
        assert result.status == "success"
        assert result.branch == "feature/CMH-016-pipeline-state-file-readers"
        assert result.pr_number == 43
        assert len(result.files_changed) == 2
        assert "src/claude_code_helper_mcp/hooks/state_reader.py" in result.files_changed
        assert result.summary == "StateReader class implemented"
        assert result.ticket_id == "CMH-016"
        assert result.self_check["no_mocks"] is True

    def test_read_build_output_error_status(self, project_dir: Path) -> None:
        """Reads a build output with error status."""
        data = _sample_build_output(
            status="error",
            branch=None,
            pr_number=None,
            files_changed=[],
        )
        _write_json(project_dir, BUILD_OUTPUT_PATH, data)
        reader = StateReader(str(project_dir))

        result = reader.read_build_output()
        assert result is not None
        assert result.status == "error"
        assert result.branch is None
        assert result.pr_number is None

    def test_read_build_output_missing_files_changed(self, project_dir: Path) -> None:
        """Handles missing files_changed field gracefully."""
        data = _sample_build_output()
        del data["files_changed"]
        _write_json(project_dir, BUILD_OUTPUT_PATH, data)
        reader = StateReader(str(project_dir))

        result = reader.read_build_output()
        assert result is not None
        assert result.files_changed == []

    def test_read_build_output_invalid_files_changed_type(self, project_dir: Path) -> None:
        """Handles non-list files_changed value gracefully."""
        data = _sample_build_output(files_changed="not-a-list")
        _write_json(project_dir, BUILD_OUTPUT_PATH, data)
        reader = StateReader(str(project_dir))

        result = reader.read_build_output()
        assert result is not None
        assert result.files_changed == []

    def test_read_missing_build_output_returns_none(self, project_dir: Path) -> None:
        """Returns None when build-output.json does not exist."""
        reader = StateReader(str(project_dir))
        result = reader.read_build_output()
        assert result is None

    def test_read_build_output_preserves_raw(self, project_dir: Path) -> None:
        """The raw dict is preserved."""
        data = _sample_build_output()
        _write_json(project_dir, BUILD_OUTPUT_PATH, data)
        reader = StateReader(str(project_dir))

        result = reader.read_build_output()
        assert result is not None
        assert result.raw["pr_number"] == 43


# ===========================================================================
# TestReadPlanOutput
# ===========================================================================


class TestReadPlanOutput:
    """Tests for reading plan-output.json."""

    def test_read_valid_plan_output(self, project_dir: Path) -> None:
        """Reads and parses a valid plan-output.json correctly."""
        data = _sample_plan_output()
        _write_json(project_dir, PLAN_OUTPUT_PATH, data)
        reader = StateReader(str(project_dir))

        result = reader.read_plan_output()
        assert result is not None
        assert result.status == "success"
        assert result.next_ticket == "CMH-016"
        assert result.ticket_id == "CMH-016"
        assert result.next_ticket_details is not None
        assert result.next_ticket_details["id"] == "CMH-016"
        assert result.backlog_summary is not None
        assert result.backlog_summary["total_tickets"] == 27

    def test_read_plan_output_no_next_ticket(self, project_dir: Path) -> None:
        """Handles plan output where next_ticket is null."""
        data = _sample_plan_output(next_ticket=None)
        _write_json(project_dir, PLAN_OUTPUT_PATH, data)
        reader = StateReader(str(project_dir))

        result = reader.read_plan_output()
        assert result is not None
        assert result.next_ticket is None

    def test_read_missing_plan_output_returns_none(self, project_dir: Path) -> None:
        """Returns None when plan-output.json does not exist."""
        reader = StateReader(str(project_dir))
        result = reader.read_plan_output()
        assert result is None

    def test_read_plan_output_preserves_raw(self, project_dir: Path) -> None:
        """The raw dict is preserved."""
        data = _sample_plan_output()
        data["session"] = "pipeline-run-CMH-016"
        _write_json(project_dir, PLAN_OUTPUT_PATH, data)
        reader = StateReader(str(project_dir))

        result = reader.read_plan_output()
        assert result is not None
        assert result.raw["session"] == "pipeline-run-CMH-016"


# ===========================================================================
# TestReadBacklog
# ===========================================================================


class TestReadBacklog:
    """Tests for reading backlog.json."""

    def test_read_valid_backlog(self, project_dir: Path) -> None:
        """Reads and parses a valid backlog.json correctly."""
        data = _sample_backlog_data()
        _write_json(project_dir, BACKLOG_PATH, data)
        reader = StateReader(str(project_dir))

        result = reader.read_backlog()
        assert result is not None
        assert result.project == "claude-code-helper-mcp"
        assert len(result.phases) == 2
        assert len(result.tickets) == 5

    def test_read_backlog_phases(self, project_dir: Path) -> None:
        """Backlog phases are parsed correctly."""
        data = _sample_backlog_data()
        _write_json(project_dir, BACKLOG_PATH, data)
        reader = StateReader(str(project_dir))

        result = reader.read_backlog()
        assert result is not None
        phase1 = result.phases[0]
        assert phase1.id == "phase-1"
        assert phase1.name == "Foundation"
        assert phase1.status == "completed"
        assert len(phase1.tickets) == 2

        phase4 = result.phases[1]
        assert phase4.id == "phase-4"
        assert phase4.name == "Agent Forge Integration"
        assert phase4.status == "in-progress"

    def test_read_backlog_tickets(self, project_dir: Path) -> None:
        """Backlog tickets are parsed correctly."""
        data = _sample_backlog_data()
        _write_json(project_dir, BACKLOG_PATH, data)
        reader = StateReader(str(project_dir))

        result = reader.read_backlog()
        assert result is not None

        # Find CMH-016.
        ticket = result.get_ticket("CMH-016")
        assert ticket is not None
        assert ticket.title == "Pipeline state file readers"
        assert ticket.phase == "phase-4"
        assert ticket.priority == "high"
        assert ticket.status == "in-progress"
        assert ticket.dependencies == ["CMH-005", "CMH-015"]
        assert ticket.estimated_complexity == "medium"
        assert ticket.pr_number is None

    def test_read_backlog_empty_phases_and_tickets(self, project_dir: Path) -> None:
        """Handles backlog with no phases or tickets."""
        _write_json(
            project_dir,
            BACKLOG_PATH,
            {"project": "empty", "phases": [], "tickets": []},
        )
        reader = StateReader(str(project_dir))

        result = reader.read_backlog()
        assert result is not None
        assert result.project == "empty"
        assert result.phases == []
        assert result.tickets == []

    def test_read_missing_backlog_returns_none(self, project_dir: Path) -> None:
        """Returns None when backlog.json does not exist."""
        reader = StateReader(str(project_dir))
        result = reader.read_backlog()
        assert result is None

    def test_read_backlog_preserves_raw(self, project_dir: Path) -> None:
        """The raw dict is preserved."""
        data = _sample_backlog_data()
        _write_json(project_dir, BACKLOG_PATH, data)
        reader = StateReader(str(project_dir))

        result = reader.read_backlog()
        assert result is not None
        assert result.raw["project"] == "claude-code-helper-mcp"


# ===========================================================================
# TestBacklogQueries
# ===========================================================================


class TestBacklogQueries:
    """Tests for BacklogState query methods."""

    def test_get_ticket_found(self, project_dir: Path) -> None:
        """get_ticket returns the correct ticket."""
        data = _sample_backlog_data()
        _write_json(project_dir, BACKLOG_PATH, data)
        reader = StateReader(str(project_dir))

        backlog = reader.read_backlog()
        assert backlog is not None

        ticket = backlog.get_ticket("CMH-015")
        assert ticket is not None
        assert ticket.title == "Agent Forge hook integration"
        assert ticket.status == "done"

    def test_get_ticket_not_found(self, project_dir: Path) -> None:
        """get_ticket returns None for unknown ticket ID."""
        data = _sample_backlog_data()
        _write_json(project_dir, BACKLOG_PATH, data)
        reader = StateReader(str(project_dir))

        backlog = reader.read_backlog()
        assert backlog is not None
        assert backlog.get_ticket("CMH-999") is None

    def test_get_phase_found(self, project_dir: Path) -> None:
        """get_phase returns the correct phase."""
        data = _sample_backlog_data()
        _write_json(project_dir, BACKLOG_PATH, data)
        reader = StateReader(str(project_dir))

        backlog = reader.read_backlog()
        assert backlog is not None

        phase = backlog.get_phase("phase-4")
        assert phase is not None
        assert phase.name == "Agent Forge Integration"

    def test_get_phase_not_found(self, project_dir: Path) -> None:
        """get_phase returns None for unknown phase ID."""
        data = _sample_backlog_data()
        _write_json(project_dir, BACKLOG_PATH, data)
        reader = StateReader(str(project_dir))

        backlog = reader.read_backlog()
        assert backlog is not None
        assert backlog.get_phase("phase-99") is None

    def test_get_tickets_for_phase(self, project_dir: Path) -> None:
        """get_tickets_for_phase returns all tickets in a given phase."""
        data = _sample_backlog_data()
        _write_json(project_dir, BACKLOG_PATH, data)
        reader = StateReader(str(project_dir))

        backlog = reader.read_backlog()
        assert backlog is not None

        phase4_tickets = backlog.get_tickets_for_phase("phase-4")
        ids = [t.id for t in phase4_tickets]
        assert "CMH-015" in ids
        assert "CMH-016" in ids
        assert "CMH-017" in ids
        assert "CMH-001" not in ids

    def test_get_tickets_for_empty_phase(self, project_dir: Path) -> None:
        """get_tickets_for_phase returns empty list for unknown phase."""
        data = _sample_backlog_data()
        _write_json(project_dir, BACKLOG_PATH, data)
        reader = StateReader(str(project_dir))

        backlog = reader.read_backlog()
        assert backlog is not None
        assert backlog.get_tickets_for_phase("phase-99") == []

    def test_get_tickets_by_status_done(self, project_dir: Path) -> None:
        """get_tickets_by_status returns all done tickets."""
        data = _sample_backlog_data()
        _write_json(project_dir, BACKLOG_PATH, data)
        reader = StateReader(str(project_dir))

        backlog = reader.read_backlog()
        assert backlog is not None

        done = backlog.get_tickets_by_status("done")
        ids = [t.id for t in done]
        assert "CMH-001" in ids
        assert "CMH-002" in ids
        assert "CMH-015" in ids
        assert "CMH-016" not in ids

    def test_get_tickets_by_status_planned(self, project_dir: Path) -> None:
        """get_tickets_by_status returns planned tickets."""
        data = _sample_backlog_data()
        _write_json(project_dir, BACKLOG_PATH, data)
        reader = StateReader(str(project_dir))

        backlog = reader.read_backlog()
        assert backlog is not None

        planned = backlog.get_tickets_by_status("planned")
        assert len(planned) == 1
        assert planned[0].id == "CMH-017"

    def test_get_tickets_by_status_empty(self, project_dir: Path) -> None:
        """get_tickets_by_status returns empty for unknown status."""
        data = _sample_backlog_data()
        _write_json(project_dir, BACKLOG_PATH, data)
        reader = StateReader(str(project_dir))

        backlog = reader.read_backlog()
        assert backlog is not None
        assert backlog.get_tickets_by_status("nonexistent") == []


# ===========================================================================
# TestGetCurrentTicketContext
# ===========================================================================


class TestGetCurrentTicketContext:
    """Tests for the get_current_ticket_context convenience method."""

    def test_full_context_with_all_files(self, project_dir: Path) -> None:
        """Returns fully populated context when all state files are present."""
        _write_json(project_dir, PIPELINE_STATE_PATH, _sample_pipeline_data())
        _write_json(project_dir, BUILD_OUTPUT_PATH, _sample_build_output())
        _write_json(project_dir, BACKLOG_PATH, _sample_backlog_data())
        _write_json(project_dir, PLAN_OUTPUT_PATH, _sample_plan_output())
        reader = StateReader(str(project_dir))

        ctx = reader.get_current_ticket_context()
        assert ctx is not None
        assert ctx.ticket_id == "CMH-016"
        assert ctx.title == "Pipeline state file readers"
        assert ctx.description == "Implement StateReader class"
        assert ctx.phase == "phase-4"
        assert ctx.phase_name == "Agent Forge Integration"
        assert ctx.priority == "high"
        assert ctx.complexity == "medium"
        assert ctx.dependencies == ["CMH-005", "CMH-015"]
        assert ctx.pipeline_status == "running"
        assert ctx.pipeline_step == "build"
        assert ctx.pipeline_agent == "build"
        assert ctx.last_completed_step == "plan"
        assert ctx.branch == "feature/CMH-016-pipeline-state-file-readers"
        assert ctx.pr_number == 43
        assert ctx.build_status == "success"
        assert len(ctx.files_changed) == 2

    def test_context_without_build_output(self, project_dir: Path) -> None:
        """Returns context with build fields empty when no build output."""
        _write_json(project_dir, PIPELINE_STATE_PATH, _sample_pipeline_data())
        _write_json(project_dir, BACKLOG_PATH, _sample_backlog_data())
        reader = StateReader(str(project_dir))

        ctx = reader.get_current_ticket_context()
        assert ctx is not None
        assert ctx.ticket_id == "CMH-016"
        assert ctx.title == "Pipeline state file readers"
        assert ctx.branch is None
        assert ctx.pr_number is None
        assert ctx.build_status is None
        assert ctx.files_changed == []

    def test_context_without_backlog(self, project_dir: Path) -> None:
        """Returns context with ticket details empty when no backlog."""
        _write_json(project_dir, PIPELINE_STATE_PATH, _sample_pipeline_data())
        reader = StateReader(str(project_dir))

        ctx = reader.get_current_ticket_context()
        assert ctx is not None
        assert ctx.ticket_id == "CMH-016"
        assert ctx.title is None
        assert ctx.description is None
        assert ctx.phase is None
        assert ctx.pipeline_status == "running"

    def test_context_idle_pipeline_falls_back_to_plan(self, project_dir: Path) -> None:
        """Falls back to plan output when pipeline has no current ticket."""
        _write_json(
            project_dir,
            PIPELINE_STATE_PATH,
            _sample_pipeline_data(
                status="idle",
                current_ticket=None,
                current_step=None,
                current_agent=None,
            ),
        )
        _write_json(project_dir, PLAN_OUTPUT_PATH, _sample_plan_output())
        _write_json(project_dir, BACKLOG_PATH, _sample_backlog_data())
        reader = StateReader(str(project_dir))

        ctx = reader.get_current_ticket_context()
        assert ctx is not None
        assert ctx.ticket_id == "CMH-016"
        assert ctx.title == "Pipeline state file readers"

    def test_context_returns_none_when_no_ticket(self, project_dir: Path) -> None:
        """Returns None when no ticket can be determined."""
        _write_json(
            project_dir,
            PIPELINE_STATE_PATH,
            _sample_pipeline_data(status="idle", current_ticket=None),
        )
        _write_json(
            project_dir,
            PLAN_OUTPUT_PATH,
            _sample_plan_output(next_ticket=None),
        )
        reader = StateReader(str(project_dir))

        ctx = reader.get_current_ticket_context()
        assert ctx is None

    def test_context_returns_none_when_no_pipeline(self, project_dir: Path) -> None:
        """Returns None when pipeline.json is missing."""
        reader = StateReader(str(project_dir))
        ctx = reader.get_current_ticket_context()
        assert ctx is None

    def test_context_build_output_ticket_mismatch(self, project_dir: Path) -> None:
        """Build output is ignored when its ticket_id does not match."""
        _write_json(project_dir, PIPELINE_STATE_PATH, _sample_pipeline_data())
        _write_json(
            project_dir,
            BUILD_OUTPUT_PATH,
            _sample_build_output(ticket_id="CMH-015"),  # different ticket
        )
        _write_json(project_dir, BACKLOG_PATH, _sample_backlog_data())
        reader = StateReader(str(project_dir))

        ctx = reader.get_current_ticket_context()
        assert ctx is not None
        assert ctx.ticket_id == "CMH-016"
        assert ctx.branch is None  # build output ignored
        assert ctx.pr_number is None

    def test_context_with_blocked_pipeline(self, project_dir: Path) -> None:
        """Context reflects blocked pipeline state correctly."""
        _write_json(
            project_dir,
            PIPELINE_STATE_PATH,
            _sample_pipeline_data(
                status="blocked",
                current_step="validate",
                current_agent="validate",
                last_completed_step="build",
                failed_step="validate",
                failure_reason="Validation found issues",
            ),
        )
        _write_json(project_dir, BACKLOG_PATH, _sample_backlog_data())
        reader = StateReader(str(project_dir))

        ctx = reader.get_current_ticket_context()
        assert ctx is not None
        assert ctx.pipeline_status == "blocked"
        assert ctx.pipeline_step == "validate"
        assert ctx.last_completed_step == "build"


# ===========================================================================
# TestCaching
# ===========================================================================


class TestCaching:
    """Tests for cache behaviour, TTL, and invalidation."""

    def test_read_is_cached(self, project_dir: Path) -> None:
        """A successful read populates the cache."""
        _write_json(project_dir, PIPELINE_STATE_PATH, _sample_pipeline_data())
        reader = StateReader(str(project_dir), cache_ttl_seconds=60.0)

        reader.read_pipeline()
        assert reader.is_cached(PIPELINE_STATE_PATH)

    def test_cached_read_returns_same_data(self, project_dir: Path) -> None:
        """Subsequent reads within TTL return cached data."""
        _write_json(
            project_dir,
            PIPELINE_STATE_PATH,
            _sample_pipeline_data(status="running"),
        )
        reader = StateReader(str(project_dir), cache_ttl_seconds=60.0)

        result1 = reader.read_pipeline()
        assert result1 is not None
        assert result1.status == "running"

        # Modify the file on disk.
        _write_json(
            project_dir,
            PIPELINE_STATE_PATH,
            _sample_pipeline_data(status="completed"),
        )

        # Second read should still return cached "running" data.
        result2 = reader.read_pipeline()
        assert result2 is not None
        assert result2.status == "running"

    def test_cache_expiry(self, project_dir: Path) -> None:
        """Cache expires after TTL and re-reads from disk."""
        _write_json(
            project_dir,
            PIPELINE_STATE_PATH,
            _sample_pipeline_data(status="running"),
        )
        reader = StateReader(str(project_dir), cache_ttl_seconds=0.1)

        result1 = reader.read_pipeline()
        assert result1 is not None
        assert result1.status == "running"

        # Modify the file.
        _write_json(
            project_dir,
            PIPELINE_STATE_PATH,
            _sample_pipeline_data(status="completed"),
        )

        # Wait for cache to expire.
        time.sleep(0.15)

        result2 = reader.read_pipeline()
        assert result2 is not None
        assert result2.status == "completed"

    def test_invalidate_specific_key(self, project_dir: Path) -> None:
        """Invalidating a specific key forces re-read on next call."""
        _write_json(
            project_dir,
            PIPELINE_STATE_PATH,
            _sample_pipeline_data(status="running"),
        )
        reader = StateReader(str(project_dir), cache_ttl_seconds=60.0)

        reader.read_pipeline()
        assert reader.is_cached(PIPELINE_STATE_PATH)

        reader.invalidate_cache(PIPELINE_STATE_PATH)
        assert not reader.is_cached(PIPELINE_STATE_PATH)

    def test_invalidate_all(self, project_dir: Path) -> None:
        """Invalidating all keys clears the entire cache."""
        _write_json(project_dir, PIPELINE_STATE_PATH, _sample_pipeline_data())
        _write_json(project_dir, BUILD_OUTPUT_PATH, _sample_build_output())
        reader = StateReader(str(project_dir), cache_ttl_seconds=60.0)

        reader.read_pipeline()
        reader.read_build_output()
        assert len(reader.cached_keys()) == 2

        reader.invalidate_cache()
        assert len(reader.cached_keys()) == 0

    def test_refresh_clears_cache(self, project_dir: Path) -> None:
        """refresh() is an alias for invalidate_cache(None)."""
        _write_json(project_dir, PIPELINE_STATE_PATH, _sample_pipeline_data())
        reader = StateReader(str(project_dir), cache_ttl_seconds=60.0)

        reader.read_pipeline()
        assert len(reader.cached_keys()) == 1

        reader.refresh()
        assert len(reader.cached_keys()) == 0

    def test_cached_keys_reflects_read_files(self, project_dir: Path) -> None:
        """cached_keys returns all files that have been read."""
        _write_json(project_dir, PIPELINE_STATE_PATH, _sample_pipeline_data())
        _write_json(project_dir, BUILD_OUTPUT_PATH, _sample_build_output())
        _write_json(project_dir, BACKLOG_PATH, _sample_backlog_data())
        reader = StateReader(str(project_dir), cache_ttl_seconds=60.0)

        reader.read_pipeline()
        reader.read_build_output()
        reader.read_backlog()

        keys = reader.cached_keys()
        assert PIPELINE_STATE_PATH in keys
        assert BUILD_OUTPUT_PATH in keys
        assert BACKLOG_PATH in keys

    def test_none_results_are_cached(self, project_dir: Path) -> None:
        """Missing file reads are cached to avoid repeated disk hits."""
        reader = StateReader(str(project_dir), cache_ttl_seconds=60.0)

        result1 = reader.read_pipeline()  # file does not exist
        assert result1 is None
        assert PIPELINE_STATE_PATH in reader.cached_keys()

        # Create the file after the first read.
        _write_json(project_dir, PIPELINE_STATE_PATH, _sample_pipeline_data())

        # Should still return None from cache.
        result2 = reader.read_pipeline()
        assert result2 is None

    def test_invalidate_after_none_cache_allows_reread(self, project_dir: Path) -> None:
        """Invalidating after a None cache entry allows fresh read."""
        reader = StateReader(str(project_dir), cache_ttl_seconds=60.0)

        # First read returns None (file missing).
        result1 = reader.read_pipeline()
        assert result1 is None

        # Create the file.
        _write_json(project_dir, PIPELINE_STATE_PATH, _sample_pipeline_data())

        # Invalidate and re-read.
        reader.invalidate_cache(PIPELINE_STATE_PATH)
        result2 = reader.read_pipeline()
        assert result2 is not None
        assert result2.status == "running"

    def test_each_reader_has_independent_cache(self, project_dir: Path) -> None:
        """Different StateReader instances have independent caches."""
        _write_json(project_dir, PIPELINE_STATE_PATH, _sample_pipeline_data())
        reader1 = StateReader(str(project_dir), cache_ttl_seconds=60.0)
        reader2 = StateReader(str(project_dir), cache_ttl_seconds=60.0)

        reader1.read_pipeline()
        assert reader1.is_cached(PIPELINE_STATE_PATH)
        assert not reader2.is_cached(PIPELINE_STATE_PATH)


# ===========================================================================
# TestGracefulDegradation
# ===========================================================================


class TestGracefulDegradation:
    """Tests for graceful handling of missing/malformed files and edge cases."""

    def test_malformed_json_returns_none(self, project_dir: Path) -> None:
        """Malformed JSON files return None without raising."""
        file_path = project_dir / AGENT_FORGE_DIR / PIPELINE_STATE_PATH
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as fp:
            fp.write("{invalid json!")

        reader = StateReader(str(project_dir))
        result = reader.read_pipeline()
        assert result is None

    def test_empty_file_returns_none(self, project_dir: Path) -> None:
        """Empty files return None without raising."""
        file_path = project_dir / AGENT_FORGE_DIR / PIPELINE_STATE_PATH
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as fp:
            fp.write("")

        reader = StateReader(str(project_dir))
        result = reader.read_pipeline()
        assert result is None

    def test_json_array_returns_none(self, project_dir: Path) -> None:
        """A JSON file containing an array (not object) returns None."""
        file_path = project_dir / AGENT_FORGE_DIR / PIPELINE_STATE_PATH
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as fp:
            json.dump([1, 2, 3], fp)

        reader = StateReader(str(project_dir))
        result = reader.read_pipeline()
        assert result is None

    def test_json_string_returns_none(self, project_dir: Path) -> None:
        """A JSON file containing a string (not object) returns None."""
        file_path = project_dir / AGENT_FORGE_DIR / PIPELINE_STATE_PATH
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as fp:
            json.dump("just a string", fp)

        reader = StateReader(str(project_dir))
        result = reader.read_pipeline()
        assert result is None

    def test_no_agent_forge_directory(self, tmp_path: Path) -> None:
        """All reads return None when .agent-forge/ does not exist."""
        reader = StateReader(str(tmp_path))
        assert reader.read_pipeline() is None
        assert reader.read_build_output() is None
        assert reader.read_plan_output() is None
        assert reader.read_backlog() is None
        assert reader.get_current_ticket_context() is None

    def test_permission_error_returns_none(self, project_dir: Path) -> None:
        """Permission errors return None gracefully."""
        file_path = project_dir / AGENT_FORGE_DIR / PIPELINE_STATE_PATH
        _write_json(project_dir, PIPELINE_STATE_PATH, _sample_pipeline_data())

        # Make file unreadable.
        original_mode = file_path.stat().st_mode
        try:
            os.chmod(file_path, 0o000)
            reader = StateReader(str(project_dir))
            result = reader.read_pipeline()
            assert result is None
        finally:
            # Restore permissions for cleanup.
            os.chmod(file_path, original_mode)

    def test_empty_dict_pipeline(self, project_dir: Path) -> None:
        """An empty dict in pipeline.json produces defaults."""
        _write_json(project_dir, PIPELINE_STATE_PATH, {})
        reader = StateReader(str(project_dir))

        result = reader.read_pipeline()
        assert result is not None
        assert result.status == "unknown"
        assert result.current_ticket is None

    def test_empty_dict_backlog(self, project_dir: Path) -> None:
        """An empty dict in backlog.json produces defaults."""
        _write_json(project_dir, BACKLOG_PATH, {})
        reader = StateReader(str(project_dir))

        result = reader.read_backlog()
        assert result is not None
        assert result.project == ""
        assert result.phases == []
        assert result.tickets == []

    def test_partial_ticket_data(self, project_dir: Path) -> None:
        """A ticket with missing fields still parses."""
        data = {
            "project": "test",
            "phases": [],
            "tickets": [
                {
                    "id": "T-001",
                    "title": "Minimal ticket",
                }
            ],
        }
        _write_json(project_dir, BACKLOG_PATH, data)
        reader = StateReader(str(project_dir))

        backlog = reader.read_backlog()
        assert backlog is not None
        ticket = backlog.get_ticket("T-001")
        assert ticket is not None
        assert ticket.id == "T-001"
        assert ticket.title == "Minimal ticket"
        assert ticket.status == "unknown"
        assert ticket.dependencies == []
        assert ticket.pr_number is None

    def test_partial_phase_data(self, project_dir: Path) -> None:
        """A phase with missing fields still parses."""
        data = {
            "project": "test",
            "phases": [{"id": "p-1", "name": "Phase One"}],
            "tickets": [],
        }
        _write_json(project_dir, BACKLOG_PATH, data)
        reader = StateReader(str(project_dir))

        backlog = reader.read_backlog()
        assert backlog is not None
        phase = backlog.get_phase("p-1")
        assert phase is not None
        assert phase.name == "Phase One"
        assert phase.status == "unknown"
        assert phase.tickets == []


# ===========================================================================
# TestDataClasses
# ===========================================================================


class TestDataClasses:
    """Tests for data class construction and from_dict methods."""

    def test_pipeline_state_from_dict(self) -> None:
        """PipelineState.from_dict creates correct instance."""
        data = {
            "status": "completed",
            "current_phase": "phase-3",
            "current_ticket": None,
        }
        state = PipelineState.from_dict(data)
        assert state.status == "completed"
        assert state.current_phase == "phase-3"
        assert state.current_ticket is None
        assert state.raw == data

    def test_build_output_from_dict(self) -> None:
        """BuildOutput.from_dict creates correct instance."""
        data = {
            "status": "success",
            "branch": "feature/test",
            "pr_number": 42,
            "files_changed": ["a.py", "b.py"],
            "summary": "Done",
            "ticket_id": "T-001",
            "self_check": {"no_mocks": True},
        }
        output = BuildOutput.from_dict(data)
        assert output.status == "success"
        assert output.branch == "feature/test"
        assert output.pr_number == 42
        assert output.files_changed == ["a.py", "b.py"]
        assert output.self_check == {"no_mocks": True}

    def test_plan_output_from_dict(self) -> None:
        """PlanOutput.from_dict creates correct instance."""
        data = {
            "status": "success",
            "next_ticket": "T-002",
            "ticket_id": "T-002",
            "next_ticket_details": {"id": "T-002"},
            "backlog_summary": {"total": 10},
        }
        output = PlanOutput.from_dict(data)
        assert output.next_ticket == "T-002"
        assert output.next_ticket_details["id"] == "T-002"

    def test_backlog_ticket_from_dict_full(self) -> None:
        """BacklogTicket.from_dict with all fields."""
        data = {
            "id": "T-003",
            "github_issue": 3,
            "title": "Test ticket",
            "description": "A test",
            "phase": "phase-1",
            "priority": "high",
            "type": "feature",
            "status": "done",
            "dependencies": ["T-001"],
            "assigned_agent": "build",
            "estimated_complexity": "small",
            "pr_number": 10,
            "completed_at": "2026-01-01T00:00:00Z",
        }
        ticket = BacklogTicket.from_dict(data)
        assert ticket.id == "T-003"
        assert ticket.github_issue == 3
        assert ticket.priority == "high"
        assert ticket.pr_number == 10
        assert ticket.completed_at == "2026-01-01T00:00:00Z"

    def test_backlog_phase_from_dict(self) -> None:
        """BacklogPhase.from_dict creates correct instance."""
        data = {
            "id": "phase-2",
            "name": "Core",
            "description": "Core features",
            "milestone": "M-2",
            "status": "completed",
            "tickets": ["T-004", "T-005"],
        }
        phase = BacklogPhase.from_dict(data)
        assert phase.id == "phase-2"
        assert phase.name == "Core"
        assert phase.tickets == ["T-004", "T-005"]

    def test_ticket_context_defaults(self) -> None:
        """TicketContext has sensible defaults."""
        ctx = TicketContext()
        assert ctx.ticket_id is None
        assert ctx.title is None
        assert ctx.dependencies == []
        assert ctx.files_changed == []
        assert ctx.pipeline_status is None

    def test_ticket_context_full(self) -> None:
        """TicketContext can be fully populated."""
        ctx = TicketContext(
            ticket_id="T-010",
            title="Full context",
            description="All fields set",
            phase="phase-3",
            phase_name="Recovery",
            priority="critical",
            complexity="large",
            dependencies=["T-001", "T-002"],
            pipeline_status="running",
            pipeline_step="build",
            pipeline_agent="build",
            last_completed_step="plan",
            branch="feature/T-010",
            pr_number=55,
            build_status="success",
            files_changed=["a.py"],
        )
        assert ctx.ticket_id == "T-010"
        assert ctx.phase_name == "Recovery"
        assert ctx.pr_number == 55

    def test_pipeline_state_is_frozen(self) -> None:
        """PipelineState is immutable (frozen dataclass)."""
        state = PipelineState.from_dict({"status": "idle"})
        with pytest.raises(AttributeError):
            state.status = "running"  # type: ignore[misc]

    def test_build_output_is_frozen(self) -> None:
        """BuildOutput is immutable (frozen dataclass)."""
        output = BuildOutput.from_dict({"status": "success"})
        with pytest.raises(AttributeError):
            output.status = "error"  # type: ignore[misc]

    def test_backlog_ticket_is_frozen(self) -> None:
        """BacklogTicket is immutable (frozen dataclass)."""
        ticket = BacklogTicket.from_dict({"id": "T-001", "title": "Test"})
        with pytest.raises(AttributeError):
            ticket.status = "done"  # type: ignore[misc]

    def test_ticket_context_is_mutable(self) -> None:
        """TicketContext is mutable (used as an accumulator by get_current_ticket_context)."""
        ctx = TicketContext()
        ctx.ticket_id = "T-001"
        ctx.title = "Updated"
        assert ctx.ticket_id == "T-001"
        assert ctx.title == "Updated"
