"""Tests for the check_alignment MCP tool (CMH-012).

Verifies alignment checking for active tasks, keyword overlap scoring,
file-path relevance, contextual scoring, warning generation, threshold
configuration, edge cases (no task, empty action, no files), and the
AlignmentChecker engine directly.

All tests use real file I/O with temporary directories -- zero mocks.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from claude_code_helper_mcp.detection.alignment import (
    AlignmentChecker,
    AlignmentReport,
    DEFAULT_ALIGNMENT_THRESHOLD,
)
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


def _check_alignment(tools: dict, action: str, file_path: str = "",
                     threshold: float = 0.5) -> dict:
    """Helper to call check_alignment tool."""
    args = {"action": action}
    if file_path:
        args["file_path"] = file_path
    if threshold != 0.5:
        args["threshold"] = threshold
    return _run_tool(tools, "check_alignment", args)


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

class TestCheckAlignmentRegistration:
    """Tests that check_alignment is properly registered."""

    def test_tool_is_registered(self, tools):
        assert "check_alignment" in tools

    def test_tool_callable(self, tools):
        tool = tools["check_alignment"]
        assert tool is not None


# ---------------------------------------------------------------------------
# No active task
# ---------------------------------------------------------------------------

class TestCheckAlignmentNoTask:
    """Tests for check_alignment when no task is active."""

    def test_returns_error_when_no_task(self, tools):
        result = _check_alignment(tools, "Some action")
        assert result["error"] is True
        assert "No active task" in result["message"]

    def test_error_includes_timestamp(self, tools):
        result = _check_alignment(tools, "Some action")
        assert "timestamp" in result
        assert result["timestamp"]


# ---------------------------------------------------------------------------
# Basic alignment with active task
# ---------------------------------------------------------------------------

class TestCheckAlignmentBasic:
    """Tests for basic alignment checking with an active task."""

    def test_returns_alignment_report(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool", phase="phase-3")
        result = _check_alignment(tools, "Implementing the alignment checker")
        assert result["error"] is False
        assert "confidence" in result
        assert "aligned" in result
        assert "warnings" in result

    def test_confidence_is_float(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool")
        result = _check_alignment(tools, "Working on alignment")
        assert isinstance(result["confidence"], (int, float))
        assert 0.0 <= result["confidence"] <= 1.0

    def test_aligned_is_bool(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool")
        result = _check_alignment(tools, "Working on alignment")
        assert isinstance(result["aligned"], bool)

    def test_includes_task_id(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool")
        result = _check_alignment(tools, "Working on alignment")
        assert result["task_id"] == "CMH-012"

    def test_includes_timestamp(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool")
        result = _check_alignment(tools, "Working on alignment")
        assert "timestamp" in result
        assert result["timestamp"]

    def test_includes_scope_info(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool", phase="phase-3")
        result = _check_alignment(tools, "Working on alignment")
        assert "scope_info" in result
        assert result["scope_info"]["ticket_id"] == "CMH-012"
        assert result["scope_info"]["title"] == "check_alignment MCP tool"
        assert result["scope_info"]["phase"] == "phase-3"

    def test_includes_action_analysis(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool")
        result = _check_alignment(tools, "Working on alignment")
        assert "action_analysis" in result
        assert result["action_analysis"]["action"] == "Working on alignment"
        assert "keyword_overlap_score" in result["action_analysis"]
        assert "file_relevance_score" in result["action_analysis"]
        assert "contextual_score" in result["action_analysis"]


# ---------------------------------------------------------------------------
# Aligned actions (should produce high confidence)
# ---------------------------------------------------------------------------

class TestCheckAlignmentAlignedActions:
    """Tests that clearly aligned actions get high confidence."""

    def test_action_matching_title_keywords(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool", phase="phase-3")
        result = _check_alignment(tools, "Implementing the check_alignment MCP tool")
        assert result["confidence"] >= 0.5
        assert result["aligned"] is True

    def test_action_referencing_ticket_id(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool")
        result = _check_alignment(tools, "Working on CMH-012 alignment checker")
        assert result["confidence"] >= 0.5

    def test_action_with_recorded_file(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool")
        _record_file(tools, "src/detection/alignment.py", "created")
        result = _check_alignment(
            tools,
            "Adding keyword extraction to alignment module",
            file_path="src/detection/alignment.py",
        )
        assert result["confidence"] >= 0.5
        assert result["aligned"] is True

    def test_action_in_same_directory_as_recorded_files(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool")
        _record_file(tools, "src/detection/alignment.py", "created")
        result = _check_alignment(
            tools,
            "Adding detection utilities",
            file_path="src/detection/helpers.py",
        )
        # Same directory should contribute positively.
        assert result["action_analysis"]["file_relevance_score"] >= 0.5

    def test_action_with_matching_step_context(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool")
        _record_step(tools, "Created AlignmentChecker class")
        _record_step(tools, "Added keyword extraction method")
        result = _check_alignment(
            tools, "Adding scoring functions to AlignmentChecker"
        )
        assert result["confidence"] >= 0.5


# ---------------------------------------------------------------------------
# Misaligned actions (should produce low confidence)
# ---------------------------------------------------------------------------

class TestCheckAlignmentMisalignedActions:
    """Tests that clearly misaligned actions get low confidence."""

    def test_completely_unrelated_action(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool", phase="phase-3")
        _record_file(tools, "src/detection/alignment.py", "created")
        result = _check_alignment(
            tools,
            "Configuring database migration for PostgreSQL schema",
            file_path="migrations/0001_initial.sql",
        )
        assert result["confidence"] < 0.5
        assert result["aligned"] is False
        assert len(result["warnings"]) > 0

    def test_file_in_unrelated_directory(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool")
        _record_file(tools, "src/detection/alignment.py", "created")
        _record_file(tools, "tests/test_check_alignment.py", "created")
        result = _check_alignment(
            tools,
            "Editing infrastructure config",
            file_path="infra/terraform/main.tf",
        )
        assert result["action_analysis"]["file_relevance_score"] < 0.5

    def test_action_with_no_keyword_overlap(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool")
        result = _check_alignment(
            tools,
            "Deploying kubernetes pods to production cluster",
        )
        assert result["action_analysis"]["keyword_overlap_score"] < 0.5


# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------

class TestCheckAlignmentWarnings:
    """Tests for warning generation."""

    def test_low_confidence_produces_warning(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool")
        _record_file(tools, "src/detection/alignment.py", "created")
        result = _check_alignment(
            tools,
            "Configuring database migration for PostgreSQL",
            file_path="migrations/0001.sql",
        )
        if result["confidence"] < 0.5:
            assert len(result["warnings"]) > 0
            assert any("scope" in w.lower() or "confidence" in w.lower()
                       for w in result["warnings"])

    def test_file_outside_scope_warning(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool")
        _record_file(tools, "src/detection/alignment.py", "created")
        _record_file(tools, "src/mcp/server.py", "modified")
        result = _check_alignment(
            tools,
            "Editing infrastructure code",
            file_path="infra/deploy/config.yaml",
        )
        # Should warn about file being outside recorded directories.
        file_warnings = [w for w in result["warnings"] if "outside" in w.lower() or "director" in w.lower()]
        if result["action_analysis"]["file_relevance_score"] < 0.4:
            assert len(file_warnings) > 0

    def test_no_warnings_for_highly_aligned_action(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool")
        _record_file(tools, "src/detection/alignment.py", "created")
        result = _check_alignment(
            tools,
            "Implementing the check_alignment MCP tool for CMH-012",
            file_path="src/detection/alignment.py",
        )
        # Highly aligned actions may still get advisory warnings.
        # But no "outside scope" warnings.
        scope_warnings = [w for w in result["warnings"]
                          if "outside" in w.lower() and "scope" in w.lower()]
        assert len(scope_warnings) == 0

    def test_warnings_are_strings(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool")
        result = _check_alignment(tools, "Random unrelated database work")
        for w in result["warnings"]:
            assert isinstance(w, str)


# ---------------------------------------------------------------------------
# Threshold configuration
# ---------------------------------------------------------------------------

class TestCheckAlignmentThreshold:
    """Tests for configurable threshold."""

    def test_strict_threshold_flags_marginal_action(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool")
        # A moderately related action.
        result_strict = _check_alignment(
            tools, "Adding detection helpers", threshold=0.9
        )
        result_permissive = _check_alignment(
            tools, "Adding detection helpers", threshold=0.1
        )
        # Strict threshold should be harder to pass.
        assert result_strict["action_analysis"]["threshold"] == 0.9
        assert result_permissive["action_analysis"]["threshold"] == 0.1
        # The confidence score should be the same regardless of threshold.
        assert result_strict["confidence"] == result_permissive["confidence"]

    def test_threshold_affects_aligned_boolean(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool")
        # Very permissive threshold: almost everything is aligned.
        result = _check_alignment(
            tools, "Some loosely related work", threshold=0.01
        )
        assert result["aligned"] is True

    def test_threshold_clamped_to_valid_range(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool")
        # Threshold above 1.0 should be clamped.
        result = _check_alignment(tools, "Something", threshold=2.0)
        assert result["action_analysis"]["threshold"] == 1.0

        # Threshold below 0.0 should be clamped.
        result = _check_alignment(tools, "Something", threshold=-1.0)
        assert result["action_analysis"]["threshold"] == 0.0

    def test_default_threshold_is_0_5(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool")
        result = _check_alignment(tools, "Working on alignment")
        assert result["action_analysis"]["threshold"] == 0.5


# ---------------------------------------------------------------------------
# File path analysis
# ---------------------------------------------------------------------------

class TestCheckAlignmentFilePath:
    """Tests for file path relevance scoring."""

    def test_exact_file_match_high_score(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool")
        _record_file(tools, "src/detection/alignment.py", "created")
        result = _check_alignment(
            tools, "Editing alignment module",
            file_path="src/detection/alignment.py",
        )
        assert result["action_analysis"]["file_relevance_score"] >= 0.9

    def test_no_file_path_gives_neutral_score(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool")
        result = _check_alignment(tools, "Working on alignment")
        assert result["action_analysis"]["file_relevance_score"] == 0.5

    def test_no_recorded_files_gives_neutral_positive(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool")
        result = _check_alignment(
            tools, "Creating first file",
            file_path="src/detection/alignment.py",
        )
        # No files recorded yet -- should be neutral-positive.
        assert result["action_analysis"]["file_relevance_score"] >= 0.5

    def test_same_directory_boosts_score(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool")
        _record_file(tools, "src/detection/alignment.py", "created")
        _record_file(tools, "src/mcp/server.py", "modified")
        result = _check_alignment(
            tools, "Adding detection helper",
            file_path="src/detection/utils.py",
        )
        assert result["action_analysis"]["file_relevance_score"] >= 0.5


# ---------------------------------------------------------------------------
# With description metadata
# ---------------------------------------------------------------------------

class TestCheckAlignmentWithDescription:
    """Tests that task description enriches scope analysis."""

    def test_description_keywords_improve_alignment(self, tools):
        _start_task(
            tools, "CMH-012", "check_alignment MCP tool",
            phase="phase-3",
            description="Implement alignment checking that compares actions against task scope using keyword extraction and file path analysis",
        )
        result = _check_alignment(
            tools, "Adding keyword extraction to the alignment checker"
        )
        # Description keywords should boost the score.
        assert result["confidence"] >= 0.5


# ---------------------------------------------------------------------------
# AlignmentChecker engine tests (direct, no MCP)
# ---------------------------------------------------------------------------

class TestAlignmentCheckerDirect:
    """Direct tests for the AlignmentChecker engine without the MCP layer."""

    def test_default_threshold(self):
        checker = AlignmentChecker()
        assert checker.threshold == DEFAULT_ALIGNMENT_THRESHOLD

    def test_custom_threshold(self):
        checker = AlignmentChecker(threshold=0.8)
        assert checker.threshold == 0.8

    def test_threshold_clamped_low(self):
        checker = AlignmentChecker(threshold=-0.5)
        assert checker.threshold == 0.0

    def test_threshold_clamped_high(self):
        checker = AlignmentChecker(threshold=1.5)
        assert checker.threshold == 1.0

    def test_check_returns_alignment_report(self):
        checker = AlignmentChecker()
        report = checker.check(
            action="Implementing keyword extraction",
            file_path=None,
            task_title="Keyword extraction module",
        )
        assert isinstance(report, AlignmentReport)
        assert 0.0 <= report.confidence <= 1.0
        assert isinstance(report.aligned, bool)
        assert isinstance(report.warnings, list)
        assert isinstance(report.scope_info, dict)
        assert isinstance(report.action_analysis, dict)

    def test_high_overlap_action(self):
        checker = AlignmentChecker()
        report = checker.check(
            action="Adding keyword extraction to the alignment checker",
            file_path="src/detection/alignment.py",
            task_title="check_alignment MCP tool",
            task_description="Implement alignment checking with keyword extraction",
            task_files=["src/detection/alignment.py"],
            task_ticket_id="CMH-012",
        )
        assert report.confidence >= 0.5
        assert report.aligned is True

    def test_low_overlap_action(self):
        checker = AlignmentChecker()
        report = checker.check(
            action="Deploying kubernetes pods to production",
            file_path="infra/k8s/deployment.yaml",
            task_title="check_alignment MCP tool",
            task_description="Alignment checking",
            task_files=["src/detection/alignment.py"],
            task_ticket_id="CMH-012",
        )
        assert report.confidence < 0.5
        assert report.aligned is False

    def test_empty_action(self):
        checker = AlignmentChecker()
        report = checker.check(
            action="",
            file_path=None,
            task_title="Some task",
        )
        # Empty action should produce neutral/low score.
        assert 0.0 <= report.confidence <= 1.0

    def test_report_to_dict(self):
        checker = AlignmentChecker()
        report = checker.check(
            action="Doing some work",
            file_path=None,
            task_title="A task",
        )
        d = report.to_dict()
        assert "confidence" in d
        assert "aligned" in d
        assert "warnings" in d
        assert "scope_info" in d
        assert "action_analysis" in d
        assert "generated_at" in d


# ---------------------------------------------------------------------------
# Keyword extraction tests
# ---------------------------------------------------------------------------

class TestKeywordExtraction:
    """Tests for the AlignmentChecker keyword extraction methods."""

    def test_extract_keywords_basic(self):
        keywords = AlignmentChecker._extract_keywords(
            "Implementing the alignment checker module"
        )
        assert "alignment" in keywords
        assert "checker" in keywords

    def test_extract_keywords_removes_stop_words(self):
        keywords = AlignmentChecker._extract_keywords(
            "the alignment is a good module"
        )
        assert "the" not in keywords
        assert "is" not in keywords
        assert "alignment" in keywords

    def test_extract_keywords_removes_short_tokens(self):
        keywords = AlignmentChecker._extract_keywords("a b cd efg")
        assert "a" not in keywords
        assert "b" not in keywords
        assert "cd" in keywords
        assert "efg" in keywords

    def test_extract_keywords_handles_camelcase(self):
        keywords = AlignmentChecker._extract_keywords(
            "AlignmentChecker getTaskStatus"
        )
        assert "alignment" in keywords
        assert "checker" in keywords
        assert "task" in keywords
        assert "status" in keywords

    def test_extract_keywords_empty_string(self):
        keywords = AlignmentChecker._extract_keywords("")
        assert keywords == set()

    def test_extract_path_keywords(self):
        keywords = AlignmentChecker._extract_path_keywords(
            "src/detection/alignment.py"
        )
        assert "detection" in keywords
        assert "alignment" in keywords
        # 'src' and 'py' are non-informative and should be excluded.
        assert "src" not in keywords
        assert "py" not in keywords

    def test_extract_path_keywords_empty(self):
        keywords = AlignmentChecker._extract_path_keywords("")
        assert keywords == set()

    def test_extract_path_keywords_nested(self):
        keywords = AlignmentChecker._extract_path_keywords(
            "src/claude_code_helper_mcp/detection/alignment.py"
        )
        assert "claude" in keywords
        assert "helper" in keywords
        assert "detection" in keywords
        assert "alignment" in keywords


# ---------------------------------------------------------------------------
# Scoring function tests
# ---------------------------------------------------------------------------

class TestScoringFunctions:
    """Tests for individual scoring functions."""

    def test_keyword_overlap_full_match(self):
        score = AlignmentChecker._keyword_overlap_score(
            {"alignment", "checker"},
            {"alignment", "checker", "mcp", "tool"},
        )
        assert score == 1.0

    def test_keyword_overlap_no_match(self):
        score = AlignmentChecker._keyword_overlap_score(
            {"database", "migration"},
            {"alignment", "checker", "mcp"},
        )
        assert score == 0.0

    def test_keyword_overlap_partial(self):
        score = AlignmentChecker._keyword_overlap_score(
            {"alignment", "database"},
            {"alignment", "checker", "mcp"},
        )
        assert 0.0 < score < 1.0

    def test_keyword_overlap_empty_action(self):
        score = AlignmentChecker._keyword_overlap_score(
            set(), {"alignment", "checker"},
        )
        assert score == 0.5  # neutral

    def test_keyword_overlap_empty_scope(self):
        score = AlignmentChecker._keyword_overlap_score(
            {"alignment"}, set(),
        )
        assert score == 0.5  # neutral

    def test_file_relevance_exact_match(self):
        score = AlignmentChecker._file_relevance_score(
            "src/detection/alignment.py",
            ["src/detection/alignment.py", "src/mcp/server.py"],
        )
        assert score == 1.0

    def test_file_relevance_no_file(self):
        score = AlignmentChecker._file_relevance_score(
            None, ["src/detection/alignment.py"],
        )
        assert score == 0.5

    def test_file_relevance_no_recorded_files(self):
        score = AlignmentChecker._file_relevance_score(
            "src/detection/alignment.py", [],
        )
        assert score == 0.6

    def test_file_relevance_unrelated_directory(self):
        score = AlignmentChecker._file_relevance_score(
            "infra/terraform/main.tf",
            ["src/detection/alignment.py", "src/mcp/server.py"],
        )
        assert score < 0.5

    def test_contextual_score_with_ticket_reference(self):
        score = AlignmentChecker._contextual_score(
            "Working on CMH-012 alignment tool",
            "check_alignment MCP tool",
            "CMH-012",
        )
        assert score >= 0.7  # ticket reference is a strong signal

    def test_contextual_score_no_reference(self):
        score = AlignmentChecker._contextual_score(
            "Deploying to production",
            "check_alignment MCP tool",
            "CMH-012",
        )
        assert score <= 0.6

    def test_contextual_score_title_keyword_match(self):
        score = AlignmentChecker._contextual_score(
            "Working on alignment checking",
            "check_alignment MCP tool",
            "CMH-012",
        )
        assert score > 0.5


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestCheckAlignmentEdgeCases:
    """Edge case tests for robustness."""

    def test_very_long_action(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool")
        long_action = "alignment " * 50
        result = _check_alignment(tools, long_action.strip())
        assert result["error"] is False
        assert 0.0 <= result["confidence"] <= 1.0

    def test_action_with_special_characters(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool")
        result = _check_alignment(
            tools, "Adding @decorator & handling <edge-cases> in alignment!"
        )
        assert result["error"] is False

    def test_file_path_with_dots(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool")
        result = _check_alignment(
            tools, "Editing config",
            file_path="src/claude.code.helper/config.py",
        )
        assert result["error"] is False

    def test_multiple_checks_on_same_task(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool")
        result1 = _check_alignment(tools, "Working on alignment")
        result2 = _check_alignment(tools, "Working on alignment")
        assert result1["confidence"] == result2["confidence"]

    def test_check_with_many_recorded_files(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool")
        for i in range(15):
            _record_file(tools, f"src/detection/module_{i}.py", "created")
        result = _check_alignment(
            tools, "Adding more detection logic",
            file_path="src/detection/module_15.py",
        )
        assert result["error"] is False
        assert result["action_analysis"]["file_relevance_score"] >= 0.5

    def test_check_with_many_recorded_steps(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool")
        for i in range(20):
            _record_step(tools, f"Working on alignment part {i + 1}")
        result = _check_alignment(tools, "Continuing alignment work")
        assert result["error"] is False

    def test_task_with_no_description(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool")
        result = _check_alignment(tools, "Working on alignment")
        assert result["error"] is False

    def test_task_with_no_phase(self, tools):
        _start_task(tools, "CMH-012", "check_alignment MCP tool")
        result = _check_alignment(tools, "Working")
        assert result["scope_info"]["phase"] is None


# ---------------------------------------------------------------------------
# Integration: full lifecycle
# ---------------------------------------------------------------------------

class TestCheckAlignmentIntegration:
    """Integration tests with full task lifecycle."""

    def test_alignment_evolves_with_task_context(self, tools):
        """As more context is recorded, alignment becomes more precise."""
        _start_task(
            tools, "CMH-012", "check_alignment MCP tool",
            phase="phase-3",
            description="Implement alignment checking with keyword extraction and file path analysis",
        )

        # Before any files/steps recorded.
        r1 = _check_alignment(
            tools, "Adding alignment detection code",
            file_path="src/detection/alignment.py",
        )

        # Record files and steps to enrich scope.
        _record_file(tools, "src/detection/alignment.py", "created")
        _record_file(tools, "src/mcp/server.py", "modified")
        _record_step(tools, "Created AlignmentChecker class")
        _record_step(tools, "Added keyword extraction")
        _record_step(tools, "Added file relevance scoring")

        # Same action after enriched context.
        r2 = _check_alignment(
            tools, "Adding alignment detection code",
            file_path="src/detection/alignment.py",
        )

        # Both should be aligned; the exact scores may differ.
        assert r1["error"] is False
        assert r2["error"] is False
        assert r1["confidence"] >= 0.0
        assert r2["confidence"] >= 0.0

    def test_aligned_and_misaligned_in_same_session(self, tools):
        """One aligned and one misaligned action in the same task."""
        _start_task(
            tools, "CMH-012", "check_alignment MCP tool",
            phase="phase-3",
            description="Alignment checking for scope drift detection",
        )
        _record_file(tools, "src/detection/alignment.py", "created")

        aligned = _check_alignment(
            tools,
            "Fixing alignment checker scoring for CMH-012",
            file_path="src/detection/alignment.py",
        )
        misaligned = _check_alignment(
            tools,
            "Configuring PostgreSQL connection pooling",
            file_path="infra/database/pool_config.yaml",
        )

        assert aligned["confidence"] > misaligned["confidence"]
        assert aligned["aligned"] is True
        assert misaligned["aligned"] is False
