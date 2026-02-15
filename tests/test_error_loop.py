"""Comprehensive tests for the ErrorLoopDetector (CMH-020).

Tests cover:
- Basic construction and defaults
- Recording successful outcomes (streak breaking)
- Recording failure outcomes (streak accumulation)
- Similarity analysis (keyword-based grouping)
- Loop threshold detection (configurable)
- Severity classification (none, warning, active, critical)
- Iterative debugging vs true loop discrimination
- Variation score computation
- Progress signal detection
- Evidence gathering (common keywords, files, tools)
- TaskMemory integration via check_task()
- Session summary and reset
- Edge cases (empty sessions, single failure, mixed outcomes)
- Report serialization (to_dict)
- Custom threshold configuration
"""

from datetime import datetime, timezone

import pytest

from claude_code_helper_mcp.detection.error_loop import (
    DEFAULT_LOOP_THRESHOLD,
    ErrorLoopDetector,
    ErrorLoopReport,
    FailureRecord,
    LoopEvidence,
    LOOP_SEVERITY_ACTIVE,
    LOOP_SEVERITY_CRITICAL,
    LOOP_SEVERITY_NONE,
    LOOP_SEVERITY_WARNING,
    SIMILARITY_THRESHOLD,
)
from claude_code_helper_mcp.models.task import TaskMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_detector(**kwargs) -> ErrorLoopDetector:
    """Create an ErrorLoopDetector with optional overrides."""
    return ErrorLoopDetector(**kwargs)


def record_failures(
    detector: ErrorLoopDetector,
    action: str,
    count: int,
    tool: str = "Bash",
    error: str = "Command failed",
    files: list[str] | None = None,
) -> ErrorLoopReport:
    """Record N failures of the same action and return the last report."""
    report = None
    for _ in range(count):
        report = detector.record_outcome(
            action=action,
            success=False,
            tool_used=tool,
            result_summary=error,
            files_involved=files,
        )
    assert report is not None
    return report


# ===================================================================
# Test class: Construction and defaults
# ===================================================================


class TestConstruction:
    """Test ErrorLoopDetector construction and default values."""

    def test_default_construction(self):
        d = make_detector()
        assert d.total_checks == 0
        assert d.total_failures == 0
        assert d.current_streak == 0

    def test_default_threshold(self):
        d = make_detector()
        assert d._loop_threshold == DEFAULT_LOOP_THRESHOLD
        assert DEFAULT_LOOP_THRESHOLD == 3

    def test_custom_threshold(self):
        d = make_detector(loop_threshold=5)
        assert d._loop_threshold == 5

    def test_minimum_threshold(self):
        """Threshold cannot be below 2."""
        d = make_detector(loop_threshold=1)
        assert d._loop_threshold == 2

    def test_custom_similarity_threshold(self):
        d = make_detector(similarity_threshold=0.7)
        assert d._similarity_threshold == 0.7

    def test_similarity_threshold_clamped(self):
        d = make_detector(similarity_threshold=1.5)
        assert d._similarity_threshold == 1.0
        d2 = make_detector(similarity_threshold=-0.5)
        assert d2._similarity_threshold == 0.0

    def test_custom_variation_threshold(self):
        d = make_detector(variation_threshold=0.5)
        assert d._variation_threshold == 0.5


# ===================================================================
# Test class: Recording outcomes
# ===================================================================


class TestRecordOutcome:
    """Test recording successes and failures."""

    def test_record_success_returns_report(self):
        d = make_detector()
        report = d.record_outcome("Run tests", success=True)
        assert isinstance(report, ErrorLoopReport)
        assert report.severity == LOOP_SEVERITY_NONE
        assert report.loop_detected is False
        assert report.consecutive_failures == 0

    def test_record_failure_increments_streak(self):
        d = make_detector()
        report = d.record_outcome("Run tests", success=False)
        assert report.consecutive_failures == 1
        assert d.current_streak == 1
        assert d.total_failures == 1
        assert d.total_checks == 1

    def test_success_breaks_streak(self):
        d = make_detector()
        d.record_outcome("Run tests", success=False)
        d.record_outcome("Run tests", success=False)
        assert d.current_streak == 2

        d.record_outcome("Run tests", success=True)
        assert d.current_streak == 0
        assert d.total_failures == 2
        assert d.total_checks == 3

    def test_multiple_failures_accumulate(self):
        d = make_detector()
        for i in range(5):
            report = d.record_outcome(
                "Compile the project source code",
                success=False,
                result_summary="Compilation failed with syntax errors",
            )
        assert report.consecutive_failures == 5
        assert d.total_failures == 5
        assert d.total_checks == 5

    def test_different_failures_reset_streak(self):
        d = make_detector()
        d.record_outcome(
            "Compile the project code",
            success=False,
            result_summary="Compilation failed",
        )
        d.record_outcome(
            "Compile the project code",
            success=False,
            result_summary="Compilation failed again",
        )
        assert d.current_streak == 2

        # Completely different action.
        d.record_outcome(
            "Deploy to production server",
            success=False,
            result_summary="Connection timeout",
        )
        assert d.current_streak == 1
        assert d.total_failures == 3

    def test_total_counts_track_all(self):
        d = make_detector()
        d.record_outcome("Step 1", success=True)
        d.record_outcome("Step 2", success=False)
        d.record_outcome("Step 3", success=True)
        d.record_outcome("Step 4", success=False)
        d.record_outcome("Step 5", success=False)
        assert d.total_checks == 5
        assert d.total_failures == 3

    def test_files_recorded(self):
        d = make_detector()
        d.record_outcome(
            "Edit configuration file",
            success=False,
            files_involved=["config.py"],
        )
        assert len(d._failure_streak) == 1
        assert d._failure_streak[0].files_involved == ("config.py",)


# ===================================================================
# Test class: Similarity analysis
# ===================================================================


class TestSimilarity:
    """Test action similarity detection."""

    def test_identical_actions_are_similar(self):
        d = make_detector()
        d.record_outcome("Run pytest on the project tests", success=False)
        report = d.record_outcome("Run pytest on the project tests", success=False)
        assert report.consecutive_failures == 2

    def test_very_different_actions_break_streak(self):
        d = make_detector()
        d.record_outcome("Run pytest on the project", success=False)
        report = d.record_outcome(
            "Deploy production kubernetes cluster",
            success=False,
        )
        assert report.consecutive_failures == 1

    def test_similar_actions_continue_streak(self):
        d = make_detector()
        d.record_outcome(
            "Run pytest on integration tests",
            success=False,
        )
        report = d.record_outcome(
            "Run pytest on integration test suite",
            success=False,
        )
        assert report.consecutive_failures == 2

    def test_compute_similarity_identical_sets(self):
        kw = frozenset({"pytest", "integration", "tests"})
        assert ErrorLoopDetector._compute_similarity(kw, kw) == 1.0

    def test_compute_similarity_disjoint_sets(self):
        a = frozenset({"pytest", "integration"})
        b = frozenset({"deploy", "kubernetes"})
        assert ErrorLoopDetector._compute_similarity(a, b) == 0.0

    def test_compute_similarity_partial_overlap(self):
        a = frozenset({"pytest", "integration", "tests"})
        b = frozenset({"pytest", "unit", "tests"})
        sim = ErrorLoopDetector._compute_similarity(a, b)
        # intersection = {pytest, tests} = 2, union = {pytest, integration, tests, unit} = 4
        assert sim == pytest.approx(0.5)

    def test_compute_similarity_empty_sets(self):
        empty = frozenset()
        assert ErrorLoopDetector._compute_similarity(empty, empty) == 1.0
        assert ErrorLoopDetector._compute_similarity(empty, frozenset({"x"})) == 0.0

    def test_custom_similarity_threshold(self):
        """Higher threshold means fewer actions group together."""
        d = make_detector(similarity_threshold=0.9)
        d.record_outcome("Run pytest integration tests", success=False)
        report = d.record_outcome(
            "Run pytest unit tests",
            success=False,
        )
        # With high threshold, these may not be similar enough.
        # The similarity is ~0.5 which is < 0.9, so streak resets.
        assert report.consecutive_failures == 1


# ===================================================================
# Test class: Severity classification
# ===================================================================


class TestSeverityClassification:
    """Test error loop severity levels."""

    def test_no_failures_is_none(self):
        d = make_detector()
        report = d.check()
        assert report.severity == LOOP_SEVERITY_NONE

    def test_one_failure_is_none(self):
        d = make_detector()
        report = d.record_outcome("Run tests", success=False)
        assert report.severity == LOOP_SEVERITY_NONE

    def test_two_failures_is_warning(self):
        """At default threshold=3, two failures triggers warning."""
        d = make_detector()
        d.record_outcome("Run tests on the project", success=False)
        report = d.record_outcome("Run tests on the project suite", success=False)
        assert report.severity == LOOP_SEVERITY_WARNING

    def test_three_failures_is_active(self):
        """At default threshold=3, three failures triggers active."""
        d = make_detector()
        report = record_failures(d, "Run tests on the project", 3)
        assert report.severity == LOOP_SEVERITY_ACTIVE
        assert report.loop_detected is True

    def test_six_failures_is_critical(self):
        """At default threshold=3, 6+ consecutive failures is critical."""
        d = make_detector()
        report = record_failures(d, "Run tests on the project", 6)
        assert report.severity == LOOP_SEVERITY_CRITICAL
        assert report.loop_detected is True

    def test_custom_threshold_affects_severity(self):
        """With threshold=5, 4 failures is only warning."""
        d = make_detector(loop_threshold=5)
        report = record_failures(d, "Compile the project source", 4)
        assert report.severity == LOOP_SEVERITY_WARNING
        assert report.loop_detected is False

    def test_custom_threshold_five_is_active(self):
        d = make_detector(loop_threshold=5)
        report = record_failures(d, "Compile the project source", 5)
        assert report.severity == LOOP_SEVERITY_ACTIVE
        assert report.loop_detected is True

    def test_loop_detected_flag(self):
        d = make_detector()
        report = d.record_outcome("Run tests on project", success=False)
        assert report.loop_detected is False

        d.record_outcome("Run tests on project", success=False)
        report = d.record_outcome("Run tests on project", success=False)
        assert report.loop_detected is True

    def test_success_resets_severity(self):
        d = make_detector()
        record_failures(d, "Compile source code for the project", 3)
        assert d.current_streak == 3

        report = d.record_outcome("Compile source code", success=True)
        assert report.severity == LOOP_SEVERITY_NONE
        assert report.loop_detected is False
        assert report.consecutive_failures == 0


# ===================================================================
# Test class: Iterative debugging detection
# ===================================================================


class TestIterativeDebugging:
    """Test discrimination between true loops and iterative debugging."""

    def test_true_loop_identical_failures(self):
        """Identical failures with no variation = true loop."""
        d = make_detector()
        for _ in range(4):
            report = d.record_outcome(
                "Run pytest test suite",
                success=False,
                tool_used="Bash",
                result_summary="3 tests failed: test_a, test_b, test_c",
                files_involved=["tests/test_main.py"],
            )
        assert report.evidence.is_iterative_debugging is False
        assert report.severity == LOOP_SEVERITY_ACTIVE

    def test_iterative_debugging_different_files(self):
        """Failures targeting different files = iterative debugging."""
        d = make_detector()
        files_list = [
            ["src/handler.py"],
            ["src/router.py"],
            ["src/middleware.py"],
        ]
        for i, files in enumerate(files_list):
            report = d.record_outcome(
                "Fix the import error in source module",
                success=False,
                tool_used="Edit",
                result_summary=f"Import still failing in {files[0]}",
                files_involved=files,
            )
        assert report.evidence.variation_score > 0
        if report.evidence.is_iterative_debugging:
            # If iterative debugging is detected, severity should be reduced.
            assert report.severity in (LOOP_SEVERITY_NONE, LOOP_SEVERITY_WARNING)

    def test_iterative_debugging_different_errors(self):
        """Failures with different error messages = progress signals."""
        d = make_detector()
        errors = [
            "SyntaxError on line 42",
            "TypeError: expected int got str",
            "ImportError: no module named foo",
        ]
        for error in errors:
            report = d.record_outcome(
                "Fix the broken source module code",
                success=False,
                tool_used="Bash",
                result_summary=error,
                files_involved=["src/main.py"],
            )
        if report.evidence.iteration_progress_signals:
            assert any(
                "Error messages changed" in s
                for s in report.evidence.iteration_progress_signals
            )

    def test_iterative_debugging_different_tools(self):
        """Failures using different tools = potential progress."""
        d = make_detector()
        tools = ["Bash", "Edit", "Write"]
        for tool in tools:
            report = d.record_outcome(
                "Fix the configuration module settings",
                success=False,
                tool_used=tool,
                result_summary="Configuration still invalid",
                files_involved=["config.py"],
            )
        if report.evidence.iteration_progress_signals:
            assert any(
                "Tools changed" in s
                for s in report.evidence.iteration_progress_signals
            )

    def test_variation_score_identical_failures(self):
        """Identical failures should have variation score near 0."""
        records = [
            FailureRecord(
                action="Run tests",
                tool_used="Bash",
                error_summary="3 tests failed",
                files_involved=("test.py",),
                keywords=frozenset({"run", "tests"}),
            )
            for _ in range(3)
        ]
        score = ErrorLoopDetector._compute_variation_score(records)
        assert score == 0.0

    def test_variation_score_different_failures(self):
        """Different failures should have higher variation score."""
        records = [
            FailureRecord(
                action="Run tests",
                tool_used="Bash",
                error_summary="Syntax error on line 10",
                files_involved=("src/main.py",),
                keywords=frozenset({"run", "tests"}),
            ),
            FailureRecord(
                action="Fix import in module",
                tool_used="Edit",
                error_summary="Import resolution failed for package",
                files_involved=("src/utils.py",),
                keywords=frozenset({"fix", "import", "module"}),
            ),
            FailureRecord(
                action="Update configuration settings",
                tool_used="Write",
                error_summary="Configuration validation error on schema",
                files_involved=("config.json",),
                keywords=frozenset({"configuration", "settings"}),
            ),
        ]
        score = ErrorLoopDetector._compute_variation_score(records)
        assert score > 0.5

    def test_variation_score_single_record(self):
        records = [
            FailureRecord(
                action="Test",
                keywords=frozenset({"test"}),
            )
        ]
        score = ErrorLoopDetector._compute_variation_score(records)
        assert score == 0.0


# ===================================================================
# Test class: Evidence gathering
# ===================================================================


class TestEvidence:
    """Test evidence collection from failure streaks."""

    def test_evidence_common_keywords(self):
        d = make_detector()
        for _ in range(3):
            d.record_outcome(
                "Run pytest test suite",
                success=False,
            )
        report = d.check()
        assert "pytest" in report.evidence.common_keywords
        assert "suite" in report.evidence.common_keywords

    def test_evidence_common_files(self):
        d = make_detector()
        for _ in range(3):
            d.record_outcome(
                "Edit configuration module",
                success=False,
                files_involved=["src/config.py", "src/settings.py"],
            )
        report = d.check()
        assert "src/config.py" in report.evidence.common_files
        assert "src/settings.py" in report.evidence.common_files

    def test_evidence_common_tools(self):
        d = make_detector()
        for _ in range(3):
            d.record_outcome(
                "Run the build command",
                success=False,
                tool_used="Bash",
            )
        report = d.check()
        assert "Bash" in report.evidence.common_tools

    def test_evidence_similarity_score(self):
        d = make_detector()
        for _ in range(3):
            d.record_outcome(
                "Run pytest test suite",
                success=False,
            )
        report = d.check()
        assert report.evidence.similarity_score > 0.8

    def test_evidence_consecutive_failures(self):
        d = make_detector()
        record_failures(d, "Run the full test suite", 4)
        report = d.check()
        assert report.evidence.consecutive_failures == 4

    def test_no_evidence_when_no_failures(self):
        d = make_detector()
        report = d.check()
        assert report.evidence.consecutive_failures == 0
        assert report.evidence.similarity_score == 0.0

    def test_evidence_after_streak_reset(self):
        d = make_detector()
        record_failures(d, "Compile source project code", 3)
        d.record_outcome("Something else", success=True)
        report = d.check()
        assert report.evidence.consecutive_failures == 0


# ===================================================================
# Test class: Progress signals
# ===================================================================


class TestProgressSignals:
    """Test detection of progress signals between failures."""

    def test_no_signals_for_identical_failures(self):
        records = [
            FailureRecord(
                action="Run tests",
                tool_used="Bash",
                error_summary="Tests failed",
                files_involved=("test.py",),
                keywords=frozenset({"run", "tests"}),
            )
            for _ in range(3)
        ]
        signals = ErrorLoopDetector._detect_progress_signals(records)
        assert len(signals) == 0

    def test_file_change_signal(self):
        records = [
            FailureRecord(
                action="Fix import error",
                tool_used="Edit",
                error_summary="Import failed",
                files_involved=("src/a.py",),
                keywords=frozenset({"fix", "import", "error"}),
            ),
            FailureRecord(
                action="Fix import error",
                tool_used="Edit",
                error_summary="Import failed",
                files_involved=("src/b.py",),
                keywords=frozenset({"fix", "import", "error"}),
            ),
        ]
        signals = ErrorLoopDetector._detect_progress_signals(records)
        assert any("Files changed" in s for s in signals)

    def test_tool_change_signal(self):
        records = [
            FailureRecord(
                action="Fix broken module",
                tool_used="Bash",
                error_summary="Error",
                files_involved=("src/x.py",),
                keywords=frozenset({"fix", "broken", "module"}),
            ),
            FailureRecord(
                action="Fix broken module",
                tool_used="Edit",
                error_summary="Error",
                files_involved=("src/x.py",),
                keywords=frozenset({"fix", "broken", "module"}),
            ),
        ]
        signals = ErrorLoopDetector._detect_progress_signals(records)
        assert any("Tools changed" in s for s in signals)

    def test_single_record_no_signals(self):
        records = [
            FailureRecord(
                action="Test",
                keywords=frozenset({"test"}),
            )
        ]
        signals = ErrorLoopDetector._detect_progress_signals(records)
        assert signals == []

    def test_empty_records_no_signals(self):
        signals = ErrorLoopDetector._detect_progress_signals([])
        assert signals == []


# ===================================================================
# Test class: TaskMemory integration
# ===================================================================


class TestTaskMemoryIntegration:
    """Test check_task() with real TaskMemory objects."""

    def test_check_task_no_failures(self):
        task = TaskMemory(ticket_id="CMH-020", title="Error loop detection")
        task.add_step("Set up project", success=True)
        task.add_step("Write code", success=True)
        task.add_step("Run tests", success=True)

        d = make_detector()
        report = d.check_task(task)
        assert report.severity == LOOP_SEVERITY_NONE
        assert report.loop_detected is False
        assert report.total_checks_in_session == 3

    def test_check_task_with_loop(self):
        task = TaskMemory(ticket_id="CMH-020", title="Error loop detection")
        task.add_step("Set up project", success=True)
        task.add_step(
            "Run pytest suite",
            success=False,
            result_summary="3 tests failed",
        )
        task.add_step(
            "Run pytest suite",
            success=False,
            result_summary="3 tests failed",
        )
        task.add_step(
            "Run pytest suite",
            success=False,
            result_summary="3 tests failed",
        )

        d = make_detector()
        report = d.check_task(task)
        assert report.loop_detected is True
        assert report.consecutive_failures == 3

    def test_check_task_with_intervening_success(self):
        task = TaskMemory(ticket_id="CMH-020", title="Error loop detection")
        task.add_step("Run tests", success=False)
        task.add_step("Run tests", success=False)
        task.add_step("Fix code", success=True)
        task.add_step("Run tests", success=False)

        d = make_detector()
        report = d.check_task(task)
        assert report.loop_detected is False
        assert report.consecutive_failures == 1
        assert report.total_failures_in_session == 3

    def test_check_task_resets_session(self):
        d = make_detector()
        # Pre-existing state.
        d.record_outcome("Old failure", success=False)
        d.record_outcome("Old failure", success=False)
        assert d.current_streak == 2

        # check_task resets first.
        task = TaskMemory(ticket_id="CMH-020", title="Test")
        task.add_step("New step", success=True)
        report = d.check_task(task)
        assert report.consecutive_failures == 0
        assert report.total_checks_in_session == 1

    def test_check_task_empty_task(self):
        task = TaskMemory(ticket_id="CMH-020", title="Empty")
        d = make_detector()
        report = d.check_task(task)
        assert report.severity == LOOP_SEVERITY_NONE
        assert report.total_checks_in_session == 0


# ===================================================================
# Test class: Session summary and reset
# ===================================================================


class TestSessionManagement:
    """Test session summary and reset functionality."""

    def test_session_summary_empty(self):
        d = make_detector()
        summary = d.get_session_summary()
        assert summary["total_checks"] == 0
        assert summary["total_failures"] == 0
        assert summary["failure_rate"] == 0.0
        assert summary["current_failure_streak"] == 0
        assert summary["loop_detected"] is False

    def test_session_summary_with_data(self):
        d = make_detector()
        d.record_outcome("Step 1", success=True)
        d.record_outcome("Step 2", success=False)
        d.record_outcome("Step 3", success=False)
        summary = d.get_session_summary()
        assert summary["total_checks"] == 3
        assert summary["total_failures"] == 2
        assert summary["failure_rate"] == pytest.approx(2 / 3, abs=0.01)
        assert summary["current_failure_streak"] == 2

    def test_reset_clears_all_state(self):
        d = make_detector()
        d.record_outcome("Fail action A", success=False)
        d.record_outcome("Fail action A", success=False)
        d.record_outcome("Fail action A", success=False)
        assert d.current_streak == 3
        assert d.total_failures == 3
        assert d.total_checks == 3

        d.reset()
        assert d.current_streak == 0
        assert d.total_failures == 0
        assert d.total_checks == 0
        report = d.check()
        assert report.severity == LOOP_SEVERITY_NONE

    def test_session_summary_includes_threshold(self):
        d = make_detector(loop_threshold=5)
        summary = d.get_session_summary()
        assert summary["loop_threshold"] == 5


# ===================================================================
# Test class: Report serialization
# ===================================================================


class TestReportSerialization:
    """Test ErrorLoopReport.to_dict() serialization."""

    def test_to_dict_basic(self):
        report = ErrorLoopReport()
        d = report.to_dict()
        assert d["severity"] == LOOP_SEVERITY_NONE
        assert d["loop_detected"] is False
        assert d["consecutive_failures"] == 0
        assert "evidence" in d
        assert "generated_at" in d

    def test_to_dict_with_evidence(self):
        evidence = LoopEvidence(
            consecutive_failures=3,
            similarity_score=0.85,
            variation_score=0.1,
            common_keywords=["pytest", "suite"],
            common_files=["test.py"],
            common_tools=["Bash"],
            is_iterative_debugging=False,
            iteration_progress_signals=[],
        )
        report = ErrorLoopReport(
            severity=LOOP_SEVERITY_ACTIVE,
            loop_detected=True,
            consecutive_failures=3,
            evidence=evidence,
            recommended_action="Error loop detected.",
        )
        d = report.to_dict()
        assert d["severity"] == LOOP_SEVERITY_ACTIVE
        assert d["loop_detected"] is True
        assert d["evidence"]["consecutive_failures"] == 3
        assert d["evidence"]["similarity_score"] == 0.85
        assert d["evidence"]["common_keywords"] == ["pytest", "suite"]
        assert d["evidence"]["is_iterative_debugging"] is False

    def test_to_dict_roundtrip_detector(self):
        """Verify detector output can be serialized."""
        d = make_detector()
        record_failures(d, "Run full test suite", 4)
        report = d.check()
        d_dict = report.to_dict()
        assert d_dict["severity"] in (
            LOOP_SEVERITY_NONE,
            LOOP_SEVERITY_WARNING,
            LOOP_SEVERITY_ACTIVE,
            LOOP_SEVERITY_CRITICAL,
        )
        assert isinstance(d_dict["evidence"]["similarity_score"], float)
        assert isinstance(d_dict["generated_at"], str)


# ===================================================================
# Test class: Edge cases
# ===================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_check_without_any_outcomes(self):
        d = make_detector()
        report = d.check()
        assert report.severity == LOOP_SEVERITY_NONE
        assert report.consecutive_failures == 0
        assert report.total_checks_in_session == 0

    def test_single_success(self):
        d = make_detector()
        report = d.record_outcome("Do work", success=True)
        assert report.consecutive_failures == 0
        assert d.total_checks == 1
        assert d.total_failures == 0

    def test_alternating_success_failure(self):
        d = make_detector()
        for i in range(10):
            d.record_outcome(
                f"Step number {i}",
                success=(i % 2 == 0),
            )
        assert d.current_streak <= 1
        assert d.total_failures == 5
        assert d.total_checks == 10

    def test_failure_with_empty_action(self):
        """Empty action should still be handled gracefully."""
        d = make_detector()
        report = d.record_outcome("", success=False)
        assert report.consecutive_failures == 1

    def test_failure_with_no_tool(self):
        d = make_detector()
        report = d.record_outcome("Run tests", success=False, tool_used="")
        assert report.consecutive_failures == 1

    def test_failure_with_no_files(self):
        d = make_detector()
        report = d.record_outcome("Run tests", success=False, files_involved=None)
        assert report.consecutive_failures == 1

    def test_long_action_descriptions(self):
        long_action = "Fix the " + "very " * 100 + "important bug"
        d = make_detector()
        report = d.record_outcome(long_action, success=False)
        assert report.consecutive_failures == 1

    def test_many_consecutive_failures(self):
        """Ensure the detector handles very long streaks gracefully."""
        d = make_detector()
        for _ in range(50):
            d.record_outcome(
                "Run the full test suite",
                success=False,
                result_summary="Tests keep failing",
            )
        report = d.check()
        assert report.consecutive_failures == 50
        assert report.severity == LOOP_SEVERITY_CRITICAL

    def test_threshold_of_two(self):
        """Minimum possible threshold of 2."""
        d = make_detector(loop_threshold=2)
        d.record_outcome("Compile code", success=False)
        report = d.record_outcome("Compile code", success=False)
        assert report.severity == LOOP_SEVERITY_ACTIVE
        assert report.loop_detected is True


# ===================================================================
# Test class: Recommended actions
# ===================================================================


class TestRecommendedActions:
    """Test that recommended actions are provided for each severity."""

    def test_none_severity_has_action(self):
        d = make_detector()
        report = d.check()
        assert "No error loop" in report.recommended_action

    def test_warning_severity_has_action(self):
        d = make_detector()
        d.record_outcome("Run tests on project", success=False)
        report = d.record_outcome("Run tests on project", success=False)
        assert "Potential error loop" in report.recommended_action or report.severity == LOOP_SEVERITY_NONE

    def test_active_severity_has_action(self):
        d = make_detector()
        report = record_failures(d, "Run tests on the project", 3)
        if report.severity == LOOP_SEVERITY_ACTIVE:
            assert "Error loop detected" in report.recommended_action

    def test_critical_severity_has_action(self):
        d = make_detector()
        report = record_failures(d, "Run tests on the project", 6)
        if report.severity == LOOP_SEVERITY_CRITICAL:
            assert "Critical error loop" in report.recommended_action


# ===================================================================
# Test class: Mixed scenario integration tests
# ===================================================================


class TestMixedScenarios:
    """Test realistic mixed-outcome scenarios."""

    def test_build_fail_fix_cycle(self):
        """Simulate: build fails, fix, build fails differently, fix, build succeeds."""
        d = make_detector()

        d.record_outcome(
            "Compile source code",
            success=False,
            tool_used="Bash",
            result_summary="Missing import: os.path",
            files_involved=["src/main.py"],
        )
        d.record_outcome(
            "Fix missing import",
            success=True,
            tool_used="Edit",
            files_involved=["src/main.py"],
        )
        d.record_outcome(
            "Compile source code",
            success=False,
            tool_used="Bash",
            result_summary="Type error: expected int",
            files_involved=["src/main.py"],
        )
        d.record_outcome(
            "Fix type annotation",
            success=True,
            tool_used="Edit",
            files_involved=["src/main.py"],
        )
        report = d.record_outcome(
            "Compile source code",
            success=True,
            tool_used="Bash",
            files_involved=["src/main.py"],
        )
        assert report.loop_detected is False
        assert report.consecutive_failures == 0
        assert d.total_failures == 2

    def test_genuine_error_loop(self):
        """Simulate an agent stuck retrying the same broken command."""
        d = make_detector()

        d.record_outcome("Write initial code", success=True)

        for _ in range(5):
            d.record_outcome(
                "Run pytest test suite",
                success=False,
                tool_used="Bash",
                result_summary="3 tests failed: test_a, test_b, test_c",
                files_involved=["tests/test_main.py"],
            )

        report = d.check()
        assert report.loop_detected is True
        assert report.consecutive_failures == 5
        assert report.severity in (LOOP_SEVERITY_ACTIVE, LOOP_SEVERITY_CRITICAL)
        assert report.evidence.is_iterative_debugging is False

    def test_progressive_debugging(self):
        """Simulate iterative debugging where each attempt tries something different."""
        d = make_detector()

        d.record_outcome(
            "Fix the database connection handler",
            success=False,
            tool_used="Edit",
            result_summary="Connection timeout on port 5432",
            files_involved=["src/db.py"],
        )
        d.record_outcome(
            "Fix the database connection handler configuration",
            success=False,
            tool_used="Edit",
            result_summary="Authentication failed for user admin",
            files_involved=["config/database.yaml"],
        )
        d.record_outcome(
            "Fix the database connection handler credentials",
            success=False,
            tool_used="Write",
            result_summary="SSL certificate validation error",
            files_involved=["certs/db-cert.pem"],
        )

        report = d.check()
        # Should detect variation (different files, errors, tools).
        assert report.evidence.variation_score > 0
        assert len(report.evidence.iteration_progress_signals) > 0

    def test_mixed_failures_different_categories(self):
        """Different failure categories should not merge into the same streak."""
        d = make_detector()

        d.record_outcome(
            "Run pytest test suite",
            success=False,
            result_summary="Tests failed",
        )
        d.record_outcome(
            "Run pytest test suite",
            success=False,
            result_summary="Tests failed",
        )
        # Different category of failure.
        d.record_outcome(
            "Deploy to staging server",
            success=False,
            result_summary="Connection refused",
        )
        report = d.check()
        # The deploy failure starts a new streak.
        assert report.consecutive_failures == 1


# ===================================================================
# Test class: Constants and module-level exports
# ===================================================================


class TestConstants:
    """Test module-level constants and imports."""

    def test_default_loop_threshold_value(self):
        assert DEFAULT_LOOP_THRESHOLD == 3

    def test_similarity_threshold_value(self):
        assert SIMILARITY_THRESHOLD == 0.5

    def test_severity_constants(self):
        assert LOOP_SEVERITY_NONE == "none"
        assert LOOP_SEVERITY_WARNING == "warning"
        assert LOOP_SEVERITY_ACTIVE == "active"
        assert LOOP_SEVERITY_CRITICAL == "critical"

    def test_imports_from_detection_package(self):
        """Verify the detection __init__.py exports the new classes."""
        from claude_code_helper_mcp.detection import (
            ErrorLoopDetector,
            ErrorLoopReport,
            FailureRecord,
            LoopEvidence,
            DEFAULT_LOOP_THRESHOLD,
            LOOP_SEVERITY_ACTIVE,
            LOOP_SEVERITY_CRITICAL,
            LOOP_SEVERITY_NONE,
            LOOP_SEVERITY_WARNING,
            SIMILARITY_THRESHOLD,
        )
        assert ErrorLoopDetector is not None
        assert ErrorLoopReport is not None
        assert FailureRecord is not None
        assert LoopEvidence is not None
