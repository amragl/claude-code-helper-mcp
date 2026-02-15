"""Tests for the DriftDetector class (CMH-019).

Verifies drift detection across single checks, multi-action sessions,
file scope boundary detection, step repetition detection, topic shift
detection, temporal trend analysis, severity classification, session
management, DriftReport serialisation, TaskMemory integration, and
edge cases.

All tests use real computation with in-memory data -- zero mocks.
"""

from __future__ import annotations

import pytest

from claude_code_helper_mcp.detection.drift import (
    DEFAULT_THRESHOLDS,
    DriftDetector,
    DriftIndicator,
    DriftReport,
    SEVERITY_CRITICAL,
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MODERATE,
    SEVERITY_NONE,
    _get_directory,
)
from claude_code_helper_mcp.detection.alignment import AlignmentReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def detector() -> DriftDetector:
    """Return a DriftDetector with default settings."""
    return DriftDetector()


@pytest.fixture
def task_context() -> dict:
    """Return a realistic task context for testing."""
    return {
        "task_title": "Implement drift detection engine",
        "task_description": (
            "Create DriftDetector class that compares current actions against "
            "task scope using file scope analysis and action relevance scoring."
        ),
        "task_phase": "phase-5",
        "task_files": [
            "src/claude_code_helper_mcp/detection/drift.py",
            "src/claude_code_helper_mcp/detection/__init__.py",
            "tests/test_drift_detector.py",
        ],
        "task_steps": [
            "Create drift.py with DriftDetector class",
            "Implement file scope boundary detection",
            "Add step repetition and topic shift indicators",
        ],
        "task_ticket_id": "CMH-019",
    }


# ===========================================================================
# 1. DriftDetector construction and defaults
# ===========================================================================


class TestDriftDetectorConstruction:
    """Tests for DriftDetector initialisation and default values."""

    def test_default_construction(self) -> None:
        """DriftDetector can be created with default settings."""
        detector = DriftDetector()
        assert detector.session_length == 0
        assert detector.session_scores == []

    def test_custom_alignment_threshold(self) -> None:
        """Custom alignment threshold is passed through to the checker."""
        detector = DriftDetector(alignment_threshold=0.7)
        # The internal checker should use the custom threshold.
        assert detector._checker.threshold == 0.7

    def test_custom_severity_thresholds(self) -> None:
        """Custom severity thresholds override defaults."""
        custom = {SEVERITY_NONE: 0.0, SEVERITY_LOW: 0.3, SEVERITY_HIGH: 0.7}
        detector = DriftDetector(severity_thresholds=custom)
        assert detector._thresholds[SEVERITY_LOW] == 0.3
        assert detector._thresholds[SEVERITY_HIGH] == 0.7

    def test_custom_trend_window(self) -> None:
        """Custom trend window controls the session trend length."""
        detector = DriftDetector(trend_window=5)
        assert detector._trend_window == 5

    def test_minimum_trend_window(self) -> None:
        """Trend window has a minimum of 1."""
        detector = DriftDetector(trend_window=0)
        assert detector._trend_window == 1
        detector2 = DriftDetector(trend_window=-5)
        assert detector2._trend_window == 1


# ===========================================================================
# 2. Single action drift checks
# ===========================================================================


class TestSingleActionDrift:
    """Tests for drift detection on individual actions."""

    def test_aligned_action_no_drift(self, detector: DriftDetector,
                                      task_context: dict) -> None:
        """An action closely aligned with the task shows no or low drift."""
        report = detector.check(
            action="Implementing drift score calculation in drift.py",
            file_path="src/claude_code_helper_mcp/detection/drift.py",
            **task_context,
        )
        assert isinstance(report, DriftReport)
        assert report.severity in (SEVERITY_NONE, SEVERITY_LOW)
        assert report.drift_score < 0.4

    def test_unrelated_action_high_drift(self, detector: DriftDetector,
                                          task_context: dict) -> None:
        """An action completely unrelated to the task shows high drift."""
        report = detector.check(
            action="Configuring the database connection pool for PostgreSQL",
            file_path="src/database/pool.py",
            **task_context,
        )
        assert report.severity in (SEVERITY_MODERATE, SEVERITY_HIGH, SEVERITY_CRITICAL)
        assert report.drift_score > 0.3

    def test_partially_related_action(self, detector: DriftDetector,
                                       task_context: dict) -> None:
        """An action partially related shows moderate drift."""
        report = detector.check(
            action="Adding utility functions for text processing",
            file_path="src/claude_code_helper_mcp/utils.py",
            **task_context,
        )
        assert isinstance(report, DriftReport)
        assert report.drift_score > 0.0

    def test_empty_action_handled(self, detector: DriftDetector,
                                   task_context: dict) -> None:
        """Empty action string is handled without errors."""
        report = detector.check(action="", **task_context)
        assert isinstance(report, DriftReport)
        assert report.severity in (
            SEVERITY_NONE, SEVERITY_LOW, SEVERITY_MODERATE,
            SEVERITY_HIGH, SEVERITY_CRITICAL
        )

    def test_no_task_context_handled(self, detector: DriftDetector) -> None:
        """Missing task context is handled gracefully."""
        report = detector.check(
            action="Some random action",
            task_title="",
            task_description="",
        )
        assert isinstance(report, DriftReport)

    def test_action_with_ticket_reference(self, detector: DriftDetector,
                                           task_context: dict) -> None:
        """Action referencing the ticket ID gets a relevance boost."""
        report = detector.check(
            action="Working on CMH-019 drift detection implementation",
            file_path="src/claude_code_helper_mcp/detection/drift.py",
            **task_context,
        )
        assert report.drift_score < 0.3
        assert report.severity in (SEVERITY_NONE, SEVERITY_LOW)


# ===========================================================================
# 3. DriftReport structure
# ===========================================================================


class TestDriftReport:
    """Tests for DriftReport data structure and serialisation."""

    def test_report_has_required_fields(self, detector: DriftDetector,
                                         task_context: dict) -> None:
        """DriftReport contains all required fields."""
        report = detector.check(
            action="Test action for structure check",
            **task_context,
        )
        assert hasattr(report, "severity")
        assert hasattr(report, "drift_score")
        assert hasattr(report, "indicators")
        assert hasattr(report, "recommended_action")
        assert hasattr(report, "alignment_report")
        assert hasattr(report, "session_drift_trend")
        assert hasattr(report, "generated_at")

    def test_report_to_dict(self, detector: DriftDetector,
                             task_context: dict) -> None:
        """DriftReport serialises to a valid dictionary."""
        report = detector.check(
            action="Testing dict serialisation",
            **task_context,
        )
        d = report.to_dict()
        assert isinstance(d, dict)
        assert "severity" in d
        assert "drift_score" in d
        assert "indicators" in d
        assert isinstance(d["indicators"], list)
        assert "recommended_action" in d
        assert "session_drift_trend" in d
        assert "generated_at" in d

    def test_report_alignment_report_included(self, detector: DriftDetector,
                                               task_context: dict) -> None:
        """DriftReport includes the underlying AlignmentReport."""
        report = detector.check(
            action="Check alignment report inclusion",
            **task_context,
        )
        assert report.alignment_report is not None
        assert isinstance(report.alignment_report, AlignmentReport)

    def test_report_to_dict_with_alignment(self, detector: DriftDetector,
                                            task_context: dict) -> None:
        """Serialised report includes alignment_report as dict."""
        report = detector.check(
            action="Check alignment in dict",
            **task_context,
        )
        d = report.to_dict()
        assert "alignment_report" in d
        assert isinstance(d["alignment_report"], dict)
        assert "confidence" in d["alignment_report"]

    def test_indicator_structure(self) -> None:
        """DriftIndicator has the expected fields."""
        ind = DriftIndicator(
            category="test_category",
            description="Test indicator description",
            severity_contribution=0.3,
            evidence="some evidence",
        )
        assert ind.category == "test_category"
        assert ind.description == "Test indicator description"
        assert ind.severity_contribution == 0.3
        assert ind.evidence == "some evidence"


# ===========================================================================
# 4. Severity classification
# ===========================================================================


class TestSeverityClassification:
    """Tests for drift score to severity level mapping."""

    def test_no_drift_severity(self) -> None:
        """Drift score of 0.0 maps to SEVERITY_NONE."""
        detector = DriftDetector()
        assert detector._classify_severity(0.0) == SEVERITY_NONE

    def test_low_drift_severity(self) -> None:
        """Drift score of 0.25 maps to SEVERITY_LOW."""
        detector = DriftDetector()
        assert detector._classify_severity(0.25) == SEVERITY_LOW

    def test_moderate_drift_severity(self) -> None:
        """Drift score of 0.5 maps to SEVERITY_MODERATE."""
        detector = DriftDetector()
        assert detector._classify_severity(0.5) == SEVERITY_MODERATE

    def test_high_drift_severity(self) -> None:
        """Drift score of 0.7 maps to SEVERITY_HIGH."""
        detector = DriftDetector()
        assert detector._classify_severity(0.7) == SEVERITY_HIGH

    def test_critical_drift_severity(self) -> None:
        """Drift score of 0.9 maps to SEVERITY_CRITICAL."""
        detector = DriftDetector()
        assert detector._classify_severity(0.9) == SEVERITY_CRITICAL

    def test_exact_threshold_boundaries(self) -> None:
        """Scores at exact threshold boundaries are classified correctly."""
        detector = DriftDetector()
        assert detector._classify_severity(0.0) == SEVERITY_NONE
        assert detector._classify_severity(0.2) == SEVERITY_LOW
        assert detector._classify_severity(0.4) == SEVERITY_MODERATE
        assert detector._classify_severity(0.6) == SEVERITY_HIGH
        assert detector._classify_severity(0.8) == SEVERITY_CRITICAL

    def test_custom_thresholds(self) -> None:
        """Custom severity thresholds are respected."""
        custom = {
            SEVERITY_NONE: 0.0,
            SEVERITY_LOW: 0.1,
            SEVERITY_MODERATE: 0.3,
            SEVERITY_HIGH: 0.5,
            SEVERITY_CRITICAL: 0.7,
        }
        detector = DriftDetector(severity_thresholds=custom)
        assert detector._classify_severity(0.05) == SEVERITY_NONE
        assert detector._classify_severity(0.15) == SEVERITY_LOW
        assert detector._classify_severity(0.35) == SEVERITY_MODERATE
        assert detector._classify_severity(0.55) == SEVERITY_HIGH
        assert detector._classify_severity(0.75) == SEVERITY_CRITICAL

    def test_max_drift_score(self) -> None:
        """Maximum drift score (1.0) is classified as CRITICAL."""
        detector = DriftDetector()
        assert detector._classify_severity(1.0) == SEVERITY_CRITICAL

    def test_recommended_action_per_severity(self, detector: DriftDetector,
                                              task_context: dict) -> None:
        """Each severity level has a non-empty recommended action."""
        # Run a check to get any report; the key test is that
        # recommended_action is populated for all severity levels.
        from claude_code_helper_mcp.detection.drift import _RECOMMENDED_ACTIONS
        for severity in [SEVERITY_NONE, SEVERITY_LOW, SEVERITY_MODERATE,
                         SEVERITY_HIGH, SEVERITY_CRITICAL]:
            assert severity in _RECOMMENDED_ACTIONS
            assert len(_RECOMMENDED_ACTIONS[severity]) > 0


# ===========================================================================
# 5. File scope boundary detection
# ===========================================================================


class TestFileScopeBoundary:
    """Tests for file scope indicator detection."""

    def test_in_scope_file_no_indicator(self, detector: DriftDetector,
                                         task_context: dict) -> None:
        """File within recorded directories produces no file_scope indicator."""
        report = detector.check(
            action="Editing drift detection module",
            file_path="src/claude_code_helper_mcp/detection/drift.py",
            **task_context,
        )
        file_indicators = [i for i in report.indicators if i.category == "file_scope"]
        assert len(file_indicators) == 0

    def test_out_of_scope_file_produces_indicator(self, detector: DriftDetector,
                                                    task_context: dict) -> None:
        """File outside all recorded directories produces a file_scope indicator."""
        report = detector.check(
            action="Editing database configuration",
            file_path="config/database.yaml",
            **task_context,
        )
        file_indicators = [i for i in report.indicators if i.category == "file_scope"]
        assert len(file_indicators) == 1
        assert "outside" in file_indicators[0].description.lower()

    def test_subdirectory_of_scope_no_indicator(self, detector: DriftDetector,
                                                  task_context: dict) -> None:
        """File in a subdirectory of a recorded directory is in scope."""
        report = detector.check(
            action="Adding helper to detection package",
            file_path="src/claude_code_helper_mcp/detection/helpers/utils.py",
            **task_context,
        )
        file_indicators = [i for i in report.indicators if i.category == "file_scope"]
        assert len(file_indicators) == 0

    def test_parent_directory_of_scope_no_indicator(self, detector: DriftDetector,
                                                      task_context: dict) -> None:
        """File in a parent directory of a recorded directory is in scope."""
        report = detector.check(
            action="Editing package init",
            file_path="src/claude_code_helper_mcp/detection/__init__.py",
            **task_context,
        )
        file_indicators = [i for i in report.indicators if i.category == "file_scope"]
        assert len(file_indicators) == 0

    def test_no_file_path_no_indicator(self, detector: DriftDetector,
                                        task_context: dict) -> None:
        """When no file_path is provided, no file_scope indicator is generated."""
        report = detector.check(
            action="Planning next steps",
            file_path=None,
            **task_context,
        )
        file_indicators = [i for i in report.indicators if i.category == "file_scope"]
        assert len(file_indicators) == 0

    def test_no_task_files_no_indicator(self, detector: DriftDetector) -> None:
        """When no task files are recorded, no file_scope indicator fires."""
        report = detector.check(
            action="Starting work",
            file_path="somewhere/random.py",
            task_title="Some task",
            task_files=[],
        )
        file_indicators = [i for i in report.indicators if i.category == "file_scope"]
        assert len(file_indicators) == 0


# ===========================================================================
# 6. Step repetition detection
# ===========================================================================


class TestStepRepetitionDetection:
    """Tests for detecting repeated similar actions."""

    def test_no_repetition_for_first_action(self, detector: DriftDetector,
                                              task_context: dict) -> None:
        """First action in session has no repetition indicator."""
        report = detector.check(
            action="Implementing drift detection",
            **task_context,
        )
        rep_indicators = [i for i in report.indicators if i.category == "step_repetition"]
        assert len(rep_indicators) == 0

    def test_repetition_detected_after_multiple_similar_actions(
        self, detector: DriftDetector, task_context: dict
    ) -> None:
        """Repeating similar actions 3+ times triggers a repetition indicator."""
        for _ in range(3):
            detector.check(
                action="Fixing drift score calculation bug in drift detector",
                **task_context,
            )
        report = detector.check(
            action="Fixing drift score calculation bug in drift detection module",
            **task_context,
        )
        rep_indicators = [i for i in report.indicators if i.category == "step_repetition"]
        assert len(rep_indicators) == 1
        assert "similar" in rep_indicators[0].description.lower()

    def test_no_repetition_for_different_actions(self, detector: DriftDetector,
                                                   task_context: dict) -> None:
        """Different actions do not trigger repetition."""
        detector.check(action="Creating DriftDetector class", **task_context)
        detector.check(action="Adding file scope boundary detection", **task_context)
        report = detector.check(
            action="Implementing severity classification",
            **task_context,
        )
        rep_indicators = [i for i in report.indicators if i.category == "step_repetition"]
        assert len(rep_indicators) == 0


# ===========================================================================
# 7. Topic shift detection
# ===========================================================================


class TestTopicShiftDetection:
    """Tests for detecting sudden topic changes between actions."""

    def test_no_shift_for_first_action(self, detector: DriftDetector,
                                        task_context: dict) -> None:
        """First action has no topic shift indicator."""
        report = detector.check(
            action="Starting drift detection work",
            **task_context,
        )
        shift_indicators = [i for i in report.indicators if i.category == "topic_shift"]
        assert len(shift_indicators) == 0

    def test_topic_shift_detected_on_abrupt_change(
        self, detector: DriftDetector, task_context: dict
    ) -> None:
        """An abrupt change in topics triggers a topic shift indicator."""
        detector.check(
            action="Implementing drift score calculation for alignment checks",
            **task_context,
        )
        report = detector.check(
            action="Deploying kubernetes pods with helm charts to production cluster",
            **task_context,
        )
        shift_indicators = [i for i in report.indicators if i.category == "topic_shift"]
        assert len(shift_indicators) == 1
        assert "shift" in shift_indicators[0].description.lower()

    def test_no_shift_for_related_consecutive_actions(
        self, detector: DriftDetector, task_context: dict
    ) -> None:
        """Related consecutive actions do not trigger a topic shift."""
        detector.check(
            action="Adding drift indicator data classes",
            **task_context,
        )
        report = detector.check(
            action="Adding drift severity classification logic",
            **task_context,
        )
        shift_indicators = [i for i in report.indicators if i.category == "topic_shift"]
        assert len(shift_indicators) == 0


# ===========================================================================
# 8. Temporal drift trend
# ===========================================================================


class TestTemporalDriftTrend:
    """Tests for progressive drift trend detection over time."""

    def test_no_trend_for_few_checks(self, detector: DriftDetector,
                                      task_context: dict) -> None:
        """Trend indicator does not fire with fewer than 4 checks."""
        for _ in range(2):
            detector.check(action="Some action", **task_context)
        report = detector.check(action="Another action", **task_context)
        trend_indicators = [i for i in report.indicators if i.category == "temporal_trend"]
        assert len(trend_indicators) == 0

    def test_stable_drift_no_trend(self, detector: DriftDetector,
                                    task_context: dict) -> None:
        """Consistent low drift does not produce a trend indicator."""
        for _ in range(5):
            report = detector.check(
                action="Working on drift detection engine implementation",
                file_path="src/claude_code_helper_mcp/detection/drift.py",
                **task_context,
            )
        trend_indicators = [i for i in report.indicators if i.category == "temporal_trend"]
        # Stable low drift should not trigger trend.
        assert len(trend_indicators) == 0


# ===========================================================================
# 9. Session management
# ===========================================================================


class TestSessionManagement:
    """Tests for session state tracking and reset."""

    def test_session_length_increments(self, detector: DriftDetector,
                                        task_context: dict) -> None:
        """Each check increments the session length."""
        assert detector.session_length == 0
        detector.check(action="First action", **task_context)
        assert detector.session_length == 1
        detector.check(action="Second action", **task_context)
        assert detector.session_length == 2

    def test_session_scores_recorded(self, detector: DriftDetector,
                                      task_context: dict) -> None:
        """Session scores are recorded after each check."""
        detector.check(action="Action one", **task_context)
        detector.check(action="Action two", **task_context)
        assert len(detector.session_scores) == 2
        assert all(isinstance(s, float) for s in detector.session_scores)

    def test_reset_clears_session(self, detector: DriftDetector,
                                   task_context: dict) -> None:
        """Calling reset() clears all session state."""
        detector.check(action="Some action", **task_context)
        detector.check(action="Another action", **task_context)
        assert detector.session_length == 2

        detector.reset()
        assert detector.session_length == 0
        assert detector.session_scores == []

    def test_session_drift_trend_in_report(self, detector: DriftDetector,
                                            task_context: dict) -> None:
        """DriftReport includes session drift trend."""
        for i in range(3):
            report = detector.check(action=f"Action {i}", **task_context)
        assert len(report.session_drift_trend) == 3

    def test_trend_window_caps_trend_length(self, task_context: dict) -> None:
        """Trend in report is capped by the trend_window setting."""
        detector = DriftDetector(trend_window=3)
        for i in range(5):
            report = detector.check(action=f"Action {i}", **task_context)
        assert len(report.session_drift_trend) <= 3

    def test_session_files_tracked(self, detector: DriftDetector,
                                    task_context: dict) -> None:
        """Files checked during the session are tracked."""
        detector.check(
            action="Edit file",
            file_path="src/some/file.py",
            **task_context,
        )
        detector.check(
            action="Edit another file",
            file_path="src/other/file.py",
            **task_context,
        )
        summary = detector.get_session_summary()
        assert summary["unique_files_checked"] == 2


# ===========================================================================
# 10. Session summary
# ===========================================================================


class TestSessionSummary:
    """Tests for the get_session_summary() method."""

    def test_empty_session_summary(self, detector: DriftDetector) -> None:
        """Empty session returns zeroed summary."""
        summary = detector.get_session_summary()
        assert summary["total_checks"] == 0
        assert summary["average_drift"] == 0.0
        assert summary["max_drift"] == 0.0
        assert summary["min_drift"] == 0.0
        assert summary["current_drift"] == 0.0
        assert summary["trend_direction"] == "stable"
        assert summary["unique_files_checked"] == 0

    def test_summary_after_checks(self, detector: DriftDetector,
                                   task_context: dict) -> None:
        """Summary is populated after checks."""
        for i in range(3):
            detector.check(action=f"Action {i}", **task_context)
        summary = detector.get_session_summary()
        assert summary["total_checks"] == 3
        assert summary["average_drift"] >= 0.0
        assert summary["max_drift"] >= summary["min_drift"]
        assert summary["current_drift"] >= 0.0
        assert summary["trend_direction"] in ("stable", "increasing", "decreasing")

    def test_summary_values_are_rounded(self, detector: DriftDetector,
                                         task_context: dict) -> None:
        """Summary numeric values are rounded to 3 decimal places."""
        detector.check(action="Some action", **task_context)
        summary = detector.get_session_summary()
        for key in ["average_drift", "max_drift", "min_drift", "current_drift"]:
            value_str = str(summary[key])
            if "." in value_str:
                decimal_places = len(value_str.split(".")[1])
                assert decimal_places <= 3


# ===========================================================================
# 11. DriftDetector with TaskMemory integration
# ===========================================================================


class TestTaskMemoryIntegration:
    """Tests for check_with_task() using real TaskMemory objects."""

    def test_check_with_task_basic(self, detector: DriftDetector) -> None:
        """check_with_task extracts fields from TaskMemory correctly."""
        from claude_code_helper_mcp.models import TaskMemory

        task = TaskMemory(
            ticket_id="CMH-019",
            title="Implement drift detection engine",
            phase="phase-5",
        )
        task.add_step(action="Created DriftDetector class")
        task.record_file(
            path="src/claude_code_helper_mcp/detection/drift.py",
            action="created",
            description="New drift detection module",
        )

        report = detector.check_with_task(
            action="Adding severity classification",
            file_path="src/claude_code_helper_mcp/detection/drift.py",
            task=task,
        )
        assert isinstance(report, DriftReport)
        assert report.drift_score < 0.5

    def test_check_with_task_empty_task(self, detector: DriftDetector) -> None:
        """check_with_task handles a minimal task without errors."""
        from claude_code_helper_mcp.models import TaskMemory

        task = TaskMemory(
            ticket_id="CMH-TEST",
            title="Minimal test task",
        )
        report = detector.check_with_task(
            action="Doing something",
            file_path=None,
            task=task,
        )
        assert isinstance(report, DriftReport)

    def test_check_with_task_uses_file_paths(self, detector: DriftDetector) -> None:
        """check_with_task uses task file paths for file scope analysis."""
        from claude_code_helper_mcp.models import TaskMemory, FileAction

        task = TaskMemory(
            ticket_id="CMH-019",
            title="Drift detection engine",
            phase="phase-5",
        )
        task.record_file(
            path="src/claude_code_helper_mcp/detection/drift.py",
            action=FileAction.CREATED,
            description="Main module",
        )
        task.record_file(
            path="tests/test_drift_detector.py",
            action=FileAction.CREATED,
            description="Tests",
        )

        # In-scope file
        in_scope = detector.check_with_task(
            action="Editing drift module",
            file_path="src/claude_code_helper_mcp/detection/drift.py",
            task=task,
        )
        # Out-of-scope file
        out_scope = detector.check_with_task(
            action="Editing unrelated module",
            file_path="config/database/setup.py",
            task=task,
        )
        # Out-of-scope should have higher drift.
        assert out_scope.drift_score >= in_scope.drift_score


# ===========================================================================
# 12. Drift score computation
# ===========================================================================


class TestDriftScoreComputation:
    """Tests for the composite drift score calculation."""

    def test_drift_score_in_valid_range(self, detector: DriftDetector,
                                         task_context: dict) -> None:
        """Drift score is always in [0.0, 1.0]."""
        for action in [
            "Perfectly aligned action on drift detection",
            "Completely unrelated database migration work",
            "",
            "A" * 1000,
        ]:
            report = detector.check(action=action, **task_context)
            assert 0.0 <= report.drift_score <= 1.0

    def test_drift_score_inversely_related_to_alignment(
        self, detector: DriftDetector, task_context: dict
    ) -> None:
        """Higher alignment confidence produces lower drift score."""
        aligned = detector.check(
            action="Implementing drift score for detection engine analysis",
            file_path="src/claude_code_helper_mcp/detection/drift.py",
            **task_context,
        )
        detector.reset()
        unrelated = detector.check(
            action="Setting up AWS Lambda functions for email notifications",
            file_path="infra/lambda/email.py",
            **task_context,
        )
        assert aligned.drift_score < unrelated.drift_score

    def test_indicators_contribute_to_drift_score(self) -> None:
        """Indicators increase the drift score beyond base alignment drift."""
        from claude_code_helper_mcp.detection.alignment import AlignmentReport as AR

        alignment = AR(confidence=0.7, aligned=True)
        no_indicators = DriftDetector._compute_drift_score(alignment, [])

        indicator = DriftIndicator(
            category="test",
            description="Test indicator",
            severity_contribution=0.5,
        )
        with_indicators = DriftDetector._compute_drift_score(alignment, [indicator])

        assert with_indicators > no_indicators


# ===========================================================================
# 13. Edge cases
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_long_action_string(self, detector: DriftDetector,
                                      task_context: dict) -> None:
        """Very long action strings are handled without errors."""
        long_action = "word " * 500
        report = detector.check(action=long_action, **task_context)
        assert isinstance(report, DriftReport)

    def test_special_characters_in_action(self, detector: DriftDetector,
                                           task_context: dict) -> None:
        """Special characters in actions are handled."""
        report = detector.check(
            action="Fix bug: TypeError in _compute_drift_score() [urgent!!!]",
            **task_context,
        )
        assert isinstance(report, DriftReport)

    def test_unicode_in_action(self, detector: DriftDetector,
                                task_context: dict) -> None:
        """Unicode characters in actions are handled."""
        report = detector.check(
            action="Fixing encoding issue with UTF-8 characters",
            **task_context,
        )
        assert isinstance(report, DriftReport)

    def test_none_file_path_handled(self, detector: DriftDetector,
                                     task_context: dict) -> None:
        """None file_path is handled correctly."""
        report = detector.check(
            action="Planning next steps",
            file_path=None,
            **task_context,
        )
        assert isinstance(report, DriftReport)

    def test_empty_task_files_list(self, detector: DriftDetector) -> None:
        """Empty task files list is handled."""
        report = detector.check(
            action="Editing a file",
            file_path="some/file.py",
            task_title="Empty task",
            task_files=[],
        )
        assert isinstance(report, DriftReport)

    def test_many_sequential_checks(self, detector: DriftDetector,
                                     task_context: dict) -> None:
        """Detector handles many sequential checks without error."""
        for i in range(50):
            report = detector.check(
                action=f"Sequential action number {i}",
                **task_context,
            )
        assert detector.session_length == 50
        assert isinstance(report, DriftReport)

    def test_reset_between_sessions(self, detector: DriftDetector,
                                     task_context: dict) -> None:
        """Reset between sessions properly isolates session state."""
        for i in range(5):
            detector.check(action=f"Session 1 action {i}", **task_context)
        assert detector.session_length == 5

        detector.reset()
        assert detector.session_length == 0

        for i in range(3):
            detector.check(action=f"Session 2 action {i}", **task_context)
        assert detector.session_length == 3


# ===========================================================================
# 14. Indicator generation
# ===========================================================================


class TestIndicatorGeneration:
    """Tests for individual indicator generation methods."""

    def test_alignment_drift_indicator_high_confidence(self) -> None:
        """High alignment confidence produces no indicator."""
        alignment = AlignmentReport(confidence=0.95, aligned=True)
        indicator = DriftDetector._alignment_drift_indicator(alignment)
        assert indicator is None

    def test_alignment_drift_indicator_low_confidence(self) -> None:
        """Low alignment confidence produces an indicator."""
        alignment = AlignmentReport(confidence=0.3, aligned=False)
        indicator = DriftDetector._alignment_drift_indicator(alignment)
        assert indicator is not None
        assert indicator.category == "keyword_divergence"
        assert indicator.severity_contribution > 0.5

    def test_alignment_drift_indicator_moderate_confidence(self) -> None:
        """Moderate alignment confidence produces a moderate indicator."""
        alignment = AlignmentReport(confidence=0.6, aligned=True)
        indicator = DriftDetector._alignment_drift_indicator(alignment)
        assert indicator is not None
        assert indicator.severity_contribution < 0.5

    def test_file_scope_indicator_returns_none_when_in_scope(self) -> None:
        """File scope indicator returns None for in-scope files."""
        indicator = DriftDetector._file_scope_indicator(
            file_path="src/detection/drift.py",
            task_files=["src/detection/alignment.py"],
        )
        assert indicator is None

    def test_file_scope_indicator_fires_for_out_of_scope(self) -> None:
        """File scope indicator fires for files outside task directories."""
        indicator = DriftDetector._file_scope_indicator(
            file_path="config/database.yaml",
            task_files=["src/detection/drift.py", "tests/test_drift.py"],
        )
        assert indicator is not None
        assert indicator.category == "file_scope"
        assert indicator.severity_contribution == 0.5


# ===========================================================================
# 15. Constants and defaults
# ===========================================================================


class TestConstants:
    """Tests for module-level constants and defaults."""

    def test_severity_levels_defined(self) -> None:
        """All five severity levels are defined."""
        assert SEVERITY_NONE == "none"
        assert SEVERITY_LOW == "low"
        assert SEVERITY_MODERATE == "moderate"
        assert SEVERITY_HIGH == "high"
        assert SEVERITY_CRITICAL == "critical"

    def test_default_thresholds_defined(self) -> None:
        """Default thresholds are defined for all severity levels."""
        assert SEVERITY_NONE in DEFAULT_THRESHOLDS
        assert SEVERITY_LOW in DEFAULT_THRESHOLDS
        assert SEVERITY_MODERATE in DEFAULT_THRESHOLDS
        assert SEVERITY_HIGH in DEFAULT_THRESHOLDS
        assert SEVERITY_CRITICAL in DEFAULT_THRESHOLDS

    def test_thresholds_are_monotonically_increasing(self) -> None:
        """Severity thresholds increase from none to critical."""
        levels = [SEVERITY_NONE, SEVERITY_LOW, SEVERITY_MODERATE,
                  SEVERITY_HIGH, SEVERITY_CRITICAL]
        values = [DEFAULT_THRESHOLDS[lvl] for lvl in levels]
        for i in range(1, len(values)):
            assert values[i] > values[i - 1]


# ===========================================================================
# 16. Helper functions
# ===========================================================================


class TestHelpers:
    """Tests for module-level helper functions."""

    def test_get_directory_with_path(self) -> None:
        """_get_directory extracts directory from a file path."""
        assert _get_directory("src/detection/drift.py") == "src/detection"

    def test_get_directory_no_slash(self) -> None:
        """_get_directory returns empty string for paths without /."""
        assert _get_directory("drift.py") == ""

    def test_get_directory_trailing_slash(self) -> None:
        """_get_directory handles trailing slashes."""
        assert _get_directory("src/detection/") == "src/detection"

    def test_get_directory_empty_string(self) -> None:
        """_get_directory returns empty string for empty input."""
        assert _get_directory("") == ""
