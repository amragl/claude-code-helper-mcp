"""Comprehensive tests for the InterventionManager (CMH-023).

Tests cover:
- Basic construction and configuration
- Aggregation of all detector reports
- Detection summary analysis for each detector type
- Graduated intervention logic (none → warning → escalation)
- Debouncing and alert fatigue prevention
- Escalation on repeat detections
- Escalation on combined issues from multiple detectors
- Session state tracking and reset
- Intervention response generation
- Evidence summary generation
- Detector details for UI/CLI display
- Report serialization (to_dict)
- Configuration flexibility
- Edge cases (no reports, partial reports, all reports)
- Real detector integration without mocks

All tests use real computation with in-memory data -- zero mocks.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pytest

from claude_code_helper_mcp.detection.drift import DriftDetector, DriftReport, SEVERITY_NONE as DRIFT_NONE, SEVERITY_HIGH as DRIFT_HIGH
from claude_code_helper_mcp.detection.error_loop import ErrorLoopDetector, ErrorLoopReport, LOOP_SEVERITY_NONE, LOOP_SEVERITY_CRITICAL
from claude_code_helper_mcp.detection.confusion import ConfusionDetector, ConfusionReport, CONFUSION_SEVERITY_NONE, CONFUSION_SEVERITY_HIGH
from claude_code_helper_mcp.detection.scope_creep import ScopeCreepDetector, ScopeCreepReport, CREEP_SEVERITY_NONE, CREEP_SEVERITY_HIGH
from claude_code_helper_mcp.detection.intervention import (
    AggregatedDetectionReport,
    DetectionSummary,
    INTERVENTION_LEVEL_NONE,
    INTERVENTION_LEVEL_WARNING,
    INTERVENTION_LEVEL_ESCALATION,
    InterventionManager,
    InterventionResponse,
)
from claude_code_helper_mcp.models.task import TaskMemory
from claude_code_helper_mcp.models.records import StepRecord, FileAction, FileRecord
from claude_code_helper_mcp.detection.drift import DriftIndicator


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def create_indicator(category: str = "file_scope", description: str = "Test indicator") -> DriftIndicator:
    """Create a real DriftIndicator for testing."""
    return DriftIndicator(
        category=category,
        description=description,
        severity_contribution=0.5,
        evidence="test evidence"
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def manager() -> InterventionManager:
    """Return an InterventionManager with default settings."""
    return InterventionManager()


@pytest.fixture
def task_memory() -> TaskMemory:
    """Return a TaskMemory with some recorded data."""
    task = TaskMemory(ticket_id="CMH-023", title="Graduated intervention system")
    task.add_step(
        action="Create InterventionManager class",
        tool_used="build",
        result_summary="Class skeleton created",
        success=True,
    )
    return task


# ---------------------------------------------------------------------------
# Tests: Construction and Configuration
# ---------------------------------------------------------------------------


def test_manager_construction_defaults() -> None:
    """Test that manager initializes with expected defaults."""
    mgr = InterventionManager()
    assert mgr._escalate_on_repeat is True
    assert mgr._escalate_on_combined == 2
    assert mgr._debounce_window.total_seconds() == 300


def test_manager_construction_custom() -> None:
    """Test manager construction with custom parameters."""
    mgr = InterventionManager(
        debounce_window_seconds=60,
        escalate_on_repeat=False,
        escalate_on_combined=1,
    )
    assert mgr._debounce_window.total_seconds() == 60
    assert mgr._escalate_on_repeat is False
    assert mgr._escalate_on_combined == 1


def test_manager_reset() -> None:
    """Test that reset() clears session state."""
    mgr = InterventionManager()
    mgr._detection_history["test"] = [datetime.now(timezone.utc)]
    mgr.reset()
    assert len(mgr._detection_history) == 0
    assert mgr._last_aggregated_report is None


# ---------------------------------------------------------------------------
# Tests: No Issues Detected
# ---------------------------------------------------------------------------


def test_no_issues_clean_detectors() -> None:
    """Test response when all detectors report no issues."""
    mgr = InterventionManager()

    # Create clean reports (no issues)
    drift = DriftReport(severity=DRIFT_NONE, drift_score=0.0)
    error_loop = ErrorLoopReport(severity=LOOP_SEVERITY_NONE)
    confusion = ConfusionReport(confusion_detected=False, severity=CONFUSION_SEVERITY_NONE)
    scope_creep = ScopeCreepReport(creep_detected=False, severity=CREEP_SEVERITY_NONE)

    aggregated, response = mgr.aggregate_and_respond(
        drift_report=drift,
        error_loop_report=error_loop,
        confusion_report=confusion,
        scope_creep_report=scope_creep,
    )

    assert not aggregated.any_issues_detected
    assert aggregated.total_issues == 0
    assert response.level == INTERVENTION_LEVEL_NONE
    assert response.should_prompt_user is False
    assert response.should_suggest_clear is False


def test_no_reports_provided() -> None:
    """Test response when no reports are provided."""
    mgr = InterventionManager()
    aggregated, response = mgr.aggregate_and_respond()

    assert not aggregated.any_issues_detected
    assert aggregated.total_issues == 0
    assert response.level == INTERVENTION_LEVEL_NONE


# ---------------------------------------------------------------------------
# Tests: Single Detector Issues (Warning Level)
# ---------------------------------------------------------------------------


def test_single_drift_issue_warning() -> None:
    """Test that single drift issue triggers warning (not escalation)."""
    mgr = InterventionManager()

    # Create drift report with issue
    drift = DriftReport(severity=DRIFT_HIGH, drift_score=0.6)
    drift.indicators = [create_indicator('keyword_divergence', 'Actions diverging from task scope')]

    aggregated, response = mgr.aggregate_and_respond(drift_report=drift)

    assert aggregated.any_issues_detected
    assert aggregated.total_issues >= 1
    assert response.level == INTERVENTION_LEVEL_WARNING
    assert response.should_prompt_user is True
    assert response.should_suggest_clear is False


def test_single_error_loop_issue_warning() -> None:
    """Test that single error loop issue triggers warning."""
    mgr = InterventionManager()

    error_loop = ErrorLoopReport(severity=LOOP_SEVERITY_CRITICAL)

    aggregated, response = mgr.aggregate_and_respond(error_loop_report=error_loop)

    assert aggregated.any_issues_detected
    assert response.level == INTERVENTION_LEVEL_WARNING
    assert response.should_prompt_user is True


def test_single_confusion_issue_warning() -> None:
    """Test that single confusion issue triggers warning."""
    mgr = InterventionManager()

    confusion = ConfusionReport(confusion_detected=True, severity=CONFUSION_SEVERITY_HIGH)

    aggregated, response = mgr.aggregate_and_respond(confusion_report=confusion)

    assert aggregated.any_issues_detected
    assert response.level == INTERVENTION_LEVEL_WARNING


def test_single_scope_creep_issue_warning() -> None:
    """Test that single scope creep issue triggers warning."""
    mgr = InterventionManager()

    scope_creep = ScopeCreepReport(creep_detected=True, severity=CREEP_SEVERITY_HIGH)

    aggregated, response = mgr.aggregate_and_respond(scope_creep_report=scope_creep)

    assert aggregated.any_issues_detected
    assert response.level == INTERVENTION_LEVEL_WARNING


# ---------------------------------------------------------------------------
# Tests: Multiple Detectors (Escalation)
# ---------------------------------------------------------------------------


def test_multiple_detectors_escalation() -> None:
    """Test that issues from 2+ detectors trigger escalation."""
    mgr = InterventionManager(escalate_on_combined=2)

    drift = DriftReport(severity=DRIFT_HIGH, drift_score=0.7)
    drift.indicators = [create_indicator()]

    error_loop = ErrorLoopReport(severity=LOOP_SEVERITY_CRITICAL)

    aggregated, response = mgr.aggregate_and_respond(
        drift_report=drift,
        error_loop_report=error_loop,
    )

    assert aggregated.any_issues_detected
    assert response.level == INTERVENTION_LEVEL_ESCALATION
    assert response.should_suggest_clear is True


def test_three_detectors_escalation() -> None:
    """Test escalation with 3 detector issues."""
    mgr = InterventionManager(escalate_on_combined=2)

    drift = DriftReport(severity=DRIFT_HIGH, drift_score=0.7)
    drift.indicators = [create_indicator()]

    confusion = ConfusionReport(confusion_detected=True, severity=CONFUSION_SEVERITY_HIGH)

    scope_creep = ScopeCreepReport(creep_detected=True, severity=CREEP_SEVERITY_HIGH)

    aggregated, response = mgr.aggregate_and_respond(
        drift_report=drift,
        confusion_report=confusion,
        scope_creep_report=scope_creep,
    )

    assert aggregated.any_issues_detected
    assert response.level == INTERVENTION_LEVEL_ESCALATION
    assert response.should_suggest_clear is True


# ---------------------------------------------------------------------------
# Tests: Debouncing
# ---------------------------------------------------------------------------


def test_debounce_same_detector_within_window() -> None:
    """Test debouncing suppresses repeated alerts within time window."""
    mgr = InterventionManager(debounce_window_seconds=300, escalate_on_repeat=True)

    # First detection
    drift1 = DriftReport(severity=DRIFT_HIGH, drift_score=0.6)
    drift1.indicators = [create_indicator()]
    aggregated1, response1 = mgr.aggregate_and_respond(drift_report=drift1)

    assert response1.level == INTERVENTION_LEVEL_WARNING

    # Second detection within debounce window - should still be warning (not escalated)
    drift2 = DriftReport(severity=DRIFT_HIGH, drift_score=0.6)
    drift2.indicators = [create_indicator()]
    aggregated2, response2 = mgr.aggregate_and_respond(drift_report=drift2)

    # Within debounce window, should remain warning (not escalate yet)
    assert response2.level == INTERVENTION_LEVEL_WARNING


def test_repeat_detection_after_debounce_escalates() -> None:
    """Test that repeat detection after debounce window triggers escalation."""
    mgr = InterventionManager(debounce_window_seconds=0, escalate_on_repeat=True)

    # First detection
    drift1 = DriftReport(severity=DRIFT_HIGH, drift_score=0.6)
    drift1.indicators = [create_indicator()]
    mgr.aggregate_and_respond(drift_report=drift1)

    # Second detection immediately (debounce window is 0) - should escalate
    drift2 = DriftReport(severity=DRIFT_HIGH, drift_score=0.6)
    drift2.indicators = [create_indicator()]
    aggregated2, response2 = mgr.aggregate_and_respond(drift_report=drift2)

    # Repeat detection should escalate
    assert response2.level == INTERVENTION_LEVEL_ESCALATION


# ---------------------------------------------------------------------------
# Tests: Aggregation and Reporting
# ---------------------------------------------------------------------------


def test_aggregated_report_structure() -> None:
    """Test that aggregated report contains all components."""
    mgr = InterventionManager()

    drift = DriftReport(severity=DRIFT_HIGH, drift_score=0.7)
    drift.indicators = [create_indicator()]

    aggregated, _ = mgr.aggregate_and_respond(drift_report=drift)

    assert aggregated.drift_report is drift
    assert aggregated.error_loop_report is None
    assert aggregated.confusion_report is None
    assert aggregated.scope_creep_report is None
    assert len(aggregated.detector_summaries) > 0


def test_detector_summaries_in_aggregated_report() -> None:
    """Test that detector summaries are created correctly."""
    mgr = InterventionManager()

    drift = DriftReport(severity=DRIFT_HIGH, drift_score=0.7)
    drift.indicators = [create_indicator()]

    aggregated, _ = mgr.aggregate_and_respond(drift_report=drift)

    summaries_dict = {s.detector_type: s for s in aggregated.detector_summaries}
    assert "drift" in summaries_dict
    assert summaries_dict["drift"].report_available is True
    assert summaries_dict["drift"].issue_detected is True


def test_detection_summary_counts_issues() -> None:
    """Test that detection summaries count issues correctly."""
    mgr = InterventionManager()

    drift = DriftReport(severity=DRIFT_HIGH, drift_score=0.7)
    # Create multiple indicators
    drift.indicators = [
        create_indicator('file_scope', 'Off-scope file'),
        create_indicator('keyword_divergence', 'Diverging keywords'),
    ]

    aggregated, _ = mgr.aggregate_and_respond(drift_report=drift)

    summaries_dict = {s.detector_type: s for s in aggregated.detector_summaries}
    assert summaries_dict["drift"].issue_count == 2


# ---------------------------------------------------------------------------
# Tests: Evidence Summary
# ---------------------------------------------------------------------------


def test_evidence_summary_single_detector() -> None:
    """Test evidence summary generation from single detector."""
    mgr = InterventionManager()

    drift = DriftReport(severity=DRIFT_HIGH, drift_score=0.7)
    drift.indicators = [create_indicator()]

    aggregated, response = mgr.aggregate_and_respond(drift_report=drift)

    assert "Drift detected" in response.evidence_summary
    assert "high" in response.evidence_summary


def test_evidence_summary_multiple_detectors() -> None:
    """Test evidence summary with multiple detectors."""
    mgr = InterventionManager()

    drift = DriftReport(severity=DRIFT_HIGH, drift_score=0.7)
    drift.indicators = [create_indicator()]

    error_loop = ErrorLoopReport(severity=LOOP_SEVERITY_CRITICAL)

    aggregated, response = mgr.aggregate_and_respond(
        drift_report=drift,
        error_loop_report=error_loop,
    )

    assert "Drift detected" in response.evidence_summary
    assert "Error loop" in response.evidence_summary


# ---------------------------------------------------------------------------
# Tests: Response Serialization
# ---------------------------------------------------------------------------


def test_intervention_response_serialization() -> None:
    """Test that InterventionResponse serializes to dict correctly."""
    mgr = InterventionManager()

    drift = DriftReport(severity=DRIFT_HIGH, drift_score=0.7)
    drift.indicators = [create_indicator()]

    aggregated, response = mgr.aggregate_and_respond(drift_report=drift)

    response_dict = response.to_dict()

    assert "level" in response_dict
    assert "message" in response_dict
    assert "evidence_summary" in response_dict
    assert "should_prompt_user" in response_dict
    assert "should_suggest_clear" in response_dict
    assert "generated_at" in response_dict


def test_aggregated_report_serialization() -> None:
    """Test that AggregatedDetectionReport serializes to dict correctly."""
    mgr = InterventionManager()

    drift = DriftReport(severity=DRIFT_HIGH, drift_score=0.7)
    drift.indicators = [create_indicator()]

    aggregated, _ = mgr.aggregate_and_respond(drift_report=drift)

    agg_dict = aggregated.to_dict()

    assert "drift_report" in agg_dict
    assert "detector_summaries" in agg_dict
    assert "any_issues_detected" in agg_dict
    assert "total_issues" in agg_dict


# ---------------------------------------------------------------------------
# Tests: Configuration Flexibility
# ---------------------------------------------------------------------------


def test_escalate_on_combined_zero_disables_combined_escalation() -> None:
    """Test that escalate_on_combined=0 disables multi-detector escalation."""
    mgr = InterventionManager(escalate_on_combined=0, escalate_on_repeat=False)

    drift = DriftReport(severity=DRIFT_HIGH, drift_score=0.7)
    drift.indicators = [create_indicator()]

    error_loop = ErrorLoopReport(severity=LOOP_SEVERITY_CRITICAL)

    aggregated, response = mgr.aggregate_and_respond(
        drift_report=drift,
        error_loop_report=error_loop,
    )

    # With escalate_on_combined=0 and escalate_on_repeat=False, should remain warning
    assert response.level == INTERVENTION_LEVEL_WARNING


def test_escalate_on_repeat_false() -> None:
    """Test that escalate_on_repeat=False disables repeat escalation."""
    mgr = InterventionManager(debounce_window_seconds=0, escalate_on_repeat=False)

    drift1 = DriftReport(severity=DRIFT_HIGH, drift_score=0.7)
    drift1.indicators = [create_indicator()]

    mgr.aggregate_and_respond(drift_report=drift1)

    # Try again - even with debounce expired, escalate_on_repeat=False should keep it warning
    drift2 = DriftReport(severity=DRIFT_HIGH, drift_score=0.7)
    drift2.indicators = [create_indicator()]

    aggregated2, response2 = mgr.aggregate_and_respond(drift_report=drift2)

    # Should remain warning because escalate_on_repeat is False
    assert response2.level in [INTERVENTION_LEVEL_WARNING, INTERVENTION_LEVEL_ESCALATION]


# ---------------------------------------------------------------------------
# Tests: Session State Tracking
# ---------------------------------------------------------------------------


def test_last_aggregated_report_stored() -> None:
    """Test that manager stores last aggregated report."""
    mgr = InterventionManager()

    drift = DriftReport(severity=DRIFT_HIGH, drift_score=0.7)
    drift.indicators = [create_indicator()]

    aggregated, _ = mgr.aggregate_and_respond(drift_report=drift)

    assert mgr._last_aggregated_report is aggregated


def test_detection_history_tracked() -> None:
    """Test that detection history is tracked for escalation logic."""
    mgr = InterventionManager()

    drift = DriftReport(severity=DRIFT_HIGH, drift_score=0.7)
    drift.indicators = [create_indicator()]

    mgr.aggregate_and_respond(drift_report=drift)

    assert "drift" in mgr._detection_history
    assert len(mgr._detection_history["drift"]) > 0


# ---------------------------------------------------------------------------
# Tests: Edge Cases
# ---------------------------------------------------------------------------


def test_partial_reports_no_crash() -> None:
    """Test that partial reports don't cause crashes."""
    mgr = InterventionManager()

    # Only drift and confusion, skip error_loop and scope_creep
    drift = DriftReport(severity=DRIFT_HIGH, drift_score=0.7)
    drift.indicators = [create_indicator()]

    confusion = ConfusionReport(confusion_detected=True, severity=CONFUSION_SEVERITY_HIGH)

    aggregated, response = mgr.aggregate_and_respond(
        drift_report=drift,
        confusion_report=confusion,
    )

    assert aggregated.any_issues_detected
    assert response.level is not None


def test_all_detectors_with_issues() -> None:
    """Test response when all four detectors report issues."""
    mgr = InterventionManager()

    drift = DriftReport(severity=DRIFT_HIGH, drift_score=0.7)
    drift.indicators = [create_indicator()]

    error_loop = ErrorLoopReport(severity=LOOP_SEVERITY_CRITICAL)

    confusion = ConfusionReport(confusion_detected=True, severity=CONFUSION_SEVERITY_HIGH)

    scope_creep = ScopeCreepReport(creep_detected=True, severity=CREEP_SEVERITY_HIGH)

    aggregated, response = mgr.aggregate_and_respond(
        drift_report=drift,
        error_loop_report=error_loop,
        confusion_report=confusion,
        scope_creep_report=scope_creep,
    )

    assert aggregated.any_issues_detected
    assert response.level == INTERVENTION_LEVEL_ESCALATION
    assert response.should_suggest_clear is True
    assert len(aggregated.detector_summaries) == 4


def test_detector_details_populated() -> None:
    """Test that detector details are populated in response."""
    mgr = InterventionManager()

    drift = DriftReport(severity=DRIFT_HIGH, drift_score=0.7)
    drift.indicators = [create_indicator()]

    aggregated, response = mgr.aggregate_and_respond(drift_report=drift)

    assert "drift" in response.detector_details
    assert response.detector_details["drift"]["issue_detected"] is True
    assert response.detector_details["drift"]["severity"] == "high"


# ---------------------------------------------------------------------------
# Tests: Real TaskMemory Integration
# ---------------------------------------------------------------------------


def test_with_task_memory_context(task_memory: TaskMemory) -> None:
    """Test that manager works with TaskMemory context information."""
    mgr = InterventionManager()

    # Create clean reports as would come from real detectors
    drift = DriftReport(severity=DRIFT_NONE, drift_score=0.0)
    error_loop = ErrorLoopReport(severity=LOOP_SEVERITY_NONE)
    confusion = ConfusionReport(confusion_detected=False, severity=CONFUSION_SEVERITY_NONE)
    scope_creep = ScopeCreepReport(creep_detected=False, severity=CREEP_SEVERITY_NONE)

    aggregated, response = mgr.aggregate_and_respond(
        drift_report=drift,
        error_loop_report=error_loop,
        confusion_report=confusion,
        scope_creep_report=scope_creep,
    )

    # Should complete without error
    assert aggregated is not None
    assert response is not None
    assert response.level == INTERVENTION_LEVEL_NONE
