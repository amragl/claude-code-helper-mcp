"""Graduated intervention system for aggregate drift detection and response.

The InterventionManager aggregates reports from all four detection engines
(DriftDetector, ErrorLoopDetector, ConfusionDetector, ScopeCreepDetector)
and applies a graduated intervention strategy:

- Level 1 (Warning): First detection of any issue type triggers an actionable
  warning message without disrupting the workflow.
- Level 2 (Escalation): Second or later detection of the same or related issues
  suggests auto-clear recovery with comprehensive context restoration.

This module provides:
- Debouncing and alert fatigue prevention via configurable thresholds
- Configurable intervention thresholds and response behaviors
- AggregatedDetectionReport combining all detector outputs
- InterventionResponse with recommendation level and action summary
- Session-level tracking of detection history for graduated escalation
- Integration with TaskMemory for automatic analysis across full task lifecycle

Design decisions:
- All analysis is local and deterministic (no external API calls).
- InterventionManager maintains session state to track detection history.
  Call reset() to start a new analysis session.
- Debouncing uses time windows (default 5 minutes) to avoid duplicate alerts
  on the same issue.
- Graduation logic: first unique issue type → warning; repeat or combination
  of issues → escalation to auto-clear recommendation.
- Configurable thresholds allow different projects to tune sensitivity.

Depends on:
- CMH-019: DriftDetector (plan drift detection)
- CMH-020: ErrorLoopDetector (error loop detection)
- CMH-021: ConfusionDetector (confusion pattern detection)
- CMH-022: ScopeCreepDetector (scope creep detection)
- CMH-009: TaskMemory (task context and step records)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from claude_code_helper_mcp.detection.confusion import ConfusionReport
from claude_code_helper_mcp.detection.drift import DriftReport
from claude_code_helper_mcp.detection.error_loop import ErrorLoopReport
from claude_code_helper_mcp.detection.scope_creep import ScopeCreepReport


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Intervention response levels.
INTERVENTION_LEVEL_NONE = "none"
INTERVENTION_LEVEL_WARNING = "warning"
INTERVENTION_LEVEL_ESCALATION = "escalation"

# Debounce window for alert fatigue prevention (in seconds).
DEFAULT_DEBOUNCE_WINDOW = 300  # 5 minutes

# Thresholds for graduating from warning to escalation.
# - escalate_on_repeat: escalate if same issue type appears again
# - escalate_on_combined: escalate if N different detectors report issues
DEFAULT_ESCALATE_ON_REPEAT = True
DEFAULT_ESCALATE_ON_COMBINED = 2

# Recommended actions per level.
_RECOMMENDED_ACTIONS = {
    INTERVENTION_LEVEL_NONE: (
        "All detectors report clean state. Continue current work."
    ),
    INTERVENTION_LEVEL_WARNING: (
        "One or more detectors have flagged a potential issue. "
        "Review the evidence and consider adjusting your approach. "
        "Use 'memory show' to see task context and recent decisions."
    ),
    INTERVENTION_LEVEL_ESCALATION: (
        "Multiple detectors are reporting related issues, or the same issue "
        "has occurred repeatedly. An automatic /clear and recovery may help. "
        "This will pause current work, save full context, and allow you to "
        "resume with a fresh perspective and recovery prompts."
    ),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DetectionSummary:
    """Summary of a single detector's findings.

    Attributes:
        detector_type: The detector name ('drift', 'error_loop',
            'confusion', 'scope_creep').
        report_available: Whether a report was provided.
        issue_detected: Whether this detector found any issues.
        severity: The severity level from the detector (or 'none').
        issue_count: Number of distinct issues found.
        last_event_time: When this detector last reported an issue.
    """

    detector_type: str
    report_available: bool = False
    issue_detected: bool = False
    severity: str = "none"
    issue_count: int = 0
    last_event_time: Optional[datetime] = None


@dataclass
class AggregatedDetectionReport:
    """Combined output from all detection engines.

    Attributes:
        drift_report: The DriftReport (if provided).
        error_loop_report: The ErrorLoopReport (if provided).
        confusion_report: The ConfusionReport (if provided).
        scope_creep_report: The ScopeCreepReport (if provided).
        detector_summaries: List of DetectionSummary objects from each detector.
        any_issues_detected: Boolean. True if any detector found issues.
        total_issues: Total count of issues across all detectors.
        generated_at: UTC timestamp when this report was generated.
    """

    drift_report: Optional[DriftReport] = None
    error_loop_report: Optional[ErrorLoopReport] = None
    confusion_report: Optional[ConfusionReport] = None
    scope_creep_report: Optional[ScopeCreepReport] = None
    detector_summaries: list[DetectionSummary] = field(default_factory=list)
    any_issues_detected: bool = False
    total_issues: int = 0
    generated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary."""
        return {
            "drift_report": (
                self.drift_report.to_dict() if self.drift_report else None
            ),
            "error_loop_report": (
                self.error_loop_report.to_dict()
                if self.error_loop_report
                else None
            ),
            "confusion_report": (
                self.confusion_report.to_dict() if self.confusion_report else None
            ),
            "scope_creep_report": (
                self.scope_creep_report.to_dict()
                if self.scope_creep_report
                else None
            ),
            "detector_summaries": [
                {
                    "detector_type": s.detector_type,
                    "report_available": s.report_available,
                    "issue_detected": s.issue_detected,
                    "severity": s.severity,
                    "issue_count": s.issue_count,
                    "last_event_time": (
                        s.last_event_time.isoformat()
                        if s.last_event_time
                        else None
                    ),
                }
                for s in self.detector_summaries
            ],
            "any_issues_detected": self.any_issues_detected,
            "total_issues": self.total_issues,
            "generated_at": self.generated_at.isoformat(),
        }


@dataclass
class InterventionResponse:
    """Recommended intervention action based on aggregated detection results.

    Attributes:
        level: One of 'none', 'warning', 'escalation'.
        message: Human-readable summary of the recommended action.
        evidence_summary: Brief summary of the evidence triggering this response.
        detector_details: Detailed findings from each detector (for the UI/CLI
            to display).
        should_prompt_user: Boolean. If True, recommend showing the user this
            message (vs. silent logging).
        should_suggest_clear: Boolean. If True, include auto-clear recovery as
            a suggested remediation step.
        generated_at: UTC timestamp when this response was generated.
    """

    level: str = INTERVENTION_LEVEL_NONE
    message: str = ""
    evidence_summary: str = ""
    detector_details: dict = field(default_factory=dict)
    should_prompt_user: bool = False
    should_suggest_clear: bool = False
    generated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary."""
        return {
            "level": self.level,
            "message": self.message,
            "evidence_summary": self.evidence_summary,
            "detector_details": self.detector_details,
            "should_prompt_user": self.should_prompt_user,
            "should_suggest_clear": self.should_suggest_clear,
            "generated_at": self.generated_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# InterventionManager
# ---------------------------------------------------------------------------


class InterventionManager:
    """Aggregates all detector reports and applies graduated intervention logic.

    The InterventionManager maintains session state to track:
    - What issues have been detected in this session
    - When each issue type was last detected (for debouncing)
    - Whether a user has already seen a warning for this issue type

    This allows the system to apply graduated responses: first unique issue
    → warning, repeat or combination → escalation to auto-clear recovery.

    Parameters
    ----------
    debounce_window_seconds:
        Time window for duplicate alert suppression. Default 300 (5 minutes).
        Identical issues detected within this window are not escalated again.
    escalate_on_repeat:
        If True, escalate to auto-clear suggestion on second detection of
        same issue type. Default True.
    escalate_on_combined:
        Number of different detector types that must report issues to
        escalate. Default 2. Set to 0 to disable combined escalation.
    """

    def __init__(
        self,
        debounce_window_seconds: int = DEFAULT_DEBOUNCE_WINDOW,
        escalate_on_repeat: bool = DEFAULT_ESCALATE_ON_REPEAT,
        escalate_on_combined: int = DEFAULT_ESCALATE_ON_COMBINED,
    ) -> None:
        self._debounce_window = timedelta(seconds=debounce_window_seconds)
        self._escalate_on_repeat = escalate_on_repeat
        self._escalate_on_combined = max(0, escalate_on_combined)

        # Session state: track detection history
        self._detection_history: dict[str, list[datetime]] = {}
        self._last_aggregated_report: Optional[AggregatedDetectionReport] = None

    def reset(self) -> None:
        """Reset session state. Start fresh detection tracking."""
        self._detection_history.clear()
        self._last_aggregated_report = None

    def aggregate_and_respond(
        self,
        drift_report: Optional[DriftReport] = None,
        error_loop_report: Optional[ErrorLoopReport] = None,
        confusion_report: Optional[ConfusionReport] = None,
        scope_creep_report: Optional[ScopeCreepReport] = None,
    ) -> tuple[AggregatedDetectionReport, InterventionResponse]:
        """Aggregate all detector reports and generate an intervention response.

        Combines all available detector reports, checks debounce/repeat
        conditions, and applies graduated intervention logic to determine
        whether to warning or escalate.

        Parameters
        ----------
        drift_report:
            DriftReport from DriftDetector, or None if not run.
        error_loop_report:
            ErrorLoopReport from ErrorLoopDetector, or None if not run.
        confusion_report:
            ConfusionReport from ConfusionDetector, or None if not run.
        scope_creep_report:
            ScopeCreepReport from ScopeCreepDetector, or None if not run.

        Returns
        -------
        Tuple[AggregatedDetectionReport, InterventionResponse]
            - AggregatedDetectionReport: Combined findings from all detectors
            - InterventionResponse: Recommended action (none/warning/escalation)
        """
        now = datetime.now(timezone.utc)

        # Build aggregated report
        aggregated = AggregatedDetectionReport(
            drift_report=drift_report,
            error_loop_report=error_loop_report,
            confusion_report=confusion_report,
            scope_creep_report=scope_creep_report,
            generated_at=now,
        )

        # Analyze each detector's findings
        detectors_with_issues = []
        total_issues = 0

        if drift_report:
            summary = self._analyze_drift(drift_report, now)
            aggregated.detector_summaries.append(summary)
            if summary.issue_detected:
                detectors_with_issues.append("drift")
                total_issues += summary.issue_count

        if error_loop_report:
            summary = self._analyze_error_loop(error_loop_report, now)
            aggregated.detector_summaries.append(summary)
            if summary.issue_detected:
                detectors_with_issues.append("error_loop")
                total_issues += summary.issue_count

        if confusion_report:
            summary = self._analyze_confusion(confusion_report, now)
            aggregated.detector_summaries.append(summary)
            if summary.issue_detected:
                detectors_with_issues.append("confusion")
                total_issues += summary.issue_count

        if scope_creep_report:
            summary = self._analyze_scope_creep(scope_creep_report, now)
            aggregated.detector_summaries.append(summary)
            if summary.issue_detected:
                detectors_with_issues.append("scope_creep")
                total_issues += summary.issue_count

        aggregated.any_issues_detected = len(detectors_with_issues) > 0
        aggregated.total_issues = total_issues

        # Generate intervention response
        response = self._generate_response(
            aggregated, detectors_with_issues, now
        )

        # Store for session tracking
        self._last_aggregated_report = aggregated

        return aggregated, response

    def _analyze_drift(
        self, report: DriftReport, now: datetime
    ) -> DetectionSummary:
        """Analyze DriftReport and create a DetectionSummary."""
        issue_detected = report.severity != "none"
        if issue_detected:
            self._record_detection("drift", now)

        return DetectionSummary(
            detector_type="drift",
            report_available=True,
            issue_detected=issue_detected,
            severity=report.severity,
            issue_count=len(report.indicators) if issue_detected else 0,
            last_event_time=now if issue_detected else None,
        )

    def _analyze_error_loop(
        self, report: ErrorLoopReport, now: datetime
    ) -> DetectionSummary:
        """Analyze ErrorLoopReport and create a DetectionSummary."""
        issue_detected = report.severity != "none"
        if issue_detected:
            self._record_detection("error_loop", now)

        return DetectionSummary(
            detector_type="error_loop",
            report_available=True,
            issue_detected=issue_detected,
            severity=report.severity,
            issue_count=(
                1 if issue_detected else 0
            ),  # ErrorLoopReport reports one loop
            last_event_time=now if issue_detected else None,
        )

    def _analyze_confusion(
        self, report: ConfusionReport, now: datetime
    ) -> DetectionSummary:
        """Analyze ConfusionReport and create a DetectionSummary."""
        issue_detected = report.confusion_detected
        if issue_detected:
            self._record_detection("confusion", now)

        return DetectionSummary(
            detector_type="confusion",
            report_available=True,
            issue_detected=issue_detected,
            severity=report.severity,
            issue_count=len(report.patterns) if issue_detected else 0,
            last_event_time=now if issue_detected else None,
        )

    def _analyze_scope_creep(
        self, report: ScopeCreepReport, now: datetime
    ) -> DetectionSummary:
        """Analyze ScopeCreepReport and create a DetectionSummary."""
        issue_detected = report.creep_detected
        if issue_detected:
            self._record_detection("scope_creep", now)

        return DetectionSummary(
            detector_type="scope_creep",
            report_available=True,
            issue_detected=issue_detected,
            severity=report.severity,
            issue_count=len(report.signals) if issue_detected else 0,
            last_event_time=now if issue_detected else None,
        )

    def _record_detection(self, detector_type: str, now: datetime) -> None:
        """Record a detection event for debouncing and escalation logic."""
        if detector_type not in self._detection_history:
            self._detection_history[detector_type] = []
        self._detection_history[detector_type].append(now)

    def _should_debounce(
        self, detector_type: str, now: datetime
    ) -> bool:
        """Check if a detection should be debounced (duplicate alert suppression)."""
        if detector_type not in self._detection_history:
            return False
        if not self._detection_history[detector_type]:
            return False

        # Get the most recent detection time
        last_detection = self._detection_history[detector_type][-1]
        time_since = now - last_detection

        return time_since < self._debounce_window

    def _should_escalate(
        self, detectors_with_issues: list[str], now: datetime
    ) -> bool:
        """Determine if intervention should escalate to auto-clear suggestion."""
        # Escalate if multiple detector types are reporting issues
        if (
            self._escalate_on_combined > 0
            and len(detectors_with_issues) >= self._escalate_on_combined
        ):
            return True

        # Escalate if same detector reports an issue after debounce window
        if self._escalate_on_repeat:
            for detector in detectors_with_issues:
                if (
                    detector in self._detection_history
                    and len(self._detection_history[detector]) > 1
                ):
                    # This detector has reported before
                    if not self._should_debounce(detector, now):
                        # Last report was outside debounce window
                        return True

        return False

    def _generate_response(
        self,
        aggregated: AggregatedDetectionReport,
        detectors_with_issues: list[str],
        now: datetime,
    ) -> InterventionResponse:
        """Generate the InterventionResponse based on aggregated findings."""
        response = InterventionResponse(generated_at=now)

        if not aggregated.any_issues_detected:
            response.level = INTERVENTION_LEVEL_NONE
            response.message = _RECOMMENDED_ACTIONS[INTERVENTION_LEVEL_NONE]
            return response

        # Determine intervention level
        should_escalate = self._should_escalate(detectors_with_issues, now)

        if should_escalate:
            response.level = INTERVENTION_LEVEL_ESCALATION
            response.should_prompt_user = True
            response.should_suggest_clear = True
            response.message = _RECOMMENDED_ACTIONS[INTERVENTION_LEVEL_ESCALATION]
        else:
            response.level = INTERVENTION_LEVEL_WARNING
            response.should_prompt_user = True
            response.should_suggest_clear = False
            response.message = _RECOMMENDED_ACTIONS[INTERVENTION_LEVEL_WARNING]

        # Build evidence summary
        evidence_parts = []
        if aggregated.drift_report and aggregated.drift_report.severity != "none":
            evidence_parts.append(
                f"Drift detected ({aggregated.drift_report.severity}): "
                f"{len(aggregated.drift_report.indicators)} indicators"
            )
        if (
            aggregated.error_loop_report
            and aggregated.error_loop_report.severity != "none"
        ):
            evidence_parts.append(
                f"Error loop ({aggregated.error_loop_report.severity}): "
                f"repeated failures detected"
            )
        if (
            aggregated.confusion_report
            and aggregated.confusion_report.confusion_detected
        ):
            evidence_parts.append(
                f"Confusion ({aggregated.confusion_report.severity}): "
                f"{len(aggregated.confusion_report.patterns)} patterns"
            )
        if aggregated.scope_creep_report and aggregated.scope_creep_report.creep_detected:
            evidence_parts.append(
                f"Scope creep ({aggregated.scope_creep_report.severity}): "
                f"{len(aggregated.scope_creep_report.signals)} signals"
            )

        response.evidence_summary = "; ".join(evidence_parts)

        # Build detector details for UI/CLI display
        for summary in aggregated.detector_summaries:
            if summary.report_available:
                response.detector_details[summary.detector_type] = {
                    "issue_detected": summary.issue_detected,
                    "severity": summary.severity,
                    "issue_count": summary.issue_count,
                }

        return response
