"""Plan drift detection engine for identifying when actions diverge from task scope.

The DriftDetector compares current actions against the active task's defined scope
using file scope analysis, action relevance scoring, step pattern analysis, and
temporal drift tracking.  It produces a DriftReport with severity, indicators,
and recommended actions.

This module builds on the AlignmentChecker (CMH-012) by adding:
- Historical drift tracking across a sequence of actions (not just single checks)
- Severity classification (none, low, moderate, high, critical)
- File scope boundary detection (when edits leave the expected directory tree)
- Step pattern analysis (detecting repetitive actions or sudden topic changes)
- Recommended actions based on drift severity and pattern

Design decisions:
- All analysis is local and deterministic (no external API calls).
- DriftDetector maintains state across check() calls within a session for
  temporal drift tracking.  Call reset() to start a new session.
- Severity thresholds are configurable at construction time.
- The detector re-uses AlignmentChecker's keyword and file analysis internally
  to avoid duplication.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from claude_code_helper_mcp.detection.alignment import (
    AlignmentChecker,
    AlignmentReport,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DriftIndicator:
    """A single signal that drift may be occurring.

    Attributes:
        category: The type of indicator (e.g., 'file_scope', 'keyword_divergence',
            'step_repetition', 'topic_shift').
        description: Human-readable description of the drift signal.
        severity_contribution: Float in [0.0, 1.0] indicating how much this
            indicator contributes to overall drift severity.
        evidence: Optional evidence string (e.g., the off-scope file path).
    """

    category: str
    description: str
    severity_contribution: float
    evidence: str = ""


@dataclass
class DriftReport:
    """Result of a drift detection check.

    Attributes:
        severity: One of 'none', 'low', 'moderate', 'high', 'critical'.
        drift_score: Float in [0.0, 1.0] where 0.0 is no drift and 1.0 is
            maximum drift.
        indicators: List of DriftIndicator objects describing what triggered
            the drift score.
        recommended_action: Suggested response based on drift severity.
        alignment_report: The underlying AlignmentReport from AlignmentChecker.
        session_drift_trend: List of drift scores from the current session
            (most recent last), showing how drift has evolved over time.
        generated_at: UTC timestamp when this report was generated.
    """

    severity: str
    drift_score: float
    indicators: list[DriftIndicator] = field(default_factory=list)
    recommended_action: str = ""
    alignment_report: Optional[AlignmentReport] = None
    session_drift_trend: list[float] = field(default_factory=list)
    generated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> dict:
        """Serialise to a plain dictionary."""
        return {
            "severity": self.severity,
            "drift_score": round(self.drift_score, 3),
            "indicators": [
                {
                    "category": ind.category,
                    "description": ind.description,
                    "severity_contribution": round(ind.severity_contribution, 3),
                    "evidence": ind.evidence,
                }
                for ind in self.indicators
            ],
            "recommended_action": self.recommended_action,
            "alignment_report": (
                self.alignment_report.to_dict()
                if self.alignment_report
                else None
            ),
            "session_drift_trend": [round(s, 3) for s in self.session_drift_trend],
            "generated_at": self.generated_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Constants and severity thresholds
# ---------------------------------------------------------------------------

# Severity level names in ascending order.
SEVERITY_NONE = "none"
SEVERITY_LOW = "low"
SEVERITY_MODERATE = "moderate"
SEVERITY_HIGH = "high"
SEVERITY_CRITICAL = "critical"

# Default severity thresholds (drift_score ranges).
DEFAULT_THRESHOLDS = {
    SEVERITY_NONE: 0.0,       # [0.0, 0.2)
    SEVERITY_LOW: 0.2,        # [0.2, 0.4)
    SEVERITY_MODERATE: 0.4,   # [0.4, 0.6)
    SEVERITY_HIGH: 0.6,       # [0.6, 0.8)
    SEVERITY_CRITICAL: 0.8,   # [0.8, 1.0]
}

# Recommended actions per severity level.
_RECOMMENDED_ACTIONS = {
    SEVERITY_NONE: "No drift detected.  Continue current work.",
    SEVERITY_LOW: (
        "Minor drift detected.  Verify the current action is relevant to the "
        "active task before proceeding."
    ),
    SEVERITY_MODERATE: (
        "Moderate drift detected.  Review task scope and confirm the current "
        "action is intentional.  Consider recording a decision if expanding scope."
    ),
    SEVERITY_HIGH: (
        "Significant drift detected.  The current actions appear to be moving "
        "away from the task scope.  Pause and review the task requirements.  "
        "Consider running check_alignment to verify."
    ),
    SEVERITY_CRITICAL: (
        "Critical drift detected.  The current work appears substantially "
        "outside the task scope.  Strongly recommend pausing, reviewing the "
        "task definition, and either refocusing or recording a deliberate "
        "scope change decision."
    ),
}


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------


class DriftDetector:
    """Detects plan drift by tracking how actions diverge from task scope over time.

    The detector maintains a session history of alignment checks and uses
    temporal analysis to identify progressive drift.  Each call to :meth:`check`
    produces a :class:`DriftReport` with severity classification.

    Parameters
    ----------
    alignment_threshold:
        The AlignmentChecker threshold (passed through to AlignmentChecker).
        Default 0.5.
    severity_thresholds:
        Custom severity thresholds.  Keys are severity level names, values are
        the minimum drift_score for that level.  Defaults to DEFAULT_THRESHOLDS.
    trend_window:
        Number of recent checks to include in the session drift trend.  Default 10.
    """

    def __init__(
        self,
        alignment_threshold: float = 0.5,
        severity_thresholds: Optional[dict[str, float]] = None,
        trend_window: int = 10,
    ) -> None:
        self._checker = AlignmentChecker(threshold=alignment_threshold)
        self._thresholds = dict(severity_thresholds or DEFAULT_THRESHOLDS)
        self._trend_window = max(1, trend_window)

        # Session state: accumulated across check() calls until reset().
        self._session_scores: list[float] = []
        self._session_actions: list[str] = []
        self._session_files: list[str] = []

    @property
    def session_length(self) -> int:
        """Return the number of checks performed in the current session."""
        return len(self._session_scores)

    @property
    def session_scores(self) -> list[float]:
        """Return a copy of all drift scores in the current session."""
        return list(self._session_scores)

    def reset(self) -> None:
        """Clear session state for a new detection session."""
        self._session_scores.clear()
        self._session_actions.clear()
        self._session_files.clear()

    def check(
        self,
        action: str,
        file_path: Optional[str] = None,
        task_title: str = "",
        task_description: str = "",
        task_phase: Optional[str] = None,
        task_files: Optional[list[str]] = None,
        task_steps: Optional[list[str]] = None,
        task_ticket_id: Optional[str] = None,
    ) -> DriftReport:
        """Run a drift detection check for a single action.

        Parameters
        ----------
        action:
            Description of the action being checked.
        file_path:
            Optional file path the action targets.
        task_title:
            The active task's title.
        task_description:
            Optional task description.
        task_phase:
            Optional roadmap phase (e.g., "phase-5").
        task_files:
            List of file paths already recorded in the task scope.
        task_steps:
            List of action strings from steps already recorded.
        task_ticket_id:
            The ticket identifier (e.g., "CMH-019").

        Returns
        -------
        DriftReport
            The drift assessment with severity and indicators.
        """
        task_files = task_files or []
        task_steps = task_steps or []

        # 1. Run the underlying alignment check.
        alignment = self._checker.check(
            action=action,
            file_path=file_path,
            task_title=task_title,
            task_description=task_description,
            task_phase=task_phase,
            task_files=task_files,
            task_steps=task_steps,
            task_ticket_id=task_ticket_id,
        )

        # 2. Collect drift indicators.
        indicators: list[DriftIndicator] = []

        # 2a. Alignment-based drift indicator.
        alignment_drift = self._alignment_drift_indicator(alignment)
        if alignment_drift:
            indicators.append(alignment_drift)

        # 2b. File scope boundary indicator.
        file_indicator = self._file_scope_indicator(file_path, task_files)
        if file_indicator:
            indicators.append(file_indicator)

        # 2c. Step repetition indicator.
        repetition_indicator = self._step_repetition_indicator(action)
        if repetition_indicator:
            indicators.append(repetition_indicator)

        # 2d. Topic shift indicator (sudden change in action keywords).
        topic_indicator = self._topic_shift_indicator(action)
        if topic_indicator:
            indicators.append(topic_indicator)

        # 2e. Temporal drift trend indicator (progressive drift over session).
        trend_indicator = self._temporal_trend_indicator(alignment.confidence)
        if trend_indicator:
            indicators.append(trend_indicator)

        # 3. Compute the composite drift score.
        drift_score = self._compute_drift_score(alignment, indicators)

        # 4. Record session state.
        self._session_scores.append(drift_score)
        self._session_actions.append(action)
        if file_path:
            self._session_files.append(file_path)

        # 5. Classify severity and get recommended action.
        severity = self._classify_severity(drift_score)
        recommended = _RECOMMENDED_ACTIONS.get(severity, "")

        # 6. Build the trend snapshot.
        trend = self._session_scores[-self._trend_window:]

        return DriftReport(
            severity=severity,
            drift_score=drift_score,
            indicators=indicators,
            recommended_action=recommended,
            alignment_report=alignment,
            session_drift_trend=list(trend),
        )

    # ------------------------------------------------------------------
    # Indicator generation
    # ------------------------------------------------------------------

    @staticmethod
    def _alignment_drift_indicator(
        alignment: AlignmentReport,
    ) -> Optional[DriftIndicator]:
        """Generate an indicator from the alignment confidence score.

        Low alignment confidence directly implies the action is drifting
        from the task scope.
        """
        # Invert the confidence to get drift contribution.
        # High confidence = low drift, low confidence = high drift.
        drift_contribution = 1.0 - alignment.confidence

        if drift_contribution <= 0.1:
            # Very well aligned -- no indicator needed.
            return None

        if drift_contribution < 0.3:
            desc = (
                f"Alignment confidence is {alignment.confidence:.0%}, indicating "
                f"minor divergence from task scope."
            )
        elif drift_contribution < 0.6:
            desc = (
                f"Alignment confidence is {alignment.confidence:.0%}, indicating "
                f"moderate divergence from task scope."
            )
        else:
            desc = (
                f"Alignment confidence is {alignment.confidence:.0%}, indicating "
                f"significant divergence from task scope."
            )

        return DriftIndicator(
            category="keyword_divergence",
            description=desc,
            severity_contribution=drift_contribution,
            evidence=f"confidence={alignment.confidence:.3f}",
        )

    @staticmethod
    def _file_scope_indicator(
        file_path: Optional[str],
        task_files: list[str],
    ) -> Optional[DriftIndicator]:
        """Detect when a file is outside the directories touched by the task.

        Returns an indicator if the file's directory does not overlap with
        any of the task's recorded file directories.
        """
        if not file_path or not task_files:
            return None

        file_dir = _get_directory(file_path)
        if not file_dir:
            return None

        task_dirs = {_get_directory(f) for f in task_files if _get_directory(f)}
        if not task_dirs:
            return None

        # Check if the file's directory is within or contains any task directory.
        for td in task_dirs:
            if file_dir.startswith(td) or td.startswith(file_dir):
                return None

        # File is outside all recorded directories.
        return DriftIndicator(
            category="file_scope",
            description=(
                f"File '{file_path}' is outside all directories previously "
                f"touched by this task ({len(task_dirs)} directory/ies in scope)."
            ),
            severity_contribution=0.5,
            evidence=f"file_dir={file_dir}, task_dirs={sorted(task_dirs)[:5]}",
        )

    def _step_repetition_indicator(
        self,
        action: str,
    ) -> Optional[DriftIndicator]:
        """Detect when the same action is repeated multiple times.

        Repetitive actions (3+ similar actions in recent history) may indicate
        an error loop or confusion.
        """
        if len(self._session_actions) < 2:
            return None

        # Compare action keywords against recent session actions.
        action_kw = AlignmentChecker._extract_keywords(action)
        if not action_kw:
            return None

        similar_count = 0
        recent_actions = self._session_actions[-6:]  # look at last 6 actions
        for prev_action in recent_actions:
            prev_kw = AlignmentChecker._extract_keywords(prev_action)
            if not prev_kw:
                continue
            overlap = action_kw & prev_kw
            if len(overlap) >= max(1, len(action_kw) * 0.7):
                similar_count += 1

        if similar_count < 2:
            return None

        contribution = min(0.6, 0.15 * similar_count)
        return DriftIndicator(
            category="step_repetition",
            description=(
                f"Action is very similar to {similar_count} recent actions.  "
                f"This may indicate an error loop or repeated failed attempts."
            ),
            severity_contribution=contribution,
            evidence=f"similar_actions_in_recent_history={similar_count}",
        )

    def _topic_shift_indicator(
        self,
        action: str,
    ) -> Optional[DriftIndicator]:
        """Detect a sudden topic shift from the previous action.

        If the previous action had mostly different keywords from the current
        action, and the shift is large, this may indicate unintentional drift.
        """
        if not self._session_actions:
            return None

        prev_action = self._session_actions[-1]
        current_kw = AlignmentChecker._extract_keywords(action)
        prev_kw = AlignmentChecker._extract_keywords(prev_action)

        if not current_kw or not prev_kw:
            return None

        overlap = current_kw & prev_kw
        all_keywords = current_kw | prev_kw
        if not all_keywords:
            return None

        similarity = len(overlap) / len(all_keywords)

        # A similarity below 0.1 between consecutive actions is a sudden shift.
        if similarity >= 0.1:
            return None

        return DriftIndicator(
            category="topic_shift",
            description=(
                f"Sudden topic shift detected between consecutive actions.  "
                f"Keyword similarity is {similarity:.0%} (expected > 10%).  "
                f"This may indicate a context switch or confusion."
            ),
            severity_contribution=0.35,
            evidence=(
                f"similarity={similarity:.3f}, "
                f"current_keywords={sorted(current_kw)[:5]}, "
                f"previous_keywords={sorted(prev_kw)[:5]}"
            ),
        )

    def _temporal_trend_indicator(
        self,
        current_confidence: float,
    ) -> Optional[DriftIndicator]:
        """Detect a progressive decline in alignment confidence over time.

        If the last N checks show a consistent downward trend in confidence,
        this indicates progressive drift even if no single check is critically low.
        """
        if len(self._session_scores) < 3:
            return None

        # Look at the last few alignment-derived drift scores.
        # We approximate by using session scores (which include alignment info).
        recent = self._session_scores[-4:]

        # Check if scores are monotonically increasing (increasing drift).
        increasing_count = 0
        for i in range(1, len(recent)):
            if recent[i] > recent[i - 1]:
                increasing_count += 1

        # If most recent scores are increasing, we have a trend.
        if increasing_count < len(recent) - 1:
            return None

        avg_drift = sum(recent) / len(recent)
        if avg_drift < 0.2:
            # Drift is consistently low even if trending up -- no concern.
            return None

        return DriftIndicator(
            category="temporal_trend",
            description=(
                f"Progressive drift trend detected over {len(recent)} recent "
                f"checks.  Average drift score is {avg_drift:.0%}.  Drift is "
                f"consistently increasing."
            ),
            severity_contribution=min(0.4, avg_drift * 0.6),
            evidence=(
                f"recent_scores={[round(s, 3) for s in recent]}, "
                f"avg_drift={avg_drift:.3f}"
            ),
        )

    # ------------------------------------------------------------------
    # Scoring and classification
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_drift_score(
        alignment: AlignmentReport,
        indicators: list[DriftIndicator],
    ) -> float:
        """Compute the composite drift score from alignment and indicators.

        The base drift is the inverse of alignment confidence.  Indicators
        contribute additively, capped at 1.0.

        The formula:
            base = 1.0 - alignment.confidence
            indicator_sum = sum(ind.severity_contribution * weight for ind)
            drift_score = base * 0.6 + indicator_sum * 0.4

        The indicator weights are normalised so that a single severe indicator
        can push drift_score high, but multiple small indicators can too.
        """
        base_drift = 1.0 - alignment.confidence

        if not indicators:
            # No indicators beyond alignment -- use base drift directly.
            return max(0.0, min(1.0, base_drift))

        # Sum indicator contributions, capped at 1.0.
        indicator_sum = min(1.0, sum(ind.severity_contribution for ind in indicators))

        # Weighted combination: base alignment drift 60%, indicators 40%.
        drift_score = base_drift * 0.6 + indicator_sum * 0.4

        return max(0.0, min(1.0, drift_score))

    def _classify_severity(self, drift_score: float) -> str:
        """Classify a drift score into a severity level.

        Uses the threshold map to find the highest severity whose threshold
        is at or below the drift score.
        """
        severity = SEVERITY_NONE
        for level in [
            SEVERITY_NONE,
            SEVERITY_LOW,
            SEVERITY_MODERATE,
            SEVERITY_HIGH,
            SEVERITY_CRITICAL,
        ]:
            if drift_score >= self._thresholds.get(level, 1.0):
                severity = level
        return severity

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def check_with_task(
        self,
        action: str,
        file_path: Optional[str],
        task: "TaskMemory",
    ) -> DriftReport:
        """Run a drift check using a TaskMemory object as the scope source.

        This is a convenience wrapper that extracts the required fields from
        a TaskMemory instance and delegates to :meth:`check`.

        Parameters
        ----------
        action:
            Description of the action being checked.
        file_path:
            Optional file path the action targets.
        task:
            A TaskMemory instance providing the task scope.

        Returns
        -------
        DriftReport
            The drift assessment.
        """
        return self.check(
            action=action,
            file_path=file_path,
            task_title=task.title,
            task_description=task.summary or "",
            task_phase=task.phase,
            task_files=task.get_file_paths(),
            task_steps=[step.action for step in task.steps],
            task_ticket_id=task.ticket_id,
        )

    def get_session_summary(self) -> dict:
        """Return a summary of the current detection session.

        Useful for including in reports and monitoring dashboards.
        """
        if not self._session_scores:
            return {
                "total_checks": 0,
                "average_drift": 0.0,
                "max_drift": 0.0,
                "min_drift": 0.0,
                "current_drift": 0.0,
                "trend_direction": "stable",
                "unique_files_checked": 0,
            }

        avg = sum(self._session_scores) / len(self._session_scores)
        max_drift = max(self._session_scores)
        min_drift = min(self._session_scores)
        current = self._session_scores[-1]

        # Determine trend direction from last 3+ scores.
        if len(self._session_scores) >= 3:
            recent = self._session_scores[-3:]
            if all(recent[i] > recent[i - 1] for i in range(1, len(recent))):
                trend = "increasing"
            elif all(recent[i] < recent[i - 1] for i in range(1, len(recent))):
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"

        return {
            "total_checks": len(self._session_scores),
            "average_drift": round(avg, 3),
            "max_drift": round(max_drift, 3),
            "min_drift": round(min_drift, 3),
            "current_drift": round(current, 3),
            "trend_direction": trend,
            "unique_files_checked": len(set(self._session_files)),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_directory(file_path: str) -> str:
    """Extract the directory portion of a file path (everything before the last /)."""
    idx = file_path.rfind("/")
    if idx >= 0:
        return file_path[:idx]
    return ""
